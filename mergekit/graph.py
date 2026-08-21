# graph_v18.py - Optimized for 3060 TI (8GB VRAM) and similar low-VRAM GPUs  
# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
"""
Module for computational graph execution.

Classes:
    Task: Abstract base class representing a computational task.
    Executor: Class for scheduling and executing directed acyclic task graphs.
"""
  
import os  
import sys  
import gc  
import logging  
import networkx  
import torch  
import tqdm  
from pydantic import BaseModel  
from typing_extensions import Generic, TypeVar  
from abc import ABC, abstractmethod  
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union  
  
from mergekit.common import get_torch_accelerator_module  
  
# ============================================================================  
# CONFIGURATION SECTION - TUNE THESE PARAMETERS FOR YOUR GPU  
# ============================================================================  
  
# --- PRIMARY VRAM TARGETS ---  
# For 3060 TI (8GB): Start with 7.2-7.4GB. Increase if stable, decrease if OOM.  
# For 3060 (12GB): Try 10.5-11.0GB  
# For 4GB cards: Try 3.2-3.5GB  
TARGET_VRAM_GB = 7.7  # Target VRAM usage in GB (TUNE THIS FIRST)  
  
# Safety margin to account for PyTorch overhead and fragmentation  
# Windows typically needs ~0.8GB, Linux ~0.5GB  
VRAM_SAFETY_MARGIN_GB = 0.2  # Reduce to 0.5-0.6 on Linux, increase to 1.0 if unstable  
  
# --- CUDA MEMORY ALLOCATOR CONFIGURATION ---  
# Smaller values = less fragmentation but more overhead  
# 24MB is optimal for 8GB cards, 32MB for 12GB+ cards  
CUDA_MAX_SPLIT_SIZE_MB = 24  # Options: 16, 24, 32, 64  
  
# --- CHUNK SIZE BEHAVIOR ---  
# How aggressively to reduce chunk size on OOM (0.5-0.9 range)  
# Lower = more conservative (slower but safer), Higher = more aggressive  
CHUNK_REDUCTION_FACTOR = 0.75  # Options: 0.5 (safe), 0.7 (balanced), 0.85 (aggressive)  
  
# Minimum chunk size before giving up and falling back to CPU  
MIN_CHUNK_SIZE = 1  # Usually keep at 1, increase to 4-8 if seeing micro-chunk overhead  
  
# Enable power-of-2 alignment for chunk sizes (following measure.py strategy)  
# This improves memory allocation efficiency  
ENABLE_POWER_OF_2_ALIGNMENT = True  # Set False if causing issues  
  
# --- TASK-SPECIFIC MEMORY MULTIPLIERS ---  
# These control how much extra VRAM to reserve for specific task types  
# Increase if task OOMs, decrease if underutilizing VRAM  
TASK_MULTIPLIERS = {  
    "ModelStock": 2.2,      # Options: 1.8-2.5 (needs room for pairwise similarities)  
    "Karcher": 3.0,         # Options: 2.5-3.5 (iterative, needs working memory)  
    "Consensus": 3.0,       # Options: 2.5-3.5 (similar to Karcher)  
    "Prometheus": 8.0,      # The PrometheusTask is very heavy! High multiplier to force small chunks on 8GB VRAM
    "Tensorguard": 6.0,     # Performs a backward() pass to generate gradients. This requires storing the activation graph and intermediate tensors, which roughly triples the memory requirement compared to a simple linear merge
    "default": 1.2,         # Options: 1.0-1.5 (general tasks)  
}  
  
# --- MEMORY CLEANUP BEHAVIOR ---  
# Enable aggressive garbage collection and cache clearing  
# True = slower but more stable, False = faster but may fragment memory  
ENABLE_AGGRESSIVE_CLEANUP = False  # Set False if merges are very stable  
  
# How often to force cleanup (every N tasks). 0 = after every task  
CLEANUP_FREQUENCY = 10  # Options: 0 (always), 1, 2, 5, 10  
  
# --- FALLBACK STRATEGY ---  
# Fixed chunk sizes to try if adaptive chunking fails  
# Powers of 2 work best for GPU memory alignment  
FALLBACK_CHUNK_SIZES = [4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2]  
  
# --- FAST PATH OPTIMIZATION ---  
# Try to execute entire task at once before chunking  
# True = faster when it works, False = always chunk (more conservative)  
ENABLE_FAST_PATH = True  # Set False if getting frequent OOM on large tasks  
  
# --- TASK ROUTING ---  
# Tasks that should always run on CPU (typically I/O bound)  
CPU_ONLY_TASKS = [  
    "LoadTensor",  
    "GatherTensors",   
    "SaveTensor",  
    "TensorWriterTask",  
    "FinalizeModel",  
    "PermutedEmbeddings",  # Gather operations don't benefit from GPU  
]  
  
# ============================================================================  
# END OF CONFIGURATION SECTION  
# ============================================================================  
  
if sys.platform == "win32":  
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = f"max_split_size_mb:{CUDA_MAX_SPLIT_SIZE_MB}"  
  
ValueT = TypeVar("ValueT")  
LOG = logging.getLogger(__name__)  
  
  
def _round_to_power_of_2(n: int, prefer_lower: bool = True) -> int:  
    """Round to nearest power of 2 for memory alignment."""  
    if n <= 0:  
        return 1  
    if n == 1:  
        return 1  
      
    # Find the two nearest powers of 2  
    power = n.bit_length() - 1  
    lower = 1 << power  
    upper = 1 << (power + 1)  
      
    if prefer_lower or (n - lower) < (upper - n):  
        return lower  
    return upper  
  
  
class Task(ABC, BaseModel, Generic[ValueT], frozen=True):  
    @abstractmethod  
    def arguments(self) -> Dict[str, "Task"]:  
        ...  
  
    @abstractmethod  
    def execute(self, **kwargs) -> ValueT:  
        ...  
  
    def priority(self) -> int:  
        return 0  
  
    def group_label(self) -> Optional[str]:  
        return None  
  
    def uses_accelerator(self) -> bool:  
        return False  
  
    def main_thread_only(self) -> bool:  
        return False  
  
    def duplicate_per_gpu(self) -> bool:  
        return False  
  
  
class TaskUniverse:  
    tasks: List[Task]  
    task_to_index: Dict[Task, int]  
    task_arguments: Dict[int, Dict[str, int]]  
    _type_id_to_index: Dict[Tuple[type, int], int]  
  
    def __init__(self, tasks: Optional[Iterable[Task]] = None):  
        self.tasks = []  
        self.task_to_index = {}  
        self.task_arguments = {}  
        self._type_id_to_index = {}  
        if tasks is not None:  
            for task in tasks:  
                self.add_task(task)  
  
    def add_task(self, task: Task, recursive: bool = True) -> "TaskHandle":  
        _ti_key = (type(task), id(task))  
        if _ti_key in self._type_id_to_index:  
            index = self._type_id_to_index[_ti_key]  
            return TaskHandle(self, index)  
  
        index = self.task_to_index.setdefault(task, len(self.tasks))  
        if index < len(self.tasks):  
            return TaskHandle(self, index)  
        self.tasks.append(task)  
        self._type_id_to_index[_ti_key] = index  
  
        if recursive:  
            self.task_arguments[index] = {}  
            for k, v in task.arguments().items():  
                self.task_arguments[index][k] = self.add_task(v, recursive=True)._index  
        return TaskHandle(self, index)  
  
    def get_handle(self, task: Task) -> Optional["TaskHandle"]:  
        if task not in self.task_to_index:  
            return None  
        return TaskHandle(self, self.task_to_index[task])  
  
  
class TaskHandle:  
    __slots__ = ["_universe", "_index"]  
    _universe: TaskUniverse  
    _index: int  
  
    def __init__(self, universe: TaskUniverse, index: int):  
        self._universe = universe  
        self._index = index  
  
    def task(self) -> Task:  
        return self._universe.tasks[self._index]  
  
    def arguments(self) -> Dict[str, "TaskHandle"]:  
        return {  
            k: TaskHandle(self._universe, v)  
            for k, v in self._universe.task_arguments[self._index].items()  
        }  
  
    def __eq__(self, other):  
        if not isinstance(other, TaskHandle):  
            return False  
        return self._index == other._index and self._universe is other._universe  
  
    def __hash__(self):  
        return self._index  
  
    def __str__(self):  
        return f"TaskHandle({type(self.task()).__name__}, {self._index})"  
  
    __repr__ = __str__  
  
  
class ExecutionSchedule:  
    tasks: List[TaskHandle]  
    last_use_index: Dict[TaskHandle, int]  
  
    def __init__(self, tasks: List[TaskHandle], last_use_index: Dict[TaskHandle, int]):  
        self.tasks = tasks  
        self.last_use_index = last_use_index  
  
  
def build_schedule(  
    targets: List[TaskHandle], cached_values: Dict[TaskHandle, Any]  
) -> ExecutionSchedule:  
    if not targets:  
        return ExecutionSchedule(tasks=[], last_use_index={})  
  
    universe = targets[0]._universe  
    dummy_handle = TaskHandle(universe, -1)  
    edge_tups: List[Tuple[TaskHandle, TaskHandle]] = []  
  
    explored = set()  
    to_explore = set(targets)  
    while to_explore:  
        task = to_explore.pop()  
        if task in explored:  
            continue  
        explored.add(task)  
        if task in (cached_values or {}):  
            continue  
        for dep in task.arguments().values():  
            to_explore.add(dep)  
            edge_tups.append((dep, task))  
  
    for target in targets:  
        edge_tups.append((dummy_handle, target))  
  
    def _compare_key(node: TaskHandle) -> Tuple[str, int]:  
        if node._index < 0:  
            return ("", 0)  
        task = node.task()  
        return (task.group_label() or "", -task.priority())  
  
    graph = networkx.DiGraph(edge_tups)  
    schedule: List[TaskHandle] = [  
        node  
        for node in networkx.lexicographical_topological_sort(graph, key=_compare_key)  
        if (node != dummy_handle) and node not in (cached_values or {})  
    ]  
  
    last_use_index = {}  
    for idx, task in reversed(list(enumerate(schedule))):  
        for dep in task.arguments().values():  
            if dep not in last_use_index:  
                last_use_index[dep] = idx  
        if task not in last_use_index:  
            last_use_index[task] = idx  
    for task in cached_values or {}:  
        if task not in last_use_index:  
            last_use_index[task] = len(schedule) + 1  
  
    return ExecutionSchedule(tasks=schedule, last_use_index=last_use_index)  
  
  
class Executor:  
    math_device: torch.device  
    storage_device: torch.device  
    universe: TaskUniverse  
    targets: List[TaskHandle]  
    schedule: ExecutionSchedule  
    cached_values: Optional[Dict[TaskHandle, Any]]  
    _task_counter: int  
  
    def __init__(  
        self,  
        targets: Union[List[Task], List[TaskHandle]],  
        math_device: torch.device = torch.device("cpu"),  
        storage_device: torch.device = torch.device("cpu"),  
        cached_values: Optional[Dict[TaskHandle, Any]] = None,  
    ):  
        self.cached_values = cached_values  
        self._task_counter = 0  
          
        if isinstance(math_device, str):  
            math_device = torch.device(math_device)  
        if isinstance(storage_device, str):  
            storage_device = torch.device(storage_device)  
        self.math_device = math_device  
        self.storage_device = storage_device  
          
        if targets and isinstance(targets[0], Task):  
            universe = TaskUniverse(targets)  
            targets = [universe.add_task(t) for t in targets]  
        elif targets and isinstance(targets[0], TaskHandle):  
            universe = targets[0]._universe  
        elif not targets:  
            universe = TaskUniverse()  
        else:  
            raise ValueError("Targets must be a list of Task or TaskHandle instances")  
              
        self.universe = universe  
        self.targets = targets  
        self.schedule = build_schedule(targets, cached_values=cached_values)  
  
    def _slice_argument(self, arg: Any, start: int, end: int) -> Any:  
        """Recursively slice tensors within nested structures."""  
        if isinstance(arg, torch.Tensor):  
            if arg.shape[0] > 1:  
                return arg[start:end]  
            return arg  
        elif isinstance(arg, dict):  
            return {k: self._slice_argument(v, start, end) for k, v in arg.items()}  
        elif isinstance(arg, list):  
            return [self._slice_argument(v, start, end) for v in arg]  
        elif isinstance(arg, tuple):  
            return tuple(self._slice_argument(v, start, end) for v in arg)  
        return arg  
  
    def _get_memory_stats(self) -> Dict[str, float]:  
        """Get current VRAM statistics in GB."""  
        if self.math_device.type != "cuda":  
            return {}  
          
        allocated = torch.cuda.memory_allocated(self.math_device) / (1024**3)  
        reserved = torch.cuda.memory_reserved(self.math_device) / (1024**3)  
        total = torch.cuda.get_device_properties(self.math_device).total_memory / (1024**3)  
          
        return {  
            "allocated_gb": allocated,  
            "reserved_gb": reserved,  
            "total_gb": total,  
            "free_gb": total - allocated,  
        }  
  
    def _get_adaptive_chunk_size(self, task: Task, arguments: Dict[str, Any]) -> int:  
        """  
        Calculate optimal chunk size based on available VRAM and task requirements.  
          
        This implements the "measure.py strategy" of targeting a specific VRAM fill level  
        rather than using currently available memory, which prevents oscillation.  
        """  
        if self.math_device.type == "cpu":  
            return 1024  # Large default for CPU  
          
        # Get hardware capacity  
        total_vram = torch.cuda.get_device_properties(self.math_device).total_memory  
        target_bytes = TARGET_VRAM_GB * (1024**3)  
          
        # Analyze tensor dimensions and count  
        num_tensors = 0  
        width = 0  
        bytes_per_element = 4  # Default float32  
          
        for arg in arguments.values():  
            if isinstance(arg, torch.Tensor):  
                num_tensors += 1  
                width = max(width, arg.shape[-1] if len(arg.shape) > 1 else arg.shape[0])  
                bytes_per_element = arg.element_size()  
            elif isinstance(arg, dict):  
                for v in arg.values():  
                    if isinstance(v, torch.Tensor):  
                        num_tensors += 1  
                        width = max(width, v.shape[-1] if len(v.shape) > 1 else v.shape[0])  
                        bytes_per_element = v.element_size()  
          
        if num_tensors == 0 or width == 0:  
            return 512  # Safe default  
          
        # Get task-specific multiplier  
        task_name = type(task).__name__  
        multiplier = TASK_MULTIPLIERS.get("default", 1.2)  
          
        for key, mult in TASK_MULTIPLIERS.items():  
            if key in task_name:  
                multiplier = mult  
                break  
          
        # Calculate bytes per row with multiplier for working memory  
        bytes_per_row = num_tensors * width * bytes_per_element * multiplier  
          
        # Calculate usable VRAM (target minus current allocation and safety margin)  
        current_allocated = torch.cuda.memory_allocated(self.math_device)  
        safety_bytes = VRAM_SAFETY_MARGIN_GB * (1024**3)  
        usable_vram = max(target_bytes - current_allocated - safety_bytes, 1024 * (1024**2))  
          
        # Calculate chunk size  
        chunk_size = max(MIN_CHUNK_SIZE, int(usable_vram // bytes_per_row))  
          
        # Apply power-of-2 alignment if enabled (measure.py strategy)  
        if ENABLE_POWER_OF_2_ALIGNMENT and chunk_size > MIN_CHUNK_SIZE:  
            chunk_size = _round_to_power_of_2(chunk_size, prefer_lower=True)  
          
        LOG.debug(f"Calculated chunk size: {chunk_size} (tensors={num_tensors}, width={width}, mult={multiplier:.2f})")  
        return chunk_size  
  
    def _execute_chunked(self, task: Task, arguments: Dict[str, Any]) -> Any:  
        """  
        Execute task in chunks with progressive fallback strategy.  
          
        Strategy:  
        1. Try adaptive chunk size  
        2. On OOM, reduce by CHUNK_REDUCTION_FACTOR  
        3. Continue until success or MIN_CHUNK_SIZE reached  
        """  
        # Find total rows to process  
        total_rows = 0  
        for arg in arguments.values():  
            if isinstance(arg, torch.Tensor):  
                total_rows = arg.shape[0]  
                break  
            elif isinstance(arg, dict):  
                for v in arg.values():  
                    if isinstance(v, torch.Tensor):  
                        total_rows = v.shape[0]  
                        break  
            if total_rows > 0:  
                break  
          
        if total_rows == 0:  
            return task.execute(**arguments)  
          
        # Calculate initial chunk size  
        chunk_size = self._get_adaptive_chunk_size(task, arguments)  
          
        # FAST PATH: Try to execute all at once if chunk size >= total rows  
        if ENABLE_FAST_PATH and chunk_size >= total_rows:  
            try:  
                gpu_args = {  
                    k: self._move_tensors(v, self.math_device)  
                    for k, v in arguments.items()  
                }  
                res = task.execute(**gpu_args)  
                result = self._move_tensors(res, self.storage_device)  
                del gpu_args, res  
                if ENABLE_AGGRESSIVE_CLEANUP:  
                    torch.cuda.empty_cache()  
                return result  
            except torch.OutOfMemoryError:  
                LOG.warning(f"Fast path OOM, falling back to chunking")  
                torch.cuda.empty_cache()  
                gc.collect()  
                chunk_size = max(MIN_CHUNK_SIZE, total_rows // 2)  
          
        # Chunked execution with progressive reduction  
        results = []  
        i = 0  
        oom_count = 0  
          
        while i < total_rows:  
            end = min(i + chunk_size, total_rows)  
              
            try:  
                chunk_args_gpu = {  
                    k: self._move_tensors(self._slice_argument(v, i, end), self.math_device)  
                    for k, v in arguments.items()  
                }  
                chunk_res = task.execute(**chunk_args_gpu)  
                results.append(self._move_tensors(chunk_res, self.storage_device))  
                  
                del chunk_args_gpu, chunk_res  
                  
                # Aggressive cleanup per measure.py strategy  
                if ENABLE_AGGRESSIVE_CLEANUP:  
                    torch.cuda.empty_cache()  
                  
                i = end  # Move to next chunk  
                oom_count = 0  # Reset OOM counter on success  
                  
            except torch.OutOfMemoryError:  
                oom_count += 1  
                torch.cuda.empty_cache()  
                gc.collect()  
                  
                # Progressive reduction  
                old_chunk = chunk_size  
                chunk_size = max(MIN_CHUNK_SIZE, int(chunk_size * CHUNK_REDUCTION_FACTOR))  
                  
                # Apply power-of-2 alignment  
                if ENABLE_POWER_OF_2_ALIGNMENT:  
                    chunk_size = _round_to_power_of_2(chunk_size, prefer_lower=True)  
                  
                if chunk_size < MIN_CHUNK_SIZE:  
                    LOG.error(f"Chunk size below minimum ({MIN_CHUNK_SIZE}), cannot continue")  
                    raise  
                  
                LOG.warning(  
                    f"OOM at chunk {old_chunk}, reducing to {chunk_size} "  
                    f"(attempt {oom_count}, progress: {i}/{total_rows})"  
                )  
                  
                # Safety: if we OOM too many times, something is wrong  
                if oom_count > 10:  
                    LOG.error("Too many OOM errors, giving up")  
                    raise  
          
        # Concatenate results  
        if not results:  
            return None  
          
        if isinstance(results[0], torch.Tensor):  
            return torch.cat(results, dim=0)  
        elif isinstance(results[0], dict):  
            out = {}  
            for k in results[0].keys():  
                out[k] = torch.cat([r[k] for r in results], dim=0)  
            return out  
          
        return results  
  
    def _execute_with_fallback(self, task: Task, arguments: Dict[str, Any], accelerator) -> Any:  
        """  
        Execute task with comprehensive fallback strategy.  
          
        Strategy:  
        1. Try full GPU execution  
        2. Try adaptive chunking  
        3. Try fixed chunk sizes  
        4. Fall back to CPU  
        """  
        task_name = type(task).__name__  
          
        # Strategy 1: Try full GPU execution for light tasks  
        try:  
            gpu_args = {  
                k: self._move_tensors(v, self.math_device)  
                for k, v in arguments.items()  
            }  
            res = task.execute(**gpu_args)  
            result = self._move_tensors(res, self.storage_device)  
            del gpu_args, res  
            return result  
        except torch.OutOfMemoryError:  
            LOG.debug(f"Full GPU execution failed for {task_name}, trying chunked")  
            torch.cuda.empty_cache()  
            gc.collect()  
        except Exception as e:  
            LOG.warning(f"GPU execution error for {task_name}: {e}")  
            torch.cuda.empty_cache()  
            raise  
          
        # Strategy 2: Try adaptive chunking  
        try:  
            result = self._execute_chunked(task, arguments)  
            return result  
        except torch.OutOfMemoryError:  
            LOG.warning(f"Adaptive chunking failed for {task_name}, trying fixed sizes")  
            torch.cuda.empty_cache()  
            gc.collect()  
        except Exception as e:  
            LOG.warning(f"Chunking error for {task_name}: {e}")  
            raise  
          
        # Strategy 3: Try fixed chunk sizes  
        for chunk_size in FALLBACK_CHUNK_SIZES:  
            if chunk_size < MIN_CHUNK_SIZE:  
                continue  
              
            try:  
                LOG.info(f"Trying fixed chunk size {chunk_size} for {task_name}")  
                  
                # Get total rows  
                total_rows = 0  
                for arg in arguments.values():  
                    if isinstance(arg, torch.Tensor):  
                        total_rows = arg.shape[0]  
                        break  
                    elif isinstance(arg, dict):  
                        for v in arg.values():  
                            if isinstance(v, torch.Tensor):  
                                total_rows = v.shape[0]  
                                break  
                    if total_rows > 0:  
                        break  
                  
                if total_rows == 0:  
                    break  
                  
                results = []  
                for i in range(0, total_rows, chunk_size):  
                    end = min(i + chunk_size, total_rows)  
                    chunk_args = {  
                        k: self._slice_argument(v, i, end)  
                        for k, v in arguments.items()  
                    }  
                    chunk_args_gpu = {  
                        k: self._move_tensors(v, self.math_device)  
                        for k, v in chunk_args.items()  
                    }  
                    chunk_res = task.execute(**chunk_args_gpu)  
                    results.append(self._move_tensors(chunk_res, self.storage_device))  
                    del chunk_args, chunk_args_gpu, chunk_res  
                      
                    if ENABLE_AGGRESSIVE_CLEANUP:  
                        torch.cuda.empty_cache()  
                  
                if isinstance(results[0], torch.Tensor):  
                    return torch.cat(results, dim=0)  
                elif isinstance(results[0], dict):  
                    out = {}  
                    for k in results[0].keys():  
                        out[k] = torch.cat([r[k] for r in results], dim=0)  
                    return out  
                return results  
                  
            except torch.OutOfMemoryError:  
                torch.cuda.empty_cache()  
                gc.collect()  
                continue  
            except Exception as e:  
                LOG.warning(f"Fixed chunk {chunk_size} failed: {e}")  
                break  
          
        # Strategy 4: CPU fallback  
        LOG.warning(f"All GPU strategies failed for {task_name}, using CPU")  
        raise torch.OutOfMemoryError("Forcing CPU fallback")  
  
    def _run(  
        self,  
        quiet: bool = False,  
        desc: Optional[str] = None,  
    ) -> Iterator[Tuple[TaskHandle, Any]]:  
        last_use_index = self.schedule.last_use_index  
  
        values: Dict[TaskHandle, Any] = {}  
        if self.cached_values:  
            for task, value in self.cached_values.items():  
                values[task] = value  
          
        is_gpu_execution = self.math_device.type != "cpu"  
        accelerator = get_torch_accelerator_module(self.math_device.type) if is_gpu_execution else None  
  
        for idx, task_handle in (  
            pbar := tqdm.tqdm(  
                list(enumerate(self.schedule.tasks)),  
                disable=quiet,  
                desc=desc or "Executing graph",  
            )  
        ):  
            task = task_handle.task()  
            task_type = type(task).__name__  
              
            # Log memory stats periodically  
            if is_gpu_execution and idx % 10 == 0:  
                stats = self._get_memory_stats()  
                LOG.debug(  
                    f"Memory: {stats.get('allocated_gb', 0):.2f}GB allocated, "  
                    f"{stats.get('free_gb', 0):.2f}GB free of {stats.get('total_gb', 0):.2f}GB"  
                )  
              
            # Determine execution strategy  
            is_cpu_only_task = task_type in CPU_ONLY_TASKS  
            want_gpu = is_gpu_execution and task.uses_accelerator() and not is_cpu_only_task  
              
            # Collect arguments  
            arguments = {k: values[h] for k, h in task_handle.arguments().items()}  
              
            success = False  
              
            # Try GPU execution  
            if want_gpu:  
                try:  
                    res = self._execute_with_fallback(task, arguments, accelerator)  
                    values[task_handle] = res  
                    success = True  
                except torch.OutOfMemoryError:  
                    LOG.warning(f"All GPU strategies exhausted for {task_type}, falling back to CPU")  
                    success = False  
                except Exception as e:  
                    LOG.error(f"GPU execution failed for {task_type}: {e}")  
                    success = False  
                  
                # Cleanup after GPU attempt  
                if is_gpu_execution and ENABLE_AGGRESSIVE_CLEANUP:  
                    gc.collect()  
                    if accelerator:  
                        accelerator.empty_cache()  
              
            # CPU fallback  
            if not success:  
                if want_gpu:  
                    LOG.info(f"Executing {task_type} on CPU")  
                  
                # Ensure cleanup before CPU execution  
                if is_gpu_execution:  
                    gc.collect()  
                    if accelerator:  
                        accelerator.empty_cache()  
                  
                # Move arguments to CPU  
                cpu_arguments = {  
                    k: self._move_tensors(v, torch.device("cpu"))  
                    for k, v in arguments.items()  
                }  
                  
                res = task.execute(**cpu_arguments)  
                del cpu_arguments  
                res = self._move_tensors(res, self.storage_device)  
                values[task_handle] = res  
              
            del res  
            del arguments  
  
            if task_handle in self.targets:  
                yield (task_handle, values[task_handle])  
  
            # Evict unreferenced values  
            expired = []  
            for key in values:  
                if idx >= last_use_index[key]:  
                    expired.append(key)  
            for key in expired:  
                del values[key]  
              
            # Periodic cleanup (measure.py strategy)  
            self._task_counter += 1  
            if is_gpu_execution and ENABLE_AGGRESSIVE_CLEANUP:  
                if CLEANUP_FREQUENCY == 0 or self._task_counter % max(1, CLEANUP_FREQUENCY) == 0:  
                    gc.collect()  
                    if accelerator:  
                        accelerator.empty_cache()  
  
        del values  
        del pbar  
  
    def run(  
        self,  
        quiet: bool = False,  
        desc: Optional[str] = None,  
    ) -> Iterator[Tuple[Task, Any]]:  
        for handle, value in self._run(quiet=quiet, desc=desc):  
            yield (handle.task(), value)  
  
    def execute(self, desc: Optional[str] = None) -> None:  
        for _ in self.run(desc=desc):  
            pass  
  
    def _move_tensors(  
        self, value: Any, device: torch.device, non_blocking: Optional[bool] = None  
    ) -> Any:  
        """Move tensors to specified device, handling nested structures."""  
        if non_blocking is None:  
            non_blocking = device.type in ["cuda", "xpu"]  
          
        if isinstance(value, torch.Tensor):  
            if value.device == device:  
                return value  
            return value.to(device=device, non_blocking=non_blocking)  
        elif isinstance(value, dict):  
            return {  
                k: self._move_tensors(v, device, non_blocking)  
                for k, v in value.items()  
            }  
        elif isinstance(value, list):  
            return [self._move_tensors(v, device, non_blocking) for v in value]  
        elif isinstance(value, tuple):  
            return tuple(self._move_tensors(v, device, non_blocking) for v in value)  
          
        return value