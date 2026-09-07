# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# mergekit/merge_methods/moe_sce.py by Naphula # this version is not yet functional

"""
MoE-aware SCE (Select, Calculate, Erase) merge method.
Combines corresponding experts across MoE models using exact matrix-level 
variance-based selection, relative energy scaling, and sign consensus.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
from typing_extensions import override

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)
from mergekit.merge_methods.rectify_embed import rectify_embed_sizes
from mergekit.merge_methods.generalized_task_arithmetic import (
    get_mask as sign_consensus_mask,
)

LOG = logging.getLogger(__name__)


def log_moe_sce_audit(
    layer_name: str, 
    base_name: str, 
    donor_names: List[str], 
    variances: List[float], 
    calculated_weights: List[float],
    select_topk: float
):
    """Prints and saves a visual bar chart of MoE SCE selection and energy scaling."""
    bar_char = "█"
    lines = [f"\n[MoE_SCE Audit] Layer: {layer_name} | select_topk={select_topk:.2f}"]
    clean_base = base_name.split("\\")[-1].split("/")[-1][:50]
    lines.append(f"  [BASE] {clean_base:<50}")
    
    total_var = sum(variances) if sum(variances) > 0 else 1.0
    
    for name, var, weight in zip(donor_names, variances, calculated_weights):
        pct = (var / total_var) * 100
        bar_len = int(max(0, min(50, pct / 2)))
        bar = bar_char * bar_len
        clean_name = name.split("\\")[-1].split("/")[-1][:50]
        
        lines.append(
            f"  {clean_name:<50}: {bar:<50} (Var: {pct:5.1f}%, Calc_W: {weight:.4f})"
        )
        
    log_entry = "\n".join(lines)
    print(log_entry)
    with open("moe_sce_audit.log", "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")


def sce_mask(
    tvs: torch.Tensor, 
    density: float, 
    mask_dtype: torch.dtype
) -> torch.Tensor:
    """Mathematical Objective: Variance-based parameter-wise selection mask."""
    if density <= 0:
        return torch.zeros_like(tvs[0], dtype=mask_dtype)
    if density >= 1:
        return torch.ones_like(tvs[0], dtype=mask_dtype)

    # If chunked to 1D, return ones (no variance can be calculated on a single parameter)
    if tvs[0].numel() <= 1:
        return torch.ones_like(tvs[0], dtype=mask_dtype)

    # Calculate exact parameter-wise variance across the stacked task vectors
    var = torch.var(tvs.float(), dim=0, unbiased=False)
    nonzero = torch.count_nonzero(var)
    k = int(nonzero * density)
    if k == 0:
        return torch.zeros_like(tvs[0], dtype=mask_dtype)

    # Retrieve indices of the top-k highest variance values
    _, indices = torch.topk(var.abs().view(-1), k=k, largest=True)
    mask = torch.zeros_like(var, dtype=mask_dtype)
    mask.view(-1)[indices] = 1
    return mask


def sce_weight(tvs: torch.Tensor) -> torch.Tensor:
    """Mathematical Objective: Exact matrix-level energy weight based on squared norm."""
    # Robust multi-dim handler: Flatten everything except the model stack dimension (dim=0)
    # This prevents calculations from crashing or producing un-averaged values during VRAM chunking fallbacks.
    num_models = tvs.shape[0]
    tvs_flat = tvs.float().view(num_models, -1)
    
    weights = torch.mean(tvs_flat ** 2, dim=1)
    weight_sum = torch.sum(weights).item()
    if abs(weight_sum) < 1e-6:
        return torch.ones(num_models, dtype=tvs.dtype, device=tvs.device) / num_models
    return (weights / weight_sum).to(tvs.dtype)


class MoESCETask(Task[torch.Tensor]):
    """
    Task for merging MoE weights using Select, Calculate, Erase (SCE).
    Runs the exact SCE algorithm across experts and attention parameters,
    while handling router weights via configurable fallback strategies.
    """

    gather_tensors: MergeTensorInput
    weight_info: WeightInfo
    base_model: ModelReference
    tensor_parameters: ImmutableMap[ModelReference, Any]
    
    int8_mask: bool
    select_topk: float
    router_strategy: str
    blend_experts: bool

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def _is_expert_weight(self, name: str) -> Optional[int]:
        if ".experts." in name:
            return 0 
        match = re.search(r'experts\.(\d+)\.', name)
        if match:
            return int(match.group(1))
        return None

    def _is_router_weight(self, name: str) -> bool:
        return (
            'block_sparse_moe.gate.weight' in name or
            'mlp.gate.weight' in name or
            'shared_expert_gate.weight' in name
        )

    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> torch.Tensor:
        if not tensors or any(t is None for t in tensors.values()):
            return None
            
        if self.base_model not in tensors:
            raise RuntimeError("Base model required for MoE SCE merge")
            
        base_tensor = tensors.pop(self.base_model)
        
        if len(tensors) == 0:
            return base_tensor

        model_tensors = list(tensors.values())
        all_tensors = [base_tensor] + model_tensors
        
        for i in range(1, len(all_tensors)):
            rectify_embed_sizes(self.weight_info, [all_tensors[0], all_tensors[i]])

        weight_name = self.weight_info.name
        expert_idx = self._is_expert_weight(weight_name)
        is_router = self._is_router_weight(weight_name)

        # --- Handle Router Weights ---
        if is_router:
            if self.router_strategy == "average":
                return torch.stack(all_tensors).mean(dim=0)
            elif self.router_strategy == "first":
                return base_tensor
            elif self.router_strategy == "random_init":
                return torch.randn_like(base_tensor) * 0.02
            elif self.router_strategy != "sce":
                pass # Fallback to SCE if unspecified

        # --- Handle Expert Weights (if blending is disabled) ---
        if expert_idx is not None and not self.blend_experts:
            return base_tensor

        # --- Exact SCE Formulation ---
        # Initialize task vectors (v_i = T_i - T_base)
        tvs = []
        donor_names = []
        for model in self.tensor_parameters.keys():
            if model != self.base_model and model in tensors:
                delta = tensors[model].to(base_tensor.dtype) - base_tensor
                tvs.append(delta)
                donor_names.append(str(model.model.path))

        task_vectors = torch.stack(tvs, dim=0)

        # 1. SELECT (Variance-Based Masking)
        mask_dtype = torch.int8 if self.int8_mask else base_tensor.dtype
        if self.select_topk < 1.0:
            mask = sce_mask(task_vectors, self.select_topk, mask_dtype)
            task_vectors = task_vectors * mask.unsqueeze(0)

        # 2. CALCULATE (Matrix-Level Energy Weighting)
        tv_weights = sce_weight(task_vectors)
        
        # Trigger Live Audit HUD
        if "mlp" in weight_name or "self_attn.q_proj" in weight_name or "lm_head" in weight_name:
            variances = [torch.var(tv.float()).item() if tv.numel() > 1 else 0.0 for tv in task_vectors]
            log_moe_sce_audit(
                weight_name, 
                str(self.base_model.model.path), 
                donor_names, 
                variances, 
                tv_weights.tolist(),
                self.select_topk
            )

        # Reshape calculated weights to broadcast across the stacked task vectors
        weight_broadcaster = tv_weights.clone()
        while weight_broadcaster.dim() < task_vectors.dim():
            weight_broadcaster = weight_broadcaster.unsqueeze(-1)

        # 3. ERASE (TIES Sign Consensus)
        erase_mask = sign_consensus_mask(task_vectors, method="sum", mask_dtype=mask_dtype)

        # 4. SYNTHESIS
        erased_weights = weight_broadcaster * erase_mask
        merged_tv = (task_vectors * erased_weights).sum(dim=0)
        
        # Clamp divisor to prevent division by zero in zero-consensus parameters
        divisor = torch.sum(erased_weights, dim=0).clamp(min=1e-6)
        final_tv = merged_tv / divisor

        return (base_tensor + final_tv).to(base_tensor.dtype)

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()


class MoESCEMerge(MergeMethod):
    """
    MoE-aware SCE merge method.
    
    Intelligently blends MoE models by:
    - Isolating task vectors relative to a base model
    - Applying exact variance selection (Select) and magnitude scaling (Calculate)
    - Resolving sign interference on MoE experts via TIES masking (Erase)
    """

    def name(self) -> str:
        return "moe_sce"

    @override
    def pretty_name(self) -> Optional[str]:
        return "MoE SCE"

    @override
    def reference_url(self) -> Optional[str]:
        return "https://arxiv.org/abs/2408.07990"

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="int8_mask", required=False, default_value=False),
            ConfigParameterDef(name="select_topk", required=False, default_value=1.0),
            ConfigParameterDef(name="router_strategy", required=False, default_value="sce"),
            ConfigParameterDef(name="blend_experts", required=False, default_value=True),
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        # Defined for parameter resolver compatibility (SCE evaluates matrix-level weights internally)
        return [
            ConfigParameterDef(name="weight", required=True),
        ]

    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        base_model: Optional[ModelReference],
        **_kwargs,
    ) -> Task:
        return MoESCETask(
            gather_tensors=tensors,
            weight_info=output_weight,
            base_model=base_model,
            tensor_parameters=tensor_parameters,
            int8_mask=parameters["int8_mask"],
            select_topk=parameters["select_topk"],
            router_strategy=parameters["router_strategy"],
            blend_experts=parameters["blend_experts"],
        )