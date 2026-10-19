# Copyright (C) 2026 Arcee AI  
# SPDX-License-Identifier: LGPL-3.0-only  
# mergekit/merge_methods/nearswap.py  version 2 by Naphula

"""
NearSwap merge method with full MoE support and real-time swap audit HUD.
Interpolates the base model with parameters from a donor model primarily
where they are already similar using inverse difference weighting.
"""

import logging
import re
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

LOG = logging.getLogger(__name__)


def log_nearswap_audit(
    layer_name: str,
    base_model: ModelReference,
    donor_model: ModelReference,
    weight: torch.Tensor,
    t_val: float,
):
    """Prints and saves a visual ASCII audit bar of NearSwap influence per layer."""
    base_name = str(base_model.model.path).split("\\")[-1].split("/")[-1][:35]
    donor_name = str(donor_model.model.path).split("\\")[-1].split("/")[-1][:35]

    # Calculate statistics
    with torch.no_grad():
        w_float = weight.float()
        mean_swap = w_float.mean().item()
        # Active swapped parameters (influence >= 0.5)
        swapped_pct = (w_float >= 0.5).float().mean().item() * 100.0

    bar_char = "█"
    bar_len = int(max(0, min(40, mean_swap * 40)))
    bar = bar_char * bar_len

    lines = [f"\n[NearSwap Audit] Layer: {layer_name} | t={t_val:.6f}"]
    lines.append(f"  [BASE]  {base_name:<35}")
    lines.append(
        f"  [DONOR] {donor_name:<35}: {bar:<40} {mean_swap * 100:5.1f}% avg "
        f"({swapped_pct:5.1f}% params >= 0.5)"
    )

    log_entry = "\n".join(lines)
    print(log_entry)

    with open("nearswap_audit.log", "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")


def nearswap_kernel(a: torch.Tensor, b: torch.Tensor, t: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes NearSwap blending between base tensor `a` and donor tensor `b`.
    Returns: (blended_tensor, weight_tensor)
    """
    absdiff = torch.abs(a - b)
    weight = (t / absdiff.clamp(min=1e-6)).clamp(min=0.0, max=1.0)
    blended = weight * b + (1.0 - weight) * a
    return blended, weight


class NearSwapTask(Task[torch.Tensor]):
    gather_tensors: MergeTensorInput
    base_model: ModelReference
    weight_info: WeightInfo
    t: float
    router_strategy: str

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def _is_expert_weight(self, name: str) -> Optional[int]:
        """Detect expert weights across standard and packed MoE architectures."""
        if ".experts." in name:
            return 0
        match = re.search(r'block_sparse_moe\.experts\.(\d+)\.w[123]', name)
        if match:
            return int(match.group(1))
        match = re.search(r'mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)', name)
        if match:
            return int(match.group(1))
        return None

    def _is_router_weight(self, name: str) -> bool:
        """Detect MoE gate/router weights."""
        return (
            'block_sparse_moe.gate.weight' in name or
            'mlp.gate.weight' in name or
            'router.proj.weight' in name or
            'router.scale' in name or
            'shared_expert_gate.weight' in name
        )

    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> torch.Tensor:
        if not tensors or any(t is None for t in tensors.values()):
            return None

        if len(tensors) == 1:
            return list(tensors.values())[0]

        if len(tensors) != 2:
            raise RuntimeError("NearSwap merge expects exactly two models (one base and one donor)")

        if self.base_model not in tensors:
            raise RuntimeError(f"Base model {self.base_model} not in input tensors")

        donor_model = [m for m in tensors.keys() if m != self.base_model][0]

        base_t = tensors[self.base_model]
        donor_t = tensors[donor_model]

        prepped = [base_t, donor_t]
        rectify_embed_sizes(self.weight_info, prepped)
        base_t, donor_t = prepped[0], prepped[1]

        # Handle size mismatches if non-embedding layers differ
        if base_t.shape != donor_t.shape:
            LOG.warning(
                f"Skipping donor tensor for {self.weight_info.name} due to shape mismatch: "
                f"{base_t.shape} vs {donor_t.shape}"
            )
            return base_t

        weight_name = self.weight_info.name
        is_router = self._is_router_weight(weight_name)
        is_expert = self._is_expert_weight(weight_name) is not None

        # --- Router Weight Routing ---
        if is_router:
            if self.router_strategy == "first":
                return base_t
            elif self.router_strategy == "average":
                return (base_t + donor_t) * 0.5
            elif self.router_strategy == "lerp":
                return 0.5 * base_t + 0.5 * donor_t
            # Default to NearSwap kernel if router_strategy == "nearswap"

        # --- Compute NearSwap Merge ---
        res, weight = nearswap_kernel(base_t, donor_t, self.t)
        res = res.to(dtype=base_t.dtype, device=base_t.device)

        # --- Live Audit Trigger ---
        # Log dense projections, expert slices, and attention blocks
        should_audit = (
            is_expert
            or "self_attn.q_proj" in weight_name
            or "self_attn.o_proj" in weight_name
            or "self_attn.v_proj" in weight_name
            or "self_attn.k_proj" in weight_name
            or "mlp.down_proj" in weight_name
            or "mlp.gate_proj" in weight_name
            or "mlp.up_proj" in weight_name
            or "lm_head" in weight_name
            or "embed_tokens" in weight_name
            or is_router
        )
        if should_audit:
            log_nearswap_audit(weight_name, self.base_model, donor_model, weight, self.t)

        return res

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()


class NearSwapMerge(MergeMethod):
    """
    NearSwap merge method.
    
    Selectively pulls in parameters from a secondary model where they are
    already close to the base model in Euclidean distance. Supports both
    dense and MoE architectures.
    """

    def name(self) -> str:
        return "nearswap"

    @override
    def pretty_name(self) -> Optional[str]:
        return "NearSwap"

    @override
    def reference_url(self) -> Optional[str]:
        return "https://huggingface.co/alchemonaut/QuartetAnemoi-70B-t0.0001"

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="t", required=False, default_value=0.0),
            ConfigParameterDef(
                name="router_strategy",
                required=False,
                default_value="nearswap",
            ),
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="t", required=False, default_value=None),
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
        router_strategy = (
            parameters["router_strategy"]
            if "router_strategy" in parameters
            else "nearswap"
        )

        # Resolve t: check donor model parameters first, then fall back to global parameters
        t_val = None
        for model_ref, m_params in tensor_parameters.items():
            if model_ref != base_model and "t" in m_params and m_params["t"] is not None:
                t_val = float(m_params["t"])
                break

        if t_val is None:
            t_val = float(parameters.get("t", 0.0) if hasattr(parameters, "get") else parameters["t"])

        return NearSwapTask(
            gather_tensors=tensors,
            base_model=base_model,
            weight_info=output_weight,
            t=t_val,
            router_strategy=router_strategy,
        )


from mergekit.merge_methods.registry import REGISTERED_MERGE_METHODS
REGISTERED_MERGE_METHODS["nearswap"] = NearSwapMerge()