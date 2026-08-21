# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# mergekit/merge_methods/moe_della.py by Naphula

"""
MoE-aware DELLA merge method.
Blends corresponding experts across MoE models using DELLA magnitude pruning 
and TIES sign consensus.
"""

import re
import logging
from typing import Any, Dict, List, Optional

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
from mergekit.sparsify import RescaleNorm, SparsificationMethod, sparsify

LOG = logging.getLogger(__name__)


def log_moe_della_audit(
    layer_name: str, 
    base_model: ModelReference, 
    tvs: List[Dict[str, Any]], 
    global_lambda: float
):
    """Prints and saves a bar chart of MoE DELLA distribution based on actual Delta Norms."""
    
    base_name = str(base_model.model.path).split("\\")[-1].split("/")[-1][:50]
    
    bar_char = "█"
    lines = [f"\n[MoE_DELLA Audit] Layer: {layer_name} | Lambda={global_lambda:.2f}"]
    lines.append(f"  [BASE] {base_name:<50}")

    stats = []
    total_impact = 0.0
    
    for tv in tvs:
        model_name = str(tv['model'].model.path).split("\\")[-1].split("/")[-1][:50]
        weight = tv.get('weight', 0.0)
        density = tv.get('density', 1.0)
        epsilon = tv.get('epsilon', 0.15)
        delta = tv.get('delta', None)
        
        norm = 0.0
        if delta is not None:
            norm = torch.norm(delta.float()).item()
            
        # Effective contribution magnitude = Weight * Norm
        impact = weight * norm
        total_impact += impact
            
        stats.append({
            'name': model_name,
            'weight': weight,
            'density': density,
            'epsilon': epsilon,
            'norm': norm,
            'impact': impact
        })

    stats.sort(key=lambda x: x['name'])

    for s in stats:
        pct = (s['impact'] / total_impact * 100) if total_impact > 0 else 0.0
        bar_len = int(max(0, min(50, pct / 2)))
        bar = bar_char * bar_len
        
        info = f"W:{s['weight']:.2f} D:{s['density']:.2f} E:{s['epsilon']:.2f} N:{s['norm']:.2f}"
        lines.append(f"  {s['name']:<50}: {bar:<50} {pct:5.1f}% ({info})")

    log_entry = "\n".join(lines)
    print(log_entry)
    
    with open("moe_della_audit.log", "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")


def get_mask(delta: torch.Tensor, mask_dtype: torch.dtype):
    """Returns a mask determining which delta vectors should be merged using TIES sum consensus."""
    sign = delta.sign().to(mask_dtype)
    sign_weight = delta.sum(dim=0)
    majority_sign = (sign_weight >= 0).to(mask_dtype) * 2 - 1
    return sign == majority_sign


class MoEDellaTask(Task[torch.Tensor]):
    gather_tensors: MergeTensorInput
    weight_info: WeightInfo
    base_model: ModelReference
    tensor_parameters: ImmutableMap[ModelReference, Any]
    
    int8_mask: bool
    normalize_router: bool
    normalize_weights: bool
    rescale: bool
    lambda_: float
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
            raise RuntimeError("Base model required for MoE DELLA merge")
            
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
            elif self.router_strategy != "della":
                pass # Fallback to DELLA if unknown

        # --- Handle Expert Weights (if blending is disabled) ---
        if expert_idx is not None and not self.blend_experts:
            return base_tensor

        # --- DELLA Logic (Task Vectors, Sparsification, Consensus) ---
        tvs = []
        for model, tensor in tensors.items():
            delta = tensor.to(base_tensor.dtype) - base_tensor
            tvs.append({
                "model": model,
                "delta": delta,
                "weight": self.tensor_parameters[model]["weight"],
                "density": self.tensor_parameters[model]["density"],
                "epsilon": self.tensor_parameters[model]["epsilon"]
            })

        # Trigger Audit HUD
        if "mlp" in weight_name or "self_attn.q_proj" in weight_name or "lm_head" in weight_name:
            log_moe_della_audit(weight_name, self.base_model, tvs, self.lambda_)

        # 1. Sparsify (DELLA Magnitude Pruning)
        for tv in tvs:
            tv["delta"] = sparsify(
                tv["delta"],
                density=tv["density"],
                method=SparsificationMethod.della_magprune,
                epsilon=tv["epsilon"],
                rescale_norm=RescaleNorm.l1 if self.rescale else None
            )

        deltas = torch.stack([tv["delta"] for tv in tvs], dim=0)
        weights = torch.tensor([tv["weight"] for tv in tvs], dtype=deltas.dtype, device=deltas.device)

        while len(deltas.shape) > len(weights.shape):
            weights.unsqueeze_(-1)

        weighted_deltas = deltas * weights

        # 2. Sign Consensus (TIES)
        mask_dtype = torch.int8 if self.int8_mask else base_tensor.dtype
        mask = get_mask(weighted_deltas, mask_dtype=mask_dtype)

        mixed_delta = (weighted_deltas * mask).sum(dim=0)
        divisor = (weights * mask).sum(dim=0)
        divisor[divisor == 0] = 1

        # 3. Normalize & Scale
        do_normalize = self.normalize_router if is_router else self.normalize_weights
        if do_normalize:
            mixed_delta /= divisor

        if self.lambda_ != 1:
            mixed_delta *= self.lambda_

        return (base_tensor + mixed_delta).to(base_tensor.dtype)

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()


class MoEDellaMerge(MergeMethod):
    """
    MoE-aware DELLA merge method.
    
    Intelligently blends MoE models by:
    - Computing task vectors relative to a base model
    - Applying DELLA magnitude pruning and TIES sign consensus to experts
    - Handling router weights with configurable strategy
    """

    def name(self) -> str:
        return "moe_della"

    @override
    def pretty_name(self) -> Optional[str]:
        return "MoE DELLA"

    @override
    def reference_url(self) -> Optional[str]:
        return "https://arxiv.org/abs/2406.11617"

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="int8_mask", required=False, default_value=False),
            ConfigParameterDef(name="normalize_router", required=False, default_value=True),
            ConfigParameterDef(name="normalize_weights", required=False, default_value=False),
            ConfigParameterDef(name="rescale", required=False, default_value=True),
            ConfigParameterDef(name="lambda", required=False, default_value=1.0),
            ConfigParameterDef(name="router_strategy", required=False, default_value="della"),
            ConfigParameterDef(name="blend_experts", required=False, default_value=True),
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="weight", required=True),
            ConfigParameterDef(name="density", required=False, default_value=1.0),
            ConfigParameterDef(name="epsilon", required=False, default_value=0.15),
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
        return MoEDellaTask(
            gather_tensors=tensors,
            weight_info=output_weight,
            base_model=base_model,
            tensor_parameters=tensor_parameters,
            int8_mask=parameters["int8_mask"],
            normalize_router=parameters["normalize_router"],
            normalize_weights=parameters["normalize_weights"],
            rescale=parameters["rescale"],
            lambda_=parameters["lambda"],
            router_strategy=parameters["router_strategy"],
            blend_experts=parameters["blend_experts"],
        )