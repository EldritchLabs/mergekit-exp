# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# mergekit/merge_methods/moe_karcher.py by Naphula

"""
MoE-aware Karcher merge method.
Blends corresponding experts across MoE models using geometric mean.
"""

import re
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
from mergekit.merge_methods.karcher import karcher_merge_tensors
from mergekit.merge_methods.rectify_embed import rectify_embed_sizes

def log_moe_karcher_audit(layer_name: str, donor_names: List[str], final_tensor: torch.Tensor, donor_tensors: List[torch.Tensor]):
    """Prints and saves a visual bar chart of MoE Karcher influence with exact magnitudes."""
    final_norm = torch.linalg.norm(final_tensor.float()).item()
    
    bar_char = "█"
    lines = [f"\n[MoE_Karcher Audit] Layer: {layer_name}"]
    lines.append(f"{'Donor Model':<55} {'Norm':<10} {'Rel Energy':<10}")
    lines.append("-" * 85)
    
    for i, (name, donor_t) in enumerate(zip(donor_names, donor_tensors)):
        donor_norm = torch.linalg.norm(donor_t.float()).item()
        rel_energy = donor_norm / (final_norm + 1e-8)
        
        bar_len = int(min(1.0, rel_energy) * 40)
        clean_name = name.split("\\")[-1].split("/")[-1][:50]
        
        # Identity Check
        dup_msg = ""
        if i > 0 and torch.equal(donor_tensors[0].view(-1)[:10], donor_t.view(-1)[:10]):
            dup_msg = " [!! DUPLICATE !!]"
        
        lines.append(f"  {clean_name:<50}: {bar_char * bar_len:<40} {donor_norm:>8.4f} ({rel_energy:5.2f}x){dup_msg}")

    log_entry = "\n".join(lines)
    print(log_entry)
    with open("moe_karcher_audit.log", "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")

class MoEKarcherTask(Task[torch.Tensor]):
    """
    MoE-aware Karcher merge that:
    1. Identifies expert weights by pattern matching
    2. Blends corresponding experts across MoE models
    3. Handles router weights separately with optional strategies
    """

    gather_tensors: MergeTensorInput
    weight_info: WeightInfo
    max_iter: int
    tol: float
    router_strategy: str  # "average", "karcher", "first", "random_init"
    blend_experts: bool  # If True, blend corresponding experts; if False, interleave

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def _is_expert_weight(self, name: str) -> Optional[int]:
        # Gemma 4 packed experts: model.language_model.layers.X.experts.gate_up_proj
        if ".experts." in name:
            return 0 
        
        # Standard MoE experts (Mixtral/DeepSeek)
        match = re.search(r'experts\.(\d+)\.', name)
        if match:
            return int(match.group(1))
            
        return None

    def _is_router_weight(self, name: str) -> bool:
        """Detect if this is a router/gate weight."""
        return (
            'block_sparse_moe.gate.weight' in name or
            'mlp.gate.weight' in name or
            'shared_expert_gate.weight' in name
        )

    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> torch.Tensor:
        # Skip if no tensors were found (None returned from LoadTensor)
        if not tensors or any(t is None for t in tensors.values()):
            return None
            
        if len(tensors) == 1:
            return list(tensors.values())[0]

        # Capture donor names for the audit
        donor_names = [str(m.model.path) for m in tensors.keys()]
        model_tensors = list(tensors.values())
        
        # Ensure compatible shapes
        for i in range(1, len(model_tensors)):
            rectify_embed_sizes(self.weight_info, [model_tensors[0], model_tensors[i]])

        weight_name = self.weight_info.name
        expert_idx = self._is_expert_weight(weight_name)
        is_router = self._is_router_weight(weight_name)

        # --- Perform Merge ---
        res_tensor = None
        
        # Handle expert weights
        if expert_idx is not None and self.blend_experts:
            alphas = [1.0 / len(model_tensors)] * len(model_tensors)
            res_tensor = karcher_merge_tensors(
                model_tensors, alphas, max_iter=self.max_iter, tol=self.tol
            )

        # Handle router weights
        elif is_router:
            if self.router_strategy == "average":
                res_tensor = torch.stack(model_tensors).mean(dim=0)
            elif self.router_strategy == "first":
                res_tensor = model_tensors[0]
            elif self.router_strategy == "random_init":
                res_tensor = torch.randn_like(model_tensors[0]) * 0.02
            else: # karcher
                alphas = [1.0 / len(model_tensors)] * len(model_tensors)
                res_tensor = karcher_merge_tensors(
                    model_tensors, alphas, max_iter=self.max_iter, tol=self.tol
                )
        
        # Handle everything else
        else:
            alphas = [1.0 / len(model_tensors)] * len(model_tensors)
            res_tensor = karcher_merge_tensors(
                model_tensors, alphas, max_iter=self.max_iter, tol=self.tol
            )

        # --- Trigger Audit HUD ---
        # Only audit every few layers or specific types to avoid console spam
        if "mlp" in weight_name or "self_attn.q_proj" in weight_name:
            log_moe_karcher_audit(weight_name, donor_names, res_tensor, model_tensors)

        return res_tensor
        
        # Ensure compatible shapes
        for i in range(1, len(model_tensors)):
            rectify_embed_sizes(self.weight_info, [model_tensors[0], model_tensors[i]])

        weight_name = self.weight_info.name
        expert_idx = self._is_expert_weight(weight_name)
        is_router = self._is_router_weight(weight_name)

        # Handle expert weights
        if expert_idx is not None and self.blend_experts:
            # Blend corresponding experts across MoE models using Karcher mean
            # This assumes expert[i] in MoE1 corresponds to expert[i] in MoE2
            alphas = [1.0 / len(model_tensors)] * len(model_tensors)
            return karcher_merge_tensors(
                model_tensors, alphas, max_iter=self.max_iter, tol=self.tol
            )

        # Handle router weights
        if is_router:
            if self.router_strategy == "karcher":
                alphas = [1.0 / len(model_tensors)] * len(model_tensors)
                return karcher_merge_tensors(
                    model_tensors, alphas, max_iter=self.max_iter, tol=self.tol
                )
            elif self.router_strategy == "average":
                # Simple arithmetic mean
                return torch.stack(model_tensors).mean(dim=0)
            elif self.router_strategy == "first":
                # Use router from first model
                return model_tensors[0]
            elif self.router_strategy == "random_init":
                # Random initialization (scaled normal)
                return torch.randn_like(model_tensors[0]) * 0.02
            else:
                # Default to Karcher
                alphas = [1.0 / len(model_tensors)] * len(model_tensors)
                return karcher_merge_tensors(
                    model_tensors, alphas, max_iter=self.max_iter, tol=self.tol
                )

        # Handle non-expert, non-router weights (attention, norms, embeddings)
        # Use standard Karcher mean
        alphas = [1.0 / len(model_tensors)] * len(model_tensors)
        return karcher_merge_tensors(
            model_tensors, alphas, max_iter=self.max_iter, tol=self.tol
        )

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()


class MoEKarcherMerge(MergeMethod):
    """
    MoE-aware Karcher merge method.
    
    Intelligently blends MoE models by:
    - Applying Karcher mean to corresponding experts
    - Handling router weights with configurable strategy
    - Preserving MoE structure in the output
    """

    def name(self) -> str:
        return "moe_karcher"

    @override
    def pretty_name(self) -> Optional[str]:
        return "MoE Karcher Mean"

    @override
    def reference_url(self) -> Optional[str]:
        return "https://en.wikipedia.org/wiki/Karcher_mean"

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="max_iter", required=False, default_value=10),
            ConfigParameterDef(name="tol", required=False, default_value=1e-5),
            ConfigParameterDef(
                name="router_strategy",
                required=False,
                default_value="karcher",
            ),
            ConfigParameterDef(
                name="blend_experts",
                required=False,
                default_value=True,
            ),
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
        return MoEKarcherTask(
            gather_tensors=tensors,
            weight_info=output_weight,
            max_iter=parameters["max_iter"],
            tol=parameters["tol"],
            router_strategy=parameters["router_strategy"],
            blend_experts=parameters["blend_experts"],
        )