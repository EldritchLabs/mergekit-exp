# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# mergekit/merge_methods/moe_slerp.py by Naphula

"""
MoE-aware SLERP merge method.
Blends corresponding experts across exactly 2 MoE models using spherical interpolation.
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
from mergekit.merge_methods.slerp import slerp
from mergekit.merge_methods.rectify_embed import rectify_embed_sizes


class MoESlerpTask(Task[torch.Tensor]):
    gather_tensors: MergeTensorInput
    base_model: ModelReference
    t: float
    weight_info: WeightInfo
    router_strategy: str

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def _is_expert_weight(self, name: str) -> Optional[int]:
        """Detect expert weights."""
        # Gemma 4 packed experts
        if ".experts." in name:
            return 0
        # Mixtral pattern
        match = re.search(r'block_sparse_moe\.experts\.(\d+)\.w[123]', name)
        if match:
            return int(match.group(1))
        # DeepSeek/Qwen pattern
        match = re.search(r'mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)', name)
        if match:
            return int(match.group(1))
        return None

    def _is_router_weight(self, name: str) -> bool:
        """Detect router weights."""
        return (
            'block_sparse_moe.gate.weight' in name or
            'mlp.gate.weight' in name or
            'shared_expert_gate.weight' in name
        )

    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> torch.Tensor:
        # Gemma 4 Heterogeneous Guard: Skip if no tensors were found for this task
        if not tensors or any(t is None for t in tensors.values()):
            return None

        if len(tensors) == 1:
            return list(tensors.values())[0]
        elif len(tensors) != 2:
            raise RuntimeError("MoE Slerp merge expects exactly two models")
        elif self.base_model not in tensors:
            raise RuntimeError("Base model not in input tensors")

        [a, b] = list(tensors.items())
        if a[0] != self.base_model:
            [a, b] = [b, a]
        prepped_tensors = [a[1], b[1]]

        rectify_embed_sizes(self.weight_info, prepped_tensors)

        weight_name = self.weight_info.name
        expert_idx = self._is_expert_weight(weight_name)
        is_router = self._is_router_weight(weight_name)

        # Handle expert weights with SLERP
        if expert_idx is not None:
            return (
                slerp(self.t, prepped_tensors[0], prepped_tensors[1])
                .to(prepped_tensors[0].dtype)
                .to(prepped_tensors[0].device)
            )

        # Handle router weights
        if is_router:
            if self.router_strategy == "slerp":
                return (
                    slerp(self.t, prepped_tensors[0], prepped_tensors[1])
                    .to(prepped_tensors[0].dtype)
                    .to(prepped_tensors[0].device)
                )
            elif self.router_strategy == "lerp":
                # Linear interpolation for routers
                return (1 - self.t) * prepped_tensors[0] + self.t * prepped_tensors[1]
            elif self.router_strategy == "first":
                return prepped_tensors[0]
            else:
                # Default to slerp
                return (
                    slerp(self.t, prepped_tensors[0], prepped_tensors[1])
                    .to(prepped_tensors[0].dtype)
                    .to(prepped_tensors[0].device)
                )

        # Handle non-expert weights with standard SLERP
        return (
            slerp(self.t, prepped_tensors[0], prepped_tensors[1])
            .to(prepped_tensors[0].dtype)
            .to(prepped_tensors[0].device)
        )

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()


class MoESlerpMerge(MergeMethod):
    """MoE-aware SLERP merge for exactly 2 MoE models."""

    def name(self) -> str:
        return "moe_slerp"

    @override
    def pretty_name(self) -> Optional[str]:
        return "MoE SLERP"

    @override
    def reference_url(self) -> Optional[str]:
        return "https://en.wikipedia.org/wiki/Slerp"

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="t", required=True),
            ConfigParameterDef(
                name="router_strategy",
                required=False,
                default_value="slerp",
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
        # ImmutableMap does not have .get(); use direct access with fallback logic
        r_strat = parameters["router_strategy"] if "router_strategy" in parameters else "slerp"
        
        return MoESlerpTask(
            gather_tensors=tensors,
            base_model=base_model,
            weight_info=output_weight,
            t=parameters["t"],
            router_strategy=r_strat,
        )