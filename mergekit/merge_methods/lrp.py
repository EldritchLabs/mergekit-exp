# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from typing_extensions import override

from mergekit.architecture import WeightInfo
from mergekit.merge_methods.generalized_task_arithmetic import get_mask as sign_consensus_mask
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)
from mergekit.merge_methods.rectify_embed import rectify_embed_sizes


class LRPMergeTask(Task[torch.Tensor]):
    gather_tensors: MergeTensorInput
    base_model: Optional[ModelReference]
    inversion_mode: int = 0 # 0 = Standard, 1 = MAGIC Inversion

    def _compute_in_memory_lrp(self, delta: torch.Tensor) -> torch.Tensor:
        """
        Calculates functional importance without external files.
        Uses the 'Self-Relevance' proxy: |Delta| * Variance(Delta).
        # Performs LRP-based merging using Layer-wise Relevance Propagation scores
        # to determine which weights to keep during model merging.
        """
        # High variance in the delta indicates a 'Hot Zone' of learning
        importance = delta.abs() * (delta.var(dim=-1, keepdim=True) + 1e-8)
        return importance
    model_weights: ImmutableMap[ModelReference, float]
    density: float
    weight_info: WeightInfo

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def _compute_topk_mask(self, importance: torch.Tensor, density: float) -> torch.Tensor:
        """
        Compute binary mask for top-k most important weights.

        Args:
            importance: Importance scores tensor
            density: Fraction of weights to keep (0.0 to 1.0)

        Returns:
            Binary mask (1 = keep, 0 = discard)
        """
        if density <= 0:
            return torch.zeros_like(importance, dtype=torch.bool)
        if density >= 1.0:
            return torch.ones_like(importance, dtype=torch.bool)

        numel = importance.numel()
        k = max(1, int(density * numel))
        k = min(k, numel)

        # Use topk for efficiency
        flat_importance = importance.flatten()
        top_k_values, _ = torch.topk(flat_importance, k)
        threshold = top_k_values[-1]

        return (importance >= threshold).to(importance.dtype)

    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> torch.Tensor:
        # Get base tensor
        base_tensor = tensors.get(self.base_model) if self.base_model else None

        if base_tensor is None:
            first_tensor = list(tensors.values())[0] if tensors else None
            if first_tensor is None:
                raise ValueError("No tensors provided for merging")
            base_tensor = torch.zeros_like(first_tensor)

        # Collect non-base tensors
        weight_tensors = {ref: t for ref, t in tensors.items() if ref != self.base_model}

        if not weight_tensors:
            return base_tensor

        # Rectify embedding sizes
        rectify_embed_sizes(self.weight_info, [base_tensor] + list(weight_tensors.values()))
        base_tensor = base_tensor.to(list(weight_tensors.values())[0].dtype)

        # 1. Calculate all Task Vectors
        deltas = []
        model_refs = list(weight_tensors.keys())
        for ref in model_refs:
            deltas.append(weight_tensors[ref] - base_tensor)
        
        stacked_deltas = torch.stack(deltas, dim=0)

        # 2. MAGIC Conflict Inversion
        if self.inversion_mode == 1 and len(deltas) > 1:
            majority_sign = torch.sign(stacked_deltas.sum(dim=0))
            stacked_deltas = torch.where(
                (torch.sign(stacked_deltas) * majority_sign) < 0,
                -stacked_deltas, 
                stacked_deltas
            )

        # 3. In-Memory LRP Sparsification & Weighted Average
        total_weight = sum(self.model_weights.values())
        final_merged_delta = torch.zeros_like(base_tensor)
        
        for i, ref in enumerate(model_refs):
            delta = stacked_deltas[i]
            importance = self._compute_in_memory_lrp(delta)
            mask = self._compute_topk_mask(importance, self.density)
            
            weight = self.model_weights[ref]
            final_merged_delta += (weight / total_weight) * (delta * mask)

        return base_tensor + final_merged_delta

    def uses_accelerator(self) -> bool:
        return True

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()

    def priority(self) -> int:
        return 0


class LRPMerge(MergeMethod):
    """
    LRP-based merge method using Layer-wise Relevance Propagation scores.

    Merges fine-tuned models by:
    1. Computing task vectors (deltas from base)
    2. Using LRP importance scores to determine which weights are most relevant
    3. Sparsifying based on importance (LRP scores or magnitude fallback)
    4. Weighted averaging of sparse deltas
    """

    @override
    def name(self) -> str:
        return "lrp"

    @override
    def pretty_name(self) -> Optional[str]:
        return "LRP Merge"

    @override
    def reference_url(self) -> Optional[str]:
        return "https://github.com/arcee-ai/mergekit"

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="density", required=False, default_value=0.7),
            ConfigParameterDef(name="inversion_mode", required=False, default_value=0),
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [ConfigParameterDef(name="weight", required=False, default_value=1.0)]

    @override
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
        """Create the LRP merge task with proper validation."""
        # Collect model weights from non-base models
        model_weights = {}
        for model_ref, params in tensor_parameters.items():
            if model_ref != base_model:
                try:
                    weight = params["weight"]
                except (KeyError, TypeError):
                    weight = 1.0
                model_weights[model_ref] = weight

        if not model_weights:
            raise ValueError("At least one fine-tuned model (other than base) is required for LRP merge")

        # Get density parameter
        try:
            density = parameters["density"]
        except (KeyError, TypeError):
            density = 0.7

        # Validate density
        if not 0 <= density <= 1:
            raise ValueError(f"density must be between 0 and 1, got {density}")

        # Convert ImmutableMap to dict to use .get() safely
        params_dict = dict(parameters)
        
        return LRPMergeTask(
            gather_tensors=tensors,
            base_model=base_model,
            model_weights=ImmutableMap(model_weights),
            density=density,
            weight_info=output_weight,
            inversion_mode=int(params_dict.get("inversion_mode", 0)),
        )
