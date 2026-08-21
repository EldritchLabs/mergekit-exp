# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only  
# mergekit/merge_methods/multi_fusion.py by Naphula
  
from typing import Any, Dict, List, Optional  
  
import torch  
import torch.nn.functional as F  
from typing_extensions import override  
  
from mergekit.architecture import WeightInfo  
from mergekit.common import ModelReference  
from mergekit.graph import Task  
from mergekit.merge_methods.base import (  
    ConfigParameterDef,  
    MergeMethod,  
    MergeTensorInput,  
)  
from mergekit.merge_methods.rectify_embed import rectify_embed_sizes  
  
  
class DynamicThresholdFusion:  
    def approximate_quantiles(self, tensor, q):  
        # Flatten the tensor  
        flat_tensor = tensor.view(-1)  
  
        # If tensor is too large, sample it  
        if flat_tensor.numel() > 1e6:  
            flat_tensor = flat_tensor[torch.randperm(flat_tensor.numel())[:1000000]]  
  
        # Sort the (possibly sampled) tensor  
        sorted_tensor, _ = torch.sort(flat_tensor)  
  
        # Compute quantile indices  
        quantile_indices = (q * (sorted_tensor.numel() - 1)).long()  
  
        # Return quantiles  
        return sorted_tensor[quantile_indices]  
  
    def calculate_dynamic_threshold(self, importance_scores, tukey_fence=1.5):  
        # Approximate median and quantiles  
        median = self.approximate_quantiles(importance_scores, torch.tensor([0.5]))[0]  
        q1, q3 = self.approximate_quantiles(  
            importance_scores, torch.tensor([0.25, 0.75])  
        )  
  
        # Calculate IQR  
        iqr = q3 - q1  
  
        # Set threshold as median + tukey_fence * IQR  
        dynamic_threshold = median + tukey_fence * iqr  
  
        return dynamic_threshold
  
    def compute_fusion_mask(self, importance_scores, tukey_fence=1.5):  
        threshold = self.calculate_dynamic_threshold(importance_scores, tukey_fence)  
        fusion_mask = (importance_scores >= threshold).float()  
        return fusion_mask, threshold 
  
  
class MultiFusionMergeTask(Task[torch.Tensor]):  
    gather_tensors: MergeTensorInput  
    base_model: ModelReference  
    weight_info: WeightInfo  
    importance_metric: str = "delta_mag"  
    tukey_fence: float = 1.5
  
    def uses_accelerator(self) -> bool:  
        return True  
  
    def arguments(self) -> Dict[str, Task]:  
        return {"tensors": self.gather_tensors}  
  
    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> torch.Tensor:  
        if len(tensors) == 1:  
            return list(tensors.values())[0]  
        elif len(tensors) != 2:  
            raise RuntimeError("MutliFusion merge expects exactly two models")  
        elif self.base_model not in tensors:  
            raise RuntimeError("Base model not in input tensors")  
  
        [a, b] = list(tensors.items())  
        if a[0] != self.base_model:  
            [a, b] = [b, a]  
        prepped_tensors = [a[1], b[1]]  
  
        rectify_embed_sizes(self.weight_info, prepped_tensors)  
  
        importance_scores = self._compute_importance(  
            prepped_tensors[1], prepped_tensors[0]  
        )  
        dynamic_threshold_fusion = DynamicThresholdFusion()  
        fusion_mask, _threshold = dynamic_threshold_fusion.compute_fusion_mask(  
            importance_scores, tukey_fence=self.tukey_fence   
        )  
  
        delta = prepped_tensors[1] - prepped_tensors[0]  
        masked_delta = delta * fusion_mask  
        fused = prepped_tensors[0] + masked_delta  
  
        return fused  
  
    def _compute_importance(  
        self, params: torch.Tensor, base_params: torch.Tensor, eps: float = 1e-8  
    ) -> torch.Tensor:  
        if self.importance_metric == "kl_div":  
            return self._compute_kl_div_importance(params, base_params, eps)  
        elif self.importance_metric == "delta_mag":  
            return self._compute_delta_mag_importance(params, base_params)  
        elif self.importance_metric == "cosine_sim":  
            return self._compute_cosine_sim_importance(params, base_params)  
        elif self.importance_metric == "fisher_grad":  
            return self._compute_fisher_grad_importance(params, base_params)  
        else:  
            raise ValueError(f"Unknown importance metric: {self.importance_metric}")  
  
    def _compute_kl_div_importance(  
        self, params: torch.Tensor, base_params: torch.Tensor, eps: float = 1e-8  
    ) -> torch.Tensor:  
        diff = (params - base_params).abs()  
        p = F.softmax(params, dim=-1) + eps  
        q = F.softmax(base_params, dim=-1) + eps  
        kl_div = torch.sum(p * torch.log(p / q), dim=-1)  
        return diff * kl_div.unsqueeze(-1)  
  
    def _compute_delta_mag_importance(  
        self, params: torch.Tensor, base_params: torch.Tensor  
    ) -> torch.Tensor:  
        # Magnitude of delta - used by TIES/DARE/DELLA  
        delta = params - base_params  
        return delta.abs()  
  
    def _compute_cosine_sim_importance(  
        self, params: torch.Tensor, base_params: torch.Tensor  
    ) -> torch.Tensor:  
        # Cosine similarity based - inspired by Model Stock  
        delta = params - base_params  
        delta_flat = delta.view(-1)  
        base_flat = base_params.view(-1)  
          
        # Compute cosine similarity between delta and base  
        dot_product = torch.dot(delta_flat, base_flat)  
        norm_delta = torch.norm(delta_flat)  
        norm_base = torch.norm(base_flat)  
          
        # Avoid division by zero  
        if norm_delta == 0 or norm_base == 0:  
            return torch.zeros_like(delta)  
          
        cosine_sim = dot_product / (norm_delta * norm_base)  
        # Convert similarity to importance (higher similarity = more important)  
        importance = delta.abs() * (1 + cosine_sim.abs())  
        return importance.view_as(delta)  
  
    def _compute_fisher_grad_importance(  
        self, params: torch.Tensor, base_params: torch.Tensor  
    ) -> torch.Tensor:  
        # Fisher/gradient-based importance - inspired by Karcher/Fisher information  
        # Since we don't have access to gradients/data, we use a proxy based on  
        # the magnitude and variance of the delta  
        delta = params - base_params  
          
        # Compute variance along the last dimension as a proxy for Fisher information  
        if delta.dim() > 1:  
            variance = torch.var(delta, dim=-1, keepdim=True)  
        else:  
            variance = delta.var().unsqueeze(0)  
          
        ## # Importance combines magnitude and variance  
        ## importance = delta.abs() * (variance + 1e-8)  
        ## return importance  
        
        # Use variance directly as importance (SCE-style) rather than variance * magnitude  
        importance = variance + 1e-8  
        return importance
  
class MultiFusionMerge(MergeMethod):  
    def name(self) -> str:  
        return "multi_fusion"  
  
    @override  
    def pretty_name(self) -> Optional[str]:  
        return "Multi Fusion"  
  
    @override  
    def reference_url(self) -> Optional[str]:  
        return "https://huggingface.co/Naphula/Slimaki-Tavern-24B-v1.3"  
  
    def parameters(self) -> List[ConfigParameterDef]:  
        return [  
            ConfigParameterDef(  
                name="importance_metric",  
                required=False,  
                default_value="delta_mag",  
            ),  
            ConfigParameterDef(  
                name="tukey_fence",  
                required=False,  
                default_value=1.5,  
            )  
        ] 
  
    def make_task(  
        self,  
        output_weight: WeightInfo,  
        tensors: MergeTensorInput,  
        base_model: Optional[ModelReference],  
        parameters: Dict[str, Any],  
        **kwargs,  
    ) -> Task[torch.Tensor]:  
        return MultiFusionMergeTask(  
            gather_tensors=tensors,  
            weight_info=output_weight,  
            base_model=base_model,  
            importance_metric=parameters["importance_metric"],  
            tukey_fence=parameters["tukey_fence"]
        )