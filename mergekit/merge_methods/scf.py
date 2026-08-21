# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# mergekit/merge_methods/scf.py by Naphula

from typing import List, Optional
import torch
import torch.nn.functional as F
from mergekit.merge_methods.easy_define import merge_method

def _calculate_dynamic_threshold(scores: torch.Tensor) -> torch.Tensor:
    """Approximates a robust dynamic threshold using IQR."""
    # Use a sample for very large tensors to speed up quantile calculation
    if scores.numel() > 1_000_000:
        scores = scores.flatten()[torch.randperm(scores.numel())[:1_000_000]]
    
    q1, median, q3 = torch.quantile(scores.float(), torch.tensor([0.25, 0.5, 0.75], device=scores.device))
    iqr = q3 - q1
    return median + 1.5 * iqr

def _compute_importance_scores(params: torch.Tensor, base_params: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Calculates Arcee Fusion's importance scores."""
    diff = (params - base_params).abs()
    # Ensure softmax is applied on a meaningful dimension
    softmax_dim = -1 if params.dim() > 1 else 0
    p = F.softmax(params, dim=softmax_dim) + eps
    q = F.softmax(base_params, dim=softmax_dim) + eps
    
    # Add a dimension for broadcasting if needed
    kl_div = torch.sum(p * torch.log(p / q), dim=softmax_dim)
    if kl_div.dim() < diff.dim():
        kl_div = kl_div.unsqueeze(-1)
        
    return diff * kl_div

@merge_method(
    name="scf",
    pretty_name="Selective Coherence Fusion",
)
@torch.no_grad()
def selective_coherence_fusion(
    tensors: List[torch.Tensor],
    base_tensor: torch.Tensor,
    select_topk: float = 0.9,
) -> torch.Tensor:
    """
    Selective Coherence Fusion (SCF):
    A two-stage filtering method that combines the strengths of SCE and Arcee Fusion.
    1.  **SCE Stage (Select):** A variance-based mask is created to identify the most
        structurally active parameter positions across all donor models.
    2.  **Arcee Fusion Stage (Fuse):** Within these active regions, Arcee Fusion's
        importance scores (magnitude * KL divergence) are calculated for each donor.
        A dynamic threshold is used to create a final fusion mask, merging only the
        most semantically significant changes.

    - select_topk: The fraction of parameter positions to retain based on variance,
                   defining the "coherent subspace" for the fusion.
    """
    if not tensors:
        return base_tensor

    task_vectors = torch.stack([t.float() - base_tensor.float() for t in tensors])
    
    # --- Stage 1: SCE - Select Coherent Subspace ---
    variance_mask = torch.ones_like(base_tensor, dtype=torch.bool)
    if select_topk < 1.0 and task_vectors.numel() > 0:
        variances = torch.var(task_vectors, dim=0, unbiased=False)
        k = int(select_topk * variances.numel())
        if k > 0:
            threshold = torch.kthvalue(variances.flatten(), variances.numel() - k).values
            variance_mask = variances >= threshold

    # --- Stage 2: Arcee Fusion - Fuse within Subspace ---
    final_delta = torch.zeros_like(base_tensor)
    
    # Calculate importance scores for all donors *within the coherent subspace*
    all_importance_scores = torch.stack([
        _compute_importance_scores(t, base_tensor) * variance_mask
        for t in tensors
    ])
    
    # Average the importance scores to get a consensus on what's important
    avg_importance_scores = all_importance_scores.mean(dim=0)
    
    # Use a single dynamic threshold for all donors based on the consensus importance
    dynamic_threshold = _calculate_dynamic_threshold(avg_importance_scores[variance_mask])
    
    # Create a fusion mask where the average importance exceeds the threshold
    fusion_mask = (avg_importance_scores >= dynamic_threshold).float()
    
    # Apply the fusion mask to the average of the task vectors
    avg_task_vector = task_vectors.mean(dim=0)
    final_delta = avg_task_vector * fusion_mask
    
    return (base_tensor + final_delta).to(base_tensor.dtype)