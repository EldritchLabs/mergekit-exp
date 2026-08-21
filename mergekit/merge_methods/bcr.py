# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# mergekit/merge_methods/bcr.py by Naphula

from typing import List, Optional
import torch
from mergekit.merge_methods.easy_define import merge_method

@merge_method(
    name="bcr",
    pretty_name="Barycentric Conflict Resolution",
)
@torch.no_grad()
def barycentric_conflict_resolution(
    tensors: List[torch.Tensor],
    base_tensor: torch.Tensor,
    weight: List[float],
    dominance: List[float],
    density: List[float],
) -> torch.Tensor:
    """
    Barycentric Conflict Resolution (BCR) Merge:
    An advanced task vector method that resolves conflicts constructively.
    It separates parameter changes into "agreement" and "conflict" sets based
    on sign. Agreed-upon changes are merged like in TIES. Conflicting changes
    are resolved via a weighted barycentric mean, where a model's `dominance`
    factor allows it to "outvote" others and pull the result closer to its
    own value, rather than being discarded.

    - weight: Standard per-model weighting for its task vector.
    - dominance: Per-model factor (0-1) controlling its influence in conflicts.
    - density: Per-model sparsity for the initial task vector.
    """
    if not tensors:
        return base_tensor

    task_vectors = torch.stack([t.float() - base_tensor.float() for t in tensors])
    n, *shape = task_vectors.shape
    device = task_vectors.device
    dtype = task_vectors.dtype

    # --- Stage 1: Sparsify Task Vectors ---
    for i in range(n):
        d = density[i]
        if d < 1.0:
            flat_tv = task_vectors[i].flatten()
            k = int(d * flat_tv.numel())
            if k == 0:
                task_vectors[i] = torch.zeros_like(task_vectors[i])
                continue
            
            # Magnitude pruning
            abs_tv = flat_tv.abs()
            threshold = torch.kthvalue(abs_tv, flat_tv.numel() - k).values
            mask = abs_tv >= threshold
            task_vectors[i] = task_vectors[i] * mask.view(*shape)

    # --- Stage 2: Identify Agreement and Conflict Sets ---
    signs = torch.sign(task_vectors)
    
    # Sum of signs determines agreement. 0 means perfect conflict or all zeros.
    # A sum equal to the number of non-zero models means all agree.
    num_nonzero = (signs != 0).sum(dim=0)
    sign_sum_abs = signs.sum(dim=0).abs()

    # Agreement mask is where the absolute sum of signs equals the number of active models
    agreement_mask = (sign_sum_abs == num_nonzero).float()
    conflict_mask = 1.0 - agreement_mask

    # --- Stage 3: Resolve Agreement Set (TIES-like) ---
    weights_tensor = torch.tensor(weight, device=device, dtype=dtype).view(n, *([1] * len(shape)))
    
    # Weighted sum of vectors that agree
    agreed_delta = (task_vectors * weights_tensor * agreement_mask).sum(dim=0)
    
    # Normalize by the sum of weights of models that agreed
    agreed_divisor = (weights_tensor * agreement_mask * (signs != 0)).sum(dim=0)
    agreed_delta = torch.nan_to_num(agreed_delta / agreed_divisor, 0.0)

    # --- Stage 4: Resolve Conflict Set (Barycentric Mean) ---
    dominance_tensor = torch.tensor(dominance, device=device, dtype=dtype).view(n, *([1] * len(shape)))
    
    # Numerator: sum of (vector * weight * dominance) for conflicting params
    conflict_numerator = (task_vectors * weights_tensor * dominance_tensor * conflict_mask).sum(dim=0)
    
    # Denominator: sum of (weight * dominance) for conflicting params
    conflict_denominator = (weights_tensor * dominance_tensor * conflict_mask * (signs != 0)).sum(dim=0)
    
    conflicted_delta = torch.nan_to_num(conflict_numerator / conflict_denominator, 0.0)

    # --- Stage 5: Combine and Finalize ---
    final_task_vector = agreed_delta + conflicted_delta
    
    return (base_tensor + final_task_vector).to(base_tensor.dtype)