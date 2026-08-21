# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# mergekit/merge_methods/brf.py by Naphula

from typing import List, Optional
import torch
import torch.nn.functional as F
from mergekit.merge_methods.easy_define import merge_method

def _deterministic_hash(tensor: torch.Tensor) -> int:
    """Creates a deterministic hash from a tensor's content for reproducible sampling."""
    # Use a combination of checksum and norm for a stable hash
    # Using .sum() on a slice is faster than a full checksum for large tensors
    sample_sum = tensor.flatten()[:4096].sum().item()
    l2_norm = torch.linalg.norm(tensor.float()).item()
    # Combine them in a way that produces a large integer
    return int((abs(sample_sum) + 1.0) * (l2_norm + 1.0)) & 0xFFFFFFFF

def _get_resonant_blocks(tensor: torch.Tensor, num_blocks: int, block_size: int, seed: int) -> torch.Tensor:
    """Selects a deterministic, pseudo-random set of blocks from a tensor."""
    generator = torch.Generator(device=tensor.device).manual_seed(seed)
    flat_tensor = tensor.flatten()
    num_elements = flat_tensor.numel()
    
    if num_elements <= num_blocks * block_size:
        return flat_tensor.unsqueeze(0) # Not enough elements, return the whole thing

    # Generate deterministic random start indices
    start_indices = torch.randint(0, num_elements - block_size, (num_blocks,), generator=generator)
    
    # Gather blocks efficiently
    blocks = torch.stack([flat_tensor[i : i + block_size] for i in start_indices])
    return blocks

def _calculate_true_karcher_mean(
    tensors: List[torch.Tensor], 
    alphas: List[float], 
    max_iter: int, 
    tol: float
) -> torch.Tensor:
    """
    A direct adaptation of the official mergekit `karcher_merge_tensors` function.
    This computes the precise Riemannian barycenter for a list of tensors.
    """
    if not tensors:
        return torch.empty(0)
    if len(tensors) == 1:
        return tensors[0]

    # --- Start of direct adaptation from karcher.py ---

    # Calculate norms and unit vectors
    norms = []
    units = []
    for t in tensors:
        t_float = t.float()
        n = torch.linalg.norm(t_float)
        n_val = n.item()
        if n_val == 0.0:
            norms.append(0.0)
            units.append(torch.zeros_like(t))
        else:
            norms.append(n_val)
            units.append((t / n).to(t.dtype))

    # Select non-zero weight vectors
    valid_indices = [i for i, n in enumerate(norms) if n > tol]
    if not valid_indices:
        return torch.zeros_like(tensors[0])

    valid_alphas = [alphas[i] for i in valid_indices]
    alpha_sum = sum(valid_alphas)
    normalized_alphas = [a / alpha_sum for a in valid_alphas]
    valid_units = [units[i] for i in valid_indices]

    # Initial guess: Normalized weighted arithmetic mean
    u = torch.zeros_like(valid_units[0])
    for a, ui in zip(normalized_alphas, valid_units):
        u += a * ui
    norm_u = torch.linalg.norm(u.float()).item()
    if norm_u < tol:
        u = valid_units[0].clone()
    else:
        u = (u / norm_u).to(u.dtype)

    # Iterative Karcher mean computation
    for _ in range(max_iter):
        T = torch.zeros_like(u)
        for a, ui in zip(normalized_alphas, valid_units):
            # Flatten tensor for dot product calculation
            dot = torch.clamp(torch.dot(u.flatten(), ui.flatten()), -1.0, 1.0)
            theta = torch.arccos(dot)
            theta_val = theta.item()
            if theta_val < tol:
                continue
            else:
                # Ensure tensor operations
                sin_theta = torch.sin(theta)
                T += a * (theta / sin_theta) * (ui - dot * u)

        # Convert norm_T to tensor
        norm_T = torch.linalg.norm(T.float())
        if norm_T.item() < tol:
            break

        # Use tensor for trigonometric calculations
        cos_norm_T = torch.cos(norm_T)
        sin_norm_T = torch.sin(norm_T)
        u = (cos_norm_T * u + sin_norm_T * (T / norm_T)).to(u.dtype)

        # Ensure u is a unit vector
        u_norm = torch.linalg.norm(u.float())
        if u_norm.item() > tol:
            u = (u / u_norm).to(u.dtype)

    # Global scale: Weighted sum of original tensor norms (including zero vectors)
    s = 0.0
    for a, n in zip(alphas, norms):
        s += a * n

    return s * u

@merge_method(
    name="brf",
    pretty_name="Barycentric Resonance Flow",
)
@torch.no_grad()
def barycentric_resonance_flow(
    tensors: List[torch.Tensor],
    base_tensor: torch.Tensor,
    weight: List[float],
    strength: float = 1.0,
    resonance_blend: float = 0.5,
    dominance_floor: float = 0.2,
    dominance_roof: float = 1.5,
    # Default Method
    resonance_num_blocks: int = 20,
    resonance_block_size: int = 1024,
    # Microburst Method
    # resonance_num_blocks: int = 100,
    # resonance_block_size: int = 128,
    # Use a very small number of large blocks for resonance analysis.
    # This makes the coherence calculation less stable and more prone to weird results.
    # resonance_num_blocks: int = 5,
    # resonance_block_size: int = 8192,
) -> torch.Tensor:
    """
    Barycentric Resonance Flow (BRF): A multi-stage, geometry-aware, and
    conflict-resolving task vector merge.

    Stage 1: Resonance Analysis - Deterministically samples blocks from each task
             vector to analyze spatial and spectral similarity, producing a
             `coherence_weight` for each model.
    Stage 2: Conflict Resolution (BCR) - Uses coherence to modulate dominance in a
             constructive conflict resolution, producing a synthesized `consensus_vector`.
    Stage 3: Directional Flow - Grounds the `consensus_vector` using the geometric
             magnitude (Karcher Mean norm) of the group and applies it as a
             directional shift to the base model.
    """
    if not tensors:
        return base_tensor

    # --- Stage 0: Setup and Task Vector Calculation ---
    task_vectors = torch.stack([t.float() - base_tensor.float() for t in tensors])
    n, *shape = task_vectors.shape
    device = task_vectors.device
    dtype = task_vectors.dtype
    eps = 1e-8

    # --- Stage 1: Resonance Analysis ---
    coherence_weights = torch.ones(n, device=device, dtype=dtype)
    if n > 1:
        all_blocks_spatial = []
        all_blocks_spectral = []

        for i in range(n):
            tv = task_vectors[i]
            seed = _deterministic_hash(tv)
            blocks = _get_resonant_blocks(tv, resonance_num_blocks, resonance_block_size, seed)
            
            # Spatial features (normalized blocks)
            spatial_blocks = F.normalize(blocks, p=2, dim=1)
            all_blocks_spatial.append(spatial_blocks)

            # Spectral features (FFT magnitude)
            if resonance_blend > 0 and blocks.shape[1] > 1:
                fft_blocks = torch.fft.rfft(blocks, dim=1).abs()
                spectral_blocks = F.normalize(fft_blocks, p=2, dim=1)
                all_blocks_spectral.append(spectral_blocks)

        # Calculate pairwise similarity matrices
        avg_sims = torch.zeros(n, device=device, dtype=dtype)
        
        # Spatial similarity
        if resonance_blend < 1.0:
            spatial_cat = torch.cat(all_blocks_spatial, dim=0)
            sim_matrix_spatial = torch.matmul(spatial_cat, spatial_cat.T)
            # Average similarity of a model's blocks to all other blocks
            for i in range(n):
                start, end = i * resonance_num_blocks, (i + 1) * resonance_num_blocks
                # Exclude self-similarity within the model's own blocks
                sim_matrix_spatial[start:end, start:end] = 0
                avg_sims[i] += torch.mean(sim_matrix_spatial[start:end]) * (1.0 - resonance_blend)

        # Spectral similarity
        if resonance_blend > 0 and all_blocks_spectral:
            spectral_cat = torch.cat(all_blocks_spectral, dim=0)
            sim_matrix_spectral = torch.matmul(spectral_cat, spectral_cat.T)
            for i in range(n):
                start, end = i * resonance_num_blocks, (i + 1) * resonance_num_blocks
                sim_matrix_spectral[start:end, start:end] = 0
                avg_sims[i] += torch.mean(sim_matrix_spectral[start:end]) * resonance_blend
        
        # Normalize coherence weights
        if torch.std(avg_sims) > eps:
            coherence_weights = (avg_sims - avg_sims.min()) / (avg_sims.max() - avg_sims.min() + eps)

    # --- Stage 2: Barycentric Conflict Resolution ---
    # Modulate dominance with coherence
    dominance = dominance_floor + (dominance_roof - dominance_floor) * coherence_weights
    dominance_tensor = dominance.view(n, *([1] * len(shape)))
    weights_tensor = torch.tensor(weight, device=device, dtype=dtype).view(n, *([1] * len(shape)))

    # Identify agreement and conflict sets
    signs = torch.sign(task_vectors)
    num_nonzero = (signs != 0).sum(dim=0)
    sign_sum_abs = signs.sum(dim=0).abs()
    agreement_mask = (sign_sum_abs == num_nonzero).float()
    conflict_mask = 1.0 - agreement_mask

    # Resolve agreement (TIES-like)
    agreed_divisor = (weights_tensor * agreement_mask * (signs != 0)).sum(dim=0)
    agreed_delta = torch.nan_to_num((task_vectors * weights_tensor * agreement_mask).sum(dim=0) / agreed_divisor, 0.0)

    # Resolve conflict (Barycentric Mean)
    conflict_denominator = (weights_tensor * dominance_tensor * conflict_mask * (signs != 0)).sum(dim=0)
    conflict_numerator = (task_vectors * weights_tensor * dominance_tensor * conflict_mask).sum(dim=0)
    conflicted_delta = torch.nan_to_num(conflict_numerator / conflict_denominator, 0.0)
    
    consensus_vector = agreed_delta + conflicted_delta

    # --- Stage 3: Riemannian-Centered Directional Flow ---
    if n > 1:
        # 1. Flatten task vectors
        flat_tvs = task_vectors.flatten(1)
        
        # 2. Normalize to project onto the hypersphere
        # We need to separate Magnitude (Norm) from Direction (Unit Vector)
        norms = torch.linalg.norm(flat_tvs, dim=1, keepdim=True)
        unit_tvs = flat_tvs / (norms + eps)
        
        # 3. Calculate the TRUE Karcher Mean vector
        # (Uses the rigorous Riemannian Log/Exp map function)
        # --- FIX: The function expects a list of tensors and a list of weights (alphas).
        # ---      We provide equal weights since this is a democratic calculation.
        alphas = [1.0 / n] * n
        karcher_mean_vector = _calculate_true_karcher_mean(
            tensors=list(torch.unbind(task_vectors, dim=0)), # Pass as a list of tensors
            alphas=alphas,
            max_iter=100, 
            tol=1e-8
        )
        
        # 4. Calculate the Riemannian Magnitude
        # We need the unit direction of the Karcher Mean to project onto.
        # --- FIX: Ensure the direction vector is flat for broadcasting.
        karcher_direction = F.normalize(karcher_mean_vector.flatten(), p=2, dim=0)
        
        # Project the original vectors onto the Karcher Direction to find the 
        # "Geometric Magnitude" that aligns with the group consensus.
        # This penalizes conflict in the magnitude calculation.
        
        # 4. Calculate the Riemannian Magnitude
        # Instead of just averaging the norms (Arithmetic), we project the 
        # original vectors onto the Karcher Direction to find the 
        # "Geometric Magnitude" that aligns with the group consensus.
        # This penalizes conflict in the magnitude calculation.
        projected_magnitudes = (flat_tvs * karcher_direction).sum(dim=1)
        barycentric_norm = projected_magnitudes.mean()
        
        # Rescale the synthesized consensus vector to match this stable magnitude
        consensus_norm = torch.linalg.norm(consensus_vector)
        resonant_flow_vector = consensus_vector * (barycentric_norm / (consensus_norm + eps))
    
    # --- FIX: This block was missing. It handles the case of a single tensor
    # ---      and applies the final calculated vector to the base tensor.
    else:
        resonant_flow_vector = consensus_vector

    # Apply the final flow vector to the base model
    final_task_vector = strength * resonant_flow_vector
    return (base_tensor + final_task_vector).to(base_tensor.dtype)