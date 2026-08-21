# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# mergekit/merge_methods/scream.py by Naphula 
  
from typing import List, Optional  
import torch  
import torch.nn.functional as F  
from mergekit.merge_methods.easy_define import merge_method  
from mergekit.merge_methods.generalized_task_arithmetic import get_mask as sign_consensus_mask  
  
  
@merge_method(  
    name="scream",  
    pretty_name="SCREAM (Similarity-Consensus Resolved Enhanced Adaptive Merging)",  
    reference_url="https://arxiv.org/abs/2403.19522",  
)  
def scream_merge(  
    tensors: List[torch.Tensor],  
    base_tensor: torch.Tensor,  
    # Model stock parameters  
    stock_weight: float = 0.4,  
    filter_wise: bool = False,  
    # DELLA parameters    
    density: float = 0.7,  
    epsilon: float = 0.05,  
    # SCE parameters  
    select_topk: float = 0.8,  
    # Novelty distribution parameters  
    della_novelty_weight: float = 0.3,  
    sce_novelty_weight: float = 0.3,  
    int8_mask: bool = False,  
) -> torch.Tensor:  
    """  
    SCREAM: Similarity-Consensus Resolved Enhanced Adaptive Merging  
      
    Combines three powerful merging paradigms:  
    1. Model Stock: Geometric interpolation based on cosine similarity  
    2. DELLA: Adaptive magnitude-based pruning with novelty preservation  
    3. SCE: Variance-based selection with sign consensus  
      
    The method distributes weights between:  
    - Model Stock center (conservative, similarity-based)  
    - DELLA novelty (adaptive magnitude pruning)  
    - SCE novelty (variance-based selective merging)  
    """  
    if not tensors:  
        return base_tensor  
      
    mask_dtype = torch.int8 if int8_mask else base_tensor.dtype  
    device = base_tensor.device  
      
    # Calculate task vectors (differences from base)  
    task_vectors = torch.stack([t - base_tensor for t in tensors], dim=0)  
    num_models = len(tensors)  
      
    # === MODEL STOCK COMPONENT ===  
    # Calculate pairwise cosine similarities between task vectors  
    stock_component = _calculate_model_stock_component(  
        task_vectors, base_tensor, filter_wise, device  
    )  
      
    # === DELLA COMPONENT ===  
    # Apply adaptive magnitude-based pruning  
    della_component = _calculate_della_component(  
        task_vectors, density, epsilon, mask_dtype, device  
    )  
      
    # === SCE COMPONENT ===  
    # Apply variance-based selection and sign consensus  
    sce_component = _calculate_sce_component(  
        task_vectors, select_topk, mask_dtype, device  
    )  
      
    # === WEIGHT DISTRIBUTION ===  
    # Combine components with learned weight distribution  
    total_weight = stock_weight + della_novelty_weight + sce_novelty_weight  
    stock_weight /= total_weight  
    della_novelty_weight /= total_weight    
    sce_novelty_weight /= total_weight  
      
    # Final weighted combination  
    final_delta = (  
        stock_weight * stock_component +  
        della_novelty_weight * della_component +  
        sce_novelty_weight * sce_component  
    )  
      
    return base_tensor + final_delta  
  
  
def _calculate_model_stock_component(  
    task_vectors: torch.Tensor,   
    base_tensor: torch.Tensor,  
    filter_wise: bool,  
    device: torch.device  
) -> torch.Tensor:  
    """Calculate Model Stock component based on cosine similarity."""  
    num_models = task_vectors.shape[0]  
      
    # Calculate pairwise cosine similarities  
    cos_thetas = []  
    for i in range(num_models):  
        for j in range(i + 1, num_models):  
            tv_i = task_vectors[i].view(-1)  
            tv_j = task_vectors[j].view(-1)  
              
            cos_theta = F.cosine_similarity(  
                tv_i.unsqueeze(0), tv_j.unsqueeze(0), dim=1  
            ).clamp(-1, 1)  
            cos_thetas.append(cos_theta)  
      
    cos_theta = torch.stack(cos_thetas).mean(dim=0)  
      
    # Calculate interpolation weight  
    t = (num_models * cos_theta) / (1 + (num_models - 1) * cos_theta)  
      
    # Average task vectors and interpolate with base  
    avg_task_vector = task_vectors.mean(dim=0)  
    stock_delta = t * avg_task_vector  
      
    return stock_delta  
  
  
def _calculate_della_component(
    task_vectors: torch.Tensor,
    density: float, 
    epsilon: float,
    mask_dtype: torch.dtype,
    device: torch.device
) -> torch.Tensor:
    """Calculate DELLA component with adaptive magnitude-based pruning."""
    orig_shape = task_vectors.shape
    
    # Flatten to 2D for vectorized rank calculation: (batch_size, features)
    if task_vectors.dim() == 2:
        tv_work = task_vectors.unsqueeze(1)
    else:
        tv_work = task_vectors.view(-1, task_vectors.shape[-1])
        
    magnitudes = torch.abs(tv_work)
    
    # Vectorized rank calculation (Replaces the extremely slow Python for-loop)
    sorted_indices = torch.argsort(magnitudes, dim=-1, descending=False)
    ranks = sorted_indices.argsort(dim=-1).float() + 1
    
    min_ranks = ranks.min(dim=-1, keepdim=True).values
    max_ranks = ranks.max(dim=-1, keepdim=True).values
    
    rank_norm = ((ranks - min_ranks) / (max_ranks - min_ranks + 1e-8)).clamp(0, 1)
    probs = (density - epsilon) + rank_norm * 2 * epsilon
    
    mask = torch.bernoulli(probs.clamp(0, 1)).to(mask_dtype)
    
    # Reshape mask back to original shape
    mask = mask.view(orig_shape)
    
    # Apply mask (Fixes the .unsqueeze(-1) broadcasting crash)
    masked_tvs = task_vectors * mask
    
    # DELLA Rescaling (L1 norm preservation per model)
    dims_to_sum = list(range(1, task_vectors.dim()))
    orig_norm = task_vectors.abs().sum(dim=dims_to_sum, keepdim=True)
    new_norm = masked_tvs.abs().sum(dim=dims_to_sum, keepdim=True)
    rescaled_tvs = masked_tvs * (orig_norm / (new_norm + 1e-8))
    
    della_delta = rescaled_tvs.mean(dim=0)
    
    return della_delta
  
  
def _calculate_sce_component(  
    task_vectors: torch.Tensor,  
    select_topk: float,  
    mask_dtype: torch.dtype,   
    device: torch.device  
) -> torch.Tensor:  
    """Calculate SCE component with variance-based selection and sign consensus."""  
    # Variance-based selection  
    if select_topk < 1.0:  
        variance = torch.var(task_vectors, dim=0, unbiased=False)  
          
        # Select top-k positions by variance  
        nonzero = torch.count_nonzero(variance)  
        k = int(nonzero * select_topk)  
          
        if k > 0:  
            _, indices = torch.topk(variance.abs().view(-1), k=k, largest=True)  
            variance_mask = torch.zeros_like(variance, dtype=mask_dtype)  
            variance_mask.view(-1)[indices] = 1  
            task_vectors = task_vectors * variance_mask.unsqueeze(0)  
      
    # Sign consensus (from SCE)  
    erase_mask = sign_consensus_mask(task_vectors, method="sum", mask_dtype=mask_dtype)  
      
    # Calculate weights based on mean squared magnitude  
    tv_weights = torch.mean(task_vectors**2, dim=list(range(1, task_vectors.dim())))  
    weight_sum = torch.sum(tv_weights).item()  
      
    if abs(weight_sum) < 1e-6:  
        tv_weights = torch.ones_like(tv_weights) / tv_weights.shape[0]  
    else:  
        tv_weights = tv_weights / weight_sum  
      
    # Apply weights and mask  
    while tv_weights.dim() < task_vectors.dim():  
        tv_weights = tv_weights.unsqueeze(-1)  
      
    erased_weights = tv_weights * erase_mask  
    sce_delta = (task_vectors * erased_weights).sum(dim=0)  
    sce_delta = sce_delta / torch.sum(erased_weights, dim=0).clamp(min=1e-6)  
      
    return sce_delta