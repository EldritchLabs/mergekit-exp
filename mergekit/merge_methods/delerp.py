# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

from typing import List, Optional      
import numpy as np    
import torch      
from mergekit.merge_methods.easy_define import merge_method  
    
@merge_method(    
    name="delerp",    
    pretty_name="Decomposed Linear Interpolation",    
    reference_url="https://huggingface.co/blog/grimjim/delerp-merge-method"    
)    
def delerp(    
    tensors: List[torch.Tensor],    
    t: float,    
    base_tensor: Optional[torch.Tensor] = None,    
    eps: float = 1e-8,    
) -> torch.Tensor:    
    """  
    Decomposed Linear Interpolation (DeLERP)
    Created by GrimJim, Modified by Naphula to fix various bugs
      
    Separates direction and magnitude: direction via NLERP (Normalized Linear  
    Interpolation), magnitude via maximum norm. Operates on each weight  
    parameter, treating the entire tensor as a vector in high-dimensional space  
    for norm calculations.  
      
    Direction: NLERP on unit sphere (coordinate-free geometric interpolation)  
    Magnitude: max(||v0||, ||v1||) to preserve stronger importance signal  
      
    Args:  
        tensors: List of input tensors (1 if base_tensor provided, 2 otherwise)  
        t (float): Float value between 0.0 and 1.0  
        base_tensor: Optional base model tensor  
        eps (float): Small epsilon for numerical stability  
    Returns:  
        torch.Tensor: Interpolation vector between v0 and v1  
    """  
    # Handle base_tensor case  
    if base_tensor is not None:  
        if len(tensors) != 1:  
            raise ValueError(f"delerp with base_tensor requires exactly 1 tensor, got {len(tensors)}")  
        v0 = base_tensor  
        v1 = tensors[0]  
    else:  
        # No base_tensor - expect exactly 2 tensors  
        if len(tensors) != 2:  
            raise ValueError(f"delerp requires exactly 2 tensors, got {len(tensors)}")  
        v0, v1 = tensors[0], tensors[1]  
        
    # In delerp() function, after getting v0 and v1 tensors:  
    is_torch = False    
    if not isinstance(v0, np.ndarray):    
        is_torch = True    
        v0 = v0.detach().cpu().float().numpy()    
    if not isinstance(v1, np.ndarray):    
        is_torch = True    
        v1 = v1.detach().cpu().float().numpy()    
  
    # Handle shape mismatches for embedding tensors    
    if v0.shape != v1.shape:    
        # Take common submatrix for embedding weights    
        min_shape = tuple(min(s0, s1) for s0, s1 in zip(v0.shape, v1.shape))    
    else:    
        min_shape = v0.shape  # Initialize when shapes match  

    if v0.shape != min_shape:    
        v0 = v0[:min_shape[0], :min_shape[1]] if len(min_shape) == 2 else v0[:min_shape[0]]    
    if v1.shape != min_shape:    
        v1 = v1[:min_shape[0], :min_shape[1]] if len(min_shape) == 2 else v1[:min_shape[0]]

    # Compute norms (treating entire tensor as single vector)    
    norm_v0 = np.linalg.norm(v0)  
    norm_v1 = np.linalg.norm(v1)  
      
    # Handle zero vectors  
    if norm_v0 < eps and norm_v1 < eps:  
        return maybe_torch(np.zeros_like(v0), is_torch)  
      
    # NLERP: Linear interpolation followed by normalization  
    d_merged = (1 - t) * v0 + t * v1  
    norm_d_merged = np.linalg.norm(d_merged)  
      
    if norm_d_merged > eps:  
        # Normalize to get unit direction vector  
        d_merged = d_merged / norm_d_merged  
    else:  
        # Vectors cancel out - fallback to linear interpolation  
        # This handles the edge case where v0 and v1 are nearly opposite  
        return maybe_torch((1 - t) * v0 + t * v1, is_torch)  
      
    # Max norm: Preserve the stronger magnitude signal  
    m_merged = max(norm_v0, norm_v1)  
      
    # Recombine: direction * magnitude  
    result = d_merged * m_merged  
      
    return maybe_torch(result, is_torch)  
      
def maybe_torch(v: np.ndarray, is_torch: bool):    
    if is_torch:    
        return torch.from_numpy(v)    
    return v