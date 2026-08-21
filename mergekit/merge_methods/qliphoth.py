# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# mergekit/merge_methods/qliphoth_v2.py by Naphula

"""QLIPHOTH v2 - Conservative Orthogonal Injection with Stability Safeguards"""  
  
import logging  
from typing import List, Optional  
import torch  
from mergekit.merge_methods.easy_define import merge_method  
  
LOG = logging.getLogger(__name__)  
  
def _conservative_null_project(  
    tau_p: torch.Tensor,  
    rival_matrix: torch.Tensor,  
    k: int,  
    preserve_ratio: float = 0.7,  
    eps: float = 1e-8,  
) -> torch.Tensor:  
    """Conservative projection that preserves essential dimensions."""  
    if rival_matrix.shape[0] == 1:  
        # Single rival: simple rejection with preservation  
        mu = rival_matrix[0]  
        dot_vu = torch.dot(tau_p, mu)  
        dot_uu = torch.dot(mu, mu).clamp(min=eps)  
        projection = (dot_vu / dot_uu) * mu  
        # Preserve 70% of original, remove 30% projection  
        return tau_p - preserve_ratio * projection  
      
    # Compute thin SVD  
    U, S, Vh = torch.linalg.svd(rival_matrix, full_matrices=False)  
      
    # Limit rank to preserve more dimensions  
    max_k = max(1, int(k * preserve_ratio))  
    V_top = Vh[:max_k, :]  
      
    # Conservative projection  
    projection = V_top.T @ (V_top @ tau_p)  
    return tau_p - 0.5 * projection  # Reduced projection strength  
  
def _adaptive_variance_scale(  
    V_flat: torch.Tensor,  
    tau_p_perp: torch.Tensor,  
    base_gamma: float,  
    max_scale: float = 3.0,  
) -> torch.Tensor:  
    """Adaptive scaling that prevents explosion."""  
    # Normalize variance  
    V_norm = V_flat / (V_flat.mean().clamp(min=1e-8))  
      
    # Adaptive gamma based on orthogonal vector magnitude  
    ortho_norm = tau_p_perp.norm()  
    base_norm = V_flat.norm()  
      
    if ortho_norm > 0 and base_norm > 0:  
        adaptive_gamma = base_gamma * min(1.0, base_norm / ortho_norm)  
    else:  
        adaptive_gamma = base_gamma  
      
    # Clamp scaling to prevent explosion  
    variance_scale = 1.0 + adaptive_gamma * V_norm  
    return torch.clamp(variance_scale, max=max_scale)  
  
def _preserve_generation_weights(  
    result: torch.Tensor,  
    base: torch.Tensor,  
    max_deviation_ratio: float = 0.1,  # Accept parameter
    preserve_layers: List[str] = None,  
) -> torch.Tensor:  
    """Ensure critical generation layers remain functional."""  
    # This would be called for output layers specifically  
    # For now, ensure we don't deviate too far from base  
    deviation = result - base  
    # Use the passed ratio
    max_deviation = base.norm() * max_deviation_ratio  
      
    if deviation.norm() > max_deviation:  
        # Scale down deviation 
        scale = max_deviation / deviation.norm()  
        result = base + scale * deviation  
      
    return result
  
@merge_method(  
    name="qliphoth",  
    pretty_name="QLIPHOTH v2",  
    reference_url=None,  
)  
def qliphoth_merge(  
    tensors: List[torch.Tensor],  
    base_tensor: torch.Tensor,  
    pinocchio: List[float],  
    gamma: float = 0.5,  # Reduced default  
    lambda_enslaved: float = 0.05,  # Reduced default  
    svd_rank: int = 2,  # More conservative  
    use_svd: bool = True,  
    normalize_variance: bool = True,  
    pinocchio_weight: float = 0.8,  # Reduced  
    rival_weight: float = 1.0,  
    preserve_ratio: float = 0.7,  # New parameter  
    stability_check: bool = True,  # New parameter  
    max_deviation: float = 0.1,  # New YAML parameter
) -> torch.Tensor:  
    """QLIPHOTH v2: Conservative orthogonal injection with safeguards."""  
    if not tensors:  
        return base_tensor  
      
    base = base_tensor.float()  
    orig_dtype = base_tensor.dtype  
      
    # Separate models  
    pinocchio_tvs = []  
    rival_tvs_list = []  
      
    for tensor, p_flag in zip(tensors, pinocchio):  
        tv = tensor.float() - base  
        if p_flag > 0.5:  
            pinocchio_tvs.append(tv)  
        else:  
            rival_tvs_list.append(tv)  
      
    if not pinocchio_tvs:  
        LOG.warning("QLIPHOTH v2: No Pinocchio model found. Returning base.")  
        return base_tensor  
      
    tau_p = torch.stack(pinocchio_tvs, dim=0).mean(dim=0)  
    tau_p_flat = tau_p.view(-1)  
      
    if not rival_tvs_list:  
        LOG.warning("QLIPHOTH v2: No rival models. Conservative Pinocchio addition.")  
        result = base + pinocchio_weight * tau_p  
        return result.to(orig_dtype)  
      
    rival_matrix = torch.stack([tv.view(-1) for tv in rival_tvs_list], dim=0)  
    N = rival_matrix.shape[0]  
      
    # Step 1: Rival consensus  
    mu_R_flat = rival_matrix.mean(dim=0)  
      
    # Step 2: Conflict map  
    V_flat = rival_matrix.var(dim=0, unbiased=False)  
    if normalize_variance:  
        V_norm = V_flat / (V_flat.mean().clamp(min=1e-8))  
    else:  
        V_norm = V_flat  
      
    # Step 3: Conservative orthogonal projection  
    if use_svd and N > 1:  
        k = min(svd_rank, N)  
        tau_p_perp_flat = _conservative_null_project(  
            tau_p_flat, rival_matrix, k=k, preserve_ratio=preserve_ratio  
        )  
    else:  
        # Simple rejection with preservation  
        dot_vu = torch.dot(tau_p_flat, mu_R_flat)  
        dot_uu = torch.dot(mu_R_flat, mu_R_flat).clamp(min=1e-8)  
        projection = (dot_vu / dot_uu) * mu_R_flat  
        tau_p_perp_flat = tau_p_flat - 0.5 * projection  
      
    # Step 4: Adaptive variance scaling  
    variance_scale = _adaptive_variance_scale(V_flat, tau_p_perp_flat, gamma)  
    tau_p_amplified_flat = tau_p_perp_flat * variance_scale  
      
    # Step 5: Reduced Aikido flip  
    tau_enslaved_flat = torch.zeros_like(tau_p_flat)  
    if lambda_enslaved > 0 and N > 1:  
        # Simplified enslavement with reduced impact  
        rival_signs = torch.sign(rival_matrix)  
        consensus_sign = torch.sign(mu_R_flat)  
        minority_mask = (rival_signs != consensus_sign.unsqueeze(0)) & (rival_signs != 0)  
          
        p_sign = torch.sign(tau_p_perp_flat)  
        enslaved_raw = (rival_matrix.abs() * minority_mask.float() * p_sign.unsqueeze(0)).sum(dim=0)  
        count = minority_mask.float().sum(dim=0).clamp(min=1.0)  
        tau_enslaved_flat = (enslaved_raw / count) * 0.3  # Reduced impact  
      
    # Step 6: Conservative synthesis  
    result_flat = (  
        base.view(-1) +   
        rival_weight * mu_R_flat +   
        pinocchio_weight * tau_p_amplified_flat +   
        lambda_enslaved * tau_enslaved_flat  
    )  
      
    # Stability check  
    if stability_check:  
        result = _preserve_generation_weights(  
            result_flat.reshape_as(base), 
            base,
            max_deviation_ratio=max_deviation  # Pass YAML value here
        )  
    else:  
        result = result_flat.reshape_as(base)  
      
    return result.to(orig_dtype)