# mergekit/merge_methods/karcher_stock.py # version 37
# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only
# mergekit/merge_methods/karcher_stock.py by Naphula

import logging
import math
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
from mergekit.merge_methods.rectify_embed import rectify_embed_sizes


# ─────────────────────────────────────────────────────────────────────────────
# Core math helpers
# ─────────────────────────────────────────────────────────────────────────────

def log_karcher_stock_audit(layer_name: str, t_value: float, alphas: torch.Tensor, base_name: str, donor_names: List[str],   
                           cos_theta: float, iters: int, norm_magnitudes: Optional[List[float]] = None,   
                           final_influence: Optional[List[float]] = None):
    """Prints and saves a visual bar chart of Karcher-Stock influence."""  
    # Base influence is (1-t). Donor total influence is t.  
    base_weight = 1.0 - t_value  
    # alphas are already normalized to sum to 1.0  
    donor_weights = (alphas * t_value).tolist()  
      
    bar_char = "█"  
    lines = [f"\n[Karcher_Stock Audit] Layer: {layer_name}"]  
    lines.append(f"Stats: Cos(θ): {cos_theta:.3f} | t-factor: {t_value:.4f} | Karcher Iters: {iters}")  
      
    # Header for three columns  
    lines.append(f"{'':<60} {'Weight':<8} {'Norm':<8} {'Final':<8}")  
    lines.append(f"{'':<60} {'Contrib':<8} {'Mag':<8} {'Influence':<8}")  
      
    # Base Model Bar  
    pct_base = max(0, base_weight * 100)  
    bar_len = int(min(100, pct_base) / 2)  
    clean_base = base_name.split("\\")[-1].split("/")[-1][:60]  
    lines.append(f"  (Base)  {clean_base:<52}: {bar_char * bar_len:<50} ({pct_base:6.2f}%) {'':>16}")  
      
    # Donor Bars  
    for i, (name, weight) in enumerate(zip(donor_names, donor_weights)):  
        pct_donor = max(0, weight * 100)  
        bar_len = int(min(100, pct_donor) / 2)  
        clean_name = name.split("\\")[-1].split("/")[-1][:60]  
          
       # Calculate norm magnitude percentage  
        if norm_magnitudes is not None:  
            norm_pct = norm_magnitudes[i] * 100  
            norm_str = f"{norm_pct:6.2f}%"  
        else:  
            norm_str = " N/A   "  
              
        # Calculate final influence percentage    
        if final_influence is not None:  
            final_pct = final_influence[i] * 100  
            final_str = f"{final_pct:6.2f}%"  
        else:  
            final_str = " N/A   "  
              
        lines.append(f"  (Donor) {clean_name:<52}: {bar_char * bar_len:<50} ({pct_donor:6.2f}%) {norm_str} {final_str}")  
  
    log_entry = "\n".join(lines)  
    print(log_entry)  
    with open("karcher_stock_audit.log", "a", encoding="utf-8") as f:  
        f.write(log_entry + "\n")

def _batched_karcher_mean(
    unit_vecs: torch.Tensor,   # (N, D) or (N, R, D) for filter_wise
    alphas: torch.Tensor,      # (N,)
    max_iter: int,
    tol: float,
    adaptive_iters: Optional[torch.Tensor] = None,  # (,) or (R,) per-row budget
) -> torch.Tensor:
    """
    Weighted Karcher mean on the unit hypersphere.

    unit_vecs: (N, D) — N unit vectors in R^D
                or (N, R, D) — N unit vectors per R rows (filter_wise)
    alphas:    (N,) — normalized weights summing to 1
    adaptive_iters: if provided, per-row iteration budget (filter_wise only)

    Returns: (D,) or (R, D) unit vector(s)
    """
    filter_wise = unit_vecs.dim() == 3
    N = unit_vecs.shape[0]

    if filter_wise:
        # unit_vecs: (N, R, D), alphas: (N,)
        # u: (R, D)
        a = alphas.view(N, 1, 1)  # broadcast over R, D
        u = (a * unit_vecs).sum(dim=0)  # (R, D)
        u_norm = torch.linalg.norm(u.float(), dim=-1, keepdim=True)  # (R, 1)
        u = torch.where(u_norm > tol, u / u_norm.to(u.dtype), unit_vecs[0])

        # active mask: rows that haven't converged
        active = torch.ones(u.shape[0], dtype=torch.bool, device=u.device)  # (R,)
        max_budget = int(adaptive_iters.max().item()) if adaptive_iters is not None else max_iter

        for step in range(max_budget):
            if not active.any():
                break

            # dot: (N, R) = sum over D of u[R,D] * unit_vecs[N,R,D]
            dot = (u.unsqueeze(0) * unit_vecs).sum(dim=-1)  # (N, R)
            dot = dot.clamp(-1.0, 1.0)
            theta = torch.arccos(dot)  # (N, R)
            sin_theta = torch.sin(theta).clamp(min=tol)

            # log map: (N, R, D)
            coeff = (theta / sin_theta).unsqueeze(-1)  # (N, R, 1)
            log_vecs = coeff * (unit_vecs - dot.unsqueeze(-1) * u.unsqueeze(0))

            # weighted tangent vector T: (R, D)
            T = (alphas.view(N, 1, 1) * log_vecs).sum(dim=0)

            # per-row norm of T
            norm_T = torch.linalg.norm(T.float(), dim=-1)  # (R,)

            # convergence check
            converged = norm_T < tol
            if adaptive_iters is not None:
                converged = converged | (step >= adaptive_iters)
            active = active & ~converged

            # exp map update (only active rows)
            safe_norm = norm_T.clamp(min=tol).unsqueeze(-1)  # (R, 1)
            u_new = (
                torch.cos(norm_T).unsqueeze(-1) * u
                + torch.sin(norm_T).unsqueeze(-1) * (T / safe_norm)
            )
            u_new_norm = torch.linalg.norm(u_new.float(), dim=-1, keepdim=True)
            u_new = u_new / u_new_norm.clamp(min=tol).to(u_new.dtype)

            # only update active rows
            u = torch.where(active.unsqueeze(-1), u_new, u)

        return u  # (R, D)

    else:
        # unit_vecs: (N, D), u: (D,)
        u = (alphas.unsqueeze(-1) * unit_vecs).sum(dim=0)  # (D,)
        u_norm = torch.linalg.norm(u.float()).item()
        u = (u / u_norm).to(unit_vecs.dtype) if u_norm > tol else unit_vecs[0].clone()

        budget = int(adaptive_iters.item()) if adaptive_iters is not None else max_iter

        for _ in range(budget):
            dot = (u.unsqueeze(0) * unit_vecs).sum(dim=-1).clamp(-1.0, 1.0)  # (N,)
            theta = torch.arccos(dot)  # (N,)
            sin_theta = torch.sin(theta).clamp(min=tol)

            coeff = (theta / sin_theta).unsqueeze(-1)  # (N, 1)
            log_vecs = coeff * (unit_vecs - dot.unsqueeze(-1) * u.unsqueeze(0))  # (N, D)

            T = (alphas.unsqueeze(-1) * log_vecs).sum(dim=0)  # (D,)
            norm_T = torch.linalg.norm(T.float()).item()

            if norm_T < tol:
                break

            u = (
                math.cos(norm_T) * u
                + math.sin(norm_T) * (T / norm_T)
            ).to(unit_vecs.dtype)
            u_norm = torch.linalg.norm(u.float()).item()
            if u_norm > tol:
                u = (u / u_norm).to(u.dtype)

        return u  # (D,)


def _compute_pairwise_cos_theta(
    offsets: List[torch.Tensor],  # each (D,) or (R, D)
) -> torch.Tensor:
    """
    Compute mean pairwise cosine similarity of task vectors.
    Returns scalar or (R,) tensor for filter_wise.
    """
    cos_thetas = []
    for i in range(len(offsets)):
        for j in range(i + 1, len(offsets)):
            a, b = offsets[i], offsets[j]
            norm_a = torch.linalg.norm(a.float(), dim=-1)
            norm_b = torch.linalg.norm(b.float(), dim=-1)
            denom = (norm_a * norm_b).clamp(min=1e-8)
            cos_ab = ((a.float() * b.float()).sum(dim=-1) / denom).clamp(-1.0, 1.0)
            cos_thetas.append(cos_ab)
    return torch.stack(cos_thetas, dim=0).mean(dim=0)  # scalar or (R,)


def _compute_adaptive_iters(
    cos_theta: torch.Tensor,  # scalar or (R,)
    min_iter: int,
    max_iter: int,
) -> torch.Tensor:
    """
    Maps cos_theta in [-1, 1] → iteration budget in [min_iter, max_iter].
    cos_theta = 1  (identical directions) → min_iter
    cos_theta = -1 (opposing directions)  → max_iter
    """
    fraction = ((1.0 - cos_theta.float()) / 2.0).clamp(0.0, 1.0)
    iters = min_iter + ((max_iter - min_iter) * fraction).round().long()
    return iters


# ─────────────────────────────────────────────────────────────────────────────
# Task
# ─────────────────────────────────────────────────────────────────────────────

class KarcherStockTask(Task[torch.Tensor]):
    gather_tensors: MergeTensorInput
    base_model: ModelReference
    weight_info: WeightInfo
    filter_wise: bool
    max_iter: int
    min_iter: int
    tol: float
    magnitude_mean_geometric: bool
    tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]]

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> torch.Tensor:
        # ── 1. Guard: passthrough if only base model present ──────────────────
        if len(tensors) == 1 and self.base_model in tensors:
            return tensors[self.base_model]
        if len(tensors) < 3:
            if self.weight_info.optional:
                logging.warning(
                    f"Optional weight {self.weight_info.name} not present in "
                    "enough models, discarding"
                )
                return None
            raise ValueError(
                "KarcherStockMerge requires at least 3 models (base + 2 fine-tuned)"
            )

        # ── 2. Rectify embed sizes, separate base from fine-tuned ─────────────
        if self.base_model not in tensors:
            raise ValueError("Base model tensor not found")

        all_weights = [tensors[self.base_model]] + [
            tensors[k] for k in tensors if k != self.base_model
        ]
        rectify_embed_sizes(self.weight_info, all_weights)
        w_0 = all_weights[0].float()
        ws_raw = [w.float() for w in all_weights[1:]]
        fine_tuned_models = [k for k in tensors if k != self.base_model]
        out_shape = all_weights[0].shape
        orig_dtype = all_weights[0].dtype

        # ── 3. Per-model weights (normalized) ─────────────────────────────────
        raw_weights = [
            float(self.tensor_parameters[m]["weight"])
            if m in self.tensor_parameters and "weight" in self.tensor_parameters[m]
            else 1.0
            for m in fine_tuned_models
        ]
        weight_sum = sum(raw_weights)
        alphas = torch.tensor(
            [w / weight_sum for w in raw_weights],
            dtype=torch.float32,
            device=w_0.device,
        )  # (N,)

        # ── 4. Reshape for filter_wise or flatten ─────────────────────────────
        if self.filter_wise and w_0.dim() >= 2:
            # treat each row as an independent filter
            # shape: (R, D) where R = rows, D = cols (or flattened remaining dims)
            R = w_0.shape[0]
            D = w_0[0].numel()
            w_0_2d = w_0.reshape(R, D)
            ws_2d = [w.reshape(R, D) for w in ws_raw]
        else:
            # flatten everything
            w_0_2d = w_0.reshape(1, -1)   # (1, D) — treat as single "row"
            ws_2d = [w.reshape(1, -1) for w in ws_raw]

        # ── 5. Task vectors and their norms ───────────────────────────────────
        offsets = [w - w_0_2d for w in ws_2d]  # list of (R, D)
        norms = torch.stack(
            [torch.linalg.norm(v.float(), dim=-1) for v in offsets], dim=0
        )  # (N, R)

        # ── 6. Weighted mean norm L_bar ───────────────────────────────────────
        if self.magnitude_mean_geometric:
            # Geometric Mean Logic
            safe_norms_for_log = norms.clamp(min=1e-8)
            log_norms = torch.log(safe_norms_for_log)
            L_bar = torch.exp((alphas.unsqueeze(-1) * log_norms).sum(dim=0))  # (R,)
        else:
            # Standard v12 Arithmetic Mean
            L_bar = (alphas.unsqueeze(-1) * norms).sum(dim=0)  # (R,)

        # ── 7. Normalize task vectors to unit sphere ──────────────────────────
        safe_norms = norms.clamp(min=self.tol).unsqueeze(-1)  # (N, R, 1)
        unit_offsets = torch.stack(offsets, dim=0) / safe_norms  # (N, R, D)

        # zero-norm task vectors: replace unit vector with zeros
        # (they'll contribute nothing to the Karcher mean via alpha weighting,
        #  but we need a valid unit vector placeholder)
        zero_mask = (norms < self.tol).unsqueeze(-1)  # (N, R, 1)
        unit_offsets = torch.where(zero_mask, torch.zeros_like(unit_offsets), unit_offsets)

        # ── 8. Pairwise cosine similarity of task vectors ─────────────────────
        # offsets_list: list of (R, D)
        cos_theta = _compute_pairwise_cos_theta(offsets)  # (R,) or scalar

        # ── 9. Adaptive iteration budget ──────────────────────────────────────
        adaptive_iters = _compute_adaptive_iters(
            cos_theta, self.min_iter, self.max_iter
        )  # (R,) or scalar

        # ── 10. Karcher mean of unit task vectors ─────────────────────────────
        # unit_offsets: (N, R, D)
        # For non-filter_wise, R=1, so we squeeze after
        if self.filter_wise and w_0.dim() >= 2:
            u_star = _batched_karcher_mean(
                unit_offsets,          # (N, R, D)
                alphas,
                max_iter=self.max_iter,
                tol=self.tol,
                adaptive_iters=adaptive_iters,  # (R,)
            )  # (R, D)
        else:
            # unit_offsets: (N, 1, D) → squeeze to (N, D)
            u_star = _batched_karcher_mean(
                unit_offsets.squeeze(1),   # (N, D)
                alphas,
                max_iter=self.max_iter,
                tol=self.tol,
                adaptive_iters=adaptive_iters,  # scalar
            )  # (D,)
            u_star = u_star.unsqueeze(0)   # (1, D) to match w_0_2d shape

        # ── 11. Model Stock t factor with Adaptive Soft-Clamp ─────────────
        N = len(ws_2d)
        ct = cos_theta.unsqueeze(-1) if cos_theta.dim() > 0 else cos_theta
        
        # Raw Model Stock formula
        denom = 1.0 + (N - 1) * ct
        # Add a tiny epsilon to prevent literal division by zero
        t_raw = (N * ct) / denom.clamp(min=1e-6) 

        # --- BULLETPROOF TANH CLAMP ---
        # 1. Prevent negative infinity spikes (fallback to base model)
        t_clamped_bottom = torch.clamp(t_raw, min=0.0)
        
        # 2. Smoothly asymptote positive spikes to L (Maximum allowed t-factor)
        L = 1.5 
        excess = torch.clamp(t_clamped_bottom - 1.0, min=0.0)
        t_soft_top = 1.0 + (L - 1.0) * torch.tanh(excess / (L - 1.0))
        
        # 3. Apply: If t <= 1.0, use exact math. If t > 1.0, use soft curve.
        t = torch.where(t_clamped_bottom <= 1.0, t_clamped_bottom, t_soft_top)
        # ------------------------------

        # ── 12. Final merge ────────────────────────────────────────────────────
        # w_0_2d: (R, D), u_star: (R, D), L_bar: (R,), t: (R, 1)
        L_bar_2d = L_bar.unsqueeze(-1)  # (R, 1)
        w_h = w_0_2d + t * L_bar_2d * u_star  # (R, D)

        # --- V12 AUDIT ---  
        avg_cos = cos_theta.mean().item() if isinstance(cos_theta, torch.Tensor) else cos_theta  
        avg_t = t.mean().item() if isinstance(t, torch.Tensor) else t  
        avg_iters = adaptive_iters.float().mean().item() if isinstance(adaptive_iters, torch.Tensor) else adaptive_iters  
          
        # Get donor names in the same order as alphas  
        donor_names = [str(m.model.path) for m in fine_tuned_models]  
  
        # Calculate norm magnitude contributions (normalized per row)
        # norms: (N, R), norm_sum: (R,)
        norm_sum = norms.sum(dim=0, keepdim=True) + 1e-8
        norm_contributions = norms / norm_sum  # (N, R)
          
        # Compositional normalization: Ensure Base % + Donor % = 100%
        # t represents the total share of the donor pool.
        # (alphas * norm_contributions) represents how that share is split among donors.
        t_val = t.mean().item() if isinstance(t, torch.Tensor) else t
        
        # The 'norms' tensor now contains the already-suppressed magnitudes.
        # We calculate the share of the 'Energy Pool' based on these suppressed values.
        norm_sum = norms.sum(dim=0, keepdim=True) + 1e-8
        actual_compositional_share = norms / norm_sum # (N, R)
        
        if self.filter_wise and w_0.dim() >= 2:
            # Calculate relative share within the donor pool
            rel_donor_share = actual_compositional_share.mean(dim=1)
            # Scale by t to get percentage of the TOTAL model
            final_influence = (rel_donor_share * t_val).cpu().tolist()
            norm_magnitudes = actual_compositional_share.mean(dim=1).cpu().tolist()
        else:
            rel_donor_share = actual_compositional_share.flatten()
            final_influence = (rel_donor_share * t_val).cpu().tolist()
            norm_magnitudes = actual_compositional_share.flatten().cpu().tolist()
  
        log_karcher_stock_audit(  
            layer_name=self.weight_info.name,  
            t_value=avg_t,  
            alphas=alphas,  
            base_name=str(self.base_model.model.path),  
            donor_names=donor_names,  
            cos_theta=avg_cos,  
            iters=int(avg_iters),  
            norm_magnitudes=norm_magnitudes,  
            final_influence=final_influence  
        )  
        # ---------------------------------

        return w_h.reshape(out_shape).to(orig_dtype)

    def group_label(self) -> Optional[str]:
        return self.gather_tensors.group_label()


# ─────────────────────────────────────────────────────────────────────────────
# Method
# ─────────────────────────────────────────────────────────────────────────────

class KarcherStockMerge(MergeMethod):
    def name(self) -> str:
        return "karcher_stock"

    @override
    def pretty_name(self) -> Optional[str]:
        return "Karcher-Stock"

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="filter_wise", required=False, default_value=False),
            ConfigParameterDef(name="max_iter",    required=False, default_value=300),
            ConfigParameterDef(name="min_iter",    required=False, default_value=5),
            ConfigParameterDef(name="tol",         required=False, default_value=1e-7),
            ConfigParameterDef(name="magnitude_mean_geometric", required=False, default_value=False),
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="weight", required=False, default_value=1.0),
        ]

    def make_task(
        self,
        *,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        base_model: Optional[ModelReference],
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        **_kwargs,
    ) -> Task:
        return KarcherStockTask(
            gather_tensors=tensors,
            base_model=base_model,
            weight_info=output_weight,
            filter_wise=parameters["filter_wise"],
            max_iter=parameters["max_iter"],
            min_iter=parameters["min_iter"],
            tol=parameters["tol"],
            magnitude_mean_geometric=parameters["magnitude_mean_geometric"],
            tensor_parameters=tensor_parameters,
        )