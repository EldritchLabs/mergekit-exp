# Copyright (C) 2025 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only  
# multi_fusion.py version 4 by Naphula

import math
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
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

EPS = 1e-8


def _is_router_weight(name: str) -> bool:
    """Detects gating and routing matrices across MoE architectures."""
    return (
        "block_sparse_moe.gate.weight" in name
        or "mlp.gate.weight" in name
        or "shared_expert_gate.weight" in name
    )


def _is_expert_weight(name: str) -> Optional[int]:
    """Detects whether this weight tensor belongs to an MoE expert branch."""
    if ".experts." in name or ".block_sparse_moe.experts." in name:
        return 0
    match = re.search(r"experts\.(\d+)\.", name)
    if match:
        return int(match.group(1))
    return None


def _get_param_val(p_map: Any, key: str, default: Any = None) -> Any:
    """Safely extracts a key from dicts or MergeKit ImmutableMap objects."""
    if p_map is None:
        return default
    if hasattr(p_map, "get"):
        return p_map.get(key, default)
    if hasattr(p_map, "__contains__") and key in p_map:
        return p_map[key]
    return default


class GlobalMetricTracker:
    """Thread-safe persistent tracker for metric win frequency across all layers."""
    lock = threading.Lock()
    METRICS = ["delta_mag", "kl_div", "cosine_sim", "variance_sce", "fisher_grad"]
    ABBR = {"delta_mag": "MAG", "kl_div": "KL", "cosine_sim": "COS", "variance_sce": "SCE", "fisher_grad": "FSH"}
    
    total_layers_optimized: int = 0
    metric_win_counts: Dict[str, int] = {m: 0 for m in METRICS}
    cumulative_metric_weights: Dict[str, float] = {m: 0.0 for m in METRICS}
    FILE_PATH = "multi_fusion_metric_stats.json"

    @classmethod
    def record_win(cls, best_blend_weights: List[float]):
        with cls.lock:
            cls.total_layers_optimized += 1
            # Primary winner is metric with highest coefficient in Dirichlet blend
            max_idx = int(torch.tensor(best_blend_weights).argmax().item())
            primary_winner = cls.METRICS[max_idx]
            cls.metric_win_counts[primary_winner] += 1

            for m, w in zip(cls.METRICS, best_blend_weights):
                cls.cumulative_metric_weights[m] += float(w)

            # Persist running statistics
            try:
                data = {
                    "total_layers": cls.total_layers_optimized,
                    "win_counts": cls.metric_win_counts,
                    "cumulative_weights": {k: round(v, 4) for k, v in cls.cumulative_metric_weights.items()},
                    "win_percentages": {
                        cls.ABBR[k]: f"{(v / max(1, cls.total_layers_optimized)) * 100:.1f}%"
                        for k, v in cls.metric_win_counts.items()
                    }
                }
                with open(cls.FILE_PATH, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception:
                pass

    @classmethod
    def get_summary_str(cls) -> str:
        with cls.lock:
            tot = max(1, cls.total_layers_optimized)
            chunks = []
            for m in cls.METRICS:
                abbr = cls.ABBR[m]
                cnt = cls.metric_win_counts[m]
                pct = (cnt / tot) * 100.0
                chunks.append(f"{abbr}:{cnt}({pct:.0f}%)")
            return " | ".join(chunks)


def log_multi_fusion_audit(
    layer_name: str,
    base_name: str,
    donor_names: List[str],
    delta_norms: List[float],
    mask_pass_rates: List[float],
    effective_shares: List[float],
    metric_str: str,
    tukey_fence: float,
    winning_trial_info: Optional[str] = None,
    winning_blend_str: Optional[str] = None,
):
    """Prints and persists a live visual influence audit of each donor."""
    bar_char = "█"
    global_stats = GlobalMetricTracker.get_summary_str()
    tot_layers = GlobalMetricTracker.total_layers_optimized

    lines = [
        f"\n╔{'═'*92}╗",
        f"║ [MultiFusion v4 Audit] {layer_name[:66]:<68} ║",
        f"║ Base: {base_name[:34]:<34} | Mode: {metric_str[:22]:<22} | Fence: {tukey_fence:4.2f}   ║",
    ]

    if winning_trial_info:
        lines.append(f"║ Pareto Telemetry: {winning_trial_info[:70]:<70} ║")
    if winning_blend_str:
        lines.append(f"║ Winning Blend:    {winning_blend_str[:70]:<70} ║")
    
    lines.append(f"║ Global Wins [{tot_layers:>3}]: {global_stats[:67]:<67} ║")

    lines.extend([
        f"╠{'═'*92}╣",
        f"║ {'Donor Model':<38} | {'Norm':<6} | {'Pass%':<6} | {'Share%':<6} | {'Effective Impact':<24} ║",
        f"╟{'-'*38}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*26}╢",
    ])

    tot_share = sum(effective_shares) + EPS
    for name, d_norm, p_rate, eff in zip(donor_names, delta_norms, mask_pass_rates, effective_shares):
        pct = (eff / tot_share) * 100.0
        bar_len = min(22, int(pct / 4.54))
        bar = bar_char * bar_len
        clean_name = name.split("\\")[-1].split("/")[-1][:38]
        lines.append(
            f"║ {clean_name:<38} | {d_norm:6.2f} | {p_rate*100:5.1f}% | {pct:5.1f}% | {bar:<22} ║"
        )

    lines.append(f"╚{'═'*90}╝\n")
    log_entry = "\n".join(lines)
    print(log_entry)

    try:
        with open("multi_fusion_audit.log", "a", encoding="utf-8") as f:
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"--- [{ts}] ---\n{log_entry}\n")
    except Exception:
        pass
  
  
class DynamicThresholdFusion:
    def approximate_quantiles(self, tensor: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        flat_tensor = tensor.view(-1)
        if flat_tensor.numel() > 1e6:
            # Deterministic strided subsampling is faster and avoids GPU RNG synchronization locks
            step = max(1, flat_tensor.numel() // 1000000)
            flat_tensor = flat_tensor[::step]

        sorted_tensor, _ = torch.sort(flat_tensor)
        quantile_indices = (q.to(sorted_tensor.device) * (sorted_tensor.numel() - 1)).long()
        return sorted_tensor[quantile_indices]

    def calculate_dynamic_threshold(self, importance_scores: torch.Tensor, tukey_fence: float = 1.5) -> torch.Tensor:
        q_vals = self.approximate_quantiles(importance_scores, torch.tensor([0.25, 0.50, 0.75]))
        q1, median, q3 = q_vals[0], q_vals[1], q_vals[2]
        iqr = q3 - q1
        return median + tukey_fence * iqr

    def compute_fusion_mask(
        self, importance_scores: torch.Tensor, tukey_fence: float = 1.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        threshold = self.calculate_dynamic_threshold(importance_scores, tukey_fence)
        fusion_mask = (importance_scores >= threshold).float()
        return fusion_mask, threshold


class HierarchicalMonteCarloParetoEngine:
    """
    Evaluates candidate fusion masks across multi-resolution powers of 2.
    Runs synthetic forward projections to benchmark signal recovery,
    base manifold preservation, and noise amplification.
    """

    @staticmethod
    def sample_power_of_two_subspace(
        donor: torch.Tensor, base: torch.Tensor, delta: torch.Tensor, sample_size: int = 4096
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        numel = delta.numel()
        target_len = min(numel, sample_size)
        
        # Power-of-2 uniform strided slice to sample global matrix behavior
        stride = max(1, numel // target_len)
        indices = torch.arange(0, target_len * stride, stride, device=delta.device)[:target_len]

        d_sub = donor.flatten()[indices].float()
        b_sub = base.flatten()[indices].float()
        delta_sub = delta.flatten()[indices].float()
        return d_sub, b_sub, delta_sub

    @staticmethod
    def simulate_synthetic_forward_benchmark(
        fused_sub: torch.Tensor,
        base_sub: torch.Tensor,
        donor_sub: torch.Tensor,
        delta_sub: torch.Tensor,
        mask_sub: torch.Tensor,
    ) -> float:
        """
        Runs synthetic forward projection benchmarking signal recovery vs noise:
          1. Base Manifold Directional Drift (Cosine preservation)
          2. Donor Novelty Signal Recovery (L1/L2 alignment)
          3. Noise Amplification (High-frequency kurtosis / outlier energy)
        """
        # A. Directional Alignment
        cos_drift = 1.0 - F.cosine_similarity(fused_sub.unsqueeze(0), base_sub.unsqueeze(0), eps=EPS).item()

        # B. Novelty Signal Recovery: measures whether high-magnitude donor features survived
        intended_novelty = donor_sub - base_sub
        realized_novelty = fused_sub - base_sub
        norm_intended = intended_novelty.norm().clamp_min(EPS)
        recovery_err = (intended_novelty - realized_novelty).norm().item() / norm_intended.item()

        # C. Noise Amplification: measures variance inflation in regions where mask is inactive
        inactive_variance = torch.var(fused_sub * (1.0 - mask_sub), unbiased=False).item()
        base_variance = torch.var(base_sub, unbiased=False).clamp_min(EPS).item()
        noise_ratio = abs(inactive_variance - base_variance) / base_variance

        # D. Sparsity Prior: Prevents all-zero or all-one degeneracy
        mean_density = mask_sub.mean().item()
        sparsity_loss = (1.0 / (mean_density + 0.05)) + (1.0 / (1.05 - mean_density))

        # Pareto Multi-Objective Compound Loss (Lower is better)
        pareto_loss = (
            2.5 * cos_drift +
            1.2 * recovery_err +
            0.4 * noise_ratio +
            0.02 * sparsity_loss
        )
        return float(pareto_loss)
  
  
class MultiFusionMergeTask(Task[torch.Tensor]):
    gather_tensors: MergeTensorInput
    base_model: ModelReference
    weight_info: WeightInfo
    tensor_parameters: ImmutableMap[ModelReference, Any]
    importance_metric: str = "pareto_auto"
    tukey_fence: float = 1.5
    is_moe: bool = True
    router_strategy: str = "metric"
    router_metric: str = "kl_div"
    blend_experts: bool = True
    sce_inversion_mode: bool = False
    anti_monopoly_power: float = 1.0
    mc_trials: int = 16
    power_of_two_scale: int = 4096

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.gather_tensors}

    def execute(self, tensors: Dict[ModelReference, torch.Tensor]) -> torch.Tensor:
        if not tensors or any(t is None for t in tensors.values()):
            return None

        if self.base_model not in tensors:
            raise RuntimeError(f"Base model '{self.base_model}' missing from input tensors.")

        base_t = tensors[self.base_model]
        donor_refs = [m for m in tensors.keys() if m != self.base_model]

        if len(donor_refs) == 0:
            return base_t

        # Shape reconciliation across all donors
        all_tensors = [base_t] + [tensors[m] for m in donor_refs]
        rectify_embed_sizes(self.weight_info, all_tensors)
        base_t = all_tensors[0]
        donor_tensors = all_tensors[1:]

        weight_name = self.weight_info.name
        is_router = _is_router_weight(weight_name)
        is_expert = _is_expert_weight(weight_name) is not None

        # --- MoE Gating & Routing Logic ---
        if is_router and self.is_moe:
            if self.router_strategy == "average":
                return torch.stack([t.float() for t in all_tensors]).mean(dim=0).to(base_t.dtype)
            elif self.router_strategy == "first" or self.router_strategy == "base":
                return base_t
            elif self.router_strategy == "random_init":
                return torch.randn_like(base_t) * 0.02

        if is_expert and self.is_moe and not self.blend_experts:
            return base_t

        active_metric = self.router_metric if (is_router and self.is_moe) else self.importance_metric

        # Extract normalized per-donor weights
        donor_weights = []
        for m in donor_refs:
            p = _get_param_val(self.tensor_parameters, m, None)
            w = _get_param_val(p, "weight", 1.0)
            donor_weights.append(float(w))
        w_sum = sum(donor_weights) + EPS
        norm_donor_weights = [w / w_sum for w in donor_weights]

        # Multi-Donor Mask & Outlier Synthesis
        fused = self._synthesize_multi_donor_fusion(
            donor_tensors, base_t, norm_donor_weights, active_metric
        )
        return fused.to(base_t.dtype)

    def _synthesize_multi_donor_fusion(
        self,
        donors: List[torch.Tensor],
        base: torch.Tensor,
        donor_weights: List[float],
        metric: str,
    ) -> torch.Tensor:
        num_donors = len(donors)
        dtf = DynamicThresholdFusion()
        base_f = base.float()

        # Compute raw task vectors: tau_i = W_donor_i - W_base
        task_vectors = torch.stack([d.float() - base_f for d in donors], dim=0)

        # Anti-Monopoly Energy Normalization
        if self.anti_monopoly_power < 1.0:
            norms = torch.stack([torch.norm(tv).clamp_min(EPS) for tv in task_vectors])
            median_norm = torch.median(norms)
            scaling = torch.pow(median_norm / norms, 1.0 - self.anti_monopoly_power)
            while scaling.dim() < task_vectors.dim():
                scaling = scaling.unsqueeze(-1)
            task_vectors = task_vectors * scaling

        # Optional MAGIC Aikido Minority Sign Inversion
        if self.sce_inversion_mode:
            maj_sign = torch.sign(task_vectors.sum(dim=0))
            disagreement = (task_vectors.sign() != maj_sign.unsqueeze(0)) & (task_vectors != 0)
            task_vectors = torch.where(disagreement, -task_vectors, task_vectors)

        if metric == "pareto_auto":
            return self._solve_hierarchical_mc_pareto_multi(task_vectors, base_f, donor_weights)

        # Compute importance masks per donor & capture audit metrics
        fused_delta = torch.zeros_like(base_f)
        total_weight_mask = torch.zeros_like(base_f)
        
        delta_norms = []
        mask_pass_rates = []
        effective_shares = []

        for i in range(num_donors):
            tv = task_vectors[i]
            d_tensor = donors[i]
            imp = self._compute_importance(d_tensor, base, tv, metric)
            mask, _ = dtf.compute_fusion_mask(imp, tukey_fence=self.tukey_fence)

            w_i = donor_weights[i]
            contribution = w_i * (tv * mask)
            fused_delta += contribution
            total_weight_mask += w_i * mask

            # Telemetry metrics
            delta_norms.append(torch.norm(tv.flatten()).item())
            mask_pass_rates.append(mask.mean().item())
            effective_shares.append(torch.norm(contribution.flatten()).item())

        # Safe divisor normalization (fallback to 1 where mask is unselected)
        total_weight_mask[total_weight_mask == 0] = 1.0
        result = base_f + (fused_delta / total_weight_mask)

        # Trigger Live Audit HUD for non-Pareto discrete metric passes
        w_name = self.weight_info.name
        if "mlp" in w_name or "self_attn" in w_name or "embed" in w_name or "gate" in w_name:
            clean_donor_names = [
                str(m.model.path).split("\\")[-1].split("/")[-1] 
                for m in list(self.tensor_parameters.keys() if self.tensor_parameters else []) if m != self.base_model
            ]
            base_str = str(self.base_model.model.path).split("\\")[-1].split("/")[-1]

            # Record discrete metric win in the cumulative tracker
            single_blend = [1.0 if m == metric else 0.0 for m in GlobalMetricTracker.METRICS]
            GlobalMetricTracker.record_win(single_blend)

            trial_info = f"Discrete Mode: Single Metric Locked ({metric.upper()})"
            blend_str = f"100% {metric.upper()} Primary Mask Formulation"

            log_multi_fusion_audit(
                layer_name=w_name,
                base_name=base_str,
                donor_names=clean_donor_names,
                delta_norms=delta_norms,
                mask_pass_rates=mask_pass_rates,
                effective_shares=effective_shares,
                metric_str=metric,
                tukey_fence=self.tukey_fence,
                winning_trial_info=trial_info,
                winning_blend_str=blend_str,
            )

        return result

    def _resolve_fusion_mask_and_delta(
        self, donor: torch.Tensor, base: torch.Tensor, metric: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        delta = donor.float() - base.float()
        dtf = DynamicThresholdFusion()

        # MAGIC Minority Sign Inversion
        delta_working = delta
        if "variance_sce" in metric and self.sce_inversion_mode:
            sign_disagreement = (donor.sign() != base.sign()) & (donor != 0) & (base != 0)
            delta_working = torch.where(sign_disagreement, -delta, delta)

        if metric == "pareto_auto":
            return self._solve_hierarchical_mc_pareto_mask(donor, base, delta_working)

        importance_scores = self._compute_importance(donor, base, delta_working, metric)
        fusion_mask, _threshold = dtf.compute_fusion_mask(importance_scores, tukey_fence=self.tukey_fence)
        return fusion_mask, delta_working

    def _compute_importance(
        self, donor: torch.Tensor, base: torch.Tensor, delta: torch.Tensor, metric: str
    ) -> torch.Tensor:
        if metric == "kl_div":
            return self._compute_kl_div_importance(donor, base)
        elif metric == "delta_mag":
            return self._compute_delta_mag_importance(delta)
        elif metric == "cosine_sim":
            return self._compute_cosine_sim_importance(donor, base, delta)
        elif metric == "variance_sce":
            return self._compute_variance_sce_importance(delta)
        elif metric == "fisher_grad":
            return self._compute_karcher_fisher_importance(donor, base)
        else:
            raise ValueError(f"Unknown importance metric: {metric}")

    def _compute_kl_div_importance(self, donor: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        diff = (donor.float() - base.float()).abs()
        # For MoE gate matrices, ensure softmax is taken across the expert dimension
        if _is_router_weight(self.weight_info.name):
            # If shape is [hidden_dim, num_experts], experts are dim=-1
            # If shape is [num_experts, hidden_dim], experts are dim=0
            if donor.dim() == 2 and donor.shape[0] < donor.shape[1] and donor.shape[0] <= 128:
                softmax_dim = 0
            else:
                softmax_dim = -1
        else:
            softmax_dim = -1

        p = F.softmax(donor.float(), dim=softmax_dim) + EPS
        q = F.softmax(base.float(), dim=softmax_dim) + EPS
        kl_div = torch.sum(p * torch.log(p / q), dim=softmax_dim, keepdim=True)
        return (diff * kl_div).view_as(donor)

    def _compute_delta_mag_importance(self, delta: torch.Tensor) -> torch.Tensor:
        return delta.abs()

    def _compute_cosine_sim_importance(
        self, donor: torch.Tensor, base: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates row-wise/channel-wise angular divergence:
        (1 - cos(theta_channel)) * ||delta_channel||.
        """
        if delta.dim() <= 1:
            dot = torch.dot(donor.flatten().float(), base.flatten().float())
            norm_d = donor.flatten().float().norm().clamp_min(EPS)
            norm_b = base.flatten().float().norm().clamp_min(EPS)
            cos_sim = (dot / (norm_d * norm_b)).clamp(-1.0, 1.0)
            return delta.abs() * (1.0 - cos_sim)

        donor_flat = donor.float().flatten(1)
        base_flat = base.float().flatten(1)
        delta_flat = delta.float().flatten(1)

        norm_d = donor_flat.norm(dim=1, keepdim=True).clamp_min(EPS)
        norm_b = base_flat.norm(dim=1, keepdim=True).clamp_min(EPS)
        dot_row = (donor_flat * base_flat).sum(dim=1, keepdim=True)
        cos_row = (dot_row / (norm_d * norm_b)).clamp(-1.0, 1.0)

        importance_flat = delta_flat.abs() * (1.0 + (1.0 - cos_row))
        return importance_flat.view_as(delta)

    def _compute_variance_sce_importance(self, delta: torch.Tensor) -> torch.Tensor:
        """Calculates channel-wise variance scaled by magnitude."""
        if delta.dim() > 1:
            var = torch.var(delta.float(), dim=-1, keepdim=True, unbiased=False)
            importance = delta.abs() * (var.sqrt() + EPS)
        else:
            var = delta.float().var(unbiased=False)
            importance = delta.abs() * (var.sqrt() + EPS)
        return importance.view_as(delta)

    def _compute_karcher_fisher_importance(
        self, donor: torch.Tensor, base: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates Riemannian Log-Map tangent energy on the hypersphere:
        Log_u(v) = (theta / sin(theta)) * (v - cos(theta) * u)
        """
        u_base = F.normalize(base.flatten().float(), p=2, dim=0, eps=EPS)
        u_donor = F.normalize(donor.flatten().float(), p=2, dim=0, eps=EPS)

        dot = torch.dot(u_base, u_donor).clamp(-1.0, 1.0)
        theta = torch.arccos(dot)
        sin_theta = torch.sin(theta).clamp_min(EPS)

        scaling = theta / sin_theta if theta.item() > EPS else 1.0
        tangent_vector = scaling * (u_donor - dot * u_base)
        tangent_energy = tangent_vector.abs().view_as(base)

        diff = (donor.float() - base.float()).abs()
        return (diff * (1.0 + tangent_energy)).view_as(base)

    def _solve_hierarchical_mc_pareto_multi(
        self, task_vectors: torch.Tensor, base: torch.Tensor, donor_weights: List[float]
    ) -> torch.Tensor:
        """
        Multi-Donor Monte Carlo Pareto Optimization:
        Evaluates candidate importance metric mixtures and Tukey threshold fences
        on power-of-2 strided subspaces across all donor task vectors.
        """
        metrics = ["delta_mag", "kl_div", "cosine_sim", "variance_sce", "fisher_grad"]
        dtf = DynamicThresholdFusion()
        engine = HierarchicalMonteCarloParetoEngine()
        num_donors = task_vectors.shape[0]

        # Extract representative aggregated donor and delta for subspace trial
        w_t = torch.tensor(donor_weights, device=task_vectors.device, dtype=torch.float32)
        while w_t.dim() < task_vectors.dim():
            w_t = w_t.unsqueeze(-1)
        mean_delta = (task_vectors * w_t).sum(dim=0)
        mean_donor = base + mean_delta

        d_sub, b_sub, delta_sub = engine.sample_power_of_two_subspace(
            mean_donor, base, mean_delta, sample_size=self.power_of_two_scale
        )

        # Precompute candidate importance fields on subspace
        sub_imp_fields = []
        for m in metrics:
            imp_f = self._compute_importance(mean_donor, base, mean_delta, m)
            _, _, imp_sub = engine.sample_power_of_two_subspace(
                mean_donor, base, imp_f, sample_size=self.power_of_two_scale
            )
            sub_imp_fields.append(imp_sub / imp_sub.norm().clamp_min(EPS))

        best_loss = float("inf")
        best_blend_weights = [1.0 / len(metrics)] * len(metrics)
        best_fence = self.tukey_fence
        winning_trial_idx = 0
        winning_type = "Pure Baseline"

        # Discrete Baseline Check (Trials 1 through 5)
        for i in range(len(metrics)):
            w = [1.0 if j == i else 0.0 for j in range(len(metrics))]
            cand_imp_sub = sub_imp_fields[i]
            q_vals = dtf.approximate_quantiles(cand_imp_sub, torch.tensor([0.25, 0.75]))
            thresh = q_vals[0] + self.tukey_fence * (q_vals[1] - q_vals[0])
            mask_sub = (cand_imp_sub >= thresh).float()

            fused_sub = b_sub + delta_sub * mask_sub
            loss = engine.simulate_synthetic_forward_benchmark(fused_sub, b_sub, d_sub, delta_sub, mask_sub)
            if loss < best_loss:
                best_loss = loss
                best_blend_weights = w
                winning_trial_idx = i + 1
                winning_type = f"Pure {metrics[i].upper()}"

        # Continuous Dirichlet Monte Carlo Exploration (Trials 1 through K)
        if self.mc_trials > 0:
            dirichlet_dist = torch.distributions.Dirichlet(torch.ones(len(metrics)))
            for trial_step in range(1, self.mc_trials + 1):
                cand_w = dirichlet_dist.sample().tolist()
                fence_jitter = max(0.5, self.tukey_fence + (torch.rand(1).item() * 0.6 - 0.3))

                blended_sub = sum(w * sub_f for w, sub_f in zip(cand_w, sub_imp_fields))
                q_vals = dtf.approximate_quantiles(blended_sub, torch.tensor([0.25, 0.75]))
                thresh = q_vals[0] + fence_jitter * (q_vals[1] - q_vals[0])
                mask_sub = (blended_sub >= thresh).float()

                fused_sub = b_sub + delta_sub * mask_sub
                loss = engine.simulate_synthetic_forward_benchmark(fused_sub, b_sub, d_sub, delta_sub, mask_sub)
                
                # Check for Pareto improvement
                if loss < best_loss:
                    best_loss = loss
                    best_blend_weights = cand_w
                    best_fence = fence_jitter
                    winning_trial_idx = trial_step
                    winning_type = f"Trial #{trial_step}/{self.mc_trials}"

        # Register win and blend coefficients in Global Metric Tracker
        GlobalMetricTracker.record_win(best_blend_weights)

        # Apply winning mixture across all donors on the full-dimensional tensor
        full_imp = torch.zeros_like(base)
        for idx, m in enumerate(metrics):
            full_imp_f = self._compute_importance(mean_donor, base, mean_delta, m)
            full_imp += best_blend_weights[idx] * (full_imp_f / full_imp_f.norm().clamp_min(EPS))

        final_mask, _ = dtf.compute_fusion_mask(full_imp, tukey_fence=best_fence)
        
        # Multi-donor weighted sum masked by optimal Pareto threshold
        fused_delta = (task_vectors * w_t).sum(dim=0) * final_mask

        # Telemetry metrics for Audit HUD
        delta_norms = [torch.norm(task_vectors[i].flatten()).item() for i in range(num_donors)]
        p_rate = final_mask.mean().item()
        mask_pass_rates = [p_rate] * num_donors
        effective_shares = [
            (donor_weights[i] * torch.norm((task_vectors[i] * final_mask).flatten())).item()
            for i in range(num_donors)
        ]

        w_name = self.weight_info.name
        if "mlp" in w_name or "self_attn" in w_name or "embed" in w_name or "gate" in w_name:
            clean_donor_names = [
                str(m.model.path).split("\\")[-1].split("/")[-1] 
                for m in list(self.tensor_parameters.keys() if self.tensor_parameters else []) if m != self.base_model
            ]
            base_str = str(self.base_model.model.path).split("\\")[-1].split("/")[-1]

            trial_info = f"Winner: {winning_type} | Loss: {best_loss:.4f} | Pass Rate: {p_rate*100:4.1f}%"
            blend_str = (
                f"MAG:{best_blend_weights[0]:.2f} | KL:{best_blend_weights[1]:.2f} | "
                f"COS:{best_blend_weights[2]:.2f} | SCE:{best_blend_weights[3]:.2f} | "
                f"FSH:{best_blend_weights[4]:.2f}"
            )

            log_multi_fusion_audit(
                layer_name=w_name,
                base_name=base_str,
                donor_names=clean_donor_names,
                delta_norms=delta_norms,
                mask_pass_rates=mask_pass_rates,
                effective_shares=effective_shares,
                metric_str=f"MC-Pareto (K={self.mc_trials})",
                tukey_fence=best_fence,
                winning_trial_info=trial_info,
                winning_blend_str=blend_str,
            )

        return base + fused_delta
  
class MultiFusionMerge(MergeMethod):
    def name(self) -> str:
        return "multi_fusion"

    @override
    def pretty_name(self) -> Optional[str]:
        return "Multi Fusion v4 (Multi-Donor Monte Carlo Pareto MoE/Dense Telemetry Engine)"

    @override
    def reference_url(self) -> Optional[str]:
        return "https://huggingface.co/Naphula/Slimaki-Tavern-24B-v1.3"

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(
                name="importance_metric",
                required=False,
                default_value="pareto_auto",
                description="Importance metric: 'pareto_auto', 'delta_mag', 'kl_div', 'cosine_sim', 'variance_sce', 'fisher_grad'",
            ),
            ConfigParameterDef(
                name="tukey_fence",
                required=False,
                default_value=1.5,
                description="Tukey IQR multiplier (median + fence * IQR) for dynamic outlier selection threshold",
            ),
            ConfigParameterDef(
                name="anti_monopoly_power",
                required=False,
                default_value=1.0,
                description="Power law exponent for donor norm damping (1.0=unconstrained, 0.40=anti-monopoly damping)",
            ),
            ConfigParameterDef(
                name="mc_trials",
                required=False,
                default_value=16,
                description="Number of continuous Dirichlet Monte Carlo trials for pareto_auto (0=pure discrete check, 8-64 recommended)",
            ),
            ConfigParameterDef(
                name="power_of_two_scale",
                required=False,
                default_value=4096,
                description="Power of 2 sample resolution for Monte Carlo benchmark slices (e.g. 1024, 4096, 16384)",
            ),
            ConfigParameterDef(
                name="is_moe",
                required=False,
                default_value=True,
                description="Enables MoE architecture guards for router gates and expert parameter matrices",
            ),
            ConfigParameterDef(
                name="router_strategy",
                required=False,
                default_value="metric",
                description="Router gate fusion strategy: 'metric' (uses router_metric), 'average', 'first', 'base', 'random_init'",
            ),
            ConfigParameterDef(
                name="router_metric",
                required=False,
                default_value="kl_div",
                description="Dedicated metric for router weights: 'kl_div', 'pareto_auto', 'delta_mag', 'cosine_sim'",
            ),
            ConfigParameterDef(
                name="blend_experts",
                required=False,
                default_value=True,
                description="Blends MoE expert layers (if False, passes base model expert weights untouched)",
            ),
            ConfigParameterDef(
                name="sce_inversion_mode",
                required=False,
                default_value=False,
                description="Aikido/MAGIC minority inversion: Inverts conflicting parameter delta signs instead of zeroing them out",
            ),
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="weight", required=False, default_value=1.0)
        ]

    def make_task(
        self,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        base_model: Optional[ModelReference],
        parameters: Dict[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
        **kwargs,
    ) -> Task[torch.Tensor]:
        params = dict(parameters)
        return MultiFusionMergeTask(
            gather_tensors=tensors,
            weight_info=output_weight,
            base_model=base_model,
            tensor_parameters=tensor_parameters,
            importance_metric=str(params.get("importance_metric", "pareto_auto")),
            tukey_fence=float(params.get("tukey_fence", 1.5)),
            anti_monopoly_power=float(params.get("anti_monopoly_power", 1.0)),
            mc_trials=int(params.get("mc_trials", 16)),
            power_of_two_scale=int(params.get("power_of_two_scale", 4096)),
            is_moe=bool(params.get("is_moe", True)),
            router_strategy=str(params.get("router_strategy", "metric")),
            router_metric=str(params.get("router_metric", "kl_div")),
            blend_experts=bool(params.get("blend_experts", True)),
            sce_inversion_mode=bool(params.get("sce_inversion_mode", False)),
        )