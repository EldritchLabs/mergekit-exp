# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

from typing import List, Optional

import torch

from mergekit.merge_methods.easy_define import merge_method
from mergekit.merge_methods.generalized_task_arithmetic import (
    get_mask as sign_consensus_mask,
)
from mergekit.architecture import WeightInfo
from mergekit.common import ModelReference
import inspect


def log_sce_audit(layer_name: str, base_name: str, donor_names: List[str], variances: List[float], select_topk: float):
    bar_char = "█"
    lines = [f"\n[SCE Audit] Layer: {layer_name} | select_topk={select_topk}"]
    clean_base = base_name.split("\\")[-1].split("/")[-1][:50]
    lines.append(f"  [BASE] {clean_base:<50}")
    
    total_var = sum(variances) if sum(variances) > 0 else 1.0
    
    for name, var in zip(donor_names, variances):
        pct = (var / total_var) * 100
        bar_len = int(max(0, min(50, pct / 2)))
        bar = bar_char * bar_len
        clean_name = name.split("\\")[-1].split("/")[-1][:50]
        # lines.append(f"  {clean_name:<50}: {bar:<50} {pct:5.1f}% (Var: {var:.6f})")
        lines.append(f"  {clean_name:<50}: {bar:<50} (Variance {pct:5.1f}%)")
        
    log_entry = "\n".join(lines)
    print(log_entry)
    with open("sce_audit.log", "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")


@merge_method(
    name="sce",
    pretty_name="SCE",
    reference_url="https://arxiv.org/abs/2408.07990",
)
def sce_merge(
    tensors: List[torch.Tensor],
    base_tensor: torch.Tensor,
    output_weight: WeightInfo,
    base_model: ModelReference,
    int8_mask: bool = False,
    select_topk: float = 1.0,
) -> torch.Tensor:
    if not tensors:
        return base_tensor
    mask_dtype = torch.int8 if int8_mask else base_tensor.dtype
    task_vectors = torch.stack([t - base_tensor for t in tensors], dim=0)

    if select_topk < 1:
        mask = sce_mask(task_vectors, select_topk, mask_dtype)
        task_vectors = task_vectors * mask.unsqueeze(0)
        
        # --- LIVE AUDIT CHART ---
        model_refs = inspect.currentframe().f_back.f_locals.get('model_refs')
        donor_names =[str(m.model.path) for m in model_refs] if model_refs else [f"Donor_{i}" for i in range(len(tensors))]
        variances =[torch.var(tv.float()).item() for tv in task_vectors]
        log_sce_audit(output_weight.name, str(base_model.model.path), donor_names, variances, select_topk)
        # ------------------------

    erase_mask = sign_consensus_mask(task_vectors, method="sum", mask_dtype=mask_dtype)

    tv_weights = sce_weight(task_vectors)
    while tv_weights.dim() < task_vectors.dim():
        tv_weights = tv_weights.unsqueeze(-1)

    erased_weights = tv_weights * erase_mask
    merged_tv = (task_vectors * erased_weights).sum(dim=0)
    final_tv = merged_tv / torch.sum(erased_weights, dim=0).clamp(min=1e-6)

    return base_tensor + final_tv


def sce_weight(tvs: torch.Tensor) -> torch.Tensor:
    weights = torch.mean(tvs**2, dim=list(range(1, tvs.dim())))
    weight_sum = torch.sum(weights).item()
    if abs(weight_sum) < 1e-6:
        return torch.ones_like(weights) / weights.shape[0]
    return weights / weight_sum


def sce_mask(
    tvs: torch.Tensor, density: float, mask_dtype: Optional[torch.dtype] = None
):
    if density <= 0:
        return torch.zeros_like(tvs, dtype=mask_dtype)
    if density >= 1:
        return torch.ones_like(tvs, dtype=mask_dtype)

    var = torch.var(tvs, dim=0, unbiased=False)
    nonzero = torch.count_nonzero(var)
    k = int(nonzero * density)
    if k == 0:
        return torch.zeros_like(tvs, dtype=mask_dtype)

    _, indices = torch.topk(var.abs().view(-1), k=k, largest=True)
    mask = torch.zeros_like(var, dtype=mask_dtype)
    mask.view(-1)[indices] = 1
    return mask
