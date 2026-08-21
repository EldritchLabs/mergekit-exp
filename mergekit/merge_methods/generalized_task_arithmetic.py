# Copyright (C) 2026 Arcee AI
# SPDX-License-Identifier: LGPL-3.0-only

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import torch
from pydantic import BaseModel
from typing_extensions import Literal, override

from mergekit.architecture import WeightInfo
from mergekit.common import ImmutableMap, ModelReference
from mergekit.graph import Task
from mergekit.merge_methods.base import (
    ConfigParameterDef,
    MergeMethod,
    MergeTensorInput,
)
from mergekit.sparsify import RescaleNorm, SparsificationMethod, sparsify


class ConsensusMethod(str, Enum):
    count = "count"
    sum = "sum"


class GeneralizedTaskArithmeticMerge(MergeMethod, BaseModel, frozen=True):
    consensus_method: Optional[ConsensusMethod]
    sparsification_method: Optional[SparsificationMethod]
    default_normalize: bool
    default_rescale: bool
    method_name: str
    method_pretty_name: Optional[str]
    method_reference_url: Optional[str]

    def name(self) -> str:
        return self.method_name

    @override
    def pretty_name(self) -> Optional[str]:
        return self.method_pretty_name

    @override
    def reference_url(self) -> Optional[str]:
        return self.method_reference_url

    def parameters(self) -> List[ConfigParameterDef]:
        return [
            ConfigParameterDef(name="int8_mask", required=False, default_value=False),
            ConfigParameterDef(
                name="normalize", required=False, default_value=self.default_normalize
            ),
            ConfigParameterDef(
                name="rescale", required=False, default_value=self.default_rescale
            ),
            ConfigParameterDef(name="lambda", required=False, default_value=1.0),
        ]

    def tensor_parameters(self) -> List[ConfigParameterDef]:
        res = [
            ConfigParameterDef(name="weight", required=True),
            ConfigParameterDef(name="density", required=False, default_value=1.0),
        ]
        if self.sparsification_method == SparsificationMethod.magnitude_outliers:
            res.append(
                ConfigParameterDef(
                    name="gamma",
                    default_value=0.01,
                )
            )
        if self.sparsification_method == SparsificationMethod.della_magprune:
            res.append(
                ConfigParameterDef(
                    name="epsilon",
                    default_value=0.15,
                )
            )
        return res

    def make_task(
        self,
        output_weight: WeightInfo,
        tensors: MergeTensorInput,
        base_model: Optional[ModelReference],
        parameters: ImmutableMap[str, Any],
        tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
    ) -> Task:
        return GTATask(
            method=self,
            tensors=tensors,
            base_model=base_model,
            tensor_parameters=tensor_parameters,
            int8_mask=parameters["int8_mask"],
            normalize=parameters["normalize"],
            lambda_=parameters["lambda"],
            rescale_norm=RescaleNorm.l1 if parameters["rescale"] else None,
            weight_info=output_weight,
        )


class GTATask(Task[torch.Tensor]):
    method: GeneralizedTaskArithmeticMerge
    tensors: MergeTensorInput
    base_model: ModelReference
    weight_info: WeightInfo
    tensor_parameters: ImmutableMap[ModelReference, Any]
    int8_mask: bool
    normalize: bool
    lambda_: float
    rescale_norm: Optional[RescaleNorm]

    def uses_accelerator(self) -> bool:
        return True

    def arguments(self) -> Dict[str, Task]:
        return {"tensors": self.tensors}

    def execute(
        self,
        tensors: Dict[ModelReference, torch.Tensor],
        **_kwargs,
    ) -> torch.Tensor:
        # collect task vectors
        tvs, base = get_task_vectors(
            self.weight_info,
            self.base_model,
            tensors,
            tensor_parameters=self.tensor_parameters.data,
        )
        
        # --- LIVE AUDIT CHART ---
        if tvs:
            log_della_audit(
                self.weight_info.name, 
                self.base_model, 
                tvs, 
                self.lambda_, 
                self.method.method_pretty_name
            )
        # ------------------------

        if not tvs:
            return base

        # sparsify
        if self.method.sparsification_method:
            for tv_info in tvs:
                kwargs = {}
                if "gamma" in tv_info:
                    kwargs["gamma"] = tv_info["gamma"]

                if "epsilon" in tv_info:
                    kwargs["epsilon"] = tv_info["epsilon"]

                tv_info["delta"] = sparsify(
                    tv_info["delta"],
                    density=tv_info["density"],
                    method=self.method.sparsification_method,
                    rescale_norm=self.rescale_norm,
                    **kwargs,
                )

        deltas = torch.stack([tv["delta"] for tv in tvs], dim=0)

        weights = torch.tensor(
            [tv["weight"] for tv in tvs], dtype=deltas.dtype, device=deltas.device
        )
        while len(deltas.shape) > len(weights.shape):
            weights.unsqueeze_(-1)

        weighted_deltas = deltas * weights

        # get sign consensus and mix deltas
        if self.method.consensus_method:
            mask_dtype = torch.int8 if self.int8_mask else base.dtype
            mask = get_mask(
                weighted_deltas,
                method=self.method.consensus_method,
                mask_dtype=mask_dtype,
            )
            mixed_delta = (weighted_deltas * mask).sum(dim=0)
            divisor = (weights * mask).sum(dim=0)
            divisor[divisor == 0] = 1
        else:
            mixed_delta = weighted_deltas.sum(dim=0)
            divisor = weights.sum(dim=0)
            divisor[divisor.abs() < 1e-8] = 1

        if self.normalize:
            mixed_delta /= divisor

        if self.lambda_ != 1:
            mixed_delta *= self.lambda_

        return (base + mixed_delta).to(base.dtype)

    def group_label(self) -> Optional[str]:
        return self.tensors.group_label()


def get_task_vectors(
    weight_info: WeightInfo,
    base_model: ModelReference,
    tensors: ImmutableMap[ModelReference, torch.Tensor],
    tensor_parameters: ImmutableMap[ModelReference, ImmutableMap[str, Any]],
) -> Tuple[List[Dict[str, Any]], torch.Tensor]:
    keys = list(tensors.keys())
    base = tensors[base_model]

    parameter_name = weight_info.name

    res = []
    for model in keys:
        if model == base_model:
            continue

        x = tensors[model].to(base.dtype)
        if x.shape != base.shape:
            if weight_info.is_embed:
                x = x[: base.shape[0], : base.shape[1]]
                logging.warning(f"Using submatrix of {model}:{parameter_name}")
            else:
                logging.warning(
                    f"skipping {model}:{parameter_name} due to size mismatch"
                )
                continue

        delta = x - base
        del x
        del tensors[model]

        d = {}
        d["model"] = model
        d["delta"] = delta
        for p in tensor_parameters[model]:
            d[p] = tensor_parameters[model][p]
        res.append(d)
    return res, base


def get_mask(
    delta: torch.Tensor,
    method: Literal["sum", "count"] = "sum",
    mask_dtype: Optional[torch.dtype] = None,
):
    """Returns a mask determining which delta vectors should be merged
    into the final model.

    For the methodology described in the TIES paper use 'sum'. For a
    simpler naive count of signs, use 'count'."""
    if mask_dtype is None:
        mask_dtype = delta.dtype

    sign = delta.sign().to(mask_dtype)

    if method == "sum":
        sign_weight = delta.sum(dim=0)
        majority_sign = (sign_weight >= 0).to(mask_dtype) * 2 - 1
        del sign_weight
    elif method == "count":
        majority_sign = (sign.sum(dim=0) >= 0).to(mask_dtype) * 2 - 1
    else:
        raise RuntimeError(f'Unimplemented mask method "{method}"')

    return sign == majority_sign


def log_della_audit(
    layer_name: str, 
    base_model: ModelReference, 
    tvs: List[Dict[str, Any]], 
    global_lambda: float,
    method_name: str
):
    """Prints and saves a bar chart of DELLA/Task Arithmetic distribution based on actual Delta Norms."""
    
    base_name = str(base_model.model.path).split("\\")[-1].split("/")[-1][:50]
    
    bar_char = "█"
    lines = [f"\n[{method_name} Audit] Layer: {layer_name} | Lambda={global_lambda:.2f}"]
    lines.append(f"  [BASE] {base_name:<50}")

    # 1. Calculate stats
    stats = []
    total_impact = 0.0
    
    for tv in tvs:
        model_name = str(tv['model'].model.path).split("\\")[-1].split("/")[-1][:50]
        weight = tv.get('weight', 0.0)
        density = tv.get('density', 1.0)
        epsilon = tv.get('epsilon', None)
        delta = tv.get('delta', None)
        
        norm = 0.0
        if delta is not None:
            # Use float32 for norm calculation to be safe
            norm = torch.norm(delta.float()).item()
            
        # Effective contribution magnitude = Weight * Norm
        # This shows how much this model is actually moving the weights
        impact = weight * norm
        total_impact += impact
            
        stats.append({
            'name': model_name,
            'weight': weight,
            'density': density,
            'epsilon': epsilon,
            'norm': norm,
            'impact': impact
        })

    # Sort by name for consistent logs
    stats.sort(key=lambda x: x['name'])

    # 2. Generate bars
    for s in stats:
        # Calculate percentage relative to the sum of all impacts (Share of Voice)
        pct = (s['impact'] / total_impact * 100) if total_impact > 0 else 0.0
        
        # Bar length (max 50 chars for 100%)
        bar_len = int(max(0, min(50, pct / 2)))
        bar = bar_char * bar_len
        
        # Format info string
        # W=Weight, D=Density, N=DeltaNorm
        info = f"W:{s['weight']:.2f} D:{s['density']:.2f} N:{s['norm']:.2f}"
        if s['epsilon'] is not None:
            info += f" E:{s['epsilon']:.2f}"
            
        lines.append(f"  {s['name']:<50}: {bar:<50} {pct:5.1f}% ({info})")

    log_entry = "\n".join(lines)
    print(log_entry)
    
    with open("della_audit.log", "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")