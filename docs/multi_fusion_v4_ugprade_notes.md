### Summary of `multi_fusion` v2a Enhancements:
1. **MoE + Dense Architecture Handling**: Adds `is_moe` parameter, weight name regex dispatch for expert weights (`.experts.`, `.block_sparse_moe.experts.`), and router gates (`.gate.weight`, `.shared_expert_gate.weight`), plus configurable `router_strategy` (`"kl_div"`, `"average"`, `"first"`, etc.) and `blend_experts`.
2. **`variance_sce` with Optional MAGIC Minority Inversion**: Replaces the old stub with true matrix-level row-wise variance selection, and supports `sce_inversion_mode` (Aikido/MAGIC minority inversion where dissenters flip sign instead of being masked out).
3. **True Karcher-Style `fisher_grad`**: Evaluates tangent projections on the Riemannian hypersphere manifold ($u_{\text{base}}$ vs. $u_{\text{donor}}$), calculating geodesic angle $\theta = \arccos(\langle u, v \rangle)$ and the Riemannian Log-Map tangent energy $\|\text{Log}_u(v)\|^2$ to represent true Fisher information metric curvature.
4. **Row-Wise / Cosine Angle Subspace `cosine_sim`**: Fixes the scalar-tensor bug reported by Devin by computing row-wise/filter-wise angular divergence $(1 - \cos \theta_r) \cdot \|\Delta W_r\|$, producing true Model Stock-style geometry variations.
5. **Dedicated Router Metric Override**: Adds `router_metric` (defaulting to `"kl_div"`), allowing router matrices to evaluate exact behavioral gating shifts while expert matrices use `delta_mag` or `variance_sce`.
6. **Multi-Objective Pareto Frontier Mask Auto-Selector**: Adds `importance_metric: "pareto_auto"`, which evaluates candidate masks across the available metrics on a low-rank proxy slice, scoring them on gradient conservation, sparsity variance, and directional alignment to autonomously select the optimal metric per layer.

---

### Summary of `multi_fusion` v2b Enhancements:
### 1. Mathematical Audit: How Candidates Were Evaluated vs. What Was Broken

In `multi_fusion_v2a.py`, the evaluation in `_solve_pareto_auto_mask` was:

```python
# From v2a:
candidate_fused = base_slice + delta_slice * mask_slice
cand_delta = candidate_fused - base_slice

dot_cancel = abs(torch.dot(cand_delta / cand_delta.norm().clamp_min(EPS),
                           base_slice / base_slice.norm().clamp_min(EPS)).item())
var_score = 1.0 / (torch.var(candidate_fused).item() + EPS)
sparsity_ratio = mask_slice.mean().item()
sparsity_penalty = abs(sparsity_ratio - 0.25) * 2.0

pareto_loss = dot_cancel * 2.0 + var_score * 0.05 + sparsity_penalty
```

#### Why this analysis metric was inadequate:
1. **Evaluating Raw Weights in Static Slices**: It only evaluated raw static weight coordinates on the first contiguous `8192` elements. For 2D projection matrices or stacked MoE experts, the first 8,192 elements represent just a fraction of the first few rows (e.g., token 0 in an embedding or expert 0's initial channels), ignoring the global manifold topology.
2. **Hardcoded Sparsity Target (`0.25`)**: Penalizing `abs(sparsity_ratio - 0.25)` forced every layer to select roughly 25% of elements regardless of whether the layer needed 2% outlier grafting or 60% feature injection.
3. **Discrete Metric Winner-Take-All**: It evaluated 5 discrete candidate masks and selected a single winner. It lacked continuous convex or Pareto blending across metrics.
4. **No Functional Simulation of Output Manifolds**: As Vortex5 noted in the chat, evaluation should not just look at passive weight numbers. It should measure **signal preservation vs. noise amplification vs. directional distortion** through synthetic forward projection trials.

---

### 2. Audit of Devin's Claim: Is `kl_div` Truly Appropriate for Routers?

#### Devin's Claim:
> *"Router / gate (`mlp.gate.weight`) -> `kl_div`: The row softmax over `num_experts` logits is the routing distribution — KL here measures real behavioral divergence, not a proxy."*

#### The Verdict: **Partially True, but Architecturally Inverted.**
* **Where Devin is right**: Unlike dense projections or MLP hidden activations (where weights are linear transformations, not probability distributions), router weights $W_g \in \mathbb{R}^{E \times D}$ directly compute routing logits:
$$p(\text{expert } e \mid x) = \text{Softmax}(W_g \cdot x)_e$$
  A softmax across expert logits actually represents a categorical probability distribution.
* **Where Devin is wrong**: Look at how `kl_div` was calculated across matrices:
  ```python
  p = F.softmax(donor.float(), dim=-1) + EPS
  q = F.softmax(base.float(), dim=-1) + EPS
  kl_div = torch.sum(p * torch.log(p / q), dim=-1, keepdim=True)
  ```
  In Transformers (including Gemma-4, Mixtral, and Qwen), gate weight tensors are formatted as:
$$\text{Shape: } [\text{num\_experts}, \text{hidden\_dim}] \quad \text{or} \quad [\text{hidden\_dim}, \text{num\_experts}]$$
  * If the tensor is `[num_experts, hidden_dim]` (e.g., Gemma 4 / Mistral), taking `dim=-1` computes a softmax **across all 2,048 or 4,096 hidden input dimensions**, **not** across the experts!
  * Furthermore, static gate weights are unconditioned on hidden state activations $x$. If the donor and base gates differ by a constant scale factor, their softmax distributions can diverge while their top-1/top-2 ranking of tokens remains completely identical.
* **Conclusion**: We keep `kl_div` as a fast analytical router default, but provide `router_strategy: "pareto_auto"` so the Monte Carlo Pareto engine can automatically benchmark `kl_div`, `cosine_sim`, and `delta_mag` on gate tensors as well.

---

### 3. The Upgrade: Power-of-2 Hierarchical Monte Carlo Pareto Engine (`v2b`)

In `v2b`, we replace the static slice evaluation with an **autonomous hierarchical Monte Carlo simulation**:

1. **Power-of-2 Hierarchical Multi-Resolution Sampling ($N \in \{2^8, 2^{10}, 2^{12}, 2^{14}\}$)**:
   Instead of testing a static slice, the engine draws strided, randomly seeded sub-tensors across powers of 2 ($256 \to 1024 \to 4096 \to 16384$), capturing both local row-wise structure and global matrix variance.
2. **Configurable Monte Carlo Trials (`mc_trials: 8` to `64`)**:
   Users can choose between rapid runs (`8` trials) or deep, multi-pass Pareto searches (`32`–`64` trials).
3. **Multi-Metric Pareto Loss Function**:
   Evaluates each candidate mask $M$ through simulated forward passes:
$$\mathcal{L}_{\text{Pareto}} = w_{\text{drift}} \cdot \mathcal{E}_{\text{drift}} + w_{\text{noise}} \cdot \mathcal{E}_{\text{noise}} + w_{\text{rec}} \cdot \mathcal{E}_{\text{rec}}$$
   * **Base Manifold Preservation ($\mathcal{E}_{\text{drift}}$)**: Measures directional cosine drift of base representations under simulated projection vectors $x \sim \mathcal{N}(0, I)$.
   * **Noise Amplification ($\mathcal{E}_{\text{noise}}$)**: Penalizes high-frequency variance spikes in inactive orthogonal null-spaces.
   * **Donor Signal Recovery ($\mathcal{E}_{\text{rec}}$)**: Measures how effectively the donor's salient task-vector signal is transferred without overwriting base features.
4. **Continuous Dirichlet Pareto Blending**:
   Rather than selecting a single metric as a discrete winner, high trial counts sample continuous mixture weights $\vec{\alpha} \sim \text{Dirichlet}(\mathbf{1})$, blending the importance scoring functions into an optimal hybrid importance metric.

---

### Architectural Comparison: Base Anchor Geometry

When merging an ensemble of fine-tunes along with compound synthesizers (DELLA/Novelty), the choice of the `base_model` geometry completely determines the coordinate frame for task vectors $\tau_i = W_i - W_{\text{base}}$.

```
Option A: Karcher Consensus as Base Anchor (Geometric Barycenter Frame)
                [Donor 1]  [Donor 2] ... [Donor 10]   [Novelty Blend]
                     \          |         /               /
                      \         |        /               /
                       ▼        ▼       ▼               ▼
                       ----------------------------------
                              [Base: Karcher Mean] 
                         (Centered at manifold centroid)

Option B: Original Gemma 4 as Base Anchor (Pretrained Origin Frame)
          [Karcher]   [Donor 1] ... [Donor 10]   [Novelty Blend]
              \            \             /              /
               \            \           /              /
                ▼            ▼         ▼              ▼
                ----------------------------------------
                        [Base: Google Gemma 4 Base]
                       (Zero-point pretrained origin)
```

---

### Trade-Offs: Karcher Consensus Base vs. Gemma 4 Pretrained Base

| Dimension | Option A: Karcher as Base (`base_model: karcher`) | Option B: Gemma 4 as Base (Karcher as Donor or Excluded) |
| :--- | :--- | :--- |
| **Origin Point** | **Riemannian Centroid $\mu_{\mathbb{S}}$** of the 10 models. | Pretrained ancestor checkpoint ($W_{\text{pt}}$). |
| **Task Vector Semantics** | $\tau_i = W_i - \mu_{\mathbb{S}}$ represents **pure idiosyncratic divergence** from consensus. | $\tau_i = W_i - W_{\text{pt}}$ represents **cumulative fine-tuning drift** + shared training shift. |
| **Sparsity & Noise** | **Maximally Sparse**: Shared direction cancels out ($\sum \tau_i \approx \mathbf{0}$). Tukey outlier detection captures true individual specialisms. | **Dense & Correlated**: All 10 models share common instruction-tuning shifts, creating directional collinearity and mask collisions. |
| **Novelty Grafting Mechanics** | `Novelty_Blend` acts as an expansive outward tangent force relative to the stable centroid. | `Karcher` acts as an inward gravitational pull fighting against the outward fine-tune deltas. |
| **Failure Mode / Risks** | If a layer had degenerate Karcher convergence (e.g. numerical drift on extreme shapes), the base inherits that offset. | High risk of the "outlier monopoly" seen earlier, where large-magnitude fine-tunes dominate the consensus centroid. |
| **Recommendation** | **Superior for Final Synthesis**: Establishes the Karcher barycenter as the default anchor, grafting high-confidence outliers from the donors and Novelty Blend. | Best for raw primary training passes (TIES, DARE), not for multi-model consensus arbitration. |

---

### What Multi-Fusion v3 Needs for Scale ($N \ge 2$)

To scale beyond 2 models while preventing signal dilution, `multi_fusion_v3` introduces four architectural upgrades:

1. **N-Way Model Support with Donor-Specific Temperature & Weights**: Accepts an arbitrary number of donor models ($N \ge 2$) alongside the designated base anchor.
2. **Channel-Wise Anti-Monopoly Normalization (`anti_monopoly_power`)**: Applies a sub-linear power transform ($0.35\text{--}0.50$) to task-vector variance across donors, preventing large-norm models from crowding out specialized donors.
3. **Multi-Donor TIES Sign Agreement on Selected Outliers**: When multiple donors pass the Tukey outlier threshold at the same parameter coordinate, their directions are resolved using majority sign consensus rather than additive cancellation.
4. **Hierarchical Monte Carlo Pareto Mask Arbiter ($N \times K$)**: Evaluates candidate masks across all donors simultaneously, optimizing a composite Pareto loss (Base Manifold Drift, Signal Recovery, Sparsity Floor) using multi-resolution power-of-2 subspace trials.

---

Here are the `<<<< before` and `==== after >>>>` patch chunks to add a live visual Audit HUD to `multi_fusion_v3.py`. 

### Features of this HUD:
1. **Dynamic Influence Breakdown**: Tracks each donor's Task Vector Norm ($\|\tau_i\|$), Tukey Outlier Mask Pass Rate (Sparsity %), and Net Fused Impact on the tensor.
2. **True Force Bar Charts**: Formats terminal visual bars (`█████`) showing exactly how much of each donor survived into the final weights.
3. **Multi-Metric Winning Formulation Display**: Prints the winning metric blend discovered by the Hierarchical Monte Carlo Pareto Engine.
4. **MoE Layer Discrimination**: Specifically logs expert and gate layers to the console and appends detailed records to `multi_fusion_audit.log`.

---

### Upgrades in `v4`:
1. **Live Trial Progress & Winner Telemetry**: Pass `winning_trial_info` and `winning_blend_str` to the HUD. Shows the exact trial number (`#42/64`), the winning type (`MC Trial #42/64` vs `Pure MAG`), the simulated forward Pareto loss, and the full weight mix breakdown:
   ```text
   ║ Pareto Telemetry: Winner Trial #19/64 (Loss: 0.0412) | Fence: 1.22                        ║
   ║ Winning Blend:    MAG: 0.35 | KL: 0.10 | COS: 0.20 | SCE: 0.25 | FSH: 0.10             ║
   ```
2. **Cumulative Global Metric Tracker**: Thread-safe persistent registry (`GlobalMetricTracker`) tracking total wins, percentage of layers won, and cumulative weight influence for each of the 5 metrics (`delta_mag`, `kl_div`, `cosine_sim`, `variance_sce`, `fisher_grad`) across the entire run.
3. **Global Metric Win HUD**: Adds a dedicated cumulative score box to the audit view:
   ```text
   ║ Global Metric Wins: MAG: 14 (38.9%) | SCE: 10 (27.8%) | KL: 6 (16.7%) | COS: 4 | FSH: 2 ║
   ```
4. **Persistent JSON Telemetry**: Automatically saves running metric win counts to `multi_fusion_metric_stats.json` so you can analyze the distribution after the merge finishes.
```

---

# MultiFusion v4: Autonomous Multi-Donor Monte Carlo Pareto Optimization and Geometric Manifold Gating for Large Language Models

**Author:** Naphula  
**Organization:** Arcee AI & Open-Source Community  
**Date:** October 2026  
**License:** LGPL-3.0-only  
**Reference Codebase:** `multi_fusion_v4.py` (Upgrade from `multi_fusion_v1.py`)  
**Old Reference Model:** [`Naphula/Slimaki-Tavern-24B-v1.3`](https://huggingface.co/Naphula/Slimaki-Tavern-24B-v1.3)
**New Reference Model:** [`Naphula/Goetia-26B-A4B-v1.7`](https://huggingface.co/Naphula/Goetia-26B-A4B-v1.7)

---

## Abstract

Model merging operates in weight space to combine specialized neural capabilities without the compute expenditure of multi-task training or the latency footprint of model ensembling. Early heuristic frameworks—including **MultiFusion v1** (an outlier-gated generalization of Arcee Fusion)—were strictly constrained to pairwise ($N = 2$) operations, relied on naive discrete scoring metrics ($L_1$ delta magnitude, scalar-broadcasted cosine similarity, or unconditioned last-dimension softmax KL divergence), and suffered from high-variance quantile approximations. 

We introduce **MultiFusion v4**, a complete architectural leap from v1. MultiFusion v4 generalizes dynamic outlier thresholding to an arbitrary number of donors ($N \ge 2$), introduces an **Anti-Monopoly Energy Normalization** engine to neutralize high-norm donor dominance, supports **MAGIC minority sign inversion** (Aikido parameter arbitration), incorporates MoE-native routing guards with dimension-aware categorical divergences, and introduces a **Hierarchical Monte Carlo Pareto Engine (`pareto_auto`)**. Through power-of-two strided subspace projections ($N \in \{1024, 4096, 16384\}$) and continuous Dirichlet stochastic exploration ($\vec{\alpha} \sim \text{Dirichlet}(\mathbf{1}_5)$), MultiFusion v4 dynamically discovers the optimal continuous mixture of five core geometric metrics—$L_1$ Delta Magnitude, MoE-Calibrated KL Divergence, Row-Wise Cosine Angular Divergence, Channel-Wise Variance (SCE), and Riemannian Log-Map Tangent Energy (Fisher Information)—tailored to each individual layer. We also provide a terminal-rendered, thread-safe live visual **Audit Telemetry HUD** with persistent global metric win tracking.

> **Note on Research Horizon:** While our internal research roadmap includes the 12-manifold batched tensor collider known as *Pareto Fusion*, that system remains in active laboratory testing and is **not** released in this distribution. **MultiFusion v4** constitutes the official production release for high-fidelity multi-model synthesis.

---

## 1. Introduction: The Jump from v1 to v4

When MultiFusion v1 was published, it provided a simple yet effective mechanism for combining two models ($A$ and $B$) by identifying salient delta weights via Tukey’s Interquartile Range (IQR) fence:

$$\tau = \text{median}(I) + 1.5 \cdot \text{IQR}(I)$$

However, as model scale advanced into the 20B–70B regime and Mixture-of-Experts (MoE) architectures became prevalent, the limitations of v1 became critical bottlenecks:

| Dimension | MultiFusion v1 | MultiFusion v4 |
| :--- | :--- | :--- |
| **Model Arity** | Strictly pairwise ($N = 1$ donor $+ 1$ base). | Arbitrary donors ($N \ge 2$) with individual donor weighting. |
| **Metric Formulation** | Hardcoded, mutually exclusive discrete metrics (`delta_mag`, flawed `cosine_sim`, `kl_div`, scalar `fisher_grad`). | Continuous Dirichlet convex metric blends discovered via forward simulation (`pareto_auto`). |
| **Quantile Computation** | Stochastic `torch.randperm` (non-deterministic, triggers GPU synchronization locks). | Deterministic strided uniform subsampling (lock-free, reproducible). |
| **Norm Imbalance** | None. Aggressive fine-tunes with massive parameter norms completely erase conservative donors. | Sub-linear **Anti-Monopoly Energy Normalization** ($\tau_i \leftarrow \tau_i \cdot (\text{median} / \|\tau_i\|)^{1-p}$). |
| **Interference Resolution** | Pure additive superposition; destructive interference zeroes out coordinates. | **MAGIC Minority Sign Inversion** (Aikido reflection across majority hyperplanes). |
| **MoE Architecture** | Blind matrix flattening; destroys gating topologies and mixes routing projections. | Dedicated MoE dispatch (`is_moe`, `blend_experts`, `router_strategy`, `router_metric`) with expert-axis softmax. |
| **Observability** | Silent execution. | Thread-safe live visual ASCII HUD and persistent JSON win tracker. |

---

## 2. Mathematical Formulation

Let $\mathbf{W}_{\text{base}} \in \mathbb{R}^{d_1 \times d_2}$ be the anchor model parameters (e.g., a pretrained foundation model or a Riemannian Karcher consensus centroid) and let $\{\mathbf{W}_i\}_{i=1}^N$ be a collection of $N$ fine-tuned donor checkpoints.

### 2.1 Task Vector Extraction and Anti-Monopoly Normalization
Raw task vectors are formed by:
$$\boldsymbol{\tau}_i = \mathbf{W}_i - \mathbf{W}_{\text{base}}, \quad \forall i \in \{1, \dots, N\}$$

In unconstrained multi-donor merges, heavily fine-tuned models often exhibit task vector norms $\|\boldsymbol{\tau}_i\|_F$ an order of magnitude larger than conservative domain adaptations. When $p_{\text{mono}} < 1.0$, MultiFusion v4 applies a global energy compression:

$$\tilde{\boldsymbol{\tau}}_i = \boldsymbol{\tau}_i \cdot \left( \frac{\text{median}_j \|\boldsymbol{\tau}_j\|_F}{\|\boldsymbol{\tau}_i\|_F + \epsilon} \right)^{1 - p_{\text{mono}}}$$

When $p_{\text{mono}} = 1.0$, the original norms are preserved. Setting $p_{\text{mono}} \in [0.35, 0.50]$ compresses outlier disparity, ensuring conservative donors retain voice in the final synthesis.

### 2.2 Constructive Interference via Aikido / MAGIC Minority Inversion
Standard consensus schemes (e.g., TIES, DARE) erase or zero out weights where donor models disagree on the direction of gradient travel. Under `sce_inversion_mode = True`, MultiFusion v4 adopts an Aikido-inspired strategy: rather than nullifying dissenting energy, minority signs are reflected across the majority consensus plane:

$$\mathbf{S}_{\text{maj}} = \text{sgn}\left(\sum_{i=1}^N \tilde{\boldsymbol{\tau}}_i\right)$$
$$\hat{\boldsymbol{\tau}}_i = \begin{cases} -\tilde{\boldsymbol{\tau}}_i & \text{if } \text{sgn}(\tilde{\boldsymbol{\tau}}_i) \neq \mathbf{S}_{\text{maj}} \text{ and } \tilde{\boldsymbol{\tau}}_i \neq 0 \\ \tilde{\boldsymbol{\tau}}_i & \text{otherwise} \end{cases}$$

This preserves high-frequency novelty energy while eliminating destructive directional cancellation.

---

## 3. The 5-Metric Geometric Manifold

MultiFusion v4 replaces the ad-hoc approximations of v1 with five mathematically grounded importance metrics:

```
                                  [ Task Vector Tau_i ]
                                            │
        ┌───────────────┬───────────────────┼───────────────────┬───────────────┐
        ▼               ▼                   ▼                   ▼               ▼
   1. DELTA_MAG     2. KL_DIV         3. COSINE_SIM       4. VARIANCE_SCE  5. FISHER_GRAD
   (L1 Departure)  (MoE Routing)     (Angular Divergence) (Channel Energy)  (Riemannian Log)
        │               │                   │                   │               │
        └───────────────┴───────────────────┼───────────────────┴───────────────┘
                                            ▼
                           [ Continuous Dirichlet Alpha-Mixture ]
                                            │
                                            ▼
                           [ Deterministic Tukey Outlier Gate ]
                                            │
                                            ▼
                           [ Fused Base + Masked Synthesized Delta ]
```

### 1. $L_1$ Magnitude Departure (`delta_mag`):
$$S_{\text{mag}} = |\hat{\boldsymbol{\tau}}_i|$$
Measures raw absolute parameter deviation from the base manifold, serving as the primary novelty metric utilized in TIES and DELLA.

### 2. Dimension-Calibrated Distributional Shift (`kl_div`):
$$S_{\text{kl}} = |\mathbf{W}_i - \mathbf{W}_{\text{base}}| \odot \sum_{k} p_k \log\left(\frac{p_k}{q_k} + \epsilon\right)$$
Where $p = \text{Softmax}(\mathbf{W}_i, \text{dim}=d)$ and $q = \text{Softmax}(\mathbf{W}_{\text{base}}, \text{dim}=d)$.

* **MoE Dimension Correction:** In v1, softmax was blindly computed across `dim=-1`. In v4, router gate matrices (`*.gate.weight`) are inspected: if the tensor dimensions are structured as $[\text{num\_experts}, \text{hidden\_dim}]$ with $\text{dim}_0 \le 128$, the softmax is evaluated across $\text{dim}=0$. This guarantees that KL measures genuine **categorical expert routing probability shifts**, rather than meaningless normalization across the hidden input dimension.

### 3. Row-Wise Angular Divergence (`cosine_sim`):
Unlike v1 (which computed a single global scalar dot product and broadcasted it across the entire matrix), v4 computes channel/row-wise angular divergence:
$$S_{\text{cos}} = |\hat{\boldsymbol{\tau}}_{i, r}| \odot \left(2.0 - \frac{\langle \mathbf{W}_{i, r}, \mathbf{W}_{\text{base}, r} \rangle}{\|\mathbf{W}_{i, r}\|_2 \|\mathbf{W}_{\text{base}, r}\|_2 + \epsilon}\right)$$
This aligns with Model Stock geometry: weights that execute sharp angular rotations relative to the base manifold receive proportionally higher saliency scores.

### 4. Channel-Wise Energy Variance (`variance_sce`):
$$S_{\text{sce}} = |\hat{\boldsymbol{\tau}}_i| \odot \sqrt{\text{Var}_{\text{channel}}(\hat{\boldsymbol{\tau}}_i) + \epsilon}$$
Evaluates the variance of parameter deltas within each functional neuron or channel, capturing structured activation dispersion.

### 5. Riemannian Tangent Energy Proxy (`fisher_grad`):
In data-free merging, the true Fisher Information Matrix cannot be explicitly evaluated without calibration batches. MultiFusion v4 implements a Riemannian hyperspherical Log-Map proxy:
$$S_{\text{fsh}} = |\mathbf{W}_i - \mathbf{W}_{\text{base}}| \odot \left(1.0 + \left\|\frac{\theta}{\sin\theta} \left(\hat{\mathbf{u}}_i - \cos\theta \hat{\mathbf{u}}_{\text{base}}\right)\right\|\right)$$
Where $\hat{\mathbf{u}} = \mathbf{W} / \|\mathbf{W}\|_2$ and $\theta = \arccos(\langle \hat{\mathbf{u}}_i, \hat{\mathbf{u}}_{\text{base}} \rangle)$. This measures geodesic tangent energy along the Riemannian hypersphere.

---

## 4. The Hierarchical Monte Carlo Pareto Engine (`pareto_auto`)

Rather than forcing the practitioner to guess which metric suits a given layer, `importance_metric: "pareto_auto"` frames mask generation as an online multi-objective forward simulation.

### 4.1 Power-of-Two Multi-Resolution Subspace Projection
To prevent GPU execution bottlenecks and avoid memory explosion on multi-billion parameter layers, the engine projects the target tensors onto a power-of-two strided subspace $\Omega_N$ ($N \in \{1024, 4096, 16384\}$):

$$\text{stride} = \max\left(1, \left\lfloor \frac{\text{numel}(\boldsymbol{\tau})}{N} \right\rfloor\right)$$
$$\Omega_N = \{ j \cdot \text{stride} \mid j \in [0, N-1] \}$$

$$\mathbf{W}_{\text{sub}} = \mathbf{W}[\Omega_N], \quad \boldsymbol{\tau}_{\text{sub}} = \boldsymbol{\tau}[\Omega_N]$$

The candidate importance fields $\{S_m\}_{m=1}^5$ are precomputed on $\Omega_N$ and $L_2$-normalized:
$$\tilde{S}_m = \frac{S_m}{\|S_m\|_2 + \epsilon}$$

### 4.2 Continuous Dirichlet Stochastic Exploration
The engine executes a baseline evaluation across all five pure metrics ($\vec{\alpha} \in \{\mathbf{e}_1, \dots, \mathbf{e}_5\}$), followed by $K$ stochastic Monte Carlo trials (`mc_trials: 16` to `64`). For each trial $t \in [1, K]$, a random candidate mixture vector is sampled from a symmetric Dirichlet distribution:

$$\vec{\alpha}^{(t)} \sim \text{Dirichlet}(\mathbf{1}_5), \quad \sum_{m=1}^5 \alpha_m^{(t)} = 1$$

Simultaneously, the Tukey IQR fence is jittered to explore threshold sensitivity:
$$\eta^{(t)} \sim \mathcal{U}(\text{fence} - 0.3, \text{fence} + 0.3)$$

The candidate subspace mask is generated via lock-free deterministic quantiles:
$$\mathbf{M}_{\text{sub}}^{(t)} = \mathbb{I}\left( \sum_{m=1}^5 \alpha_m^{(t)} \tilde{S}_m \ge Q_2 + \eta^{(t)} (Q_3 - Q_1) \right)$$

### 4.3 Multi-Objective Compound Pareto Loss
Each candidate mask $\mathbf{M}_{\text{sub}}^{(t)}$ synthesizes a candidate tensor $\mathbf{W}_{\text{fused, sub}}^{(t)} = \mathbf{W}_{\text{base, sub}} + \boldsymbol{\tau}_{\text{sub}} \odot \mathbf{M}_{\text{sub}}^{(t)}$ and is evaluated against four competing objectives:

$$\mathcal{L}_{\text{Pareto}} = 2.5 \cdot \mathcal{E}_{\text{drift}} + 1.2 \cdot \mathcal{E}_{\text{rec}} + 0.4 \cdot \mathcal{E}_{\text{noise}} + 0.02 \cdot \mathcal{R}_{\text{sparsity}}$$

1. **Base Manifold Directional Drift ($\mathcal{E}_{\text{drift}}$):** Prevents degradation of core foundational representations:
$$\mathcal{E}_{\text{drift}} = 1.0 - \frac{\langle \mathbf{W}_{\text{fused, sub}}, \mathbf{W}_{\text{base, sub}} \rangle}{\|\mathbf{W}_{\text{fused, sub}}\|_2 \|\mathbf{W}_{\text{base, sub}}\|_2 + \epsilon}$$
2. **Donor Novelty Signal Recovery ($\mathcal{E}_{\text{rec}}$):** Measures whether salient donor features survived gating:
$$\mathcal{E}_{\text{rec}} = \frac{\|(\mathbf{W}_{\text{donor, sub}} - \mathbf{W}_{\text{base, sub}}) - (\mathbf{W}_{\text{fused, sub}} - \mathbf{W}_{\text{base, sub}})\|_2}{\|\mathbf{W}_{\text{donor, sub}} - \mathbf{W}_{\text{base, sub}}\|_2 + \epsilon}$$
3. **Null-Space Noise Amplification ($\mathcal{E}_{\text{noise}}$):** Penalizes variance explosions in inactive regions:
$$\mathcal{E}_{\text{noise}} = \frac{|\text{Var}(\mathbf{W}_{\text{fused, sub}} \odot (1 - \mathbf{M}_{\text{sub}})) - \text{Var}(\mathbf{W}_{\text{base, sub}})|}{\text{Var}(\mathbf{W}_{\text{base, sub}}) + \epsilon}$$
4. **Regularized Sparsity Prior ($\mathcal{R}_{\text{sparsity}}$):** An asymmetric barrier function preventing trivial collapse into all-zero or all-one masks:
$$\mathcal{R}_{\text{sparsity}} = \frac{1}{\bar{\mathbf{M}}_{\text{sub}} + 0.05} + \frac{1}{1.05 - \bar{\mathbf{M}}_{\text{sub}}}$$

The optimal tuple $(\vec{\alpha}^*, \eta^*)$ minimizing $\mathcal{L}_{\text{Pareto}}$ is extracted and applied to the full-dimensional tensor.

---

## 5. Live Visual Audit Telemetry HUD

To provide complete transparency during long-running merges, MultiFusion v4 incorporates a thread-safe monitoring interface.

### Console Output Example
```text
╔════════════════════════════════════════════════════════════════════════════════════════════╗
║ [MultiFusion v4 Audit] model.layers.14.mlp.experts.3.up_proj.weight                        ║
║ Base: Gemma4_MoE_Karcher_Consensus       | Mode: MC-Pareto (K=32)       | Fence: 1.48      ║
║ Pareto Telemetry: Winner: Trial #19/32 | Loss: 0.0412 | Pass Rate: 16.4%                   ║
║ Winning Blend:    MAG:0.35 | KL:0.08 | COS:0.24 | SCE:0.21 | FSH:0.12                      ║
║ Global Wins [ 42]: MAG:14(33%) | KL:3(7%) | COS:9(21%) | SCE:12(29%) | FSH:4(10%)         ║
╠════════════════════════════════════════════════════════════════════════════════════════════╣
║ Donor Model                            | Norm   | Pass%  | Share% | Effective Impact       ║
╟----------------------------------------+--------+--------+--------+------------------------╢
║ Gemma4-Writer-26BA4B                   |   1.12 |  16.4% |   8.2% | █                      ║
║ Pantheon-Reasoning-26B-A4B-1.1         |   1.45 |  16.4% |  11.4% | ██                     ║
║ Promethean-Dawn-26B-A4B                |   3.82 |  16.4% |  42.1% | █████████              ║
║ Orion-26B-A4B-v1.1                     |   0.94 |  16.4% |   6.8% | █                      ║
║ Novelty_Compound_Vector                |   2.80 |  16.4% |  31.5% | ██████                 ║
╚════════════════════════════════════════════════════════════════════════════════════════════╝
```

### Telemetry Persistence
Statistics are automatically recorded across threads and written to `multi_fusion_metric_stats.json`:
```json
{
  "total_layers": 42,
  "win_counts": {
    "delta_mag": 14,
    "kl_div": 3,
    "cosine_sim": 9,
    "variance_sce": 12,
    "fisher_grad": 4
  },
  "win_percentages": {
    "MAG": "33.3%",
    "KL": "7.1%",
    "COS": "21.4%",
    "SCE": "28.6%",
    "FSH": "9.5%"
  }
}
```

---

## 6. Configuration & Usage

### 6.1 Installation
MultiFusion v4 is implemented as a drop-in `MergeMethod` within the MergeKit ecosystem. Copy `multi_fusion_v4.py` into your MergeKit installation:

```bash
cp multi_fusion_v4.py mergekit/mergekit/merge_methods/multi_fusion.py
```
Register the method in `mergekit/merge_methods/registry.py`:
```python
from mergekit.merge_methods.multi_fusion import MultiFusionMerge

STATIC_MERGE_METHODS: List[MergeMethod] = [
    # ...
    MultiFusionMerge(),
]
```

### 6.2 Production Multi-Donor Recipe (`recipe.yaml`)
Below is an example showing an ensemble of specialized fine-tunes synthesized onto a Karcher Consensus base anchor:

```yaml
architecture: Gemma4ForConditionalGeneration
base_model: B:/26B/Gemma4_26B_MoE_Karcher_Consensus

models:
  - model: B:/26B/ConicCat--Gemma4-Writer-26BA4B
    parameters:
      weight: 1.0
  - model: B:/26B/Cyclone-Labs--Eclipsed-Phoenix-26B-A4B
    parameters:
      weight: 1.0
  - model: B:/26B/Gryphe--Pantheon-Reasoning-26B-A4B-1.1
    parameters:
      weight: 1.0
  - model: B:/26B/Naphula--Promethean-Dawn-26B-A4B
    parameters:
      weight: 1.2
  - model: B:/26B/TheDrummer--Orion-26B-A4B-v1.1
    parameters:
      weight: 1.0
  - model: B:/26B/Gemma4_26B_MoE_Novelty_Blend
    parameters:
      weight: 2.0

merge_method: multi_fusion
parameters:
  # Pareto Auto-Arbiter Settings
  importance_metric: "pareto_auto"    # Online multi-objective metric optimization
  mc_trials: 32                       # 32 Dirichlet Monte Carlo trials per tensor
  power_of_two_scale: 8192            # 8K resolution strided subspace projection
  tukey_fence: 1.5                    # Outlier rejection fence (median + 1.5 * IQR)

  # Energy Balance & Inversion
  anti_monopoly_power: 0.40           # Compresses norm dominance across donors
  sce_inversion_mode: true            # Invert minority sign conflicts (Aikido resolution)

  # MoE Architecture Handling
  is_moe: true
  router_strategy: "metric"
  router_metric: "kl_div"             # Calibrated expert-axis categorical divergence
  blend_experts: true

dtype: float32
out_dtype: bfloat16
tokenizer:
  source: union
chat_template: auto
```

Execute using the standard MergeKit CLI:
```bash
mergekit-yaml recipe.yaml ./output_model --cuda --low-cpu-memory
```

---

## 7. Empirical Recommendations & Best Practices

1. **Base Anchor Selection:**
   * When merging large multi-donor ensembles, establish a **geometric barycenter** (such as a Karcher Mean merge) as the `base_model`. In this coordinate system, $\boldsymbol{\tau}_i = \mathbf{W}_i - \mathbf{W}_{\text{karcher}}$ isolates pure idiosyncratic task specialisms, yielding maximal sparsity and clean Tukey outlier detection.
2. **Setting Anti-Monopoly Power (`anti_monopoly_power`):**
   * Default: `1.0` (unconstrained).
   * For ensembles containing 5+ models with uneven parameter norms, set `anti_monopoly_power: 0.35 - 0.45` to prevent over-trained checkpoints from muting subtle domain donors.
3. **Monte Carlo Trial Tuning (`mc_trials`):**
   * Rapid Prototyping: `mc_trials: 0` evaluates pure metrics in under 2 ms per tensor.
   * Standard Production: `mc_trials: 16 - 32` strikes an optimal balance between quality and synthesis time.
   * Deep Exploration: `mc_trials: 64` achieves maximal Pareto frontier exploration for final model synthesis.
4. **Mixture of Experts Routing:**
   * When in doubt, keep `router_strategy: "metric"` and `router_metric: "kl_div"` (or "pareto_auto") when merging MoE architectures to guarantee that gating matrices preserve sharp categorical dispatch distributions across expert banks.