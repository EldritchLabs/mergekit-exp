# mergekit-exp
*AI analysis of the updates followed by the original mergekit readme.*

## Summary

`mergekit-exp` is a fork of Arcee AI's `mergekit` that adds a large number of new **experimental merge methods**, plus supporting infrastructure like MoE-aware task variants, support for new architectures like **Gemma 4**, and a simplified method-definition API.

---

## New Merge Methods

| Method | File | Idea |
|---|---|---|
| BCR (Barycentric Conflict Resolution) | `bcr.py` | Resolves sign conflicts via weighted barycentric mean instead of discarding, using a `dominance` factor <cite repo="EldritchLabs/mergekit-exp" path="mergekit/merge_methods/bcr.py" start="1-33" end="1-33" /> |
| BRF (Barycentric Resonance Flow) | `brf.py` | Multi-stage: resonance analysis, BCR-style conflict resolution, Karcher-mean-grounded directional flow <cite repo="EldritchLabs/mergekit-exp" path="mergekit/merge_methods/brf.py" start="35-49" end="35-49" /> |
| MoE-Karcher | `moe_karcher.py` | Blends MoE experts via geometric (Karcher) mean, with router-weight handling strategies <cite repo="EldritchLabs/mergekit-exp" path="mergekit/merge_methods/moe_karcher.py" start="1-24" end="1-24" /> |
| MoE-DELLA | `moe_della.py` | MoE-aware DELLA magnitude pruning + TIES sign consensus on experts <cite repo="EldritchLabs/mergekit-exp" path="mergekit/merge_methods/moe_della.py" start="1-29" end="1-29" /> |
| MoE-SLERP | `moe_slerp.py` | Spherical interpolation between exactly two MoE models' experts <cite repo="EldritchLabs/mergekit-exp" path="mergekit/merge_methods/moe_slerp.py" start="1-25" end="1-25" /> |
| SCF (Selective Coherence Fusion) | `scf.py` | Two-stage: SCE-style variance selection, then Arcee-Fusion-style importance scoring/thresholding <cite repo="EldritchLabs/mergekit-exp" path="mergekit/merge_methods/scf.py" start="1-57" end="1-57" /> |
| SCREAM | `scream.py` | Combines Model Stock, DELLA, and SCE components with weighted blending <cite repo="EldritchLabs/mergekit-exp" path="mergekit/merge_methods/scream.py" start="1-45" end="1-45" /> |
| LRP (Layer-wise Relevance Propagation) merge | `lrp.py` | Uses `[Delta] * Variance(Delta)` "self-relevance" proxy, with "MAGIC" sign-inversion mode <cite repo="EldritchLabs/mergekit-exp" path="mergekit/merge_methods/lrp.py" start="27-36" end="27-36" /> |
| Multi-Fusion | `multi_fusion.py` | Dynamic-threshold (Tukey fence) fusion mask over multiple importance metrics (`kl_div`, `delta_mag`, `cosine_sim`, `fisher_grad`) <cite repo="EldritchLabs/mergekit-exp" path="mergekit/merge_methods/multi_fusion.py" start="1-19" end="1-19" /> |
| PCB (Parameter Competition Balancing) | `pcb.py` | Uses intra-model/inter-model importance (`b_intra`, `b_inter`) to weight task vectors (per wiki) |
| QLIPHOTH v2 | `qliphoth.py` | Orthogonal-injection method with SVD rank, stability checks, deviation bounds <cite repo="EldritchLabs/mergekit-exp" path="mergekit/merge_methods/qliphoth.py" start="85-104" end="85-104" /> |

## The `easy_define` API

A significant piece of new infrastructure is the `@merge_method` decorator in `mergekit/merge_methods/easy_define.py`, which lets a plain Python function (with `tensors: List[torch.Tensor]` and optional `base_tensor`) be auto-wrapped into a full `MergeMethod`/`Task` pair <cite repo="EldritchLabs/mergekit-exp" path="mergekit/merge_methods/easy_define.py" start="26-49" end="26-49" />. Nearly all of the custom methods (`bcr.py`, `scf.py`, `scream.py`, `brf.py`) use this decorator instead of hand-writing a `MergeMethod`/`Task` subclass <cite repo="EldritchLabs/mergekit-exp" path="mergekit/merge_methods/nearswap.py" start="11-18" end="11-18" />, <cite repo="EldritchLabs/mergekit-exp" path="mergekit/merge_methods/ram.py" start="11-15" end="11-15" />. This is a developer-experience/graph-construction change: instead of manually building `Task` subclasses like `MoEKarcherTask` <cite repo="EldritchLabs/mergekit-exp" path="mergekit/merge_methods/moe_karcher.py" start="55-74" end="55-74" />, method authors write a plain function and decorate it.

## MoE Task/Graph Additions

The MoE-aware methods (`MoEKarcherTask`, `MoEDellaTask`, `MoESlerpTask`) all inherit from `Task` in `mergekit/graph.py` (the core task-graph system) and add expert/router-detection logic not present in standard (non-MoE) merge methods — e.g. `_is_expert_weight` and `_is_router_weight` regex matching in `MoEKarcherTask` <cite repo="EldritchLabs/mergekit-exp" path="mergekit/merge_methods/moe_karcher.py" start="76-94" end="76-94" />, and per-router strategy handling (`average`, `karcher`, `first`, `random_init`) <cite repo="EldritchLabs/mergekit-exp" path="mergekit/merge_methods/moe_karcher.py" start="126-138" end="126-138" />. This is graph-level: these `Task` subclasses plug into the same `arguments()`/`execute()`/`group_label()` interface as core tasks, e.g. `KarcherTask` <cite repo="EldritchLabs/mergekit-exp" path="mergekit/merge_methods/karcher.py" start="20-38" end="20-38" />, so they integrate into the existing merge planning/execution graph without altering its core mechanics.

## Summary of differences: `mergekit-exp` vs. official `arcee-ai/mergekit`

`mergekit-exp` (EldritchLabs) forks the official `arcee-ai/mergekit` at roughly the `a6e40288` commit (June 2026) and layers on a set of experimental changes across 5 recent "Add files via upload" commits [1](#0-0) . The changes fall into three buckets: **new experimental merge methods**, **graph/executor rewrites** for low-VRAM execution, and a handful of **compatibility patches/hacks** (notably for a "Gemma 4" architecture that doesn't exist in the upstream repo).

### 1. New merge methods added

All added in `mergekit/merge_methods/`, most headed with `# mergekit/merge_methods/X.py by Naphula`:

- **`bcr.py`** — Barycentric Conflict Resolution: TIES-style consensus for agreeing task-vector signs, and a weighted barycentric mean (using a per-model `dominance` factor) for conflicting signs, instead of simply discarding conflicts [2](#0-1) .
- **`brf.py`** — Barycentric Resonance Flow: a 3-stage method (spatial/spectral "resonance" similarity analysis → BCR-style conflict resolution → directional flow rescaled by a true Riemannian/Karcher-mean magnitude) [3](#0-2) .
- **`delerp.py`** — Decomposed Linear Interpolation (originally by "GrimJim", "Modified by Naphula to fix various bugs"): splits direction (NLERP) from magnitude (max norm) [4](#0-3) .
- **`lrp.py`** — LRP Merge: uses a "Layer-wise Relevance Propagation" proxy (`|delta| * variance(delta)`) to select top-k important weights per model, with an optional "MAGIC" sign-inversion mode for conflicting deltas [5](#0-4) .
- **`moe_della.py`** — MoE-aware DELLA: applies DELLA magnitude pruning + TIES sign consensus per-expert for MoE models, with configurable router-merging strategy (`della`/`average`/`first`/`random_init`) [6](#0-5) .
- **`moe_karcher.py`** — MoE-aware Karcher mean merge, blending corresponding experts geometrically with the same router-strategy options [7](#0-6) .
- **`moe_slerp.py`** — MoE-aware SLERP for exactly 2 MoE models, with per-expert SLERP and configurable router strategy [8](#0-7) .
- **`multi_fusion.py`** — Dynamic-threshold fusion using Tukey-fence (IQR-based) outlier detection to build an importance/fusion mask [9](#0-8) .
- **`pcb.py`** — Parameter Competition Balancing (this one is *not* Naphula's — it carries the original arXiv-2410.02396 implementation license header from Charles O. Goddard) [10](#0-9) .
- **`qliphoth.py`** — "QLIPHOTH v2": orthogonal task-vector injection against a "rival" consensus with conservative SVD-based null-space projection, adaptive variance scaling, and a stability/deviation safety clamp [11](#0-10) .
- **`scf.py`** — Selective Coherence Fusion: two-stage filter combining SCE-style variance masking with Arcee-Fusion-style KL-divergence importance scoring [12](#0-11) .
- **`scream.py`** — "SCREAM": blends Model Stock, DELLA, and SCE components into one weighted merge [13](#0-12) .

These new methods are **now wired directly into the method registry** and are selectable via YAML config out of the box.

### 2. Modifications to existing methods

- **`generalized_task_arithmetic.py`** (used by TIES/DELLA/task-arithmetic methods) — adds a `log_della_audit()` function that prints/writes a bar-chart-style audit log of task-vector norms/weights/densities per layer to `della_audit.log` [1](#0-0) .
- **`model_stock.py`** — similarly adds a `log_model_stock_audit()` writing to `model_stock_audit.log`, showing base vs. donor interpolation weight (`t`) per layer.
- **`sce.py`** — modified (34 additions), presumably similar auditing/robustness tweaks.

### 3. Core graph/execution engine rewrite (`mergekit/graph.py`, "graph_v18")

The commit `05d95cec` ("Updated to graph_v18 and other fixes for Gemma 4") heavily rewrites `mergekit/graph.py` (769 additions/537 deletions) to add **adaptive VRAM-aware chunked execution**, targeted at low-VRAM GPUs like an RTX 3060 Ti (8GB) [14](#0-13) :

- A large configuration block at the top of the file (`TARGET_VRAM_GB`, `VRAM_SAFETY_MARGIN_GB`, `CUDA_MAX_SPLIT_SIZE_MB`, `CHUNK_REDUCTION_FACTOR`, `TASK_MULTIPLIERS`, etc.) hardcodes GPU-tuning knobs directly in source.
- `Executor` gains `_get_adaptive_chunk_size`, `_execute_chunked`, and `_execute_with_fallback` methods implementing a multi-tier fallback: try full-GPU execution → adaptive chunking → fixed chunk-size list → CPU fallback, with OOM-triggered progressive chunk shrinking and power-of-2 alignment.
- Task-specific VRAM multipliers are defined for made-up/experimental task names (`ModelStock`, `Karcher`, `Consensus`, `Prometheus`, `Tensorguard`) that don't correspond to anything in upstream `mergekit`, suggesting there are other in-progress/unlisted experimental tasks not fully shown in what was uploaded.
- This is a significant divergence from upstream's simpler `Executor._run`/`_move_tensors` implementation, which has no chunking or VRAM-target logic at all [15](#0-14) .

Related smaller fixes in the same commit: `mergekit/common.py::get_config_value` gains a fallback into `text_config` for nested/multimodal configs; `mergekit/sparsify.py::della_magprune` adds safety clamping of `density`/`epsilon` instead of raising a `ValueError`; `mergekit/plan.py` tweaks alias handling to accept tuples as well as lists.

### 4. Gemma 4 support (unofficial/speculative)

Several commits add support for a `"gemma4"` / `Gemma4ForConditionalGeneration` architecture that does not exist in upstream mergekit (upstream only has Gemma/Gemma2/Gemma3):

- New file `mergekit/_data/architectures/gemma4.json` defining a MoE-capable Gemma-4-like layer template with vision-tower fields, packed experts (`experts.gate_up_proj`/`down_proj`), and a router [16](#0-15) .
- `mergekit/architecture/auto.py` is regressed/rewritten to remove the newer upstream tensor-name/checkpoint-conversion logic (`can_convert_checkpoint_keys`, `convert_checkpoint_tensors`) and hardcode `optional = True` for every weight plus force `tied_keys = None`, with a comment: *"Force untying for Gemma 4 configurations to ensure lm_head is compiled"* and *"Gemma 4 alternates attention types; tensors present in layer 0 may be missing in layer 5"* [17](#0-16) .
- `mergekit/io/tasks.py` adds a hardcoded "GGUF MAPPING HACK" translating HF tensor names to GGUF-style names (`token_embd.weight`, `blk.N.attn_q.weight`, etc.) and changes the `LoaderCache` key to use the string model path instead of the `ModelReference` object "to prevent Pydantic object hash collisions on Windows" [18](#0-17) . It also silently skips missing required tensors instead of raising `RuntimeError`.

### Net effect

This is not a clean feature branch — it removes some newer upstream capabilities (auto-architecture's checkpoint-key conversion system) while bolting on Gemma-4-specific hacks, a VRAM-chunking executor rewrite, and a dozen registered experimental merge methods.

---

Comparing your pasted snippet to the current `GeneralizedTaskArithmeticMerge`/`GTATask` implementation in `mergekit/merge_methods/generalized_task_arithmetic.py`, the proposed update adds four things that don't exist in the current GTA code:

1. **MoE-safety guard** — skips execution early if any input tensor is `None`/missing, and raises a clear `RuntimeError` if the `base_model` tensor isn't present, instead of failing silently or with an obscure `KeyError`.
2. **Embedding-size rectification** — calls `rectify_embed_sizes` across all input tensors before computing deltas, so mismatched vocabulary/embedding sizes across MoE models don't crash the merge.
3. **Router-weight detection** — a new `_is_router_weight` helper method recognizing MoE gate/router parameter names.
4. **Regex/filter-aware per-model weight resolution** — in `get_task_vectors`, instead of a flat `d[p] = tensor_parameters[model][p]` assignment, the new code lets the `weight` parameter be a *list of filter/value conditions*, resolving the effective weight based on substring or `|`-delimited pattern matches against the tensor's name (falling back to `1.0` if nothing matches).

---

### Detailed breakdown

#### 1. MoE guard + base_model check
Currently, `GTATask.execute` goes straight into `get_task_vectors` without checking for missing tensors. Your proposal inserts explicit guards:
```python
if not tensors or any(t is None for t in tensors.values()):
    return None
if self.base_model not in tensors:
    raise RuntimeError(...)
```
This mirrors defensive patterns already used elsewhere in the codebase, e.g. `MoEDellaTask.execute`'s own guard `if not tensors or any(t is None ...)` and base-model check.

#### 2. `rectify_embed_sizes` integration
The current GTA implementation has no embedding-size handling at all in `execute()` — mismatches are only handled inside `get_task_vectors` by truncating embed weights when `weight_info.is_embed`. The proposal imports and calls `rectify_embed_sizes` directly in `execute()`, the same utility already used by `MoEDellaTask` to align vocab/embedding dims before diffing.

#### 3. Router-weight detection (`_is_router_weight`)
This method is added to `GTATask` but, notably, is **not called anywhere** in the pasted `execute()` body — it's defined but unused in what you shared. It closely parallels the router-detection logic already used in `MoEDellaMerge`'s task, `_is_router_weight`, but extends the pattern list with `router.proj.weight` and `router.scale`. Since it's unused, the "offer" here is essentially scaffolding for future router-aware branching in plain GTA (task_arithmetic/ties/dare/della), analogous to what `moe_della.py` already does for routing strategy (`average`, `first`, `random_init`, `della`).

#### 4. Filter-based per-model weight resolution
This is the most functionally significant addition. Today, `get_task_vectors` just copies whatever tensor parameter value was resolved upstream. The proposal instead lets `weight` be a list of `{filter, value}` conditions and resolves per-tensor:
- `filter is None` or `"*"` → wildcard default
- `filt` containing `"|"` → OR-matches any of the pipe-separated substrings against the parameter name
- otherwise → plain substring match against `parameter_name`

This would let a single merge config assign different `weight` values to different layers/modules (e.g., attention vs. MLP vs. router) within the same model entry, which the current `ConfigParameterDef(name="weight", required=True)` scalar/simple-list model doesn't support.

## Notes
- I compared your pasted code against the current `mergekit/merge_methods/generalized_task_arithmetic.py` as indexed in the `EldritchLabs/mergekit-exp` repo; the diffs above are the concrete "offers" of the update relative to that baseline.
- The `_is_router_weight` addition appears incomplete/dead code in the snippet you shared (defined but never invoked in `execute()`), so I can't confirm its intended integration without seeing more of the file.

---

# mergekit

[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/arcee-ai/mergekit/pre-commit.yml?label=Tests)](https://github.com/arcee-ai/mergekit/actions/workflows/pre-commit.yml)
[![Arcee Discord](https://img.shields.io/badge/Arcee%20Discord-Arcee%20Discord?logo=discord&logoColor=white&color=5865F2)](https://discord.gg/arceeai)

`mergekit` is a toolkit for merging pre-trained language models. `mergekit` uses an out-of-core approach to perform unreasonably elaborate merges in resource-constrained situations. Merges can be run entirely on CPU or accelerated with as little as 8 GB of VRAM. Many merging algorithms are supported, with more coming as they catch my attention.

## Contents

- [Why Merge Models?](#why-merge-models)
- [Features](#features)
- [Installation](#installation)
- [Community & Support](#community--support)
  - [Contributing](#contributing)
  - [Community Tools](#community-tools)
- [Usage](#usage)
- [Merge Configuration](#merge-configuration)
  - [Parameter Specification](#parameter-specification)
  - [Tokenizer Configuration](#tokenizer-configuration)
  - [Chat Template Configuration](#chat-template-configuration)
  - [Examples](#examples)
- [Merge Methods](#merge-methods)
- [LoRA Extraction](#lora-extraction)
- [Mixture of Experts Merging](#mixture-of-experts-merging)
- [Evolutionary Merge Methods](#evolutionary-merge-methods)
- [Multi-Stage Merging (`mergekit-multi`)](#multi-stage-merging-mergekit-multi)
- [Raw PyTorch Model Merging (`mergekit-pytorch`)](#raw-pytorch-model-merging-mergekit-pytorch)
- [Tokenizer Transplantation (`mergekit-tokensurgeon`)](#tokenizer-transplantation-mergekit-tokensurgeon)
- [Citation](#citation)

## Why Merge Models?

Model merging is a powerful technique that allows combining the strengths of different models without the computational overhead of ensembling or the need for additional training. By operating directly in the weight space of models, merging can:

- Combine multiple specialized models into a single versatile model
- Transfer capabilities between models without access to training data
- Find optimal trade-offs between different model behaviors
- Improve performance while maintaining inference costs
- Create new capabilities through creative model combinations

Unlike traditional ensembling which requires running multiple models, merged models maintain the same inference cost as a single model while often achieving comparable or superior performance.

## Features

Key features of `mergekit` include:

- Supports Llama, Mistral, GPT-NeoX, StableLM, and more
- Many [merge methods](#merge-methods)
- GPU or CPU execution
- Lazy loading of tensors for low memory use
- Interpolated gradients for parameter values (inspired by Gryphe's [BlockMerge_Gradient](https://github.com/Gryphe/BlockMerge_Gradient) script)
- Piecewise assembly of language models from layers ("Frankenmerging")
- [Mixture of Experts merging](#mixture-of-experts-merging)
- [LORA extraction](#lora-extraction)
- [Evolutionary merge methods](#evolutionary-merge-methods)
- [Multi-stage merging](#multi-stage-merging-mergekit-multi) for complex workflows.
- [Merging of raw PyTorch models (`mergekit-pytorch`)](#raw-pytorch-model-merging-mergekit-pytorch).

## Installation

```sh
git clone https://github.com/arcee-ai/mergekit.git
cd mergekit

pip install -e .  # install the package and make scripts available
```

If the above fails with the error of:

```
ERROR: File "setup.py" or "setup.cfg" not found. Directory cannot be installed in editable mode:
(A "pyproject.toml" file was found, but editable mode currently requires a setuptools-based build.)
```

You may need to upgrade pip to > 21.3 with the command `python3 -m pip install --upgrade pip`.

## Community & Support

- **Issues**: [GitHub Issues](https://github.com/arcee-ai/mergekit/issues)
- **Discussions**: [Arcee Discord](https://discord.gg/arceeai)

### Contributing

We welcome contributions to `mergekit`! If you have ideas for new merge methods, features, or other improvements, please check out our [contributing guide](CONTRIBUTING.md) for details on how to get started.

### Community Tools

- **[FrankensteinAI](https://frankenstein-ai.com/)**: For those who prefer a browser-based experience without local setup or hardware wrangling, the team at FrankensteinAI has built a hosted platform powered by `mergekit`. Also features a community gallery and leaderboard for sharing and comparing merged models.

## Usage

The script `mergekit-yaml` is the main entry point for `mergekit`. It takes a YAML configuration file and an output path, like so:

```sh
mergekit-yaml path/to/your/config.yml ./output-model-directory [--cuda] [--lazy-unpickle] [--allow-crimes] [... other options]
```

This will run the merge and write your merged model to `./output-model-directory`.

For more information on the arguments accepted by `mergekit-yaml` run the command `mergekit-yaml --help`.

### Uploading to Huggingface

When you have a merged model you're happy with, you may want to share it on the Hugging Face Hub. `mergekit` generates a `README.md` for your merge with some basic information for a model card. You can edit it to include more details about your merge, like giving it a good name or explaining what it's good at; rewrite it entirely; or use the generated `README.md` as-is. It is also possible to edit your `README.md` online once it has been uploaded to the Hub.

Once you're happy with your model card and merged model, you can upload it to the Hugging Face Hub using the [huggingface_hub](https://huggingface.co/docs/huggingface_hub/index) Python library.

```sh
# log in to huggingface with an access token (must have write permission)
huggingface-cli login
# upload your model
huggingface-cli upload your_hf_username/my-cool-model ./output-model-directory .
```

The [documentation](https://huggingface.co/docs/huggingface_hub/guides/cli#huggingface-cli-upload) for `huggingface_hub` goes into more detail about other options for uploading.

## Merge Configuration

Merge configurations are YAML documents specifying the operations to perform in order to produce your merged model.
Below are the primary elements of a configuration file:

- `merge_method`: Specifies the method to use for merging models. See [Merge Methods](#merge-methods) for a list.
- `slices`: Defines slices of layers from different models to be used. This field is mutually exclusive with `models`.
- `models`: Defines entire models to be used for merging. This field is mutually exclusive with `slices`.
- `base_model`: Specifies the base model used in some merging methods.
- `parameters`: Holds various parameters such as weights and densities, which can also be specified at different levels of the configuration.
- `dtype`: Specifies the data type used for the merging operation.
- `tokenizer` or `tokenizer_source`: Determines how to construct a tokenizer for the merged model.
- `chat_template`: Specifies a chat template for the merged model.

### Parameter Specification

Parameters are flexible and can be set with varying precedence. They can be specified conditionally using tensor name filters, which allows finer control such as differentiating between attention heads and fully connected layers.

Parameters can be specified as:

- **Scalars**: Single floating-point values.
- **Gradients**: List of floating-point values, specifying an interpolated gradient.

The parameters can be set at different levels, with decreasing precedence as follows:

1. `slices.*.sources.parameters` - applying to a specific input slice
2. `slices.*.parameters` - applying to a specific output slice
3. `models.*.parameters` or `input_model_parameters` - applying to any tensors coming from specific input models
4. `parameters` - catchall

### Tokenizer Configuration

The tokenizer behavior can be configured in two ways: using the new `tokenizer` field (recommended) or the legacy `tokenizer_source` field (maintained for backward compatibility). These fields are mutually exclusive - you should use one or the other, not both.

#### Modern Configuration (tokenizer)

The `tokenizer` field provides fine-grained control over vocabulary and embeddings:

```yaml
tokenizer:
  source: "union"  # or "base" or a specific model path
  tokens:          # Optional: configure specific tokens
    <token_name>:
      source: ...  # Specify embedding source
      force: false # Optional: force this embedding for all models
  pad_to_multiple_of: null  # Optional: pad vocabulary size
```

##### Tokenizer Source

The `source` field determines the vocabulary of the output model:

- `union`: Combine vocabularies from all input models (default)
- `base`: Use vocabulary from the base model
- `"path/to/model"`: Use vocabulary from a specific model

##### Token Embedding Handling

When a tokenizer is configured, each input model's embedding matrix is adjusted to match the output vocabulary before being passed to the merge method. For tokens a model already has, its own embedding is used. For tokens a model is *missing*, a fallback embedding is assigned using these rules:

- If the base model has the token, use the base model's embedding
- If only one model has the token, use that model's embedding
- Otherwise, use an average of all available embeddings

The merge method then combines these per-model embeddings (original and filled-in) to produce the final output. This means the final embedding for a token present in multiple models is determined by your merge method (SLERP, linear, TIES, etc.), not simply taken from one model.

You can override these defaults for specific tokens. Any tokens listed here that don't already exist in the output vocabulary will be added automatically, making this useful for introducing new special tokens.

```yaml
tokenizer:
  source: union
  tokens:
    # Use embedding from a specific model
    <|im_start|>:
      source: "path/to/chatml/model"

    # Force a specific embedding for all models
    <|special|>:
      source: "path/to/model"
      force: true

    # Map a token to another model's token embedding
    <|renamed_token|>:
      source:
        kind: "model_token"
        model: "path/to/model"
        token: "<|original_token|>"  # or use token_id: 1234

    # Use a zero embedding
    <|unused|>:
      source:
        kind: "zero"
```

##### Practical Example

Here's how you might preserve both Llama 3 Instruct and ChatML prompt formats when merging models:

```yaml
tokenizer:
  source: union
  tokens:
    # ChatML tokens
    <|im_start|>:
      source: "chatml_model"
    <|im_end|>:
      source: "chatml_model"

    # Llama 3 tokens - force original embeddings
    <|start_header_id|>:
      source: "llama3_model"
      force: true
    <|end_header_id|>:
      source: "llama3_model"
      force: true
    <|eot_id|>:
      source: "llama3_model"
      force: true
```

#### Legacy Configuration (tokenizer_source)

For backward compatibility, the `tokenizer_source` field is still supported:

```yaml
tokenizer_source: "union"  # or "base" or a model path
```

This provides basic tokenizer selection but lacks the fine-grained control of the modern `tokenizer` field.

### Chat Template Configuration

The optional `chat_template` field allows overriding the chat template used for the merged model.

```yaml
chat_template: "auto"  # or a template name or Jinja2 template
```

Options include:

- `"auto"`: Automatically select the most common template among input models
- Built-in templates: `"alpaca"`, `"chatml"`, `"llama3"`, `"mistral"`, `"exaone"`
- A Jinja2 template string for custom formatting

### Examples

Several examples of merge configurations are available in [`examples/`](examples/).

## Merge Methods

`mergekit` offers many methods for merging models, each with its own strengths and weaknesses. Choosing the right method depends on your specific goals, the relationship between the models you're merging, and the desired characteristics of the final model.

For detailed explanations, parameter descriptions, and use cases for each method, please see our [**Merge Method Guide**](docs/merge_methods.md).

### Method Overview

| Method (`value`)                                                                                                      | Core Idea                                                            | # Models | Base Model | Key Strengths / Use Cases                                       |
|:----------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------|:--------:|:----:|:---------------------------------------------------------------|
| [**Linear** (`linear`)](docs/merge_methods.md#linear-linear)                                                          | Simple weighted average of model parameters.                         |    ≥2    |  -   | Averaging similar checkpoints, model soups.                     |
| [**SLERP** (`slerp`)](docs/merge_methods.md#slerp-slerp)                                                              | Spherical linear interpolation between two models.                   |     2    |  ✓   | Smoothly transitioning between two models.                      |
| [**NuSLERP** (`nuslerp`)](docs/merge_methods.md#nuslerp-nuslerp)                                                        | Enhanced SLERP with flexible weighting.                              |     2    |  *   | More intuitive SLERP; task vector SLERP.                        |
| [**Multi-SLERP** (`multislerp`)](docs/merge_methods.md#multi-slerp-multislerp)                                          | Barycentric SLERP for multiple models.                               |    ≥2    |  *   | Spherical interpolation for >2 models.                          |
| [**Karcher Mean** (`karcher`)](docs/merge_methods.md#karcher-mean-karcher)                                              | Riemannian barycenter of model parameters.                           |    ≥2    |  -   | Geometrically sound averaging on manifolds.                     |
| [**Task Arithmetic** (`task_arithmetic`)](docs/merge_methods.md#task-arithmetic-task_arithmetic)                      | Linearly combine "task vectors" (differences from a base).           |    ≥2    |  ✓   | Transferring/combining fine-tuned skills.                       |
| [**TIES** (`ties`)](docs/merge_methods.md#ties-merging-ties)                                                          | Task arithmetic + sparsification & sign consensus.                   |    ≥2    |  ✓   | Merging many models, reducing interference.                     |
| [**DARE** (`dare_linear`, `dare_ties`)](docs/merge_methods.md#dare-dare_linear-dare_ties)                               | Task arithmetic + random pruning & rescaling.                        |    ≥2    |  ✓   | Robust skill retention, similar to TIES.                        |
| [**DELLA** (`della`, `della_linear`)](docs/merge_methods.md#della-della-della_linear)                                   | Task arithmetic + adaptive magnitude-based pruning.                  |    ≥2    |  ✓   | Prioritizing important changes, reducing interference.          |
| [**Model Breadcrumbs** (`breadcrumbs`, `breadcrumbs_ties`)](docs/merge_methods.md#model-breadcrumbs-breadcrumbs_ties)   | Task arithmetic + outlier removal (small & large diffs).             |    ≥2    |  ✓   | Refining task vectors by removing extreme changes.              |
| [**SCE** (`sce`)](docs/merge_methods.md#sce-sce)                                                                      | Task arithmetic + adaptive matrix-level weighting based on variance. |    ≥2    |  ✓   | Dynamically weighting models based on parameter variance.       |
| [**Model Stock** (`model_stock`)](docs/merge_methods.md#model-stock-model_stock)                                        | Geometric weight calculation for linear interpolation.               |    ≥3    |  ✓   | Finding good linear interpolation weights for many checkpoints. |
| [**Nearswap** (`nearswap`)](docs/merge_methods.md#nearswap-nearswap)                                                    | Interpolate where parameters are similar.                            |     2    |  ✓   | Selective merging based on parameter similarity.                |
| [**Arcee Fusion** (`arcee_fusion`)](docs/merge_methods.md#arcee-fusion-arcee_fusion)                                    | Dynamic thresholding for fusing important changes.                   |     2    |  ✓   | Identifying and merging salient features.                       |
| [**Passthrough** (`passthrough`)](docs/merge_methods.md#passthrough-passthrough)                                        | Directly copies tensors from a single input model.                      |     1    |  -   | Frankenmerging, layer stacking, model surgery.                  |

**Key for `Base Model` Column:**

- ✓: **Required** - One of the input models *must* be designated as the `base_model`.
- *: **Optional** - One of the input models *can* be designated as the `base_model`.
- -: **Not Applicable** - `base_model` has no effect on this method.

## LoRA Extraction

Mergekit allows extracting PEFT-compatible low-rank approximations of finetuned models.

### Usage

```sh
mergekit-extract-lora --model finetuned_model_id_or_path --base-model base_model_id_or_path --out-path output_path [--no-lazy-unpickle] [--cuda] [--max-rank=desired_rank] [--sv-epsilon=tol]
```

## Mixture of Experts Merging

The `mergekit-moe` script supports merging multiple dense models into a mixture of experts, either for direct use or for further training. For more details see the [`mergekit-moe` documentation](docs/moe.md).

## Evolutionary Merge Methods

See [`docs/evolve.md`](docs/evolve.md) for details.

## Multi-Stage Merging (`mergekit-multi`)

`mergekit-multi` enables the execution of complex, multi-stage model merging workflows. You can define multiple merge configurations in a single YAML file, where later merges can use the outputs of earlier ones as inputs. This is useful for building up sophisticated models through a series of targeted merges.

See the [`mergekit-multi` documentation](docs/multimerge.md) for usage details and examples.

## Raw PyTorch Model Merging (`mergekit-pytorch`)

For merging arbitrary PyTorch models (not necessarily Hugging Face Transformers), `mergekit-pytorch` provides a way to apply mergekit's algorithms directly to `.pt` or `.safetensors` checkpoints. The configuration is similar to the YAML format used in `mergekit-yaml`, but does not support layer slicing or tokenizer configuration.

### Usage

```sh
mergekit-pytorch path/to/your/raw_config.yml ./output_pytorch_model_directory [options]
```

Use `mergekit-pytorch --help` for detailed options.

## Tokenizer Transplantation (`mergekit-tokensurgeon`)

`mergekit-tokensurgeon` is a specialized tool for transplanting tokenizers between models, allowing you to align the vocabulary of one model with another. This is particularly useful for cheaply producing draft models for speculative decoding or for cross-tokenizer knowledge distillation. See the [documentation](docs/tokensurgeon.md) for more details and how to use it.

## Citation

If you find `mergekit` useful in your research, please consider citing the [paper](https://aclanthology.org/2024.emnlp-industry.36/):

```bibtex
@inproceedings{goddard-etal-2024-arcees,
    title = "Arcee{'}s {M}erge{K}it: A Toolkit for Merging Large Language Models",
    author = "Goddard, Charles  and
      Siriwardhana, Shamane  and
      Ehghaghi, Malikeh  and
      Meyers, Luke  and
      Karpukhin, Vladimir  and
      Benedict, Brian  and
      McQuade, Mark  and
      Solawetz, Jacob",
    editor = "Dernoncourt, Franck  and
      Preo{\c{t}}iuc-Pietro, Daniel  and
      Shimorina, Anastasia",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track",
    month = nov,
    year = "2024",
    address = "Miami, Florida, US",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-industry.36",
    doi = "10.18653/v1/2024.emnlp-industry.36",
    pages = "477--485",
    abstract = "The rapid growth of open-source language models provides the opportunity to merge model checkpoints, combining their parameters to improve performance and versatility. Advances in transfer learning have led to numerous task-specific models, which model merging can integrate into powerful multitask models without additional training. MergeKit is an open-source library designed to support this process with an efficient and extensible framework suitable for any hardware. It has facilitated the merging of thousands of models, contributing to some of the world{'}s most powerful open-source model checkpoints. The library is accessible at: https://github.com/arcee-ai/mergekit.",
}
```
