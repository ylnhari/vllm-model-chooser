# vLLM Model Chooser

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen?style=flat-square)](https://ylnhari.github.io/vllm-model-chooser)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)
[![vLLM](https://img.shields.io/badge/vLLM-compatible-6366f1?style=flat-square)](https://docs.vllm.ai)
[![Models](https://img.shields.io/badge/models-115-success?style=flat-square)](index.html)
[![GitHub Pages](https://img.shields.io/badge/hosted-GitHub%20Pages-222?style=flat-square)](https://ylnhari.github.io/vllm-model-chooser)

Interactive web tool to find the optimal vLLM-compatible LLM for any GPU setup. Select your GPU, filter by context length and quantization, and instantly see which models fit.

[vLLM Recipes](https://recipes.vllm.ai)                ,                [vLLM  Model Chooser Tool Web Page](https://ylnhari.github.io/vllm-model-chooser)

## Features

- **6 GPU types**: L4, A100, H100, H200, B100, B200 — with 1–8× multi-GPU
- **VRAM + quantization dual check**: Models are filtered by BOTH memory budget AND quantization format compatibility, tri-state native/software/unsupported (NVFP4 → Blackwell only; MXFP4 → software on A100/Hopper, native on Blackwell)
- **vLLM-faithful memory budget**: `budget = physical × gpu-memory-utilization` is the ceiling for the whole vLLM instance, and `KV pool ≤ budget − weights` — the same accounting vLLM does. The utilization is a **user control** (default 0.92, vLLM's own), not a buried constant. The activation peak vLLM measures at startup is deliberately **not** modelled, so the KV pool is shown as an upper bound.
- **"Tight" fit verdict**: a fit that leaves under **8 GB per GPU** of headroom (budget − weights − KV) is shown as an amber ✓ instead of green. The 8 GB band is a heuristic warning sized from real vLLM startup logs (per-GPU activation peaks of about 0.1–18 GiB), never a subtracted quantity — a tight fit is still a fit.
- **Real KV-cache geometry**: computed per model from its HuggingFace `config.json` — including MLA (DeepSeek) and **sliding-window attention**, where only some layers cache the full context and the rest are capped at the window. Selectable FP16/FP8 KV dtype.
- **Worst-case concurrency**: how many full-context requests fit in the KV pool left after weights
- **Shareable URLs**: filter state (including util / KV dtype) is reflected into query params
- **115 models**: params, weight-only VRAM, context length, quantization variants, KV geometry
- **GPU compatibility info modal**: Native HW vs vLLM SW support per format, GPU specs + memory-budget table, quantization compatibility matrix
- **vLLM Recipe & HuggingFace links** per model

### What this tool deliberately does *not* show

**Benchmark scores.** They were removed, not hidden. Only **2 of 115** models expose structured eval metrics on HuggingFace (`cardData.model-index`), and even those aren't comparable (`pass@1`/`acc`/`exact_match` on unnamed tasks); the Open LLM Leaderboard is archived and doesn't cover these models. The previous MMLU/HumanEval numbers were hand-curated with **no verifiable source** — and were rendered as confident bars *and* used as a sort key. Unsourced scores driving a model-selection decision are worse than no scores. A test now guards against them creeping back in.

## Quick Start

```bash
python3 -m http.server 5000
open http://localhost:5000
```

Or use the provided script:

```bash
./start-server.sh
```

## Project Structure

```
vllm-model-chooser/
├── index.html       # Markup + styles; loads data.js then app.js
├── data.js          # GPU_CONFIG, GPU_QUANT_COMPAT, MODELS_DATA (auto-generated)
├── app.js           # Application logic + rendering
├── scripts/
│   ├── sync-data.mjs  # Rebuild data.js from live vLLM recipes (npm run factcheck's sibling)
│   └── factcheck.mjs  # Audit data.js against live recipes (npm run factcheck)
├── tests/           # Node built-in test runner (npm test)
├── AGENTS.md        # Maintenance guide for AI agents
├── README.md        # This file
└── start-server.sh  # Convenience dev server script
```

## Development

```bash
npm test           # logic + data-integrity tests (no dependencies)
npm run factcheck  # audit model data against live vLLM recipes
node scripts/sync-data.mjs   # regenerate data.js from recipes
```

## GPU Quantization Compatibility

Tri-state: **Native** = hardware tensor cores · **vLLM SW** = software path (loads, no speedup) · ✗ = unsupported.

| Format | L4 (Ada) | A100 (Ampere) | H100/H200 (Hopper) | B100/B200 (Blackwell) |
|--------|----------|---------------|--------------------|---------------------|
| BF16 | Native ✓ | Native ✓ | Native ✓ | Native ✓ |
| FP8 | Native ✓ | vLLM SW ✓ | Native ✓ | Native ✓ |
| INT8 (W8A8) | Native ✓ | Native ✓ | Native ✓ | Native ✓ |
| INT4/AWQ/GPTQ | vLLM SW ✓ | vLLM SW ✓ | vLLM SW ✓ | vLLM SW ✓ |
| NVFP4 | ✗ | ✗ | ✗ | Native ✓ |
| MXFP4 | vLLM SW ✓ | vLLM SW ✓ | vLLM SW ✓ | Native ✓ |
| MXFP8 | ✗ | ✗ | ✗ | Native ✓ |

## GPU Specs & the memory budget

`GPU_CONFIG` carries **physical** VRAM only. The usable figure is derived at runtime, mirroring vLLM:

```
budget          = physical × gpu-memory-utilization     # default 0.92 — vLLM's own default
KV cache pool   ≤ budget − weights                      # vLLM fills this greedily; it caps concurrency
```

`--gpu-memory-utilization` is the ceiling for the **whole** vLLM instance — weights, activations, CUDA graphs and KV cache all live inside it (`vllm/v1/worker/gpu_worker.py`, `determine_available_memory`). The utilization is exposed in the filter bar. What vLLM actually subtracts before sizing the KV cache is the activation/CUDA-graph peak it **measures** at startup by profiling a dummy forward pass; that figure scales with `--max-num-batched-tokens` and the model's hidden size (real logs run from under 0.1 GiB for a 0.5B model to over 30 GiB for a large model at a large batch), and nothing in this app's data can source it. It is therefore **not modelled**: every KV pool is shown as an upper bound (`≤`), and a fit with only a few GB to spare may not survive the real profile. An earlier version exposed a flat per-GPU "activation reserve" control; it was removed because a number with no source violates the rule that every displayed figure traces to one.

Instead, the app flags a **tight** fit: when `budget − weights − KV` is under `8 GB × GPUs`, the ✓ turns amber, the card's memory bar reads "(tight)", and the model modal shows a "Headroom for activations / CUDA graphs" row. The 8 GB/GPU band (`TIGHT_HEADROOM_GB_PER_GPU` in `app.js`) is a **heuristic warning band, not a measurement and not a subtraction** — a tight fit still counts as a fit everywhere. It is sized from real per-GPU "PyTorch activation peak memory" figures in vLLM startup logs:

| Model / setup | Activation peak per GPU | Source |
|---|---|---|
| Gemma-3-27b-it, 4× A100 40GB, vLLM 0.9.1, `max_model_len` 131072 | 17.91 GiB (1.41 GiB at `max_model_len` 4096) | [HF discussion](https://huggingface.co/google/gemma-3-27b-it/discussions/75) |
| DeepSeek-R1, L40S, TP8 PP4, `max_model_len` 16384, vLLM 0.7.3 | 1.52 GiB, then OOM during CUDA-graph capture at util 0.98 | [vllm#15598](https://github.com/vllm-project/vllm/issues/15598) |
| Qwen3-0.6B-FP8, RTX 5090, `max_model_len` 40960 | 0.52 GiB torch peak increase | [NVIDIA forum](https://forums.developer.nvidia.com/t/with-the-same-model-and-vllm-image-gb10-uses-more-vram-than-x86-gpu/353523) |
| Qwen2.5-0.5B-Instruct-AWQ, `max_num_batched_tokens` 2048, `max_num_seqs` 1 | 0.09 GiB | [vllm#16141](https://github.com/vllm-project/vllm/issues/16141) |

At vLLM defaults, mid-size models land in the 1–18 GiB range; 8 GB flags the danger zone without claiming precision. If a fit is tight, lower `--max-num-batched-tokens` / `--max-model-len`, pick a smaller quantization, or add a GPU.

| GPU | Physical | Architecture | Memory |
|-----|----------|--------------|--------|
| L4 24GB | 24 GB | Ada Lovelace (sm_89) | GDDR6 |
| A100 80GB | 80 GB | Ampere (sm_80) | HBM2e |
| H100 80GB | 80 GB | Hopper (sm_90) | HBM3 |
| H200 141GB | 141 GB | Hopper (sm_90) | HBM3e |
| B100 192GB | 192 GB | Blackwell (sm_100) | HBM3e |
| B200 192GB | 192 GB | Blackwell (sm_100) | HBM3e |

## KV cache

```
KV(tokens) = kvBytesPerToken × tokens
           + kvSlidingBytesPerToken × min(tokens, kvWindow)
```

Sliding-window layers **cannot** be folded into a single bytes-per-token constant: their cache is capped at the window no matter how long the context grows. Gemma-3/4 and Step-3.7 are mostly sliding layers — counting every layer as full attention overstates `gemma-4-31B` at 128K as **128.8 GB when the real figure is 22.3 GB (5.8×)**, wrongly excluding it from GPUs it fits.

But the inverse trap is worse. **A declared `sliding_window` does not mean the cache is capped**, and reading it naively *under*-counts — which makes a model claim to fit on GPUs it doesn't:

- **Qwen2/2.5** declare `sliding_window` but disable it via `use_sliding_window: false` → plain full attention.
- **DeepSeek-V4** pairs it with **sparse attention** (`index_topk`): sparsity changes which tokens you *attend to*, not which you *store*. The full MLA cache is kept. Reading the window as a cap reported **0.016 GB of KV for a 284B model at 1M context** — the real figure is ~52 GB.
- **MiMo-V2** uses `hybrid_layer_pattern` (`0` = full, `1` = SWA with separate `swa_*` head geometry).

A test now fails on any model whose KV comes out implausibly small for its own advertised context.

**When is 0 GB legitimate?** Only for models that never autoregressively decode — diffusion pipelines (image/video/audio) denoise a whole latent iteratively, so there are no previous tokens to cache. Embedding/rerankers are **not** zero: vLLM still fills a transient KV cache during prefill.

Geometry is read from each model's HF `config.json` by the generator. `kvSource` records provenance, and the UI marks anything that isn't a first-hand read:

| Marker | `kvSource` | Meaning |
|---|---|---|
| *(none)* | `config` | read from the model's own `config.json` |
| `°` | `mirror` | read from an ungated same-architecture mirror (original repo is gated) |
| `†` | `estimate` | hand-sourced geometry — no config available anywhere |
| `‡` | — | no geometry at all; coarse `params × tokens` proxy |
| — | `none` | single-pass diffusion/generative model: **no KV cache at any context** |

## Data Sources (provenance — read before updating any data)

All factual data is derived from the sources below. **Trust these sources, in this order — not model knowledge, blog posts, or general web search.** (A "Marlin FP4 fallback" web claim once led to NVFP4 being wrongly marked as Hopper-compatible; the recipes are explicit that NVFP4 is Blackwell-only. When in doubt, the recipe wins.)

| What you're updating | Authoritative source | Field / how to read it |
|----------------------|----------------------|------------------------|
| **Model inventory** (which models exist) | [`recipes.vllm.ai/models.json`](https://recipes.vllm.ai/models.json) | `hf_id` is the **case-sensitive recipe path** (e.g. `Google/gemma-4-12B-it`). `json` is the per-model recipe URL. |
| **Model specs** (params, active params, context length, dense/MoE) | `recipes.vllm.ai/<hf_id>.json` → `.model` | `parameter_count`, `active_parameters`, `context_length`, `architecture`. Cross-check context with the HuggingFace `config.json` (`max_position_embeddings`). |
| **Variants & precisions** | same recipe → `.variants.*` | `precision`, `vram_minimum_gb` (**KV-INCLUSIVE — do not copy into our weight-only `vram`**), and `description` (carries hardware hints like "for Blackwell GPUs", "fits on 1xA100"). |
| **Quantization × GPU compatibility** (`GPU_QUANT_COMPAT`) | [vLLM quantization "supported hardware" docs](https://docs.vllm.ai/en/latest/features/quantization/supported_hardware/) | Primary matrix of method × architecture. Corroborate per-model with the recipe variant `description` and `hardware_overrides` keys (`blackwell` / `hopper` / `amd`). |
| **GPU hardware specs** (`GPU_CONFIG`) | NVIDIA datasheets | **Physical** VRAM, architecture, sm version, memory type. No `usableVram` — that's derived at runtime from the utilization knob, so there's no second value to drift. The recipe `recommended_command.hardware_profile.description` gives authoritative blurbs (e.g. "NVIDIA H200 SXM 141 GB HBM3e"). |
| **Memory budget** (util) | [vLLM engine args](https://docs.vllm.ai/en/latest/configuration/engine_args.html), `vllm/config/cache.py` (default 0.92) | `budget = physical × --gpu-memory-utilization` for the whole instance; KV is allocated greedily into what's left after weights + the activation peak vLLM *measures* at startup. The utilization is a runtime user control; the measured peak is not modelled, so the KV pool is an upper bound. |
| **Weight-only VRAM** (our `vram` field) | computed, not copied | `totalParams × bytesPerParam` — BF16 ×2, FP8/MXFP8/INT8 ×1, INT4/NVFP4/MXFP4/W4A16 ×0.5. Recipe `vram_minimum_gb` includes KV cache, so don't use it directly. Mixed-precision (MXFP4/`FP4+FP8`/`AMD-FP8`) stays curated — the naive formula understates it. Real checkpoints run a few % larger (quant scales, unquantized embeddings/lm_head). |
| **KV-cache geometry** (`kvBytesPerToken`, `kvSlidingBytesPerToken`, `kvWindow`, `kvSource`) | HuggingFace `config.json` | Per attention layer: `2 × num_kv_heads × head_dim × 2B` (GQA/MHA) or `(kv_lora_rank + qk_rope_head_dim) × 2B` (MLA — a single compressed latent, **not** ×heads, **not** ×2). Layers are then split into full-attention vs sliding-window (`layer_types`, or `sliding_window` + `sliding_window_pattern`); Mamba/linear layers cache nothing and are excluded. See `kvBytesPerTokenFromConfig` in the generator. |
| **Benchmarks** | ❌ **none — deliberately removed** | Only 2/115 models expose structured eval metrics on HF, and they aren't comparable. There is no authoritative source, so the tool shows no scores rather than unsourced ones. Do not re-add hand-curated numbers; a test blocks it. |

### Keeping data correct
- `node scripts/sync-data.mjs` — rebuilds `data.js` from recipes (context length, variant precisions, `recipe_id` casing), preserving curated fields.
- `npm run factcheck` — audits `data.js` against live recipes and reports mismatches.
- `GPU_CONFIG` and `GPU_QUANT_COMPAT` are **hand-maintained inside `scripts/sync-data.mjs`** (they are not in any recipe) — edit them there, with a source citation in the comment, then re-run the generator.

## License

MIT
