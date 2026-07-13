# vLLM Model Chooser

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen?style=flat-square)](https://ylnhari.github.io/vllm-model-chooser)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)
[![vLLM](https://img.shields.io/badge/vLLM-compatible-6366f1?style=flat-square)](https://docs.vllm.ai)
[![Models](https://img.shields.io/badge/models-108-success?style=flat-square)](index.html)
[![GitHub Pages](https://img.shields.io/badge/hosted-GitHub%20Pages-222?style=flat-square)](https://ylnhari.github.io/vllm-model-chooser)

Interactive web tool to find the optimal vLLM-compatible LLM for any GPU setup. Select your GPU, filter by context length and quantization, and instantly see which models fit.

[vLLM Recipes](https://recipes.vllm.ai)                ,                [vLLM  Model Chooser Tool Web Page](https://ylnhari.github.io/vllm-model-chooser)

## Features

- **6 GPU types**: L4, A100, H100, H200, B100, B200 — with 1/2/4/8× multi-GPU
- **VRAM + quantization dual check**: Models are filtered by BOTH weight VRAM AND quantization format compatibility, tri-state native/software/unsupported (NVFP4 → Blackwell only; MXFP4 → software on A100/Hopper, native on Blackwell)
- **Optional KV-cache estimate**: factor an approximate context-length KV footprint into the fit check
- **Shareable URLs**: filter state is reflected into query params
- **108 models** with detailed specs: params, weight-only VRAM, context length, quantization variants, benchmarks
- **GPU compatibility info modal**: Native HW vs vLLM SW support per format, full GPU specs table, quantization compatibility matrix
- **vLLM Recipe & HuggingFace links** per model

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

## GPU Specs

Usable VRAM = `floor(physical × 0.95)`.

| GPU | VRAM | Usable (95%) | Architecture | Memory |
|-----|------|-------------|--------------|--------|
| L4 24GB | 24 GB | 22 GB | Ada Lovelace (sm_89) | GDDR6 |
| A100 80GB | 80 GB | 76 GB | Ampere (sm_80) | HBM2e |
| H100 80GB | 80 GB | 76 GB | Hopper (sm_90) | HBM3 |
| H200 141GB | 141 GB | 133 GB | Hopper (sm_90) | HBM3e |
| B100 192GB | 192 GB | 182 GB | Blackwell (sm_100) | HBM3e |
| B200 192GB | 192 GB | 182 GB | Blackwell (sm_100) | HBM3e |

## Data Sources (provenance — read before updating any data)

All factual data is derived from the sources below. **Trust these sources, in this order — not model knowledge, blog posts, or general web search.** (A "Marlin FP4 fallback" web claim once led to NVFP4 being wrongly marked as Hopper-compatible; the recipes are explicit that NVFP4 is Blackwell-only. When in doubt, the recipe wins.)

| What you're updating | Authoritative source | Field / how to read it |
|----------------------|----------------------|------------------------|
| **Model inventory** (which models exist) | [`recipes.vllm.ai/models.json`](https://recipes.vllm.ai/models.json) | `hf_id` is the **case-sensitive recipe path** (e.g. `Google/gemma-4-12B-it`). `json` is the per-model recipe URL. |
| **Model specs** (params, active params, context length, dense/MoE) | `recipes.vllm.ai/<hf_id>.json` → `.model` | `parameter_count`, `active_parameters`, `context_length`, `architecture`. Cross-check context with the HuggingFace `config.json` (`max_position_embeddings`). |
| **Variants & precisions** | same recipe → `.variants.*` | `precision`, `vram_minimum_gb` (**KV-INCLUSIVE — do not copy into our weight-only `vram`**), and `description` (carries hardware hints like "for Blackwell GPUs", "fits on 1xA100"). |
| **Quantization × GPU compatibility** (`GPU_QUANT_COMPAT`) | [vLLM quantization "supported hardware" docs](https://docs.vllm.ai/en/latest/features/quantization/supported_hardware/) | Primary matrix of method × architecture. Corroborate per-model with the recipe variant `description` and `hardware_overrides` keys (`blackwell` / `hopper` / `amd`). |
| **GPU hardware specs** (`GPU_CONFIG`) | NVIDIA datasheets | VRAM, architecture, sm version, memory type. The recipe `recommended_command.hardware_profile.description` gives authoritative blurbs (e.g. "NVIDIA H200 SXM 141 GB HBM3e"). |
| **Weight-only VRAM** (our `vram` field) | computed, not copied | `totalParams × bytesPerParam` — BF16 ×2, FP8/MXFP8/INT8 ×1, INT4/NVFP4/MXFP4/W4A16 ×0.5. Recipe `vram_minimum_gb` includes KV cache, so don't use it directly. |
| **Benchmarks** | curated (model cards / leaderboards) | Hand-maintained; preserved across syncs — the generator never overwrites them. |

### Keeping data correct
- `node scripts/sync-data.mjs` — rebuilds `data.js` from recipes (context length, variant precisions, `recipe_id` casing), preserving curated fields.
- `npm run factcheck` — audits `data.js` against live recipes and reports mismatches.
- `GPU_CONFIG` and `GPU_QUANT_COMPAT` are **hand-maintained inside `scripts/sync-data.mjs`** (they are not in any recipe) — edit them there, with a source citation in the comment, then re-run the generator.

## License

MIT
