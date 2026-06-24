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
- **VRAM + quantization dual check**: Models are filtered by BOTH weight VRAM AND quantization format compatibility, tri-state native/software/unsupported (MXFP4 on A100? ✗. MXFP4 on H100? software ✓. NVFP4 native only on Blackwell)
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
| INT8/INT4/AWQ/GPTQ | vLLM SW ✓ | vLLM SW ✓ | vLLM SW ✓ | vLLM SW ✓ |
| NVFP4 | ✗ | vLLM SW ✓ | vLLM SW ✓ | Native ✓ |
| MXFP4 | ✗ | ✗ | vLLM SW ✓ | Native ✓ |
| MXFP8 | ✗ | ✗ | ✗ | vLLM SW ✓ |

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

## Data Sources

| Source | URL |
|--------|-----|
| vLLM Recipes | https://recipes.vllm.ai/models.json |
| vLLM Quantization Docs | https://docs.vllm.ai/en/stable/features/quantization/ |
| HuggingFace Models | https://huggingface.co/models |

## License

MIT
