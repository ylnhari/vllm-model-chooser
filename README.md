# vLLM Model Chooser

Interactive web tool to find the optimal vLLM-compatible LLM for any GPU setup. Select your GPU, filter by context length and quantization, and instantly see which models fit.

**[Live Demo](https://ylnhari.github.io/vllm-model-chooser)** · **[vLLM Recipes](https://recipes.vllm.ai)**

## Features

- **6 GPU types**: L4, A100, H100, H200, B100, B200 — with 1/2/4/8× multi-GPU
- **VRAM + quantization dual check**: Models are filtered by BOTH weight VRAM AND quantization format compatibility (NVFP4 on A100? ✗. NVFP4 on B200? ✓)
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
modal-chooser/
├── index.html       # Single-file application (all HTML/CSS/JS)
├── AGENTS.md        # Maintenance guide for AI agents
├── README.md        # This file
└── start-server.sh  # Convenience dev server script
```

## GPU Quantization Compatibility

| Format | L4 (Ada) | A100 (Ampere) | H100/H200 (Hopper) | B100/B200 (Blackwell) |
|--------|----------|---------------|--------------------|---------------------|
| BF16 | Native ✓ | Native ✓ | Native ✓ | Native ✓ |
| FP8 | Native ✓ | vLLM SW ✓ | Native ✓ | Native ✓ |
| INT4/AWQ | vLLM SW ✓ | vLLM SW ✓ | vLLM SW ✓ | vLLM SW ✓ |
| NVFP4 | ✗ | ✗ | ✗ | Native ✓ |
| MXFP4 | ✗ | ✗ | vLLM SW ✓ | Native ✓ |
| MXFP8 | ✗ | ✗ | ✗ | vLLM SW ✓ |

## GPU Specs

| GPU | VRAM | Usable (95%) | Architecture | Memory |
|-----|------|-------------|--------------|--------|
| L4 24GB | 24 GB | 23 GB | Ada Lovelace (sm_89) | GDDR6 |
| A100 80GB | 80 GB | 76 GB | Ampere (sm_80) | HBM2 |
| H100 80GB | 80 GB | 76 GB | Hopper (sm_90) | HBM3 |
| H200 141GB | 141 GB | 134 GB | Hopper (sm_90) | HBM3e |
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
