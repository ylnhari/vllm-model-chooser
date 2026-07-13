# Development Guide - vLLM Model Chooser

This file contains technical documentation for maintaining and extending the vLLM Model Chooser application.

## Application Overview

Static, no-build web application that helps users select vLLM-compatible LLM models based on GPU hardware, VRAM requirements, context length, and quantization options. Served by any static server (`python3 -m http.server`).

**File layout** (split as of the v1 refactor — sections further down may still say "single-file"; mentally substitute these files):
- `index.html` — markup + styles; loads `data.js` then `app.js` as plain globals (no modules/bundler).
- `data.js` — `GPU_CONFIG`, `GPU_QUANT_COMPAT`, `MODELS_DATA`. **Auto-generated** by `scripts/sync-data.mjs` from live vLLM recipes; curated fields (benchmark, weight-only vram, type, tested) are preserved across syncs. Don't hand-edit recipe-sourced fields (contextLength, variant precisions) — re-run the generator.
- `app.js` — all logic + rendering.
- `scripts/sync-data.mjs` (regenerate data) · `scripts/factcheck.mjs` (audit vs recipes — `npm run factcheck`).
- `tests/` — `npm test` (Node built-in runner, no deps). Run after any logic/data change.

**`GPU_QUANT_COMPAT` is now a tri-state map**, not an array: `{ CANONICAL: 'native' | 'sw' }` — present key = supported (`native` = HW tensor cores, `sw` = vLLM software path), absent = unsupported. `precSupportLevel(prec, gpuType)` returns `'native' | 'sw' | null`; `isPrecCompatible` is just `precSupportLevel(...) !== null`.

**Critical**: VRAM values in `MODELS_DATA` must be **weight-only** (model stored weights only). KV cache is NOT included in the `vram` field — `app.js` has an optional, clearly-labelled KV-cache *estimate* (`estKVCacheGB`, a coarse params×tokens proxy) the user can toggle into the fit check.

---

## Sources of Truth (READ THIS before changing any data)

**Trust these sources, in this priority order. Do NOT use model knowledge, blog posts, or general web search to set compatibility or specs.** A real regression happened this way: a web claim about a "Marlin FP4 fallback" led to NVFP4 being marked Hopper/Ampere-compatible, but every recipe says NVFP4 is *for Blackwell GPUs*. **When a source disagrees with your prior, the recipe/vLLM-docs win.** Verify, don't assume.

| Data | Authoritative source | How to read it |
|------|----------------------|----------------|
| Model inventory | `recipes.vllm.ai/models.json` | `hf_id` = **case-sensitive** recipe path (e.g. `Google/gemma-4-12B-it`, not `google/`). `json` = recipe URL. |
| Model specs (params, active, context, dense/MoE) | `recipes.vllm.ai/<hf_id>.json` → `.model` | `parameter_count`, `active_parameters`, `context_length`, `architecture`. Cross-check context with HuggingFace `config.json` (`max_position_embeddings`). |
| Variants & precisions | recipe `.variants.*` | `precision`, `vram_minimum_gb` (**KV-INCLUSIVE — never copy into our weight-only `vram`**), `description` (free-text hardware hints, e.g. "for Blackwell GPUs", "fits on 1xA100"). There is **no structured per-variant GPU list** — the hardware hints live in `description` + `hardware_overrides` keys (`blackwell`/`hopper`/`amd`). |
| Quant × GPU compatibility (`GPU_QUANT_COMPAT`) | [vLLM quantization "supported hardware" docs](https://docs.vllm.ai/en/latest/features/quantization/supported_hardware/) | The canonical method × architecture matrix. Corroborate per-model with recipe variant `description` + `hardware_overrides`. |
| GPU hardware specs (`GPU_CONFIG`) | NVIDIA datasheets | VRAM / arch / sm / memory. Recipe `recommended_command.hardware_profile.description` gives authoritative blurbs ("NVIDIA H200 SXM 141 GB HBM3e"). |
| Weight-only VRAM (our `vram`) | **computed** | `totalParams × bytesPerParam` (see formula below). Never copy the recipe's KV-inclusive `vram_minimum_gb`. |
| Benchmarks | curated | Hand-maintained; the generator preserves them. |

**Verified facts (re-confirm against sources if you touch them):**
- NVFP4 → **Blackwell native only** (DeepSeek-R1, Llama-3.3-70B, MiniMax-M2.7 recipes all say "for Blackwell GPUs").
- MXFP4 → native on Blackwell, **software elsewhere incl. A100** (gpt-oss-120b: "fits on 1xA100 80GB").
- MXFP8 → **native on Blackwell MX tensor cores only** (MiniMax-M3: "Blackwell ... for native MX tensor core").
- FP8 → native on Ada/Hopper/Blackwell, **software (Marlin) on Ampere**.
- INT8 (W8A8) → **native on all listed GPUs** (all are Ampere+/sm_80+ with int8 tensor cores; vLLM's CUTLASS int8 GEMM is the native path).
- INT4/AWQ/GPTQ (W4A16) → **software on all** (weights dequantised to FP16 for compute via Marlin — no int4 tensor-core math).

**Where these live & how to update:**
- `GPU_CONFIG` and `GPU_QUANT_COMPAT` are **hand-maintained inside `scripts/sync-data.mjs`** (no recipe carries them). Edit them there *with a source citation in the comment*, then run the generator.
- `node scripts/sync-data.mjs` rebuilds `data.js` (context, variant precisions, `recipe_id` casing) preserving curated fields. `npm run factcheck` audits the result. `npm test` validates logic. Run all three after any change.
- `recipe_id` is set per-model only when the recipe path casing differs from `hf_url` (HuggingFace link uses `hf_url`; recipe link uses `recipe_id || hf_url`).

---

## GPU Quantization Compatibility System

This is the most architecturally significant part of the application. Models are checked against BOTH available VRAM AND quantization format compatibility when determining if a model fits a GPU configuration.

### Data Flow

```
User selects GPU type + GPU count
                ↓
modelFitsGPU(model, gpus)
    ├── getGPUVRAM(gpus)          → total usable VRAM
    ├── model.vram <= vram        → weight-only VRAM check
    ├── isPrecCompatible(prec, gpuType) → quantization format check
    └── (variant loop)            → try each variant (VRAM + quant check)
                ↓
    Returns { fits: true/false, variant: {...} }
```

### Key Data Structures

```javascript
// ~Line 254: GPU hardware specs
const GPU_CONFIG = {
    'L4-24GB': { vram: 24, usableVram: 23, architecture: 'Ada Lovelace', name: 'L4 24GB' },
    'A100-80GB': { vram: 80, usableVram: 76, architecture: 'Ampere', name: 'A100 80GB' },
    'H100-80GB': { vram: 80, usableVram: 76, architecture: 'Hopper', name: 'H100 80GB' },
    'H200-141GB': { vram: 141, usableVram: 134, architecture: 'Hopper', name: 'H200 141GB' },
    'B100-192GB': { vram: 192, usableVram: 182, architecture: 'Blackwell', name: 'B100 192GB' },
    'B200-192GB': { vram: 192, usableVram: 182, architecture: 'Blackwell', name: 'B200 192GB' },
};

// ~Line 263: Quantization formats supported per GPU
const GPU_QUANT_COMPAT = {
    'L4-24GB': ['BF16', 'FP8', 'INT4', 'AWQ', 'GPTQ', 'W4A16'],
    'A100-80GB': ['BF16', 'FP8', 'INT4', 'AWQ', 'GPTQ', 'W4A16'],
    'H100-80GB': ['BF16', 'FP8', 'INT4', 'AWQ', 'GPTQ', 'W4A16', 'MXFP4'],
    'H200-141GB': ['BF16', 'FP8', 'INT4', 'AWQ', 'GPTQ', 'W4A16', 'MXFP4'],
    'B100-192GB': ['BF16', 'FP8', 'INT4', 'AWQ', 'GPTQ', 'W4A16', 'NVFP4', 'NVFP4-QAD', 'MXFP4', 'MXFP8'],
    'B200-192GB': ['BF16', 'FP8', 'INT4', 'AWQ', 'GPTQ', 'W4A16', 'NVFP4', 'NVFP4-QAD', 'MXFP4', 'MXFP8'],
};
```

### GPU Quantization Compatibility Reference

| Format    | L4 (Ada)       | A100 (Ampere)   | H100/H200 (Hopper) | B100/B200 (Blackwell) |
|-----------|----------------|-----------------|---------------------|----------------------|
| BF16      | Native HW ✓    | Native HW ✓     | Native HW ✓         | Native HW ✓          |
| FP8       | Native HW ✓    | vLLM SW ✓       | Native HW ✓         | Native HW ✓          |
| INT4/AWQ  | vLLM SW ✓      | vLLM SW ✓       | vLLM SW ✓           | vLLM SW ✓            |
| NVFP4     | ✗              | ✗               | ✗                   | Native HW ✓          |
| MXFP4     | ✗              | ✗               | vLLM SW ✓           | Native HW ✓          |
| MXFP8     | ✗              | ✗               | ✗                   | vLLM SW ✓            |

**Key rules:**
- `Native HW` = GPU has dedicated tensor cores for this format
- `vLLM SW` = Format is emulated in software (Marlin kernel, etc.) — no native hardware support
- `✗` = Format is completely unsupported on this architecture

### normalizePrec() — Format Normalization

Maps any precision string to a canonical format name. **Priority order matters** — more specific formats are checked before generic ones:

1. `NVFP4` — matches "NVFP4", "NVFP4-QAD"
2. `MXFP4` — matches "MXFP4" (checked before FP8 to avoid "MXFP4" matching "FP8" substring)
3. `MXFP8` — matches "MXFP8" (checked before FP8 to avoid "MXFP8" matching "FP8" substring)
4. `NVFP4` (FP4 fallback) — matches "FP4+FP8" (compound format from DeepSeek) or bare "FP4"
5. `FP8` — matches "FP8", "AMD-FP8"
6. `BF16` — exact match only
7. `INT4` — matches "INT4", "AWQ", "GPTQ", "W4A16", "QAT-W4A16"
8. `null` — unrecognized formats (e.g. "300B-A47B")

**Rule**: FP4 variants must be checked BEFORE FP8 because "FP4+FP8" also contains "FP8".

### isPrecCompatible(prec, gpuType) — Compatibility Gate

```javascript
function isPrecCompatible(prec, gpuType) {
    const formats = GPU_QUANT_COMPAT[gpuType];
    if (!formats) return true;        // unknown GPU → allow
    const norm = normalizePrec(prec);
    if (!norm) return true;            // unknown format → allow
    return formats.includes(norm);     // check against supported list
}
```

**Edge cases:**
- Unknown GPU type: returns `true` (allow, don't break)
- Unknown precision string: returns `true` (allow, don't break)
- null/undefined precision: returns `true` (allow)

### modelFitsGPU(model, gpus) — Feasibility Check

```javascript
function modelFitsGPU(model, gpus) {
    if (gpus === 0) return { fits: true, reason: "Any configuration" };
    const gpuType = document.getElementById('gpuTypeSelect')?.value || 'L4-24GB';
    const vram = getGPUVRAM(gpus);
    // Check 1: Base model precision fits VRAM AND is compatible
    if (model.vram <= vram && isPrecCompatible(model.prec, gpuType)) return { fits: true };
    // Check 2: Try each variant
    if (model.variants.length > 0) {
        for (const v of model.variants) {
            if (v.vram && v.vram <= vram && isPrecCompatible(v.prec || model.prec, gpuType))
                return { fits: true, variant: v };
        }
    }
    return { fits: false };
}
```

**Both VRAM AND quantization must pass** for a model to be marked as fitting.

### GPU_QUANT_COMPAT Maintenance Rules

#### Adding a new GPU
1. Add entry to `GPU_CONFIG` with VRAM specs
2. Add entry to `GPU_QUANT_COMPAT` with supported formats
3. Add `<option>` in HTML select element
4. Add GPU card in `openGPUInfoModal()` (Native HW / vLLM SW sections)
5. Add rows to GPU specifications table and quantization compatibility table in `openGPUInfoModal()`

#### Adding a new quantization format
1. Add canonical name to `normalizePrec()` priority chain
2. Add canonical name to `GPU_QUANT_COMPAT` entries for compatible GPUs
3. Add format row to quantization compatibility table in `openGPUInfoModal()`
4. Add badge CSS class if needed
5. Add entry in `getQuantBadgeClass()`

#### When vLLM adds software support for an existing format on a new GPU
1. Add the format's canonical name to that GPU's array in `GPU_QUANT_COMPAT`
2. Update the quantization compatibility table in `openGPUInfoModal()`

---

## Code Architecture

### Main Data Structures

```javascript
// GPU Configuration
const GPU_CONFIG = {
    'A100-80GB': { vram: 80, usableVram: 76, architecture: 'Ampere', name: 'A100 80GB' },
    // ... other GPUs (see section above)
};

// GPU Quantization Compatibility
const GPU_QUANT_COMPAT = {
    'A100-80GB': ['BF16', 'FP8', 'INT4', 'AWQ', 'GPTQ', 'W4A16'],
    // ... other GPUs (see section above)
};

// Model Data Array - one entry per model
const MODELS_DATA = [
    {
        id: 1,
        name: "Model-Name",
        provider: "ProviderName",
        params: "100B/10B",        // display string
        totalParams: 100,          // total parameters
        activeParams: 10,          // active parameters (for MoE)
        prec: "BF16",              // default precision
        vram: 200,                 // WEIGHT-ONLY VRAM in GB
        variants: [
            { prec: "FP8", vram: 100 },
            { prec: "NVFP4", vram: 50, note: "Blackwell only" }
        ],
        type: "text",              // text, vision, moe, embedding
        contextLength: 131072,     // context window (optional)
        benchmark: { mmlu: 85.2, humaneval: 72.4 },
        hf_url: "provider/model-name"
    }
];
```

### Critical Rules

1. **VRAM is Weight-Only Only**: Never include KV cache in `vram` field
2. **95% GPU Utilization**: `usableVram = Math.floor(vram * 0.95)`
3. **Single Function Definition**: Never duplicate `getGPUVRAM`, `modelFitsGPU`, or any core function
4. **Dual Check in modelFitsGPU**: Always check BOTH VRAM capacity AND quantization compatibility
5. **vLLM URL Format**: `https://recipes.vllm.ai/{Provider}/{Model}.json` (Provider capitalized)
6. **normalizePrec Priority**: Specific formats (NVFP4, MXFP4, MXFP8) before generic (FP8, FP4)

---

## Essential Functions

### VRAM Calculation

```javascript
// ~Line 293: Single source of truth for GPU VRAM
function getGPUVRAM(gpus) {
    const gpuType = document.getElementById('gpuTypeSelect')?.value || 'L4-24GB';
    return gpus * (GPU_CONFIG[gpuType]?.usableVram || 72);  // 72 is fallback for unknown GPU
}
```

### Quantization Format Detection (Badge Styling)

```javascript
// ~Line 515: Used for badge CSS class only (NOT for compatibility logic)
function getQuantBadgeClass(prec) {
    if (prec.includes('BF16')) return 'badge-bf16';
    if (prec.includes('FP8')) return 'badge-fp8';
    if (prec.includes('INT4') || prec.includes('GPTQ') || prec.includes('AWQ')) return 'badge-int4';
    if (prec.includes('NVFP4') || prec.includes('FP4')) return 'badge-nvfp4';
    if (prec.includes('MXFP4')) return 'badge-mxfp4';
    if (prec.includes('QAT')) return 'badge-int4';
    return 'badge-bf16';
}
```

### Bytes Per Parameter by Precision

| Precision | Bytes/Param |
|-----------|-------------|
| BF16      | 2           |
| FP8       | 1           |
| INT4/AWQ/GPTQ | 0.5     |
| NVFP4/FP4 | 0.5         |
| MXFP4     | 0.5         |
| MXFP8     | 1           |
| W4A16     | 0.5         |

---

## Common Bugs and How to Fix Them

### Bug: Model Shows as Fitting When It Shouldn't

**Cause 1**: Duplicate `getGPUVRAM` function with hardcoded value.

**Fix**: Search for `function getGPUVRAM` — there should be exactly ONE. Remove duplicates.

**Cause 2**: `modelFitsGPU` only checks VRAM, not quantization compatibility.

**Fix**: Ensure `modelFitsGPU` calls `isPrecCompatible()` for both base precision and each variant:
```javascript
// CORRECT
if (model.vram <= vram && isPrecCompatible(model.prec, gpuType)) return { fits: true };
if (v.vram && v.vram <= vram && isPrecCompatible(v.prec || model.prec, gpuType)) return { fits: true, variant: v };
```

### Bug: Wrong GPU VRAM Being Used

**Cause**: Hardcoded fallback like `|| 72` or `* 72`.

**Fix**: Ensure `getGPUVRAM` reads from `GPU_CONFIG` and uses a fallback for unknown GPUs:
```javascript
// CORRECT
return gpus * (GPU_CONFIG[gpuType]?.usableVram || 72);
```

### Bug: VRAM Values Don't Match vLLM Recipes

**Cause**: VRAM includes KV cache or uses wrong precision multiplier.

**Fix**: Calculate weight-only VRAM:
- BF16: `totalParams * 2`
- FP8: `totalParams * 1`
- INT4/AWQ/GPTQ/NVFP4/MXFP4/W4A16: `totalParams * 0.5`
- MXFP8: `totalParams * 1`

### Bug: normalizePrec Returns Wrong Format

**Cause**: Priority order is wrong — a generic check (e.g. `FP8`) matches before a specific one (e.g. `MXFP8` or `NVFP4`).

**Fix**: Always place more specific formats FIRST in the if-else chain:
```javascript
// Specific formats first
if (upper.includes('NVFP4')) return 'NVFP4';
if (upper.includes('MXFP4')) return 'MXFP4';
if (upper.includes('MXFP8')) return 'MXFP8';
// Compound format (DeepSeek FP4+FP8 → treated as NVFP4)
if (upper === 'FP4+FP8' || (upper.includes('FP4') && !upper.includes('FP8'))) return 'NVFP4';
// Then generic formats
if (upper.includes('FP8')) return 'FP8';
```

### Bug: MXFP8 Shows as FP8

**Cause**: "MXFP8".includes("FP8") is true → caught by FP8 check before MXFP8 check.

**Fix**: MXFP8 must be checked before FP8 in `normalizePrec()`.

### Bug: modelFitsGPU VRAM Not Updating When GPU Type Changes

**Cause**: `modelFitsGPU` doesn't read from DOM, caches old value.

**Fix**: `modelFitsGPU` reads `document.getElementById('gpuTypeSelect')?.value` fresh each call.

---

## Data Update Workflow

### Step 1: Check for New Models

```bash
# Get list of all vLLM recipes
curl https://recipes.vllm.ai/models.json | jq '.[] | .hf_id'

# Compare with current MODELS_DATA hf_url values
```

### Step 2: Fetch Model Recipe

```bash
curl https://recipes.vllm.ai/{Provider}/{Model}.json | jq '{
  name: .meta.title,
  provider: .meta.provider,
  param_count: .model.parameter_count,
  active_params: .model.active_parameters,
  context_length: .model.context_length,
  default_vram: .variants.default.vram_minimum_gb,
  variants: .variants
}'
```

### Step 3: Calculate Weight-Only VRAM

Use the formula directly (vLLM recipe `vram_minimum_gb` includes KV cache, so don't rely on it for weight-only):
- BF16: `totalParams * 2`
- FP8: `totalParams * 1`
- INT4/AWQ/GPTQ/NVFP4/MXFP4/W4A16: `totalParams * 0.5`
- MXFP8: `totalParams * 1`

### Step 4: Verify HuggingFace Model ID

Check that `hf_url` matches the actual HuggingFace model ID:
```
https://huggingface.co/{hf_url}
```

### Step 5: Validate JavaScript

```bash
node -e "
const fs = require('fs');
const html = fs.readFileSync('index.html', 'utf8');
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];
const script = scripts[scripts.length-1][1];
try {
    new Function(script);
    console.log('JavaScript syntax: OK');
} catch(e) {
    console.log('ERROR:', e.message);
}
"
```

---

## GPU Types Reference

When adding a new GPU, update ALL of:
1. `GPU_CONFIG` object
2. `GPU_QUANT_COMPAT` object  
3. HTML `<select>` element
4. GPU compatibility card (Native HW / vLLM SW sections in `openGPUInfoModal()`)
5. GPU specifications table in `openGPUInfoModal()`
6. Quantization compatibility table in `openGPUInfoModal()`

```javascript
// JavaScript format
'GPU-NAME': { vram: XXX, usableVram: Math.floor(XXX * 0.95), architecture: 'Name', name: 'Display Name' }

// Also add to GPU_QUANT_COMPAT
'GPU-NAME': ['BF16', 'FP8', /* ... supported formats */]
```

### Current GPUs

| Key | Name | VRAM | Usable (95%) | Architecture | SM Version |
|-----|------|------|--------------|--------------|------------|
| L4-24GB | L4 24GB | 24 | 23 | Ada Lovelace | sm_89 |
| A100-80GB | A100 80GB | 80 | 76 | Ampere | sm_80 |
| H100-80GB | H100 80GB | 80 | 76 | Hopper | sm_90 |
| H200-141GB | H200 141GB | 141 | 134 | Hopper | sm_90 |
| B100-192GB | B100 192GB | 192 | 182 | Blackwell | sm_100 |
| B200-192GB | B200 192GB | 192 | 182 | Blackwell | sm_100 |

---

## Adding New Model Example

```javascript
// 1. Fetch recipe
// curl https://recipes.vllm.ai/Qwen/Qwen3-32B.json

// 2. Extract data:
// - name, provider, params, totalParams, activeParams, prec, vram, contextLength, hf_url

// 3. Calculate weight-only VRAM:
// - IF prec is BF16: vram = totalParams * 2
// - IF prec is FP8: vram = totalParams * 1
// - IF prec is MXFP8: vram = totalParams * 1
// - IF prec is INT4/NVFP4/MXFP4/W4A16: vram = totalParams * 0.5

// 4. Add variants (map from recipe .variants object)

// 5. Add to MODELS_DATA with unique ID
{
    id: 200,  // use next available ID
    name: "Qwen3-32B",
    provider: "Qwen",
    params: "32B",
    totalParams: 32,
    activeParams: 32,
    prec: "BF16",
    vram: 64,              // 32 × 2 for BF16
    variants: [
        { prec: "FP8", vram: 32 },     // 32 × 1 for FP8
        { prec: "AWQ", vram: 16 }      // 32 × 0.5 for AWQ
    ],
    type: "text",
    contextLength: 40960,
    benchmark: { mmlu: 76.8, humaneval: 55.4, math: 52.5 },
    hf_url: "Qwen/Qwen3-32B"
}
```

---

## Adding New Quantization Format Example

If a new format like "FP6" is introduced:

```javascript
// 1. Add to normalizePrec() before the generic FP8 check
function normalizePrec(prec) {
    if (upper.includes('FP6')) return 'FP6';
    // ... existing checks ...
}

// 2. Add to GPU_QUANT_COMPAT for supported GPUs
'B200-192GB': ['BF16', 'FP8', 'FP6', 'INT4', 'AWQ', 'GPTQ', 'W4A16', 'NVFP4', 'NVFP4-QAD', 'MXFP4', 'MXFP8'],

// 3. Add badge CSS class
.badge-fp6 { background: rgba(...); color: ...; border: ...; }

// 4. Add to getQuantBadgeClass()
if (prec.includes('FP6')) return 'badge-fp6';

// 5. Add to GPU compatibility modal table
```

---

## Validation Before Changes

1. JavaScript syntax check (see Step 5 above)
2. Verify `getGPUVRAM()` exists exactly once
3. Test with L4 24GB — models >23GB should NOT fit on 1 GPU
4. Test NVFP4 variant on A100 — should NOT fit (NVFP4 unsupported on Ampere)
5. Test NVFP4 variant on B200 — should fit (Blackwell native)
6. Test MXFP4 variant on H100 — should fit (vLLM SW support)
7. Test MXFP8 variant on H100 — should NOT fit (unsupported)
8. Verify modal warns about KV cache overhead for long contexts (not included in VRAM)
9. Test `FP4+FP8` compound format (DeepSeek models) — treated as NVFP4

---

## Quick Reference

| Component | Location | Notes |
|-----------|----------|-------|
| `GPU_CONFIG` | ~Line 232 | GPU hardware specs |
| `GPU_QUANT_COMPAT` | ~Line 243 | GPU quantization support matrix |
| `normalizePrec()` | ~Line 255 | Format → canonical name (priority matters!) |
| `isPrecCompatible()` | ~Line 270 | Quant format compatibility check |
| `getGPUVRAM()` | ~Line 279 | Must be EXACTLY ONE instance |
| `MODELS_DATA` | ~Line 286 | 108 models, each with weight-only VRAM |
| `modelFitsGPU()` | ~Line 405 | Dual check: VRAM + quant compatibility |
| `filterModels()` | ~Line 426 | Applies all filters and sorts |
| `getQuantBadgeClass()` | ~Line 467 | Badge CSS only (not for compatibility logic) |
| `renderModels()` | ~Line 498 | Renders model cards grid |
| `updateStats()` | ~Line 599 | Updates stats cards |
| `openModal()` | ~Line 613 | Model detail modal |
| `openGPUInfoModal()` | ~Line 726 | GPU compatibility info modal |

---

## Reference Sources

| Resource | URL | Used For |
|----------|-----|----------|
| vLLM Recipes List | https://recipes.vllm.ai/models.json | Model inventory |
| vLLM Model Recipe | https://recipes.vllm.ai/{Provider}/{Model}.json | Per-model VRAM/variant data |
| vLLM Quantization Docs | https://docs.vllm.ai/en/stable/features/quantization/ | Hardware compatibility matrix |
| vLLM Quantization API | https://docs.vllm.ai/en/v0.19.1/api/vllm/model_executor/layers/quantization/ | Per-method implementation details |
| NVIDIA GPU Specs | https://www.nvidia.com/en-us/ | Tensor core capabilities |
| HuggingFace Models | https://huggingface.co/models | contextLength from config.json |
| NVIDIA Tensor Core Guide | https://www.nvidia.com/en-us/data-center/tensor-core/ | Native HW format support by architecture |

## Key Principles for Data Accuracy

- `GPU_QUANT_COMPAT` lists all formats a GPU can run (native + software). Only completely unsupported formats are excluded.
- `normalizePrec()` maps variant strings (like "GPTQ-Int4", "NVFP4-QAD", "W4A16") to 6 canonical format categories: BF16, FP8, INT4, NVFP4, MXFP4, MXFP8.
- "Blackwell only" on variant notes is a human hint; the actual enforcement is through `isPrecCompatible()` checking the GPU's format list.
- Legacy `gpuFeasibility` field on models is unused programmatically — VRAM + quant compatibility are computed at runtime by `modelFitsGPU()`.
- Context length should be sourced from HuggingFace config.json `max_position_embeddings` (or `model_max_length` / `rope_scaling` as fallback).
