# Development Guide - vLLM Model Chooser

This file contains technical documentation for maintaining and extending the vLLM Model Chooser application.

## Application Overview

Static, no-build web application that helps users select vLLM-compatible LLM models based on GPU hardware, VRAM requirements, context length, and quantization options. Served by any static server (`python3 -m http.server`).

**File layout:**
- `index.html` — markup + styles; loads `data.js` then `app.js` as plain globals (no modules/bundler).
- `data.js` — `GPU_CONFIG`, `GPU_QUANT_COMPAT`, `MODELS_DATA`. **Auto-generated** by `scripts/sync-data.mjs` from live vLLM recipes + HuggingFace configs. Only `type` and mixed-precision `vram` are curated. Don't hand-edit generated fields (`contextLength`, variant precisions, `recipe_id`, all `kv*`) — re-run the generator.
- `app.js` — all logic + rendering.
- `shared/prec.mjs` — `normalizePrec` + `BYTES_PER_PARAM`, imported by both Node scripts (and mirrored byte-for-byte into `app.js`, which is a classic browser script).
- `scripts/sync-data.mjs` (regenerate data) · `scripts/factcheck.mjs` (audit vs recipes — `npm run factcheck`).
- `tests/` — `npm test` (Node built-in runner, no deps). Run after any logic/data change.

**`GPU_QUANT_COMPAT` is a tri-state map**, not an array: `{ CANONICAL: 'native' | 'sw' }` — present key = supported (`native` = HW tensor cores, `sw` = vLLM software path), absent = unsupported. `precSupportLevel(prec, gpuType)` returns `'native' | 'sw' | null`; `isPrecCompatible` is just `precSupportLevel(...) !== null`.

**`GPU_CONFIG` carries physical VRAM only** — no `usableVram`. The usable budget is derived at runtime (`physical × util − activation reserve`), both of which are user controls mirroring vLLM's `--gpu-memory-utilization`. See "The memory budget" below.

**Critical**: `vram` in `MODELS_DATA` is **weight-only**. KV cache is never folded into it — it's computed separately from each model's real HF attention geometry (`estKVCacheGB`), including sliding-window layers, and the user toggles it into the fit check.

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

## Extending the tool (READ before adding models / precisions / GPUs / architectures)

The golden rule: **`data.js` is generated, not hand-authored.** You add the minimum
seed to `MODELS_DATA` (or edit a table in `scripts/sync-data.mjs`), then let the
generator fill in everything derivable, then verify. Always finish with the loop:

```
node scripts/sync-data.mjs      # rebuild data.js from live recipes + HF configs
npm run factcheck               # must end "Total discrepancies: 0"
npm test                        # must be all-pass
```

`normalizePrec` + `BYTES_PER_PARAM` live in **`shared/prec.mjs`** and are imported by
both scripts. `app.js` keeps a byte-identical copy of `normalizePrec` (it runs as a
classic browser `<script>`, no ESM import); a drift-guard test fails if the two
diverge, so **edit `shared/prec.mjs` and mirror the change into `app.js`**.

### Add a new MODEL
1. Confirm it exists in the source of truth: it must appear in
   [`recipes.vllm.ai/models.json`](https://recipes.vllm.ai/models.json) (`factcheck`
   flags "not in inventory" otherwise).
2. Append a seed row to `MODELS_DATA` in `data.js` with a fresh unique `id` and the
   fields you can't derive: `name`, `provider`, `params`, `totalParams`,
   `activeParams`, base `prec`, `type`, `hf_url` (the case-sensitive HF path). Set
   `vram`/`variants`/`contextLength` to rough placeholders — the generator overwrites
   the derivable ones.
3. Run `node scripts/sync-data.mjs`. It fills `contextLength`, adds recipe variant
   precisions, sets `recipe_id` casing, **computes weight-only `vram`** for pure
   precisions, and **computes `kvBytesPerToken`** from the HF `config.json`.
4. Only these stay curated (the generator never overwrites them): `type`, and `vram`
   **only for mixed-precision** models (MXFP4/`FP4+FP8`/`AMD-FP8`, where the naive
   formula is wrong — e.g. gpt-oss).
5. **Do not add benchmark scores.** They were removed on purpose: only 2/108 models
   expose structured eval metrics on HF (`cardData.model-index`), and those aren't
   comparable across models. Hand-curated scores have no verifiable source, and this
   tool's whole contract is that every number traces to one. A test enforces this.

### Add a new QUANTIZATION format
1. `shared/prec.mjs`: add the recognizer to `normalizePrec` **before** any generic
   substring it could be mistaken for (specific-before-generic is the whole contract —
   see the MXFP8-vs-FP8 tests), and add its weight bytes/param to `BYTES_PER_PARAM`.
   Mirror the `normalizePrec` change into `app.js`.
2. `scripts/sync-data.mjs → GPU_QUANT_COMPAT`: add the format key to every GPU that
   supports it, `'native'` (HW tensor cores) or `'sw'` (vLLM software/dequant path),
   **with a source citation** (vLLM "supported hardware" docs + recipe variant
   `description`/`hardware_overrides`). Absent key = unsupported.
3. `app.js`: add a `badge-<fmt>` CSS class mapping in `getQuantBadgeClass`, add the
   format to `FORMAT_ORDER`/`FORMAT_NOTE` in `openGPUInfoModal` (the modal renders
   from the data, so it picks up the new column/row automatically), and add the
   `<option>` to `#quantFilter` in `index.html`.
4. Regenerate + `factcheck` + `test`. Add/adjust a `normalizePrec`/`precSupportLevel`
   test for the new format.

### Add a new GPU
Edit **both** `GPU_CONFIG` and `GPU_QUANT_COMPAT` in `scripts/sync-data.mjs` (not
`data.js` — it's generated). `GPU_CONFIG` carries **physical `vram` only** — do NOT add a
`usableVram`: the usable budget is derived at runtime from the user's memory-utilization
knob, and a second baked-in value would just drift. Add a `<option>` to `#gpuTypeSelect`
in `index.html`, regenerate, verify. The info-modal table and snapshot cards are
data-driven and update themselves.

### The memory budget (why there's no `usableVram`)
The app mirrors vLLM's own accounting:

```
budget        = physical × gpu-memory-utilization   # #memUtilSelect, default 0.95
weights + KV  = budget − activation reserve         # #reserveSelect, default 2 GB/GPU
KV pool       = (weights + KV) − weights            # vLLM fills this greedily
```

`getGPUVRAM(gpus)` is the single source of truth for the capacity check. The **activation
reserve is an assumption, and must stay visibly labelled as one** — vLLM does not compute
it, it *measures* it by profiling a forward pass at startup, and it scales with
`--max-num-batched-tokens`. Never bury it in a constant; it's a user control (settable to 0)
and carries a `*` marker in the UI. Same principle for the util default (0.95 is more
optimistic than vLLM's own 0.90–0.92 — so the user gets to choose).

### Add a new ARCHITECTURE (KV-cache geometry)
KV is **not** a single bytes-per-token constant. The app computes:

```
KV(tokens) = kvBytesPerToken × tokens
           + kvSlidingBytesPerToken × min(tokens, kvWindow)
```

because **sliding-window layers are capped at the window** however long the context grows.
Getting this wrong is not academic: treating every layer as full attention overstated
`gemma-4-31B` at 128K as 128.8 GB when the truth is 22.3 GB (**5.8×**), wrongly excluding
it from GPUs it fits. Gemma-3/4, Step-3.7, Voxtral and DeepSeek-V4 are all mostly sliding.

`kvBytesPerTokenFromConfig()` in sync-data.mjs returns `{ full, sliding, window }` from the
HF `config.json`. **Every model should carry geometry**, plus a `kvSource` provenance tag —
absent geometry means the app falls back to a coarse params proxy (`‡`), a gap to close.

Fallback chain (`fetchConfigKV`), which also sets `kvSource`:
1. **Direct** `config.json`, then `params.json` (Mistral consolidated) → `kvSource: 'config'`.
2. **Ungated mirror** (`HF_CONFIG_MIRROR`) — gated repos (HF 401, e.g. Meta Llama) map to a
   same-architecture public mirror so geometry is *fetched, not guessed* → `'mirror'` (`°`).
3. **Explicit override** (`KV_GEOMETRY_OVERRIDES`) — `0` for generative/no-KV models
   (→ `'none'`); `{ layers, kvHeads, headDim }` for the rare gated model with no mirror
   (→ `'estimate'`, `†` — cite the source, as `plamo-3` does).

The two halves of the computation, keyed off config fields:
- **per-layer bytes** (`perLayerKVBytes`): MLA (`kv_lora_rank + qk_rope_head_dim` — a single
  compressed latent, **not** ×heads, **not** ×2 — DeepSeek); else GQA/MHA
  (`2 × num_key_value_heads × head_dim × 2B`). `head_dim` falls back to `attention_head_dim`
  or `hidden_size / num_attention_heads`.
- **layer classification** (`classifyLayers`): `hybrid_override_pattern` (count `*`) and
  `layers_block_type` (count `*attention*`) for Mamba/attention hybrids — **Mamba layers
  cache nothing and are excluded from both counts**; `layer_types` → split `full_attention`
  vs `sliding_attention`; else `sliding_window` + `sliding_window_pattern` (Gemma-3 style:
  every Nth layer is global, the rest slide); else all layers are full attention.
  A sliding split with no declared window is treated as full attention rather than silently
  under-counting.
- **nesting**: `descendToAttn()` unwraps `text_config` / `thinker_config` / `decoder` / … for
  multimodal/omni configs; `tolerantJson()` survives `Infinity`/`NaN` literals.

For a genuinely new scheme, add a branch keyed on a distinctive field or `model_type`,
re-sync, and spot-check `@128K` looks sane: dense-70B ≈ 43GB, MLA-671B ≈ 9GB, Mamba-hybrid
or mostly-sliding ≈ single-digit to low-tens GB. **Beware the under-count direction** — it
makes a model claim to fit when it won't, which is the harmful error.

---

## Reference

The compatibility matrix, GPU specs, memory-budget formula and KV formula live in
**`README.md`** — and, authoritatively, in the code itself (`scripts/sync-data.mjs` for
`GPU_CONFIG` / `GPU_QUANT_COMPAT` / KV geometry, `app.js` for the fit logic). This file
does not restate them: a duplicated matrix is a matrix that goes stale, and this section
previously did exactly that (it still described `GPU_QUANT_COMPAT` as an array of format
strings long after it became a tri-state `{ FORMAT: 'native' | 'sw' }` map, and had MXFP4
marked unsupported on A100 when the recipes explicitly run gpt-oss on one).

### The invariants that actually matter

1. **`data.js` is generated.** Never hand-edit recipe- or config-sourced fields
   (`contextLength`, variant precisions, `recipe_id`, `vram` for pure precisions, all `kv*`
   fields). Re-run `node scripts/sync-data.mjs`.
2. **`vram` is weight-only.** Never fold KV cache into it. Recipe `vram_minimum_gb` is
   KV-inclusive — do not copy it into our `vram` field.
3. **One `getGPUVRAM`.** It is the single source of truth for capacity
   (`physical × util − reserve`). Never add a second definition or a hardcoded fallback.
4. **Dual check.** `modelFitsGPU` must gate on BOTH capacity AND `precSupportLevel`.
   It returns the numbers the UI draws (`weights`/`kv`/`usable`) so the card's bar and its
   ✓/✗ are physically incapable of disagreeing — they used to, badly.
5. **`normalizePrec`: specific before generic.** `MXFP8` must be matched before `FP8`, and
   the FP4 family before `FP8`, or `"MXFP8"`/`"FP4+FP8"` silently become `FP8`. It lives in
   `shared/prec.mjs`, mirrored byte-for-byte into `app.js` (drift-guarded by a test).
6. **Every displayed number traces to a source.** Recipes, HF `config.json`, or an NVIDIA
   datasheet — or it is computed from those and marked as an estimate (`~ ≈ ° † ‡ *`).
   If you cannot source it, do not display it. This is why benchmarks are gone.

### Verification loop (run all three, always)

```bash
node scripts/sync-data.mjs   # rebuild data.js from live recipes + HF configs
npm run factcheck            # must end "Total discrepancies: 0"
npm test                     # must be all-pass
```

`factcheck` audits `data.js` against the live recipes (inventory, context length, parameter
counts, variant precisions, and the pure-precision VRAM formula). `npm test` additionally
guards the things no external source can check for us: normalizer drift, fit monotonicity,
KV sliding-window capping, the tri-state compat map, and the absence of unsourced scores.

### Sanity numbers (spot-check after any KV change)

| Model shape | KV @ 128K, FP16 |
|---|---|
| dense 70B (GQA, full attention) | ≈ 43 GB |
| MLA 671B (DeepSeek) | ≈ 9 GB |
| mostly-sliding 31B (Gemma-class) | ≈ 22 GB (**not** ~129 GB — that's the full-attention error) |
| diffusion / single-pass generative | 0 GB at any context |

If a change makes KV *smaller*, be suspicious: under-counting makes a model claim to fit
when it won't, which is the harmful direction. Over-counting merely hides a usable model.
