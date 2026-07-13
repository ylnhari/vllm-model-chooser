// Logic + data-integrity tests for vLLM Model Chooser.
// Run with:  node --test
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loadApp } from './harness.mjs';
import { normalizePrec as sharedNormalizePrec } from '../shared/prec.mjs';

const app = loadApp();
const { normalizePrec, isPrecCompatible, precSupportLevel, getGPUVRAM, getGPUBudget,
        estKVCacheGB, modelFitsGPU, MODELS_DATA, GPU_CONFIG, GPU_QUANT_COMPAT } = app;

// ---------------------------------------------------------------------------
// normalizePrec — priority order is the whole point (specific before generic)
// ---------------------------------------------------------------------------
test('normalizePrec: specific FP4/FP8 formats win over generic substrings', () => {
  assert.equal(normalizePrec('NVFP4'), 'NVFP4');
  assert.equal(normalizePrec('NVFP4-QAD'), 'NVFP4');
  assert.equal(normalizePrec('MXFP4'), 'MXFP4');
  assert.equal(normalizePrec('MXFP8'), 'MXFP8');          // must NOT become FP8
  assert.equal(normalizePrec('FP4+FP8'), 'NVFP4');        // DeepSeek compound
  assert.equal(normalizePrec('FP8'), 'FP8');
  assert.equal(normalizePrec('AMD-FP8'), 'FP8');
  assert.equal(normalizePrec('BF16'), 'BF16');
});

test('normalizePrec: INT4 family maps to INT4, INT8 family maps to INT8', () => {
  for (const p of ['INT4', 'AWQ', 'GPTQ', 'W4A16', 'GPTQ-Int4', 'QAT-W4A16']) {
    assert.equal(normalizePrec(p), 'INT4', `${p} should map to INT4`);
  }
  for (const p of ['INT8', 'W8A8', 'INT8-W8A8']) {
    assert.equal(normalizePrec(p), 'INT8', `${p} should map to INT8`);
  }
});

test('normalizePrec: unknown / empty returns null', () => {
  assert.equal(normalizePrec('300B-A47B'), null);
  assert.equal(normalizePrec(''), null);
  assert.equal(normalizePrec(null), null);
  assert.equal(normalizePrec(undefined), null);
});

// Drift guard: app.js keeps its own copy of normalizePrec (it runs as a classic
// browser <script>, no ESM import), while the Node scripts import ../shared/prec.mjs.
// This asserts the two never diverge — the exact failure that made factcheck's old
// private copy mis-handle INT8/W8A8 and unknown formats.
test('normalizePrec: app.js copy matches the shared ../shared/prec.mjs module', () => {
  const inputs = ['BF16', 'FP8', 'AMD-FP8', 'FP4+FP8', 'NVFP4', 'NVFP4-QAD', 'MXFP4',
    'MXFP8', 'INT8', 'W8A8', 'INT8-W8A8', 'INT4', 'AWQ', 'GPTQ', 'GPTQ-Int4', 'W4A16',
    'QAT-W4A16', 'QAT-mobile', '300B-A47B', 'INT2/4/8', 'fp8', '', null, undefined];
  for (const p of inputs) {
    assert.equal(normalizePrec(p), sharedNormalizePrec(p), `mismatch on ${JSON.stringify(p)}`);
  }
});

// ---------------------------------------------------------------------------
// isPrecCompatible — the quantization gate
// ---------------------------------------------------------------------------
test('precSupportLevel: NVFP4 is Blackwell-native ONLY (recipes gate it to Blackwell)', () => {
  assert.equal(precSupportLevel('NVFP4', 'B200-192GB'), 'native');
  assert.equal(precSupportLevel('NVFP4', 'B100-192GB'), 'native');
  assert.equal(precSupportLevel('NVFP4', 'H100-80GB'), null);
  assert.equal(precSupportLevel('NVFP4', 'A100-80GB'), null);
  assert.equal(precSupportLevel('NVFP4', 'L4-24GB'), null);
});

test('isPrecCompatible: NVFP4 only compatible on Blackwell', () => {
  assert.equal(isPrecCompatible('NVFP4', 'B200-192GB'), true);
  assert.equal(isPrecCompatible('NVFP4', 'B100-192GB'), true);
  assert.equal(isPrecCompatible('NVFP4', 'H100-80GB'), false);
  assert.equal(isPrecCompatible('NVFP4', 'A100-80GB'), false);
  assert.equal(isPrecCompatible('NVFP4', 'L4-24GB'), false);
});

test('precSupportLevel: FP8 native on Ada/Hopper/Blackwell, software on Ampere', () => {
  assert.equal(precSupportLevel('FP8', 'L4-24GB'), 'native');
  assert.equal(precSupportLevel('FP8', 'H100-80GB'), 'native');
  assert.equal(precSupportLevel('FP8', 'B200-192GB'), 'native');
  assert.equal(precSupportLevel('FP8', 'A100-80GB'), 'sw');
});

test('precSupportLevel: MXFP4 native on Blackwell, software everywhere else (recipes run it on A100)', () => {
  assert.equal(precSupportLevel('MXFP4', 'B200-192GB'), 'native');
  assert.equal(precSupportLevel('MXFP4', 'H100-80GB'), 'sw');
  assert.equal(precSupportLevel('MXFP4', 'A100-80GB'), 'sw');
  assert.equal(precSupportLevel('MXFP4', 'L4-24GB'), 'sw');
});

test('precSupportLevel: MXFP8 native on Blackwell only', () => {
  assert.equal(precSupportLevel('MXFP8', 'B200-192GB'), 'native');
  assert.equal(precSupportLevel('MXFP8', 'B100-192GB'), 'native');
  assert.equal(precSupportLevel('MXFP8', 'H100-80GB'), null);
  assert.equal(precSupportLevel('MXFP8', 'A100-80GB'), null);
  assert.equal(isPrecCompatible('MXFP8', 'B200-192GB'), true);
  assert.equal(isPrecCompatible('MXFP8', 'H100-80GB'), false);
});

test('isPrecCompatible: FP8 everywhere (native or vLLM SW), BF16/INT4 everywhere', () => {
  for (const g of Object.keys(GPU_CONFIG)) {
    assert.equal(isPrecCompatible('FP8', g), true, `FP8 on ${g}`);
    assert.equal(isPrecCompatible('BF16', g), true, `BF16 on ${g}`);
    assert.equal(isPrecCompatible('AWQ', g), true, `AWQ on ${g}`);
  }
});

test('isPrecCompatible: fail-safe — unknown GPU or unknown format returns true', () => {
  assert.equal(isPrecCompatible('NVFP4', 'NONEXISTENT-GPU'), true);
  assert.equal(isPrecCompatible('SOME-FUTURE-FORMAT', 'B200-192GB'), true);
});

// ---------------------------------------------------------------------------
// Memory budget — mirrors vLLM: budget = physical x util; weights+KV = budget - reserve
// ---------------------------------------------------------------------------
test('getGPUBudget: physical x gpu-memory-utilization, scaled by GPU count', () => {
  app.setMemUtil(0.95);
  app.setGpuType('A100-80GB');
  assert.equal(getGPUBudget(1), 76);        // 80 * 0.95
  assert.equal(getGPUBudget(4), 304);
  app.setMemUtil(0.90);                     // vLLM's own default
  assert.equal(getGPUBudget(1), 72);        // 80 * 0.90
  app.setMemUtil(0.95);
});

test('getGPUVRAM: usable = budget - activation reserve (per GPU)', () => {
  app.setGpuType('A100-80GB');
  app.setMemUtil(0.95);
  app.setReserve(2);
  assert.equal(getGPUVRAM(1), 74);          // 80*0.95 - 2
  assert.equal(getGPUVRAM(4), 296);         // 4*(76 - 2)
  app.setReserve(0);
  assert.equal(getGPUVRAM(4), 304);         // reserve is genuinely subtractable
  app.setReserve(app.DEFAULT_RESERVE_GB);
});

test('getGPUVRAM: the reserve is per-GPU, so it scales with GPU count', () => {
  app.setGpuType('H100-80GB');
  app.setMemUtil(0.95);
  app.setReserve(4);
  assert.equal(getGPUVRAM(8), 8 * (80 * 0.95 - 4));
  app.setReserve(app.DEFAULT_RESERVE_GB);
});

test('GPU_CONFIG: carries PHYSICAL vram only — no baked-in usable value to drift', () => {
  for (const [key, cfg] of Object.entries(GPU_CONFIG)) {
    assert.ok(cfg.vram > 0, `${key} vram`);
    assert.equal(cfg.usableVram, undefined, `${key} must not bake in a usable value`);
  }
});

// ---------------------------------------------------------------------------
// modelFitsGPU — the dual check (VRAM AND quantization)
// ---------------------------------------------------------------------------
test('modelFitsGPU: gpus=0 (Any) always fits', () => {
  for (const m of MODELS_DATA.slice(0, 10)) {
    assert.equal(modelFitsGPU(m, 0).fits, true);
  }
});

test('modelFitsGPU: tiny model fits a single L4; huge model does not', () => {
  app.setGpuType('L4-24GB');
  const tiny = MODELS_DATA.find(m => m.vram <= 23 && m.variants.length === 0);
  assert.equal(modelFitsGPU(tiny, 1).fits, true);
  const huge = MODELS_DATA.find(m => m.vram > 184); // exceeds 8x L4 even before quant
  assert.equal(modelFitsGPU(huge, 8).fits, false);
});

test('modelFitsGPU: the quant gate blocks a Blackwell-only NVFP4 variant on Hopper', () => {
  // DeepSeek-V3 (id 120): base FP8 671GB (weight-only), NVFP4 variant 336GB (Blackwell only).
  const m = MODELS_DATA.find(x => x.id === 120);
  app.setMemUtil(0.95);
  app.setReserve(2);
  app.setGpuType('H100-80GB');
  // 8x H100 = 8*(80*0.95 - 2) = 592GB: base FP8 (671) doesn't fit; NVFP4 (336) fits
  // VRAM but is unsupported on Hopper, so the model is blocked by the quant gate.
  const blocked = modelFitsGPU(m, 8);
  assert.equal(blocked.fits, false);
  assert.equal(blocked.reason, 'quant');
  // 3x B200 = 3*(192*0.95 - 2) = 541GB: base FP8 671 doesn't fit; NVFP4 336 fits VRAM
  // AND is Blackwell-native → the variant is selected.
  app.setGpuType('B200-192GB');
  const r = modelFitsGPU(m, 3);
  assert.equal(r.fits, true);
  assert.equal(r.variant.prec, 'NVFP4');
  assert.equal(r.level, 'native');
});

// The bar and the ✓/✗ used to disagree: the bar was drawn from base-precision weights
// against a hardcoded 8-GPU denominator, while the verdict came from the fitting
// variant against the SELECTED count. modelFitsGPU now returns the numbers the bar is
// drawn from, so the two cannot drift apart.
test('modelFitsGPU: returns the weights/kv/usable it actually decided on', () => {
  app.setGpuType('B200-192GB');
  app.setKVContext(0);
  const m = MODELS_DATA.find(x => x.id === 120);
  const r = modelFitsGPU(m, 3);
  assert.equal(r.fits, true);
  assert.equal(r.weights, r.variant.vram, 'weights must be the variant that was chosen, not the base');
  assert.equal(r.usable, getGPUVRAM(3));
  assert.ok(r.weights + r.kv <= r.usable, 'the reported numbers must satisfy the fit they claim');

  // On a miss, weights = the model's SMALLEST candidate (its best case), so the bar
  // says "even the smallest quantization overflows" rather than overstating.
  const miss = modelFitsGPU(m, 1);
  assert.equal(miss.fits, false);
  const smallest = Math.min(m.vram, ...m.variants.map(v => v.vram));
  assert.equal(miss.weights, smallest);
});

test('modelFitsGPU: fit is monotonic in GPU count', () => {
  app.setGpuType('H100-80GB');
  for (const m of MODELS_DATA) {
    let prev = false;
    for (const n of [1, 2, 3, 4, 5, 6, 7, 8]) {
      const fits = modelFitsGPU(m, n).fits;
      if (prev) assert.equal(fits, true, `${m.name} fit regressed from ${n - 1}→${n} GPUs`);
      prev = fits;
    }
  }
});

// ---------------------------------------------------------------------------
// KV-cache estimate (#12)
// ---------------------------------------------------------------------------
test('estKVCacheGB: zero when disabled, positive and monotonic in context', () => {
  const m = MODELS_DATA.find(x => x.id === 112); // Llama-3.3-70B
  assert.equal(estKVCacheGB(m, 0), 0);
  const a = estKVCacheGB(m, 32768);
  const b = estKVCacheGB(m, 131072);
  assert.ok(a > 0 && b > a, 'KV should grow with context length');
});

test('estKVCacheGB: uses exact attention geometry (kvBytesPerToken) when present', () => {
  const m = MODELS_DATA.find(x => x.kvBytesPerToken && !x.kvSlidingBytesPerToken);
  assert.ok(m, 'expected at least one full-attention model with synced KV geometry');
  // geometry path is exact: bytes/token × tokens / 1e9, independent of totalParams
  assert.equal(estKVCacheGB(m, 100000), m.kvBytesPerToken * 100000 / 1e9);
  assert.equal(estKVCacheGB(m, 0), 0);
});

// The correction that matters most: sliding-window layers are CAPPED at the window,
// so their KV stops growing once context exceeds it. Treating every layer as full
// attention over-counted Gemma-class models by several × at long context.
test('estKVCacheGB: sliding-window KV is capped at the window, not the context', () => {
  const m = MODELS_DATA.find(x => x.kvSlidingBytesPerToken > 0 && x.kvWindow > 0);
  assert.ok(m, 'expected at least one sliding-window model');
  const win = m.kvWindow;
  const expected = t => (m.kvBytesPerToken * t + m.kvSlidingBytesPerToken * Math.min(t, win)) / 1e9;
  assert.equal(estKVCacheGB(m, win), expected(win));
  assert.equal(estKVCacheGB(m, 131072), expected(131072));
  // Beyond the window, ONLY the full-attention layers keep growing.
  const growth = estKVCacheGB(m, 131072 * 2) - estKVCacheGB(m, 131072);
  assert.equal(growth, m.kvBytesPerToken * 131072 / 1e9);
  // And it must be strictly cheaper than pretending every layer were full attention.
  const naive = (m.kvBytesPerToken + m.kvSlidingBytesPerToken) * 131072 / 1e9;
  assert.ok(estKVCacheGB(m, 131072) < naive, 'sliding KV must be below the full-attention upper bound');
});

test('estKVCacheGB: FP8 KV cache (--kv-cache-dtype fp8) halves the footprint', () => {
  const m = MODELS_DATA.find(x => x.kvBytesPerToken > 0);
  app.setKVDtype('fp16');
  const fp16 = estKVCacheGB(m, 131072);
  app.setKVDtype('fp8');
  assert.equal(estKVCacheGB(m, 131072), fp16 / 2);
  app.setKVDtype('fp16');
});

test('estKVCacheGB: no-KV models (diffusion/generative) never consume KV at any context', () => {
  const none = MODELS_DATA.filter(m => m.kvSource === 'none');
  assert.ok(none.length > 0, 'expected some single-pass generative models');
  for (const m of none) {
    assert.equal(estKVCacheGB(m, 1048576), 0, `${m.name} must not consume KV`);
  }
});

test('data: KV geometry fields are non-negative integers with a known provenance', () => {
  const SOURCES = new Set(['config', 'mirror', 'estimate', 'none']);
  for (const m of MODELS_DATA) {
    if (m.kvBytesPerToken == null) continue;
    assert.ok(Number.isInteger(m.kvBytesPerToken) && m.kvBytesPerToken >= 0, `${m.name} kvBytesPerToken`);
    assert.ok(SOURCES.has(m.kvSource), `${m.name} kvSource must be one of ${[...SOURCES]}, got ${m.kvSource}`);
    // A sliding-layer budget is meaningless without the window that caps it.
    if (m.kvSlidingBytesPerToken) {
      assert.ok(m.kvWindow > 0, `${m.name} has sliding KV but no kvWindow to cap it`);
    }
  }
});

// ---------------------------------------------------------------------------
// Worst-case concurrency — the KV pool left after weights
// ---------------------------------------------------------------------------
test('maxConcurrentRequests: derives from the KV pool, and shrinks as context grows', () => {
  app.setGpuType('H200-141GB');
  app.setMemUtil(0.95);
  app.setReserve(2);
  const m = MODELS_DATA.find(x => x.kvBytesPerToken > 0 && x.vram < 40);
  const at32k = app.maxConcurrentRequests(m, m.vram, 2, 32768);
  const at128k = app.maxConcurrentRequests(m, m.vram, 2, 131072);
  assert.ok(at32k > 0, 'should fit at least one request at 32K');
  assert.ok(at128k <= at32k, 'longer context must not increase concurrency');
  assert.equal(app.maxConcurrentRequests(m, m.vram, 2, 0), null, 'null when KV estimate is off');
});

test('modelFitsGPU: enabling the KV estimate can turn a fit into a non-fit', () => {
  app.setGpuType('H100-80GB');
  // find a model that fits on 1 GPU on weights alone but is close to the limit
  const m = MODELS_DATA.find(x => {
    app.setKVContext(0);
    return modelFitsGPU(x, 1).fits && x.vram > 60 && x.variants.length === 0;
  });
  if (m) {
    app.setKVContext(0);
    assert.equal(modelFitsGPU(m, 1).fits, true);
    app.setKVContext(1048576);                 // 1M tokens → large KV
    assert.equal(modelFitsGPU(m, 1).fits, false);
  }
  app.setKVContext(0);                          // reset shared state
});

// ---------------------------------------------------------------------------
// Data integrity — guards against future MODELS_DATA edits
// ---------------------------------------------------------------------------
test('data: all model ids are unique', () => {
  const ids = MODELS_DATA.map(m => m.id);
  assert.equal(new Set(ids).size, ids.length);
});

test('data: every model has required, well-typed fields', () => {
  for (const m of MODELS_DATA) {
    assert.equal(typeof m.id, 'number', `${m.name} id`);
    assert.ok(m.name && typeof m.name === 'string');
    assert.ok(m.provider && typeof m.provider === 'string');
    assert.equal(typeof m.totalParams, 'number');
    assert.ok(m.vram > 0, `${m.name} vram must be > 0`);
    assert.ok(Array.isArray(m.variants), `${m.name} variants must be an array`);
    assert.ok(m.hf_url && m.hf_url.includes('/'), `${m.name} hf_url must look like provider/model`);
  }
});

// Benchmarks were REMOVED, not merely hidden: only 2 of 108 models expose structured
// eval metrics on HuggingFace, so the old hand-curated MMLU/HumanEval numbers had no
// verifiable source. Unsourced scores rendered as confident bars (and sortable!) are
// worse than none — this guards against them creeping back in.
test('data: no unsourced benchmark scores are carried in the dataset', () => {
  for (const m of MODELS_DATA) {
    assert.equal(m.benchmark, undefined, `${m.name} must not carry unsourced benchmark scores`);
    assert.equal(m.tested, undefined, `${m.name} must not carry a meaningless tested flag`);
  }
});

test('data: every precision string normalizes to a known canonical (or null) without throwing', () => {
  const known = new Set(['BF16', 'FP8', 'INT8', 'INT4', 'NVFP4', 'MXFP4', 'MXFP8', null]);
  for (const m of MODELS_DATA) {
    assert.ok(known.has(normalizePrec(m.prec)), `${m.name} base prec ${m.prec}`);
    for (const v of m.variants) {
      // variants may carry exotic notes (e.g. "300B-A47B") that normalize to null — allowed
      assert.doesNotThrow(() => normalizePrec(v.prec));
    }
  }
});

test('data: recipe_id, when present, is a same-target path differing only by casing from hf_url', () => {
  for (const m of MODELS_DATA) {
    if (!m.recipe_id) continue;
    assert.notEqual(m.recipe_id, m.hf_url, `${m.name} recipe_id should differ from hf_url when set`);
    assert.equal(m.recipe_id.toLowerCase(), m.hf_url.toLowerCase(), `${m.name} recipe_id must match hf_url case-insensitively`);
  }
});

test('data: every GPU has a tri-state quant-compat map with valid levels', () => {
  for (const g of Object.keys(GPU_CONFIG)) {
    const map = GPU_QUANT_COMPAT[g];
    assert.ok(map && typeof map === 'object' && !Array.isArray(map), `${g} missing quant-compat map`);
    for (const [fmt, level] of Object.entries(map)) {
      assert.ok(level === 'native' || level === 'sw', `${g}.${fmt} level must be native|sw, got ${level}`);
    }
    assert.equal(map.BF16, 'native', `${g} must support BF16 natively`);
  }
});
