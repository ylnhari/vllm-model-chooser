// Logic + data-integrity tests for vLLM Model Chooser.
// Run with:  node --test
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loadApp } from './harness.mjs';
import { normalizePrec as sharedNormalizePrec } from '../shared/prec.mjs';

const app = loadApp();
const { normalizePrec, isPrecCompatible, precSupportLevel, getGPUVRAM, estKVCacheGB,
        modelFitsGPU, MODELS_DATA, GPU_CONFIG, GPU_QUANT_COMPAT } = app;

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
// getGPUVRAM — single source of truth, reads selected GPU type
// ---------------------------------------------------------------------------
test('getGPUVRAM: scales usable VRAM by GPU count (usable = floor(vram * 0.95))', () => {
  app.setGpuType('L4-24GB');
  assert.equal(getGPUVRAM(1), 22);     // floor(24 * 0.95) = 22
  assert.equal(getGPUVRAM(8), 176);
  app.setGpuType('H200-141GB');
  assert.equal(getGPUVRAM(1), 133);    // floor(141 * 0.95) = 133
  app.setGpuType('A100-80GB');
  assert.equal(getGPUVRAM(2), 152);
});

test('GPU_CONFIG: usableVram equals floor(vram * 0.95) for every GPU', () => {
  for (const [key, cfg] of Object.entries(GPU_CONFIG)) {
    assert.equal(cfg.usableVram, Math.floor(cfg.vram * 0.95), `${key} usableVram`);
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
  app.setGpuType('H100-80GB');
  // 8x H100 = 608GB: base FP8 (671) doesn't fit; NVFP4 (336) fits VRAM but is
  // unsupported on Hopper, so the model is blocked by the quant gate.
  const blocked = modelFitsGPU(m, 8);
  assert.equal(blocked.fits, false);
  assert.equal(blocked.reason, 'quant');
  // 3x B200 = 546GB: base FP8 671 doesn't fit, NVFP4 336 fits VRAM AND is native → variant selected.
  app.setGpuType('B200-192GB');
  const r = modelFitsGPU(m, 3);
  assert.equal(r.fits, true);
  assert.equal(r.variant.prec, 'NVFP4');
  assert.equal(r.level, 'native');
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
    assert.ok(m.benchmark && typeof m.benchmark === 'object');
    assert.ok(m.hf_url && m.hf_url.includes('/'), `${m.name} hf_url must look like provider/model`);
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
