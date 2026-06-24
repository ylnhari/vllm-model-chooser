// Logic + data-integrity tests for vLLM Model Chooser.
// Run with:  node --test
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { loadApp } from './harness.mjs';

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

// ---------------------------------------------------------------------------
// isPrecCompatible — the quantization gate
// ---------------------------------------------------------------------------
test('precSupportLevel: NVFP4 native on Blackwell, software on Hopper/Ampere, unsupported on Ada', () => {
  assert.equal(precSupportLevel('NVFP4', 'B200-192GB'), 'native');
  assert.equal(precSupportLevel('NVFP4', 'B100-192GB'), 'native');
  assert.equal(precSupportLevel('NVFP4', 'H100-80GB'), 'sw');
  assert.equal(precSupportLevel('NVFP4', 'A100-80GB'), 'sw');
  assert.equal(precSupportLevel('NVFP4', 'L4-24GB'), null);
});

test('isPrecCompatible: NVFP4 compatible everywhere except Ada L4 (software fallback counts)', () => {
  assert.equal(isPrecCompatible('NVFP4', 'B200-192GB'), true);
  assert.equal(isPrecCompatible('NVFP4', 'H100-80GB'), true);   // vLLM SW
  assert.equal(isPrecCompatible('NVFP4', 'A100-80GB'), true);   // vLLM SW
  assert.equal(isPrecCompatible('NVFP4', 'L4-24GB'), false);    // unsupported
});

test('precSupportLevel: FP8 native on Ada/Hopper/Blackwell, software on Ampere', () => {
  assert.equal(precSupportLevel('FP8', 'L4-24GB'), 'native');
  assert.equal(precSupportLevel('FP8', 'H100-80GB'), 'native');
  assert.equal(precSupportLevel('FP8', 'B200-192GB'), 'native');
  assert.equal(precSupportLevel('FP8', 'A100-80GB'), 'sw');
});

test('isPrecCompatible: MXFP4 software-supported on Hopper, native on Blackwell, not on Ada/Ampere', () => {
  assert.equal(isPrecCompatible('MXFP4', 'H100-80GB'), true);
  assert.equal(isPrecCompatible('MXFP4', 'B200-192GB'), true);
  assert.equal(isPrecCompatible('MXFP4', 'A100-80GB'), false);
  assert.equal(isPrecCompatible('MXFP4', 'L4-24GB'), false);
});

test('isPrecCompatible: MXFP8 only on Blackwell', () => {
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

test('modelFitsGPU: the quant gate blocks an unsupported format even when VRAM is fine', () => {
  // gpt-oss-20b (id 90): base MXFP4, 16GB. MXFP4 is unsupported on Ada L4.
  const m = MODELS_DATA.find(x => x.id === 90);
  assert.equal(m.prec.includes('MXFP4'), true);
  app.setGpuType('L4-24GB');               // 1x L4 = 22GB usable, so 16GB fits on VRAM...
  const blocked = modelFitsGPU(m, 1);
  assert.equal(blocked.fits, false);       // ...but MXFP4 is unsupported on Ada
  assert.equal(blocked.reason, 'quant');
  app.setGpuType('H100-80GB');             // MXFP4 is vLLM-software on Hopper
  const ok = modelFitsGPU(m, 1);
  assert.equal(ok.fits, true);
  assert.equal(ok.level, 'sw');
});

test('modelFitsGPU: selects a lower-precision variant when the base does not fit', () => {
  // DeepSeek-V3 (id 120): base FP8 805GB, NVFP4 variant 403GB.
  const m = MODELS_DATA.find(x => x.id === 120);
  app.setGpuType('B200-192GB');
  const r = modelFitsGPU(m, 4);            // 4x B200 = 728GB: base 805 no, NVFP4 403 yes
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
