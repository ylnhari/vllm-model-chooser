// Canonical precision normalizer — the ONE source of truth for mapping any
// precision string to a canonical format name. Imported by the Node scripts
// (sync-data.mjs, factcheck.mjs). app.js keeps a byte-identical copy because it
// runs as a classic browser <script> (no ESM import); tests/logic.test.mjs
// guards the two against drift.
//
// PRIORITY ORDER MATTERS: specific formats (NVFP4, MXFP4, MXFP8) MUST be checked
// before generic ones (FP8, FP4) so e.g. "MXFP8" does not match "FP8" and
// "FP4+FP8" does not match "FP8".
export function normalizePrec(prec) {
  const upper = (prec || '').toUpperCase();
  if (!upper) return null;
  if (upper.includes('NVFP4')) return 'NVFP4';
  if (upper.includes('MXFP4')) return 'MXFP4';
  if (upper.includes('MXFP8')) return 'MXFP8';
  if (upper === 'FP4+FP8' || (upper.includes('FP4') && !upper.includes('FP8'))) return 'NVFP4';
  if (upper.includes('FP8')) return 'FP8';
  if (upper === 'BF16') return 'BF16';
  if (upper.includes('INT8') || upper.includes('W8A8')) return 'INT8';
  if (upper.includes('INT4') || upper.includes('AWQ') || upper.includes('GPTQ') || upper.includes('W4A16')) return 'INT4';
  return null;
}

// Weight-only VRAM bytes-per-parameter by canonical precision.
//   BF16 ×2 · FP8/INT8/MXFP8 ×1 · INT4/NVFP4/MXFP4 ×0.5
export const BYTES_PER_PARAM = { BF16: 2, FP8: 1, INT8: 1, INT4: 0.5, NVFP4: 0.5, MXFP4: 0.5, MXFP8: 1 };
