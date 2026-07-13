// Auto-sync generator: rebuilds data.js from the live vLLM recipes source of
// truth (https://recipes.vllm.ai) while preserving curated fields that recipes
// don't carry (benchmarks, weight-only VRAM, type, tested flag).
//
//   node scripts/sync-data.mjs            # fetch + rewrite data.js, print a diff summary
//   node scripts/sync-data.mjs --dry-run  # report changes without writing
//
// Objectively-sourced fields that get reconciled from recipes:
//   - contextLength      (recipe model.context_length, when > 0)
//   - variants           (adds canonical precisions present in the recipe but missing here)
// Curated fields preserved as-is: benchmark, type, vram (weight-only), tested, notes.
//
// Network access required. No external dependencies.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { normalizePrec, BYTES_PER_PARAM } from '../shared/prec.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, '..');
const INDEX = path.join(ROOT, 'index.html');
const DATA = path.join(ROOT, 'data.js');
const BASE = 'https://recipes.vllm.ai';
const dryRun = process.argv.includes('--dry-run');

// Corrected, hand-maintained GPU tables (not sourced from recipes).
// usableVram = floor(vram * 0.95). Quant support is tri-state: 'native' | 'sw'.
const GPU_CONFIG = {
  'L4-24GB':    { vram: 24,  usableVram: 22,  architecture: 'Ada Lovelace', sm: 'sm_89',  memory: 'GDDR6', name: 'L4 24GB' },
  'A100-80GB':  { vram: 80,  usableVram: 76,  architecture: 'Ampere',       sm: 'sm_80',  memory: 'HBM2e', name: 'A100 80GB' },
  'H100-80GB':  { vram: 80,  usableVram: 76,  architecture: 'Hopper',       sm: 'sm_90',  memory: 'HBM3',  name: 'H100 80GB' },
  'H200-141GB': { vram: 141, usableVram: 133, architecture: 'Hopper',       sm: 'sm_90',  memory: 'HBM3e', name: 'H200 141GB' },
  'B100-192GB': { vram: 192, usableVram: 182, architecture: 'Blackwell',    sm: 'sm_100', memory: 'HBM3e', name: 'B100 192GB' },
  'B200-192GB': { vram: 192, usableVram: 182, architecture: 'Blackwell',    sm: 'sm_100', memory: 'HBM3e', name: 'B200 192GB' },
};

// Tri-state quantization support. Present key = supported; value = how.
//  'native' = hardware tensor cores; 'sw' = vLLM software path (dequant/Marlin, no speedup).
//  Absent = unsupported.
// Grounded in vLLM recipes (the source of truth this app is built from):
//  - NVFP4 is gated to Blackwell ("NVFP4 quantized weights for Blackwell GPUs"; the
//    recipe site shows H100 as "requires NVIDIA Blackwell"). Blackwell native ONLY.
//  - MXFP4 (OCP microscaling, e.g. gpt-oss) runs broadly via software dequant — the
//    gpt-oss-120b recipe states it "fits on 1xA100 80GB" — and is native on Blackwell.
//  - FP8 is native on Ada/Hopper/Blackwell, software (Marlin) on Ampere.
//  - MXFP8 is native on Blackwell MX tensor cores ("Blackwell ... for native MX
//    tensor core"), unsupported elsewhere.
//  - INT8 (W8A8) runs on native INT8 tensor cores on every arch here — all are
//    Ampere+ (sm_80+) and have IMMA/int8 tensor cores; vLLM's CUTLASS int8 scaled_mm
//    is the native path. INT4/AWQ/GPTQ (W4A16) differ: weights are dequantised to
//    FP16 for compute (Marlin), so they stay 'sw' (no int4 tensor-core math).
const GPU_QUANT_COMPAT = {
  'L4-24GB':    { BF16: 'native', FP8: 'native', INT8: 'native', INT4: 'sw', MXFP4: 'sw' },
  'A100-80GB':  { BF16: 'native', FP8: 'sw',     INT8: 'native', INT4: 'sw', MXFP4: 'sw' },
  'H100-80GB':  { BF16: 'native', FP8: 'native', INT8: 'native', INT4: 'sw', MXFP4: 'sw' },
  'H200-141GB': { BF16: 'native', FP8: 'native', INT8: 'native', INT4: 'sw', MXFP4: 'sw' },
  'B100-192GB': { BF16: 'native', FP8: 'native', INT8: 'native', INT4: 'sw', NVFP4: 'native', MXFP4: 'native', MXFP8: 'native' },
  'B200-192GB': { BF16: 'native', FP8: 'native', INT8: 'native', INT4: 'sw', NVFP4: 'native', MXFP4: 'native', MXFP8: 'native' },
};

function loadCurrentModels() {
  // Prefer existing data.js if present, else extract from index.html.
  let src;
  if (fs.existsSync(DATA)) src = fs.readFileSync(DATA, 'utf8');
  else src = fs.readFileSync(INDEX, 'utf8');
  const m = src.match(/const MODELS_DATA = (\[[\s\S]*?\]);/);
  if (!m) throw new Error('Could not locate MODELS_DATA');
  return eval(m[1]);
}

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  return res.json();
}

function emit(models) {
  const fmtVariant = v => {
    const parts = [`prec: ${JSON.stringify(v.prec)}`, `vram: ${v.vram}`];
    if (v.note) parts.push(`note: ${JSON.stringify(v.note)}`);
    return `{${parts.join(', ')}}`;
  };
  const fmtModel = m => {
    const variants = `[${(m.variants || []).map(fmtVariant).join(', ')}]`;
    const bench = JSON.stringify(m.benchmark || {});
    const fields = [
      `id: ${m.id}`,
      `name: ${JSON.stringify(m.name)}`,
      `provider: ${JSON.stringify(m.provider)}`,
      `params: ${JSON.stringify(m.params)}`,
      `totalParams: ${m.totalParams}`,
      `activeParams: ${m.activeParams}`,
      `prec: ${JSON.stringify(m.prec)}`,
      `vram: ${m.vram}`,
      `variants: ${variants}`,
      `type: ${JSON.stringify(m.type)}`,
      `contextLength: ${m.contextLength}`,
      `benchmark: ${bench}`,
      `hf_url: ${JSON.stringify(m.hf_url)}`,
    ];
    // Weight-only-KV footprint per token (bytes, FP16), from HF attention geometry.
    if (m.kvBytesPerToken) fields.push(`kvBytesPerToken: ${m.kvBytesPerToken}`);
    // recipe_id is set only when the recipe path casing differs from hf_url
    // (e.g. HuggingFace "google/..." vs recipe "Google/..."); the recipe link
    // is case-sensitive, the HuggingFace link uses hf_url.
    if (m.recipe_id) fields.push(`recipe_id: ${JSON.stringify(m.recipe_id)}`);
    if (m.tested) fields.push(`tested: true`);
    return `    { ${fields.join(', ')} },`;
  };

  const header = `// AUTO-GENERATED by scripts/sync-data.mjs — do not hand-edit data sourced from
// vLLM recipes (contextLength, variant precisions, recipe_id). Curated fields
// (benchmark, type, tested) are preserved across syncs. Weight-only vram is
// COMPUTED (totalParams x bytes-per-param) for pure precisions; mixed-precision
// values (MXFP4/compound like gpt-oss, "FP4+FP8", "AMD-FP8") stay curated.
// Last sync: ${new Date().toISOString().slice(0, 10)}.
//
// SOURCES OF TRUTH (see README "Data Sources" / AGENTS.md):
//   model specs (params, context, arch) ........ recipes.vllm.ai/<id>.json -> .model.*
//   variant precisions + per-model VRAM hints ... same .variants.* (vram is KV-INCLUSIVE)
//   quant x GPU compatibility (GPU_QUANT_COMPAT)  vLLM quantization "supported hardware"
//                                                 docs + recipe variant descriptions
//   GPU hardware specs .......................... NVIDIA datasheets / recipe hardware_profile
// VRAM values here are WEIGHT-ONLY (no KV cache). usableVram = floor(physical * 0.95).
// GPU_QUANT_COMPAT is tri-state: 'native' (HW tensor cores) | 'sw' (vLLM software path).
`;
  const gpuCfg = 'const GPU_CONFIG = ' + JSON.stringify(GPU_CONFIG, null, 4) + ';\n';
  const quant = 'const GPU_QUANT_COMPAT = ' + JSON.stringify(GPU_QUANT_COMPAT, null, 4) + ';\n';
  const data = 'const MODELS_DATA = [\n' + models.map(fmtModel).join('\n') + '\n];\n';
  return `${header}\n${gpuCfg}\n${quant}\n${data}`;
}

// Weight-only VRAM is COMPUTED (totalParams x bytes-per-param), not curated, for
// uniformly-quantised "pure" precisions. Mixed / compound precisions where the naive
// formula is wrong (MXFP4 = 4-bit experts + higher-precision rest e.g. gpt-oss;
// "FP4+FP8" compound; "AMD-FP8" = MXFP4 weights + FP8 acts) keep their curated value.
const PURE_PRECISIONS = new Set(['BF16', 'FP8', 'INT8', 'INT4', 'NVFP4']);
function pureWeightVram(totalParams, prec) {
  const raw = (prec || '').toUpperCase();
  const canon = normalizePrec(prec);
  if (!PURE_PRECISIONS.has(canon)) return null;                 // mixed/unknown → leave curated
  if (raw.includes('+') || raw.includes('AMD') || raw.includes('MX')) return null; // compound/mixed
  const bpp = BYTES_PER_PARAM[canon];
  if (bpp == null) return null;
  return Math.max(1, Math.round(totalParams * bpp));
}

// Exact-ish KV-cache footprint per token (bytes), computed from the model's
// HuggingFace config.json attention geometry. FP16 KV assumed (2 bytes/elem).
//   KV/token = (full-attention layers) x 2 (K+V) x num_kv_heads x head_dim x 2
// Handles the three cases that matter for these models:
//   - hybrid linear/full attention (layer_types): only 'full_attention' layers cache KV
//   - GQA/MHA: num_key_value_heads x head_dim (exact)
//   - MLA (DeepSeek): approximated via the config's compressed kv-head dims
// Returns null when geometry is unavailable (gated repo / missing fields) so the
// app falls back to its coarse params-based proxy. VL/multimodal configs nest the
// text stack under text_config/language_config/llm_config — unwrap that first.
function kvBytesPerTokenFromConfig(rawCfg) {
  const c = (rawCfg && (rawCfg.text_config || rawCfg.language_config || rawCfg.llm_config)) || rawCfg || {};
  const layersAll = c.num_hidden_layers ?? c.num_layers;
  if (!layersAll) return null;
  let layers = layersAll;
  if (Array.isArray(c.layer_types)) {
    const full = c.layer_types.filter(t => String(t).includes('full')).length;
    if (full > 0) layers = full;                 // hybrid: linear-attn layers hold no KV cache
  }
  // MLA (DeepSeek-style): the cache is a single compressed latent (kv_lora_rank) plus
  // the rope key per token per layer — NOT num_heads × head_dim, and NOT doubled for
  // K/V (the latent is shared). Using the GQA formula here overcounts ~20×.
  if (c.kv_lora_rank) {
    return Math.round(layers * (c.kv_lora_rank + (c.qk_rope_head_dim || 0)) * 2);
  }
  // GQA / MHA: separate K and V, each num_kv_heads × head_dim, FP16.
  const nHeads = c.num_attention_heads;
  const kvHeads = c.num_key_value_heads ?? nHeads;
  let headDim = c.head_dim;
  if (!headDim && c.hidden_size && nHeads) headDim = c.hidden_size / nHeads;
  if (!kvHeads || !headDim) return null;
  return Math.round(layers * 2 * kvHeads * headDim * 2);
}

async function fetchConfigKV(hfUrl) {
  try {
    const res = await fetch(`https://huggingface.co/${hfUrl}/resolve/main/config.json`);
    if (!res.ok) return null;                     // 401 gated / 404 → fall back to proxy
    return kvBytesPerTokenFromConfig(await res.json());
  } catch { return null; }
}

async function main() {
  const models = loadCurrentModels();
  const inventory = await fetchJson(`${BASE}/models.json`);
  const byId = new Map(inventory.map(r => [r.hf_id.toLowerCase(), r]));

  const changes = { context: [], variants: [], gpuFeasibilityRemoved: 0, recipeIds: [], vram: [], kv: [] };

  const queue = [...models];
  async function worker() {
    while (queue.length) {
      const m = queue.shift();
      if ('gpuFeasibility' in m) { delete m.gpuFeasibility; changes.gpuFeasibilityRemoved++; }
      const entry = byId.get((m.hf_url || '').toLowerCase());
      if (!entry) continue;
      // Capture the exact recipe-path casing when it differs from hf_url so the
      // (case-sensitive) recipe link doesn't 404 (e.g. "Google/" vs "google/").
      if (entry.hf_id !== m.hf_url) { m.recipe_id = entry.hf_id; changes.recipeIds.push(`${m.hf_url} -> ${entry.hf_id}`); }
      else if ('recipe_id' in m) delete m.recipe_id;
      // KV-cache geometry from HuggingFace config.json (best-effort; null on gated/missing).
      const kv = await fetchConfigKV(m.hf_url);
      if (kv && kv !== m.kvBytesPerToken) { changes.kv.push({ name: m.name, to: kv }); m.kvBytesPerToken = kv; }
      else if (!kv && 'kvBytesPerToken' in m) { /* keep last-known geometry if config now unavailable */ }

      let j;
      try { j = await fetchJson(BASE + entry.json); } catch { continue; }
      const mod = j.model || {};

      // 1) context length
      if (mod.context_length && mod.context_length !== 0 && m.contextLength !== mod.context_length) {
        changes.context.push({ name: m.name, from: m.contextLength, to: mod.context_length });
        m.contextLength = mod.context_length;
      }

      // 2) add canonical variant precisions present in recipe but missing here
      const have = new Set([normalizePrec(m.prec), ...(m.variants || []).map(v => normalizePrec(v.prec))]);
      const recipePrecs = [...new Set(Object.values(j.variants || {})
        .map(v => normalizePrec(v.precision)).filter(Boolean))];
      for (const canon of recipePrecs) {
        if (have.has(canon)) continue;
        const bpp = BYTES_PER_PARAM[canon];
        if (bpp == null) continue;                 // can't compute weight-only VRAM → skip
        const vram = Math.max(1, Math.round(m.totalParams * bpp));
        m.variants = m.variants || [];
        m.variants.push({ prec: canon, vram, note: 'synced from vLLM recipe' });
        have.add(canon);
        changes.variants.push({ name: m.name, prec: canon, vram });
      }
    }
  }
  await Promise.all(Array.from({ length: 8 }, worker));

  // 3) normalise weight-only VRAM to the pure formula (base + variants). Strips any
  // serving-headroom multiplier that leaked into the curated values so the field is
  // consistently weight-only (the KV-cache footprint is a separate app-side estimate).
  for (const m of models) {
    const bv = pureWeightVram(m.totalParams, m.prec);
    if (bv != null && bv !== m.vram) { changes.vram.push({ name: m.name, prec: m.prec, from: m.vram, to: bv }); m.vram = bv; }
    for (const v of m.variants || []) {
      if (v.vram == null) continue;
      const vv = pureWeightVram(m.totalParams, v.prec);
      if (vv != null && vv !== v.vram) { changes.vram.push({ name: `${m.name} [${v.prec}]`, prec: v.prec, from: v.vram, to: vv }); v.vram = vv; }
    }
  }

  models.sort((a, b) => a.id - b.id);
  const out = emit(models);

  console.log(`Reconciled ${models.length} models against ${inventory.length} recipes.`);
  console.log(`  gpuFeasibility removed: ${changes.gpuFeasibilityRemoved}`);
  console.log(`  context lengths updated: ${changes.context.length}`);
  changes.context.forEach(c => console.log(`    - ${c.name}: ${c.from} → ${c.to}`));
  console.log(`  variants added: ${changes.variants.length}`);
  changes.variants.forEach(c => console.log(`    - ${c.name}: +${c.prec} @${c.vram}GB`));
  console.log(`  weight-only VRAM normalised: ${changes.vram.length}`);
  changes.vram.forEach(c => console.log(`    - ${c.name}: ${c.from} → ${c.to}GB`));
  console.log(`  KV geometry set (bytes/token): ${changes.kv.length}`);
  console.log(`  recipe_id (case) set: ${changes.recipeIds.length}`);
  changes.recipeIds.forEach(c => console.log(`    - ${c}`));

  if (dryRun) { console.log('\n(dry run — data.js not written)'); return; }
  fs.writeFileSync(DATA, out);
  console.log(`\nWrote ${DATA}`);
}

main().catch(e => { console.error(e); process.exitCode = 1; });
