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
    // KV footprint per token (bytes, FP16) from HF attention geometry; 0 = no
    // autoregressive KV cache (diffusion/generative). Absent = geometry unavailable.
    if (m.kvBytesPerToken != null) fields.push(`kvBytesPerToken: ${m.kvBytesPerToken}`);
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
// Descend through nested wrapper configs (multimodal/omni models bury the text
// decoder under text_config / thinker_config / …) to the block carrying attention geometry.
function descendToAttn(cfg) {
  const WRAP = ['text_config', 'language_config', 'llm_config', 'thinker_config', 'decoder', 'llm'];
  let c = cfg || {};
  for (let i = 0; i < 5; i++) {
    const geom = c && (c.num_hidden_layers || c.n_layers || c.hybrid_override_pattern) && (c.num_attention_heads || c.n_heads);
    if (geom) return c;
    let next = null;
    for (const k of WRAP) { if (c && typeof c[k] === 'object' && c[k]) { next = c[k]; break; } }
    if (!next) break;
    c = next;
  }
  return c;
}

function kvBytesPerTokenFromConfig(rawCfg) {
  const c = descendToAttn(rawCfg);
  if (!c) return null;
  // Layer count: for Mamba/attention hybrids (Nemotron-H) the KV-bearing layers are
  // the '*' entries of hybrid_override_pattern; else count full_attention layer_types;
  // else num_hidden_layers / n_layers (Mistral params.json).
  let layers = null;
  if (typeof c.hybrid_override_pattern === 'string') {
    const attn = (c.hybrid_override_pattern.match(/\*/g) || []).length;
    if (attn > 0) layers = attn;
  }
  if (layers == null && Array.isArray(c.layers_block_type)) {          // Nemotron-H hybrid
    const attn = c.layers_block_type.filter(t => String(t).toLowerCase().includes('attention')).length;
    if (attn > 0) layers = attn;
  }
  if (layers == null && Array.isArray(c.layer_types)) {
    const full = c.layer_types.filter(t => String(t).includes('full')).length;
    if (full > 0) layers = full;
  }
  if (layers == null) layers = c.num_hidden_layers ?? c.n_layers;
  if (!layers) return null;
  // MLA (DeepSeek): a single compressed latent (kv_lora_rank) + rope key per token per
  // layer — NOT num_heads × head_dim, NOT doubled for K/V. GQA formula overcounts ~20×.
  if (c.kv_lora_rank) {
    return Math.round(layers * (c.kv_lora_rank + (c.qk_rope_head_dim || 0)) * 2);
  }
  // GQA / MHA: separate K and V, each num_kv_heads × head_dim, FP16.
  const nHeads = c.num_attention_heads ?? c.n_heads;
  const kvHeads = c.num_key_value_heads ?? c.n_kv_heads ?? nHeads;
  let headDim = c.head_dim ?? c.attention_head_dim;
  if (!headDim && (c.hidden_size || c.dim) && nHeads) headDim = (c.hidden_size || c.dim) / nHeads;
  if (!kvHeads || !headDim) return null;
  return Math.round(layers * 2 * kvHeads * headDim * 2);
}

// Some HF configs embed non-JSON literals (Infinity / NaN in rope-scaling fields).
function tolerantJson(text) {
  try { return JSON.parse(text); }
  catch { return JSON.parse(text.replace(/-?\bInfinity\b/g, '1e309').replace(/\bNaN\b/g, 'null')); }
}

// Ungated config mirrors for gated repos (HF 401): identical architecture, public
// config — lets the KV sync fetch REAL geometry instead of guessing. Verified same-arch.
const HF_CONFIG_MIRROR = {
  'meta-llama/Llama-3.1-8B-Instruct': 'unsloth/Meta-Llama-3.1-8B-Instruct',
  'meta-llama/Llama-3.3-70B-Instruct': 'unsloth/Llama-3.3-70B-Instruct',
  'meta-llama/Llama-4-Scout-17B-16E-Instruct': 'unsloth/Llama-4-Scout-17B-16E-Instruct',
  'google/translategemma-27b-it': 'unsloth/gemma-3-27b-it',   // translategemma = Gemma-3-27B backbone
  'zai-org/GLM-GA': 'zai-org/GLM-4-9B-0414',                  // GLM-GA = GLM-4 9B family
};

// Explicit KV overrides for models with no fetchable config AND no mirror.
//   0  → NO autoregressive KV cache (diffusion/image/video/audio generators, single-pass).
//   { layers, kvHeads, headDim } → hand-sourced GQA geometry (cited), computed like a config.
const KV_GEOMETRY_OVERRIDES = {
  'zai-org/GLM-Image': 0,
  'meituan-longcat/LongCat-Image-Edit': 0,
  'Qwen/Qwen-Image': 0,
  'Wan-AI/Wan2.2-T2V-A14B-Diffusers': 0,
  'stabilityai/stable-diffusion-3.5-medium': 0,
  'stabilityai/stable-audio-open-1.0': 0,
  // PLaMo-3 31B: gated, no public mirror. Estimated from a typical 31B dense GQA layout
  // (48 layers, 8 KV heads, head_dim 128); refine if the config ever opens.
  'pfnet/plamo-3-nict-31b-base': { layers: 48, kvHeads: 8, headDim: 128 },
};

async function fetchRepoKV(repo) {
  const tryFile = async (file) => {
    try { const r = await fetch(`https://huggingface.co/${repo}/resolve/main/${file}`);
      return r.ok ? kvBytesPerTokenFromConfig(tolerantJson(await r.text())) : null;
    } catch { return null; }
  };
  return (await tryFile('config.json')) ?? (await tryFile('params.json')) ?? null;
}

async function fetchConfigKV(hfUrl) {
  // 1) direct config.json / params.json → 2) ungated mirror repo → 3) explicit override.
  let kv = await fetchRepoKV(hfUrl);
  if (kv == null && HF_CONFIG_MIRROR[hfUrl]) kv = await fetchRepoKV(HF_CONFIG_MIRROR[hfUrl]);
  if (kv == null && hfUrl in KV_GEOMETRY_OVERRIDES) {
    const o = KV_GEOMETRY_OVERRIDES[hfUrl];
    kv = (o === 0) ? 0 : Math.round(o.layers * 2 * o.kvHeads * o.headDim * 2);
  }
  return kv;
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
      if (kv != null && kv !== m.kvBytesPerToken) { changes.kv.push({ name: m.name, to: kv }); m.kvBytesPerToken = kv; }

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
