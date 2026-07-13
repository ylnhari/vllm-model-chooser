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
// `vram` is the PHYSICAL board capacity from NVIDIA datasheets. There is deliberately
// no `usableVram` here: the usable budget is physical x gpu-memory-utilization, and
// that utilization is a runtime user control (mirroring vLLM's --gpu-memory-utilization),
// so baking one value in would give the app a second, drifting source of truth.
// Quant support is tri-state: 'native' | 'sw'.
const GPU_CONFIG = {
  'L4-24GB':    { vram: 24,  architecture: 'Ada Lovelace', sm: 'sm_89',  memory: 'GDDR6', name: 'L4 24GB' },
  'A100-80GB':  { vram: 80,  architecture: 'Ampere',       sm: 'sm_80',  memory: 'HBM2e', name: 'A100 80GB' },
  'H100-80GB':  { vram: 80,  architecture: 'Hopper',       sm: 'sm_90',  memory: 'HBM3',  name: 'H100 80GB' },
  'H200-141GB': { vram: 141, architecture: 'Hopper',       sm: 'sm_90',  memory: 'HBM3e', name: 'H200 141GB' },
  'B100-192GB': { vram: 192, architecture: 'Blackwell',    sm: 'sm_100', memory: 'HBM3e', name: 'B100 192GB' },
  'B200-192GB': { vram: 192, architecture: 'Blackwell',    sm: 'sm_100', memory: 'HBM3e', name: 'B200 192GB' },
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
      `hf_url: ${JSON.stringify(m.hf_url)}`,
    ];
    // KV geometry (bytes/token, FP16) from the HF config — see kvBytesPerTokenFromConfig.
    //   KV(tokens) = kvBytesPerToken * tokens + kvSlidingBytesPerToken * min(tokens, kvWindow)
    // kvSource records provenance so the UI can mark estimated figures.
    if (m.kvBytesPerToken != null) fields.push(`kvBytesPerToken: ${m.kvBytesPerToken}`);
    if (m.kvSlidingBytesPerToken) fields.push(`kvSlidingBytesPerToken: ${m.kvSlidingBytesPerToken}`);
    if (m.kvWindow) fields.push(`kvWindow: ${m.kvWindow}`);
    if (m.kvUpperBound) fields.push(`kvUpperBound: true`);
    if (m.kvSource) fields.push(`kvSource: ${JSON.stringify(m.kvSource)}`);
    // recipe_id is set only when the recipe path casing differs from hf_url
    // (e.g. HuggingFace "google/..." vs recipe "Google/..."); the recipe link
    // is case-sensitive, the HuggingFace link uses hf_url.
    if (m.recipe_id) fields.push(`recipe_id: ${JSON.stringify(m.recipe_id)}`);
    return `    { ${fields.join(', ')} },`;
  };

  const header = `// AUTO-GENERATED by scripts/sync-data.mjs — do not hand-edit data sourced from
// vLLM recipes (contextLength, variant precisions, recipe_id) or from HF configs
// (kv* fields). Curated fields (type) are preserved across syncs. Weight-only vram
// is COMPUTED (totalParams x bytes-per-param) for pure precisions; mixed-precision
// values (MXFP4/compound like gpt-oss, "FP4+FP8", "AMD-FP8") stay curated.
// Last sync: ${new Date().toISOString().slice(0, 10)}.
//
// SOURCES OF TRUTH (see README "Data Sources" / AGENTS.md):
//   model specs (params, context, arch) ........ recipes.vllm.ai/<id>.json -> .model.*
//   variant precisions + per-model VRAM hints ... same .variants.* (vram is KV-INCLUSIVE)
//   quant x GPU compatibility (GPU_QUANT_COMPAT)  vLLM quantization "supported hardware"
//                                                 docs + recipe variant descriptions
//   GPU hardware specs .......................... NVIDIA datasheets / recipe hardware_profile
//   KV geometry (kv* fields) ................... each model's HuggingFace config.json
//
// vram is WEIGHT-ONLY (no KV cache). GPU_CONFIG carries PHYSICAL vram only — the
// usable budget is physical x gpu-memory-utilization, a runtime user control that
// mirrors vLLM's --gpu-memory-utilization, so it is deliberately NOT baked in here.
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

// KV-cache geometry from the model's HuggingFace config.json. FP16 KV (2 bytes/elem).
// Returns { full, sliding, window } in BYTES PER TOKEN, where the app computes
//
//     KV(tokens) = full * tokens + sliding * min(tokens, window)
//
// Sliding-window layers cannot be folded into a single bytes-per-token constant:
// their cache is CAPPED at `window` tokens no matter how long the context is. Gemma-3
// is the case that forces this — 62 layers, sliding_window 1024, sliding_window_pattern 6
// (5 local : 1 global), and NO layer_types array. Counting all 62 layers at full context
// overestimates its 128K KV by ~6x (67 GB vs ~11 GB actual).
//
// Per-layer KV bytes:
//   - MLA (DeepSeek): one compressed latent (kv_lora_rank + rope) — NOT heads x head_dim, NOT x2
//   - GQA/MHA: 2 (K+V) x num_kv_heads x head_dim x 2 bytes
// Layer classification (which layers cache, and how):
//   - hybrid_override_pattern / layers_block_type: Mamba layers hold NO KV — excluded
//   - layer_types: 'full_attention' vs 'sliding_attention'
//   - sliding_window + sliding_window_pattern (Gemma-3 style, no layer_types):
//     every Nth layer is global, the rest are sliding
//   - sliding_window alone: every layer is sliding
// Returns null when geometry is unavailable (gated repo / missing fields).
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

// GQA/MHA per-layer KV bytes per token (K + V, FP16), optionally for the model's
// separate sliding-window head geometry (`swa_*`, e.g. MiMo-V2).
function gqaBytes(c, swa = false) {
  const p = swa ? 'swa_' : '';
  const nHeads = c[`${p}num_attention_heads`] ?? c.num_attention_heads ?? c.n_heads;
  const kvHeads = c[`${p}num_key_value_heads`] ?? (swa ? null : (c.num_key_value_heads ?? c.n_kv_heads ?? nHeads));
  let kDim = c[`${p}head_dim`] ?? (swa ? null : (c.head_dim ?? c.attention_head_dim));
  if (!swa && !kDim && (c.hidden_size || c.dim) && nHeads) kDim = (c.hidden_size || c.dim) / nHeads;
  if (!kvHeads || !kDim) return null;
  const vDim = c[`${p}v_head_dim`] ?? c.v_head_dim ?? kDim;   // K and V dims can differ
  return (kvHeads * kDim + kvHeads * vDim) * 2;              // FP16
}

// MLA (DeepSeek): the cache is ONE compressed latent + a rope key per token per layer —
// not heads x head_dim, and not doubled for K/V. V3 spells the latent as `kv_lora_rank`;
// V4 expresses the same shape as num_key_value_heads:1 + head_dim:512 + qk_rope_head_dim.
function mlaBytes(c) {
  const latent = c.kv_lora_rank ?? ((c.num_key_value_heads === 1 && c.qk_rope_head_dim) ? c.head_dim : null);
  if (!latent) return null;
  return (latent + (c.qk_rope_head_dim || 0)) * 2;
}

// Does this config use SPARSE attention (DeepSeek DSA "lightning indexer", index_topk)?
// Critical distinction: sparse attention changes which tokens you ATTEND TO, not which
// you STORE — the full KV cache is still kept so the indexer has something to select
// from. Such configs may also carry a `sliding_window`, but it does NOT cap the cache.
// Treating it as a cap made DeepSeek-V4 report ~0 GB of KV at a 1M context.
function isSparseAttention(c) {
  return c.index_topk != null || c.index_n_heads != null;
}

// Split layers into KV-bearing { full, sliding } counts, or null if we don't confidently
// recognise the scheme. Layers holding no KV (Mamba/linear blocks) are excluded from both.
function classifyLayers(c) {
  // Mamba/attention hybrids: only the '*' (attention) entries cache KV.
  if (typeof c.hybrid_override_pattern === 'string') {
    const attn = (c.hybrid_override_pattern.match(/\*/g) || []).length;
    if (attn > 0) return { full: attn, sliding: 0 };
  }
  if (Array.isArray(c.layers_block_type)) {                            // Nemotron-H hybrid
    const attn = c.layers_block_type.filter(t => String(t).toLowerCase().includes('attention')).length;
    if (attn > 0) return { full: attn, sliding: 0 };
  }
  // MiMo-V2 style: 0 = full attention, 1 = sliding-window (with its own swa_* geometry).
  if (Array.isArray(c.hybrid_layer_pattern)) {
    const sliding = c.hybrid_layer_pattern.filter(x => Number(x) === 1).length;
    const full = c.hybrid_layer_pattern.filter(x => Number(x) === 0).length;
    if (full + sliding > 0) return { full, sliding, swaGeom: true };
  }
  // Explicit per-layer types (Gemma-3 newer configs, Qwen3-Next, …).
  if (Array.isArray(c.layer_types)) {
    const full = c.layer_types.filter(t => String(t).includes('full')).length;
    const sliding = c.layer_types.filter(t => String(t).includes('sliding')).length;
    if (full + sliding > 0) return { full, sliding };
  }
  const layers = c.num_hidden_layers ?? c.n_layers;
  if (!layers) return null;

  // A declared `sliding_window` only caps the cache if it is actually IN EFFECT:
  //  - Qwen2/Qwen2.5 declare it but gate it behind `use_sliding_window: false` — the window
  //    is disabled and the model is plain full attention. Honouring it capped Qwen2.5-VL-7B
  //    at its 32K window against a real 128K context: a 4x UNDER-count.
  //  - Sparse-attention models (DeepSeek DSA) keep the full cache regardless (see above).
  const windowActive = c.use_sliding_window !== false && !isSparseAttention(c) && c.sliding_window;
  if (windowActive) {
    const pattern = c.sliding_window_pattern;
    if (pattern && pattern > 1) {                    // Gemma-3: every Nth layer is global
      const full = Math.floor(layers / pattern);
      return { full, sliding: layers - full };
    }
    return { full: 0, sliding: layers };             // pure SWA (Mistral/Voxtral family)
  }
  return { full: layers, sliding: 0 };
}

// Returns { full, sliding, window, upperBound } in BYTES PER TOKEN (FP16 KV), or null.
//   KV(tokens) = full * tokens + sliding * min(tokens, window)
// `upperBound` marks a deliberately CONSERVATIVE result: we could not model the
// architecture's cache exactly, so every layer is counted as full attention. Over-counting
// merely hides a usable model; UNDER-counting makes a model claim to fit on GPUs it does
// not — always fail toward the upper bound.
function kvBytesPerTokenFromConfig(rawCfg) {
  const c = descendToAttn(rawCfg);
  if (!c) return null;

  const split = classifyLayers(c);
  if (!split || (split.full + split.sliding) === 0) return null;

  const mla = mlaBytes(c);
  const fullPerLayer = mla ?? gqaBytes(c);
  if (fullPerLayer == null) return null;

  // Sliding layers may carry their own head geometry (MiMo swa_*); fall back to the
  // full-attention geometry when they don't.
  const slidingPerLayer = (split.swaGeom ? gqaBytes(c, true) : null) ?? fullPerLayer;

  const window = c.sliding_window || 0;
  // Sliding layers with no window to cap them are unbounded — count them as full
  // attention rather than silently under-counting.
  if (split.sliding > 0 && !window) {
    return { full: Math.round(split.full * fullPerLayer + split.sliding * slidingPerLayer), sliding: 0, window: 0, upperBound: true };
  }
  return {
    full: Math.round(split.full * fullPerLayer),
    sliding: Math.round(split.sliding * slidingPerLayer),
    window: split.sliding > 0 ? window : 0,
    upperBound: false,
  };
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

// Resolves KV geometry + its provenance. kvSource is surfaced in the UI so an
// estimated figure is never presented with the same confidence as a real config:
//   'config'   — read from the model's own HF config.json / params.json
//   'mirror'   — read from an ungated same-architecture mirror repo (gated original)
//   'estimate' — hand-sourced geometry, no fetchable config anywhere (cited below)
//   'none'     — model has NO autoregressive KV cache (single-pass diffusion/generative)
async function fetchConfigKV(hfUrl) {
  let kv = await fetchRepoKV(hfUrl);
  if (kv) return { kv, kvSource: 'config' };

  const mirror = HF_CONFIG_MIRROR[hfUrl];
  if (mirror) {
    kv = await fetchRepoKV(mirror);
    if (kv) return { kv, kvSource: 'mirror' };
  }

  if (hfUrl in KV_GEOMETRY_OVERRIDES) {
    const o = KV_GEOMETRY_OVERRIDES[hfUrl];
    if (o === 0) return { kv: { full: 0, sliding: 0, window: 0, upperBound: false }, kvSource: 'none' };
    return {
      kv: { full: Math.round(o.layers * 2 * o.kvHeads * o.headDim * 2), sliding: 0, window: 0, upperBound: false },
      kvSource: 'estimate',
    };
  }
  return null;
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
      const got = await fetchConfigKV(m.hf_url);
      if (got) {
        // Normalise both sides: the emitter omits zero-valued kvSliding/kvWindow, so a
        // round-tripped model reads them back as `undefined`. Comparing raw would report
        // a change on every re-sync and make the generator look non-idempotent.
        const sig = (mm) => `${mm.kvBytesPerToken ?? 'x'}/${mm.kvSlidingBytesPerToken || 0}/${mm.kvWindow || 0}/${mm.kvUpperBound || false}/${mm.kvSource || 'x'}`;
        const before = sig(m);
        m.kvBytesPerToken = got.kv.full;
        m.kvSlidingBytesPerToken = got.kv.sliding || 0;
        m.kvWindow = got.kv.window || 0;
        m.kvUpperBound = got.kv.upperBound || false;
        m.kvSource = got.kvSource;
        const after = sig(m);
        if (before !== after) changes.kv.push({ name: m.name, to: after, src: got.kvSource });
      }

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
  console.log(`  KV geometry changed (full/sliding/window): ${changes.kv.length}`);
  changes.kv.forEach(c => console.log(`    - ${c.name}: ${c.to} [${c.src}]`));
  console.log(`  recipe_id (case) set: ${changes.recipeIds.length}`);
  changes.recipeIds.forEach(c => console.log(`    - ${c}`));

  if (dryRun) { console.log('\n(dry run — data.js not written)'); return; }
  fs.writeFileSync(DATA, out);
  console.log(`\nWrote ${DATA}`);
}

main().catch(e => { console.error(e); process.exitCode = 1; });
