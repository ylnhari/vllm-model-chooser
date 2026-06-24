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

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, '..');
const INDEX = path.join(ROOT, 'index.html');
const DATA = path.join(ROOT, 'data.js');
const BASE = 'https://recipes.vllm.ai';
const dryRun = process.argv.includes('--dry-run');

// Bytes-per-parameter by canonical precision (weight-only VRAM formula).
const BYTES_PER_PARAM = { BF16: 2, FP8: 1, INT8: 1, INT4: 0.5, NVFP4: 0.5, MXFP4: 0.5, MXFP8: 1 };

// Mirrors normalizePrec() in app.js — keep the two in lockstep.
function normalizePrec(prec) {
  const p = (prec || '').toUpperCase();
  if (!p) return null;
  if (p.includes('NVFP4')) return 'NVFP4';
  if (p.includes('MXFP4')) return 'MXFP4';
  if (p.includes('MXFP8')) return 'MXFP8';
  if (p === 'FP4+FP8' || (p.includes('FP4') && !p.includes('FP8'))) return 'NVFP4';
  if (p.includes('FP8')) return 'FP8';
  if (p === 'BF16') return 'BF16';
  if (p.includes('INT8') || p.includes('W8A8')) return 'INT8';
  if (p.includes('INT4') || p.includes('AWQ') || p.includes('GPTQ') || p.includes('W4A16')) return 'INT4';
  return null;
}

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
const GPU_QUANT_COMPAT = {
  'L4-24GB':    { BF16: 'native', FP8: 'native', INT8: 'sw', INT4: 'sw', MXFP4: 'sw' },
  'A100-80GB':  { BF16: 'native', FP8: 'sw',     INT8: 'sw', INT4: 'sw', MXFP4: 'sw' },
  'H100-80GB':  { BF16: 'native', FP8: 'native', INT8: 'sw', INT4: 'sw', MXFP4: 'sw' },
  'H200-141GB': { BF16: 'native', FP8: 'native', INT8: 'sw', INT4: 'sw', MXFP4: 'sw' },
  'B100-192GB': { BF16: 'native', FP8: 'native', INT8: 'sw', INT4: 'sw', NVFP4: 'native', MXFP4: 'native', MXFP8: 'native' },
  'B200-192GB': { BF16: 'native', FP8: 'native', INT8: 'sw', INT4: 'sw', NVFP4: 'native', MXFP4: 'native', MXFP8: 'native' },
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
    // recipe_id is set only when the recipe path casing differs from hf_url
    // (e.g. HuggingFace "google/..." vs recipe "Google/..."); the recipe link
    // is case-sensitive, the HuggingFace link uses hf_url.
    if (m.recipe_id) fields.push(`recipe_id: ${JSON.stringify(m.recipe_id)}`);
    if (m.tested) fields.push(`tested: true`);
    return `    { ${fields.join(', ')} },`;
  };

  const header = `// AUTO-GENERATED by scripts/sync-data.mjs — do not hand-edit data sourced from
// vLLM recipes (contextLength, variant precisions, recipe_id). Curated fields
// (benchmark, weight-only vram, type, tested) are preserved across syncs.
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

async function main() {
  const models = loadCurrentModels();
  const inventory = await fetchJson(`${BASE}/models.json`);
  const byId = new Map(inventory.map(r => [r.hf_id.toLowerCase(), r]));

  const changes = { context: [], variants: [], gpuFeasibilityRemoved: 0, recipeIds: [] };

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

  models.sort((a, b) => a.id - b.id);
  const out = emit(models);

  console.log(`Reconciled ${models.length} models against ${inventory.length} recipes.`);
  console.log(`  gpuFeasibility removed: ${changes.gpuFeasibilityRemoved}`);
  console.log(`  context lengths updated: ${changes.context.length}`);
  changes.context.forEach(c => console.log(`    - ${c.name}: ${c.from} → ${c.to}`));
  console.log(`  variants added: ${changes.variants.length}`);
  changes.variants.forEach(c => console.log(`    - ${c.name}: +${c.prec} @${c.vram}GB`));
  console.log(`  recipe_id (case) set: ${changes.recipeIds.length}`);
  changes.recipeIds.forEach(c => console.log(`    - ${c}`));

  if (dryRun) { console.log('\n(dry run — data.js not written)'); return; }
  fs.writeFileSync(DATA, out);
  console.log(`\nWrote ${DATA}`);
}

main().catch(e => { console.error(e); process.exitCode = 1; });
