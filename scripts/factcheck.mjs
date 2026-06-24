// Fact-check the app's MODELS_DATA against the live vLLM recipes source of truth
// (https://recipes.vllm.ai). Reports models missing from the inventory and field
// mismatches (context length, parameter counts, variant precisions).
//
//   node scripts/factcheck.mjs            # audit all models
//   node scripts/factcheck.mjs --json     # machine-readable output
//
// Network access required. No external dependencies.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, '..');
const BASE = 'https://recipes.vllm.ai';
const asJson = process.argv.includes('--json');

function loadModels() {
  // MODELS_DATA now lives in data.js; fall back to index.html for older layouts.
  const src = fs.existsSync(path.join(ROOT, 'data.js'))
    ? fs.readFileSync(path.join(ROOT, 'data.js'), 'utf8')
    : fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
  const m = src.match(/const MODELS_DATA = (\[[\s\S]*?\n\]);/) || src.match(/const MODELS_DATA = (\[[\s\S]*?\]);/);
  if (!m) throw new Error('Could not locate MODELS_DATA');
  return eval(m[1]); // trusted local source
}

// Canonical-format normalizer, mirrors normalizePrec() in index.html.
function norm(p) {
  p = (p || '').toUpperCase();
  if (p.includes('NVFP4')) return 'NVFP4';
  if (p.includes('MXFP4')) return 'MXFP4';
  if (p.includes('MXFP8')) return 'MXFP8';
  if (p === 'FP4+FP8' || (p.includes('FP4') && !p.includes('FP8'))) return 'NVFP4';
  if (p.includes('FP8')) return 'FP8';
  if (p === 'BF16') return 'BF16';
  if (p.includes('INT4') || p.includes('AWQ') || p.includes('GPTQ') || p.includes('W4A16')) return 'INT4';
  return p || null;
}

function parseB(s) {
  if (s == null) return null;
  if (typeof s === 'number') return s;
  const m = String(s).match(/([\d.]+)\s*([BMT])?/i);
  if (!m) return null;
  let v = parseFloat(m[1]);
  const u = (m[2] || 'B').toUpperCase();
  if (u === 'T') v *= 1000;
  if (u === 'M') v /= 1000;
  return v;
}

async function fetchJson(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  return res.json();
}

async function main() {
  const app = loadModels();
  const inventory = await fetchJson(`${BASE}/models.json`);
  const byId = new Map(inventory.map(r => [r.hf_id.toLowerCase(), r]));

  const report = { notInInventory: [], context: [], params: [], variants: [] };

  // limited-concurrency fetch of each model's recipe
  const queue = [...app];
  async function worker() {
    while (queue.length) {
      const m = queue.shift();
      const entry = byId.get((m.hf_url || '').toLowerCase());
      if (!entry) { report.notInInventory.push({ id: m.id, hf: m.hf_url }); continue; }
      let j;
      try { j = await fetchJson(BASE + entry.json); }
      catch (e) { report.notInInventory.push({ id: m.id, hf: m.hf_url, err: String(e) }); continue; }

      const mod = j.model || {};
      const rctx = mod.context_length;
      if (rctx != null && rctx !== 0 && (m.contextLength || null) !== rctx) {
        report.context.push({ id: m.id, name: m.name, app: m.contextLength ?? null, recipe: rctx });
      }
      const rp = parseB(mod.parameter_count);
      if (rp != null && Math.abs(rp - m.totalParams) > Math.max(0.5, rp * 0.05)) {
        report.params.push({ id: m.id, name: m.name, app: m.totalParams, recipe: mod.parameter_count });
      }
      const recipePrecs = new Set(Object.values(j.variants || {}).map(v => norm(v.precision)));
      const appPrecs = new Set([norm(m.prec), ...(m.variants || []).map(v => norm(v.prec))]);
      const missing = [...recipePrecs].filter(p => p && !appPrecs.has(p));
      if (missing.length) {
        report.variants.push({ id: m.id, name: m.name, missingInApp: missing,
          recipe: [...recipePrecs].filter(Boolean), app: [...appPrecs].filter(Boolean) });
      }
    }
  }
  await Promise.all(Array.from({ length: 8 }, worker));

  if (asJson) { console.log(JSON.stringify(report, null, 2)); return; }

  const line = (s = '') => console.log(s);
  line(`Audited ${app.length} models against ${inventory.length} live recipes.\n`);
  line(`Not found in recipes inventory: ${report.notInInventory.length}`);
  report.notInInventory.forEach(x => line(`  - id ${x.id} ${x.hf}${x.err ? ' (' + x.err + ')' : ''}`));
  line(`\nContext-length mismatches: ${report.context.length}`);
  report.context.forEach(x => line(`  - id ${x.id} ${x.name}: app=${x.app} recipe=${x.recipe}`));
  line(`\nParameter-count mismatches (>5%): ${report.params.length}`);
  report.params.forEach(x => line(`  - id ${x.id} ${x.name}: app=${x.app}B recipe=${x.recipe}`));
  line(`\nVariant precisions present in recipe but missing in app: ${report.variants.length}`);
  report.variants.forEach(x => line(`  - id ${x.id} ${x.name}: missing [${x.missingInApp.join(', ')}]  (recipe {${x.recipe.join(',')}})`));

  const total = report.notInInventory.length + report.context.length + report.params.length + report.variants.length;
  line(`\nTotal discrepancies: ${total}`);
  process.exitCode = total > 0 ? 1 : 0;
}

main().catch(e => { console.error(e); process.exitCode = 2; });
