// Test harness: evaluates data.js + app.js inside a Node vm with a minimal fake
// DOM/URL so the pure logic functions can be exercised without a browser.
// No external dependencies (Node built-ins only).
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, '..');

// A permissive fake element — every property the app touches is a no-op or
// returns a sane default. `value` is settable so tests can drive the filters.
function makeEl(id) {
  return {
    id,
    value: '',
    textContent: '',
    innerHTML: '',
    dataset: {},
    style: {},
    classList: { add() {}, remove() {}, toggle() {}, contains() { return false; } },
    addEventListener() {},
    appendChild() {},
  };
}

export function loadApp() {
  const data = fs.readFileSync(path.join(ROOT, 'data.js'), 'utf8');
  const code = fs.readFileSync(path.join(ROOT, 'app.js'), 'utf8');

  const els = new Map();
  const document = {
    getElementById(id) {
      if (!els.has(id)) els.set(id, makeEl(id));
      return els.get(id);
    },
    querySelectorAll() { return []; },
    addEventListener() {},
  };

  const sandbox = {
    document,
    console,
    URLSearchParams,
    location: { search: '', pathname: '/' },
    history: { replaceState() {} },
    window: {},
  };
  sandbox.window.document = document;
  vm.createContext(sandbox);

  // data.js first (defines the const tables), then app.js (logic), then expose
  // the in-scope bindings we want to test.
  const exported = [
    'MODELS_DATA', 'GPU_CONFIG', 'GPU_QUANT_COMPAT',
    'normalizePrec', 'isPrecCompatible', 'precSupportLevel',
    'getGPUVRAM', 'estKVCacheGB', 'modelFitsGPU', 'filterModels',
  ];
  const wrapped = `${data}\n${code}\n;globalThis.__app = { ${exported.join(', ')} };`;
  vm.runInContext(wrapped, sandbox);

  return {
    ...sandbox.__app,
    setGpuType(v) { document.getElementById('gpuTypeSelect').value = v; },
    setKVContext(v) { document.getElementById('kvContextSelect').value = String(v); },
    els,
    document,
  };
}
