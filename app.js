// vLLM Model Chooser — application logic. Depends on globals from data.js
// (GPU_CONFIG, GPU_QUANT_COMPAT, MODELS_DATA), loaded before this file.

// Maps any precision string to canonical format name. PRIORITY ORDER MATTERS:
// Specific formats (NVFP4, MXFP4, MXFP8) MUST be checked before generic (FP8, FP4)
// to avoid false matches like "MXFP8" matching "FP8" or "FP4+FP8" matching "FP8"
function normalizePrec(prec) {
    if (!prec) return null;
    const upper = prec.toUpperCase();
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

// Support level of a precision on a GPU: 'native' (HW tensor cores), 'sw'
// (vLLM software path — loads but no speedup), or null (unsupported).
// GPU_QUANT_COMPAT is a tri-state map { CANONICAL: 'native' | 'sw' }.
function precSupportLevel(prec, gpuType) {
    const map = GPU_QUANT_COMPAT[gpuType];
    if (!map) return 'native';          // unknown GPU → fail-safe
    const norm = normalizePrec(prec);
    if (!norm) return 'native';         // unknown format → fail-safe
    return map[norm] || null;
}

// Gate: a precision is compatible if it has any support level (native or sw).
function isPrecCompatible(prec, gpuType) {
    return precSupportLevel(prec, gpuType) !== null;
}

// --- Memory budget model ----------------------------------------------------
// Mirrors how vLLM actually accounts for GPU memory (--gpu-memory-utilization):
//
//     budget   = physical VRAM × util          ← vLLM's hard ceiling
//     kv pool  = budget − weights − activation/CUDA-graph reserve
//
// vLLM allocates the KV cache GREEDILY into whatever is left inside the budget after
// weights and activations. So weights + KV must fit under (budget − reserve).
// Docs: https://docs.vllm.ai/en/latest/configuration/engine_args.html
const DEFAULT_MEM_UTIL = 0.95;
const DEFAULT_RESERVE_GB = 2;      // per GPU — an ASSUMPTION, see getReserveGB()

function numFromSelect(id, fallback) {
    const raw = document.getElementById(id)?.value;
    const n = parseFloat(raw);
    return Number.isFinite(n) ? n : fallback;
}

// Fraction of each GPU vLLM may use. vLLM's own default is 0.90–0.92; this app
// defaults to 0.95, which is more optimistic — hence it's a user control, not a
// buried constant.
function getMemUtil() { return numFromSelect('memUtilSelect', DEFAULT_MEM_UTIL); }

// Per-GPU activation + CUDA-graph reserve. IMPORTANT: vLLM does not COMPUTE this —
// it MEASURES it by profiling a real forward pass at startup, and it scales with
// --max-num-batched-tokens and the model's hidden size. Any fixed number here is an
// assumption, which is exactly why it is a visible, user-adjustable control rather
// than a hidden fudge factor.
function getReserveGB() { return numFromSelect('reserveSelect', DEFAULT_RESERVE_GB); }

function getGPUConfig() {
    return GPU_CONFIG[document.getElementById('gpuTypeSelect')?.value || 'L4-24GB'] || {};
}

// Total vLLM memory budget across `gpus` GPUs (weights + KV + activations).
function getGPUBudget(gpus) {
    return gpus * (getGPUConfig().vram || 80) * getMemUtil();
}

// What weights + KV may actually occupy: the budget minus the activation reserve.
// Single source of truth for the capacity check — must exist EXACTLY once.
function getGPUVRAM(gpus) {
    return Math.max(0, getGPUBudget(gpus) - gpus * getReserveGB());
}

// --- KV cache ----------------------------------------------------------------
// KV(tokens) = full × tokens + sliding × min(tokens, window)
//
// Sliding-window layers CANNOT be folded into one bytes-per-token constant: their
// cache is capped at `window` tokens however long the context grows. Gemma-3/4,
// Step-3.7 and others are mostly sliding layers, so treating every layer as full
// attention overestimates their long-context KV by several ×.
// Geometry comes from each model's HF config.json (see scripts/sync-data.mjs);
// `kvSource` records whether it was read from the real config, a same-architecture
// mirror, or hand-estimated, so the UI can mark it.
const KV_GB_PER_PARAM_PER_TOKEN = 3.3e-6;   // GB per (B-params × token) — proxy, only if geometry is missing
const KV_DTYPE_BYTES = { fp16: 2, fp8: 1 };

// vLLM's --kv-cache-dtype. Geometry is stored at FP16 (2 bytes/elem); fp8 halves it.
function getKVDtype() { return document.getElementById('kvDtypeSelect')?.value || 'fp16'; }
function kvDtypeScale() { return (KV_DTYPE_BYTES[getKVDtype()] || 2) / 2; }

function estKVCacheGB(model, tokens) {
    if (!tokens) return 0;
    const scale = kvDtypeScale();
    if (model.kvBytesPerToken == null) {          // geometry unavailable → coarse proxy
        return model.totalParams * tokens * KV_GB_PER_PARAM_PER_TOKEN * scale;
    }
    const full = model.kvBytesPerToken * tokens;
    const win = model.kvWindow || 0;
    const sliding = (model.kvSlidingBytesPerToken || 0) * (win ? Math.min(tokens, win) : 0);
    return (full + sliding) * scale / 1e9;
}

// True when the model caches nothing per token (single-pass diffusion/generative).
function hasNoKVCache(model) { return model.kvSource === 'none'; }

// Selected context length (tokens) for the KV estimate; 0 = estimate off.
function getKVContextTokens() {
    return parseInt(document.getElementById('kvContextSelect')?.value || '0') || 0;
}

// Worst-case concurrent requests: how many sequences, EACH filling the full context,
// fit in the KV pool left after weights. Real serving does better (requests are
// usually shorter than max context, and PagedAttention shares prefix blocks), so this
// is a floor, not a prediction. null when there's no KV or no context selected.
function maxConcurrentRequests(model, weightsGB, gpus, tokens) {
    if (!tokens) return null;
    const perReq = estKVCacheGB(model, tokens);
    if (perReq <= 0) return null;
    return Math.max(0, Math.floor((getGPUVRAM(gpus) - weightsGB) / perReq));
}


let filteredModels = [...MODELS_DATA];
let currentGPUFilter = 1;
let currentQuantFilter = 'all';
let currentTypeFilter = 'all';
let currentContextFilter = 0;
let currentContextLength = 4096;

// Dual check: BOTH VRAM capacity AND quantization format compatibility must pass.
// weights + KV must fit under the usable budget (physical × util − activation reserve).
// Returns the numbers the UI needs to *explain* the verdict, so the card's bar and its
// ✓/✗ can never tell different stories:
//   { fits, variant?, level?, reason?, weights, kv, usable }
// `weights` is the precision actually selected on a fit, or the smallest candidate on a
// miss (i.e. the model's best case — "even its smallest quantization overflows").
function modelFitsGPU(model, gpus) {
    if (gpus === 0) return { fits: true, reason: "Any configuration" };
    const gpuType = document.getElementById('gpuTypeSelect')?.value || 'L4-24GB';
    const usable = getGPUVRAM(gpus);
    const kv = estKVCacheGB(model, getKVContextTokens());
    let vramWouldFit = false;   // did any precision fit on VRAM but get blocked by the quant gate?

    const candidates = [{ prec: model.prec, vram: model.vram, base: true }, ...(model.variants || [])];
    for (const c of candidates) {
        if (!c.vram) continue;
        if (c.vram + kv > usable) continue;
        const level = precSupportLevel(c.prec || model.prec, gpuType);
        if (level === null) { vramWouldFit = true; continue; }   // blocked by quant gate
        const base = { fits: true, level, weights: c.vram, kv, usable };
        return c.base ? base : { ...base, variant: c };
    }
    const smallest = candidates.filter(c => c.vram).sort((a, b) => a.vram - b.vram)[0];
    return {
        fits: false,
        reason: vramWouldFit ? 'quant' : 'vram',
        weights: smallest ? smallest.vram : model.vram,
        kv, usable,
    };
}

function setGPUFilter(n) {
    currentGPUFilter = n;
    document.querySelectorAll('#gpuCountFilter .filter-btn').forEach(btn => {
        btn.classList.toggle('active', parseInt(btn.dataset.value) === n);
    });
    filterModels();
}

function filterModels() {
    const gpuType = document.getElementById('gpuTypeSelect')?.value || 'L4-24GB';
    const quantFilter = document.getElementById('quantFilter')?.value || 'all';
    const typeFilter = document.getElementById('typeFilter')?.value || 'all';
    const contextFilter = parseInt(document.getElementById('contextFilter')?.value || '0');
    const searchQuery = (document.getElementById('searchInput')?.value || '').toLowerCase().trim();

    filteredModels = MODELS_DATA.filter(model => {
        if (currentGPUFilter > 0) {
            const fits = modelFitsGPU(model, currentGPUFilter);
            if (!fits.fits) return false;
        }
        if (quantFilter !== 'all') {
            const norm = normalizePrec(quantFilter);
            const matchesPrec = normalizePrec(model.prec) === norm
                || model.variants.some(v => normalizePrec(v.prec) === norm);
            if (!matchesPrec) return false;
        }
        if (typeFilter !== 'all' && model.type !== typeFilter) return false;
        if (contextFilter > 0 && (model.contextLength || 4096) < contextFilter) return false;
        if (searchQuery && !model.name.toLowerCase().includes(searchQuery) && !model.provider.toLowerCase().includes(searchQuery)) return false;
        return true;
    });
    const sortBy = document.getElementById('sortSelect')?.value || 'params';
    filteredModels.sort((a, b) => {
        switch(sortBy) {
            case 'params': return b.totalParams - a.totalParams;
            case 'vram': return a.vram - b.vram;
            case 'contextLength': return (b.contextLength || 4096) - (a.contextLength || 4096);
            case 'name': return a.name.localeCompare(b.name);
            default: return 0;
        }
    });
    renderModels();
    updateStats();
    syncStateToURL();
}

// Reset every filter/sort/search control back to defaults and re-render.
function resetFilters() {
    setSelect('quantFilter', 'all');
    setSelect('typeFilter', 'all');
    setSelect('contextFilter', '0');
    setSelect('sortSelect', 'params');
    setSelect('kvContextSelect', '0');
    setSelect('kvDtypeSelect', 'fp16');
    setSelect('memUtilSelect', String(DEFAULT_MEM_UTIL));
    setSelect('reserveSelect', String(DEFAULT_RESERVE_GB));
    const search = document.getElementById('searchInput');
    if (search) search.value = '';
    setGPUFilter(1);   // also re-runs filterModels()
}

function setSelect(id, value) {
    const el = document.getElementById(id);
    if (el) el.value = value;
}

// Badge CSS class only — NOT used for compatibility logic (that's isPrecCompatible).
// Specific FP4/FP8 formats are checked before generic ones so e.g. "MXFP8" does
// not fall through to the FP8 badge.
function getQuantBadgeClass(prec) {
    if (prec.includes('NVFP4')) return 'badge-nvfp4';
    if (prec.includes('MXFP8')) return 'badge-mxfp8';
    if (prec.includes('MXFP4')) return 'badge-mxfp4';
    if (prec.includes('FP4')) return 'badge-nvfp4';
    if (prec.includes('FP8')) return 'badge-fp8';
    if (prec.includes('BF16')) return 'badge-bf16';
    if (prec.includes('INT8') || prec.includes('W8A8')) return 'badge-int8';
    if (prec.includes('INT4') || prec.includes('GPTQ') || prec.includes('AWQ') || prec.includes('W4A16') || prec.includes('QAT')) return 'badge-int4';
    return 'badge-bf16';
}

// GB formatter — the budget is now fractional (physical × util − reserve).
function fmtGB(gb) {
    if (gb == null) return '—';
    return gb >= 100 ? Math.round(gb).toString() : gb.toFixed(1).replace(/\.0$/, '');
}

// Does the model fit on ANY supported GPU count? (1..8 — NOT just 1/2/4/8, which
// used to mis-colour models that fit on exactly 3, 5, 6 or 7 GPUs.)
function getVerdictClass(model) {
    for (let gpu = 1; gpu <= 8; gpu++) {
        if (modelFitsGPU(model, gpu).fits) return 'verdict-yes';
    }
    return 'verdict-no';
}

// Human-readable explanation for a fit result, used as a hover tooltip.
function fitTooltip(result, gpus, gpuConfig) {
    const where = `${gpus}× ${gpuConfig.name} (${fmtGB(getGPUVRAM(gpus))}GB usable)`;
    if (result.fits) {
        const via = result.variant ? `${result.variant.prec} variant` : 'base precision';
        const sw = result.level === 'sw' ? ' — vLLM software path, loads but no speedup' : '';
        const kv = result.kv ? ` — ${fmtGB(result.weights)}GB weights + ~${fmtGB(result.kv)}GB KV` : '';
        return `Fits on ${where} via ${via}${sw}${kv}`;
    }
    if (result.reason === 'quant') return `Would fit VRAM on ${where}, but the required format is unsupported on this GPU`;
    return `Exceeds available VRAM on ${where} (needs ${fmtGB(result.weights + result.kv)}GB)`;
}

function formatContextLength(tokens) {
    if (tokens >= 1048576) return (tokens / 1048576).toFixed(1) + 'M';
    if (tokens >= 1024) return (tokens / 1024).toFixed(0) + 'K';
    return tokens.toString();
}

function getContextLengthBadgeClass(tokens) {
    if (tokens >= 524288) return 'badge-nvfp4';
    if (tokens >= 131072) return 'badge-fp8';
    if (tokens >= 32768) return 'badge-bf16';
    return 'badge-int4';
}

function renderModels() {
    const grid = document.getElementById('modelsGrid');
    const noResults = document.getElementById('noResults');

    const countEl = document.getElementById('resultCount');
    if (countEl) countEl.textContent = `${filteredModels.length} of ${MODELS_DATA.length} models`;

    if (filteredModels.length === 0) {
        grid.classList.add('hidden');
        noResults.classList.remove('hidden');
        return;
    }
    
    grid.classList.remove('hidden');
    noResults.classList.add('hidden');
    
    const gpuType = document.getElementById('gpuTypeSelect')?.value || 'L4-24GB';
    const gpuConfig = GPU_CONFIG[gpuType];
    const kvTokens = getKVContextTokens();

    grid.innerHTML = filteredModels.map(model => {
        const contextLen = model.contextLength ? formatContextLength(model.contextLength) : '4K';
        const badges = `
            <div class="flex flex-wrap gap-2 mb-4">
                <span class="badge ${getQuantBadgeClass(model.prec)}">${model.prec}</span>
                ${model.variants.slice(0, 2).map(v => `<span class="badge ${getQuantBadgeClass(v.prec)}">${v.prec} ${v.vram ? '@' + v.vram + 'GB' : ''}</span>`).join('')}
                ${model.variants.length > 2 ? `<span class="badge badge-bf16">+${model.variants.length - 2}</span>` : ''}
            </div>`;

        // The budget bar is drawn for the CURRENTLY SELECTED GPU count, from the same
        // modelFitsGPU() result that decides the ✓/✗ — so the bar always explains the
        // verdict instead of contradicting it. With "Any" selected there is no config
        // to draw against, so we show the minimum viable count instead.
        let budgetHTML;
        if (currentGPUFilter === 0) {
            let min = null;
            for (let g = 1; g <= 8; g++) { if (modelFitsGPU(model, g).fits) { min = g; break; } }
            budgetHTML = `
            <div class="mb-4 text-xs text-[#8888a0]">
                Minimum to fit: <span class="${min ? 'text-[#22c55e]' : 'text-[#ef4444]'} font-semibold">${min ? `${min}× ${gpuConfig.name}` : `won't fit on 8× ${gpuConfig.name}`}</span>
                <span class="text-[#666680]">· pick a GPU count to see the memory budget</span>
            </div>`;
        } else {
            const fit = modelFitsGPU(model, currentGPUFilter);
            const usable = fit.usable;
            const wPct = Math.max(0, Math.min(100, (fit.weights / usable) * 100));
            const kPct = Math.max(0, Math.min(100 - wPct, (fit.kv / usable) * 100));
            const total = fit.weights + fit.kv;
            const over = total > usable;
            budgetHTML = `
            <div class="mb-4">
                <div class="flex justify-between text-xs text-[#8888a0] mb-1">
                    <span>${currentGPUFilter}× ${gpuConfig.name} budget</span>
                    <span class="${over ? 'text-[#ef4444]' : 'text-[#8888a0]'}">${fmtGB(total)} / ${fmtGB(usable)} GB</span>
                </div>
                <div class="gpu-bar flex">
                    <div class="gpu-bar-seg ${over ? 'seg-over' : 'seg-weights'}" style="width: ${wPct}%"></div>
                    <div class="gpu-bar-seg seg-kv" style="width: ${kPct}%"></div>
                </div>
                <div class="flex gap-3 mt-1 text-[10px] text-[#666680]">
                    <span><span class="dot ${over ? 'dot-over' : 'dot-weights'}"></span>weights ${fmtGB(fit.weights)}GB${fit.variant ? ` (${fit.variant.prec})` : ''}</span>
                    ${kvTokens
                        ? `<span><span class="dot dot-kv"></span>KV ~${fmtGB(fit.kv)}GB @ ${formatContextLength(kvTokens)}${kvMark(model)}</span>${kvNote(model)}`
                        : `<span class="text-[#666680]">KV estimate off</span>`}
                </div>
            </div>`;
        }

        return `
        <div class="card p-6 cursor-pointer" onclick="openModal(${model.id})">
            <div class="flex justify-between items-start mb-4">
                <div>
                    <h3 class="font-semibold text-lg">${model.name}</h3>
                    <p class="text-sm text-[#8888a0]">${model.provider}</p>
                </div>
            </div>

            <div class="grid grid-cols-3 gap-2 mb-4">
                <div class="bg-[#12121a] rounded-lg p-3">
                    <div class="text-sm text-[#8888a0]">Params</div>
                    <div class="font-semibold">${model.params}</div>
                </div>
                <div class="bg-[#12121a] rounded-lg p-3">
                    <div class="text-sm text-[#8888a0]">Weights</div>
                    <div class="font-semibold">${model.vram} GB</div>
                </div>
                <div class="bg-[#12121a] rounded-lg p-3">
                    <div class="text-sm text-[#8888a0]">Context</div>
                    <div class="font-semibold">${contextLen}</div>
                </div>
            </div>

            ${budgetHTML}
            ${badges}

            <div class="mt-4 pt-4 border-t border-[#2a2a3a] flex gap-2">
                ${[1, 2, 4, 8].map(gpu => {
                    const result = modelFitsGPU(model, gpu);
                    const sel = gpu === currentGPUFilter ? ' bg-[#1a1a28] rounded-lg' : '';
                    return `<div class="flex-1 text-center py-1${sel}" title="${fitTooltip(result, gpu, gpuConfig)}">
                        <div class="text-xs text-[#8888a0]">${gpu}×</div>
                        <div class="${result.fits ? 'text-[#22c55e]' : 'text-[#ef4444]'} font-bold text-lg">${result.fits ? (result.level === 'sw' ? '✓*' : '✓') : '✗'}</div>
                    </div>`;
                }).join('')}
            </div>
        </div>
    `}).join('');
}

// Estimate marker for a model's KV figure — never present an estimated number with
// the same confidence as one read from the model's real config.
//   ''  config      — read from the model's own HF config.json
//   °   mirror      — same-architecture ungated mirror (original repo is gated)
//   †   estimate    — hand-sourced geometry, no config available anywhere
//   ‡   proxy       — no geometry at all; coarse params×tokens fallback
//   ~   (always)    — every KV number is an estimate of a real allocation
function kvMark(model) {
    if (model.kvBytesPerToken == null) return '‡';
    if (model.kvSource === 'estimate') return '†';
    if (model.kvSource === 'mirror') return '°';
    return '';
}

// A strikingly small KV figure is not a bug — it's sliding-window attention, where most
// layers cap their cache at the window instead of growing with context. Say so on the
// card, or the number reads as broken (DeepSeek-V4-Flash: a 284B model with ~0GB KV).
function kvNote(model) {
    if (hasNoKVCache(model)) return `<span class="text-[#666680]">· no KV cache (single-pass)</span>`;
    // Only flag a window that actually caps within this model's context.
    if (model.kvSlidingBytesPerToken && model.kvWindow > 0 && model.kvWindow < (model.contextLength || Infinity)) {
        return `<span class="text-[#666680]" title="Sliding-window attention: most layers cap their KV at a ${formatContextLength(model.kvWindow)}-token window instead of growing with the full context, so this model's KV stays far smaller than a full-attention model of the same size.">· SWA ${formatContextLength(model.kvWindow)}</span>`;
    }
    return '';
}

function updateStats() {
    const total = MODELS_DATA.length;
    const gpuType = document.getElementById('gpuTypeSelect')?.value || 'L4-24GB';
    const gpuConfig = GPU_CONFIG[gpuType];
    const fitModels = MODELS_DATA.filter(m => modelFitsGPU(m, currentGPUFilter).fits).length;
    const quantModels = MODELS_DATA.filter(m => m.variants.length > 0).length;
    const noFit = total - fitModels;

    document.getElementById('totalModels').textContent = total;
    document.getElementById('fitModels').textContent = fitModels;
    document.getElementById('quantModels').textContent = quantModels;
    document.getElementById('noFitModels').textContent = noFit;
}

function openModal(id) {
    const model = MODELS_DATA.find(m => m.id === id);
    if (!model) return;
    
    const gpuType = document.getElementById('gpuTypeSelect')?.value || 'L4-24GB';
    const gpuConfig = GPU_CONFIG[gpuType];
    
    const minGPUs = [1, 2, 3, 4, 5, 6, 7, 8].find(gpu => modelFitsGPU(model, gpu).fits) || 'N/A';
    const minGPUStr = typeof minGPUs === 'number' ? `${minGPUs}× ${gpuConfig.name}` : minGPUs;
    
    const kvTokens = getKVContextTokens();
    const kvGB = estKVCacheGB(model, kvTokens);
    const dtypeLabel = getKVDtype() === 'fp8' ? 'FP8' : 'FP16';

    // How the KV number was derived — spelled out, never implied.
    const kvBasis = hasNoKVCache(model)
        ? `no autoregressive KV cache — this is a single-pass diffusion/generative model, so context length adds no VRAM at all`
        : model.kvBytesPerToken == null
        ? `‡ a coarse params×tokens proxy — no attention geometry was available for this model, so it ignores GQA/MLA and over-estimates MoE`
        : model.kvSource === 'estimate'
        ? `† hand-sourced geometry (this repo is gated and has no public mirror) — an estimate, not read from the model's config`
        : model.kvSource === 'mirror'
        ? `° geometry from an ungated same-architecture mirror repo (the original is gated)`
        : `${dtypeLabel} KV from the model's own HuggingFace config.json attention geometry`;

    // Sliding-window explainer. The all-sliding case (zero full-attention layers) gets a
    // stronger caveat: KV goes ~constant regardless of context, which is a big claim, and
    // if the model ALSO runs a global/sparse attention path that config.json doesn't
    // express, we'd be UNDER-counting — the direction that makes a model claim to fit when
    // it won't. Rule is data-driven (full === 0), not a per-model hardcode.
    // A window wider than the model's own context never actually caps anything (Phi-4-mini:
    // 256K window, 128K context) — warning about it would be noise.
    const windowBites = model.kvWindow > 0 && model.kvWindow < (model.contextLength || Infinity);
    const allSliding = model.kvBytesPerToken === 0 && model.kvSlidingBytesPerToken > 0 && windowBites;
    const swa = windowBites && model.kvSlidingBytesPerToken
        ? `<div class="text-xs text-[#8888a0] mt-2">
             This model uses <strong>sliding-window attention</strong>: ${allSliding
                ? `its config declares <strong>every</strong> layer as sliding, capped at a ${formatContextLength(model.kvWindow)}-token window`
                : `only some layers cache the full context, the rest are capped at a ${formatContextLength(model.kvWindow)}-token window`}.
             Its KV therefore grows far more slowly than a full-attention model of the same size.
           </div>
           ${allSliding ? `<div class="text-xs text-[#f59e0b] mt-2">
             ⚠️ <strong>Verify this one before sizing hardware.</strong> With no full-attention layers, the KV
             estimate stays nearly flat as context grows. That is what
             <a href="https://huggingface.co/${model.hf_url}/blob/main/config.json" target="_blank" class="text-[#818cf8] hover:underline">this model's <code>config.json</code></a>
             declares, and it is what vLLM reads — but if the model also runs a global or sparse attention path
             that the config doesn't express, real KV would be <em>higher</em> than shown here. Cross-check the
             <a href="https://huggingface.co/${model.hf_url}" target="_blank" class="text-[#818cf8] hover:underline">model card</a>
             and <a href="https://recipes.vllm.ai/${model.recipe_id || model.hf_url}" target="_blank" class="text-[#818cf8] hover:underline">vLLM recipe</a> before committing to a GPU count.
           </div>` : ''}`
        : '';

    const contextWarning = kvTokens
        ? `<div class="text-xs text-[#f59e0b] mt-2">≈ KV-cache estimate at ${formatContextLength(kvTokens)} ctx ≈ <strong>+${fmtGB(kvGB)} GB</strong> on top of weights (${kvBasis}).</div>${swa}`
        : `<div class="text-xs text-[#f59e0b] mt-2">⚠️ VRAM shown is weights only. Enable the KV-cache estimate (top filter bar) to factor in context length.</div>`;

    // Memory budget for the currently-selected config, itemised the way vLLM accounts
    // for it. Every line is a real quantity except the activation reserve, which is an
    // assumption and is labelled as one.
    const gpus = currentGPUFilter || 1;
    const fit = modelFitsGPU(model, gpus);
    const budget = getGPUBudget(gpus);
    const reserve = gpus * getReserveGB();
    const weights = fit.weights ?? model.vram;
    const kvPool = Math.max(0, getGPUVRAM(gpus) - weights);
    const conc = maxConcurrentRequests(model, weights, gpus, kvTokens);

    const row = (label, val, cls = '') => `
        <div class="flex justify-between py-2 border-b border-[#1a1a24]">
            <span class="text-[#8888a0]">${label}</span><span class="font-medium ${cls}">${val}</span>
        </div>`;

    const budgetHTML = `
        <div class="bg-[#12121a] rounded-xl p-4 border border-[#2a2a3a] text-sm">
            ${row(`Physical VRAM — ${gpus}× ${gpuConfig.name}`, `${gpus * gpuConfig.vram} GB`)}
            ${row(`× GPU memory utilization (${getMemUtil()})`, `${fmtGB(budget)} GB`)}
            ${row(`− Activation / CUDA-graph reserve <span class="text-[#f59e0b]">(assumption)</span>`, `−${fmtGB(reserve)} GB`, 'text-[#f59e0b]')}
            ${row(`= Usable for weights + KV`, `${fmtGB(getGPUVRAM(gpus))} GB`, 'text-[#22c55e]')}
            ${row(`− Model weights${fit.variant ? ` (${fit.variant.prec})` : ` (${model.prec})`}`, `−${fmtGB(weights)} GB`)}
            ${row(`= KV cache pool`, `${fmtGB(kvPool)} GB`, kvPool > 0 ? 'text-[#22c55e]' : 'text-[#ef4444]')}
            ${conc != null ? row(
                `≈ Concurrent requests @ ${formatContextLength(kvTokens)} <span class="text-[#666680]">(worst case)</span>`,
                `${conc}`, conc > 0 ? 'text-[#6366f1]' : 'text-[#ef4444]') : ''}
        </div>
        <p class="text-xs text-[#666680] mt-3">
            Mirrors vLLM's own accounting: the KV cache is allocated greedily into whatever is left
            inside <code>--gpu-memory-utilization</code> after weights and activations.
            ${conc != null ? `The concurrency figure assumes <strong>every</strong> request fills the full ${formatContextLength(kvTokens)} context — real serving fits more, since requests are usually shorter and PagedAttention shares prefix blocks. Treat it as a floor.` : ''}
            The activation reserve is an assumption: vLLM <em>measures</em> it by profiling a forward pass at startup, so it varies with batch size and model.
        </p>`;

    const gpuFeasHTML = [1, 2, 3, 4, 5, 6, 7, 8].map(gpu => {
        const result = modelFitsGPU(model, gpu);
        const vram = getGPUVRAM(gpu);
        const detail = result.fits
            ? (result.variant ? ` (${result.variant.prec} variant${result.level === 'sw' ? ', SW' : ''})` : (result.level === 'sw' ? ' (software path)' : ''))
            : (result.reason === 'quant' ? ' (format unsupported)' : '');
        const label = result.fits ? '✓ Fits' : '✗ Too Large';
        return `
            <div class="flex items-center justify-between bg-[#12121a] rounded-lg p-3" title="${fitTooltip(result, gpu, gpuConfig)}">
                <span class="font-medium">${gpu}× ${gpuConfig.name} (${fmtGB(vram)}GB usable)</span>
                <span class="${result.fits ? 'text-[#22c55e]' : 'text-[#ef4444]'} font-bold">
                    ${label}<span class="text-xs text-[#8888a0]">${detail}</span>
                </span>
            </div>
        `;
    }).join('');

    const hfUrl = `https://huggingface.co/${model.hf_url}`;
    // The recipe path is case-sensitive and occasionally differs from the HF id
    // (e.g. "Google/" vs "google/"); recipe_id holds the exact path when it does.
    const vllmUrl = `https://recipes.vllm.ai/${model.recipe_id || model.hf_url}`;
    
    document.getElementById('modalTitle').textContent = model.name;
    document.getElementById('modalProvider').textContent = `by ${model.provider}`;
    document.getElementById('modalContent').innerHTML = `
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            <div class="bg-[#12121a] rounded-lg p-4 text-center min-w-0">
                <div class="text-xl font-bold text-[#6366f1] break-words leading-tight">${model.params}</div>
                <div class="text-xs text-[#8888a0] mt-1">Parameters</div>
            </div>
            <div class="bg-[#12121a] rounded-lg p-4 text-center min-w-0">
                <div class="text-xl font-bold text-[#22c55e] break-words leading-tight">${model.vram} GB</div>
                <div class="text-xs text-[#8888a0] mt-1">VRAM (weights)</div>
            </div>
            <div class="bg-[#12121a] rounded-lg p-4 text-center min-w-0">
                <div class="text-xl font-bold text-[#f59e0b] break-words leading-tight">${minGPUStr}</div>
                <div class="text-xs text-[#8888a0] mt-1">GPUs Needed</div>
            </div>
            <div class="bg-[#12121a] rounded-lg p-4 text-center min-w-0">
                <div class="text-xl font-bold"><span class="badge ${getQuantBadgeClass(model.prec)}">${model.prec}</span></div>
                <div class="text-xs text-[#8888a0] mt-1">Precision</div>
            </div>
        </div>

        <div class="mb-8">
            <h3 class="font-semibold mb-4">Quantization Variants</h3>
            <div class="flex flex-wrap gap-2">
                <span class="badge ${getQuantBadgeClass(model.prec)}">${model.prec} (default)</span>
                ${model.variants.map(v => `<span class="badge ${getQuantBadgeClass(v.prec)}">${v.prec} ${v.vram ? '@ ' + v.vram + 'GB' : ''} ${v.note ? '(' + v.note + ')' : ''}</span>`).join('')}
            </div>
            ${contextWarning}
        </div>

        <div class="mb-8">
            <h3 class="font-semibold mb-4">Memory Budget <span class="text-xs font-normal text-[#8888a0]">— ${gpus}× ${gpuConfig.name}</span></h3>
            ${budgetHTML}
        </div>

        <div class="mb-8">
            <h3 class="font-semibold mb-4">GPU Feasibility</h3>
            <div class="space-y-3">${gpuFeasHTML}</div>
        </div>

        <div class="flex gap-4">
            <a href="${hfUrl}" target="_blank" class="flex-1 bg-[#FF9D00] text-black font-semibold py-3 px-6 rounded-lg text-center hover:opacity-90 transition">
                View on HuggingFace
            </a>
            <a href="${vllmUrl}" target="_blank" class="flex-1 bg-[#6366f1] text-white font-semibold py-3 px-6 rounded-lg text-center hover:opacity-90 transition">
                vLLM Recipe
            </a>
        </div>
    `;
    
    document.getElementById('modelModal').classList.remove('hidden');
}

function closeModal() {
    document.getElementById('modelModal').classList.add('hidden');
}

document.getElementById('modelModal').addEventListener('click', (e) => {
    if (e.target.id === 'modelModal') closeModal();
});

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeModal();
});

// --- Shareable URL state (#9) -------------------------------------------------
// Reflect the current filter selection into ?query=params so a configuration can
// be bookmarked/shared, and restore it on load.
const URL_CONTROLS = {
    gpuType: 'gpuTypeSelect', quant: 'quantFilter', type: 'typeFilter',
    context: 'contextFilter', sort: 'sortSelect', kv: 'kvContextSelect', q: 'searchInput',
    util: 'memUtilSelect', reserve: 'reserveSelect', kvdtype: 'kvDtypeSelect',
};
let restoringState = false;

function syncStateToURL() {
    if (restoringState) return;
    const params = new URLSearchParams();
    if (currentGPUFilter !== 1) params.set('count', String(currentGPUFilter));
    for (const [key, id] of Object.entries(URL_CONTROLS)) {
        const el = document.getElementById(id);
        if (!el || el.value == null || el.value === '') continue;
        const isDefault = (id === 'sortSelect' && el.value === 'params')
            || (id === 'gpuTypeSelect' && el.value === 'L4-24GB')
            || (['quantFilter', 'typeFilter'].includes(id) && el.value === 'all')
            || (['contextFilter', 'kvContextSelect'].includes(id) && el.value === '0')
            || (id === 'memUtilSelect' && parseFloat(el.value) === DEFAULT_MEM_UTIL)
            || (id === 'reserveSelect' && parseFloat(el.value) === DEFAULT_RESERVE_GB)
            || (id === 'kvDtypeSelect' && el.value === 'fp16');
        if (!isDefault) params.set(key, el.value);
    }
    const qs = params.toString();
    history.replaceState(null, '', qs ? `?${qs}` : location.pathname);
}

function applyStateFromURL() {
    const params = new URLSearchParams(location.search);
    if (![...params.keys()].length) return;
    restoringState = true;
    for (const [key, id] of Object.entries(URL_CONTROLS)) {
        if (params.has(key)) setSelect(id === 'searchInput' ? null : id, params.get(key));
        if (key === 'q' && params.has('q')) { const s = document.getElementById('searchInput'); if (s) s.value = params.get('q'); }
    }
    if (params.has('count')) currentGPUFilter = parseInt(params.get('count')) || 1;
    document.querySelectorAll('#gpuCountFilter .filter-btn').forEach(btn => {
        btn.classList.toggle('active', parseInt(btn.dataset.value) === currentGPUFilter);
    });
    restoringState = false;
}

// Initial paint goes through filterModels() so the default sort (Parameters) is applied
applyStateFromURL();
filterModels();

function openGPUInfoModal() {
    document.getElementById('modalProvider').textContent = '';

    // Rendered live from GPU_CONFIG / GPU_QUANT_COMPAT (the same tables the fit
    // logic uses) so this modal can never drift from the data. Add a format to
    // FORMAT_ORDER + the maps and it shows up here automatically.
    const FORMAT_ORDER = ['BF16', 'FP8', 'INT8', 'INT4', 'NVFP4', 'MXFP4', 'MXFP8'];
    const FORMAT_LABEL = { INT4: 'INT4/AWQ/GPTQ' };
    const FORMAT_NOTE = {
        BF16: 'Universal baseline — every listed GPU runs FP16/BF16 on native tensor cores.',
        FP8: 'Native on Ada (4th-gen) & Hopper+ via Transformer Engine; software (Marlin) on Ampere.',
        INT8: 'W8A8 int8 GEMM on native int8 tensor cores — all listed GPUs are Ampere+ (sm_80+).',
        INT4: 'INT4/AWQ/GPTQ (W4A16) weights are dequantised to FP16 for compute (Marlin) — software on all GPUs.',
        NVFP4: 'Blackwell FP4 tensor cores only — vLLM recipes gate NVFP4 checkpoints to Blackwell. NVIDIA ModelOpt format.',
        MXFP4: 'OCP MXFP4 format (e.g. gpt-oss). Native on Blackwell; software dequant elsewhere — recipes note it fits a single A100.',
        MXFP8: 'OCP MXFP8 microscaling format. Native on Blackwell MX tensor cores (B200/B300); unsupported on earlier GPUs.',
    };
    // One representative GPU key per architecture column (H200 ≡ H100, B200 ≡ B100).
    const COLUMNS = [
        { label: 'L4', sub: 'Ada', gpu: 'L4-24GB' },
        { label: 'A100', sub: 'Ampere', gpu: 'A100-80GB' },
        { label: 'H100/H200', sub: 'Hopper', gpu: 'H100-80GB' },
        { label: 'B100/B200', sub: 'Blackwell', gpu: 'B100-192GB' },
    ];

    const cell = (level) => {
        if (level === 'native') return `<span class="text-[#22c55e] font-bold text-lg">✓</span><br><span class="text-[10px] text-[#4ade80]">Native</span>`;
        if (level === 'sw') return `<span class="text-[#f59e0b] font-bold text-lg">✓</span><br><span class="text-[10px] text-[#fb923c]">vLLM SW</span>`;
        return `<span class="text-[#ef4444] text-lg">✗</span>`;
    };
    const chip = (fmt, kind) => {
        const cls = kind === 'native'
            ? 'bg-[#1a3a1a] text-[#4ade80] border border-[#2a5a2a]'
            : 'bg-[#3a2a1a] text-[#fb923c] border border-[#5a3a1a]';
        return `<span class="px-2 py-0.5 rounded text-xs ${cls}">${FORMAT_LABEL[fmt] || fmt}</span>`;
    };

    const snapshotCards = COLUMNS.map(col => {
        const map = GPU_QUANT_COMPAT[col.gpu] || {};
        const cfg = GPU_CONFIG[col.gpu] || {};
        const list = (kind) => {
            const fmts = FORMAT_ORDER.filter(f => map[f] === kind);
            return fmts.length ? fmts.map(f => chip(f, kind)).join('') : '<span class="text-[10px] text-[#666680]">—</span>';
        };
        return `
                    <div class="bg-[#12121a] rounded-xl p-4 border border-[#2a2a3a]">
                        <div class="text-xs text-[#8888a0] mb-1">${cfg.architecture || ''}</div>
                        <div class="font-bold mb-2">${col.label}</div>
                        <div class="space-y-2">
                            <div>
                                <div class="text-[10px] text-[#8888a0] uppercase tracking-wide mb-1">Native HW</div>
                                <div class="flex flex-wrap gap-1">${list('native')}</div>
                            </div>
                            <div>
                                <div class="text-[10px] text-[#8888a0] uppercase tracking-wide mb-1">vLLM SW</div>
                                <div class="flex flex-wrap gap-1">${list('sw')}</div>
                            </div>
                        </div>
                    </div>`;
    }).join('');

    const util = getMemUtil();
    const reserve = getReserveGB();
    const specRows = Object.values(GPU_CONFIG).map(cfg => `
                            <tr class="border-b border-[#1a1a24]">
                                <td class="py-3 px-2 font-medium">${cfg.name}</td>
                                <td class="py-3 px-2 text-[#8888a0]">${cfg.architecture} (${cfg.sm})</td>
                                <td class="py-3 px-2 text-right">${cfg.vram} GB</td>
                                <td class="py-3 px-2 text-right text-[#8888a0]">${fmtGB(cfg.vram * util)} GB</td>
                                <td class="py-3 px-2 text-right text-[#22c55e]">${fmtGB(Math.max(0, cfg.vram * util - reserve))} GB</td>
                                <td class="py-3 px-2 text-[#8888a0]">${cfg.memory}</td>
                            </tr>`).join('');

    const matrixRows = FORMAT_ORDER.map(fmt => {
        const cells = COLUMNS.map(col =>
            `<td class="py-3 px-2 text-center">${cell((GPU_QUANT_COMPAT[col.gpu] || {})[fmt])}</td>`).join('');
        return `
                            <tr class="border-b border-[#1a1a24]">
                                <td class="py-3 px-2"><span class="badge ${getQuantBadgeClass(fmt)}">${FORMAT_LABEL[fmt] || fmt}</span></td>
                                ${cells}
                                <td class="py-3 px-2 text-[#8888a0] text-xs">${FORMAT_NOTE[fmt] || ''}</td>
                            </tr>`;
    }).join('');

    const gpuInfo = `
        <div class="space-y-8">
            <div>
                <h3 class="text-xl font-bold mb-4 gradient-text">GPU Compatibility Snapshot</h3>
                <div class="grid grid-cols-2 md:grid-cols-4 gap-3">${snapshotCards}
                </div>
                <div class="mt-3 flex gap-4 text-xs text-[#666680] justify-center">
                    <span><span class="text-[#4ade80]">●</span> Native hardware tensor core support</span>
                    <span><span class="text-[#fb923c]">●</span> vLLM software-enabled (no native HW)</span>
                </div>
            </div>

            <div>
                <h3 class="text-xl font-bold mb-4 gradient-text">GPU Specifications &amp; Memory Budget</h3>
                <div class="overflow-x-auto">
                    <table class="w-full text-sm">
                        <thead>
                            <tr class="border-b border-[#2a2a3a]">
                                <th class="text-left py-3 px-2">GPU</th>
                                <th class="text-left py-3 px-2">Architecture</th>
                                <th class="text-right py-3 px-2">Physical</th>
                                <th class="text-right py-3 px-2">Budget (×${util})</th>
                                <th class="text-right py-3 px-2">Weights + KV</th>
                                <th class="text-left py-3 px-2">Memory</th>
                            </tr>
                        </thead>
                        <tbody>${specRows}
                        </tbody>
                    </table>
                </div>
                <div class="mt-3 text-xs text-[#666680] space-y-1">
                    <p><strong class="text-[#8888a0]">Budget</strong> = physical × <code>--gpu-memory-utilization</code> (currently <strong>${util}</strong>). vLLM's own default is 0.90–0.92, so 0.95 is the optimistic end — change it in the filter bar.</p>
                    <p><strong class="text-[#8888a0]">Weights + KV</strong> = budget − activation/CUDA-graph reserve (currently <strong>${fmtGB(reserve)} GB/GPU</strong>). <span class="text-[#f59e0b]">This reserve is an assumption</span> — vLLM measures it by profiling a forward pass at startup, so it varies with batch size and model. Adjust or zero it in the filter bar.</p>
                    <p>vLLM then fills the remaining <strong class="text-[#8888a0]">KV cache pool</strong> greedily with cached tokens — which is what caps concurrency.</p>
                </div>
            </div>

            <div>
                <h3 class="text-xl font-bold mb-4 gradient-text">Quantization Compatibility by GPU</h3>
                <div class="overflow-x-auto">
                    <table class="w-full text-sm">
                        <thead>
                            <tr class="border-b border-[#2a2a3a]">
                                <th class="text-left py-3 px-2">Format</th>
                                ${COLUMNS.map(c => `<th class="text-center py-3 px-2">${c.label}<br><span class="text-xs text-[#666680]">${c.sub}</span></th>`).join('')}
                                <th class="text-left py-3 px-2">Notes</th>
                            </tr>
                        </thead>
                        <tbody>${matrixRows}
                        </tbody>
                    </table>
                </div>
                <div class="mt-3 text-xs text-[#666680]">
                    <span class="text-[#22c55e]">✓</span> = Supported &nbsp;
                    <span class="text-[#f59e0b]">✓</span> = Supported (software) &nbsp;
                    <span class="text-[#ef4444]">✗</span> = Not supported &nbsp;|&nbsp;
                    <span class="text-[#4ade80]">Native</span> = Hardware tensor cores &nbsp;
                    <span class="text-[#fb923c]">vLLM SW</span> = Software-enabled
                </div>
            </div>

            <div>
                <h3 class="text-xl font-bold mb-4 gradient-text">Reference Links</h3>
                <div class="bg-[#12121a] rounded-xl p-4 border border-[#2a2a3a]">
                    <div class="text-xs text-[#8888a0] space-y-2">
                        <p>Compatibility &amp; specs are rendered live from the app's GPU_CONFIG / GPU_QUANT_COMPAT tables. Sources:</p>
                        <ul class="list-disc list-inside space-y-1">
                            <li><a href="https://docs.vllm.ai/en/latest/features/quantization/supported_hardware/" target="_blank" class="text-[#818cf8] hover:underline">vLLM Quantization — Supported Hardware</a> — official method × architecture matrix</li>
                            <li><a href="https://recipes.vllm.ai" target="_blank" class="text-[#818cf8] hover:underline">vLLM Recipes</a> — per-model precision & hardware hints</li>
                            <li><a href="https://www.nvidia.com/en-us/" target="_blank" class="text-[#818cf8] hover:underline">NVIDIA GPU Specifications</a> — tensor-core capabilities per architecture</li>
                        </ul>
                        <p class="mt-2 text-[#f59e0b]">⚠️ vLLM's support matrix changes over time — check the links above for updates.</p>
                    </div>
                </div>
            </div>
        </div>
    `;
    document.getElementById('modalTitle').textContent = 'GPU & Format Compatibility';
    document.getElementById('modalContent').innerHTML = gpuInfo;
    document.getElementById('modelModal').classList.remove('hidden');
}
    