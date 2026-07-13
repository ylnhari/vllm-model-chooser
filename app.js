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

// Single source of truth for GPU VRAM calculation — must exist EXACTLY once
function getGPUVRAM(gpus) {
    const gpuType = document.getElementById('gpuTypeSelect')?.value || 'L4-24GB';
    return gpus * (GPU_CONFIG[gpuType]?.usableVram || 72);
}

// Rough KV-cache estimate (GB). The dataset has no per-model attention geometry
// (layers / KV-heads / head-dim), so this is a deliberately coarse proxy:
// KV scales with total parameters and sequence length. Calibrated so a ~70B model
// at 128K ≈ ~30GB (FP16 KV). It IGNORES GQA/MLA and is clearly labelled an estimate
// in the UI. Returns 0 when the KV estimate is disabled (tokens = 0).
const KV_GB_PER_PARAM_PER_TOKEN = 3.3e-6;   // GB per (B-params × token)
function estKVCacheGB(model, tokens) {
    if (!tokens) return 0;
    return model.totalParams * tokens * KV_GB_PER_PARAM_PER_TOKEN;
}
// Selected context length (tokens) for the KV estimate; 0 = estimate off.
function getKVContextTokens() {
    return parseInt(document.getElementById('kvContextSelect')?.value || '0') || 0;
}


let filteredModels = [...MODELS_DATA];
let currentGPUFilter = 1;
let currentQuantFilter = 'all';
let currentTypeFilter = 'all';
let currentContextFilter = 0;
let currentContextLength = 4096;

// Dual check: BOTH VRAM capacity AND quantization format compatibility must pass.
// When the KV estimate is enabled, the estimated KV cache is added to the weight
// VRAM before the capacity check. Returns { fits, variant?, level?, reason? }.
function modelFitsGPU(model, gpus) {
    if (gpus === 0) return { fits: true, reason: "Any configuration" };
    const gpuType = document.getElementById('gpuTypeSelect')?.value || 'L4-24GB';
    const vram = getGPUVRAM(gpus);
    const kv = estKVCacheGB(model, getKVContextTokens());
    let vramWouldFit = false;   // did any precision fit on VRAM but get blocked by the quant gate?

    const candidates = [{ prec: model.prec, vram: model.vram, base: true }, ...(model.variants || [])];
    for (const c of candidates) {
        if (!c.vram) continue;
        const fitsVram = c.vram + kv <= vram;
        if (!fitsVram) continue;
        const level = precSupportLevel(c.prec || model.prec, gpuType);
        if (level === null) { vramWouldFit = true; continue; }   // blocked by quant gate
        return c.base ? { fits: true, level } : { fits: true, variant: c, level };
    }
    return { fits: false, reason: vramWouldFit ? 'quant' : 'vram' };
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
            case 'mmlu': return (b.benchmark.mmlu || 0) - (a.benchmark.mmlu || 0);
            case 'humaneval': return (b.benchmark.humaneval || 0) - (a.benchmark.humaneval || 0);
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

function getVerdictClass(model) {
    const gpus = [1, 2, 4, 8];
    for (const gpu of gpus) {
        if (modelFitsGPU(model, gpu).fits) return 'verdict-yes';
    }
    return 'verdict-no';
}

// Human-readable explanation for a fit result, used as a hover tooltip.
function fitTooltip(result, gpus, gpuConfig) {
    const where = `${gpus}× ${gpuConfig.name} (${getGPUVRAM(gpus)}GB usable)`;
    if (result.fits) {
        const via = result.variant ? `${result.variant.prec} variant` : 'base precision';
        const sw = result.level === 'sw' ? ' — vLLM software path, loads but no speedup' : '';
        return `Fits on ${where} via ${via}${sw}`;
    }
    if (result.reason === 'quant') return `Would fit VRAM on ${where}, but the required format is unsupported on this GPU`;
    return `Exceeds available VRAM on ${where}`;
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
    
    grid.innerHTML = filteredModels.map(model => {
        const gpuType = document.getElementById('gpuTypeSelect')?.value || 'L4-24GB';
        const gpuConfig = GPU_CONFIG[gpuType];
        const maxVRAM = gpuConfig.usableVram * 8;
        const vramPercent = Math.min((model.vram / maxVRAM) * 100, 100);
        const verdictClass = getVerdictClass(model);
        const contextLen = model.contextLength ? formatContextLength(model.contextLength) : '4K';
        
        return `
        <div class="card p-6 cursor-pointer" onclick="openModal(${model.id})">
            <div class="flex justify-between items-start mb-4">
                <div>
                    <h3 class="font-semibold text-lg">${model.name}</h3>
                    <p class="text-sm text-[#8888a0]">${model.provider}</p>
                </div>
                ${model.tested ? '<span class="badge bg-[#22c55e]/20 text-[#22c55e] border border-[#22c55e]/30">TESTED</span>' : ''}
            </div>
            
            <div class="grid grid-cols-3 gap-2 mb-4">
                <div class="bg-[#12121a] rounded-lg p-3">
                    <div class="text-sm text-[#8888a0]">Params</div>
                    <div class="font-semibold">${model.params}</div>
                </div>
                <div class="bg-[#12121a] rounded-lg p-3">
                    <div class="text-sm text-[#8888a0]">VRAM</div>
                    <div class="font-semibold">${model.vram} GB</div>
                </div>
                <div class="bg-[#12121a] rounded-lg p-3">
                    <div class="text-sm text-[#8888a0]">Context</div>
                    <div class="font-semibold">${contextLen}</div>
                </div>
            </div>
            
            <div class="mb-4">
                <div class="flex justify-between text-xs text-[#8888a0] mb-1">
                    <span>GPU Feasibility (${gpuConfig.name})</span>
                    <span>${model.vram}GB / ${maxVRAM}GB max</span>
                </div>
                <div class="gpu-bar">
                    <div class="gpu-bar-fill ${verdictClass}" style="width: ${vramPercent}%"></div>
                </div>
            </div>
            
            <div class="flex flex-wrap gap-2 mb-4">
                <span class="badge ${getQuantBadgeClass(model.prec)}">${model.prec}</span>
                ${model.variants.slice(0, 2).map(v => `<span class="badge ${getQuantBadgeClass(v.prec)}">${v.prec} ${v.vram ? '@' + v.vram + 'GB' : ''}</span>`).join('')}
                ${model.variants.length > 2 ? `<span class="badge badge-bf16">+${model.variants.length - 2}</span>` : ''}
            </div>
            
            ${model.benchmark.mmlu != null ? `
            <div class="space-y-2">
                <div>
                    <div class="flex justify-between text-xs mb-1">
                        <span class="text-[#8888a0]">MMLU</span>
                        <span>${model.benchmark.mmlu}%</span>
                    </div>
                    <div class="benchmark-bar">
                        <div class="benchmark-fill" style="width: ${model.benchmark.mmlu}%"></div>
                    </div>
                </div>
                ${model.benchmark.humaneval != null ? `
                <div>
                    <div class="flex justify-between text-xs mb-1">
                        <span class="text-[#8888a0]">HumanEval</span>
                        <span>${model.benchmark.humaneval}%</span>
                    </div>
                    <div class="benchmark-bar">
                        <div class="benchmark-fill" style="width: ${model.benchmark.humaneval}%"></div>
                    </div>
                </div>
                ` : ''}
            </div>
            ` : '<p class="text-xs text-[#8888a0]">No benchmark data available</p>'}
            
            <div class="mt-4 pt-4 border-t border-[#2a2a3a] flex gap-2">
                ${[1, 2, 4, 8].map(gpu => {
                    const result = modelFitsGPU(model, gpu);
                    return `<div class="flex-1 text-center" title="${fitTooltip(result, gpu, gpuConfig)}">
                        <div class="text-xs text-[#8888a0]">${gpu}×</div>
                        <div class="${result.fits ? 'text-[#22c55e]' : 'text-[#ef4444]'} font-bold text-lg">${result.fits ? (result.level === 'sw' ? '✓*' : '✓') : '✗'}</div>
                    </div>`;
                }).join('')}
            </div>
        </div>
    `}).join('');
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
    const contextWarning = kvTokens
        ? `<div class="text-xs text-[#f59e0b] mt-2">⚠️ KV-cache estimate at ${formatContextLength(kvTokens)} ctx ≈ <strong>+${kvGB.toFixed(1)} GB</strong> on top of weights (rough — a params×tokens proxy that ignores GQA/MLA and over-estimates MoE models, whose KV tracks attention depth, not total params).</div>`
        : `<div class="text-xs text-[#f59e0b] mt-2">⚠️ VRAM shown is weights only. Enable the KV-cache estimate (top filter bar) to factor in context length.</div>`;
    
    const benchmarks = model.benchmark;
    const benchHTML = Object.entries(benchmarks)
        .filter(([_, v]) => v !== null)
        .map(([key, value]) => `
            <div>
                <div class="flex justify-between text-sm mb-1">
                    <span class="text-[#8888a0]">${key.toUpperCase()}</span>
                    <span class="font-semibold">${value}%</span>
                </div>
                <div class="benchmark-bar">
                    <div class="benchmark-fill" style="width: ${value}%"></div>
                </div>
            </div>
        `).join('') || '<p class="text-sm text-[#8888a0]">No benchmark data available</p>';
    
    const gpuFeasHTML = [1, 2, 3, 4, 5, 6, 7, 8].map(gpu => {
        const result = modelFitsGPU(model, gpu);
        const vram = getGPUVRAM(gpu);
        const detail = result.fits
            ? (result.variant ? ` (${result.variant.prec} variant${result.level === 'sw' ? ', SW' : ''})` : (result.level === 'sw' ? ' (software path)' : ''))
            : (result.reason === 'quant' ? ' (format unsupported)' : '');
        const label = result.fits ? '✓ Fits' : '✗ Too Large';
        return `
            <div class="flex items-center justify-between bg-[#12121a] rounded-lg p-3" title="${fitTooltip(result, gpu, gpuConfig)}">
                <span class="font-medium">${gpu}× ${gpuConfig.name} (${vram}GB usable)</span>
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
            <h3 class="font-semibold mb-4">Benchmark Performance</h3>
            <div class="space-y-4">${benchHTML}</div>
            <p class="text-xs text-[#f59e0b] mt-3">⚠️ Indicative only — benchmarks are hand-curated from model cards/leaderboards, are <strong>not</strong> part of the vLLM recipe data, and are unverified for unreleased/preview models. Do not treat as authoritative.</p>
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
            || (['contextFilter', 'kvContextSelect'].includes(id) && el.value === '0');
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

    const specRows = Object.values(GPU_CONFIG).map(cfg => `
                            <tr class="border-b border-[#1a1a24]">
                                <td class="py-3 px-2 font-medium">${cfg.name}</td>
                                <td class="py-3 px-2 text-[#8888a0]">${cfg.architecture} (${cfg.sm})</td>
                                <td class="py-3 px-2 text-right">${cfg.vram} GB</td>
                                <td class="py-3 px-2 text-right text-[#22c55e]">${cfg.usableVram} GB</td>
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
                <h3 class="text-xl font-bold mb-4 gradient-text">GPU Specifications</h3>
                <div class="overflow-x-auto">
                    <table class="w-full text-sm">
                        <thead>
                            <tr class="border-b border-[#2a2a3a]">
                                <th class="text-left py-3 px-2">GPU</th>
                                <th class="text-left py-3 px-2">Architecture</th>
                                <th class="text-right py-3 px-2">VRAM</th>
                                <th class="text-right py-3 px-2">Usable (95%)</th>
                                <th class="text-left py-3 px-2">Memory</th>
                            </tr>
                        </thead>
                        <tbody>${specRows}
                        </tbody>
                    </table>
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
    