// Cartridge Economics — when do trained, reusable KV-cache Cartridges become
// cheaper than repeatedly prefilling conventional RAG text?
//
// Model inspired by "Cartridges at Scale: Training Modular KV Caches over
// Large Document Collections" (CAS), Hardalov, Iglesias and de Gispert,
// Amazon AGI Lab, arXiv:2606.04557, https://arxiv.org/pdf/2606.04557
//
// The paper reports hardware, optimizer steps and batch configuration, but
// not wall-clock training time or GPU-hours. The primary economics mode
// therefore requires measured GPU-hours and measured prefill wall times; it
// never derives dollar savings from the O(n^2) attention argument alone.
//
// The pure calculation functions below take plain numbers and return numbers
// (or null when a required measured value is missing) so they can be tested
// under Node without a browser. All capacity math is done in bytes: binary
// MiB/GiB/TiB for display, decimal GB for cloud billing.

const MS_PER_GPU_HOUR = 3600000
const BYTES_PER_MIB = 1024 * 1024
const BYTES_PER_GIB = 1024 * BYTES_PER_MIB
const BYTES_PER_TIB = 1024 * BYTES_PER_GIB
const BYTES_PER_DECIMAL_GB = 1e9

// BF16 KV footprint per 1K Cartridge tokens, from the CAS paper (Table of
// KV footprints; arXiv:2606.04557).
const CART_MODELS = [
    { id: 'qwen3-0.6b', label: 'Qwen3-0.6B', mibPer1K: 110 },
    { id: 'qwen3-8b', label: 'Qwen3-4B/8B', mibPer1K: 141 },
    { id: 'qwen3-32b', label: 'Qwen3-32B', mibPer1K: 250 },
    { id: 'custom-model', label: 'Custom', mibPer1K: null },
]

// KV serialization formats. The non-BF16 multipliers are ideal payload
// ratios, not measured codec output sizes.
const KV_FORMATS = [
    { id: 'bf16', label: 'BF16 K / BF16 V', multiplier: 1.0 },
    { id: 'k16v8', label: 'BF16 K / FP8 V (ideal payload)', multiplier: 0.75 },
    { id: 'fp8', label: 'FP8 K / FP8 V (ideal payload)', multiplier: 0.5 },
]

// Workload constants reported by the CAS paper (arXiv:2606.04557). These are
// paper-derived presets; training GPU-hours and wall-clock time are NOT
// reported by the paper and stay "benchmark required".
const PAPER = {
    citation:
        'Hardalov, Iglesias, de Gispert — "Cartridges at Scale: Training Modular KV Caches ' +
        'over Large Document Collections", Amazon AGI Lab, arXiv:2606.04557',
    url: 'https://arxiv.org/pdf/2606.04557',
    model: 'Qwen3-8B',
    hardware: 'H200/B200',
    selfStudyQuestionsPerDoc: '100-200',
    epochs: '80 (10-20 often reach ~95% of best)',
    optimizerSteps: '12,400-14,160',
    globalBatch: 128,
    packedSeqLen: 8192,
    longHealth: {
        corpusTokens: 236000,
        textRagTokensPerQuery: 9860, // k=10 retrieved chunks
        cartridgeTokensPerQuery20x: 2673,
        cartridgeTokensPerQuery100x: 566,
        uniqueCartridgesPerQuery: 4.4,
    },
    uniqueCartridgesPerQuery: {
        LongHealth: 4.4,
        QuALITY: 4.8,
        FinQA: 8.7,
        TechQA: 8.1,
    },
}

function isNum(x) {
    return typeof x === 'number' && isFinite(x)
}

// ── Capacity ────────────────────────────────────────────────────────────────

function cartridgeTokens(sourceTokens, compressionRatio) {
    if (!isNum(sourceTokens) || !isNum(compressionRatio) || compressionRatio <= 0) return null
    return sourceTokens / compressionRatio
}

function cartridgeBytesForTokens(cartTokens, modelMiBPer1K, kvFormatMultiplier) {
    if (!isNum(cartTokens) || !isNum(modelMiBPer1K) || !isNum(kvFormatMultiplier)) return null
    return (cartTokens / 1000) * modelMiBPer1K * BYTES_PER_MIB * kvFormatMultiplier
}

function cartridgeBytesPerDocument(sourceTokensDoc, compressionRatio, modelMiBPer1K, kvFormatMultiplier) {
    return cartridgeBytesForTokens(
        cartridgeTokens(sourceTokensDoc, compressionRatio),
        modelMiBPer1K,
        kvFormatMultiplier,
    )
}

function corpusBytes(totalSourceTokens, compressionRatio, modelMiBPer1K, kvFormatMultiplier, replicas) {
    const perReplica = cartridgeBytesForTokens(
        cartridgeTokens(totalSourceTokens, compressionRatio),
        modelMiBPer1K,
        kvFormatMultiplier,
    )
    if (perReplica === null || !isNum(replicas) || replicas < 1) return null
    return perReplica * replicas
}

function bytesToDecimalGB(bytes) {
    return bytes / BYTES_PER_DECIMAL_GB
}

// ── Storage pricing ─────────────────────────────────────────────────────────

// Resolve the $/GB-month rate for a storage preset at a given stored volume.
// Handles tiered pricing (RunPod network storage switches at 1 TB).
function storageRateForGB(preset, decimalGB) {
    if (!preset) return null
    if (Array.isArray(preset.tiers) && preset.tiers.length > 0) {
        let rate = null
        for (const tier of preset.tiers) {
            if (decimalGB >= tier.minGB) rate = tier.usdPerGBMonth
        }
        return rate
    }
    return preset.usdPerGBMonth
}

function storageCostMonth(bytes, usdPerGBMonth) {
    if (!isNum(bytes) || !isNum(usdPerGBMonth)) return null
    return bytesToDecimalGB(bytes) * usdPerGBMonth
}

// ── One-time construction cost ──────────────────────────────────────────────

// CAS trains Cartridge pools jointly, so the corpus-level cost is primary and
// the per-document number is an average, not a marginal cost.
//
// The paper identifies self-study question generation and target-model
// inference as part of the principal offline construction cost, so
// selfStudyCost is a required component: a blank value blocks the whole
// construction cost instead of silently becoming $0. Callers pass 0 only
// when the user explicitly enters zero or marks it included in the
// measured total. otherCost stays genuinely optional.
function buildCostFromMeasuredRun(gpuCount, wallHours, usdPerGpuHour, selfStudyCost, otherCost) {
    if (!isNum(gpuCount) || !isNum(wallHours) || !isNum(usdPerGpuHour)) return null
    return buildCostFromGpuHours(gpuCount * wallHours, usdPerGpuHour, selfStudyCost, otherCost)
}

function buildCostFromGpuHours(gpuHours, usdPerGpuHour, selfStudyCost, otherCost) {
    if (!isNum(gpuHours) || !isNum(usdPerGpuHour) || !isNum(selfStudyCost)) return null
    return gpuHours * usdPerGpuHour + selfStudyCost + (isNum(otherCost) ? otherCost : 0)
}

// Corpus churn: failed/preempted training retries inflate the one-time build,
// and a corpus that retrains a fraction of its documents every month turns
// part of the build cost into a recurring monthly cost.
function effectiveBuildCost(buildCost, retryOverheadFraction) {
    if (!isNum(buildCost)) return null
    const f = isNum(retryOverheadFraction) ? retryOverheadFraction : 0
    return buildCost * (1 + f)
}

function recurringRebuildCostMonth(effectiveBuild, rebuildFractionMonth) {
    if (!isNum(effectiveBuild)) return null
    const f = isNum(rebuildFractionMonth) ? rebuildFractionMonth : 0
    return effectiveBuild * f
}

// ── Measured per-query path costs ───────────────────────────────────────────

// Primary production measurement: billed GPU-hours divided by completed
// matched requests. Under continuous batching many requests share each
// GPU-second, so per-request wall latency times TP size is NOT exclusive GPU
// time; it survives below only as a labeled microbenchmark approximation.
function gpuCostPerQuery(totalGpuHours, usdPerGpuHour, completedQueries) {
    if (!isNum(totalGpuHours) || !isNum(usdPerGpuHour)) return null
    if (!isNum(completedQueries) || completedQueries <= 0) return null
    return (totalGpuHours * usdPerGpuHour) / completedQueries
}

function costPer1kToPerQuery(usdPer1kRequests) {
    return isNum(usdPer1kRequests) ? usdPer1kRequests / 1000 : null
}

// Signed per-query compute delta between the baseline and Cartridge paths.
function pathCostDelta(baselineCostQuery, cartridgeCostQuery) {
    if (!isNum(baselineCostQuery) || !isNum(cartridgeCostQuery)) return null
    return baselineCostQuery - cartridgeCostQuery
}

// ── Per-query inference savings (wall-latency microbenchmark) ───────────────

// Signed delta: a Cartridge path slower than Text RAG must surface as a
// negative saving so regressions propagate into every economic result
// instead of being silently clamped to "no saving".
function prefillGpuMsSaved(textRagPrefillWallMs, cartridgePrefillWallMs, inferenceGpuCount) {
    if (!isNum(textRagPrefillWallMs) || !isNum(cartridgePrefillWallMs) || !isNum(inferenceGpuCount)) return null
    return (textRagPrefillWallMs - cartridgePrefillWallMs) * inferenceGpuCount
}

function gpuTimeValueSavedQuery(prefillMsSaved, decodeGpuMsSaved, inferenceUsdPerGpuHour) {
    if (!isNum(prefillMsSaved) || !isNum(inferenceUsdPerGpuHour)) return null
    const decode = isNum(decodeGpuMsSaved) ? decodeGpuMsSaved : 0
    return ((prefillMsSaved + decode) / MS_PER_GPU_HOUR) * inferenceUsdPerGpuHour
}

// realizationFraction: 1.0 means elastic billing or capacity that is actually
// released; 0.0 means a fixed provisioned fleet where saved GPU time creates
// headroom but no cash saving.
function realizedComputeSavingQuery(gpuTimeValue, realizationFraction) {
    if (!isNum(gpuTimeValue) || !isNum(realizationFraction)) return null
    return gpuTimeValue * realizationFraction
}

// ── Cartridge load cost ─────────────────────────────────────────────────────

function coldCartridgesPerQuery(uniqueCartridgesQuery, cacheHitRate) {
    if (!isNum(uniqueCartridgesQuery) || !isNum(cacheHitRate)) return null
    return uniqueCartridgesQuery * (1 - cacheHitRate)
}

function loadCostQuery(
    cartBytesDoc,
    uniqueCartridgesQuery,
    cacheHitRate,
    getUsdPerRequest,
    retrievalUsdPerGB,
    egressUsdPerGB,
) {
    const cold = coldCartridgesPerQuery(uniqueCartridgesQuery, cacheHitRate)
    if (cold === null || !isNum(cartBytesDoc)) return null
    if (!isNum(getUsdPerRequest) || !isNum(retrievalUsdPerGB) || !isNum(egressUsdPerGB)) return null
    const coldBytes = cartBytesDoc * cold
    return cold * getUsdPerRequest + bytesToDecimalGB(coldBytes) * (retrievalUsdPerGB + egressUsdPerGB)
}

function coldBytesPerQuery(cartBytesDoc, uniqueCartridgesQuery, cacheHitRate) {
    const cold = coldCartridgesPerQuery(uniqueCartridgesQuery, cacheHitRate)
    if (cold === null || !isNum(cartBytesDoc)) return null
    return cartBytesDoc * cold
}

// Estimated wall milliseconds spent staging cold Cartridge objects for one
// query. Overlapped loading hides the latency behind other work; otherwise
// it is fixed per-object latency plus bytes over the effective
// storage-to-GPU bandwidth.
function coldLoadMsPerQuery(coldBytes, coldObjects, bandwidthGBps, fixedMsPerObject, overlapped) {
    if (overlapped) return 0
    if (!isNum(coldBytes) || !isNum(coldObjects) || !isNum(fixedMsPerObject)) return null
    if (!isNum(bandwidthGBps) || bandwidthGBps <= 0) return null
    return coldObjects * fixedMsPerObject + (coldBytes / (bandwidthGBps * 1e9)) * 1000
}

// Minimum effective storage-to-GPU bandwidth (GB/s) at which unoverlapped
// loading does not erase the measured wall-time benefit of the Cartridge
// path. Infinity means the fixed per-object latency alone already eats the
// entire benefit.
function minBandwidthGBps(coldBytes, coldObjects, fixedMsPerObject, wallMsSavedQuery) {
    if (!isNum(coldBytes) || !isNum(coldObjects) || !isNum(fixedMsPerObject) || !isNum(wallMsSavedQuery)) return null
    const budgetMs = wallMsSavedQuery - coldObjects * fixedMsPerObject
    if (budgetMs <= 0) return Infinity
    if (coldBytes <= 0) return 0
    return coldBytes / 1e9 / (budgetMs / 1000)
}

// ── Break-even ──────────────────────────────────────────────────────────────

function netSavingQuery(realizedSaving, loadCost) {
    if (!isNum(realizedSaving) || !isNum(loadCost)) return null
    return realizedSaving - loadCost
}

// Backblaze-style egress allowance: egress is free up to a multiple of the
// average monthly stored volume, then charged per GB. Billed at the monthly
// aggregate, never per query.
function monthlyEgressOverage(egressGBMonth, storedGB, freeMonthlyStorageMultiple, overageUsdPerGB) {
    if (!isNum(egressGBMonth) || !isNum(storedGB)) return null
    if (!isNum(freeMonthlyStorageMultiple) || !isNum(overageUsdPerGB)) return null
    return Math.max(0, egressGBMonth - freeMonthlyStorageMultiple * storedGB) * overageUsdPerGB
}

// Monthly query volume at which the Cartridge path pays for itself within the
// lifetime H (months). Non-positive per-query net saving never breaks even.
// The optional overage argument ({egressGBPerQuery, allowanceGB, usdPerGB})
// makes the solve piecewise: past the free-egress allowance every additional
// query also pays overage, so the marginal net saving shrinks.
function breakEvenQueriesMonth(buildCost, lifetimeMonths, storageMonth, netSaving, overage) {
    if (!isNum(buildCost) || !isNum(lifetimeMonths) || !isNum(storageMonth) || lifetimeMonths <= 0) return null
    if (!isNum(netSaving) || netSaving <= 0) return Infinity
    const q0 = (buildCost / lifetimeMonths + storageMonth) / netSaving
    if (!overage) return q0
    const { egressGBPerQuery, allowanceGB, usdPerGB } = overage
    if (!isNum(egressGBPerQuery) || egressGBPerQuery <= 0 || !isNum(allowanceGB) || !isNum(usdPerGB)) return q0
    const qAllowance = allowanceGB / egressGBPerQuery
    if (q0 <= qAllowance) return q0
    const netBeyond = netSaving - egressGBPerQuery * usdPerGB
    if (netBeyond <= 0) return Infinity
    return (buildCost / lifetimeMonths + storageMonth - allowanceGB * usdPerGB) / netBeyond
}

function monthlyNetSaving(queriesMonth, netSaving, storageMonth, egressOverageMonth) {
    if (!isNum(queriesMonth) || !isNum(netSaving) || !isNum(storageMonth)) return null
    if (egressOverageMonth !== undefined && egressOverageMonth !== 0 && !isNum(egressOverageMonth)) return null
    return queriesMonth * netSaving - storageMonth - (isNum(egressOverageMonth) ? egressOverageMonth : 0)
}

function paybackMonths(buildCost, monthlyNet) {
    if (!isNum(buildCost) || !isNum(monthlyNet)) return null
    if (monthlyNet <= 0) return Infinity
    return buildCost / monthlyNet
}

function lifetimeRoi(lifetimeMonths, queriesMonth, netSaving, buildCost, storageMonth, egressOverageMonth) {
    if (!isNum(lifetimeMonths) || !isNum(queriesMonth) || !isNum(netSaving)) return null
    if (!isNum(buildCost) || !isNum(storageMonth)) return null
    if (egressOverageMonth !== undefined && egressOverageMonth !== 0 && !isNum(egressOverageMonth)) return null
    const benefit = lifetimeMonths * queriesMonth * netSaving
    const cost = buildCost + lifetimeMonths * (storageMonth + (isNum(egressOverageMonth) ? egressOverageMonth : 0))
    if (cost <= 0) return null
    return (benefit - cost) / cost
}

// ── Per-document prioritization ─────────────────────────────────────────────

// Dividing query-level savings evenly between the Cartridges selected for the
// query is an allocation approximation; a measured per-document prefill delta
// should override it when available.
function netSavingPerLoad(realizedSavingQuery, uniqueCartridgesQuery, variableLoadCostPerCartridge) {
    if (!isNum(realizedSavingQuery) || !isNum(uniqueCartridgesQuery) || uniqueCartridgesQuery <= 0) return null
    if (!isNum(variableLoadCostPerCartridge)) return null
    return realizedSavingQuery / uniqueCartridgesQuery - variableLoadCostPerCartridge
}

function breakEvenDocumentHitsMonth(buildCostPerDocument, lifetimeMonths, storagePerDocumentMonth, savingPerLoad) {
    if (!isNum(buildCostPerDocument) || !isNum(lifetimeMonths) || lifetimeMonths <= 0) return null
    if (!isNum(storagePerDocumentMonth)) return null
    if (!isNum(savingPerLoad) || savingPerLoad <= 0) return Infinity
    return (buildCostPerDocument / lifetimeMonths + storagePerDocumentMonth) / savingPerLoad
}

// ── Quality gate ────────────────────────────────────────────────────────────

// Economics are conditional on quality parity: a cheaper result is irrelevant
// if accuracy fails. Returns 'unknown', 'pass' or 'fail'.
function qualityGate(textRagScore, cartridgeScore, allowedDegradation) {
    if (!isNum(textRagScore) || !isNum(cartridgeScore)) return 'unknown'
    const tol = isNum(allowedDegradation) ? allowedDegradation : 0
    return cartridgeScore >= textRagScore - tol ? 'pass' : 'fail'
}

// ── Display helpers ─────────────────────────────────────────────────────────

function formatBytesBinary(bytes) {
    if (!isNum(bytes)) return '—'
    if (bytes >= BYTES_PER_TIB) return (bytes / BYTES_PER_TIB).toFixed(2) + ' TiB'
    if (bytes >= BYTES_PER_GIB) return (bytes / BYTES_PER_GIB).toFixed(2) + ' GiB'
    if (bytes >= BYTES_PER_MIB) return (bytes / BYTES_PER_MIB).toFixed(1) + ' MiB'
    return (bytes / 1024).toFixed(1) + ' KiB'
}

function formatUSD(x) {
    if (!isNum(x)) return '—'
    const abs = Math.abs(x)
    if (abs >= 1000) return '$' + x.toLocaleString('en-US', { maximumFractionDigits: 0 })
    if (abs >= 1) return '$' + x.toFixed(2)
    if (abs >= 0.001) return '$' + x.toFixed(4)
    if (abs === 0) return '$0'
    return '$' + x.toExponential(2)
}

function formatCount(x) {
    if (!isNum(x)) return '—'
    if (!isFinite(x)) return '∞'
    if (x >= 1e9) return (x / 1e9).toFixed(2) + 'B'
    if (x >= 1e6) return (x / 1e6).toFixed(2) + 'M'
    if (x >= 1e3) return (x / 1e3).toFixed(1) + 'K'
    return x < 10 && x !== Math.round(x) ? x.toFixed(2) : String(Math.round(x))
}

const CartEcon = {
    MS_PER_GPU_HOUR,
    BYTES_PER_MIB,
    BYTES_PER_GIB,
    BYTES_PER_TIB,
    CART_MODELS,
    KV_FORMATS,
    PAPER,
    cartridgeTokens,
    cartridgeBytesForTokens,
    cartridgeBytesPerDocument,
    corpusBytes,
    bytesToDecimalGB,
    storageRateForGB,
    storageCostMonth,
    buildCostFromMeasuredRun,
    buildCostFromGpuHours,
    effectiveBuildCost,
    recurringRebuildCostMonth,
    gpuCostPerQuery,
    costPer1kToPerQuery,
    pathCostDelta,
    prefillGpuMsSaved,
    gpuTimeValueSavedQuery,
    realizedComputeSavingQuery,
    coldCartridgesPerQuery,
    coldBytesPerQuery,
    coldLoadMsPerQuery,
    minBandwidthGBps,
    loadCostQuery,
    netSavingQuery,
    monthlyEgressOverage,
    breakEvenQueriesMonth,
    monthlyNetSaving,
    paybackMonths,
    lifetimeRoi,
    netSavingPerLoad,
    breakEvenDocumentHitsMonth,
    qualityGate,
    formatBytesBinary,
    formatUSD,
    formatCount,
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = CartEcon
}

// ════════════════════════════════════════════════════════════════════════════
// Browser UI. Everything below is inert under Node.
// ════════════════════════════════════════════════════════════════════════════

if (typeof document !== 'undefined') {
    const $ = (id) => document.getElementById(id)

    const num = (id) => {
        const el = $(id)
        if (!el || el.value === '') return null
        const v = parseFloat(el.value)
        return isFinite(v) ? v : null
    }

    // ── Option lists ────────────────────────────────────────────────────────

    function fillSelect(el, options, selected) {
        el.innerHTML = options
            .map((o) => `<option value="${o.value}"${o.value === selected ? ' selected' : ''}>${o.label}</option>`)
            .join('')
    }

    function gpuPresetOptions() {
        return CARTRIDGE_PRICING.gpu.map((g) => ({
            value: g.id,
            label:
                g.price === null ? `${g.provider} — ${g.hardware}` : `${g.provider} ${g.hardware} — $${g.price}/GPU-hr`,
        }))
    }

    function initSelects() {
        fillSelect(
            $('in-model'),
            CART_MODELS.map((m) => ({
                value: m.id,
                label: m.mibPer1K ? `${m.label} (${m.mibPer1K} MiB/1K tok)` : m.label,
            })),
            'qwen3-8b',
        )
        fillSelect(
            $('in-kvformat'),
            KV_FORMATS.map((f) => ({ value: f.id, label: f.label })),
            'bf16',
        )
        fillSelect(
            $('in-compression'),
            [2, 5, 10, 20, 50, 100]
                .map((r) => ({ value: String(r), label: r + 'x' }))
                .concat([{ value: 'custom', label: 'Custom' }]),
            '20',
        )
        fillSelect(
            $('in-baseline'),
            [
                { value: 'uncached', label: 'Uncached Text RAG (repeated prefill)' },
                { value: 'kvreuse', label: 'Text RAG + ordinary KV/prefix reuse' },
                { value: 'custom', label: 'Custom measured baseline' },
            ],
            'uncached',
        )
        fillSelect($('in-train-gpu-preset'), gpuPresetOptions(), 'runpod-h200-secure')
        fillSelect($('in-inf-gpu-preset'), gpuPresetOptions(), 'runpod-h100-sxm-community')
        const storageOpts = CARTRIDGE_PRICING.storage.map((s) => ({
            value: s.id,
            label: `${s.provider} ${s.storageClass}`,
        }))
        fillSelect($('in-storage-preset'), storageOpts, 'r2-standard')
        fillSelect(
            $('st-model'),
            CART_MODELS.filter((m) => m.mibPer1K).map((m) => ({ value: m.id, label: m.label })),
            'qwen3-8b',
        )
        fillSelect(
            $('st-kvformat'),
            KV_FORMATS.map((f) => ({ value: f.id, label: f.label })),
            'bf16',
        )
        fillSelect(
            $('st-compression'),
            [2, 5, 10, 20, 50, 100].map((r) => ({ value: String(r), label: r + 'x' })),
            '20',
        )
        fillSelect($('st-storage-preset'), storageOpts, 'r2-standard')
    }

    // ── Input resolution ────────────────────────────────────────────────────

    function modelMiBPer1K(selectId, customId) {
        const id = $(selectId).value
        if (id === 'custom-model') return num(customId)
        const m = CART_MODELS.find((x) => x.id === id)
        return m ? m.mibPer1K : null
    }

    function compressionRatio() {
        const v = $('in-compression').value
        return v === 'custom' ? num('in-compression-custom') : parseFloat(v)
    }

    function gpuRate(selectId, customId) {
        const id = $(selectId).value
        const preset = CARTRIDGE_PRICING.gpu.find((g) => g.id === id)
        if (!preset || preset.price === null) return num(customId)
        return preset.price
    }

    function storagePreset() {
        return CARTRIDGE_PRICING.storage.find((s) => s.id === $('in-storage-preset').value)
    }

    // Storage cost components: preset value, overridden by a custom field when
    // the user filled one in. Unpublished (unmodeled) preset components stay
    // null unless overridden — never silently zero.
    function storageComponents(preset, decimalGB) {
        const override = (id, presetVal) => {
            const v = num(id)
            return v !== null ? v : presetVal
        }
        return {
            usdPerGBMonth: override('in-storage-rate', storageRateForGB(preset, decimalGB)),
            getUsdPerRequest: override('in-get-cost', preset ? preset.getUsdPerRequest : null),
            retrievalUsdPerGB: override('in-retrieval-cost', preset ? preset.retrievalUsdPerGB : null),
            egressUsdPerGB: override('in-egress-cost', preset ? preset.egressUsdPerGB : null),
        }
    }

    // ── Break-even computation ──────────────────────────────────────────────

    function computeBreakEven() {
        const docs = num('in-docs')
        const totalTokens = num('in-corpus-tokens')
        const docTokens = num('in-doc-tokens')
        const ratio = compressionRatio()
        const mib1k = modelMiBPer1K('in-model', 'in-model-custom')
        const kvFmt = KV_FORMATS.find((f) => f.id === $('in-kvformat').value)
        const kvMult = kvFmt ? kvFmt.multiplier : null
        const unique = num('in-unique')
        const queriesMonth = num('in-queries-month')
        const lifetime = num('in-lifetime')
        const replicas = num('in-replicas')

        const bytesDoc = cartridgeBytesPerDocument(docTokens, ratio, mib1k, kvMult)
        const corpus = corpusBytes(totalTokens, ratio, mib1k, kvMult, replicas)
        const preset = storagePreset()
        const decGB = corpus !== null ? bytesToDecimalGB(corpus) : 0
        const store = storageComponents(preset, decGB)
        const storageMonth = corpus !== null ? storageCostMonth(corpus, store.usdPerGBMonth) : null

        // Construction cost. Self-study is a required component: blank blocks
        // the construction cost unless it is marked included in the measured
        // total (which resolves it to zero without double counting).
        const trainRate = gpuRate('in-train-gpu-preset', 'in-train-rate-custom')
        const selfStudy = $('in-selfstudy-included').checked ? 0 : num('in-selfstudy')
        const otherBuild = num('in-other-build')
        const modeA = $('in-mode-a').checked
        const buildBase = modeA
            ? buildCostFromMeasuredRun(num('in-train-gpus'), num('in-train-hours'), trainRate, selfStudy, otherBuild)
            : buildCostFromGpuHours(num('in-train-gpuhours'), trainRate, selfStudy, otherBuild)
        const buildCost = effectiveBuildCost(buildBase, (num('in-retry-pct') ?? 0) / 100)
        const rebuildMonth = recurringRebuildCostMonth(buildCost, (num('in-rebuild-pct') ?? 0) / 100)

        // Per-query inference cost for both paths, three modes. The measured
        // modes (billed GPU-hours / completed requests, or $ per 1K requests)
        // are primary; wall-latency x TP is a microbenchmark approximation.
        const baseline = $('in-baseline').value
        const baselineStorageMonth = num('in-baseline-storage') ?? 0
        const infMode = $('in-inf-1k').checked ? '1k' : $('in-inf-wall').checked ? 'wall' : 'hours'
        const infGpus = num('in-inf-gpus')
        const infRate = gpuRate('in-inf-gpu-preset', 'in-inf-rate-custom')
        const cartRateOverride = num('in-inf-rate-cart')
        const cartRate = cartRateOverride !== null ? cartRateOverride : infRate
        const decodeSaved = num('in-decode-ms')
        const realization = (num('in-realization') ?? 100) / 100
        const hitRate = (num('in-hitrate') ?? 0) / 100
        const planning = infMode === 'wall' && $('in-planning').checked

        let textMs = num('in-text-ms')
        let cartMs = num('in-cart-ms')
        let textVar = null // $/query baseline-path compute
        let cartComputeVar = null // $/query Cartridge-path compute, excluding loading
        if (infMode === 'hours') {
            textVar = gpuCostPerQuery(num('in-text-gpuhours'), infRate, num('in-text-queries'))
            cartComputeVar = gpuCostPerQuery(num('in-cart-gpuhours'), cartRate, num('in-cart-queries'))
        } else if (infMode === '1k') {
            textVar = costPer1kToPerQuery(num('in-text-usd1k'))
            cartComputeVar = costPer1kToPerQuery(num('in-cart-usd1k'))
        } else {
            if (planning) {
                // Separate effective throughput per path: the Cartridge path's
                // fresh tokens still attend over the loaded Cartridge prefix,
                // so one shared tokens/s across radically different sequence
                // shapes is not valid.
                const tpsText = num('in-plan-tps-text')
                const tpsCart = num('in-plan-tps-cart')
                const textTok = num('in-plan-text-tokens')
                const cartTok = num('in-plan-cart-tokens')
                textMs = isNum(tpsText) && isNum(textTok) && tpsText > 0 ? (textTok / tpsText) * 1000 : null
                cartMs = isNum(tpsCart) && isNum(cartTok) && tpsCart > 0 ? (cartTok / tpsCart) * 1000 : null
            }
            textVar =
                isNum(textMs) && isNum(infGpus) && isNum(infRate)
                    ? ((textMs * infGpus) / MS_PER_GPU_HOUR) * infRate
                    : null
            const decodeValue = isNum(decodeSaved) && isNum(infRate) ? (decodeSaved / MS_PER_GPU_HOUR) * infRate : 0
            cartComputeVar =
                isNum(cartMs) && isNum(infGpus) && isNum(infRate)
                    ? ((cartMs * infGpus) / MS_PER_GPU_HOUR) * infRate - decodeValue
                    : null
        }

        const gpuValue = pathCostDelta(textVar, cartComputeVar)
        const realized = realizedComputeSavingQuery(gpuValue, realization)
        // Cartridge loading boundary: either the measured Cartridge-path cost
        // and latency already include load/staging (loadCost is an explicit
        // zero, never double-counted), or loading is modeled separately from
        // hit rate, bandwidth and object-store prices.
        const loadIncluded = $('in-load-included').checked
        const loadOverlap = $('in-load-overlap').checked
        const loadCost = loadIncluded
            ? 0
            : loadCostQuery(
                  bytesDoc,
                  unique,
                  hitRate,
                  store.getUsdPerRequest,
                  store.retrievalUsdPerGB,
                  store.egressUsdPerGB,
              )
        const coldObjects = coldCartridgesPerQuery(unique, hitRate)
        const coldBytes = coldBytesPerQuery(bytesDoc, unique, hitRate)
        const coldMonthGB = isNum(coldBytes) && isNum(queriesMonth) ? bytesToDecimalGB(coldBytes * queriesMonth) : null
        const loadFixedMs = num('in-load-fixed-ms')
        const loadBw = num('in-load-bw')
        const coldLoadMs = loadIncluded
            ? null
            : coldLoadMsPerQuery(coldBytes, coldObjects, loadBw, loadFixedMs, loadOverlap)
        // Wall-time benefit is only known in the wall-latency microbenchmark
        const wallMsSaved = isNum(textMs) && isNum(cartMs) ? textMs - cartMs : null
        const minBw =
            loadIncluded || loadOverlap ? null : minBandwidthGBps(coldBytes, coldObjects, loadFixedMs, wallMsSaved)
        const ttftDelta = num('in-ttft-delta')
        const ttftSlo = num('in-ttft-slo')
        const ttftFail = isNum(ttftDelta) && isNum(ttftSlo) && ttftDelta > ttftSlo
        const net = netSavingQuery(realized, loadCost)

        // Account-level egress allowance (Backblaze: free up to a multiple of
        // stored volume, then overage), applied at the monthly aggregate
        const hasAllowance =
            preset && isNum(preset.egressOverageUsdPerGB) && isNum(preset.egressFreeMonthlyStorageMultiple)
        let egressOverage = 0
        let overageParams = null
        if (hasAllowance && !loadIncluded) {
            egressOverage = monthlyEgressOverage(
                coldMonthGB,
                decGB,
                preset.egressFreeMonthlyStorageMultiple,
                preset.egressOverageUsdPerGB,
            )
            overageParams = {
                egressGBPerQuery: isNum(coldBytes) ? bytesToDecimalGB(coldBytes) : null,
                allowanceGB: preset.egressFreeMonthlyStorageMultiple * decGB,
                usdPerGB: preset.egressOverageUsdPerGB,
            }
        }

        // Storage and rebuild are deltas against the selected baseline's own
        // storage/IO bill, not charges unique to the Cartridge side
        const fixedMonth =
            storageMonth !== null && rebuildMonth !== null ? storageMonth + rebuildMonth - baselineStorageMonth : null
        const beQueries = breakEvenQueriesMonth(buildCost, lifetime, fixedMonth, net, overageParams)
        const monthly = monthlyNetSaving(queriesMonth, net, fixedMonth, egressOverage)
        const payback = paybackMonths(buildCost, monthly)
        const roi = lifetimeRoi(lifetime, queriesMonth, net, buildCost, fixedMonth, egressOverage)

        // Complete Cartridge-path variable cost per query (compute + loading)
        const cartVarTotal = isNum(cartComputeVar) && isNum(loadCost) ? cartComputeVar + loadCost : null

        // Per-document view
        const buildPerDoc = isNum(buildCost) && isNum(docs) && docs > 0 ? buildCost / docs : null
        const storagePerDocMonth =
            bytesDoc !== null && isNum(replicas) ? storageCostMonth(bytesDoc * replicas, store.usdPerGBMonth) : null
        const perLoadCost = loadCost !== null && isNum(unique) && unique > 0 ? loadCost / unique : null
        const savingPerLoad = netSavingPerLoad(realized, unique, perLoadCost)
        const beDocHits = breakEvenDocumentHitsMonth(buildPerDoc, lifetime, storagePerDocMonth, savingPerLoad)

        const quality = qualityGate(num('in-q-text'), num('in-q-cart'), num('in-q-tol'))

        return {
            docs,
            totalTokens,
            docTokens,
            ratio,
            mib1k,
            kvMult,
            unique,
            queriesMonth,
            lifetime,
            replicas,
            bytesDoc,
            corpus,
            preset,
            store,
            storageMonth,
            baseline,
            baselineStorageMonth,
            rebuildMonth,
            fixedMonth,
            buildCost,
            planning,
            infMode,
            textMs,
            cartMs,
            infGpus,
            infRate,
            gpuValue,
            realization,
            realized,
            hitRate,
            loadIncluded,
            loadOverlap,
            loadCost,
            coldObjects,
            coldBytes,
            coldMonthGB,
            coldLoadMs,
            minBw,
            ttftDelta,
            ttftSlo,
            ttftFail,
            hasAllowance,
            egressOverage,
            net,
            beQueries,
            monthly,
            payback,
            roi,
            textVar,
            cartComputeVar,
            cartVarTotal,
            buildPerDoc,
            storagePerDocMonth,
            perLoadCost,
            savingPerLoad,
            beDocHits,
            quality,
        }
    }

    const BASELINE_LABELS = {
        uncached: 'Uncached Text RAG',
        kvreuse: 'Text RAG + KV reuse',
        custom: 'Custom baseline',
    }

    // ── Status banner ───────────────────────────────────────────────────────

    function renderStatus(r) {
        const el = $('be-status')
        const benchmarkMissing =
            r.buildCost === null || r.textVar === null || r.cartComputeVar === null || r.net === null
        let cls, text
        if (benchmarkMissing) {
            cls = 'status-neutral'
            text =
                'Benchmark required — enter the measured construction cost and a measured per-query inference ' +
                'cost for both paths. The CAS paper does not report them, and this calculator refuses to invent them.'
        } else if (r.quality === 'fail') {
            cls = 'status-fail'
            text = 'Quality gate failed — Cartridge accuracy is below the allowed degradation. Economics are moot.'
        } else if (r.net <= 0 || !isFinite(r.beQueries)) {
            cls = 'status-fail'
            text = 'No economic break-even — per-query net saving is not positive at these inputs.'
        } else if (r.planning) {
            // Planning mode never emits a viable verdict
            cls = 'status-warn'
            text =
                'Illustrative only — incomplete estimate; measured production costs required. At the assumed ' +
                `per-path throughputs the sketch would break even at ${formatCount(r.beQueries)} queries/month ` +
                `vs ${BASELINE_LABELS[r.baseline]}, but assumed throughput is not a measurement.`
        } else if (r.quality === 'pass') {
            cls = 'status-good'
            text = `Economically and quality viable vs ${BASELINE_LABELS[r.baseline]} — break-even at ${formatCount(r.beQueries)} queries/month within the ${r.lifetime}-month lifetime.`
        } else {
            cls = 'status-warn'
            text =
                `Economically positive vs ${BASELINE_LABELS[r.baseline]}; quality unverified — break-even at ` +
                `${formatCount(r.beQueries)} queries/month, but no quality scores entered. ` +
                'Verify accuracy parity before trusting this.'
        }
        if (r.planning && r.baseline !== 'uncached') {
            text +=
                ' [Planning mode estimates repeated full prefill, which overstates a KV-reuse or custom baseline — ' +
                'enter measured all-in costs for this comparison.]'
        }
        if (!benchmarkMissing && r.ttftFail) {
            if (cls === 'status-good') cls = 'status-warn'
            text += ` [p95 TTFT delta ${r.ttftDelta} ms exceeds the ${r.ttftSlo} ms SLO — the latency gate fails even where dollars break even.]`
        }
        if (r.planning && !benchmarkMissing && cls === 'status-fail') {
            text += ' [Illustrative planning assumption — not measured, not paper results.]'
        } else if (!r.planning && r.infMode === 'wall' && !benchmarkMissing) {
            text +=
                ' [Wall-latency microbenchmark approximation — under continuous batching wall latency × TP ' +
                'is not exclusive GPU time; prefer measured GPU-hours per completed request.]'
        }
        el.className = 'status-banner ' + cls
        el.textContent = text
    }

    // ── Metrics rendering ───────────────────────────────────────────────────

    const BENCH = '<span class="bench-required">benchmark required</span>'

    function setMetric(id, html) {
        $(id).innerHTML = html
    }

    function metricUSD(x) {
        return x === null ? BENCH : formatUSD(x)
    }

    function renderMetrics(r) {
        setMetric(
            'm-corpus-bytes',
            r.corpus === null
                ? '—'
                : `${formatBytesBinary(r.corpus)} <span class="sub">(${bytesToDecimalGB(r.corpus).toFixed(1)} GB billed)</span>`,
        )
        setMetric('m-bytes-doc', r.bytesDoc === null ? '—' : formatBytesBinary(r.bytesDoc))
        setMetric('m-build-cost', metricUSD(r.buildCost))
        setMetric('m-build-doc', metricUSD(r.buildPerDoc))
        setMetric('m-storage-month', r.storageMonth === null ? BENCH : formatUSD(r.storageMonth) + '/mo')
        setMetric('m-rebuild-month', r.rebuildMonth === null ? BENCH : formatUSD(r.rebuildMonth) + '/mo')
        setMetric('m-gpu-value', r.gpuValue === null ? BENCH : formatUSD(r.gpuValue) + '/query')
        setMetric('m-realized', r.realized === null ? BENCH : formatUSD(r.realized) + '/query')
        setMetric(
            'm-load-cost',
            r.loadIncluded
                ? '<span class="sub">included in measured path cost</span>'
                : r.loadCost === null
                  ? BENCH
                  : formatUSD(r.loadCost) + '/query',
        )
        setMetric('m-cold-bytes', r.coldBytes === null ? '—' : formatBytesBinary(r.coldBytes) + '/query')
        setMetric('m-cold-month', r.coldMonthGB === null ? '—' : r.coldMonthGB.toFixed(1) + ' GB/mo')
        setMetric(
            'm-egress-overage',
            !r.hasAllowance
                ? '<span class="sub">n/a for this provider</span>'
                : r.egressOverage === null
                  ? '—'
                  : formatUSD(r.egressOverage) + '/mo',
        )
        setMetric(
            'm-cold-ms',
            r.loadIncluded
                ? '<span class="sub">included in measured path latency</span>'
                : r.loadOverlap
                  ? '<span class="sub">overlapped with compute</span>'
                  : r.coldLoadMs === null
                    ? '—'
                    : r.coldLoadMs.toFixed(1) + ' ms/query',
        )
        setMetric(
            'm-min-bw',
            r.minBw === null ? '—' : !isFinite(r.minBw) ? 'unreachable' : r.minBw.toFixed(2) + ' GB/s',
        )
        setMetric('m-net', r.net === null ? BENCH : formatUSD(r.net) + '/query')
        setMetric(
            'm-breakeven',
            r.beQueries === null ? BENCH : isFinite(r.beQueries) ? formatCount(r.beQueries) + ' queries/mo' : 'never',
        )
        setMetric('m-monthly-net', metricUSD(r.monthly))
        setMetric(
            'm-payback',
            r.payback === null ? BENCH : isFinite(r.payback) ? r.payback.toFixed(1) + ' months' : 'never',
        )
        setMetric('m-roi', r.roi === null ? BENCH : (r.roi * 100).toFixed(0) + '%')
        setMetric('m-saving-per-load', r.savingPerLoad === null ? BENCH : formatUSD(r.savingPerLoad) + '/load')
        setMetric(
            'm-doc-hits',
            r.beDocHits === null ? BENCH : isFinite(r.beDocHits) ? formatCount(r.beDocHits) + ' loads/mo' : 'never',
        )

        // Unmodeled-cost warning
        const warn = $('storage-unmodeled')
        const preset = r.preset
        if (preset && preset.unmodeled && preset.unmodeled.length > 0) {
            const missing = []
            if (preset.unmodeled.includes('getUsdPerRequest') && num('in-get-cost') === null)
                missing.push('GET/request')
            if (preset.unmodeled.includes('retrievalUsdPerGB') && num('in-retrieval-cost') === null)
                missing.push('retrieval $/GB')
            if (preset.unmodeled.includes('egressUsdPerGB') && num('in-egress-cost') === null)
                missing.push('egress $/GB')
            if (missing.length > 0) {
                warn.style.display = 'block'
                warn.textContent =
                    `${preset.provider} ${preset.storageClass} does not publish: ${missing.join(', ')}. ` +
                    'These are NOT assumed zero — enter overrides below or load costs stay "benchmark required".'
            } else {
                warn.style.display = 'none'
            }
        } else {
            warn.style.display = 'none'
        }
    }

    // ── Crossover graph (SVG) ───────────────────────────────────────────────

    function renderCrossoverGraph(r) {
        const host = $('be-graph')
        const W = 720
        const H = 340
        const pad = { l: 64, r: 20, t: 16, b: 44 }

        const usable =
            r.buildCost !== null &&
            r.storageMonth !== null &&
            r.textVar !== null &&
            r.cartVarTotal !== null &&
            isNum(r.lifetime)
        if (!usable) {
            host.innerHTML = `<div class="graph-empty">Crossover graph needs the measured construction cost and a measured per-query inference cost for both paths.</div>`
            return
        }
        // Complete measured path costs only: never reconstruct the Cartridge
        // line from the Text line minus a delta, and never put non-positive
        // dollar values on a logarithmic cost axis.
        if (r.textVar <= 0 || r.cartVarTotal < 0) {
            host.innerHTML = `<div class="graph-empty">Cannot plot: a non-positive per-query path cost does not fit the logarithmic cost axis. A negative complete Cartridge-path cost usually means the decode adjustment exceeds the measured compute — re-measure the path instead.</div>`
            return
        }

        const Hm = r.lifetime
        const rebuild = isNum(r.rebuildMonth) ? r.rebuildMonth : 0
        const textLine = (q) => Hm * (q * r.textVar + r.baselineStorageMonth)
        const cartLine = (q) => r.buildCost + Hm * (r.storageMonth + rebuild) + Hm * q * r.cartVarTotal

        // Log X range around break-even and selected volume
        let anchors = [r.queriesMonth || 1e4]
        if (isFinite(r.beQueries) && r.beQueries > 0) anchors.push(r.beQueries)
        const lo = Math.max(1, Math.min(...anchors) / 100)
        const hi = Math.max(...anchors) * 100
        const x0 = Math.pow(10, Math.floor(Math.log10(lo)))
        const x1 = Math.pow(10, Math.ceil(Math.log10(hi)))

        const yMaxRaw = Math.max(textLine(x1), cartLine(x1))
        const yMinRaw = Math.max(0.01, Math.min(textLine(x0), r.buildCost + Hm * (r.storageMonth + rebuild)))
        const ly0 = Math.floor(Math.log10(yMinRaw))
        const ly1 = Math.ceil(Math.log10(yMaxRaw))

        const X = (q) =>
            pad.l + ((Math.log10(q) - Math.log10(x0)) / (Math.log10(x1) - Math.log10(x0))) * (W - pad.l - pad.r)
        const Y = (c) => {
            const lc = Math.log10(Math.max(c, Math.pow(10, ly0)))
            return H - pad.b - ((lc - ly0) / (ly1 - ly0)) * (H - pad.t - pad.b)
        }

        const path = (fn) => {
            const pts = []
            const steps = 120
            for (let i = 0; i <= steps; i++) {
                const q = Math.pow(10, Math.log10(x0) + (i / steps) * (Math.log10(x1) - Math.log10(x0)))
                pts.push(`${i === 0 ? 'M' : 'L'}${X(q).toFixed(1)},${Y(fn(q)).toFixed(1)}`)
            }
            return pts.join(' ')
        }

        let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Lifetime cost crossover graph">`

        // Gridlines and ticks
        for (let e = Math.log10(x0); e <= Math.log10(x1); e++) {
            const x = X(Math.pow(10, e))
            svg += `<line x1="${x}" y1="${pad.t}" x2="${x}" y2="${H - pad.b}" stroke="rgba(255,255,255,0.07)"/>`
            svg += `<text x="${x}" y="${H - pad.b + 18}" fill="rgba(255,255,255,0.55)" font-size="11" text-anchor="middle">10<tspan baseline-shift="super" font-size="8">${e}</tspan></text>`
        }
        for (let e = ly0; e <= ly1; e++) {
            const y = Y(Math.pow(10, e))
            svg += `<line x1="${pad.l}" y1="${y}" x2="${W - pad.r}" y2="${y}" stroke="rgba(255,255,255,0.07)"/>`
            svg += `<text x="${pad.l - 8}" y="${y + 4}" fill="rgba(255,255,255,0.55)" font-size="11" text-anchor="end">$10<tspan baseline-shift="super" font-size="8">${e}</tspan></text>`
        }
        svg += `<text x="${(pad.l + W - pad.r) / 2}" y="${H - 6}" fill="rgba(255,255,255,0.65)" font-size="12" text-anchor="middle">Monthly RAG queries (log)</text>`

        // Cost lines: text RAG (orange), cartridge (cyan)
        svg += `<path d="${path(textLine)}" fill="none" stroke="#fb923c" stroke-width="2.5"/>`
        svg += `<path d="${path(cartLine)}" fill="none" stroke="#22d3ee" stroke-width="2.5"/>`

        // Break-even marker
        let beNote = ''
        if (isFinite(r.beQueries) && r.beQueries > 0) {
            if (r.beQueries >= x0 && r.beQueries <= x1) {
                const bx = X(r.beQueries)
                const by = Y(textLine(r.beQueries))
                svg += `<circle cx="${bx}" cy="${by}" r="5" fill="#f472b6"/>`
                svg += `<text x="${Math.min(bx + 8, W - 130)}" y="${Math.max(by - 10, 14)}" fill="#f472b6" font-size="12">break-even ${formatCount(r.beQueries)}/mo</text>`
            } else {
                beNote = `Break-even at ${formatCount(r.beQueries)} queries/month lies outside the plotted range.`
            }
        } else {
            beNote = 'No break-even: the Cartridge line never drops below Text RAG (net saving ≤ 0).'
        }

        // Selected volume marker
        if (isNum(r.queriesMonth) && r.queriesMonth >= x0 && r.queriesMonth <= x1) {
            const sx = X(r.queriesMonth)
            svg += `<line x1="${sx}" y1="${pad.t}" x2="${sx}" y2="${H - pad.b}" stroke="rgba(167,139,250,0.6)" stroke-dasharray="4 4" stroke-width="1.5"/>`
            svg += `<text x="${sx}" y="${pad.t + 12}" fill="#a78bfa" font-size="11" text-anchor="middle">your volume</text>`
        }

        // Legend
        const baseLabel = BASELINE_LABELS[r.baseline] || 'Text RAG'
        svg += `<line x1="${W - 240}" y1="24" x2="${W - 210}" y2="24" stroke="#fb923c" stroke-width="2.5"/>`
        svg += `<text x="${W - 204}" y="28" fill="rgba(255,255,255,0.8)" font-size="12">${baseLabel}</text>`
        svg += `<line x1="${W - 240}" y1="42" x2="${W - 210}" y2="42" stroke="#22d3ee" stroke-width="2.5"/>`
        svg += `<text x="${W - 204}" y="46" fill="rgba(255,255,255,0.8)" font-size="12">Cartridge</text>`
        svg += `</svg>`
        if (beNote) svg += `<div class="graph-note">${beNote}</div>`
        if (r.realization < 1) {
            svg += `<div class="graph-note">Lines plot billed path costs; the break-even marker uses realized cash saving (${Math.round(r.realization * 100)}% realization).</div>`
        }
        host.innerHTML = svg
    }

    // ── Cost component bars ─────────────────────────────────────────────────

    function renderComponents(r) {
        const host = $('be-components')
        if (
            r.textVar === null ||
            r.buildCost === null ||
            r.storageMonth === null ||
            r.cartComputeVar === null ||
            r.loadCost === null ||
            !isNum(r.queriesMonth)
        ) {
            host.innerHTML = ''
            return
        }
        const Hm = r.lifetime
        const Q = r.queriesMonth
        const items = [
            { label: `${BASELINE_LABELS[r.baseline]} compute`, value: Hm * Q * r.textVar, color: '#fb923c' },
            { label: 'Cartridge construction', value: r.buildCost, color: '#22d3ee' },
            { label: 'Cartridge storage', value: Hm * r.storageMonth, color: '#a78bfa' },
            { label: 'Cartridge reads', value: Hm * Q * r.loadCost, color: '#f472b6' },
            { label: 'Cartridge path compute', value: Hm * Q * r.cartComputeVar, color: '#34d399' },
        ]
        if (isNum(r.rebuildMonth) && r.rebuildMonth > 0) {
            items.push({ label: 'Recurring Cartridge rebuilds', value: Hm * r.rebuildMonth, color: '#fbbf24' })
        }
        if (r.baselineStorageMonth > 0) {
            items.push({ label: 'Baseline-side storage/IO', value: Hm * r.baselineStorageMonth, color: '#f59e0b' })
        }
        const max = Math.max(...items.map((i) => i.value), 1e-9)
        host.innerHTML =
            `<div class="comp-title">Lifetime cost components at ${formatCount(Q)} queries/month (${Hm} months). Common decode, retrieval and model-hosting costs are excluded from both sides.</div>` +
            items
                .map(
                    (i) => `
            <div class="comp-row">
                <div class="comp-label">${i.label}</div>
                <div class="comp-bar-track"><div class="comp-bar" style="width:${Math.max(1, (i.value / max) * 100).toFixed(1)}%;background:${i.color}"></div></div>
                <div class="comp-value">${formatUSD(i.value)}</div>
            </div>`,
                )
                .join('')
    }

    // ── Storage & Traffic tab ───────────────────────────────────────────────

    function computeStorage() {
        const mib1k = (CART_MODELS.find((m) => m.id === $('st-model').value) || {}).mibPer1K
        const kvFmt = KV_FORMATS.find((f) => f.id === $('st-kvformat').value)
        const kvMult = kvFmt ? kvFmt.multiplier : null
        const ratio = parseFloat($('st-compression').value)
        const corpusTokens = num('st-corpus-tokens')
        const docTokens = num('st-doc-tokens')
        const unique = num('st-unique')
        const queriesDay = num('st-queries-day')
        const hitRate = (num('st-hitrate') ?? 0) / 100
        const preset = CARTRIDGE_PRICING.storage.find((s) => s.id === $('st-storage-preset').value)

        const capacity = corpusBytes(corpusTokens, ratio, mib1k, kvMult, 1)
        const decGB = capacity !== null ? bytesToDecimalGB(capacity) : 0
        const rate = storageRateForGB(preset, decGB)
        const monthly = capacity !== null ? storageCostMonth(capacity, rate) : null
        const bytesDoc = cartridgeBytesPerDocument(docTokens, ratio, mib1k, kvMult)
        const activeTokens = isNum(unique) && isNum(docTokens) && isNum(ratio) ? unique * (docTokens / ratio) : null
        const activeBytes = bytesDoc !== null && isNum(unique) ? unique * bytesDoc : null
        const coldReadsDay = isNum(queriesDay) && isNum(unique) ? queriesDay * unique * (1 - hitRate) : null
        const trafficDay = coldReadsDay !== null && bytesDoc !== null ? coldReadsDay * bytesDoc : null

        return {
            mib1k,
            kvMult,
            ratio,
            corpusTokens,
            docTokens,
            unique,
            queriesDay,
            hitRate,
            preset,
            capacity,
            rate,
            monthly,
            bytesDoc,
            activeTokens,
            activeBytes,
            coldReadsDay,
            trafficDay,
        }
    }

    let stGraphMode = 'capacity'

    function renderStorage() {
        const s = computeStorage()
        setMetric(
            'st-m-capacity',
            s.capacity === null
                ? '—'
                : `${formatBytesBinary(s.capacity)} <span class="sub">(${bytesToDecimalGB(s.capacity).toFixed(1)} GB billed)</span>`,
        )
        setMetric('st-m-monthly', s.monthly === null ? '—' : formatUSD(s.monthly) + '/mo')
        setMetric(
            'st-m-active',
            s.activeBytes === null ? '—' : `${formatCount(s.activeTokens)} tok / ${formatBytesBinary(s.activeBytes)}`,
        )
        setMetric('st-m-cold-reads', s.coldReadsDay === null ? '—' : formatCount(s.coldReadsDay) + '/day')
        setMetric('st-m-traffic', s.trafficDay === null ? '—' : formatBytesBinary(s.trafficDay) + '/day')

        renderStorageGraph(s)
        renderTrafficGraph(s)
    }

    function logAxes(svgParts, W, H, pad, lx0, lx1, ly0, ly1, xLabel, yPrefix) {
        const X = (lx) => pad.l + ((lx - lx0) / (lx1 - lx0)) * (W - pad.l - pad.r)
        const Y = (ly) => H - pad.b - ((ly - ly0) / (ly1 - ly0)) * (H - pad.t - pad.b)
        for (let e = Math.ceil(lx0); e <= Math.floor(lx1); e++) {
            const x = X(e)
            svgParts.push(`<line x1="${x}" y1="${pad.t}" x2="${x}" y2="${H - pad.b}" stroke="rgba(255,255,255,0.07)"/>`)
            svgParts.push(
                `<text x="${x}" y="${H - pad.b + 18}" fill="rgba(255,255,255,0.55)" font-size="11" text-anchor="middle">10<tspan baseline-shift="super" font-size="8">${e}</tspan></text>`,
            )
        }
        for (let e = Math.ceil(ly0); e <= Math.floor(ly1); e++) {
            const y = Y(e)
            svgParts.push(`<line x1="${pad.l}" y1="${y}" x2="${W - pad.r}" y2="${y}" stroke="rgba(255,255,255,0.07)"/>`)
            svgParts.push(
                `<text x="${pad.l - 8}" y="${y + 4}" fill="rgba(255,255,255,0.55)" font-size="11" text-anchor="end">${yPrefix}10<tspan baseline-shift="super" font-size="8">${e}</tspan></text>`,
            )
        }
        svgParts.push(
            `<text x="${(pad.l + W - pad.r) / 2}" y="${H - 6}" fill="rgba(255,255,255,0.65)" font-size="12" text-anchor="middle">${xLabel}</text>`,
        )
        return { X, Y }
    }

    function renderStorageGraph(s) {
        const host = $('st-graph-capacity')
        if (s.mib1k === null || !isNum(s.ratio)) {
            host.innerHTML = ''
            return
        }
        const W = 720
        const H = 300
        const pad = { l: 70, r: 20, t: 16, b: 44 }
        const lx0 = 6
        const lx1 = 11 // 1M .. 100B source tokens
        const valueAt = (tokens) => {
            const cap = corpusBytes(tokens, s.ratio, s.mib1k, s.kvMult, 1)
            if (stGraphMode === 'capacity') return cap / BYTES_PER_GIB
            return storageCostMonth(cap, storageRateForGB(s.preset, bytesToDecimalGB(cap)))
        }
        const v0 = valueAt(Math.pow(10, lx0))
        const v1 = valueAt(Math.pow(10, lx1))
        if (!isNum(v0) || !isNum(v1) || v0 <= 0) {
            host.innerHTML = `<div class="graph-empty">Selected storage preset has no published rate — enter a custom rate on the Break-even tab.</div>`
            return
        }
        const ly0 = Math.floor(Math.log10(v0))
        const ly1 = Math.ceil(Math.log10(v1))
        const parts = [`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Cartridge capacity versus corpus size">`]
        const yPrefix = stGraphMode === 'capacity' ? '' : '$'
        const { X, Y } = logAxes(parts, W, H, pad, lx0, lx1, ly0, ly1, 'Source corpus tokens (log)', yPrefix)
        const pts = []
        for (let i = 0; i <= 100; i++) {
            const lx = lx0 + (i / 100) * (lx1 - lx0)
            const v = valueAt(Math.pow(10, lx))
            pts.push(`${i === 0 ? 'M' : 'L'}${X(lx).toFixed(1)},${Y(Math.log10(Math.max(v, 1e-9))).toFixed(1)}`)
        }
        parts.push(`<path d="${pts.join(' ')}" fill="none" stroke="#22d3ee" stroke-width="2.5"/>`)
        if (isNum(s.corpusTokens) && s.corpusTokens > 0) {
            const lx = Math.log10(s.corpusTokens)
            if (lx >= lx0 && lx <= lx1) {
                const v = valueAt(s.corpusTokens)
                parts.push(`<circle cx="${X(lx)}" cy="${Y(Math.log10(Math.max(v, 1e-9)))}" r="5" fill="#f472b6"/>`)
            }
        }
        parts.push(
            `<text x="${pad.l + 6}" y="${pad.t + 12}" fill="rgba(255,255,255,0.7)" font-size="12">${stGraphMode === 'capacity' ? 'Persistent capacity (GiB, log)' : 'Monthly storage cost (log)'}</text>`,
        )
        parts.push('</svg>')
        host.innerHTML = parts.join('')
    }

    function renderTrafficGraph(s) {
        const host = $('st-graph-traffic')
        if (s.bytesDoc === null || !isNum(s.unique)) {
            host.innerHTML = ''
            return
        }
        const W = 720
        const H = 300
        const pad = { l: 70, r: 20, t: 16, b: 44 }
        const lx0 = 2
        const lx1 = 8 // 100 .. 100M queries/day
        const rates = [
            { hit: 0, color: '#fb923c', label: '0% hit' },
            { hit: 0.9, color: '#a78bfa', label: '90% hit' },
            { hit: 0.99, color: '#22d3ee', label: '99% hit' },
        ]
        const gibAt = (queries, hit) => (queries * s.unique * (1 - hit) * s.bytesDoc) / BYTES_PER_GIB
        const vMin = gibAt(Math.pow(10, lx0), 0.99)
        const vMax = gibAt(Math.pow(10, lx1), 0)
        const ly0 = Math.floor(Math.log10(Math.max(vMin, 1e-9)))
        const ly1 = Math.ceil(Math.log10(Math.max(vMax, 1e-8)))
        const parts = [
            `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Daily cold-read traffic versus query volume">`,
        ]
        const { X, Y } = logAxes(parts, W, H, pad, lx0, lx1, ly0, ly1, 'RAG queries per day (log)', '')
        rates.forEach((r, ri) => {
            const pts = []
            for (let i = 0; i <= 60; i++) {
                const lx = lx0 + (i / 60) * (lx1 - lx0)
                const v = gibAt(Math.pow(10, lx), r.hit)
                pts.push(`${i === 0 ? 'M' : 'L'}${X(lx).toFixed(1)},${Y(Math.log10(Math.max(v, 1e-9))).toFixed(1)}`)
            }
            parts.push(`<path d="${pts.join(' ')}" fill="none" stroke="${r.color}" stroke-width="2.5"/>`)
            parts.push(
                `<line x1="${W - 180}" y1="${20 + ri * 18}" x2="${W - 150}" y2="${20 + ri * 18}" stroke="${r.color}" stroke-width="2.5"/>`,
            )
            parts.push(
                `<text x="${W - 144}" y="${24 + ri * 18}" fill="rgba(255,255,255,0.8)" font-size="12">${r.label}</text>`,
            )
        })
        parts.push(
            `<text x="${pad.l + 6}" y="${pad.t + 12}" fill="rgba(255,255,255,0.7)" font-size="12">Cold-read traffic (GiB/day, log)</text>`,
        )
        parts.push('</svg>')
        host.innerHTML = parts.join('')
    }

    // ── Assumptions tab ─────────────────────────────────────────────────────

    function renderAssumptions() {
        const gpuRows = CARTRIDGE_PRICING.gpu
            .filter((g) => g.price !== null)
            .map(
                (g) =>
                    `<tr><td>${g.provider}</td><td>${g.hardware}</td><td>$${g.price.toFixed(2)}</td><td>${g.unit}</td>` +
                    `<td>${g.checkedDate}</td><td><a href="${g.sourceUrl}" target="_blank" rel="noopener">source</a></td></tr>`,
            )
            .join('')
        $('price-table-gpu').innerHTML = gpuRows
        const stRows = CARTRIDGE_PRICING.storage
            .filter((s) => s.id !== 'custom-storage')
            .map((s) => {
                const price = s.tiers
                    ? s.tiers.map((t) => `$${t.usdPerGBMonth}/GB-mo ≥${t.minGB}GB`).join(', ')
                    : `$${s.usdPerGBMonth}/GB-mo`
                const un =
                    s.unmodeled && s.unmodeled.length
                        ? ' <span class="warn-inline">(transfer/request unmodeled)</span>'
                        : ''
                return (
                    `<tr><td>${s.provider}</td><td>${s.storageClass}</td><td>${price}${un}</td>` +
                    `<td>${s.checkedDate}</td><td><a href="${s.sourceUrl}" target="_blank" rel="noopener">source</a></td></tr>`
                )
            })
            .join('')
        $('price-table-storage').innerHTML = stRows

        // Stale-pricing warning
        const checked = new Date(CARTRIDGE_PRICING.snapshotDate + 'T00:00:00Z')
        const ageDays = (Date.now() - checked.getTime()) / 86400000
        const stale = $('pricing-stale')
        if (ageDays > CARTRIDGE_PRICING.staleAfterDays) {
            stale.style.display = 'block'
            stale.textContent = `Pricing may be stale: this snapshot was checked on ${CARTRIDGE_PRICING.snapshotDate}, ${Math.floor(ageDays)} days ago. Re-verify against the linked sources.`
        }
        document.querySelectorAll('.snapshot-date').forEach((el) => (el.textContent = CARTRIDGE_PRICING.snapshotDate))
    }

    // ── URL state ───────────────────────────────────────────────────────────

    const STATE_IDS = [
        'in-docs',
        'in-corpus-tokens',
        'in-doc-tokens',
        'in-compression',
        'in-compression-custom',
        'in-model',
        'in-model-custom',
        'in-kvformat',
        'in-unique',
        'in-queries-month',
        'in-lifetime',
        'in-replicas',
        'in-train-gpus',
        'in-train-hours',
        'in-train-gpuhours',
        'in-train-gpu-preset',
        'in-train-rate-custom',
        'in-selfstudy',
        'in-other-build',
        'in-rebuild-pct',
        'in-retry-pct',
        'in-baseline',
        'in-baseline-storage',
        'in-text-ms',
        'in-cart-ms',
        'in-text-gpuhours',
        'in-text-queries',
        'in-cart-gpuhours',
        'in-cart-queries',
        'in-text-usd1k',
        'in-cart-usd1k',
        'in-inf-gpus',
        'in-inf-gpu-preset',
        'in-inf-rate-custom',
        'in-inf-rate-cart',
        'in-decode-ms',
        'in-realization',
        'in-run-meta',
        'in-plan-text-tokens',
        'in-plan-cart-tokens',
        'in-plan-tps-text',
        'in-plan-tps-cart',
        'in-storage-preset',
        'in-hitrate',
        'in-load-bw',
        'in-load-fixed-ms',
        'in-ttft-delta',
        'in-ttft-slo',
        'in-storage-rate',
        'in-get-cost',
        'in-retrieval-cost',
        'in-egress-cost',
        'in-q-text',
        'in-q-cart',
        'in-q-tol',
        'st-model',
        'st-kvformat',
        'st-compression',
        'st-corpus-tokens',
        'st-doc-tokens',
        'st-unique',
        'st-queries-day',
        'st-hitrate',
        'st-storage-preset',
    ]
    const STATE_CHECKS = [
        'in-mode-a',
        'in-mode-b',
        'in-planning',
        'in-selfstudy-included',
        'in-inf-hours',
        'in-inf-1k',
        'in-inf-wall',
        'in-load-separate',
        'in-load-included',
        'in-load-overlap',
    ]

    function encodeState() {
        const p = new URLSearchParams()
        for (const id of STATE_IDS) {
            const el = $(id)
            if (el && el.value !== '') p.set(id, el.value)
        }
        for (const id of STATE_CHECKS) {
            const el = $(id)
            if (el && el.checked) p.set(id, '1')
        }
        p.set('tab', activeTab)
        return p
    }

    function decodeState() {
        const p = new URLSearchParams(location.search)
        for (const id of STATE_IDS) {
            const el = $(id)
            if (el && p.has(id)) el.value = p.get(id)
        }
        for (const id of STATE_CHECKS) {
            const el = $(id)
            if (el) el.checked = p.get(id) === '1'
        }
        if (!$('in-mode-a').checked && !$('in-mode-b').checked) $('in-mode-a').checked = true
        // Old links carried a single shared planning throughput
        if (p.has('in-plan-tps')) {
            if ($('in-plan-tps-text').value === '') $('in-plan-tps-text').value = p.get('in-plan-tps')
            if ($('in-plan-tps-cart').value === '') $('in-plan-tps-cart').value = p.get('in-plan-tps')
        }
        if (!$('in-load-separate').checked && !$('in-load-included').checked) $('in-load-separate').checked = true
        if (!$('in-inf-hours').checked && !$('in-inf-1k').checked && !$('in-inf-wall').checked) {
            // Links that predate the inference-mode selector carried wall times
            if (p.has('in-text-ms') || p.has('in-cart-ms') || p.get('in-planning') === '1') {
                $('in-inf-wall').checked = true
            } else {
                $('in-inf-hours').checked = true
            }
        }
        if (p.has('tab')) switchTab(p.get('tab'), false)
    }

    function pushState() {
        const url = location.pathname + '?' + encodeState().toString()
        history.replaceState(null, '', url)
    }

    // ── Tabs ────────────────────────────────────────────────────────────────

    let activeTab = 'breakeven'

    function switchTab(tab, updateUrl = true) {
        if (!document.getElementById('page-' + tab)) tab = 'breakeven'
        activeTab = tab
        document.querySelectorAll('.top-tab').forEach((b) => b.classList.toggle('active', b.dataset.page === tab))
        document.querySelectorAll('.page').forEach((pg) => pg.classList.toggle('active', pg.id === 'page-' + tab))
        if (updateUrl) pushState()
    }

    // ── Wiring ──────────────────────────────────────────────────────────────

    function deriveDocTokens() {
        const docs = num('in-docs')
        const total = num('in-corpus-tokens')
        if (isNum(docs) && docs > 0 && isNum(total)) {
            $('in-doc-tokens').value = Math.round(total / docs)
        }
    }

    function refreshVisibility() {
        $('mode-a-fields').style.display = $('in-mode-a').checked ? '' : 'none'
        $('mode-b-fields').style.display = $('in-mode-b').checked ? '' : 'none'
        const wall = $('in-inf-wall').checked
        $('inf-hours-fields').style.display = $('in-inf-hours').checked ? '' : 'none'
        $('inf-1k-fields').style.display = $('in-inf-1k').checked ? '' : 'none'
        $('inf-wall-fields').style.display = wall ? '' : 'none'
        $('planning-fields').style.display = wall && $('in-planning').checked ? '' : 'none'
        $('planning-banner').style.display = wall && $('in-planning').checked ? 'block' : 'none'
        $('measured-ms-fields').style.display = wall && $('in-planning').checked ? 'none' : ''
        $('load-separate-fields').style.display = $('in-load-included').checked ? 'none' : ''
        $('in-compression-custom').style.display = $('in-compression').value === 'custom' ? '' : 'none'
        $('in-model-custom').style.display = $('in-model').value === 'custom-model' ? '' : 'none'
        const trainCustom = CARTRIDGE_PRICING.gpu.find((g) => g.id === $('in-train-gpu-preset').value)
        $('in-train-rate-custom').style.display = trainCustom && trainCustom.price === null ? '' : 'none'
        const infCustom = CARTRIDGE_PRICING.gpu.find((g) => g.id === $('in-inf-gpu-preset').value)
        $('in-inf-rate-custom').style.display = infCustom && infCustom.price === null ? '' : 'none'
        $('v-realization').textContent = ($('in-realization').value || '100') + '%'
    }

    function recompute() {
        refreshVisibility()
        const r = computeBreakEven()
        renderStatus(r)
        renderMetrics(r)
        renderCrossoverGraph(r)
        renderComponents(r)
        renderStorage()
        pushState()
    }

    function init() {
        initSelects()
        decodeState()
        renderAssumptions()

        document
            .querySelectorAll('.top-tab')
            .forEach((b) => b.addEventListener('click', () => switchTab(b.dataset.page)))
        document.querySelectorAll('.graph-toggle button').forEach((b) =>
            b.addEventListener('click', () => {
                stGraphMode = b.dataset.mode
                document.querySelectorAll('.graph-toggle button').forEach((x) => x.classList.toggle('active', x === b))
                renderStorage()
            }),
        )
        ;['in-docs', 'in-corpus-tokens'].forEach((id) => $(id).addEventListener('input', deriveDocTokens))
        document.querySelectorAll('input, select').forEach((el) => {
            el.addEventListener('input', recompute)
            el.addEventListener('change', recompute)
        })
        $('copy-link').addEventListener('click', () => {
            pushState()
            navigator.clipboard.writeText(location.href).then(() => {
                $('copy-link').textContent = 'Copied!'
                setTimeout(() => ($('copy-link').textContent = 'Copy scenario link'), 1500)
            })
        })
        window.addEventListener('popstate', () => {
            decodeState()
            recompute()
        })
        recompute()
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init)
    } else {
        init()
    }
}
