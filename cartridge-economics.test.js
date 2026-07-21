// Node tests for the Cartridge Economics calculation model.
// Run with: npm test  (node --test cartridge-economics.test.js)

const test = require('node:test')
const assert = require('node:assert')
const fs = require('node:fs')
const path = require('node:path')

const E = require('./cartridge-economics.js')
const PRICING = require('./cartridge-pricing.js')

const MIB = 1024 * 1024

function closeTo(actual, expected, relTol = 1e-9) {
    assert.ok(
        Math.abs(actual - expected) <= Math.abs(expected) * relTol + 1e-12,
        `expected ${actual} to be close to ${expected}`,
    )
}

test('Qwen3-8B, 1B source tokens, 20x compression, BF16 corpus is 7,050,000 MiB', () => {
    const bytes = E.corpusBytes(1e9, 20, 141, 1.0, 1)
    assert.strictEqual(bytes / MIB, 7050000)
    assert.strictEqual(E.formatBytesBinary(bytes), '6.72 TiB')
})

test('sub-BF16 formats are labeled experimental projections', () => {
    for (const f of E.KV_FORMATS) {
        if (f.id === 'bf16') {
            assert.ok(!f.label.includes('experimental'), 'BF16 is paper-validated')
        } else {
            assert.match(f.label, /experimental projection/, `${f.id} must be marked experimental`)
        }
    }
})

test('BF16-K/FP8-V ideal payload is 75% of BF16', () => {
    const bf16 = E.corpusBytes(1e9, 20, 141, 1.0, 1)
    const k16v8 = E.corpusBytes(1e9, 20, 141, 0.75, 1)
    assert.strictEqual(k16v8, bf16 * 0.75)
    assert.strictEqual(k16v8 / MIB, 5287500)
})

test('R2 monthly cost bills decimal GB, not GiB', () => {
    const bytes = E.corpusBytes(1e9, 20, 141, 1.0, 1)
    closeTo(E.bytesToDecimalGB(bytes), 7392.4608)
    const r2 = PRICING.storage.find((s) => s.id === 'r2-standard')
    closeTo(E.storageCostMonth(bytes, r2.usdPerGBMonth), 110.886912)
    // The binary-GiB reading would be ~$115.9/month; make sure we are not doing that
    const wrongGiB = (bytes / (1024 * 1024 * 1024)) * r2.usdPerGBMonth
    assert.notStrictEqual(E.storageCostMonth(bytes, r2.usdPerGBMonth).toFixed(2), wrongGiB.toFixed(2))
    // K16/V8 ideal payload of the same corpus
    closeTo(E.storageCostMonth(bytes * 0.75, r2.usdPerGBMonth), 83.165184)
})

test('LongHealth 236K-token corpus at 20x is 11,800 Cartridge tokens and ~1,663.8 MiB', () => {
    assert.strictEqual(E.cartridgeTokens(236000, 20), 11800)
    const bytes = E.corpusBytes(236000, 20, 141, 1.0, 1)
    closeTo(bytes / MIB, 1663.8)
    const r2 = PRICING.storage.find((s) => s.id === 'r2-standard')
    closeTo(E.storageCostMonth(bytes, r2.usdPerGBMonth), 0.026169311232, 1e-6)
})

test('GPU-dollar conversion uses 3,600,000 ms per GPU-hour', () => {
    assert.strictEqual(E.MS_PER_GPU_HOUR, 3600000)
    // 100 GPU-ms saved at $2.69/GPU-hour
    closeTo(E.gpuTimeValueSavedQuery(100, 0, 2.69), (100 / 3600000) * 2.69)
    closeTo(E.gpuTimeValueSavedQuery(100, 0, 2.69), 0.0000747222222, 1e-6)
})

test('cache-hit rate reduces reads but not persistent storage', () => {
    const bytesDoc = E.cartridgeBytesPerDocument(11800, 20, 141, 1.0)
    const cold0 = E.coldCartridgesPerQuery(4.4, 0)
    const cold90 = E.coldCartridgesPerQuery(4.4, 0.9)
    assert.strictEqual(cold0, 4.4)
    closeTo(cold90, 0.44)
    const load0 = E.loadCostQuery(bytesDoc, 4.4, 0, 0.36 / 1e6, 0, 0)
    const load90 = E.loadCostQuery(bytesDoc, 4.4, 0.9, 0.36 / 1e6, 0, 0)
    closeTo(load90, load0 * 0.1)
    // Persistent corpus bytes are independent of the hit rate
    const corpus = E.corpusBytes(236000, 20, 141, 1.0, 1)
    assert.strictEqual(corpus, E.corpusBytes(236000, 20, 141, 1.0, 1))
})

test('non-positive per-query net saving never breaks even', () => {
    assert.strictEqual(E.breakEvenQueriesMonth(1000, 6, 10, 0), Infinity)
    assert.strictEqual(E.breakEvenQueriesMonth(1000, 6, 10, -0.001), Infinity)
    assert.strictEqual(E.paybackMonths(1000, 0), Infinity)
    assert.strictEqual(E.paybackMonths(1000, -5), Infinity)
    assert.strictEqual(E.breakEvenDocumentHitsMonth(50, 6, 0.1, 0), Infinity)
})

test('break-even, payback and ROI match hand-computed fixtures', () => {
    // buildCost $1000, lifetime 10 months, storage $50/month, net $0.01/query
    closeTo(E.breakEvenQueriesMonth(1000, 10, 50, 0.01), 15000)
    // At 20,000 queries/month
    closeTo(E.monthlyNetSaving(20000, 0.01, 50), 150)
    closeTo(E.paybackMonths(1000, 150), 1000 / 150)
    closeTo(E.lifetimeRoi(10, 20000, 0.01, 1000, 50), (2000 - 1500) / 1500)
    // Per-document: $50/doc build, $0.10/doc-month storage, $0.002 net/load
    closeTo(E.breakEvenDocumentHitsMonth(50, 10, 0.1, 0.002), (50 / 10 + 0.1) / 0.002)
})

test('Backblaze egress overage bills the monthly aggregate past 3x stored volume', () => {
    const b2 = PRICING.storage.find((s) => s.id === 'backblaze-b2')
    assert.strictEqual(b2.egressFreeMonthlyStorageMultiple, 3)
    assert.strictEqual(b2.egressOverageUsdPerGB, 0.01)
    // 100 GB stored: 300 GB free egress; 500 GB egress pays 200 GB overage
    closeTo(E.monthlyEgressOverage(500, 100, 3, 0.01), 2)
    // Under the allowance: zero, not negative
    assert.strictEqual(E.monthlyEgressOverage(200, 100, 3, 0.01), 0)
    // Unknown egress volume propagates as null, never zero
    assert.strictEqual(E.monthlyEgressOverage(null, 100, 3, 0.01), null)
})

test('break-even solve turns piecewise when the egress allowance binds', () => {
    // Without overage: (1000/10 + 50) / 0.01 = 15,000 queries/month
    closeTo(E.breakEvenQueriesMonth(1000, 10, 50, 0.01), 15000)
    // Allowance so large it never binds: unchanged
    closeTo(
        E.breakEvenQueriesMonth(1000, 10, 50, 0.01, { egressGBPerQuery: 0.001, allowanceGB: 100, usdPerGB: 0.01 }),
        15000,
    )
    // Allowance of 10 GB binds at 10,000 queries: marginal net drops to
    // 0.01 - 0.001*0.01 past it, so break-even moves later, not earlier
    const be = E.breakEvenQueriesMonth(1000, 10, 50, 0.01, {
        egressGBPerQuery: 0.001,
        allowanceGB: 10,
        usdPerGB: 0.01,
    })
    closeTo(be, (100 + 50 - 0.1) / (0.01 - 0.00001))
    assert.ok(be > 15000)
    // Overage larger than the net saving: never breaks even
    assert.strictEqual(
        E.breakEvenQueriesMonth(1000, 10, 50, 0.01, { egressGBPerQuery: 2, allowanceGB: 10, usdPerGB: 0.01 }),
        Infinity,
    )
    // Monthly net subtracts the aggregate overage in full
    closeTo(E.monthlyNetSaving(20000, 0.01, 50, 2), 148)
})

test('RunPod network storage tier boundary at 1 TB', () => {
    const preset = PRICING.storage.find((s) => s.id === 'runpod-network-standard')
    assert.strictEqual(E.storageRateForGB(preset, 999), 0.07)
    assert.strictEqual(E.storageRateForGB(preset, 1000), 0.05)
    assert.strictEqual(E.storageRateForGB(preset, 5000), 0.05)
    // Flat-rate presets ignore volume
    const r2 = PRICING.storage.find((s) => s.id === 'r2-standard')
    assert.strictEqual(E.storageRateForGB(r2, 1), 0.015)
    assert.strictEqual(E.storageRateForGB(r2, 1e6), 0.015)
})

test('every pricing preset carries a checked date and source URL', () => {
    const all = PRICING.gpu.concat(PRICING.storage)
    assert.ok(all.length >= 10)
    for (const p of all) {
        assert.match(p.checkedDate, /^\d{4}-\d{2}-\d{2}$/, `${p.id} checkedDate`)
        assert.match(p.sourceUrl, /^https:\/\//, `${p.id} sourceUrl`)
        assert.ok(p.id && p.provider && p.unit, `${p.id} identity fields`)
    }
})

test('benchmark records carry validity, and invalid runs never become presets', () => {
    const B = require('./cartridge-benchmarks.js')
    assert.ok(B.schemaVersion >= 1)
    assert.ok(Array.isArray(B.records))
    for (const rec of B.records) {
        assert.ok(rec.id, 'record id')
        assert.ok(['valid', 'exploratory', 'invalid'].includes(rec.validity), `${rec.id} validity`)
        assert.ok(typeof rec.notes === 'string' && rec.notes.length > 0, `${rec.id} notes`)
    }
    // The known 0.6B run with the train/eval frozen-sink mismatch is recorded
    // as invalid and its 0% score must never seed the calculator
    const bad = B.records.find((r) => r.id.includes('qwen3-0.6b'))
    assert.ok(bad, '0.6B run is recorded rather than silently dropped')
    assert.strictEqual(bad.validity, 'invalid')
    const econ = fs.readFileSync(path.join(__dirname, 'cartridge-economics.js'), 'utf8')
    for (const rec of B.records.filter((r) => r.validity !== 'valid')) {
        assert.ok(!econ.includes(rec.id), `non-valid record ${rec.id} must not be referenced by the calculator`)
    }
})

test('service worker caches the cartridge assets', () => {
    const sw = fs.readFileSync(path.join(__dirname, 'sw.js'), 'utf8')
    for (const asset of [
        './cartridge-economics.html',
        './cartridge-economics.js',
        './cartridge-pricing.js',
        './cartridge-benchmarks.js',
    ]) {
        assert.ok(sw.includes(`'${asset}'`), `sw.js caches ${asset}`)
    }
})

test('index.html links the new page and its thumbnail exists', () => {
    const index = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8')
    assert.ok(index.includes('cartridge-economics.html'), 'index.html links cartridge-economics.html')
    assert.ok(index.includes('thumbnails/cartridge-economics.png'), 'index.html references the thumbnail')
    const thumb = path.join(__dirname, 'thumbnails', 'cartridge-economics.png')
    assert.ok(fs.existsSync(thumb), 'thumbnail file exists')
    assert.ok(fs.statSync(thumb).size > 0, 'thumbnail is not empty')
})

test('cartridge-economics.html has no duplicate element IDs', () => {
    const html = fs.readFileSync(path.join(__dirname, 'cartridge-economics.html'), 'utf8')
    const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1])
    const seen = new Set()
    for (const id of ids) {
        assert.ok(!seen.has(id), `duplicate element id: ${id}`)
        seen.add(id)
    }
    assert.ok(ids.length > 30, 'expected a substantial set of element ids')
})

test('blank self-study cost blocks construction cost instead of becoming $0', () => {
    assert.strictEqual(E.buildCostFromGpuHours(100, 2.69, null, 0), null)
    assert.strictEqual(E.buildCostFromMeasuredRun(8, 10, 2.69, null, 0), null)
    // Explicit zero (user-entered or "included in measured total") is allowed
    closeTo(E.buildCostFromGpuHours(100, 2.69, 0, 0), 269)
    closeTo(E.buildCostFromGpuHours(100, 2.69, 40, null), 309)
    closeTo(E.buildCostFromGpuHours(100, 2.69, 40, 11), 320)
})

test('missing measured values propagate as null (benchmark required), never zero', () => {
    assert.strictEqual(E.buildCostFromMeasuredRun(null, 10, 2.69, 0, 0), null)
    assert.strictEqual(E.buildCostFromGpuHours(null, 2.69, 0, 0), null)
    assert.strictEqual(E.prefillGpuMsSaved(null, 100, 1), null)
    assert.strictEqual(E.loadCostQuery(1e6, 4.4, 0, null, 0, 0), null)
    assert.strictEqual(E.netSavingQuery(null, 0.001), null)
    assert.strictEqual(E.netSavingPerLoad(0.01, 4.4, null), null)
})

test('retry overhead and monthly corpus churn feed recurring costs', () => {
    closeTo(E.effectiveBuildCost(1000, 0.2), 1200)
    closeTo(E.effectiveBuildCost(1000, 0), 1000)
    assert.strictEqual(E.effectiveBuildCost(null, 0.2), null)
    // 10% of the corpus retrained monthly recurs 10% of the build cost
    closeTo(E.recurringRebuildCostMonth(1200, 0.1), 120)
    closeTo(E.recurringRebuildCostMonth(1200, 0), 0)
    assert.strictEqual(E.recurringRebuildCostMonth(null, 0.1), null)
})

test('measured production cost is billed GPU-hours over completed requests', () => {
    // 10 billed GPU-hours at $2.69 over 20,000 completed requests
    closeTo(E.gpuCostPerQuery(10, 2.69, 20000), 26.9 / 20000)
    assert.strictEqual(E.gpuCostPerQuery(null, 2.69, 20000), null)
    assert.strictEqual(E.gpuCostPerQuery(10, 2.69, 0), null)
    assert.strictEqual(E.gpuCostPerQuery(10, null, 20000), null)
    closeTo(E.costPer1kToPerQuery(1.5), 0.0015)
    assert.strictEqual(E.costPer1kToPerQuery(null), null)
})

test('path cost delta is signed and null-propagating', () => {
    closeTo(E.pathCostDelta(0.002, 0.0005), 0.0015)
    // A more expensive Cartridge path is a negative delta, not zero
    closeTo(E.pathCostDelta(0.0005, 0.002), -0.0015)
    assert.strictEqual(E.pathCostDelta(null, 0.002), null)
    assert.strictEqual(E.pathCostDelta(0.002, null), null)
})

test('cold-load bytes, latency and minimum bandwidth', () => {
    // 90% hit rate on 4.4 unique 83.19-MiB Cartridges: ~36.6 MiB cold/query
    const bytesDoc = E.cartridgeBytesPerDocument(11800, 20, 141, 1.0)
    const cold = E.coldBytesPerQuery(bytesDoc, 4.4, 0.9)
    closeTo(cold / MIB, 36.6036, 1e-4)
    assert.strictEqual(E.coldBytesPerQuery(null, 4.4, 0.9), null)
    // 1 GB/s, 5 ms/object fixed, 0.44 cold objects: fixed + transfer time
    const ms = E.coldLoadMsPerQuery(cold, 0.44, 1, 5, false)
    closeTo(ms, 0.44 * 5 + (cold / 1e9) * 1000, 1e-6)
    // Overlapped loading hides the latency entirely
    assert.strictEqual(E.coldLoadMsPerQuery(cold, 0.44, 1, 5, true), 0)
    // Missing bandwidth blocks the estimate rather than guessing
    assert.strictEqual(E.coldLoadMsPerQuery(cold, 0.44, null, 5, false), null)
    // Minimum bandwidth for loading not to erase a 100 ms wall-time benefit
    const bw = E.minBandwidthGBps(cold, 0.44, 5, 100)
    closeTo(bw, cold / 1e9 / ((100 - 2.2) / 1000), 1e-6)
    // Fixed latency alone eating the whole benefit makes it unreachable
    assert.strictEqual(E.minBandwidthGBps(cold, 4.4, 30, 100), Infinity)
})

test('quality gate distinguishes unknown, pass and fail', () => {
    assert.strictEqual(E.qualityGate(null, null, 1), 'unknown')
    assert.strictEqual(E.qualityGate(60, null, 1), 'unknown')
    assert.strictEqual(E.qualityGate(60, 59.5, 1), 'pass')
    assert.strictEqual(E.qualityGate(60, 58.9, 1), 'fail')
    assert.strictEqual(E.qualityGate(60, 60, 0), 'pass')
})

test('prefill saving is signed: a slower Cartridge path is a loss, not zero', () => {
    // Text 100 ms, Cartridge 150 ms: -50 GPU-ms per GPU
    assert.strictEqual(E.prefillGpuMsSaved(100, 150, 1), -50)
    assert.strictEqual(E.prefillGpuMsSaved(100, 150, 2), -100)
    assert.strictEqual(E.prefillGpuMsSaved(150, 100, 2), 100)
})

test('negative per-query savings never break even and hit the monthly net in full', () => {
    const msSaved = E.prefillGpuMsSaved(100, 150, 1)
    const gpuValue = E.gpuTimeValueSavedQuery(msSaved, 0, 3.6)
    closeTo(gpuValue, (-50 / 3600000) * 3.6)
    const net = E.netSavingQuery(E.realizedComputeSavingQuery(gpuValue, 1), 0)
    assert.ok(net < 0)
    assert.strictEqual(E.breakEvenQueriesMonth(1000, 6, 10, net), Infinity)
    // Monthly net reflects the full negative delta, not a floor at zero
    closeTo(E.monthlyNetSaving(100000, net, 10), 100000 * net - 10)
    assert.ok(E.monthlyNetSaving(100000, net, 10) < 0)
})

test('decode delta is signed too', () => {
    // +100 prefill ms saved, -200 decode ms lost: net negative GPU-time value
    const v = E.gpuTimeValueSavedQuery(100, -200, 3.6)
    closeTo(v, (-100 / 3600000) * 3.6)
    assert.ok(v < 0)
})
