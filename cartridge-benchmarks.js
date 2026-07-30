// Benchmark records for the Cartridge Economics calculator.
//
// Static, PWA-compatible data: every measurement that feeds (or is refused
// entry into) the calculator is recorded here with its provenance, validity
// and known defects, so numbers never circulate detached from how they were
// produced. Records marked validity 'invalid' or 'exploratory' MUST NOT be
// surfaced as presets; only 'valid' records may seed calculator fields.
//
// Field reference (one record):
//   id                    unique slug
//   validity              'valid' | 'exploratory' | 'invalid'
//   provenance            who ran it, where
//   date                  ISO date of the run, null if unknown
//   model                 exact model/checkpoint
//   dataset               dataset / corpus identity
//   corpusTokens          source corpus tokens
//   docTokens             avg source tokens per document
//   cartridgeTokens       trained Cartridge tokens per document
//   compression           source-to-Cartridge token ratio
//   serializedFormat      on-disk KV format actually written
//   serializedBytesDoc    measured serialized bytes per Cartridge
//   harness               training/eval harness and commit
//   engine                inference engine and commit
//   hardware              GPUs used
//   constructionGpuHours  measured GPU-hours for the whole corpus build
//   selfStudyCostUsd      measured self-study generation + inference cost
//   otherBuildCostUsd     other one-time construction cost
//   requestShape          input/query/output token counts
//   concurrency           concurrency and batching policy
//   baselineRun           {gpuHours, completedRequests} for the baseline path
//   cartridgeRun          {gpuHours, completedRequests} for the Cartridge path
//   loadPath              storage tier and staging path, hit rate
//   ttftMs                {baselineP50, baselineP95, cartridgeP50, cartridgeP95}
//   throughputQps         steady-state completed requests per second
//   qualityMetric         metric name
//   qualityBaseline       baseline score
//   qualityCartridge      Cartridge score
//   notes                 free text: methodology, known defects
//
// Unknown values are null, never zero — the same discipline as the
// calculator itself.

const CARTRIDGE_BENCHMARKS = {
    schemaVersion: 1,
    records: [
        {
            id: 'local-qwen3-0.6b-frozen-sink-mismatch',
            validity: 'invalid',
            provenance: 'Local exploratory run (kvcache-view author)',
            date: null,
            model: 'Qwen3-0.6B',
            dataset: null,
            corpusTokens: null,
            docTokens: null,
            cartridgeTokens: null,
            compression: null,
            serializedFormat: 'bf16',
            serializedBytesDoc: null,
            harness: null,
            engine: null,
            hardware: null,
            constructionGpuHours: null,
            selfStudyCostUsd: null,
            otherBuildCostUsd: null,
            requestShape: null,
            concurrency: null,
            baselineRun: null,
            cartridgeRun: null,
            loadPath: null,
            ttftMs: null,
            throughputQps: null,
            qualityMetric: 'accuracy',
            qualityBaseline: null,
            qualityCartridge: 0,
            notes:
                'INVALID: the Cartridge was trained with a frozen attention-sink prefix that was missing at ' +
                'evaluation, so the 0% accuracy measures the harness defect, not Cartridge quality. Neither its ' +
                'quality nor its timing may be used as economic evidence or as a calculator preset. Supersede ' +
                'this record with a corrected rerun (frozen sink present at eval) before citing any 0.6B number; ' +
                'a corrected 0.6B run validates instrumentation and cost accounting only — it does not estimate ' +
                '8B economics.',
        },
        {
            id: 'qwen3-8b-longhealth-cas-build',
            validity: 'exploratory',
            provenance: 'knlp CAS reproduction harness, 8x H100 80GB',
            date: '2026-07-30',
            model: 'Qwen3-8B',
            dataset: 'LongHealth patient records (patient_01, DLBCL)',
            corpusTokens: null,
            docTokens: 12221,
            cartridgeTokens: 611,
            compression: 20.0,
            serializedFormat: 'bf16',
            serializedBytesDoc: 90735203,
            harness: 'knlp research/cartridges_cas (HazyResearch cartridges @ 8cb6823)',
            engine: 'vLLM Qwen3-8B teacher (self-study synth) + FlexQwen3 train/serve path',
            hardware: '8x H100 80GB',
            constructionGpuHours: null,
            selfStudyCostUsd: null,
            otherBuildCostUsd: null,
            requestShape:
                'baseline prefills ~12221 document + ~131 query tokens; cartridge is a ~611-token ' +
                'KV prefix + ~131 query tokens; 32 output tokens',
            concurrency: 'single stream (latency microbenchmark; not a concurrent load test)',
            baselineRun: null,
            cartridgeRun: null,
            loadPath: null,
            ttftMs: {
                baselineP50: 757.0,
                baselineP95: 773.4,
                cartridgeP50: 80.6,
                cartridgeP95: 94.8,
            },
            throughputQps: null,
            qualityMetric: 'LongHealth option-match accuracy (thinking-on, temp 0.6, mean of >=3 runs)',
            qualityBaseline: 0.855,
            qualityCartridge: 0.50,
            notes:
                'CAS (arXiv:2606.04557) reproduction on Qwen3-8B over LongHealth. Baselines reproduce: ' +
                'no-context 0.39 (paper 0.375), full document in context 0.855 (paper 0.874). Cartridge-path ' +
                'ceiling: an untrained cartridge holding the full document KV, loaded through the cartridge ' +
                'path, scores 0.86 -- so that path has no positional/serialization loss and a perfect ' +
                'cartridge tops out at 0.86. A trained single isolated cartridge reaches 0.50 (best 0.58) ' +
                'against the paper 0.736; the collapse/rescue effect reproduces at 5 cartridges (isolated ' +
                '0.58 alone -> 0.38 co-loaded; mixed-visibility 0.44 -> 0.46). qualityCartridge here is the ' +
                'trained single-cartridge accuracy (0.50); qualityBaseline is the full document in context ' +
                '(0.855). Geometry/bytes are the trained ~611-token cartridge on disk (bf16); the serving ' +
                'A/B timing was collected with an earlier 512-token cartridge -- the prefill saving is a ' +
                'memcpy of the KV prefix and is insensitive to the small token-count difference, dropping ' +
                'TTFT from 757 ms to 81 ms median (677 ms/query) at ~35 tok/s decode both paths. Still null ' +
                '(measured, never estimated): constructionGpuHours/selfStudyCostUsd (no clean end-to-end ' +
                'build-cost measurement yet); throughputQps/loadPath (no concurrent load test yet). Stays ' +
                'exploratory: the trained-cartridge quality is below the paper and below the 0.86 path ' +
                'ceiling, so it is not a deployable preset. Method, deltas against the public Cartridges ' +
                'implementation, and where the remaining gap lives are documented at ' +
                'https://mcgrof.github.io/knlp/cas.html',
        },
    ],
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = CARTRIDGE_BENCHMARKS
}
