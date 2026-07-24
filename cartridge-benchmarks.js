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
            date: '2026-07-24',
            model: 'Qwen3-8B',
            dataset: 'LongHealth patient records',
            corpusTokens: null,
            docTokens: 7681,
            cartridgeTokens: 512,
            compression: 15.0,
            serializedFormat: 'bf16',
            serializedBytesDoc: 75546893,
            harness: 'knlp research/cartridges_cas (HazyResearch cartridges @ 8cb6823)',
            engine: 'vLLM Qwen3-8B teacher (self-study synth) + FlexQwen3 train path',
            hardware: '8x H100 80GB',
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
            qualityMetric: 'LongHealth accuracy',
            qualityBaseline: null,
            qualityCartridge: null,
            notes:
                'In-progress CAS (arXiv:2606.04557) reproduction on Qwen3-8B. Measured so far: ' +
                'cartridgeTokens (511 trained + 1 frozen sink token), serializedFormat/serializedBytesDoc ' +
                'from a trained isolated Cartridge on disk (bf16), and docTokens/compression from a LongHealth ' +
                'patient record. Still null (benchmark required, never estimated): constructionGpuHours, ' +
                'selfStudyCostUsd and otherBuildCostUsd await the running six-patient x 20000-convo self-study ' +
                'synth and Cartridge training; qualityBaseline/qualityCartridge await the combine-eval; the ' +
                'whole serving side (baselineRun/cartridgeRun, ttftMs, throughputQps, loadPath) awaits a ' +
                'serving A/B microbenchmark not yet run. Promote to valid once the build cost and Cartridge ' +
                'quality are measured. This is the 8B build the invalid 0.6B record explicitly could not estimate.',
        },
    ],
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = CARTRIDGE_BENCHMARKS
}
