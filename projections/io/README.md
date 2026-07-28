# KV-Cache Offload Storage-IO Projections

The memory envelope answers "how big is the KV cache?". This dataset
answers the next question: **what does moving it to storage actually
cost?** — per model, with the calculator's exact geometry and with
measured store/load numbers from a real NVMe device.

## What is in here

- `kvio_io_projections.json` — one record per model:
  - `kv_block_bytes_256tok`: bytes for one 256-token KV block
    (LMCache's default offload chunk), from the same math as the
    KV Cache Calculator (`kv_geometry.py` port: MHA/GQA, `head_dim`,
    DeepSeek MLA, Hunyuan CLA)
  - `kv_bytes_per_token`: the block divided by 256 — the per-token
    storage footprint
  - `store/load_cmds_per_block`, `*_bytes_per_block`: the NVMe command
    stream one block becomes at the kernel's 128 KiB command quantum
  - `measured.store/load`: p50/p99 latency and throughput for
    storing/loading blocks on a real device (see method)
- `collect_io_projections.py` — merges the generator's outputs into
  this JSON

## Method

Generated with LMCache's GPU-free workload generator
([`examples/kv_cache_offload_io/run_kv_offload_io.py`](https://github.com/mcgrof/LMCache/tree/kvio/examples/kv_cache_offload_io)),
which drives the real `raw_block` engine with fake KV bytes — storage
IO geometry depends only on block size and transfer limits, not tensor
values, so real model dimensions + fake content reproduce the real
offload IO pattern:

```sh
run_kv_offload_io.py --model <hf-model-id> --dtype bfloat16 \
    --chunk-tokens 256 --num-chunks 8 --iters 3 --warmup 1 \
    --device <file-on-xfs> --engine io_uring --odirect \
    --record <model>.record.json
```

Measured on a Micron 7450 (960 GB) through an O_DIRECT file on xfs,
io_uring engine, one block in flight at a time — the numbers are
**per-block cost**, not peak device throughput (parallel loading
saturates the drive; that is a serving-stack property, not a model
property). Command counts use the 128 KiB quantum the kernel splits at
on this host (`max_sectors_kb`). Shared machine: treat throughput as
representative, not pristine. `--model` accepts any Hugging Face model
id; the catalog models need no network.

## Reading the data

Per-token storage footprint spans 16x across today's models — from
DeepSeek-V3's 68.6 KB/token (MLA's latent compression, visible as a
storage bill, not just a VRAM one) to Llama-3.1-405B's 504 KB/token —
and per-block load latency scales linearly with block size while
per-block throughput stays flat: the drive does not care which model
the bytes belong to, only how many bytes there are and how many
commands they become.
