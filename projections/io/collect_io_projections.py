#!/usr/bin/env python3
"""Merge kvio sweep outputs into the kvcache-view IO projections JSON."""
import json
import re
import sys
from pathlib import Path

d = Path(sys.argv[1])
maxctx = {}
mc = d / "max_context.json"
if mc.exists():
    maxctx = json.loads(mc.read_text())
released = {}
rd = d / "release_dates.json"
if rd.exists():
    released = json.loads(rd.read_text())
models = []
for rec_path in sorted(d.glob("*.record.json")):
    tag = rec_path.name.replace(".record.json", "")
    rec = json.loads(rec_path.read_text())
    out = (d / f"{tag}.stdout").read_text()
    g = rec["geometry"]
    dg = rec["device_geometry"]
    m = {
        "model": rec["model"],
        "family": g.get("family"),
        "kv_block_bytes_256tok": rec["chunk_block_bytes"],
        "kv_bytes_per_token": rec["chunk_block_bytes"] / 256,
        "dtype": "bfloat16",
        "store_cmds_per_block": None,
        "store_bytes_per_block": None,
        "load_cmds_per_block": None,
        "load_bytes_per_block": None,
        "measured": {},
    }
    pm = re.search(r"store (\d+) cmds / (\d+) B, load (\d+) cmds / (\d+) B", out)
    if pm:
        m["store_cmds_per_block"] = int(pm.group(1))
        m["store_bytes_per_block"] = int(pm.group(2))
        m["load_cmds_per_block"] = int(pm.group(3))
        m["load_bytes_per_block"] = int(pm.group(4))
    for op in ("store", "load"):
        lm = re.search(
            rf"{op}\s*: p50\s+([\d.]+) ms\s+p99\s+([\d.]+) ms \|\s+([\d.]+) MB/s",
            out)
        if lm:
            m["measured"][op] = {
                "p50_ms": float(lm.group(1)),
                "p99_ms": float(lm.group(2)),
                "MBps": float(lm.group(3)),
            }
    m["cmd_bytes"] = dg.get("mdts_bytes")
    # Per-command IO size envelope. Store issues a 4 KiB header command
    # plus the payload split; load is the payload split alone, whose
    # smallest command is the block's tail (aligned up to 4 KiB).
    q = int(dg.get("mdts_bytes") or 131072)
    ba = int(dg.get("block_align") or 4096)
    blk = int(rec["chunk_block_bytes"])
    tail = blk % q
    tail_aligned = ((tail + ba - 1) // ba) * ba if tail else 0
    m["load_min_io_bytes"] = tail_aligned or min(q, blk)
    m["load_max_io_bytes"] = min(q, blk)
    m["store_min_io_bytes"] = int(dg.get("header_bytes") or ba)
    m["store_max_io_bytes"] = min(q, blk)
    mcv = maxctx.get(rec["model"])
    m["max_context_tokens"] = mcv if isinstance(mcv, int) else None
    m["released"] = released.get(rec["model"])  # HF repo createdAt (YYYY-MM-DD)
    models.append(m)

models.sort(key=lambda m: m["kv_block_bytes_256tok"])
doc = {
    "schema": 1,
    "what": "Per-model KV-cache offload storage-IO projections: the "
            "calculator's exact block geometry plus measured store/load on "
            "real NVMe",
    "method": {
        "generator": "run_kv_offload_io.py from the kvio branch of "
                     "mcgrof/LMCache (not yet upstream; kv_geometry.py "
                     "is the kv_cache_calculator math)",
        "generator_url": "https://github.com/mcgrof/LMCache/tree/kvio/"
                         "examples/kv_cache_offload_io",
        "kvio_project_url": "https://github.com/SamsungDS/ebpf-syscall/"
                            "tree/ebpf-fixes",
        "kvio_docs_url": "https://htmlpreview.github.io/?https://"
                         "github.com/SamsungDS/ebpf-syscall/blob/"
                         "ebpf-fixes/docs/kvio.html",
        "chunk_tokens": 256,
        "dtype": "bfloat16",
        "engine": "io_uring, O_DIRECT, file on xfs",
        "command_bytes": 131072,
        "command_note": "the kernel splits file I/O at max_sectors_kb "
                        "(128 KiB on this host); projected command counts "
                        "use the same 128 KiB quantum",
        "measured_workload": "8 blocks stored then loaded, 3 measured "
                             "passes after 1 warmup pass, one object in "
                             "flight (per-object cost, not peak device "
                             "throughput)",
        "device": "Micron 7450 (MTFDKBG960TFR 960 GB), shared host -- "
                  "treat measured numbers as representative, not pristine",
    },
    "transports": [
        {
            "name": "Linux kernel: POSIX / io_uring file / io_uring_cmd",
            "path": "application -> kernel -> NVMe",
            "wire_quantum": "<= 128 KiB per command on translating-IOMMU "
                            "hosts (nvme-pci clamps max_hw_sectors to "
                            "dma_opt_mapping_size = the IOVA rcache range; "
                            "host-dependent, not a device property)",
            "status": "measured",
            "notes": "The per-model table on this page was measured on "
                     "this path (io_uring file engine, O_DIRECT). Command "
                     "size sweeps on the same drive: reads scale 2,502 -> "
                     "3,727 MB/s from 16 K to 128 K at QD 16, writes flat "
                     "~1.5 GB/s at every size and depth tested.",
        },
        {
            "name": "SPDK (vfio-pci, userspace NVMe driver)",
            "path": "application -> vfio-mapped device queues",
            "wire_quantum": "device MDTS (512 KiB on the same Micron 7450 "
                            "-- 4x the kernel path); vfio maps the memory "
                            "pool once at registration, so the per-IO "
                            "DMA-mapping clamp never applies; larger "
                            "requests are software-split by SPDK",
            "status": "measured",
            "notes": "Same drive model, AMD IOMMU enabled, vfio-pci with "
                     "full translation: identify reports MDTS 512 KiB "
                     "where the kernel reports 128; 512 KiB wire commands "
                     "sustain 3,565 MB/s reads (bandwidth-flat from "
                     "128 K), writes 1,456 MB/s.",
        },
        {
            "name": "cuFile / GPUDirect Storage (NIXL GDS backend)",
            "path": "GPU <-> NVMe via nvidia-fs through the kernel driver",
            "wire_quantum": "<= 128 KiB at the device on IOMMU hosts (same "
                            "kernel clamp); the library adds its own "
                            "segmentation and bounce-pool behavior above it",
            "status": "measured (compat path)",
            "notes": "Single-stream KV-block loads 1.6-1.7 GB/s measured "
                     "through the generator's cufile/opends engines on a "
                     "Gen5 drive; GPU-direct p2p path numbers pending.",
        },
        {
            "name": "NIXL XNVME_KV (io_uring_cmd, NVMe-KV namespaces)",
            "path": "NIXL -> xNVMe -> kernel passthrough -> KV SSD",
            "wire_quantum": "value size per KV command, bounded by the "
                            "same kernel limits as passthrough",
            "status": "attribution measured, throughput pending",
            "notes": "Traced completely unmodified with per-object "
                     "attribution (the object key rides in each command); "
                     "dedicated throughput benchmarking not yet run.",
        },
        {
            "name": "NIXL UCX over NVLink (intra-node GPU <-> GPU)",
            "path": "GPU memory <-> GPU memory, no storage device",
            "wire_quantum": "no NVMe command quantum -- NVLink transfer "
                            "sizes are a different regime entirely",
            "status": "pending",
            "notes": "KV movement without storage: different limits, "
                     "different instrumentation (no NVMe tracer applies).",
        },
        {
            "name": "NIXL UCX over RDMA (inter-node)",
            "path": "initiator NIC <-> remote node; storage commands, if "
                    "any, occur on the remote target",
            "wire_quantum": "RDMA message sizing on the wire; NVMe "
                            "quanta apply only on the target node",
            "status": "pending",
            "notes": "Initiator-side kernel NVMe tracing sees nothing; "
                     "the join strategy is target-side capture plus NIXL "
                     "telemetry as the semantic layer.",
        },
    ],
    "models": models,
}
print(json.dumps(doc, indent=2))
