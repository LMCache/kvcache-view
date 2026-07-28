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
    models.append(m)

models.sort(key=lambda m: m["kv_block_bytes_256tok"])
doc = {
    "schema": 1,
    "what": "Per-model KV-cache offload storage-IO projections: the "
            "calculator's exact block geometry plus measured store/load on "
            "real NVMe",
    "method": {
        "generator": "LMCache examples/kv_cache_offload_io/"
                     "run_kv_offload_io.py (kv_geometry.py = the "
                     "kv_cache_calculator math)",
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
    "models": models,
}
print(json.dumps(doc, indent=2))
