"""Summarize sweep results into a CSV table.

Parses sweep JSON files and outputs: Config, CC, TPS/User, TPS/GPU, Cache Hit.
GPU count is derived from the filename (e.g., attn_tp4 -> 4 GPUs, attn_dp8 -> 8 GPUs).

Usage:
    python summarize_sweep.py agentic_workload/logs/
    python summarize_sweep.py agentic_workload/logs/ -o results.csv
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path


def infer_num_gpus(filename: str) -> int | None:
    """Derive GPU count from filename. The attn_{tp,dp}N field gives world size."""
    m = re.search(r"attn_(?:tp|dp)(\d+)", filename)
    if m:
        return int(m.group(1))
    return None


def extract_config(filename: str) -> str:
    """Extract config label like 'Attn TP4, MoE EP8' from filename."""
    m = re.search(r"(attn_\w+\d+)_(moe_\w+\d+)", filename)
    if not m:
        return filename
    attn_part = m.group(1).replace("attn_", "Attn ").upper().replace("ATTN ", "Attn ")
    moe_part = m.group(2).replace("moe_", "MoE ").upper().replace("MOE ", "MoE ")
    return f"{attn_part}, {moe_part}"


def process_file(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)

    filename = path.name
    num_gpus = infer_num_gpus(filename)
    config = extract_config(filename)

    rows = []
    for r in data["results"]:
        cc = r["num_clients"]
        total_tps = r.get("total_token_throughput", 0)
        p50_tpot = r.get("p50_TPOT", 0)
        user_tps = 1000.0 / p50_tpot if p50_tpot > 0 else 0
        cache_hit = r.get("cache_hit_rate", 0)

        # Skip failed runs
        if total_tps == 0:
            continue

        tps_gpu = total_tps / num_gpus if num_gpus else None

        rows.append(
            {
                "Config": config,
                "GPUs": num_gpus,
                "CC": cc,
                "TPS/User": round(user_tps, 1),
                "TPS/GPU": round(tps_gpu, 1) if tps_gpu is not None else "",
                "Cache Hit": f"{cache_hit:.1%}",
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Summarize sweep results to CSV.")
    parser.add_argument("path", type=Path, help="Directory containing sweep JSON files")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: stdout)",
    )
    args = parser.parse_args()

    json_files = sorted(args.path.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {args.path}", file=sys.stderr)
        sys.exit(1)

    all_rows = []
    for f in json_files:
        all_rows.extend(process_file(f))

    fieldnames = ["Config", "GPUs", "CC", "TPS/User", "TPS/GPU", "Cache Hit"]
    out = open(args.output, "w", newline="") if args.output else sys.stdout
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_rows)

    if args.output:
        out.close()
        print(f"Wrote {len(all_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
