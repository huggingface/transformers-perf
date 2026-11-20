"""Summarize a `tperf baselines run` results directory.

    tperf baselines summarize results/<run-dir>

Flags any measurement whose run-to-run coefficient of variation exceeds the threshold,
and any throttling that happened during a GPU's GEMM run.
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

COV_LIMIT_PCT = 1.0
COUNTER_NAMES = ["sw_power_cap", "sw_thermal", "hw_thermal", "hw_power_brake"]


def read_counters(path: Path) -> list[int]:
    fields = [f.strip() for f in path.read_text().strip().split(",")]
    return [int(f) if f.isdigit() else 0 for f in fields[1:]]


def summarize(run_dir: Path) -> int:
    noisy = 0

    print(f"GEMM throughput (TFLOPS), CoV limit {COV_LIMIT_PCT}%")
    print(f"{'gpu':>3} {'node':>4} {'dtype':>5} {'median':>8} {'min':>8} {'max':>8} {'cov%':>6}  throttled during run")
    rows = [json.loads(line) for line in (run_dir / "gemm.jsonl").read_text().splitlines() if line.strip()]
    by_dtype: dict[str, list[float]] = {}
    for r in rows:
        g = r["gpu"]
        before = read_counters(run_dir / f"gpu{g}-counters-before.csv")
        after = read_counters(run_dir / f"gpu{g}-counters-after.csv")
        throttled = (
            ", ".join(f"{name} {(a - b) / 1e6:.1f}s" for name, a, b in zip(COUNTER_NAMES, after, before) if a - b > 0)
            or "-"
        )
        flag = "  <-- noisy" if r["cov_pct"] > COV_LIMIT_PCT else ""
        noisy += bool(flag)
        by_dtype.setdefault(r["dtype"], []).append(r["tflops_median"])
        print(
            f"{g:>3} {r['numa_node']:>4} {r['dtype']:>5} {r['tflops_median']:>8.2f} {r['tflops_min']:>8.2f}"
            f" {r['tflops_max']:>8.2f} {r['cov_pct']:>6.2f}  {throttled}{flag}"
        )
    for dtype, medians in by_dtype.items():
        spread = 100 * (max(medians) - min(medians)) / statistics.median(medians)
        print(f"    {dtype}: spread of medians across GPUs {spread:.2f}%")

    runs = sorted(run_dir.glob("nvbandwidth-*run*.txt"))
    if runs:
        sums: dict[str, list[float]] = {}
        for f in runs:
            node = re.search(r"node(\d+)", f.name)
            prefix = f"node{node.group(1)} " if node else ""
            for m in re.finditer(r"^SUM (\S+) ([\d.]+)", f.read_text(), re.M):
                sums.setdefault(prefix + m.group(1), []).append(float(m.group(2)))
        print(f"\nnvbandwidth SUM over all GPUs (GB/s, or ns for latency), {len(runs)} files")
        if not sums:
            print("    no 'SUM <test> <value>' lines found: check the raw nvbandwidth-run*.txt files")
        for test, vals in sums.items():
            cov = 100 * statistics.stdev(vals) / statistics.fmean(vals) if len(vals) > 1 else 0.0
            flag = "  <-- noisy" if cov > COV_LIMIT_PCT else ""
            noisy += bool(flag)
            print(f"    {test:<42} median {statistics.median(vals):>9.2f}  cov {cov:5.2f}%{flag}")

    print(f"\n{'QUIET' if noisy == 0 else f'NOISY: {noisy} measurement(s) above {COV_LIMIT_PCT}% CoV'}")
    return 0 if noisy == 0 else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    return summarize(p.parse_args(argv).run_dir)


if __name__ == "__main__":
    sys.exit(main())
