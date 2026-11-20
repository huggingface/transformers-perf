"""Summarize `tperf processors run` results: per scenario, the median of the run medians, each run's median, how much the
runs differ (CoV of the run medians, flagged above 1%), the noise within a run, whether every run returned the same
outputs, and for encoder stages the GPU -> CPU syncs per call.

A run whose median is more than SLOW times the scenario's median is counted as slow and left out of the CoV: the whole
process was slowed down (another process on its cores, the CPU clock), not the code under test. The median of the run
medians already ignores it, with 3 runs or more.

Usage: tperf processors summarize results/<timestamp>-processors
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


SLOW = 1.5


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results_dir", type=Path)
    root = p.parse_args(argv).results_dir
    runs: dict[str, list[dict]] = defaultdict(list)
    for f in sorted(root.glob("t*/run*/processor_*.json")):
        r = json.loads(f.read_text())
        runs[r["name"]].append(r)
    print(
        f"{'scenario':<80} {'median ms':>10}  {'runs (ms)':<26} {'CoV runs':>8} {'slow':>4} {'CoV in run':>10}"
        f"  outputs  syncs"
    )
    for name, results in runs.items():
        ok = [r for r in results if r["success"]]
        if len(ok) < len(results):
            print(f"{name:<80} FAILED: {next(r['error'] for r in results if not r['success'])[:120]}")
            continue
        medians = [1e3 * statistics.median(r["samples_s"]) for r in ok]
        median = statistics.median(medians)
        normal = [m for m in medians if m <= SLOW * median]
        cov = 100 * statistics.stdev(normal) / statistics.mean(normal) if len(normal) > 1 else 0.0
        within = statistics.median(
            100 * statistics.stdev(r["samples_s"]) / statistics.mean(r["samples_s"]) for r in ok
        )
        same = len({r["output"]["sha256"] for r in ok}) == 1
        syncs = sorted({r["syncs"]["count"] for r in ok if "syncs" in r})
        print(
            f"{name:<80} {median:10.2f}  {' '.join(f'{m:.1f}' for m in medians):<26} {cov:7.2f}%"
            f"{'!' if cov > 1 else ' '} {len(medians) - len(normal):4d} {within:9.2f}%  "
            f"{'same  ' if same else 'DIFFER'}   {'/'.join(map(str, syncs)) if syncs else '-'}"
        )


if __name__ == "__main__":
    main()
