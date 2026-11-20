"""Summarize a `tperf baselines p2p` results directory.

    tperf baselines summarize-p2p results/<run-dir>

Per GPU group: the NCCL transports used, then per (op, size) the median latency over runs, the run-to-run CoV of
that median, and the bus bandwidth. Flags groups whose latency varies by more than COV_LIMIT_PCT between runs.
"""

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

COV_LIMIT_PCT = 1.0


def human(n: int) -> str:
    for shift, unit in ((30, "GiB"), (20, "MiB"), (10, "KiB")):
        if n >= 1 << shift:
            return f"{n / (1 << shift):g} {unit}"
    return f"{n} B"


def transports(log: str) -> list[str]:
    # e.g. "Channel 00 : 0[0] -> 1[1] via SHM/direct/direct"; keep the distinct "<path> via <transport>" kinds
    found = set()
    for m in re.finditer(r"Channel \d+/?\d* : (\d+)\[\w+\] -> (\d+)\[\w+\].*? via (\S+)", log):
        found.add(m.group(3))
    return sorted(found)


def summarize(run_dir: Path) -> int:
    noisy = 0
    groups = sorted(
        {p.name.rsplit("-run", 1)[0] for p in run_dir.glob("g*-run*.jsonl")},
        key=lambda g: (g.count("-"), g),
    )
    for group in groups:
        rows: dict[tuple[str, int], list[dict]] = defaultdict(list)
        logs = ""
        for path in sorted(run_dir.glob(f"{group}-run*.jsonl")):
            for line in path.read_text().splitlines():
                if line.startswith("{"):
                    row = json.loads(line)
                    rows[(row["op"], row["bytes"])].append(row)
            for log in path.parent.glob(f"{path.stem}.rank*.log"):
                logs += log.read_text()
        gpus = group[1:].replace("-", ",")
        print(f"\nGPUs {gpus}: transports {', '.join(transports(logs)) or 'unknown (no NCCL_DEBUG lines)'}")
        print(f"{'op':>10} {'size':>9} {'runs':>4} {'us':>9} {'min':>9} {'max':>9} {'cov%':>6} {'busbw GB/s':>10}")
        group_noisy = False
        for (op, nbytes), rs in sorted(rows.items()):
            meds = [r["us_median"] for r in rs]
            med = statistics.median(meds)
            cov = statistics.stdev(meds) / statistics.mean(meds) * 100 if len(meds) > 1 else 0.0
            busbw = statistics.median(r["busbw_gbs"] for r in rs)
            flag = " *" if cov > COV_LIMIT_PCT else ""
            group_noisy |= bool(flag)
            print(
                f"{op:>10} {human(nbytes):>9} {len(rs):>4} {med:>9.1f} {min(meds):>9.1f} {max(meds):>9.1f} "
                f"{cov:>6.2f} {busbw:>10.2f}{flag}"
            )
        noisy += group_noisy
    print(f"\n* run-to-run CoV above {COV_LIMIT_PCT}%")
    print("STABLE" if not noisy else f"NOISY: {noisy} group(s) above the limit")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    return summarize(p.parse_args(argv).run_dir)


if __name__ == "__main__":
    sys.exit(main())
