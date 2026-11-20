"""Compare several `tperf baselines run` runs to check that results are stable run to run.

    tperf baselines compare results/<run-dir> results/<run-dir> ...

For every GPU x dtype GEMM median and every nvbandwidth test total, prints the value from
each run and the coefficient of variation across runs. Exits 1 if anything is above the limit.
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

COV_LIMIT_PCT = 1.0


def cov(vals: list[float]) -> float:
    return 100 * statistics.stdev(vals) / statistics.fmean(vals) if len(vals) > 1 else 0.0


def load(run_dir: Path) -> dict[str, float]:
    """Flatten one run into {metric name: value}."""
    out: dict[str, float] = {}
    for line in (run_dir / "gemm.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[f"gemm gpu{r['gpu']} {r['dtype']} TFLOPS"] = r["tflops_median"]
    per_test: dict[str, list[float]] = {}
    for f in sorted(run_dir.glob("nvbandwidth-*run*.txt")):
        node = re.search(r"node(\d+)", f.name)
        prefix = f"node{node.group(1)} " if node else ""
        for m in re.finditer(r"^SUM (\S+) ([\d.]+)", f.read_text(), re.M):
            per_test.setdefault(prefix + m.group(1), []).append(float(m.group(2)))
    for test, vals in per_test.items():
        out[f"nvbw {test}"] = statistics.median(vals)
    return out


def compare(dirs: list[Path]) -> int:
    runs = [load(d) for d in dirs]
    metrics = sorted(set().union(*runs))
    print(f"{len(dirs)} runs: " + ", ".join(d.name for d in dirs))
    print(f"CoV limit across runs: {COV_LIMIT_PCT}%\n")
    width = max(len(m) for m in metrics)
    print(f"{'metric':<{width}}  " + " ".join(f"{'run' + str(i + 1):>9}" for i in range(len(runs))) + f"  {'cov%':>6}")
    noisy = 0
    for m in metrics:
        vals = [r.get(m) for r in runs]
        present = [v for v in vals if v is not None]
        c = cov(present)
        flag = "  <-- unstable" if c > COV_LIMIT_PCT else ""
        if len(present) < len(runs):
            flag += "  (missing in some runs)"
        noisy += c > COV_LIMIT_PCT
        cells = " ".join(f"{v:>9.2f}" if v is not None else f"{'-':>9}" for v in vals)
        print(f"{m:<{width}}  {cells}  {c:>6.2f}{flag}")
    print(f"\n{'STABLE' if noisy == 0 else f'UNSTABLE: {noisy} metric(s) above {COV_LIMIT_PCT}% CoV across runs'}")
    return 0 if noisy == 0 else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="+", type=Path)
    dirs = p.parse_args(argv).run_dirs
    if len(dirs) < 2:
        p.error("needs two run directories or more")
    return compare(dirs)


if __name__ == "__main__":
    sys.exit(main())
