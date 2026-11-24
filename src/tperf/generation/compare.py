"""Compare several harness runs (from `tperf generation run`) for run-to-run stability.

    tperf generation compare results/<stamp>-tbv2/run1 results/<stamp>-tbv2/run2 ...

Per scenario: median end-to-end latency, time to first token, inter-token latency and tokens/s
for each run, with the coefficient of variation across runs and the worst within-run CoV.
Then the noise probe counters (tperf.common.noise_probe) for each run. Exits 1 if any metric is above the CoV limit.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

COV_LIMIT_PCT = 1.0


def cov(vals: list[float]) -> float:
    return 100 * statistics.stdev(vals) / statistics.fmean(vals) if len(vals) > 1 and statistics.fmean(vals) else 0.0


def load(run_dir: Path) -> dict[str, dict[str, float]]:
    """{scenario name: {metric: median}} for one run, plus within-run CoV of e2e latency."""
    out = {}
    for f in run_dir.glob("*/*_benchmark_*.json"):
        for entry in json.loads(f.read_text()).values():
            if not entry["metadata"].get("success"):
                continue
            cfg, m = entry["config"], entry["measurements"]
            tokens = cfg["batch_size"] * cfg["num_tokens_to_generate"]
            e2e = m["e2e_latency"]
            out[cfg["name"]] = {
                "e2e_ms": 1e3 * statistics.median(e2e),
                "ttft_ms": 1e3 * statistics.median(m["time_to_first_token"]),
                "itl_ms": 1e3 * statistics.median(m["inter_token_latency"]),
                "tok_s": statistics.median(tokens / x for x in e2e),
                "within_cov": cov(e2e),
            }
    return out


def short(name: str) -> str:
    """Drop the parts of the scenario name that are identical in every scenario."""
    keep = [p for p in name.split("-") if not p.startswith(("w", "monitored", "unmonitored", "no_tp"))]
    return "-".join(keep)


def compare(dirs: list[Path]) -> int:
    runs = [load(d) for d in dirs]
    scenarios = sorted(set().union(*runs))
    if not scenarios:
        print("no successful scenarios found")
        return 1
    print(f"{len(dirs)} runs, CoV limit across runs {COV_LIMIT_PCT}%\n")
    unstable = 0
    for s in scenarios:
        print(short(s))
        for metric, unit in [("e2e_ms", "ms"), ("ttft_ms", "ms"), ("itl_ms", "ms"), ("tok_s", "tok/s")]:
            vals = [r[s][metric] for r in runs if s in r]
            c = cov(vals)
            flag = "  <-- unstable" if c > COV_LIMIT_PCT else ""
            unstable += bool(flag)
            cells = " ".join(f"{v:>9.2f}" for v in vals)
            print(f"   {metric:<8} {cells}  {unit:<6} cov {c:5.2f}%{flag}")
        worst = max(r[s]["within_cov"] for r in runs if s in r)
        print(f"   worst within-run e2e CoV: {worst:.2f}%")

    print("\nnoise probe (whole run)")
    print(
        f"   {'run':<6} {'elapsed s':>9} {'foreign cpu s':>13} {'steal s':>8} {'invol. csw':>10} {'irqs on cpus':>12}"
        f" {'gpu util %':>10} {'power W':>8} {'throttle ms':>12}"
    )
    for d in dirs:
        p = d / "probe.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        c, g = r["cpu"], r.get("gpu", {})
        thr = sum(g.get("throttle_ms", {}).values())
        print(
            f"   {d.name:<6} {r['elapsed_s']:>9.1f} {c['foreign_cpu_s']:>13.2f}"
            f" {c.get('steal_s_on_cpus', float('nan')):>8.2f} {c['involuntary_ctx_switches']:>10}"
            f" {c['interrupts_on_cpus']:>12} {g.get('util_mean_pct', float('nan')):>10.1f}"
            f" {g.get('avg_power_w', float('nan')):>8.1f} {thr:>12.1f}"
        )

    print(
        f"\n{'STABLE' if unstable == 0 else f'UNSTABLE: {unstable} metric(s) above {COV_LIMIT_PCT}% CoV across runs'}"
    )
    return 0 if unstable == 0 else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="+", type=Path)
    return compare(p.parse_args(argv).run_dirs)


if __name__ == "__main__":
    sys.exit(main())
