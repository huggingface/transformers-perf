"""Compare the two sides of a `tperf ab` run, round by round.

    tperf ab compare results/ab-<id> [--threshold 1.0] [--rounds]

Each round ran base and head at the same time on two GPUs, swapping GPUs every round. For every scenario and metric,
the per-round log ratio head/base is fitted as `change + gpu_effect * orientation`: `change` is the PR's effect with
the GPU difference taken out, with a 95% confidence interval from the spread between rounds (t distribution). An A/B on
one GPU ran the sides in turn, base first in odd rounds: there the orientation is which side ran first, and the fitted
`gpu` column is the run-order effect.

Verdict per metric, with T the threshold (%):
  REGRESSION / improvement  the whole interval is worse / better than T
  no change                 the whole interval is within +-T
  inconclusive              the interval crosses +-T (more rounds would narrow it)
A difference whose interval excludes 0 but stays within +-T is marked "(significant, below threshold)".
Exit status 1 if any metric is a REGRESSION.
"""

from __future__ import annotations


import argparse
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


from tperf.common.record import scenario_rows

# metric -> (row column, higher is better)
METRICS = {
    "e2e": ("e2e_ms_median", False),
    "ttft": ("ttft_ms_median", False),
    "itl": ("itl_ms_median", False),
    "itl_h1": ("itl_first_half_ms_median", False),  # first half of the generated tokens
    "itl_h2": ("itl_second_half_ms_median", False),  # second half
    "tok/s": ("tokens_per_s_median", True),
    # Collected outside the measurements, once per run (harness startup metrics)
    "load": ("startup_load_time_s", False),
    "cold": ("startup_first_generate_ms", False),  # first generation: compilation, CUDA graphs
    "mem": ("startup_measurement_peak_memory_mb", False),  # peak memory during the measurements
    # Memory read at phase boundaries (tperf.harness.framework.memory_metrics)
    "weights": ("startup_weights_mb", False),  # the model's own tensors
    "gpu ws": ("startup_measurement_working_set_mb", False),  # peak above the weights: KV cache, activations
    "reserved": ("startup_measurement_peak_reserved_mb", False),  # the caching allocator's peak reserve
    "gpu proc": ("startup_process_gpu_mb", False),  # what NVML charges the process (CUDA context included)
    "load rss": ("startup_load_host_peak_rss_mb", False),  # host peak RSS while loading
}
# Counted, not timed: printed per side (median over rounds) rather than compared as ratios
COUNTERS = {
    "compiled graphs": "compiled_graphs",
    "graph breaks": "graph_breaks",
    "alloc retries": "alloc_retries",  # first generation / measurement (no warmup value)
}
# Two-sided 95% t quantiles by degrees of freedom
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def t95(df: int) -> float:
    return T95.get(df, 2.2 if df <= 15 else 2.1 if df <= 25 else 2.0)


def fit(y: list[float], o: list[float]) -> tuple[float, float, float]:
    """Least squares y = change + gpu * o. Returns (change, half width of its 95% CI, gpu)."""
    n = len(y)
    if n >= 3 and len(set(o)) == 2:
        mo, my = statistics.fmean(o), statistics.fmean(y)
        sxx = sum((x - mo) ** 2 for x in o)
        gpu = sum((x - mo) * (v - my) for x, v in zip(o, y)) / sxx
        change = my - gpu * mo
        resid = [v - change - gpu * x for x, v in zip(o, y)]
        df = n - 2
        s2 = sum(r * r for r in resid) / df if df else 0.0
        se = math.sqrt(s2 * (1 / n + mo * mo / sxx))
        return change, (t95(df) * se if df else math.inf), gpu
    change = statistics.fmean(y)
    se = statistics.stdev(y) / math.sqrt(n) if n > 1 else math.inf
    return change, t95(n - 1) * se if n > 1 else math.inf, 0.0


def pct(log_ratio: float) -> float:
    return 100 * math.expm1(log_ratio)


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not math.isnan(x) and x > 0


def analyze(rows: list[dict], threshold: float = 1.0) -> dict:
    """Paired analysis of one A/B's rows (from scenario_rows or the dataset). Returns
    {"gpus", "scenarios": [{"scenario", "rounds", "pairs", "metrics": [...], "counters": {...}}], "failed", "regression"}."""
    runs: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    failed: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["success"]:
            runs[(row["scenario"], row["ab_round"])][row["ab_side"]] = row
        else:
            failed[row["scenario"]].add(row["ab_side"])
    gpus = sorted({r["gpu_index"] for sides in runs.values() for r in sides.values()})
    out = {"gpus": gpus, "scenarios": [], "failed": {k: sorted(v) for k, v in failed.items()}, "regression": False}
    for scenario in sorted({s for s, _ in runs}):
        pairs = sorted(
            ((r, sides) for (s, r), sides in runs.items() if s == scenario and len(sides) == 2), key=lambda x: x[0]
        )
        entry = {"scenario": scenario, "rounds": len(pairs), "pairs": pairs, "metrics": [], "counters": {}}
        out["scenarios"].append(entry)
        if not pairs:
            continue
        for name, (col, higher_is_better) in METRICS.items():
            y, o = [], []
            for _, sides in pairs:
                b, h = sides["base"].get(col), sides["head"].get(col)
                if _num(b) and _num(h) and b > 0 and h > 0:  # ratios of positive values only
                    sign = -1 if higher_is_better else 1  # positive = head worse
                    y.append(sign * math.log(h / b))
                    if len(gpus) == 1:  # sides in turn on one GPU: head ran first in even rounds
                        o.append(1.0 if sides["head"]["ab_round"] % 2 == 0 else -1.0)
                    else:
                        o.append(1.0 if sides["head"]["gpu_index"] == gpus[-1] else -1.0)
            if not y:
                continue
            change, half, gpu = fit(y, o)
            lo, hi = pct(change - half), pct(change + half)
            if lo > threshold:
                verdict = "REGRESSION"
                out["regression"] = True
            elif hi < -threshold:
                verdict = "improvement"
            elif -threshold < lo and hi < threshold:
                verdict = "no change" + (" (significant, below threshold)" if lo > 0 or hi < 0 else "")
            else:
                verdict = "inconclusive"
            entry["metrics"].append(
                {
                    "name": name,
                    "column": col,
                    "base": statistics.median(
                        sides["base"][col] for _, sides in pairs if _num(sides["base"].get(col))
                    ),
                    "head": statistics.median(
                        sides["head"][col] for _, sides in pairs if _num(sides["head"].get(col))
                    ),
                    "change_pct": pct(change),
                    "ci": [lo, hi],
                    "gpu_pct": pct(gpu),
                    "verdict": verdict,
                }
            )
        for label, counter in COUNTERS.items():
            per_side = {}
            for side in ("base", "head"):
                phases = []
                for phase in ("first_generate", "warmup", "measurement"):
                    values = [sides[side].get(f"startup_{phase}_{counter}") for _, sides in pairs]
                    values = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
                    phases.append(f"{statistics.median(values):g}" if values else "-")
                per_side[side] = "/".join(phases)
            if per_side["base"] != "-/-/-":
                entry["counters"][label] = per_side
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ab_dir")
    p.add_argument("--threshold", type=float, default=1.0, help="change (%%) that matters")
    p.add_argument("--rounds", action="store_true", help="also print every round")
    a = p.parse_args(argv)

    rows = []
    for run_dir in sorted(Path(a.ab_dir).glob("round*-*")):
        if (run_dir / "manifest.json").exists():
            rows += scenario_rows(run_dir)
    if not any(r["success"] for r in rows):
        print("no successful runs found")
        return 1
    result = analyze(rows, a.threshold)
    any_row = next(r for r in rows if r["success"])
    what = f"PR #{any_row['pr_number']}" if any_row.get("pr_number") else any_row.get("ab_id")
    print(f"{what}  model {any_row['model_id']}  GPUs {result['gpus']}  threshold {a.threshold}%")
    print("change = head vs base, positive = head is worse; 95% CI over rounds; gpu = GPU difference fitted out\n")
    for entry in result["scenarios"]:
        print(f"{entry['scenario']}  ({entry['rounds']} paired rounds)")
        for m in entry["metrics"]:
            lo, hi = m["ci"]
            print(
                f"  {m['name']:<6} {m['base']:10.3f} -> {m['head']:<10.3f} change {m['change_pct']:+6.2f}%"
                f"  CI [{lo:+6.2f}, {hi:+6.2f}]  gpu {m['gpu_pct']:+5.2f}%  {m['verdict']}"
            )
        for label, per_side in entry["counters"].items():
            flag = "".join(
                f"  <-- {side} compiles during measurements"
                for side in ("base", "head")
                if per_side[side].split("/")[-1] not in ("0", "-")
            )
            print(f"  {label} (first gen/warmup/measurement): base {per_side['base']}  head {per_side['head']}{flag}")
        if a.rounds:
            for r, sides in entry["pairs"]:
                b, h = sides["base"], sides["head"]
                print(
                    f"    round {r}: base GPU{b['gpu_index']} e2e {b['e2e_ms_median']:.2f}  "
                    f"head GPU{h['gpu_index']} e2e {h['e2e_ms_median']:.2f}"
                )
        print()
    for scenario, sides in sorted(result["failed"].items()):
        print(f"FAILED on {', '.join(sides)}: {scenario}")
    return 1 if result["regression"] else 0


if __name__ == "__main__":
    sys.exit(main())
