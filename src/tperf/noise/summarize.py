"""Compare `tperf noise soak` results side by side: one column per soak (host).

Usage: tperf noise summarize results/noise-<stamp>-<label> [results/noise-<stamp>-<label> ...]

For each soak:
  - benchmark: per scenario and metric, the median over every run of every batch, the CoV across runs, the
    spread (max - min, % of the median) and the drift (median of the second half of the runs against the first)
  - noise probe (per run): steal, foreign CPU time, interrupts on the benchmark CPUs, run-queue wait, GPU throttle
  - contention probe (between batches): core clock loop, memory copy bandwidth, L3 and DRAM gather latency, as
    median, CoV and spread across the probes
  - minute samples: steal share of the benchmark CPUs over the whole window, and the worst minute

A shared host shows up as wider CoV, spread or drift than a dedicated one at the same medians; steal points at vCPU
sharing, contention-probe spread with no steal at a neighbour taking cache or memory bandwidth.
"""

import argparse
import itertools
import json
import statistics
from pathlib import Path

from tperf.generation.compare import load, short

BENCH_METRICS = [("e2e_ms", "ms"), ("ttft_ms", "ms"), ("itl_ms", "ms"), ("tok_s", "tok/s")]
PROBE_METRICS = [("cpu_mops", "Mops/s"), ("copy_gbs", "GB/s"), ("l3_ns", "ns"), ("dram_ns", "ns")]


def describe(vals: list[float]) -> dict[str, float] | None:
    if not vals:
        return None
    med = statistics.median(vals)
    half = len(vals) // 2
    first = statistics.median(vals[:half]) if half else 0.0
    drift = 100 * (statistics.median(vals[half:]) / first - 1) if first else 0.0
    return {
        "n": len(vals),
        "median": med,
        "cov_pct": 100 * statistics.stdev(vals) / statistics.fmean(vals) if len(vals) > 1 and any(vals) else 0.0,
        "spread_pct": 100 * (max(vals) - min(vals)) / med if med else 0.0,
        "drift_pct": drift,
    }


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def run_dirs(soak: Path) -> list[Path]:
    """Every run directory of every batch, in time order."""
    meta = json.loads((soak / "soak.json").read_text())
    dirs = []
    for b in meta["batches"]:
        if b.get("dir"):
            dirs += sorted((p for p in Path(b["dir"]).glob("run*") if p.is_dir()), key=lambda p: int(p.name[3:]))
    return dirs


def summarize(soak: Path) -> dict:
    meta = json.loads((soak / "soak.json").read_text())
    runs = run_dirs(soak)
    bench: dict[str, dict[str, list[float]]] = {}
    probe: dict[str, list[float]] = {k: [] for k in ("steal_s", "foreign_s", "irqs", "rq_wait_ms", "throttle_ms")}
    for d in runs:
        for scen, m in load(d).items():
            for metric, _ in BENCH_METRICS:
                bench.setdefault(short(scen), {}).setdefault(metric, []).append(m[metric])
        p = d / "probe.json"
        if p.exists():
            r = json.loads(p.read_text())
            c = r["cpu"]
            probe["steal_s"].append(c.get("steal_s_on_cpus", 0.0))
            probe["foreign_s"].append(c["foreign_cpu_s"])
            probe["irqs"].append(c["interrupts_on_cpus"])
            probe["rq_wait_ms"].append(c["runqueue_wait_ms"])
            probe["throttle_ms"].append(sum(r.get("gpu", {}).get("throttle_ms", {}).values()))
    hostprobe = {k: [p["metrics"][k]["median"] for p in jsonl(soak / "hostprobe.jsonl")] for k, _ in PROBE_METRICS}
    samples = jsonl(soak / "samples.jsonl")
    window = {}
    if len(samples) > 1:
        total = samples[-1]["total"] - samples[0]["total"]
        minutes = [
            100 * (b["steal"] - a["steal"]) / (b["total"] - a["total"])
            for a, b in itertools.pairwise(samples)
            if b["total"] > a["total"]
        ]
        window = {
            "minutes": len(samples) - 1,
            "steal_pct": 100 * (samples[-1]["steal"] - samples[0]["steal"]) / total if total else 0.0,
            "worst_minute_steal_pct": max(minutes, default=0.0),
        }
    return {
        "label": f"{meta['label']} ({meta['host'].partition('@')[2] or meta['host']})",
        "runs": len(runs),
        "batches": len(meta["batches"]),
        "bench": {s: {m: describe(v) for m, v in ms.items()} for s, ms in bench.items()},
        "probe": {k: describe(v) for k, v in probe.items()},
        "hostprobe": {k: describe(v) for k, v in hostprobe.items()},
        "window": window,
    }


def fmt(d: dict | None, key: str, spec: str = ".2f") -> str:
    return "-" if not d or d.get(key) is None else format(d[key], spec)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("soaks", nargs="+", type=Path)
    a = p.parse_args(argv)
    sums = [summarize(s) for s in a.soaks]
    w = max(24, *(len(s["label"]) + 2 for s in sums))
    cols = "median  CoV%  spread%  drift%"

    def header(title: str) -> None:
        print(f"\n{title}")
        print(f"   {'':<28}" + "".join(f"{s['label']:<{w + 10}}" for s in sums))
        print(f"   {'':<28}" + "".join(f"{cols:<{w + 10}}" for _ in sums))

    def row(name: str, ds: list[dict | None], spec: str) -> None:
        cells = [
            f"{fmt(d, 'median', spec):>8} {fmt(d, 'cov_pct'):>5} {fmt(d, 'spread_pct'):>8} {fmt(d, 'drift_pct'):>7}"
            for d in ds
        ]
        print(f"   {name:<28}" + "".join(f"{c:<{w + 10}}" for c in cells))

    print("Soaks: " + ", ".join(f"{s['label']}: {s['batches']} batches, {s['runs']} runs" for s in sums))
    for scen in sorted(set().union(*(s["bench"] for s in sums))):
        header(f"Benchmark: {scen}")
        for metric, unit in BENCH_METRICS:
            row(f"{metric} ({unit})", [s["bench"].get(scen, {}).get(metric) for s in sums], ".2f")
    header("Noise probe, per run")
    for k in ("steal_s", "foreign_s", "irqs", "rq_wait_ms", "throttle_ms"):
        row(k, [s["probe"].get(k) for s in sums], ".2f")
    header("Contention probe, between batches")
    for k, unit in PROBE_METRICS:
        row(f"{k} ({unit})", [s["hostprobe"].get(k) for s in sums], ".2f")
    print("\nWhole window, benchmark CPUs (one sample a minute)")
    for key in ("minutes", "steal_pct", "worst_minute_steal_pct"):
        print(f"   {key:<28}" + "".join(f"{fmt(s['window'], key, '.3f'):<{w + 10}}" for s in sums))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
