"""Host side of `tperf noise`: measure what other tenants of a shared machine can take from a benchmark.

Runs on the bench host, pinned (by the caller) to one benchmark CPU and its NUMA node's memory:

    python -m tperf.noise.host_probe --cpus 1-6 [--reps 21]     # contention probe, one JSON object
    python3 -m tperf.noise.host_probe --sample --cpus 1-6      # counter snapshot, one JSON object (stdlib only)

The probe times, `reps` times each, what a neighbour on the same physical host would slow down while CPU steal stays
at 0 (AWS does not overcommit the vCPUs of non-burstable instances, but shares the L3 cache, memory bandwidth and
turbo budget):
  cpu_mops    pure Python integer loop, million iterations/s: core clock (turbo budget) and SMT sibling contention
  copy_gbs    single-thread copy of a 256 MiB array, GB/s (read + write): memory bandwidth
  l3_ns       random gather inside a 4 MiB array, ns/element: L3 contention
  dram_ns     random gather inside a 1 GiB array, ns/element: memory latency under contention
plus the steal and busy jiffies of `--cpus` over the probe. --sample prints the cumulative counters of `--cpus` (busy,
steal, total jiffies, interrupts) and the load average; `tperf noise soak` takes one a minute and diffs them.
"""

import argparse
import json
import os
import statistics
import time


def parse_cpus(spec: str) -> list[int]:
    cpus = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        cpus.extend(range(int(lo), int(hi or lo) + 1))
    return cpus


def cpu_counters(cpus: list[int]) -> dict[str, int]:
    """Cumulative jiffies of `cpus` from /proc/stat: busy (all but idle and iowait, steal included), steal, total."""
    busy = steal = total = 0
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu") and line[3].isdigit():
                name, *vals = line.split()
                if int(name[3:]) in cpus:
                    v = list(map(int, vals))
                    total += sum(v[:8])
                    busy += sum(v[:8]) - v[3] - v[4]
                    steal += v[7] if len(v) > 7 else 0
    return {"busy": busy, "steal": steal, "total": total}


def interrupt_count(cpus: list[int]) -> int:
    """Interrupts delivered to `cpus` since boot (/proc/interrupts)."""
    with open("/proc/interrupts") as f:
        header = f.readline().split()
        cols = [i for i, name in enumerate(header) if name.startswith("CPU") and int(name[3:]) in cpus]
        total = 0
        for line in f:
            fields = line.split(":", 1)[-1].split()
            total += sum(int(fields[i]) for i in cols if i < len(fields) and fields[i].isdigit())
    return total


def sample(cpus: list[int]) -> dict:
    with open("/proc/loadavg") as f:
        load = [float(x) for x in f.read().split()[:3]]
    return {"t": time.time(), **cpu_counters(cpus), "irqs": interrupt_count(cpus), "load": load}


def stats(vals: list[float]) -> dict[str, float]:
    med, mean = statistics.median(vals), statistics.fmean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return {"median": med, "min": min(vals), "max": max(vals), "cov_pct": 100 * sd / mean if mean else 0.0}  # sd/mean


def probe(cpus: list[int], reps: int) -> dict:
    import numpy as np

    rng = np.random.default_rng(0)
    src = np.ones(32 * 2**20)  # 256 MiB of float64
    dst = np.empty_like(src)
    small = np.ones(2**19)  # 4 MiB: fits in one L3 slice of every EPYC/Xeon we use
    big = np.ones(2**27)  # 1 GiB
    idx_small = rng.integers(0, small.size, 2**22)
    idx_big = rng.integers(0, big.size, 2**22)
    np.copyto(dst, src)  # first touch outside the timed loop

    def timed(fn) -> float:
        t0 = time.perf_counter()
        fn()
        return time.perf_counter() - t0

    def loop() -> None:
        x = 0
        for i in range(2_000_000):
            x += i

    results: dict[str, list[float]] = {"cpu_mops": [], "copy_gbs": [], "l3_ns": [], "dram_ns": []}
    c0, t0 = cpu_counters(cpus), time.time()
    for _ in range(reps):
        results["cpu_mops"].append(2.0 / timed(loop))
        results["copy_gbs"].append(2 * src.nbytes / 1e9 / timed(lambda: np.copyto(dst, src)))
        results["l3_ns"].append(1e9 * timed(lambda: small[idx_small].sum()) / idx_small.size)
        results["dram_ns"].append(1e9 * timed(lambda: big[idx_big].sum()) / idx_big.size)
    c1 = cpu_counters(cpus)
    hz = os.sysconf("SC_CLK_TCK")
    return {
        "t": t0,
        "elapsed_s": time.time() - t0,
        "reps": reps,
        "pinned_to": sorted(os.sched_getaffinity(0)),
        "steal_s": (c1["steal"] - c0["steal"]) / hz,
        "busy_s": (c1["busy"] - c0["busy"]) / hz,
        "metrics": {k: stats(v) for k, v in results.items()},
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cpus", required=True, help="benchmark CPUs, e.g. 1-6: counters are read for these")
    p.add_argument("--sample", action="store_true", help="print a counter snapshot (stdlib only) instead of probing")
    p.add_argument("--reps", type=int, default=21)
    a = p.parse_args(argv)
    cpus = parse_cpus(a.cpus)
    print(json.dumps(sample(cpus) if a.sample else probe(cpus, a.reps)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
