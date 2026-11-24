"""Measure how noisy a bench host is over hours: is a shared VM good enough to benchmark on?

Usage: MODEL=<hf model id> tperf noise soak [--hours 3] [--runs 5] [--label noise]
       tperf noise probe                       # one contention probe on the host, printed

Until --hours have passed, repeats: one contention probe (tperf.noise.host_probe: core clock, memory bandwidth, L3
and DRAM latency, on a benchmark CPU), then a `tperf generation run` batch of --runs runs (host tuning, noise probe,
cold compile caches: see its docstring). A thread snapshots the benchmark CPUs' busy, steal and interrupt counters
once a minute for the whole window, between and during runs. Nothing is published. Run it on two hosts at the same
time (HOST=... for each) and compare with `tperf noise summarize`.

Results: results/noise-<stamp>-<label>/ with soak.json (settings, batch result directories), hostprobe.jsonl,
samples.jsonl and soak.log (the batches' output).

Environment: HOST, MODEL (required), GPU=0, NCORES=8, BENCH_ARGS, GPU_NUMA_MAP: as `tperf generation run`, which
gets them as they are. Use the same NCORES and BENCH_ARGS on every host compared.
"""

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

from tperf.common.benchlib import (
    GPU_NUMA_MAP, HOST, PY, ROOT, cpus_for, die, log, push_package, remote_module, require_host, run_module, ssh,
    stamp,
)  # fmt: skip

MAX_FAILED_BATCHES = 3


def bench_cpus() -> tuple[int, str]:
    """(NUMA node, CPUs) the batches pin to, as `tperf generation run` picks them."""
    gpu = int(os.environ.get("GPU", "0"))
    node = GPU_NUMA_MAP[gpu]
    return node, cpus_for(node, int(os.environ.get("NCORES", "8")))


def host_probe(src: str, node: int, cpus: str) -> dict:
    """One contention probe on the first benchmark CPU (memory from the GPU's node)."""
    first = cpus.split(",")[0]
    out = ssh(
        ["numactl", f"--physcpubind={first}", f"--membind={node}",
         *remote_module(PY, "noise.host_probe", src), "--cpus", cpus],
        capture=True,
    ).stdout  # fmt: skip
    return json.loads(out)


def sampler(src: str, cpus: str, path: Path, stop: threading.Event) -> None:
    """A counter snapshot of the benchmark CPUs every minute until `stop` (the host's system python, stdlib only)."""
    with path.open("a") as f:
        while not stop.is_set():
            p = ssh([*remote_module("python3", "noise.host_probe", src), "--sample", "--cpus", cpus],
                    check=False, capture=True)  # fmt: skip
            if p.returncode == 0:
                f.write(p.stdout)
                f.flush()
            stop.wait(60)


def probe(argv: list[str] | None = None) -> int:
    """tperf noise probe: one contention probe on the host's benchmark CPUs, printed."""
    argparse.ArgumentParser(description="One contention probe (tperf.noise.host_probe) on the bench host.").parse_args(
        argv
    )
    require_host()
    node, cpus = bench_cpus()
    src = f"/opt/bench/noise/{stamp()}-probe/src"
    push_package(src)
    print(json.dumps(host_probe(src, node, cpus), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hours", type=float, default=3.0)
    p.add_argument("--runs", type=int, default=5, help="runs per generation batch")
    p.add_argument("--label", default="noise")
    a = p.parse_args(argv)
    require_host()
    if not os.environ.get("MODEL"):
        die("set MODEL to a Hugging Face model id")
    if os.environ.get("PUBLISH_REPO"):
        die("unset PUBLISH_REPO: soak runs are not published")

    st = stamp()
    out = ROOT / "results" / f"noise-{st}-{a.label}"
    out.mkdir(parents=True)
    node, cpus = bench_cpus()
    src = f"/opt/bench/noise/{st}-{a.label}/src"
    push_package(src)
    meta = {
        "host": HOST, "label": a.label, "start": st, "hours": a.hours, "runs_per_batch": a.runs, "cpus": cpus,
        "node": node, "model": os.environ["MODEL"], "gpu": os.environ.get("GPU", "0"),
        "ncores": os.environ.get("NCORES", "8"), "bench_args": os.environ.get("BENCH_ARGS", ""), "batches": [],
    }  # fmt: skip
    (out / "soak.json").write_text(json.dumps(meta, indent=2))
    log(f"{HOST}: {a.hours} h of {a.runs}-run batches on CPUs {cpus}; results in {out}")

    stop = threading.Event()
    thread = threading.Thread(target=sampler, args=(src, cpus, out / "samples.jsonl", stop), daemon=True)
    thread.start()
    deadline, failed, batch = time.time() + a.hours * 3600, 0, 0
    try:
        while time.time() < deadline and failed < MAX_FAILED_BATCHES:
            batch += 1
            try:
                with (out / "hostprobe.jsonl").open("a") as f:
                    f.write(json.dumps({"batch": batch, **host_probe(src, node, cpus)}) + "\n")
            except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
                log(f"batch {batch}: host probe failed ({e})")
            group = f"{a.label}-b{batch}"
            log(f"batch {batch}: generation run {a.runs} {group}")
            with (out / "soak.log").open("a") as f:
                rc = run_module("generation.run", a.runs, group, stdout=f, stderr=subprocess.STDOUT).returncode
            dirs = sorted((ROOT / "results").glob(f"*-{group}"))
            meta["batches"].append({"batch": batch, "exit": rc, "dir": str(dirs[-1]) if dirs else None})
            (out / "soak.json").write_text(json.dumps(meta, indent=2))
            failed = failed + 1 if rc else 0
            if rc:
                log(f"batch {batch} failed (exit {rc}), see {out / 'soak.log'}")
                time.sleep(60)
    finally:
        stop.set()
        thread.join(timeout=120)
    if failed >= MAX_FAILED_BATCHES:
        die(f"{failed} batches failed in a row, stopping: see {out / 'soak.log'}")
    log(f"done: {batch} batches. tperf noise summarize {out} <other soak>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
