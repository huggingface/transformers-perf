#!/usr/bin/env python3
"""Run tperf.baselines.nccl_bench on GPU groups, one process per GPU (runs on the bench host, with the system python:
standard library only).

Usage: PYTHONPATH=/opt/bench/src python3 -m tperf.baselines.nccl_launch <out-dir> <runs> <group> [group...]
       group = comma-separated GPU list, e.g. 0,1          (launched by `tperf baselines p2p`)
  CPUS_PER_RANK=2                  cores per rank
  BENCH_ARGS="--samples 30"        extra tperf.baselines.nccl_bench arguments
  GPU_NUMA_MAP="0 0 0 0 1 1 1 1"   GPU index -> NUMA node

Each rank is pinned to CPUS_PER_RANK cores of its GPU's NUMA node (skipping the node's first core, which takes
the device interrupts) with its memory bound to that node, like `tperf generation run` pins a single
GPU. Holds the daily lock so the daily benchmark and `tperf ab` don't start meanwhile. Writes, per group and run:
  g<group>-run<r>.jsonl   rank 0's measurements
  g<group>-run<r>.rank<i>.log  each rank's stderr, NCCL_DEBUG=INFO (the "via SHM/..." lines are the transports)
"""

import argparse
import fcntl
import glob
import os
import subprocess
import sys
from pathlib import Path


PY = "/opt/bench/venv/bin/python"
LOCK = "/tmp/bench-daily.lock"


def node_cpus(node: str) -> list[str]:
    """First hardware thread of every core of a node, the node's first core excluded."""
    out = subprocess.run(["lscpu", "-p=CPU,CORE,NODE"], stdout=subprocess.PIPE, text=True, check=True).stdout
    seen, cpus = set(), []
    for line in out.splitlines():
        if line.startswith("#"):
            continue
        cpu, core, n = line.split(",")[:3]
        if n == node and core not in seen:
            seen.add(core)
            cpus.append(cpu)
    return cpus[1:]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run tperf.baselines.nccl_bench on GPU groups, one process per GPU.")
    p.add_argument("out", type=Path)
    p.add_argument("runs", type=int)
    p.add_argument("groups", nargs="+", metavar="group")
    args = p.parse_args(argv)
    cpus_per_rank = int(os.environ.get("CPUS_PER_RANK") or 2)
    bench_args = os.environ.get("BENCH_ARGS", "").split()
    gpu_numa_map = (os.environ.get("GPU_NUMA_MAP") or "0 0 0 0 1 1 1 1").split()

    lock = open(LOCK, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"a benchmark holds {LOCK}, try later")
    args.out.mkdir(parents=True, exist_ok=True)
    pools = {n: node_cpus(n) for n in set(gpu_numa_map)}
    # Same package as this module: PYTHONPATH (inherited by the ranks) points at it
    bench = [PY, "-m", "tperf.baselines.nccl_bench"]

    port = 29511
    for r in range(1, args.runs + 1):
        for group in args.groups:
            gpus = group.split(",")
            tag = f"g{group.replace(',', '-')}-run{r}"
            used: dict[str, int] = {}
            procs = []
            port += 1
            for i, g in enumerate(gpus):
                node = gpu_numa_map[int(g)]
                k = used.get(node, 0)
                cpus = ",".join(pools[node][k : k + cpus_per_rank])
                used[node] = k + cpus_per_rank
                log = args.out / f"{tag}.rank{i}.log"
                log.write_text(f"rank {i}: GPU {g}, NUMA node {node}, CPUs {cpus}\n")
                env = {
                    **os.environ,
                    "RANK": str(i), "LOCAL_RANK": str(i), "WORLD_SIZE": str(len(gpus)),
                    "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port), "BENCH_GPU": g,
                    "NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,ENV", "NCCL_DEBUG_FILE": "/dev/stderr",
                }  # fmt: skip
                cmd = [
                    "numactl",
                    f"--physcpubind={cpus}",
                    f"--membind={node}",
                    *bench,
                    "--gpus",
                    group,
                    *bench_args,
                ]
                with open(args.out / f"{tag}.jsonl", "a") as out, open(log, "a") as err:
                    procs.append(subprocess.Popen(cmd, env=env, stdout=out, stderr=err))
            if any([proc.wait() for proc in procs]):  # a list: wait for every rank
                print(f"group {group} run {r} failed, see {args.out}/{tag}.rank*.log", file=sys.stderr)
                subprocess.run(["tail", "-20", *sorted(glob.glob(f"{args.out}/{tag}.rank*.log"))], stdout=sys.stderr)
                sys.exit(1)
            print(f"group {group} run {r}/{args.runs} done", file=sys.stderr)


if __name__ == "__main__":
    main()
