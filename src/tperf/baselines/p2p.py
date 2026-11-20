"""tperf baselines p2p: measure GPU<->GPU communication on the bench host, from your workstation.

Usage: tperf baselines p2p [label]          (HOST=ubuntu@IP to override the host)
  RUNS=3                          repetitions of every group (run-to-run spread)
  GPU_GROUPS="0,1 0,4 ..."        GPU groups; default: a pair on one NUMA node, a pair across nodes, TP4 on each
                                  node, TP8 across both
  BENCH_ARGS="--samples 30"       extra tperf.baselines.nccl_bench arguments
  NCCL_ENV="NCCL_CUMEM_HOST_ENABLE=0"  NCCL settings for every rank (recorded in topology.txt). Recommended: with
                                  the default cuMem host buffers NCCL puts every SHM buffer on NUMA node 0 (the VM
                                  hides the GPUs' node), so GPUs 4-7 write to remote memory and get 2.6x less
                                  bandwidth; disabled, buffers follow each rank's --membind.

Peer-to-peer is not supported between any GPUs of this VM (nvidia-smi topo: PHB, nvbandwidth waives its
device-to-device tests), so NCCL stages GPU<->GPU traffic through host memory. This measures what tensor
parallelism actually gets: all_reduce latency and bandwidth from 8 B to 256 MiB with torch's NCCL, one process
per GPU (tperf.baselines.nccl_launch), and which transport NCCL picked. Results land in results/<timestamp>-<label>/ and a
summary is printed.
"""

import argparse
import os
import re
import subprocess
import sys


from tperf.common.benchlib import (
    ROOT, env_args, fetch_results, log, push_package, remote_module, run_module, ssh, stamp, tune_host,
)  # fmt: skip


RUNS = int(os.environ.get("RUNS") or 3)
GPU_GROUPS = (os.environ.get("GPU_GROUPS") or "0,1 0,4 0,1,2,3 4,5,6,7 0,1,2,3,4,5,6,7").split()
NCCL_ENV = os.environ.get("NCCL_ENV", "NCCL_CUMEM_HOST_ENABLE=0")  # set but empty: no NCCL settings
BENCH_ARGS = os.environ.get("BENCH_ARGS", "")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("label", nargs="?", default="p2p")
    label = p.parse_args(argv).label

    st = stamp()
    out = ROOT / "results" / f"{st}-{label}"
    rdir = f"/opt/bench/p2p-results/{st}-{label}"
    out.mkdir(parents=True, exist_ok=True)
    log(f"Results: {out}")
    src = f"{rdir}/src"  # this run's copy of the package (tperf.baselines.nccl_launch and nccl_bench)
    push_package(src)
    tune_host("apply")
    (out / "host-status.txt").write_text(re.sub(r"\x1b\[[0-9;]*m", "", tune_host("status")))
    torch_version = 'import torch; print("torch", torch.__version__, "nccl", torch.cuda.nccl.version())'
    with open(out / "topology.txt", "w") as f:
        ssh(f"nvidia-smi topo -m; nvidia-smi topo -p2p r; /opt/bench/venv/bin/python -c '{torch_version}'", stdout=f)
    with open(out / "topology.txt", "a") as f:
        f.write(f"NCCL_ENV: {NCCL_ENV}\n")

    log(f"Groups: {' '.join(GPU_GROUPS)}, {RUNS} runs each")
    # tperf.baselines.nccl_launch (system python, standard library only) takes the daily lock itself.
    ssh([*env_args({"BENCH_ARGS": BENCH_ARGS}), *NCCL_ENV.split(), *remote_module("python3", "baselines.nccl_launch", src),
         rdir, RUNS, *GPU_GROUPS])  # fmt: skip
    fetch_results(rdir, out)

    summary = run_module("baselines.summarize_p2p", out, text=True, stdout=subprocess.PIPE)
    print(summary.stdout, end="")
    (out / "summary.txt").write_text(summary.stdout)
    return summary.returncode


if __name__ == "__main__":
    sys.exit(main())
