"""tperf baselines run: record GPU noise baselines on the bench host, from your workstation.

Usage: tperf baselines run [label]          (HOST=ubuntu@IP to override the host)
  NVBW_RUNS=5   NUMA_NODES="0 1"   GPU_NUMA_MAP="0 0 0 0 1 1 1 1"

Prerequisites on the host: `tperf host tune apply`, then `tperf host install`.
For every GPU, pinned to the NUMA node its PCIe device sits on:
  - cuBLAS GEMM throughput (bf16, the dtype harness uses), 30 timed repetitions after a 30 s warm-up
  - GPU throttle counters before/after, to catch power or thermal throttling
Then nvbandwidth (host<->device copy bandwidth and latency) from each NUMA node, NVBW_RUNS times.
Results land in results/<timestamp>-<label>/ and a summary is printed.
"""

import argparse
import os
import re
import subprocess
import sys


from tperf.common.benchlib import (
    GPU_NUMA_MAP, ROOT, env_args, log, push_package, remote_module, run_module, ssh, ssh_out, stamp, tune_host,
)  # fmt: skip


NVBW_RUNS = int(os.environ.get("NVBW_RUNS") or 5)
NUMA_NODES = (os.environ.get("NUMA_NODES") or "0 1").split()
COUNTERS = ",".join(
    f"clocks_event_reasons_counters.{c}"
    for c in ("sw_power_cap", "sw_thermal_slowdown", "hw_thermal_slowdown", "hw_power_brake_slowdown")
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("label", nargs="?", default="baseline")
    label = p.parse_args(argv).label

    st = stamp()
    out = ROOT / "results" / f"{st}-{label}"
    out.mkdir(parents=True, exist_ok=True)
    log(f"Results: {out}")
    src = f"/opt/bench/baselines-results/{st}-{label}/src"  # this run's copy of the package (tperf.baselines.gemm)
    push_package(src)

    log("Snapshotting host state")
    status = re.sub(r"\x1b\[[0-9;]*m", "", tune_host("status"))
    (out / "host-status.txt").write_text(status)
    with open(out / "nvidia-smi.txt", "w") as f:
        ssh("nvidia-smi -q; nvidia-smi topo -m", stdout=f)
    timers = status.partition("Active timers")[2].partition("Kernel:")[0]
    if re.search(r"^   [a-zA-Z].*\.timer$", timers, re.M):
        print("WARNING: timers are active on the host; run `tperf host tune apply` first", file=sys.stderr)

    ngpu = len(ssh_out(["nvidia-smi", "-L"]).splitlines())
    for g in range(ngpu):
        # NUMA node of the GPU's PCIe device (-1 on VMs that hide it: use GPU_NUMA_MAP).
        bus = ssh_out(["nvidia-smi", "-i", g, "--query-gpu=pci.bus_id", "--format=csv,noheader"]).lower()
        node = int(ssh_out(["cat", f"/sys/bus/pci/devices/{bus.removeprefix('0000')}/numa_node"]))
        if node < 0:
            node = GPU_NUMA_MAP[g] if g < len(GPU_NUMA_MAP) else 0
        log(f"GPU {g} (NUMA node {node}): GEMM")
        counters = ["nvidia-smi", "-i", g, f"--query-gpu=index,{COUNTERS}", "--format=csv,noheader,nounits"]
        with open(out / f"gpu{g}-counters-before.csv", "w") as f:
            ssh(counters, stdout=f)
        gemm = ssh(
            [*env_args({"CUDA_VISIBLE_DEVICES": g}), "numactl", f"--cpunodebind={node}", f"--membind={node}",
             *remote_module("/opt/bench/venv/bin/python", "baselines.gemm", src), "--dtype", "bf16"],
            capture=True,
        ).stdout  # fmt: skip
        with open(out / "gemm.jsonl", "a") as f:
            f.write(re.sub(r"^\{", f'{{"gpu": {g}, "numa_node": {node}, ', gemm, flags=re.M))
        with open(out / f"gpu{g}-counters-after.csv", "w") as f:
            ssh(counters, stdout=f)

    # Device-to-device tests are skipped: peer-to-peer is disabled on this VM (nvbandwidth waives them).
    for r in range(1, NVBW_RUNS + 1):
        for n in NUMA_NODES:
            log(f"nvbandwidth run {r}/{NVBW_RUNS}, host memory on NUMA node {n}")
            with open(out / f"nvbandwidth-node{n}-run{r}.txt", "w") as f:
                ssh(
                    ["numactl", f"--cpunodebind={n}", f"--membind={n}", "/opt/bench/nvbandwidth/nvbandwidth",
                     "-t", "host_to_device_memcpy_ce", "device_to_host_memcpy_ce", "host_device_latency_sm"],
                    stdout=f,
                )  # fmt: skip

    summary = run_module("baselines.summarize", out, text=True, stdout=subprocess.PIPE)
    print(summary.stdout, end="")
    (out / "summary.txt").write_text(summary.stdout)
    return summary.returncode


if __name__ == "__main__":
    sys.exit(main())
