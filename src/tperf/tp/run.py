"""tperf tp run: run harness with tensor parallelism over several GPU groups, several times, from your workstation.

Usage: tperf tp run [runs=3] [label=tp]

Environment:
  HOST          bench host, user@address (default: [host] ssh in tperf.toml)
  MODEL=meta-llama/Llama-2-7b-hf
  GPU_GROUPS="0 0,1 0,1,2,3"   one entry per configuration; a single GPU runs without torchrun (TP1 baseline)
  BENCH_ARGS   run_benchmarks.py arguments (default: -w 5 -i 10 -b 1 -s 128 -n 64, like the daily llama2-7b)
  SCENARIOS=scenarios/tp.jsonl
  MAIN_SHA     transformers commit (default: latest main), checked out as a worktree (its src/; the harness is harness/ here)
  NCORES=8     cores per rank, on the group's NUMA node (first core skipped, no hyperthread siblings)
  NCCL_ENV="NCCL_CUMEM_HOST_ENABLE=0"   see `tperf baselines p2p`: without it NCCL's SHM buffers all land on node 0

GPU groups must sit inside one NUMA node (0-3 or 4-7): across nodes the GPU->host writes NCCL relies on drop from
3.9 to 1.0 GB/s. Holds the daily lock. Results land in results/<timestamp>-<label>/<group>/run<i>/, and
`tperf generation compare` checks each group's run-to-run stability.
"""

import argparse
import os
import shlex
import subprocess
import sys


from tperf.common.benchlib import (
    GPU_NUMA_MAP,
    ROOT,
    HF_ENV,
    PY,
    Worktree,
    cmd,
    cpus_for,
    die,
    download_model,
    env_args,
    fetch_main,
    locked,
    log,
    push_harness,
    run_module,
    scp_from,
    scp_to,
    ssh,
    stamp,
    tune_host,
)


MODEL = os.environ.get("MODEL") or "meta-llama/Llama-2-7b-hf"
GPU_GROUPS = (os.environ.get("GPU_GROUPS") or "0 0,1 0,1,2,3").split()
BENCH_ARGS = shlex.split(os.environ.get("BENCH_ARGS") or "-w 5 -i 10 -b 1 -s 128 -n 64")
SCENARIOS = os.environ.get("SCENARIOS") or "scenarios/tp.jsonl"
MAIN_SHA = os.environ.get("MAIN_SHA") or None
NCORES = int(os.environ.get("NCORES") or 8)
NCCL_ENV = os.environ.get("NCCL_ENV", "NCCL_CUMEM_HOST_ENABLE=0")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="?", type=int, default=3)
    p.add_argument("label", nargs="?", default="tp")
    a = p.parse_args(argv)

    for group in GPU_GROUPS:
        if len({GPU_NUMA_MAP[int(g)] for g in group.split(",")}) != 1:
            die(f"GPU group {group} spans NUMA nodes; use 0-3 or 4-7")

    st = stamp()
    out = ROOT / "results" / f"{st}-{a.label}"
    rdir = f"/opt/bench/tp-results/{st}-{a.label}"
    out.mkdir(parents=True, exist_ok=True)
    log(f"Results: {out}")
    ssh(["mkdir", "-p", rdir, "/opt/bench/tp-work"])
    scp_to([ROOT / SCENARIOS], f"{rdir}/scenarios.jsonl")
    push_harness(rdir)

    # Worktree of main (its src/ only), removed at the end. Every run below holds the daily lock.
    sha = fetch_main(rev=MAIN_SHA)
    log(f"transformers {sha}")
    with Worktree(f"/opt/bench/tp-work/{st}", sha) as w:
        download_model(MODEL)
        env = {**HF_ENV, "HF_HUB_DISABLE_PROGRESS_BARS": 1, **dict(kv.split("=", 1) for kv in NCCL_ENV.split())}
        for i in range(1, a.runs + 1):
            tune_host("apply")
            for group in GPU_GROUPS:
                n = group.count(",") + 1
                node = GPU_NUMA_MAP[int(group.split(",")[0])]
                tag = f"tp{n}-gpu{group.replace(',', '-')}"
                d = f"{rdir}/{tag}/run{i}"
                cc = f"/mnt/cache/compile-cache/{st}-{tag}-run{i}"
                cpus = cpus_for(node, NCORES * n)
                launch = [PY, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", n] if n > 1 else [PY]
                log(f"Run {i}/{a.runs}: TP{n} on GPUs {group} (NUMA node {node})")
                bench = [
                    *env_args({**env, "PYTHONPATH": f"{w.path}/src", "TORCHINDUCTOR_CACHE_DIR": f"{cc}/inductor",
                               "TRITON_CACHE_DIR": f"{cc}/triton", "CUDA_VISIBLE_DEVICES": group,
                               "OMP_NUM_THREADS": NCORES}),
                    "numactl", f"--physcpubind={cpus}", f"--membind={node}",
                    *launch, "run_benchmarks.py", "--output-dir", d, "--model-id", MODEL, *BENCH_ARGS,
                    *(["--enable-tp"] if n > 1 else []), "--config-file", f"{rdir}/scenarios.jsonl",
                    "--no-gpu-monitoring", "--commit-id", sha, "--log-level", "WARNING",
                ]  # fmt: skip
                launch_txt = shlex.quote(f"GPUs {group}, CPUs {cpus}, {NCCL_ENV}")
                rc = ssh(
                    f"mkdir -p {d} && rm -rf {cc} && cd {rdir}/harness && echo {launch_txt} >{d}/launch.txt"
                    f" && {cmd(locked(bench))} >{d}/run.log 2>&1; rc=$?; rm -rf {cc}; exit $rc",
                    check=False,
                ).returncode
                if rc:
                    if rc == 75:
                        die("a benchmark holds the daily lock")
                    print(f"TP{n} run {i} failed, see {d}/run.log", file=sys.stderr)
                    ssh(["tail", "-30", f"{d}/run.log"], check=False, stdout=sys.stderr)
                    sys.exit(1)

    scp_from(f"{rdir}/tp*", out, recursive=True)
    for d in sorted(out.glob("tp*/")):
        log(d.name)
        runs = [f"{r}/" for r in sorted(d.glob("run*/"))]
        # tperf.generation.compare exits 1 when unstable: keep going
        res = run_module("generation.compare", *runs, text=True, stdout=subprocess.PIPE)
        print(res.stdout, end="")
        (d / "compare.txt").write_text(res.stdout)


if __name__ == "__main__":
    main()
