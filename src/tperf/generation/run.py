"""Run transformers harness several times on one GPU and check run-to-run stability.

Usage: MODEL=<hf model id> tperf generation run [runs=3] [label=tbv2]

Environment:
  HOST          bench host, user@address (default: [host] ssh in tperf.toml)
  MODEL       model id (required)
  GPU=0       physical GPU index; runs are always one at a time
  NCORES=8    physical cores to pin to, on the GPU's NUMA node. The node's first core is
              skipped (housekeeping, IRQs) and hyperthread siblings are never used.
  BENCH_ARGS  extra tperf.harness.run_benchmarks arguments (default: -w 5 -i 30 -b 1 -s 128 -n 128 --level 1)
  GPU_NUMA_MAP  GPU index -> NUMA node (default "0 0 0 0 1 1 1 1", measured on g5.48xlarge)

Per run: tperf/host/prepare.py apply (drop caches, re-lock clocks), then harness under
tperf.common.noise_probe (CPU/GPU usage and interference, read from counters before/after: no sampling).
harness's own GPU monitor is disabled: it polls NVML every 50 ms on the CPU and always
watches NVML device 0, whatever CUDA_VISIBLE_DEVICES says.
Gated models: log in once on the host with `/opt/bench/tbv2-venv/bin/hf auth login`.

Recording: before each run, tperf.common.record writes manifest.json (versions, hardware, clocks, git
identity) next to the results. With PUBLISH_REPO=<user>/<dataset>, the runs are then published to
that private HF dataset (one row per run and scenario, plus the raw files). Set BENCH_SOURCE
(main|pr|manual), BENCH_PR and BENCH_BRANCH to tag them.
"""

import argparse
import os
import re
import shlex
import subprocess
import sys


from tperf.common.benchlib import (
    GPU_NUMA_MAP, ROOT, HF_ENV, HOST, PY, SSH, TF, cmd, cpus_for, die, download_model, env_args, log, push_package,
    remote_module, run_module, scp_from, ssh, ssh_out, stamp, tune_host,
)  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="?", type=int, default=3)
    parser.add_argument("label", nargs="?", default="tbv2")
    args = parser.parse_args(argv)
    runs, label = args.runs, args.label

    model = os.environ.get("MODEL") or die("set MODEL to a Hugging Face model id")
    gpu = int(os.environ.get("GPU", "0"))
    ncores = int(os.environ.get("NCORES", "8"))
    bench_args = os.environ.get("BENCH_ARGS", "-w 5 -i 30 -b 1 -s 128 -n 128 --level 1")
    publish_repo = os.environ.get("PUBLISH_REPO", "")
    tags = {k: os.environ.get(k, d) for k, d in (("BENCH_SOURCE", "manual"), ("BENCH_PR", ""), ("BENCH_BRANCH", ""))}
    group = f"{stamp()}-{label}"
    out = ROOT / "results" / group
    rdir = f"/opt/bench/tbv2-results/{group}"
    src = f"{rdir}/src"  # this run's copy of the package (harness, record, noise probe)

    node = GPU_NUMA_MAP[gpu]
    cpus = cpus_for(node, ncores)
    commit = ssh_out(["git", "-C", TF, "rev-parse", "HEAD"])
    out.mkdir(parents=True, exist_ok=True)
    log(f"Model {model}, GPU {gpu}, NUMA node {node}, CPUs {cpus}, transformers {commit}")
    log(f"Results: {out}")

    version = push_package(src)
    log(f"harness {version}")
    log("Pre-downloading the model (not timed)")
    download_model(model)

    for i in range(1, runs + 1):
        log(f"Run {i}/{runs}")
        tune_host("apply")
        (out / f"host-status-run{i}.txt").write_text(re.sub(r"\x1b\[[0-9;]*m", "", tune_host("status")))
        ssh([
            *env_args({**tags, "BENCH_HARNESS": version}), *remote_module(PY, "common.record", src), "manifest",
            "--out", f"{rdir}/run{i}/manifest.json", "--gpu", gpu, "--cpus", cpus, "--numa-node", node,
            "--model", model, "--bench-args", bench_args, "--group", group, "--run", i, "--label", label,
        ])  # fmt: skip
        # Fresh torch.compile / Triton caches per run: every run measures a true cold start
        cc = f"/mnt/cache/compile-cache/{group}-run{i}"
        env = {**HF_ENV, "TORCHINDUCTOR_CACHE_DIR": f"{cc}/inductor", "TRITON_CACHE_DIR": f"{cc}/triton",
               "CUDA_VISIBLE_DEVICES": gpu, "PYTHONPATH": src}  # fmt: skip
        run = [
            *env_args(env), "numactl", f"--physcpubind={cpus}", f"--membind={node}",
            *remote_module(PY, "common.noise_probe", src), "--cpus", cpus, "--gpu", gpu,
            "--out", f"{rdir}/run{i}/probe.json", "--",
            PY, "-m", "tperf.harness.run_benchmarks", "--output-dir", f"{rdir}/run{i}", "--model-id", model,
            *shlex.split(bench_args),
            "--no-gpu-monitoring", "--commit-id", commit, "--log-level", "WARNING",
        ]  # fmt: skip
        remote = (
            f"rm -rf {cc} && mkdir -p {rdir}/run{i} && cd {rdir}/run{i} && {cmd(run)}; rc=$?; rm -rf {cc}; exit $rc"
        )
        run_log = out / f"run{i}.log"
        with open(run_log, "w") as f:
            failed = subprocess.run([*SSH, HOST, remote], stdout=f, stderr=subprocess.STDOUT).returncode
        if failed:
            print(f"run {i} failed, see {run_log}", file=sys.stderr)
            print("\n".join(run_log.read_text().splitlines()[-20:]), file=sys.stderr)
            sys.exit(1)
        scp_from(f"{rdir}/run{i}", out, recursive=True)

    compare = run_module(
        "generation.compare",
        *sorted(str(p) for p in out.glob("run*") if p.is_dir()),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    print(compare.stdout, end="")
    (out / "compare.txt").write_text(compare.stdout)

    if publish_repo:
        log(f"Publishing to {publish_repo}")
        hub_env = env_args({**HF_ENV, "HF_HUB_DISABLE_PROGRESS_BARS": 1})
        published = ssh(
            f"{cmd([*hub_env, *remote_module(PY, 'common.record', src), 'publish'])} {rdir}/run* "
            f"{cmd(['--repo', publish_repo])}",
            capture=True,
        ).stdout  # fmt: skip
        print(published, end="")
        (out / "publish.txt").write_text(published)


if __name__ == "__main__":
    main()
