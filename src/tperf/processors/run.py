"""tperf processors run: run tperf.processors.bench on the bench host several times, from your workstation, and report
how stable each scenario is between runs.

Usage: tperf processors run [runs=3] [label=processors]
  SCENARIOS=scenarios/processors.jsonl   THREADS=8 (torch threads; "1 8" for one pass each)   NODE=0 (NUMA node)
  GPU=0 (encoder stages; a GPU on NODE)
  PROC_ARGS="--warmup 4 --iterations 10" (default: 2 and 5)   MAIN_SHA=<commit, default latest main>

torchvision and Pillow come from /opt/bench/proc-deps (not in the tbv2-venv the GPU suites use).
Each run is a fresh process per thread count, pinned to that many cores of NODE (first core skipped, no hyperthread
siblings) with its memory on NODE, after the host tuning. Holds the daily lock. Results land in
results/<timestamp>-<label>/t<threads>/run<i>/.
"""

import argparse
import os
import shlex
import subprocess
import sys


from tperf.common.benchlib import (
    ROOT,
    HF_ENV,
    PY,
    Worktree,
    cmd,
    cpus_for,
    die,
    env_args,
    fetch_main,
    fetch_results,
    locked,
    log,
    push_package,
    remote_module,
    run_module,
    scp_to,
    ssh,
    stamp,
    tune_host,
)


SCENARIOS = os.environ.get("SCENARIOS") or "scenarios/processors.jsonl"
THREADS = [int(t) for t in (os.environ.get("THREADS") or "8").split()]
NODE = int(os.environ.get("NODE") or 0)
GPU = os.environ.get("GPU") or "0"  # for the encoder stages, on NODE
PROC_ARGS = shlex.split(os.environ.get("PROC_ARGS", ""))
MAIN_SHA = os.environ.get("MAIN_SHA") or None

# Download the processors, and the weights of the models with an encoder stage, once, not timed
DOWNLOAD = """import json, sys
from huggingface_hub import snapshot_download
from transformers import AutoProcessor
scenarios = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
for m in {s["model"] for s in scenarios}: AutoProcessor.from_pretrained(m)
for m in {s["model"] for s in scenarios if s.get("stage", "processor") != "processor"}:
    snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"])"""


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="?", type=int, default=3)
    p.add_argument("label", nargs="?", default="processors")
    a = p.parse_args(argv)

    st = stamp()
    out = ROOT / "results" / f"{st}-{a.label}"
    rdir = f"/opt/bench/proc-results/{st}-{a.label}"
    out.mkdir(parents=True, exist_ok=True)
    log(f"Results: {out}")
    ssh(["mkdir", "-p", rdir, "/opt/bench/proc-work"])
    src = f"{rdir}/src"  # this run's copy of the package (tperf.processors.bench)
    push_package(src)
    scp_to([ROOT / SCENARIOS], f"{rdir}/scenarios.jsonl")

    sha = fetch_main(rev=MAIN_SHA)
    log(f"transformers {sha}")
    with Worktree(f"/opt/bench/proc-work/{st}", sha) as w:
        pythonpath = f"{w.path}/src:/opt/bench/proc-deps"
        env = {**HF_ENV, "HF_HUB_DISABLE_PROGRESS_BARS": 1, "PYTHONPATH": pythonpath}
        ssh(f"cd /tmp && {cmd([*env_args(env), PY, '-c', DOWNLOAD, f'{rdir}/scenarios.jsonl'])} 2>/dev/null")

        for i in range(1, a.runs + 1):
            tune_host("apply")
            for t in THREADS:
                d = f"{rdir}/t{t}/run{i}"
                cpus = cpus_for(NODE, t)
                log(f"Run {i}/{a.runs}: {t} thread(s) on NUMA node {NODE}")
                bench = [
                    *env_args({**env, "OMP_NUM_THREADS": t, "CUDA_VISIBLE_DEVICES": GPU}),
                    "numactl", f"--physcpubind={cpus}", f"--membind={NODE}",
                    *remote_module(PY, "processors.bench", f"{src}:{pythonpath}"),
                    "--scenarios", f"{rdir}/scenarios.jsonl", "--output-dir", d,
                    "--threads", t, *PROC_ARGS,
                ]  # fmt: skip
                launch_txt = shlex.quote(f"commit {sha}, CPUs {cpus}")
                rc = ssh(
                    f"mkdir -p {d} && echo {launch_txt} >{d}/launch.txt"
                    f" && cd /tmp && {cmd(locked(bench))} >{d}/run.log 2>&1",
                    check=False,
                ).returncode
                if rc:
                    if rc == 75:
                        die("a benchmark holds the daily lock")
                    print(f"run {i} ({t} threads) failed", file=sys.stderr)
                    ssh(["tail", "-20", f"{d}/run.log"], check=False, stdout=sys.stderr)
                    sys.exit(1)

    fetch_results(rdir, out)
    summary = run_module("processors.summarize", out, check=True, text=True, stdout=subprocess.PIPE).stdout
    print(summary, end="")
    (out / "summary.txt").write_text(summary)


if __name__ == "__main__":
    main()
