"""tperf.daily.main: benchmark the latest transformers main and publish it (runs on the bench host).

`tperf deploy` copies the tperf package to /opt/bench/src/tperf; bench-daily.timer runs this module as ubuntu with
/usr/bin/python3 -m tperf.daily.main and PYTHONPATH=/opt/bench/src (standard library only: the system python has no
third-party packages). The other host-side modules (tperf.common.record, tperf.common.noise_probe,
tperf.processors.bench, tperf.dashboard.build) and the harness (tperf.harness.run_benchmarks) run in the tbv2-venv with
the same PYTHONPATH: the harness is the deployed package's, the worktree of main only provides transformers' src/.
Results, work trees and scenarios are under /opt/bench/daily.
Manual run: PYTHONPATH=/opt/bench/src /usr/bin/python3 -m tperf.daily.main [suite...]   (default: every suite below)

Each suite runs RUNS times on its own GPU; suites on GPUs of different NUMA nodes run in parallel
(they share no cores, memory or PCIe). Per run: host tuning re-applied (once for all suites),
manifest, noise probe, fresh compile caches. Rows are published with source=main, label daily-<suite>.
Then the "processors" suite: tperf.processors.bench on scenarios/processors.jsonl, RUNS times, pinned to NCORES cores of
node 0 and GPU 0 (free once the GPU suites are done), published to processors/ with label daily-processors.
Skipped (exit 0) if a benchmark is already running on the host.

To try other settings the same way (e.g. a shorter run), without touching the daily series:
  MAIN_SHA=<commit> ARGS="-w 5 -i 10 -b 1 -s 128 -n 64" LABEL_PREFIX=check DASHBOARD_SPACE= \\
    PYTHONPATH=/opt/bench/src /usr/bin/python3 -m tperf.daily.main llama2-7b
"""

import argparse
import fcntl
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


D = Path("/opt/bench/daily")
SRC = "/opt/bench/src"  # the tperf package, deployed by `tperf deploy`
TF = "/opt/bench/transformers"
PY = "/opt/bench/tbv2-venv/bin/python"
HARNESS_VERSION = Path(SRC) / "tperf/harness/VERSION"  # written by `tperf deploy`
REPO_URL = "https://github.com/huggingface/transformers.git"
LOCK = "/tmp/bench-daily.lock"
# Set by tperf deploy from tperf.toml, in /opt/bench/tperf.env (EnvironmentFile of the service)
PUBLISH_REPO = os.environ.get("PUBLISH_REPO") or "hf-internal-testing/transformers-perf"
DASHBOARD_SPACE = os.environ.get("DASHBOARD_SPACE", "hf-internal-testing/transformers-perf-dashboard")
RUNS = int(os.environ.get("RUNS") or 3)
MAIN_SHA = os.environ.get("MAIN_SHA", "")  # commit to benchmark instead of the latest main
ARGS = os.environ.get("ARGS", "")  # tperf.harness.run_benchmarks arguments replacing the suites' own
LABEL_PREFIX = os.environ.get("LABEL_PREFIX") or "daily"  # rows are <prefix>-<suite>; the dashboard's are daily-*
NCORES = int(os.environ.get("NCORES") or 8)
GPU_NUMA_MAP = [int(n) for n in (os.environ.get("GPU_NUMA_MAP") or "0 0 0 0 1 1 1 1").split()]
PROC_PYTHONPATH_EXTRA = "/opt/bench/proc-deps"  # torchvision and Pillow, not in the tbv2-venv

# suite name -> (gpu, model, bench args, scenarios file or None for --level 1)
SUITES = {
    "llama2-7b": (0, "meta-llama/Llama-2-7b-hf", "-w 5 -i 10 -b 1 -s 128 -n 64", None),
    "gemma3-1b-sliding": (
        4,
        "google/gemma-3-1b-it",
        "-w 5 -i 6 -b 1 -s 448 -n 128",
        D / "scenarios/sliding-window.jsonl",
    ),
}

os.environ.update(
    PYTHONPATH=SRC,  # every child below runs tperf modules with `-m`
    HF_HOME="/mnt/cache/hf",
    HF_TOKEN_PATH=f"{Path.home()}/.cache/huggingface/token",
    HF_HUB_DISABLE_PROGRESS_BARS="1",
)


def log(msg: str) -> None:
    print(f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} {msg}", flush=True)


def git(*args: str, **kw) -> str:
    return subprocess.run(["git", "-C", TF, *args], check=True, text=True, capture_output=True, **kw).stdout.strip()


def tune_host() -> None:
    subprocess.run(
        ["sudo", "/usr/bin/python3", f"{SRC}/tperf/host/prepare.py", "apply"],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def cpus_for(node: int) -> str:
    """First hardware thread of NCORES cores on a NUMA node, skipping its first core (housekeeping)."""
    out = subprocess.run(["lscpu", "-p=CPU,CORE,NODE"], check=True, text=True, capture_output=True).stdout
    seen, cpus = set(), []
    for line in out.splitlines():
        if line.startswith("#"):
            continue
        cpu, core, n = line.split(",")[:3]
        if int(n) == node and core not in seen:
            seen.add(core)
            cpus.append(cpu)
    return ",".join(cpus[1 : NCORES + 1])


def argv(*args: object) -> list[str]:
    return [str(a) for a in args]


def module(name: str) -> list[str]:
    """Run tperf.<name> in the tbv2-venv (PYTHONPATH must include SRC)."""
    return [PY, "-m", f"tperf.{name}"]


def last_line(args: list, prefix: str = "", **kw) -> bool:
    """Run a command, print the last line of its stdout (the summary); True on success."""
    p = subprocess.run(argv(*args), text=True, stdout=subprocess.PIPE, **kw)
    lines = p.stdout.splitlines()
    if lines:
        print(prefix + lines[-1], flush=True)
    return p.returncode == 0


class Daily:
    def __init__(self, stamp: str):
        self.stamp = stamp
        self.w = f"{D}/work/{stamp}"
        self.out = D / "results" / stamp
        self.sha = ""
        self.harness_version = ""

    def start_suite(self, s: str, i: int) -> tuple[subprocess.Popen | None, str]:
        """Write the manifest and launch one run of a suite; returns its process (None if the manifest failed)."""
        gpu, model, args, scen = SUITES[s]
        args = ARGS or args
        node = GPU_NUMA_MAP[gpu]
        cpus = cpus_for(node)
        d = self.out / s / f"run{i}"
        cc = f"/mnt/cache/compile-cache/daily-{self.stamp}-{s}-run{i}"
        scen_arg = ["--config-file", str(scen)] if scen else ["--level", "1"]
        d.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(cc, ignore_errors=True)
        env = dict(
            os.environ,
            PYTHONPATH=f"{SRC}:{self.w}/src",
            TRANSFORMERS_DIR=self.w,
            BENCH_SOURCE="main",
            BENCH_PR="",
            BENCH_BRANCH="main",
            BENCH_HARNESS=self.harness_version,
            TORCHINDUCTOR_CACHE_DIR=f"{cc}/inductor",
            TRITON_CACHE_DIR=f"{cc}/triton",
        )
        logf = open(d / "run.log", "w")
        io = dict(cwd=d, env=env, stdout=logf, stderr=subprocess.STDOUT)
        manifest = subprocess.run(
            argv(*module("common.record"), "manifest", "--out", d / "manifest.json", "--gpu", gpu,
             "--cpus", cpus, "--numa-node", node, "--model", model, "--bench-args", f"{args} {shlex.join(scen_arg)}",
             "--group", f"{self.stamp}-{LABEL_PREFIX}-{s}", "--run", i, "--label", f"{LABEL_PREFIX}-{s}"),
            **io,
        )  # fmt: skip
        if manifest.returncode:
            logf.close()
            return None, cc
        io["env"] = dict(env, CUDA_VISIBLE_DEVICES=str(gpu))
        proc = subprocess.Popen(
            argv("numactl", f"--physcpubind={cpus}", f"--membind={node}",
             *module("common.noise_probe"), "--cpus", cpus, "--gpu", gpu, "--out", d / "probe.json", "--",
             *module("harness.run_benchmarks"), "--output-dir", d, "--model-id", model, *shlex.split(args), *scen_arg,
             "--no-gpu-monitoring", "--commit-id", self.sha, "--log-level", "WARNING"),
            **io,
        )  # fmt: skip
        logf.close()  # the child has its own copy
        return proc, cc

    def run_gpu_suites(self, suites: list[str]) -> bool:
        ok = True
        for i in range(1, RUNS + 1):
            log(f"run {i}/{RUNS}: {' '.join(suites)}")
            tune_host()
            # Manifests first, then all suites at once, so no manifest is taken while another suite runs
            started = [(s, *self.start_suite(s, i)) for s in suites]
            for s, proc, cc in started:
                if proc is None or proc.wait():
                    log(f"error: {s} run {i} failed, see {self.out}/{s}/run{i}/run.log")
                    ok = False
                shutil.rmtree(cc, ignore_errors=True)
        return ok

    def run_processors(self) -> bool:
        """All runs of the processors suite, then publish them as one group."""
        cpus = cpus_for(0)
        scen = D / "scenarios/processors.jsonl"
        env = dict(os.environ, PYTHONPATH=f"{SRC}:{self.w}/src:{PROC_PYTHONPATH_EXTRA}")
        # Processors and the encoders' weights, downloaded once, not timed
        download = f"""import json
from huggingface_hub import snapshot_download
from transformers import AutoProcessor
scenarios = [json.loads(l) for l in open({str(scen)!r}) if l.startswith('{{')]
for m in {{s['model'] for s in scenarios}}: AutoProcessor.from_pretrained(m)
for m in {{s['model'] for s in scenarios if s.get('stage', 'processor') != 'processor'}}:
    snapshot_download(m, allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt'])"""
        quiet = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if subprocess.run([PY, "-c", download], env=env, **quiet).returncode:
            return False
        run_env = dict(env, OMP_NUM_THREADS=str(NCORES), CUDA_VISIBLE_DEVICES="0")
        dirs = []
        for i in range(1, RUNS + 1):
            log(f"processors run {i}/{RUNS}")
            tune_host()
            d = self.out / "processors" / f"run{i}"
            d.mkdir(parents=True, exist_ok=True)
            dirs.append(d)
            with open(d / "run.log", "w") as logf:
                rc = subprocess.run(
                    argv("numactl", f"--physcpubind={cpus}", "--membind=0", *module("processors.bench"),
                     "--scenarios", scen, "--output-dir", d, "--threads", NCORES),
                    cwd="/tmp", env=run_env, stdout=logf, stderr=subprocess.STDOUT,
                ).returncode  # fmt: skip
            if rc:
                log(f"error: processors run {i} failed, see {d}/run.log")
                return False
        pub_env = dict(env, TRANSFORMERS_DIR=self.w, BENCH_SOURCE="main", BENCH_BRANCH="main")
        return last_line(
            [*module("common.record"), "publish-processors", *dirs, "--repo", PUBLISH_REPO,
             "--group", f"{self.stamp}-{LABEL_PREFIX}-processors", "--label", f"{LABEL_PREFIX}-processors",
             "--gpu", "0", "--cpus", cpus],
            "processors: ", cwd="/tmp", env=pub_env,
        )  # fmt: skip

    def run(self, suites: list[str]) -> int:
        gpu_suites = [s for s in suites if s != "processors"]
        self.out.mkdir(parents=True, exist_ok=True)

        # --- Latest main ---
        git("fetch", "-q", REPO_URL, "+main:refs/daily/main")
        self.sha = git("rev-parse", MAIN_SHA or "refs/daily/main")
        git("worktree", "add", "-q", "--detach", self.w, self.sha)
        subject = subprocess.run(
            ["git", "-C", self.w, "log", "-1", "--format=%s"], check=True, text=True, capture_output=True
        ).stdout.strip()
        log(f"main {self.sha[:10]}: {subject}")
        self.harness_version = HARNESS_VERSION.read_text().strip()
        log(f"harness {self.harness_version}")

        for s in gpu_suites:
            subprocess.run(
                ["/opt/bench/tbv2-venv/bin/hf", "download", SUITES[s][1]], check=True, stdout=subprocess.DEVNULL
            )

        ok = self.run_gpu_suites(gpu_suites) if gpu_suites else True
        if "processors" in suites and not self.run_processors():
            log("error: processors failed")
            ok = False
        for s in gpu_suites:
            runs = sorted((self.out / s).glob("run*"))
            if not last_line([*module("common.record"), "publish", *runs, "--repo", PUBLISH_REPO], f"{s}: "):
                log(f"error: publishing {s} failed")
                ok = False
        if DASHBOARD_SPACE:
            dash = [*module("dashboard.build"), "--dataset", PUBLISH_REPO,
                    "--out", D / "dashboard", "--publish", DASHBOARD_SPACE]  # fmt: skip
            if not last_line(dash):
                log("error: dashboard update failed")
                ok = False
        # Raw files are in the dataset; keep a month of them here
        for p in (D / "results").iterdir():
            if (time.time() - p.stat().st_mtime) // 86400 > 30:
                shutil.rmtree(p, ignore_errors=True)
        return 0 if ok else 1

    def cleanup(self) -> None:
        subprocess.run(["git", "-C", TF, "worktree", "remove", "--force", self.w], stderr=subprocess.DEVNULL)
        subprocess.run(["git", "-C", TF, "worktree", "prune"])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("suites", nargs="*", help=f"default: {' '.join(SUITES)} processors")
    a = p.parse_args(argv)
    suites = a.suites or [*SUITES, "processors"]
    unknown = [s for s in suites if s != "processors" and s not in SUITES]
    if unknown:
        p.error(f"unknown suite(s): {' '.join(unknown)}")

    lock = open(LOCK, "w")  # held until exit
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("another daily run holds the lock, skipping")
        return 0
    if subprocess.run(["pgrep", "-f", "run_benchmarks"], stdout=subprocess.DEVNULL).returncode == 0:
        log("a benchmark is already running on this host, skipping")
        return 0

    daily = Daily(datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    try:
        failed = daily.run(suites)
    finally:
        daily.cleanup()
    log(f"done (failed={failed})")
    return failed


if __name__ == "__main__":
    sys.exit(main())
