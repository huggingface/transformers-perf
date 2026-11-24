"""tperf memory profile: record where memory goes during one generation scenario, on the bench host, as an HTML report.

Usage: MODEL=<hf model id> tperf memory profile [<pr number>|main] [label=memory]
  GPU=0                physical GPU index (pinned with its NUMA node's CPUs and memory)
  PROFILE_ARGS="-b 1 -s 128 -n 64 --attn sdpa"   tperf.memory.profile arguments (see --help there), e.g.
                       --compile-mode default, --continuous-batching, --no-native (much faster host tracking)
  MAIN_SHA=<commit>    transformers commit to profile instead of the latest main (with `main`)
  SRC_PATCHES=<patch>...   transformers changes to apply first (must apply)

Runs tperf.memory.profile under the daily lock, in a worktree of the PR head (or main), then fetches the profile
directory to results/<timestamp>-<label>/: report.html (open it in a browser), host-flamegraph.html, the raw
gpu_snapshot.pickle (pytorch.org/memory_viz) and host.memray.bin. Every allocation is recorded, so the run is slower
than a benchmark and its timings mean nothing; the measured runs read only counters (startup_* memory columns).
"""

import argparse
import os
import shlex
import sys
from pathlib import Path

from tperf.common.benchlib import (
    GPU_NUMA_MAP, HF_ENV, PY, REPO_URL, ROOT, TF, Worktree, cmd, cpus_for, die, download_model, env_args,
    fetch_main, fetch_results, locked, log, push_package, scp_to, ssh, ssh_out, stamp,
)  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("target", nargs="?", default="main", help="a transformers PR number, or main (default)")
    p.add_argument("label", nargs="?", default="memory")
    a = p.parse_args(argv)

    model = os.environ.get("MODEL") or die("set MODEL to a Hugging Face model id")
    gpu = int(os.environ.get("GPU", "0"))
    ncores = int(os.environ.get("NCORES", "8"))
    profile_args = shlex.split(os.environ.get("PROFILE_ARGS", ""))
    patches = os.environ.get("SRC_PATCHES", "").split()
    if a.target != "main" and not a.target.isdigit():
        die(f"target must be a PR number or main, not {a.target!r}")

    st = stamp()
    out = ROOT / "results" / f"{st}-{a.label}"
    rdir = f"/opt/bench/memory-results/{st}-{a.label}"
    out.mkdir(parents=True, exist_ok=True)
    log(f"Results: {out}")
    ssh(["mkdir", "-p", f"{rdir}/src-patches", "/opt/bench/memory-work"])
    src = f"{rdir}/src"  # this run's copy of the package
    push_package(src)
    if patches:
        scp_to(patches, f"{rdir}/src-patches/")

    if a.target == "main":
        sha = fetch_main(rev=os.environ.get("MAIN_SHA") or None)
    else:
        ref = f"refs/memory/pr{a.target}"
        ssh(["git", "-C", TF, "fetch", "-q", REPO_URL, f"+pull/{a.target}/head:{ref}"])
        sha = ssh_out(["git", "-C", TF, "rev-parse", ref])
    download_model(model)

    node = GPU_NUMA_MAP[gpu]
    cpus = cpus_for(node, ncores)
    with Worktree(f"/opt/bench/memory-work/{st}", sha) as w:
        for patch in patches:
            ssh(["git", "-C", w.path, "apply", f"{rdir}/src-patches/{Path(patch).name}"])
            print(f"applied {Path(patch).name}", file=sys.stderr)
        log(f"transformers {sha[:12]} ({a.target}), {model}, GPU {gpu}")
        cc = f"/mnt/cache/compile-cache/{st}-memory"
        profile = [
            *env_args({**HF_ENV, "HF_HUB_DISABLE_PROGRESS_BARS": 1, "PYTHONPATH": f"{src}:{w.path}/src",
                       "CUDA_VISIBLE_DEVICES": gpu, "OMP_NUM_THREADS": ncores,
                       "TORCHINDUCTOR_CACHE_DIR": f"{cc}/inductor", "TRITON_CACHE_DIR": f"{cc}/triton"}),
            "numactl", f"--physcpubind={cpus}", f"--membind={node}",
            PY, "-m", "tperf.memory.profile", "--model-id", model, "--out", f"{rdir}/profile", *profile_args,
        ]  # fmt: skip
        rc = ssh(f"rm -rf {cc}; {cmd(locked(profile))} >{rdir}/profile.log 2>&1; rc=$?; rm -rf {cc}; exit $rc",
                 check=False).returncode  # fmt: skip
    if rc == 75:
        die("a benchmark holds the daily lock")
    fetch_results(rdir, out)
    report = out / "profile" / "report.html"
    if rc:
        print(f"profile failed (exit {rc}): {out / 'profile.log'}", file=sys.stderr)
        ssh(["tail", "-30", f"{rdir}/profile.log"], check=False, stdout=sys.stderr)
    if report.exists():
        print(report)
    sys.exit(1 if rc else 0)


if __name__ == "__main__":
    main()
