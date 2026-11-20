"""tperf tp profile: profile compiled decode steps per TP size (tperf.tp.profile) on the bench host, from your workstation.

Usage: tperf tp profile [label=tp-profile]
  GPU_GROUPS="0 0,1 0,1,2,3"   MODEL=meta-llama/Llama-2-7b-hf   MAIN_SHA=<commit, default latest main>
  PROFILE_ARGS="--decode-tokens 16"   NCORES=8 (cores per rank)   NCCL_ENV="NCCL_CUMEM_HOST_ENABLE=0"
  SRC_PATCHES=<patch>...   transformers changes to profile (must apply)

Same setup as `tperf tp run` (worktree of main, one NUMA node per group, daily lock held). Prints each
rank's breakdown of one decode step; traces and JSON land in results/<timestamp>-<label>/.
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


from tperf.common.benchlib import (
    GPU_NUMA_MAP,
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
    run_module,
    scp_to,
    ssh,
    stamp,
    tune_host,
)


MODEL = os.environ.get("MODEL") or "meta-llama/Llama-2-7b-hf"
GPU_GROUPS = (os.environ.get("GPU_GROUPS") or "0 0,1 0,1,2,3").split()
PROFILE_ARGS = shlex.split(os.environ.get("PROFILE_ARGS", ""))
MAIN_SHA = os.environ.get("MAIN_SHA") or None
NCORES = int(os.environ.get("NCORES") or 8)
NCCL_ENV = os.environ.get("NCCL_ENV", "NCCL_CUMEM_HOST_ENABLE=0")
SRC_PATCHES = os.environ.get("SRC_PATCHES", "").split()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("label", nargs="?", default="tp-profile")
    a = p.parse_args(argv)

    st = stamp()
    out = ROOT / "results" / f"{st}-{a.label}"
    rdir = f"/opt/bench/tp-results/{st}-{a.label}"
    out.mkdir(parents=True, exist_ok=True)
    log(f"Results: {out}")
    ssh(["mkdir", "-p", f"{rdir}/src-patches", "/opt/bench/tp-work"])
    src = f"{rdir}/src"  # this run's copy of the package (tperf.tp.profile)
    push_package(src)
    if SRC_PATCHES:
        scp_to(SRC_PATCHES, f"{rdir}/src-patches/")

    sha = fetch_main(rev=MAIN_SHA)
    with Worktree(f"/opt/bench/tp-work/{st}", sha) as w:
        for patch in SRC_PATCHES:
            ssh(["git", "-C", w.path, "apply", f"{rdir}/src-patches/{Path(patch).name}"])
            print(f"applied {Path(patch).name}", file=sys.stderr)
        log(f"transformers {sha}")

        tune_host("apply")
        env = {**HF_ENV, "HF_HUB_DISABLE_PROGRESS_BARS": 1, **dict(kv.split("=", 1) for kv in NCCL_ENV.split())}
        for group in GPU_GROUPS:
            n = group.count(",") + 1
            node = GPU_NUMA_MAP[int(group.split(",")[0])]
            cpus = cpus_for(node, NCORES * n)
            cc = f"/mnt/cache/compile-cache/{st}-tp{n}"
            # `-m tperf.tp.profile`, found through PYTHONPATH (plain python, or torchrun's own -m)
            launch = [PY, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", n] if n > 1 else [PY]
            log(f"TP{n} on GPUs {group}")
            profile = [
                *env_args({**env, "PYTHONPATH": f"{src}:{w.path}/src", "TORCHINDUCTOR_CACHE_DIR": f"{cc}/inductor",
                           "TRITON_CACHE_DIR": f"{cc}/triton", "CUDA_VISIBLE_DEVICES": group,
                           "OMP_NUM_THREADS": NCORES}),
                "numactl", f"--physcpubind={cpus}", f"--membind={node}",
                *launch, "-m", "tperf.tp.profile", "--model", MODEL, "--out", f"{rdir}/traces", *PROFILE_ARGS,
            ]  # fmt: skip
            rc = ssh(
                f"rm -rf {cc}; {cmd(locked(profile))} >{rdir}/tp{n}.jsonl 2>{rdir}/tp{n}.log;"
                f" rc=$?; rm -rf {cc}; exit $rc",
                check=False,
            ).returncode
            if rc:
                if rc == 75:
                    die("a benchmark holds the daily lock")
                print(f"TP{n} failed", file=sys.stderr)
                ssh(["tail", "-30", f"{rdir}/tp{n}.log"], check=False, stdout=sys.stderr)
                sys.exit(1)

    fetch_results(rdir, out)
    summary = run_module(
        "tp.summarize", *sorted(out.glob("tp*.jsonl")), check=True, text=True, stdout=subprocess.PIPE
    ).stdout
    print(summary, end="")
    (out / "summary.txt").write_text(summary)


if __name__ == "__main__":
    main()
