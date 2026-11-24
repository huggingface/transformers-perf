"""tperf serve run: a multi-turn chat against `transformers serve` on the bench host, per-turn latency (and optionally
the server's Python time).

Usage: MODEL=<hf model id> tperf serve run [<pr number>|main] [label=serve]
  GPU=0                  physical GPU index (pinned with its NUMA node's CPUs and memory)
  NCORES=8               CPU cores for the server and the client
  SCENARIOS=scenarios/serve-multiturn.jsonl   scenario file; SCENARIO=<index or name> (default: the first)
  SERVE_BENCH_ARGS="--turns 8 --profile"      more tperf.serve.bench arguments (they win over the scenario's), e.g.
                         --profile (py-spy on the server), --concurrency 4, --server-args "--continuous-batching"
  MAIN_SHA=<commit>      transformers commit to run instead of the latest main (with `main`)
  SRC_PATCHES=<patch>... transformers changes to apply first (must apply)

Runs tperf.serve.bench under the daily lock, in a worktree of the PR head (or main), with the server and the client on
the GPU's NUMA node, then fetches results/<timestamp>-<label>/: results.json, server.log, and server.folded with
--profile. Compare two runs with `tperf serve summarize <base dir> <dir>`.
"""

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

from tperf.common.benchlib import (
    GPU_NUMA_MAP, HF_ENV, PY, REPO_URL, ROOT, TF, Worktree, cmd, cpus_for, die, download_model, env_args,
    fetch_main, fetch_results, locked, log, push_package, scp_to, ssh, ssh_out, stamp,
)  # fmt: skip


SERVE_IMPORTS = "import fastapi, openai, pydantic, uvicorn"  # transformers' `serving` extra


def scenario_index(path: Path, wanted: str) -> int:
    lines = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip().startswith("{")]
    if wanted.isdigit():
        if int(wanted) >= len(lines):
            die(f"{path} has {len(lines)} scenario(s)")
        return int(wanted)
    for i, s in enumerate(lines):
        if s.get("name") == wanted:
            return i
    die(f"no scenario named {wanted!r} in {path}: {', '.join(s.get('name', '?') for s in lines)}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("target", nargs="?", default="main", help="a transformers PR number, or main (default)")
    p.add_argument("label", nargs="?", default="serve")
    a = p.parse_args(argv)

    model = os.environ.get("MODEL") or die("set MODEL to a Hugging Face model id (a chat model)")
    gpu = int(os.environ.get("GPU", "0"))
    ncores = int(os.environ.get("NCORES", "8"))
    scenarios = Path(os.environ.get("SCENARIOS", ROOT / "scenarios/serve-multiturn.jsonl"))
    index = scenario_index(scenarios, os.environ.get("SCENARIO", "0"))
    extra = shlex.split(os.environ.get("SERVE_BENCH_ARGS", ""))
    patches = os.environ.get("SRC_PATCHES", "").split()
    if a.target != "main" and not a.target.isdigit():
        die(f"target must be a PR number or main, not {a.target!r}")

    st = stamp()
    out = ROOT / "results" / f"{st}-{a.label}"
    rdir = f"/opt/bench/serve-results/{st}-{a.label}"
    out.mkdir(parents=True, exist_ok=True)
    log(f"Results: {out}")
    ssh(["mkdir", "-p", f"{rdir}/src-patches", "/opt/bench/serve-work"])
    if ssh([PY, "-c", SERVE_IMPORTS], check=False).returncode:
        die("the bench host's harness venv lacks transformers' serving dependencies: run `tperf host install`")
    src = f"{rdir}/src"  # this run's copy of the package
    push_package(src)
    scp_to([scenarios], f"{rdir}/")
    if patches:
        scp_to(patches, f"{rdir}/src-patches/")

    if a.target == "main":
        sha = fetch_main(rev=os.environ.get("MAIN_SHA") or None)
    else:
        ref = f"refs/serve/pr{a.target}"
        ssh(["git", "-C", TF, "fetch", "-q", REPO_URL, f"+pull/{a.target}/head:{ref}"])
        sha = ssh_out(["git", "-C", TF, "rev-parse", ref])
    download_model(model)

    node = GPU_NUMA_MAP[gpu]
    cpus = cpus_for(node, ncores)
    with Worktree(f"/opt/bench/serve-work/{st}", sha) as w:
        for patch in patches:
            ssh(["git", "-C", w.path, "apply", f"{rdir}/src-patches/{Path(patch).name}"])
            print(f"applied {Path(patch).name}", file=sys.stderr)
        log(f"transformers {sha[:12]} ({a.target}), {model}, GPU {gpu}, scenario {index} of {scenarios.name}")
        cc = f"/mnt/cache/compile-cache/{st}-serve"
        bench = [
            *env_args({**HF_ENV, "HF_HUB_OFFLINE": 1, "PYTHONPATH": f"{src}:{w.path}/src", "CUDA_VISIBLE_DEVICES": gpu,
                       "OMP_NUM_THREADS": ncores, "TORCHINDUCTOR_CACHE_DIR": f"{cc}/inductor",
                       "TRITON_CACHE_DIR": f"{cc}/triton"}),
            "numactl", f"--physcpubind={cpus}", f"--membind={node}",
            PY, "-m", "tperf.serve.bench", "--model-id", model, "--out", f"{rdir}/serve", "--python", PY,
            "--commit", sha, "--config-file", f"{rdir}/{scenarios.name}", "--scenario-index", index, *extra,
        ]  # fmt: skip
        rc = ssh(f"rm -rf {cc}; {cmd(locked(bench))} >{rdir}/serve.log 2>&1; rc=$?; rm -rf {cc}; exit $rc",
                 check=False).returncode  # fmt: skip
    if rc == 75:
        die("a benchmark holds the daily lock")
    fetch_results(rdir, out)
    if rc:
        print(f"serve bench failed (exit {rc}): {out / 'serve.log'}", file=sys.stderr)
        ssh(["tail", "-30", f"{rdir}/serve.log"], check=False, stdout=sys.stderr)
    results = out / "serve" / "results.json"
    if results.exists():
        from tperf.serve.summarize import format_results

        print(format_results(json.loads(results.read_text())))
        print(results.parent)
    sys.exit(1 if rc else 0)


if __name__ == "__main__":
    main()
