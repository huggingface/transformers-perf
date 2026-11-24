"""Side-by-side benchmark of a transformers PR against the commit it is based on.

Usage: MODEL=<model> tperf ab <pr number> [rounds=6]
       MODEL=<model> HEAD_EXTRA_PATCHES=<patch> tperf ab main [rounds]   # A/B of a benchmark change itself
       MODEL=<model> HEAD_PATCHES=<patch> tperf ab main [rounds]         # A/B of a local transformers change
       MODEL=<chat model> SERVE=chat tperf ab <pr number> [rounds]       # a multi-turn chat on `transformers serve`

Environment:
  HOST          bench host, user@address (default: [host] ssh in tperf.toml)
  MODEL         model id (required)
  SCENARIOS     harness config file (.jsonl) with the scenarios to run (default: --level 1)
  SERVE         instead of the harness, run this scenario (name or index) of SERVE_SCENARIOS on each side with
                tperf.serve.bench: `transformers serve` from the side's checkout, a client resending the conversation
                each turn. Each side's requests are its samples (ttft, itl, e2e, tok/s; load = server start).
                SERVE_SCENARIOS defaults to scenarios/serve-multiturn.jsonl, SERVE_BENCH_ARGS adds tperf.serve.bench
                arguments (BENCH_ARGS and SCENARIOS are not used)
  BENCH_ARGS    tperf.harness.run_benchmarks arguments (default: -w 5 -i 10 -b 1 -s 128 -n 64). A PR changes the
                cost of prefill and/or of each decode step, so a few dozen decode steps show it as well as hundreds:
                raise -n only when the change needs a longer sequence (e.g. crossing a sliding window)
  BASE          commit to compare against (default: the merge base of the PR with main, so the only
                difference between the sides is the PR's diff; for a merged PR, head is the commit it was
                merged as and base is its parent, i.e. main right before the merge)
  HARNESS_PATCHES  patches to the harness applied to both sides (default: none), e.g. patches/*.patch; paths
                start with src/tperf/harness/ (relative to this repository's root)
  HEAD_PATCHES  patches applied to the head side's transformers checkout (with `main`: test a change before
                opening its PR; base is the main tip, head is the main tip plus the patches)
  HEAD_EXTRA_PATCHES  patches applied to the head side's harness only (a separate copy of the package), to A/B a
                change to the benchmark itself; with `main` as the PR, both sides run the main tip. Patch paths
                start with src/tperf/harness/
  GPUS="0 4"    the two GPUs used, one per NUMA node; NCORES=8 cores each on the GPU's node
  TP=1          tensor parallel size: each side runs under torchrun with --enable-tp on TP GPUs of one NUMA node
                (GPUS defaults to "0,1 4,5" for TP=2, "0,1,2,3 4,5,6,7" for TP=4), NCORES cores per rank
  NCCL_ENV="NCCL_CUMEM_HOST_ENABLE=0"   for TP > 1, see `tperf baselines p2p`
  PUBLISH_REPO  publish every run to this dataset (rows tagged ab_id / ab_side / ab_round)
  DASHBOARD_SPACE  old static Space to rebuild after publishing (default: [hub] dashboard_space in tperf.toml,
                none: results are browsed at transformers-ci /benchmarks)

Layout on the host, under /opt/bench/ab/<id>/: base/ and head/ (worktrees of the two commits), src/tperf (this
repository's package, with HARNESS_PATCHES applied: its harness drives both sides), src-head/tperf (head's copy, with
HEAD_EXTRA_PATCHES too, only when set), results/round<r>-<side>/.

Each round re-applies the host tuning, then runs base and head at the same time, one per GPU (one per NUMA
node, so they share no cores, memory or PCIe), each with PYTHONPATH pointing at the package copy and its checkout's
src/. The GPUs
swap every round, so chip-to-chip differences cancel out. `tperf ab compare` then compares the sides round by round.
"""

import argparse
import contextlib
import glob
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


from tperf.common.benchlib import (
    DASHBOARD_SPACE, GPU_NUMA_MAP, ROOT, HF_ENV, HOST, PY, REPO_URL, SSH, TF, Worktree, cmd, cpus_for, daily_lock_free, die,
    download_model, env_args, log, push_package, remote_module, run_module, scp_from, scp_to, ssh, ssh_out, stamp,
    tune_host,
)  # fmt: skip


DEFAULT_GPUS = {1: "0 4", 2: "0,1 4,5", 4: "0,1,2,3 4,5,6,7"}


def patches(var: str) -> list[str]:
    """Whitespace-separated patch paths from an environment variable; globs are expanded like the shell did."""
    return [p for word in os.environ.get(var, "").split() for p in (sorted(glob.glob(word)) or [word])]


def names(paths: list[str]) -> str:
    return ",".join(os.path.basename(p) for p in paths)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pr", help="PR number, or `main`")
    parser.add_argument("rounds", nargs="?", type=int, default=6)
    args = parser.parse_args(argv)
    pr, rounds = args.pr, args.rounds

    model = os.environ.get("MODEL") or die("set MODEL to a Hugging Face model id")
    ncores = int(os.environ.get("NCORES", "8"))
    bench_args = os.environ.get("BENCH_ARGS", "-w 5 -i 10 -b 1 -s 128 -n 64")
    tp = int(os.environ.get("TP", "1"))
    if tp not in DEFAULT_GPUS:
        die("TP must be 1, 2 or 4")
    gpus = os.environ.get("GPUS", DEFAULT_GPUS[tp]).split()
    nccl_env = os.environ.get("NCCL_ENV", "NCCL_CUMEM_HOST_ENABLE=0")
    publish_repo = os.environ.get("PUBLISH_REPO", "")
    dashboard_space = os.environ.get("DASHBOARD_SPACE", DASHBOARD_SPACE)
    scenarios = os.environ.get("SCENARIOS", "")
    harness_patches = patches("HARNESS_PATCHES")
    head_patches = patches("HEAD_PATCHES")
    head_extra_patches = patches("HEAD_EXTRA_PATCHES")
    serve = os.environ.get("SERVE", "")
    serve_scenarios = os.environ.get("SERVE_SCENARIOS", str(ROOT / "scenarios/serve-multiturn.jsonl"))
    serve_extra = shlex.split(os.environ.get("SERVE_BENCH_ARGS", ""))
    if serve:
        from tperf.serve.run import SERVE_IMPORTS, scenario_index

        if tp > 1:
            die("SERVE runs one GPU per side (TP=1)")
        serve_index = scenario_index(Path(serve_scenarios), serve)
        bench_args = shlex.join(["serve", serve, *serve_extra])

    if len(gpus) != 2:
        die("GPUS needs two GPU groups (one GPU each without TP)")
    for group in gpus:
        ids = [int(g) for g in group.split(",")]
        if len(ids) != tp:
            die(f"GPU group {group} does not have TP={tp} GPUs")
        if len({GPU_NUMA_MAP[g] for g in ids}) != 1:
            die(f"GPU group {group} spans NUMA nodes (NCCL's GPU->host writes drop 4x)")
    node_of = {group: GPU_NUMA_MAP[int(group.split(",")[0])] for group in gpus}
    if node_of[gpus[0]] == node_of[gpus[1]]:
        print(f"warning: GPUs {' '.join(gpus)} are on the same NUMA node; the sides will share memory bandwidth",
              file=sys.stderr)  # fmt: skip

    # The daily main benchmark (tperf.daily.main) holds this lock while it runs
    if not daily_lock_free():
        die(f"the daily main benchmark is running on {HOST}, try later")

    # --- Commits ---
    git = ["git", "-C", TF]
    base = os.environ.get("BASE")
    if pr == "main":
        log("Fetching main")
        ssh([*git, "fetch", "-q", REPO_URL, "+main:refs/ab/main"])
        head_sha = ssh_out([*git, "rev-parse", "refs/ab/main"])
        base_sha = ssh_out([*git, "rev-parse", base or head_sha])
        pr_num, name = "", "main"
        if head_patches:
            name = os.path.basename(head_patches[0]).removesuffix(".patch")
    else:
        info = json.loads(subprocess.run(
            ["gh", "pr", "view", pr, "-R", "huggingface/transformers", "--json", "state,mergeCommit,headRefOid"],
            check=True, capture_output=True, text=True,
        ).stdout)  # fmt: skip
        if info["state"] == "MERGED":
            # Merged PR: the commit it landed as on main, against main right before it (its first parent)
            head_sha = info["mergeCommit"]["oid"]
            log(f"Fetching main (PR #{pr} was merged as {head_sha[:10]})")
            ssh(
                f"{cmd([*git, 'fetch', '-q', REPO_URL, '+main:refs/ab/main'])} && {cmd([*git, 'cat-file', '-e', head_sha])}"
            )
            base_sha = ssh_out([*git, "rev-parse", base or f"{head_sha}^1"])
        else:
            head_sha = info["headRefOid"]
            log(f"Fetching main and PR #{pr} ({head_sha})")
            fetch = [*git, "fetch", "-q", REPO_URL, "+main:refs/ab/main", f"+pull/{pr}/head:refs/ab/pr{pr}"]
            ssh(f"{cmd(fetch)} && {cmd([*git, 'cat-file', '-e', head_sha])}")
            base_sha = ssh_out([*git, "rev-parse", base or ssh_out([*git, "merge-base", head_sha, "refs/ab/main"])])
        pr_num, name = pr, f"pr{pr}"

    ab_id = f"{stamp()}-{name}"
    ab = f"/opt/bench/ab/{ab_id}"
    out = ROOT / "results" / f"ab-{ab_id}"
    out.mkdir(parents=True, exist_ok=True)
    log(f"A/B {ab_id}: base {base_sha[:10]} vs head {head_sha[:10]}, {rounds} rounds on GPUs {' '.join(gpus)}, "
        f"results in {out}")  # fmt: skip

    ssh(["mkdir", "-p", f"{ab}/results"])
    with contextlib.ExitStack() as stack:
        stack.enter_context(Worktree(f"{ab}/base", base_sha))
        stack.enter_context(Worktree(f"{ab}/head", head_sha))

        # --- Package (harness, record, noise probe, dashboard): one copy for both sides, two with HEAD_EXTRA_PATCHES ---
        def apply_patches(target: str, paths: list[str], prefix: str = "", strip: int = 1) -> None:
            """The change is what is measured, so a patch that does not apply stops the run.

            Harness patches are relative to this repository's root (src/tperf/harness/...): applied inside a package
            copy (<copy>/tperf/harness/...) with strip=2 (`git apply -p2` drops a/src/).
            """
            for p in paths:
                remote = f"{ab}/{prefix}{os.path.basename(p)}"
                scp_to([p], remote)
                if ssh(["git", "-C", f"{ab}/{target}", "apply", f"-p{strip}", remote], check=False).returncode:
                    die(f"{os.path.basename(p)} does not apply to {target}")
                log(f"{target}: applied {os.path.basename(p)}")

        version = push_package(f"{ab}/src")
        extra_names = {os.path.basename(p) for p in head_extra_patches}
        shared = [p for p in harness_patches if p not in head_extra_patches and os.path.basename(p) not in extra_names]
        apply_patches("src", shared, strip=2)
        apply_patches("head", head_patches, prefix="head-")  # the change under test: never skip it
        head_src = "src"
        if head_extra_patches:
            head_src = "src-head"
            push_package(f"{ab}/src-head")
            apply_patches("src-head", shared + head_extra_patches, prefix="src-head-", strip=2)
        scen_arg = ["--level", "1"]
        if serve:
            scen_arg = []
            scp_to([serve_scenarios], f"{ab}/serve-scenarios.jsonl")
            if ssh([PY, "-c", SERVE_IMPORTS], check=False).returncode:
                die("the bench host's harness venv lacks transformers' serving dependencies: run `tperf host install`")
        elif scenarios:
            scp_to([scenarios], f"{ab}/scenarios.jsonl")
            scen_arg = ["--config-file", f"{ab}/scenarios.jsonl"]
        for side in ("base", "head"):
            check = (
                f"import transformers, sys; f = transformers.__file__; "
                f"sys.exit(0 if f.startswith({f'{ab}/{side}/'!r}) else f'{side} imports transformers from {{f}}')"
            )
            ssh(f"cd /tmp && {cmd([*env_args({'PYTHONPATH': f'{ab}/{side}/src'}), PY, '-c', check])}")
        log(f"Pre-downloading {model} (not timed)")
        download_model(model)

        lines = [
            f"pr={pr} base={base_sha} head={head_sha} model={model} rounds={rounds} gpus={' '.join(gpus)} tp={tp}"
        ]
        if tp > 1:
            lines.append(f"nccl_env={nccl_env}")
        lines += [
            f"head_patches={names(head_patches)}",
            f"head_extra_patches={names(head_extra_patches)}",
            f"bench_args={bench_args} scenarios={serve_scenarios if serve else scenarios or 'level 1'} "
            f"harness={version} "
            f"harness_patches={names(harness_patches)}",
        ]
        (out / "ab.txt").write_text("\n".join(lines) + "\n")
        if scenarios:
            shutil.copy(scenarios, out)

        cpus = {node: cpus_for(node, ncores * tp) for node in set(node_of.values())}
        tp_args = ["--enable-tp"] if tp > 1 else []
        launch = [PY]
        if tp > 1:
            # --standalone picks a free rendezvous port, so the two sides can start at the same time
            launch = [PY, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", tp]

        def side_cmd(side: str, group: str, rnd: int) -> str:
            """Shell command for one side of a round, run on the host concurrently with the other side."""
            gpu, node = group.split(",")[0], node_of[group]
            sha = base_sha if side == "base" else head_sha
            out_dir = f"{ab}/results/round{rnd}-{side}"
            src = f"{ab}/src" if side == "base" else f"{ab}/{head_src}"  # the package (harness) this side runs
            # Fresh torch.compile / Triton caches per run: every run measures a true cold start, whatever ran before
            cc = f"/mnt/cache/compile-cache/{ab_id}-r{rnd}-{side}"
            env = {
                **HF_ENV,
                "PYTHONPATH": f"{src}:{ab}/{side}/src",
                "TRANSFORMERS_DIR": f"{ab}/{side}",
                "BENCH_SOURCE": "main" if side == "base" else "pr",
                "BENCH_PR": pr_num,
                "BENCH_BRANCH": "",
                "BENCH_AB_ID": ab_id,
                "BENCH_AB_SIDE": side,
                "BENCH_AB_ROUND": rnd,
                "BENCH_HARNESS": version,
                "TORCHINDUCTOR_CACHE_DIR": f"{cc}/inductor",
                "TRITON_CACHE_DIR": f"{cc}/triton",
            }
            run_env = {**env, "CUDA_VISIBLE_DEVICES": group}
            if tp > 1:
                run_env.update(dict(kv.split("=", 1) for kv in shlex.split(nccl_env)), OMP_NUM_THREADS=ncores)
            manifest_args = " ".join(
                [bench_args, *scen_arg, *tp_args] + (["--nproc-per-node", str(tp)] if tp > 1 else [])
            )
            manifest = [
                *env_args(env), *remote_module(PY, "common.record", env["PYTHONPATH"]), "manifest",
                "--out", f"{out_dir}/manifest.json",
                "--gpu", gpu, "--cpus", cpus[node], "--numa-node", node, "--model", model, "--bench-args", manifest_args,
                "--group", f"{ab_id}-r{rnd}-{side}", "--run", 1,
                "--label", f"ab-{name}-{side}" + (f"-tp{tp}" if tp > 1 else ""),
            ]  # fmt: skip
            workload = [
                *launch, "-m", "tperf.harness.run_benchmarks", "--output-dir", out_dir, "--model-id", model,
                *shlex.split(bench_args), *scen_arg, *tp_args, "--no-gpu-monitoring", "--commit-id", sha,
                "--log-level", "WARNING",
            ]  # fmt: skip
            if serve:
                # the server inherits this process's CPUs and memory node; HF_HUB_OFFLINE: the model is downloaded
                run_env.update(HF_HUB_OFFLINE=1, OMP_NUM_THREADS=ncores)
                workload = [
                    PY, "-m", "tperf.serve.bench", "--model-id", model, "--out", f"{out_dir}/serve", "--python", PY,
                    "--commit", sha, "--config-file", f"{ab}/serve-scenarios.jsonl", "--scenario-index", serve_index,
                    *serve_extra,
                ]  # fmt: skip
            run = [
                *env_args(run_env),
                "numactl", f"--physcpubind={cpus[node]}", f"--membind={node}",
                *remote_module(PY, "common.noise_probe", env["PYTHONPATH"]),
                "--cpus", cpus[node], "--gpu", gpu, "--out", f"{out_dir}/probe.json", "--", *workload,
            ]  # fmt: skip
            return (
                f"mkdir -p {out_dir} && rm -rf {cc} && cd {out_dir} && {cmd(manifest)} && "
                f"{cmd(run)} >{out_dir}/run.log 2>&1; rc=$?; rm -rf {cc}; exit $rc"
            )

        for r in range(1, rounds + 1):
            # Alternate which side gets which GPU
            gb, gh = (gpus[0], gpus[1]) if r % 2 else (gpus[1], gpus[0])
            log(f"Round {r}/{rounds}: base on GPU {gb}, head on GPU {gh}" + (f" (TP{tp})" if tp > 1 else ""))
            tune_host("apply")
            procs = [subprocess.Popen([*SSH, HOST, side_cmd(side, g, r)]) for side, g in (("base", gb), ("head", gh))]
            if any([p.wait() for p in procs]):
                print(f"round {r} failed, logs in {ab}/results/round{r}-*/run.log", file=sys.stderr)
                ssh(f"tail -15 {ab}/results/round{r}-*/run.log", check=False, stdout=sys.stderr)
                sys.exit(1)
            for side in ("base", "head"):
                scp_from(f"{ab}/results/round{r}-{side}", out, recursive=True)

    # A regression must not stop publishing: keep the verdict for the exit status
    compare = run_module("ab.compare", out, stdout=subprocess.PIPE, text=True)
    print(compare.stdout, end="")
    (out / "compare.txt").write_text(compare.stdout)

    if publish_repo:
        log(f"Publishing to {publish_repo}")
        hub_env = env_args({**HF_ENV, "HF_HUB_DISABLE_PROGRESS_BARS": 1})
        published = ssh(
            f"{cmd([*hub_env, *remote_module(PY, 'common.record', f'{ab}/src'), 'publish'])} {ab}/results/round* "
            f"{cmd(['--repo', publish_repo])}",
            capture=True,
        ).stdout.strip()  # fmt: skip
        last = published.splitlines()[-1] if published else ""
        print(last)
        (out / "publish.txt").write_text(last + "\n")
        if dashboard_space:
            try:
                built = ssh_out([
                    *hub_env, "nice", "-n", 19, *remote_module(PY, "dashboard.build", f"{ab}/src"),
                    "--dataset", publish_repo, "--out", "/tmp/bench-dashboard", "--publish", dashboard_space,
                ])  # fmt: skip
                print(built.splitlines()[-1] if built else "")
            except subprocess.CalledProcessError:
                print("warning: dashboard update failed", file=sys.stderr)
    sys.exit(compare.returncode)


if __name__ == "__main__":
    main()
