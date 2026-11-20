"""Helpers shared by the commands that drive the bench host from a workstation (tperf ab, tperf generation run, ...).

Remote commands are built as argument lists and quoted with shlex.join; `ssh` runs them on HOST. Each run copies the
tperf package, the harness included, to its own directory on the host (`push_package(f"{rdir}/src")`) and runs its
host-side modules from there with `env PYTHONPATH=<rdir>/src[:...] <python> -m tperf.<module>` (`remote_module`), so it
never replaces the copy the daily run uses (REMOTE_SRC, written only by `tperf deploy`). The modules the host's system
python runs (tperf.daily.main, tperf.host.prepare, tperf.host.install, tperf.baselines.nccl_launch) only use the
standard library.
"""

import os
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NoReturn


PKG_SRC = Path(__file__).resolve().parents[2]  # the directory holding the tperf package (src/ in the checkout)


def _repo_root() -> Path:
    """The checkout this package runs from (editable install, or PYTHONPATH=src), else the current directory."""
    root = PKG_SRC.parent
    return root if (root / "pyproject.toml").is_file() and (root / "src/tperf").is_dir() else Path.cwd()


ROOT = _repo_root()  # repository root: scenarios/, patches/, results/


def _load_config() -> dict:
    """tperf.toml (not committed, see tperf.example.toml): $TPERF_CONFIG, the checkout, or ~/.config/tperf/config.toml."""
    import tomllib

    for path in (os.environ.get("TPERF_CONFIG"), ROOT / "tperf.toml", Path.home() / ".config/tperf/config.toml"):
        if path and Path(path).is_file():
            with open(path, "rb") as f:
                return tomllib.load(f)
    return {}


CONFIG = _load_config()
HOST = os.environ.get("HOST") or CONFIG.get("host", {}).get("ssh", "")  # user@address of the bench host
# Where results are published; the public defaults are overridden in tperf.toml
DATASET = CONFIG.get("hub", {}).get("dataset", "hf-internal-testing/transformers-perf")
DASHBOARD_SPACE = CONFIG.get("hub", {}).get("dashboard_space", "hf-internal-testing/transformers-perf-dashboard")
SSH = ["ssh", "-o", "BatchMode=yes"]
TF = "/opt/bench/transformers"  # transformers clone on the host (worktrees of the commits under test)
PY = "/opt/bench/tbv2-venv/bin/python"  # harness environment on the host
HF_CLI = "/opt/bench/tbv2-venv/bin/hf"
REMOTE_SRC = "/opt/bench/src"  # the deployed tperf package the daily run uses (`tperf deploy`), for PYTHONPATH
REPO_URL = "https://github.com/huggingface/transformers.git"
DAILY_LOCK = "/tmp/bench-daily.lock"  # held by the daily run, and by every run that uses the GPUs
# Hub cache and token on the host
HF_ENV = {
    "HF_HOME": "/mnt/cache/hf",
    "HF_TOKEN_PATH": CONFIG.get("host", {}).get(
        "hf_token_path", f"/home/{HOST.partition('@')[0] or 'ubuntu'}/.cache/huggingface/token"
    ),
    "HF_HUB_DISABLE_PROGRESS_BARS": 1,
}
GPU_NUMA_MAP = [int(n) for n in os.environ.get("GPU_NUMA_MAP", "0 0 0 0 1 1 1 1").split()]


def log(msg: str) -> None:
    print(f"\033[1;34m==>\033[0m {msg}", flush=True)


def die(msg: str) -> NoReturn:
    sys.exit(msg)


def cmd(args: Sequence[object]) -> str:
    """Quote an argument list for the remote shell."""
    return shlex.join(str(a) for a in args)


def env_args(env: Mapping[str, object]) -> list[str]:
    """`env K=V ...` prefix for a remote command."""
    return ["env", *(f"{k}={v}" for k, v in env.items())]


def require_host() -> str:
    if not HOST:
        die(
            'no bench host: set [host] ssh = "user@address" in tperf.toml (see tperf.example.toml), or HOST=user@address'
        )
    return HOST


def ssh(remote: str | Sequence[object], *, check: bool = True, capture: bool = False, stdin=None, stdout=None,
        host: str | None = None) -> subprocess.CompletedProcess:  # fmt: skip
    """Run a command on the host. A list is quoted; a string is passed to the remote shell as is."""
    remote = remote if isinstance(remote, str) else cmd(remote)
    return subprocess.run(
        [*SSH, host or require_host(), remote],
        check=check,
        text=True,
        stdin=stdin,
        stdout=subprocess.PIPE if capture else stdout,
    )


def ssh_out(remote: str | Sequence[object]) -> str:
    """Stdout of a remote command, stripped."""
    return ssh(remote, capture=True).stdout.strip()


def scp_to(sources: Sequence[Path | str], dest: str) -> None:
    subprocess.run(["scp", "-q", *map(str, sources), f"{require_host()}:{dest}"], check=True)


def scp_from(source: str, dest: Path | str, recursive: bool = False) -> None:
    subprocess.run(["scp", "-q", *(["-r"] if recursive else []), f"{require_host()}:{source}", str(dest)], check=True)


def fetch_results(rdir: str, dest: Path | str) -> None:
    """Copy the contents of a remote run directory to dest, without the package pushed into it (rdir/src)."""
    subprocess.run(
        ["rsync", "-a", "--exclude=/src/", "-e", " ".join(SSH), f"{require_host()}:{rdir}/", f"{dest}/"], check=True
    )


def tune_host(action: str = "apply") -> str:
    """Run tperf/host/prepare.py <action> on the host as root (drop caches, lock clocks, ...); returns its output."""
    with open(PKG_SRC / "tperf/host/prepare.py") as f:
        return ssh(["sudo", "python3", "-", action], stdin=f, capture=True).stdout


def push_package(remote_src: str) -> str:
    """Copy this tperf package, the harness included, to <remote_src>/tperf on the host; return the harness version.

    Workstation runs push to their own run directory (e.g. f"{rdir}/src") and put it first in the PYTHONPATH of every
    module they run there (`remote_module`): the daily run's copy (REMOTE_SRC) is only replaced by `tperf deploy`.
    The harness version (`tperf harness version`) is written to <remote_src>/tperf/harness/VERSION.
    """
    version = harness_version()
    ssh(["mkdir", "-p", f"{remote_src}/tperf"])
    subprocess.run(
        ["rsync", "-a", "--delete", "--exclude=__pycache__", "-e", " ".join(SSH), f"{PKG_SRC}/tperf/",
         f"{require_host()}:{remote_src}/tperf/"],
        check=True,
    )  # fmt: skip
    ssh(f"echo {shlex.quote(version)} >{shlex.quote(remote_src)}/tperf/harness/VERSION")
    return version


def remote_module(python: str, module: str, pythonpath: str) -> list[str]:
    """`env PYTHONPATH=<pythonpath> <python> -m tperf.<module>`: a host-side module of a pushed package.

    pythonpath starts with the directory the package was pushed to (push_package), then e.g. a transformers src/.
    """
    return ["env", f"PYTHONPATH={pythonpath}", python, "-m", f"tperf.{module}"]


def run_module(module: str, *args: object, **kwargs) -> subprocess.CompletedProcess:
    """Run `python -m tperf.<module> args...` locally, with this package importable whatever the install."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(PKG_SRC), env.get("PYTHONPATH", "")) if p)
    return subprocess.run([sys.executable, "-m", f"tperf.{module}", *map(str, args)], env=env, **kwargs)


def cpus_for(node: int, count: int) -> str:
    """First hardware thread of `count` cores on a NUMA node, skipping the node's first core (housekeeping)."""
    out = ssh_out(["lscpu", "-p=CPU,CORE,NODE"])
    seen, cpus = set(), []
    for line in out.splitlines():
        if line.startswith("#"):
            continue
        cpu, core, n = line.split(",")[:3]
        if int(n) == node and core not in seen:
            seen.add(core)
            cpus.append(cpu)
    if len(cpus) < count + 1:
        die(f"NUMA node {node} has {len(cpus)} cores, {count + 1} needed")
    return ",".join(cpus[1 : count + 1])


def stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def daily_lock_free() -> bool:
    """False while the daily run (or another GPU run) holds the lock."""
    return ssh(["flock", "-n", DAILY_LOCK, "true"], check=False).returncode == 0


def locked(args: Sequence[object]) -> list[object]:
    """Wrap a remote command so it runs under the daily lock, failing at once if the lock is taken."""
    return ["flock", "-n", "-E", "75", DAILY_LOCK, *args]


def harness_version() -> str:
    from tperf.common.harness import version

    return version()


def download_model(model: str) -> None:
    """Pre-download a model on the host (not timed)."""
    ssh([*env_args(HF_ENV), HF_CLI, "download", model], stdout=subprocess.DEVNULL)


class Worktree:
    """A detached worktree of the host's transformers clone, removed on exit."""

    def __init__(self, path: str, rev: str):
        self.path, self.rev = path, rev

    def __enter__(self) -> "Worktree":
        ssh(["git", "-C", TF, "worktree", "add", "-q", "--detach", self.path, self.rev])
        return self

    def __exit__(self, *exc) -> None:
        ssh(f"git -C {TF} worktree remove --force {self.path}; git -C {TF} worktree prune", check=False)


def fetch_main(ref: str = "refs/daily/main", rev: str | None = None) -> str:
    """Fetch transformers main into `ref` on the host; return the SHA of `rev` (default: that main)."""
    ssh(["git", "-C", TF, "fetch", "-q", REPO_URL, f"+main:{ref}"])
    return ssh_out(["git", "-C", TF, "rev-parse", rev or ref])
