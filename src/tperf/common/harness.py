"""Copy this repository's harness to the bench host, and report its version.

The harness copy in this repository drives every run. The transformers under test only comes from PYTHONPATH (or
the editable install in /opt/bench/tbv2-venv), so one harness measures any commit the same way.

Usage:
    tperf harness version              # last commit that changed harness/, -dirty with local changes
    tperf harness push HOST REMOTE_DIR # copy harness/ (without tests) to REMOTE_DIR/harness on HOST
"""

import argparse
import subprocess

from tperf.common.benchlib import ROOT, SSH


def version() -> str:
    git = ["git", "-C", str(ROOT)]
    commit = subprocess.run(
        [*git, "log", "-1", "--format=%h", "--", "harness"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run([*git, "diff", "--quiet", "HEAD", "--", "harness"]).returncode != 0
    return commit + ("-dirty" if dirty else "")


def push(host: str, remote_dir: str) -> str:
    v = version()
    subprocess.run([*SSH, host, f"mkdir -p {remote_dir}"], check=True)
    subprocess.run(
        ["rsync", "-a", "--delete", "--exclude=__pycache__", "--exclude=tests", "--exclude=benchmark_results",
         "-e", " ".join(SSH), f"{ROOT}/harness/", f"{host}:{remote_dir}/harness/"],
        check=True,
    )  # fmt: skip
    subprocess.run([*SSH, host, f"echo {v} >{remote_dir}/harness/VERSION"], check=True)
    return v


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("version")
    ps = sub.add_parser("push")
    ps.add_argument("host")
    ps.add_argument("remote_dir")
    a = p.parse_args(argv)
    print(version() if a.cmd == "version" else push(a.host, a.remote_dir))


if __name__ == "__main__":
    main()
