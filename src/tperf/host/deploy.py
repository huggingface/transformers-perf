"""Install or update the tooling (the tperf package, the harness, the scenarios) and the daily main benchmark on the host.

Usage: tperf deploy            # copy tooling, install bench-daily.timer (03:00 UTC)
       tperf deploy --run-now  # same, then start a run immediately (in the background)

On the host: the tperf package in /opt/bench/src/tperf (run with PYTHONPATH=/opt/bench/src), the harness in
/opt/bench/harness, the scenarios in /opt/bench/daily/scenarios.
Logs: ssh $HOST journalctl -u bench-daily -e     Results: /opt/bench/daily/results, and the dataset.

Also here: `tperf host tune [apply|status|revert]` and `tperf host install`, which pipe the standalone
tperf/host/prepare.py and tperf/host/install.py to `sudo python3 -` on the host.
"""

import argparse
import os
import shlex
import subprocess

from tperf.common.benchlib import (
    DASHBOARD_SPACE,
    DATASET,
    HOST,
    PKG_SRC,
    ROOT,
    SSH,
    push_harness,
    push_package,
    require_host,
    scp_to,
    ssh,
)


# Options of tperf/host/prepare.py and tperf/host/install.py, passed on to the host when set here
TUNE_OPTIONS = ["KEEP_SSM", "KEEP_CSTATES", "NVIDIA_DRIVER", "CACHE_MOUNT", "GPU_NUMA_MAP", "GPU_SM_CLOCK"]
INSTALL_OPTIONS = ["TF_COMMIT"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-now", action="store_true", help="start a run right after deploying")
    args = parser.parse_args(argv)

    ssh("mkdir -p /opt/bench/daily/scenarios && rm -rf /opt/bench/daily/harness-patches")
    # The harness, staged then swapped in: a running daily job keeps the copy it started with
    version = push_harness("/opt/bench/harness-new")
    ssh("rm -rf /opt/bench/harness && mv /opt/bench/harness-new/harness /opt/bench/ && rmdir /opt/bench/harness-new")
    print(f"harness {version}")
    # The package (daily run, recording, noise probe, processors, dashboard), staged then swapped in the same way
    push_package()
    # Where the daily run publishes, from tperf.toml (read by bench-daily.service)
    env = f"PUBLISH_REPO={DATASET}\nDASHBOARD_SPACE={DASHBOARD_SPACE}\n"
    subprocess.run([*SSH, require_host(), "cat >/opt/bench/tperf.env"], input=env, text=True, check=True)
    scp_to(sorted((ROOT / "scenarios").glob("*.jsonl")), "/opt/bench/daily/scenarios/")
    units = PKG_SRC / "tperf/daily"
    scp_to([units / "bench-daily.service", units / "bench-daily.timer"], "/tmp/")
    ssh(
        "sudo install -m 644 /tmp/bench-daily.service /tmp/bench-daily.timer /etc/systemd/system/ "
        "&& rm /tmp/bench-daily.service /tmp/bench-daily.timer "
        "&& sudo systemctl daemon-reload && sudo systemctl enable --now bench-daily.timer >/dev/null "
        "&& systemctl list-timers bench-daily.timer --no-pager | head -2"
    )
    if args.run_now:
        ssh(["sudo", "systemctl", "start", "--no-block", "bench-daily.service"])
        print(f"started: ssh {HOST} journalctl -u bench-daily -f")


def pipe_as_root(script: str, options: list[str], args: list[str]) -> None:
    """Run a standalone stdlib-only file of this package on the host as root: `sudo env [K=V...] python3 - args`."""
    env = [f"{k}={os.environ[k]}" for k in options if k in os.environ]
    with open(PKG_SRC / "tperf/host" / script) as f:
        ssh(f"sudo {shlex.join(['env', *env, 'python3', '-', *args])}", stdin=f)


def tune(argv: list[str] | None = None) -> None:
    """tperf host tune [apply|status|revert]: tune the host for low noise (tperf/host/prepare.py)."""
    p = argparse.ArgumentParser(
        description=f"Run tperf/host/prepare.py on {HOST} as root. Passes on {', '.join(TUNE_OPTIONS)} when set."
    )
    p.add_argument("action", nargs="?", default="apply", choices=["apply", "status", "revert"])
    pipe_as_root("prepare.py", TUNE_OPTIONS, [p.parse_args(argv).action])


def install(argv: list[str] | None = None) -> None:
    """tperf host install: CUDA toolkit, nvbandwidth, torch and transformers under /opt/bench (tperf/host/install.py)."""
    p = argparse.ArgumentParser(
        description=f"Run tperf/host/install.py on {HOST} as root. Passes on {', '.join(INSTALL_OPTIONS)} when set."
    )
    p.parse_args(argv)
    pipe_as_root("install.py", INSTALL_OPTIONS, [])


if __name__ == "__main__":
    main()
