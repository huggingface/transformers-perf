"""Install or update the tooling (the tperf package, harness included, and the scenarios) and the daily main
benchmark on the host.

Usage: tperf deploy            # copy tooling, install bench-daily.timer (03:00 UTC)
       tperf deploy --run-now  # same, then start a run immediately (in the background)
       tperf deploy --bisect-worker [--box NAME]  # same, plus bench-bisect-worker.service (`tperf bisect worker`),
                                                  # which then also runs the daily benchmark (the timer is turned off)

On the host: the tperf package in /opt/bench/src/tperf (run with PYTHONPATH=/opt/bench/src), with the harness in
/opt/bench/src/tperf/harness (its version in VERSION there), the scenarios in /opt/bench/daily/scenarios. Only this
command writes /opt/bench/src: the other commands push the package to their own run directory.
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
    REMOTE_SRC,
    ROOT,
    SSH,
    push_package,
    require_host,
    scp_to,
    ssh,
    stamp,
)


# Options of tperf/host/prepare.py and tperf/host/install.py, passed on to the host when set here
TUNE_OPTIONS = ["KEEP_SSM", "KEEP_CSTATES", "NVIDIA_DRIVER", "CACHE_MOUNT", "GPU_NUMA_MAP", "GPU_SM_CLOCK"]
INSTALL_OPTIONS = ["TF_COMMIT"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-now", action="store_true", help="start a run right after deploying")
    parser.add_argument(
        "--bisect-worker",
        action="store_true",
        help="also run `tperf bisect worker` as a service (needs ~/.config/tperf/bisect-token on the host); it runs the "
        "daily benchmark too, at 03:00 UTC, through the dashboard's bench queue",
    )
    parser.add_argument(
        "--box", default="", help="this box's name in the dashboard's box list (for the worker's logs)"
    )
    args = parser.parse_args(argv)

    # /opt/bench/harness: the harness's old place, before it moved into the package
    ssh("mkdir -p /opt/bench/daily/scenarios && rm -rf /opt/bench/daily/harness-patches /opt/bench/harness")
    # The package (harness, daily run, recording, noise probe, processors, dashboard), staged then swapped in: a
    # process that already started keeps the modules it imported
    staging = f"{REMOTE_SRC}/.tperf-{stamp()}-{os.getpid()}"
    version = push_package(staging)
    ssh(f"rm -rf {REMOTE_SRC}/tperf && mv -T {staging}/tperf {REMOTE_SRC}/tperf && rmdir {staging}")
    print(f"harness {version}")
    # Where the daily run publishes, from tperf.toml (read by bench-daily.service)
    env = f"PUBLISH_REPO={DATASET}\nDASHBOARD_SPACE={DASHBOARD_SPACE}\n"
    subprocess.run([*SSH, require_host(), "cat >/opt/bench/tperf.env"], input=env, text=True, check=True)
    scp_to(sorted((ROOT / "scenarios").glob("*.jsonl")), "/opt/bench/daily/scenarios/")
    units = PKG_SRC / "tperf/daily"
    scp_to([units / "bench-daily.service", units / "bench-daily.timer"], "/tmp/")
    # Where the worker runs the daily benchmark, the timer stays off (a plain redeploy must not turn it back on)
    worker_owns_daily = (
        args.bisect_worker
        or ssh("systemctl is-enabled --quiet bench-bisect-worker.service", check=False).returncode == 0
    )
    timer = "disable --now" if worker_owns_daily else "enable --now"
    ssh(
        "sudo install -m 644 /tmp/bench-daily.service /tmp/bench-daily.timer /etc/systemd/system/ "
        "&& rm /tmp/bench-daily.service /tmp/bench-daily.timer "
        f"&& sudo systemctl daemon-reload && sudo systemctl {timer} bench-daily.timer >/dev/null 2>&1; "
        "systemctl list-timers bench-daily.timer --no-pager | head -2"
    )
    if worker_owns_daily:
        print("daily benchmark: run by bench-bisect-worker (bench-daily.timer off)")
    if args.bisect_worker:
        install_bisect_worker(args.box)
    if args.run_now:
        ssh(["sudo", "systemctl", "start", "--no-block", "bench-daily.service"])
        print(f"started: ssh {HOST} journalctl -u bench-daily -f")


def install_bisect_worker(box: str = "") -> None:
    """bench-bisect-worker.service, and the ssh key it uses to reach this host as HOST=ubuntu@localhost."""
    ssh("test -s ~/.config/tperf/bisect-token || { echo 'missing ~/.config/tperf/bisect-token' >&2; exit 1; }")
    ssh(
        "mkdir -p /opt/bench/bisect && chmod 700 ~/.ssh "
        "&& { test -f ~/.ssh/id_bisect || ssh-keygen -q -t ed25519 -N '' -C bisect-worker -f ~/.ssh/id_bisect; } "
        '&& { grep -qF "$(cat ~/.ssh/id_bisect.pub)" ~/.ssh/authorized_keys '
        "|| cat ~/.ssh/id_bisect.pub >> ~/.ssh/authorized_keys; } "
        "&& { grep -q '^Host localhost' ~/.ssh/config 2>/dev/null "
        "|| printf 'Host localhost\\n  IdentityFile ~/.ssh/id_bisect\\n  StrictHostKeyChecking accept-new\\n' >> ~/.ssh/config; } "
        "&& chmod 600 ~/.ssh/config && ssh -o BatchMode=yes localhost true"
    )
    scp_to([PKG_SRC / "tperf/bisect/bench-bisect-worker.service"], "/tmp/")
    # this box's settings, beside the packaged unit: its name, and the daily run's time
    dropin = f"[Service]\nEnvironment=DAILY_AT_UTC=03:00\nEnvironment=BISECT_BOX={box or 'default'}\n"
    subprocess.run(
        [*SSH, require_host(), ("sudo mkdir -p /etc/systemd/system/bench-bisect-worker.service.d "
                                "&& sudo tee /etc/systemd/system/bench-bisect-worker.service.d/box.conf >/dev/null")],
        input=dropin, text=True, check=True,
    )  # fmt: skip
    ssh(
        "sudo install -m 644 /tmp/bench-bisect-worker.service /etc/systemd/system/ "
        "&& rm /tmp/bench-bisect-worker.service "
        "&& sudo systemctl daemon-reload && sudo systemctl enable bench-bisect-worker.service >/dev/null "
        "&& sudo systemctl restart bench-bisect-worker.service "
        "&& systemctl is-active bench-bisect-worker.service"
    )
    print(f"bisect worker: ssh {HOST} journalctl -u bench-bisect-worker -f")


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
