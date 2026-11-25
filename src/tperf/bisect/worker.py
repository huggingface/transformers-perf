"""Run bisect jobs queued from the Transformers CI benchmarks page, one at a time.

Usage: BISECT_TOKEN_FILE=<file> tperf bisect worker [--once]

The dashboard never reaches the bench host: this worker polls it. Each loop:

  GET  <API>/benchmarks/bisect/next          204: nothing queued; 200: a job, now claimed by this worker
                                             {"id", "model", "scenario": {<one harness config>}, "bench_args",
                                              "good", "bad", "metric", "rounds", "min_change", "expected"}
  POST <API>/benchmarks/bisect/<id>/report   {"kind": "step", "step": {...}} after every A/B,
                                             {"kind": "done", "result": {<result.json>}} at the end,
                                             {"kind": "done", "result": {"state": "failed", "reason": ...}} on error

A job is the same thing as `tperf bisect` run by hand (tperf.bisect.run.run_job), so it reaches the bench host the
same way: over ssh to HOST. On the bench host itself, set HOST=<user>@localhost. Every A/B takes the daily run's lock
for its runs; the worker also claims nothing while the lock is held or the daily run is due (QUIET_UTC), so a job does
not start just before 03:00 and lose its steps to the lock.

Environment:
  BISECT_API          dashboard base URL (default: https://transformers-ci.huggingface.cool)
  BISECT_TOKEN_FILE   file holding the worker's service token (or BISECT_TOKEN)
  POLL                seconds between polls when idle (default: 60)
  QUIET_UTC           UTC hours in which no job is claimed, "<from>-<to>" (default: 1-5, around the 03:00 daily run)
  GPUS                the two GPUs of each A/B (default "1 5": not the daily suites' GPUs 0 and 4)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import asdict
from datetime import UTC, datetime

from tperf.common.benchlib import daily_lock_free, die, log

API = os.environ.get("BISECT_API", "https://transformers-ci.huggingface.cool").rstrip("/")
POLL = int(os.environ.get("POLL") or 60)


def token() -> str:
    if path := os.environ.get("BISECT_TOKEN_FILE"):
        with open(path) as f:
            return f.read().strip()
    return os.environ.get("BISECT_TOKEN") or die("set BISECT_TOKEN_FILE (or BISECT_TOKEN) to the worker's token")


def call(method: str, path: str, body: dict | None = None) -> tuple[int, dict | None]:
    req = urllib.request.Request(
        f"{API}{path}",
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Authorization": f"Bearer {token()}",
            "Content-Type": "application/json",
            "User-Agent": "tperf-bisect",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
            return r.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        return e.code, None


def quiet_now(spec: str | None = None, now: datetime | None = None) -> bool:
    """True inside the QUIET_UTC window ("1-5": from 01:00 to 04:59 UTC; "22-2" wraps midnight)."""
    start, _, end = (spec or os.environ.get("QUIET_UTC") or "1-5").partition("-")
    hour = (now or datetime.now(UTC)).hour
    lo, hi = int(start), int(end or start)
    return lo <= hour < hi if lo <= hi else hour >= lo or hour < hi


def report(job_id: str, body: dict) -> None:
    for attempt in range(5):  # a dashboard redeploy should not lose a step
        try:
            status, _ = call("POST", f"/benchmarks/bisect/{job_id}/report", body)
            if status < 500:
                if status >= 300:
                    log(f"report for {job_id} answered {status}")
                return
        except OSError as e:
            log(f"report for {job_id} failed: {e}")
        time.sleep(10 * (attempt + 1))


def verdict(change: float, ci: tuple[float, float]) -> str:
    """Head vs base, positive = slower (the metrics are latencies)."""
    if ci[0] > 0:
        return "slower"
    if ci[1] < 0:
        return "faster"
    return "no significant change"


def run_pr(job: dict) -> None:
    """A PR bench: `tperf ab` of the PR head against its merge-base with main, one step per daily scenario."""
    from tperf.bisect.run import run_ab
    from tperf.common.benchlib import REPO_URL, TF, ssh, ssh_out

    job_id, pr, head = str(job["id"]), int(job["pr"]), str(job["head"])
    log(f"Job {job_id}: PR #{pr} at {head[:10]}, {len(job['scenarios'])} scenarios")
    git = ["git", "-C", TF]
    ssh([*git, "fetch", "-q", REPO_URL, "+main:refs/ab/main", f"+refs/pull/{pr}/head:refs/ab/pr-{pr}"])
    if ssh_out([*git, "rev-parse", f"refs/ab/pr-{pr}"]) != head:
        log(f"PR #{pr} moved since it was queued; benching the queued head {head[:10]}")
    base = ssh_out([*git, "merge-base", "refs/ab/main", head])
    measured = 0
    for sc in job["scenarios"]:
        step = {"model": sc["model"], "scenario": sc["label"], "base": base, "head": head}
        with scenario_env(sc):
            m = run_ab(base, head, int(job.get("rounds") or 3), job.get("metric") or "e2e")
        if m is None:
            step.update(change=None, ci=None, verdict="failed to run")
        else:
            measured += 1
            step.update(change=round(m.change, 2), ci=[round(c, 2) for c in m.ci], verdict=verdict(m.change, m.ci),
                        base_ms=m.base, head_ms=m.head, ab_dir=m.ab_dir)  # fmt: skip
        report(job_id, {"kind": "step", "step": step})
    if measured:
        report(job_id, {"kind": "done", "result": {"state": "done", "base": base}})
    else:
        report(
            job_id, {"kind": "done", "result": {"state": "failed", "reason": "no scenario ran (see the worker log)"}}
        )


class scenario_env:
    """MODEL, SCENARIOS (a one-line jsonl) and BENCH_ARGS for one scenario, restored afterwards."""

    KEYS = ("MODEL", "SCENARIOS", "BENCH_ARGS")

    def __init__(self, sc: dict):
        self.sc = sc

    def __enter__(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps(self.sc["scenario"]) + "\n")
            self.path = f.name
        self.saved = {k: os.environ.get(k) for k in self.KEYS}
        os.environ.update(MODEL=self.sc["model"], SCENARIOS=self.path)
        if self.sc.get("bench_args"):
            os.environ["BENCH_ARGS"] = self.sc["bench_args"]
        else:
            os.environ.pop("BENCH_ARGS", None)

    def __exit__(self, *exc):
        os.unlink(self.path)
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run(job: dict) -> None:
    if job.get("kind") == "pr":
        try:
            run_pr(job)
        except (Exception, SystemExit) as e:  # noqa: BLE001 -- die() exits; a job must still end
            reason = str(e) if isinstance(e, SystemExit) else f"{type(e).__name__}: {e}"
            traceback.print_exc()
            report(str(job["id"]), {"kind": "done", "result": {"state": "failed", "reason": reason or "failed"}})
        return
    from tperf.bisect.run import run_job

    job_id = str(job["id"])
    log(f"Job {job_id}: {job['model']} {job['good'][:10]}..{job['bad'][:10]}")
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(job["scenario"]) + "\n")
        scenarios = f.name
    saved = {k: os.environ.get(k) for k in ("MODEL", "SCENARIOS", "BENCH_ARGS")}
    os.environ.update(MODEL=job["model"], SCENARIOS=scenarios)
    if job.get("bench_args"):
        os.environ["BENCH_ARGS"] = job["bench_args"]
    try:
        data, _ = run_job(
            job["good"],
            job["bad"],
            job.get("metric") or "e2e",
            int(job.get("rounds") or 4),
            float(job.get("min_change") or 1.0),
            float(job["expected"]) if job.get("expected") is not None else None,
            lambda step: report(job_id, {"kind": "step", "step": asdict(step)}),
        )
        report(job_id, {"kind": "done", "result": data})
    except (Exception, SystemExit) as e:  # noqa: BLE001 -- die() exits; a job must still end
        reason = str(e) if isinstance(e, SystemExit) else f"{type(e).__name__}: {e}"
        traceback.print_exc()
        report(job_id, {"kind": "done", "result": {"state": "failed", "reason": reason or "the worker stopped"}})
    finally:
        os.unlink(scenarios)
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def poll_once() -> bool:
    """Claim and run one job; False if there was none (or it was not the time to take one)."""
    if quiet_now() or not daily_lock_free():
        return False
    status, job = call("GET", "/benchmarks/bisect/next")
    if status == 204:
        return False
    if status != 200 or not job:
        log(f"GET /benchmarks/bisect/next answered {status}")
        return False
    run(job)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", action="store_true", help="poll once, run the job if there is one, and exit")
    args = parser.parse_args(argv)
    token()  # fail now, not at the first job
    while True:
        try:
            ran = poll_once()
        except OSError as e:
            log(f"poll failed: {e}")
            ran = False
        if args.once:
            return 0
        if ran and os.environ.get("INVOCATION_ID"):
            # under systemd (Restart=always): exit after a job, so the next one runs the package as deployed now
            return 0
        if not ran:
            time.sleep(POLL)


if __name__ == "__main__":
    sys.exit(main())
