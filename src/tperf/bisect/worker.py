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
for its runs; the worker claims nothing while the lock is held.

The box's daily benchmark goes through the worker too when DAILY_AT_UTC is set (`tperf deploy --bisect-worker` sets it
and turns bench-daily.timer off): once that time has passed on a day it has not run yet, the worker runs
tperf.daily.main before claiming anything else (after the job in progress, if any), registers it on the dashboard
(POST <API>/benchmarks/bisect/daily) and reports each of its stages. The daily run does not depend on the dashboard:
when it cannot be reached, the run happens unreported.

Environment:
  BISECT_API          dashboard base URL (default: https://transformers-ci.huggingface.cool)
  BISECT_TOKEN_FILE   file holding the worker's service token (or BISECT_TOKEN)
  POLL                seconds between polls when idle (default: 60)
  QUIET_UTC           UTC hours in which no job is claimed, "<from>-<to>" (default: 1-5 when the timer runs the daily
                      benchmark, none when this worker does)
  DAILY_AT_UTC        "HH:MM": run the box's daily benchmark from this time each day (unset: bench-daily.timer does)
  DAILY_STATE         file holding the date of the last daily run started here (default: daily-last)
  BISECT_BOX          this box's name, for the logs (the dashboard knows the box from the token)
  GPUS                the two GPUs of each A/B (default "1 5": not the daily suites' GPUs 0 and 4)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
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
    if spec is None:
        spec = os.environ.get("QUIET_UTC", "" if os.environ.get("DAILY_AT_UTC") else "1-5")
    if not spec:
        return False
    start, _, end = spec.partition("-")
    hour = (now or datetime.now(UTC)).hour
    lo, hi = int(start), int(end or start)
    return lo <= hour < hi if lo <= hi else hour >= lo or hour < hi


def daily_due(now: datetime | None = None) -> bool:
    """The daily benchmark is due: DAILY_AT_UTC has passed today and it has not started today. Without a state file
    (a fresh install) today counts as done, so installing the worker mid-day does not start a second daily run."""
    at = os.environ.get("DAILY_AT_UTC", "")
    if not at:
        return False
    now = now or datetime.now(UTC)
    state = os.environ.get("DAILY_STATE") or "daily-last"
    try:
        with open(state) as f:
            last = f.read().strip()
    except FileNotFoundError:
        with open(state, "w") as f:
            f.write(now.date().isoformat())
        return False
    hour, _, minute = at.partition(":")
    return now.date().isoformat() != last and (now.hour, now.minute) >= (int(hour), int(minute or 0))


def run_daily() -> int:
    """Run tperf.daily.main here, registered on the dashboard as a job of kind "daily" with one step per stage."""
    with open(os.environ.get("DAILY_STATE") or "daily-last", "w") as f:
        f.write(datetime.now(UTC).date().isoformat())  # first: a crash must not start it again in a loop
    log("Daily benchmark")
    proc = subprocess.Popen(
        [sys.executable, "-m", "tperf.daily.main"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    job_id, commit = None, ""
    for line in proc.stdout:
        print(line, end="", flush=True)
        if found := re.match(r"stages (\d+)\s*$", line):
            try:
                status, body = call("POST", "/benchmarks/bisect/daily", {"total": int(found[1])})
                job_id = str(body["id"]) if status == 201 and body else None
            except OSError as e:
                log(f"could not register the daily run: {e}")
        elif found := re.match(r"stage (ok|failed): (.+?)\s*$", line):
            if job_id:
                report(job_id, {"kind": "step", "step": {"label": found[2], "verdict": found[1]}})
        elif found := re.search(r"\bmain ([0-9a-f]{10}): ", line):
            commit = found[1]
    rc = proc.wait()
    if job_id:
        result = {"state": "done", "commit": commit} if rc == 0 else {
            "state": "failed", "commit": commit, "reason": f"tperf.daily.main exited {rc} (see journalctl on the box)"}  # fmt: skip
        report(job_id, {"kind": "done", "result": result})
    return rc


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


def ab_pr(pr: int, rounds: int) -> str:
    """`tperf ab <pr> <rounds>` with the environment as set (MODEL, SCENARIOS, BENCH_ARGS, GPUS, PUBLISH_REPO), output
    streamed; the A/B's results directory, or "" if it did not start."""
    import re as _re

    from tperf.common.benchlib import PKG_SRC

    env = dict(os.environ, PROFILE="0")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(PKG_SRC), env.get("PYTHONPATH", "")) if p)
    for k in ("BASE", "HEAD"):  # the PR's merge base and head, as tperf ab picks them
        env.pop(k, None)
    proc = subprocess.Popen(
        [sys.executable, "-m", "tperf.ab.run", str(pr), str(rounds)], env=env, stdout=subprocess.PIPE, text=True
    )
    out_dir = ""
    for line in proc.stdout:
        print(line, end="", flush=True)
        if found := _re.search(r"results in (\S+)\s*$", line):
            out_dir = found[1]
    proc.wait()  # exits 1 on a REGRESSION verdict too: the rows decide
    return out_dir


def run_pr(job: dict) -> None:
    """A PR bench: `tperf ab <pr>` (head against its merge base with main, GPUs swapped each round), once per model
    with its daily scenarios, published to the dataset like any `tperf ab`, so /benchmarks/pr-<n> shows the full
    report. One step per scenario: its e2e change, CI and `tperf ab compare`'s verdict, the same as that page."""
    from pathlib import Path

    from tperf.ab.compare import analyze
    from tperf.common.record import scenario_rows

    job_id, pr = str(job["id"]), int(job["pr"])
    metric, rounds = job.get("metric") or "e2e", int(job.get("rounds") or 6)
    log(f"Job {job_id}: PR #{pr}, {len(job['scenarios'])} scenarios")
    groups: dict[tuple[str, str], list[dict]] = {}
    for sc in job["scenarios"]:
        groups.setdefault((sc["model"], sc.get("bench_args") or ""), []).append(sc)
    measured, ab_ids, commits = 0, [], {}
    for (model, bench_args), scs in groups.items():
        with scenario_env({"model": model, "bench_args": bench_args, "scenarios": [sc["scenario"] for sc in scs]}):
            out_dir = ab_pr(pr, rounds)
        rows = [r for d in sorted(Path(out_dir).glob("round*-*")) if (d / "manifest.json").exists()
                for r in scenario_rows(d)] if out_dir else []  # fmt: skip
        if out_dir:
            ab_ids.append(Path(out_dir).name.removeprefix("ab-"))
        entries = analyze(rows)["scenarios"] if rows else []
        for sc in scs:
            step = {"model": model, "scenario": sc["label"]}
            entry = next((e for e in entries if e["pairs"] and e["pairs"][0][1]["base"].get("attn_implementation")
                          == sc["scenario"].get("attn_implementation")), None)  # fmt: skip
            m = next((m for m in entry["metrics"] if m["name"] == metric), None) if entry else None
            if m is None:
                step.update(change=None, ci=None, verdict="failed to run")
            else:
                measured += 1
                base_row, head_row = entry["pairs"][0][1]["base"], entry["pairs"][0][1]["head"]
                commits = {"base": base_row.get("commit_id"), "head": head_row.get("commit_id")}
                step.update(change=round(m["change_pct"], 2), ci=[round(c, 2) for c in m["ci"]],
                            verdict=m["verdict"], base_ms=m["base"], head_ms=m["head"], rounds=entry["rounds"],
                            **commits)  # fmt: skip
            report(job_id, {"kind": "step", "step": step})
    result = {"state": "done" if measured else "failed", "page": f"/benchmarks/pr-{pr}", "ab_ids": ab_ids, **commits}
    if not measured:
        result["reason"] = "no scenario ran (see the worker log)"
    report(job_id, {"kind": "done", "result": result})


class scenario_env:
    """MODEL, SCENARIOS (a one-line jsonl) and BENCH_ARGS for one scenario, restored afterwards."""

    KEYS = ("MODEL", "SCENARIOS", "BENCH_ARGS")

    def __init__(self, sc: dict):
        self.sc = sc

    def __enter__(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for config in self.sc.get("scenarios") or [self.sc["scenario"]]:
                f.write(json.dumps(config) + "\n")
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
    os.environ.setdefault("GPUS", "1 5")  # every job: not the daily suites' GPUs 0 and 4
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
            ran = daily_due() and run_daily() is not None
            ran = ran or poll_once()
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
