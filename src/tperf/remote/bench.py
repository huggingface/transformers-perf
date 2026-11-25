"""Benchmark a transformers PR against main on the bench host, from anywhere: tperf bench <PR>.

Usage: tperf bench <PR> [--no-wait]

Queues a PR bench on the dashboard (sign in first: tperf auth login). The bench host's worker runs `tperf ab` of the
PR's head against its merge-base with main on the daily scenarios, one at a time; this command prints each scenario's
result as it lands and the results page when the bench ends. --no-wait prints the page and returns at once.
The run is posted in #transformers-benchmark too.
"""

import argparse
import sys
import time

from tperf.remote.auth import DashboardError, access_token, dashboard, request

POLL = 15


def line(step: dict) -> str:
    where = f"{step['model']} · {step['scenario']}"
    if step.get("change") is None:
        return f"  {where}: {step['verdict']}"
    lo, hi = step["ci"]
    return f"  {where}: {step['change']:+.2f}% [{lo:+.2f}, {hi:+.2f}] → {step['verdict']}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pr", type=int, help="transformers PR number")
    parser.add_argument("--no-wait", action="store_true", help="queue it and print the results page")
    args = parser.parse_args(argv)
    try:
        token = access_token()
        job = request("/benchmarks/pr-bench", body={"pr": args.pr}, token=token)
    except DashboardError as e:
        hints = {"pr_already_queued": "this PR already has a bench queued or running",
                 "bisect_queue_full": "the bench queue is full (3 jobs): try again later",
                 "pr_not_open": "the PR is not open", "pr_not_found": "no such PR in huggingface/transformers",
                 "unauthorized": "the session expired: run tperf auth login",
                 "account_not_allowed": "PR benches are for Hugging Face organization members"}  # fmt: skip
        print(f"tperf bench: {hints.get(e.error, e.error)}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"tperf bench: {e}", file=sys.stderr)
        return 1
    url = dashboard() + job["url"]
    print(f"Queued PR #{args.pr}: {url}", flush=True)
    if args.no_wait:
        return 0
    seen, state = 0, ""
    while True:
        time.sleep(POLL)
        try:
            data = request(job["url"] + "?format=json", token=token)
        except DashboardError as e:
            print(f"(poll failed: {e.error}; retrying)", file=sys.stderr)
            continue
        if data["state"] != state:
            state = data["state"]
            if state == "running":
                print(f"Running on the bench host, {len(data['scenarios'])} scenarios:", flush=True)
        for step in data["steps"][seen:]:
            print(line(step), flush=True)
        seen = len(data["steps"])
        if state == "done":
            result = data.get("result") or {}
            ok = result.get("state") == "done"
            print(f"{'Done' if ok else 'Failed: ' + str(result.get('reason', ''))}. Results: {url}")
            return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
