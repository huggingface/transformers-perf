"""Find the commit of main that changed one scenario's performance, A/B by A/B.

Usage: MODEL=<model> SCENARIOS=<one-scenario .jsonl> tperf bisect <good sha> <bad sha>

Each step is a `tperf ab` of two commits of main on the bench host (same harness, same host, same GPUs for both
sides), so a step measures transformers and nothing else:

  1. reproduce   ab good bad. The change must exclude 0 and reach MIN_CHANGE (%) to bisect. Otherwise:
                   not_reproduced  the CI's high end is below EXPECTED (the change seen on the daily chart, or
                                   MIN_CHANGE): what moved the chart was not transformers (host, driver, torch, the
                                   harness itself), and a bisect would blame a random commit
                   too_noisy       the CI covers both 0 and EXPECTED: more ROUNDS, or more iterations in BENCH_ARGS
  2. bisect      over `git rev-list --first-parent good..bad` (the commits as they landed on main). Each step is
                 ab good mid, read against the reproduced change d:
                   bad      change >= 75% of d and its CI's low end above 50% of d
                   good     change <= 25% of d and its CI's high end below 50% of d
                   unclear  anything else: stop and report the narrowed range (often two changes, or one too small
                            to split at this many rounds)
                 A commit that cannot run the scenario (the A/B fails) is skipped, as `git bisect skip`.
  3. confirm     ab culprit^ culprit: the interval reported is this one, not the bisect's.

Environment (the A/B's own variables are passed through: HOST, GPUS, NCORES, BENCH_ARGS, HARNESS_PATCHES):
  MODEL         model id (required)
  SCENARIOS     harness config file with ONE scenario (required): the bisect follows one series
  METRIC        compare.METRICS name to bisect on (default: e2e)
  ROUNDS        rounds per A/B (default: 4)
  MIN_CHANGE    smallest reproduced change worth bisecting, % (default: 1.0)
  EXPECTED      the change seen on the daily chart, % (default: MIN_CHANGE): what the reproduce step must rule out
                to answer not_reproduced
  GPUS          default "1 5": not the daily suites' GPUs 0 and 4

Results: results/bisect-<id>/, one ab-* directory per step (written by tperf ab under results/) and result.json:
  {"state": "culprit" | "range" | "not_reproduced" | "too_noisy" | "failed", "good", "bad", "commits", "steps": [...],
   "culprit": {...} | null, "range": [first possibly bad, last possibly bad] | null, "reason"}
Exit status: 0 when a culprit was found, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tperf.common.benchlib import PKG_SRC, REPO_URL, ROOT, TF, die, log, ssh, ssh_out, stamp

BAD_AT, BAD_FLOOR = 0.75, 0.5  # bad: change >= 75% of d, CI low end > 50% of d
GOOD_AT, GOOD_CEIL = 0.25, 0.5  # good: change <= 25% of d, CI high end < 50% of d


@dataclass
class Measurement:
    """One A/B's result for the bisected metric: % change of head vs base, positive = head worse."""

    change: float
    ci: tuple[float, float]
    base: float
    head: float
    ab_dir: str = ""


@dataclass
class Step:
    role: str  # reproduce | step | confirm
    base: str
    head: str
    verdict: str  # reproduced | not_reproduced | bad | good | unclear | skipped | confirmed | failed
    change: float | None = None
    ci: tuple[float, float] | None = None
    ab_dir: str = ""


@dataclass
class Result:
    state: str
    good: str
    bad: str
    commits: int
    steps: list[Step] = field(default_factory=list)
    culprit: dict | None = None
    range: list[str] | None = None
    reason: str = ""


def classify(m: Measurement, d: float) -> str:
    """A step's verdict against the reproduced change d (> 0)."""
    lo, hi = m.ci
    if m.change >= BAD_AT * d and lo > BAD_FLOOR * d:
        return "bad"
    if m.change <= GOOD_AT * d and hi < GOOD_CEIL * d:
        return "good"
    return "unclear"


def reproduced(m: Measurement, min_change: float) -> bool:
    """The full range moved, in the regression direction, by at least min_change %, and the CI excludes 0."""
    return m.ci[0] > 0 and m.change >= min_change


def bisect(
    commits: list[str],
    good: str,
    measure: Callable[[str, str], Measurement | None],
    min_change: float,
    parent: Callable[[str], str],
    expected: float | None = None,
) -> Result:
    """commits: good..bad, oldest first (the last one is bad). measure(base, head) runs one A/B, None if it failed."""
    bad = commits[-1]
    result = Result(state="failed", good=good, bad=bad, commits=len(commits))

    full = measure(good, bad)
    if full is None:
        result.steps.append(Step("reproduce", good, bad, "failed"))
        result.reason = "the A/B of good and bad failed"
        return result
    if not reproduced(full, min_change):
        expected = expected or min_change
        state = "not_reproduced" if full.ci[1] < expected else "too_noisy"
        result.steps.append(Step("reproduce", good, bad, state, full.change, full.ci, full.ab_dir))
        result.state = state
        measured = f"good vs bad: {full.change:+.2f}% [{full.ci[0]:+.2f}, {full.ci[1]:+.2f}]"
        result.reason = (
            f"{measured} rules out {expected:+.2f}%: the change is not in transformers between these commits"
            if state == "not_reproduced"
            else f"{measured} covers both 0 and {expected:+.2f}%: raise ROUNDS or the iterations in BENCH_ARGS"
        )
        return result
    d = full.change
    result.steps.append(Step("reproduce", good, bad, "reproduced", full.change, full.ci, full.ab_dir))

    # Invariant: commits[lo] is good (lo = -1: `good` itself), commits[hi] is bad
    lo, hi = -1, len(commits) - 1
    skipped: set[int] = set()
    while hi - lo > 1:
        candidates = [i for i in range(lo + 1, hi) if i not in skipped]
        if not candidates:
            break
        middle = (lo + hi) / 2
        mid = min(candidates, key=lambda i: (abs(i - middle), i))
        m = measure(good, commits[mid])
        if m is None:
            skipped.add(mid)
            result.steps.append(Step("step", good, commits[mid], "skipped"))
            continue
        verdict = classify(m, d)
        result.steps.append(Step("step", good, commits[mid], verdict, m.change, m.ci, m.ab_dir))
        if verdict == "bad":
            hi = mid
        elif verdict == "good":
            lo = mid
        else:
            result.state = "range"
            result.range = [commits[lo + 1], commits[hi]]
            result.reason = (
                f"{commits[mid][:10]} carries {100 * m.change / d:.0f}% of the change: two changes in the range, or "
                f"one too small to split at this many rounds"
            )
            return result

    if hi - lo > 1:  # every commit in between was skipped
        result.state = "range"
        result.range = [commits[lo + 1], commits[hi]]
        result.reason = "the commits in between could not run the scenario"
        return result

    culprit = commits[hi]
    before = parent(culprit)
    c = measure(before, culprit)
    if c is None:
        result.steps.append(Step("confirm", before, culprit, "failed"))
        result.state = "range"
        result.range = [culprit, culprit]
        result.reason = "the confirming A/B failed"
        return result
    ok = classify(c, d) == "bad"
    result.steps.append(Step("confirm", before, culprit, "confirmed" if ok else "unclear", c.change, c.ci, c.ab_dir))
    if not ok:
        result.state = "range"
        result.range = [culprit, culprit]
        result.reason = f"{culprit[:10]} against its parent: {c.change:+.2f}%, not the {d:+.2f}% of the full range"
        return result
    result.state = "culprit"
    result.culprit = {"sha": culprit, "parent": before, "change": c.change, "ci": list(c.ci), "base": c.base,
                      "head": c.head}  # fmt: skip
    return result


def run_ab(base: str, head: str, rounds: int, metric: str) -> Measurement | None:
    """`tperf ab main` with BASE/HEAD, output streamed; None if it failed or the scenario did not run."""
    from tperf.ab.compare import analyze
    from tperf.common.record import scenario_rows

    env = dict(os.environ, BASE=base, HEAD=head, PROFILE="0", PUBLISH_REPO="")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(PKG_SRC), env.get("PYTHONPATH", "")) if p)
    proc = subprocess.Popen(
        [sys.executable, "-m", "tperf.ab.run", "main", str(rounds)], env=env, stdout=subprocess.PIPE, text=True
    )
    out_dir = ""
    for line in proc.stdout:
        print(line, end="", flush=True)
        if found := re.search(r"results in (\S+)\s*$", line):
            out_dir = found[1]
    rc = proc.wait()
    # tperf ab exits 1 on a REGRESSION verdict too: the rows decide, not the exit status
    if not out_dir:
        print(f"warning: tperf ab exited {rc} before starting", file=sys.stderr)
        return None
    rows = [
        r for d in sorted(Path(out_dir).glob("round*-*")) if (d / "manifest.json").exists() for r in scenario_rows(d)
    ]
    entries = analyze(rows)["scenarios"]
    metrics = [m for e in entries for m in e["metrics"] if m["name"] == metric]
    if len(metrics) != 1:
        print(f"warning: {len(metrics)} results for {metric} in {out_dir} (exit {rc})", file=sys.stderr)
        return None
    m = metrics[0]
    return Measurement(m["change_pct"], tuple(m["ci"]), m["base"], m["head"], out_dir)


def first_parent_commits(good: str, bad: str) -> list[str]:
    git = ["git", "-C", TF]
    ssh([*git, "fetch", "-q", REPO_URL, "+main:refs/ab/main"])
    good, bad = (ssh_out([*git, "rev-parse", "--verify", f"{c}^{{commit}}"]) for c in (good, bad))
    if ssh([*git, "merge-base", "--is-ancestor", good, bad], check=False).returncode:
        die(f"{good[:10]} is not an ancestor of {bad[:10]}")
    return ssh_out([*git, "rev-list", "--first-parent", "--reverse", f"{good}..{bad}"]).split()


def subject(sha: str) -> str:
    return ssh_out(["git", "-C", TF, "log", "-1", "--format=%s", sha])


def summary(r: Result, metric: str) -> str:
    lines = [f"bisect {r.good[:10]}..{r.bad[:10]} ({r.commits} commits), {metric}: {r.state}"]
    for s in r.steps:
        where = f"{s.base[:10]}..{s.head[:10]}"
        if s.change is None:
            lines.append(f"  {s.role:<9} {where}  {s.verdict}")
        else:
            lines.append(f"  {s.role:<9} {where}  {s.change:+6.2f}% [{s.ci[0]:+6.2f}, {s.ci[1]:+6.2f}]  {s.verdict}")
    if r.culprit:
        c = r.culprit
        lines.append(f"culprit {c['sha'][:10]} {c.get('subject', '')}")
        lines.append(f"  vs parent: {c['change']:+.2f}% [{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}]")
    if r.range:
        lines.append(f"range {r.range[0][:10]}..{r.range[1][:10]} (inclusive)")
    if r.reason:
        lines.append(r.reason)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("good", help="a commit of main with the old performance")
    parser.add_argument("bad", help="a later commit of main with the new performance")
    args = parser.parse_args(argv)

    from tperf.ab.compare import METRICS

    os.environ.get("MODEL") or die("set MODEL to a Hugging Face model id")
    scenarios = os.environ.get("SCENARIOS") or die("set SCENARIOS to a harness config file with one scenario")
    if sum(1 for line in Path(scenarios).read_text().splitlines() if line.strip() and not line.startswith("#")) != 1:
        die(f"{scenarios} must hold exactly one scenario: a bisect follows one series")
    metric = os.environ.get("METRIC", "e2e")
    if metric not in METRICS:
        die(f"METRIC must be one of {', '.join(METRICS)}")
    rounds = int(os.environ.get("ROUNDS", "4"))
    min_change = float(os.environ.get("MIN_CHANGE", "1.0"))
    expected = float(os.environ["EXPECTED"]) if os.environ.get("EXPECTED") else None
    os.environ.setdefault("GPUS", "1 5")

    commits = first_parent_commits(args.good, args.bad)
    if not commits:
        die("nothing to bisect: bad is good")
    good = ssh_out(["git", "-C", TF, "rev-parse", f"{args.good}^{{commit}}"])
    out = ROOT / "results" / f"bisect-{stamp()}-{commits[-1][:10]}"
    out.mkdir(parents=True, exist_ok=True)
    log(f"Bisect {good[:10]}..{commits[-1][:10]}: {len(commits)} commits, {metric}, {rounds} rounds per A/B, in {out}")

    result = bisect(
        commits,
        good,
        lambda base, head: run_ab(base, head, rounds, metric),
        min_change,
        lambda sha: ssh_out(["git", "-C", TF, "rev-parse", f"{sha}^1"]),
        expected,
    )
    if result.culprit:
        result.culprit["subject"] = subject(result.culprit["sha"])
        if pr := re.search(r"\(#(\d+)\)\s*$", result.culprit["subject"]):
            result.culprit["pr"] = int(pr[1])
    data = asdict(result) | {"metric": metric, "model": os.environ["MODEL"], "rounds": rounds}
    (out / "result.json").write_text(json.dumps(data, indent=2) + "\n")
    text = summary(result, metric)
    (out / "summary.txt").write_text(text + "\n")
    print(text)
    return 0 if result.state == "culprit" else 1


if __name__ == "__main__":
    sys.exit(main())
