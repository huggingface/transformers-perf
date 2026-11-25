from datetime import UTC, datetime

import pytest

from tperf.bisect import worker


@pytest.mark.parametrize(
    ("spec", "hour", "quiet"),
    [("1-5", 0, False), ("1-5", 1, True), ("1-5", 4, True), ("1-5", 5, False), ("22-2", 23, True), ("22-2", 1, True),
     ("22-2", 2, False), ("22-2", 12, False)],
)  # fmt: skip
def test_quiet_window(spec, hour, quiet):
    assert worker.quiet_now(spec, datetime(2026, 11, 25, hour, 30, tzinfo=UTC)) is quiet


def fake_api(monkeypatch, job):
    posted = []

    def call(method, path, body=None):
        if method == "GET":
            return (200, job) if job else (204, None)
        posted.append((path, body))
        return 200, {}

    monkeypatch.setattr(worker, "call", call)
    monkeypatch.setattr(worker, "quiet_now", lambda *a: False)
    monkeypatch.setattr(worker, "daily_lock_free", lambda: True)
    return posted


JOB = {"id": 7, "model": "google/gemma-3-1b-it", "scenario": {"attn_implementation": "flash_attention_2"},
       "bench_args": "-w 5 -i 6", "good": "a" * 40, "bad": "b" * 40, "metric": "e2e", "rounds": 4,
       "min_change": 1.0, "expected": 5.5}  # fmt: skip


def test_nothing_queued(monkeypatch):
    fake_api(monkeypatch, None)
    assert worker.poll_once() is False


def test_job_reports_steps_then_the_result(monkeypatch):
    from tperf.bisect import run

    posted = fake_api(monkeypatch, JOB)
    seen_env = {}

    def run_job(good, bad, metric, rounds, min_change, expected, on_step):
        import os

        seen_env.update(MODEL=os.environ["MODEL"], BENCH_ARGS=os.environ["BENCH_ARGS"])
        with open(os.environ["SCENARIOS"]) as f:
            seen_env["scenario"] = f.read()
        on_step(run.Step("reproduce", good, bad, "not_reproduced", 0.8, (-2.9, 4.6)))
        return {"state": "not_reproduced", "reason": "rules out"}, None

    monkeypatch.setattr(run, "run_job", run_job)
    monkeypatch.delenv("BENCH_ARGS", raising=False)
    assert worker.poll_once() is True
    assert seen_env == {"MODEL": JOB["model"], "BENCH_ARGS": "-w 5 -i 6",
                        "scenario": '{"attn_implementation": "flash_attention_2"}\n'}  # fmt: skip
    assert [b["kind"] for _, b in posted] == ["step", "done"]
    assert all(p == "/benchmarks/bisect/7/report" for p, _ in posted)
    assert posted[0][1]["step"]["verdict"] == "not_reproduced"
    assert posted[1][1]["result"]["state"] == "not_reproduced"
    import os

    assert "BENCH_ARGS" not in os.environ  # restored after the job


def test_a_crashing_job_still_ends(monkeypatch):
    from tperf.bisect import run

    posted = fake_api(monkeypatch, JOB)

    def run_job(*a):
        raise SystemExit("error: aaaaaaaaaa is not an ancestor of bbbbbbbbbb")

    monkeypatch.setattr(run, "run_job", run_job)
    worker.poll_once()
    assert posted[-1][1] == {"kind": "done",
                             "result": {"state": "failed", "reason": "error: aaaaaaaaaa is not an ancestor of bbbbbbbbbb"}}  # fmt: skip


def test_no_claim_while_the_daily_run_holds_the_lock(monkeypatch):
    fake_api(monkeypatch, JOB)
    monkeypatch.setattr(worker, "daily_lock_free", lambda: False)
    assert worker.poll_once() is False


PR_JOB = {"id": "abc", "kind": "pr", "pr": 123, "head": "e" * 40, "metric": "e2e", "rounds": 3,
          "scenarios": [{"model": "google/gemma-3-1b-it", "label": "fa2", "scenario": {"attn_implementation": "flash_attention_2"},
                         "bench_args": "-w 5"},
                        {"model": "meta-llama/Llama-2-7b-hf", "label": "eager", "scenario": {"attn_implementation": "eager"},
                         "bench_args": ""}]}  # fmt: skip


def test_pr_bench_runs_tperf_ab_per_model_and_reports_its_verdicts(tmp_path, monkeypatch):
    import os
    from pathlib import Path

    from tperf.ab import compare
    from tperf.common import record

    posted = fake_api(monkeypatch, PR_JOB)
    out = tmp_path / "ab-20261125-pr123"
    (out / "round1-base").mkdir(parents=True)
    (out / "round1-base" / "manifest.json").write_text("{}")
    seen = []

    def ab_pr(pr, rounds):
        seen.append((pr, rounds, os.environ["MODEL"], Path(os.environ["SCENARIOS"]).read_text().count("\n")))
        return str(out) if os.environ["MODEL"].startswith("google") else ""

    def side(sha):
        return {"attn_implementation": "flash_attention_2", "commit_id": sha}

    entries = [{"pairs": [(1, {"base": side("m" * 40), "head": side("e" * 40)})], "rounds": 6,
                "metrics": [{"name": "e2e", "change_pct": -3.0, "ci": [-4.0, -2.0], "base": 100.0, "head": 97.0,
                             "verdict": "improvement"}]}]  # fmt: skip
    monkeypatch.setattr(worker, "ab_pr", ab_pr)
    monkeypatch.setattr(record, "scenario_rows", lambda d: [{"row": 1}])
    monkeypatch.setattr(compare, "analyze", lambda rows: {"scenarios": entries})
    monkeypatch.delenv("MODEL", raising=False)
    monkeypatch.delenv("GPUS", raising=False)
    assert worker.poll_once() is True
    assert os.environ.pop("GPUS") == "1 5"
    assert seen == [(123, 3, "google/gemma-3-1b-it", 1), (123, 3, "meta-llama/Llama-2-7b-hf", 1)]
    steps = [b["step"] for _, b in posted if b["kind"] == "step"]
    assert steps[0]["verdict"] == "improvement" and steps[0]["change"] == -3.0 and steps[0]["base"] == "m" * 40
    assert steps[1]["verdict"] == "failed to run"
    result = posted[-1][1]["result"]
    assert result["state"] == "done" and result["page"] == "/benchmarks/pr-123"
    assert result["ab_ids"] == ["20261125-pr123"]
    assert "MODEL" not in os.environ


def test_daily_due_once_a_day_after_its_time_and_not_on_a_fresh_install(tmp_path, monkeypatch):
    state = tmp_path / "daily-last"
    monkeypatch.setenv("DAILY_STATE", str(state))
    at = lambda h, m=0, d=26: datetime(2026, 11, d, h, m, tzinfo=UTC)
    monkeypatch.delenv("DAILY_AT_UTC", raising=False)
    assert worker.daily_due(at(4)) is False  # the timer runs it
    monkeypatch.setenv("DAILY_AT_UTC", "03:00")
    assert (
        worker.daily_due(at(19)) is False and state.read_text() == "2026-11-26"
    )  # fresh install: today counts as done
    assert worker.daily_due(at(2, 59, 27)) is False
    assert worker.daily_due(at(3, 0, 27)) is True
    state.write_text("2026-11-27")
    assert worker.daily_due(at(23, 0, 27)) is False


def test_the_daily_run_is_registered_and_reports_each_stage(tmp_path, monkeypatch):
    import io

    monkeypatch.setenv("DAILY_STATE", str(tmp_path / "daily-last"))
    out = io.StringIO("2026-11-27T03:00:01Z main 0123456789: Fix things\nstages 3\nstage ok: GPU suites, run 1/3\n"
                      "noise\nstage failed: processors, run 1/3\n")  # fmt: skip

    class Proc:
        stdout = out

        def wait(self):
            return 1

    monkeypatch.setattr(worker.subprocess, "Popen", lambda *a, **kw: Proc())
    posted = []

    def call(method, path, body=None):
        posted.append((path, body))
        return (201, {"id": "d1"}) if path.endswith("/daily") else (200, {})

    monkeypatch.setattr(worker, "call", call)
    assert worker.run_daily() == 1
    assert posted[0] == ("/benchmarks/bisect/daily", {"total": 3})
    steps = [b["step"] for p, b in posted if b.get("kind") == "step"]
    assert steps == [
        {"label": "GPU suites, run 1/3", "verdict": "ok"},
        {"label": "processors, run 1/3", "verdict": "failed"},
    ]
    assert posted[-1][1]["result"]["state"] == "failed" and posted[-1][1]["result"]["commit"] == "0123456789"
    assert (tmp_path / "daily-last").read_text() == datetime.now(UTC).date().isoformat()


def test_no_quiet_window_when_the_worker_runs_the_daily(monkeypatch):
    monkeypatch.delenv("QUIET_UTC", raising=False)
    monkeypatch.setenv("DAILY_AT_UTC", "03:00")
    assert worker.quiet_now(None, datetime(2026, 11, 26, 3, 30, tzinfo=UTC)) is False
    monkeypatch.delenv("DAILY_AT_UTC")
    assert worker.quiet_now(None, datetime(2026, 11, 26, 3, 30, tzinfo=UTC)) is True
