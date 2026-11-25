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
