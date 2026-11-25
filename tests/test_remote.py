import time

from tperf.remote import auth, bench


def test_credentials_are_private_and_per_dashboard(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    auth.save({auth.dashboard(): {"access_token": "t", "expires": time.time() + 60, "login": "me"}})
    path = tmp_path / "tperf" / "dashboard-auth.json"
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert auth.access_token() == "t"
    monkeypatch.setenv("TPERF_DASHBOARD", "https://other.example")
    try:
        auth.access_token()
        raise AssertionError("expected not signed in")
    except ValueError as e:
        assert "tperf auth login" in str(e)


def test_bench_queues_then_prints_each_scenario_and_the_page(monkeypatch, capsys):
    monkeypatch.setattr(bench, "access_token", lambda: "tok")
    monkeypatch.setattr(bench, "POLL", 0)
    monkeypatch.setattr(bench.time, "sleep", lambda s: None)
    step = {
        "model": "google/gemma-3-1b-it",
        "scenario": "fa2",
        "change": -3.1,
        "ci": [-4.0, -2.2],
        "verdict": "faster",
    }
    polls = iter([
        {"state": "running", "scenarios": [1, 2], "steps": [step]},
        {"state": "done", "scenarios": [1, 2], "steps": [step, {**step, "scenario": "eager", "change": None,
                                                               "verdict": "failed to run"}], "result": {"state": "done"}},
    ])  # fmt: skip
    calls = []

    def request(path, body=None, token=None):
        calls.append((path, body, token))
        return {"id": "abc", "url": "/benchmarks/pr-bench/abc"} if body else next(polls)

    monkeypatch.setattr(bench, "request", request)
    assert bench.main(["123"]) == 0
    out = capsys.readouterr().out
    assert calls[0] == ("/benchmarks/pr-bench", {"pr": 123}, "tok")
    assert "fa2: -3.10% [-4.00, -2.20] → faster" in out and "eager: failed to run" in out
    assert out.count("fa2:") == 1
    assert out.strip().endswith("Results: https://transformers-ci.huggingface.cool/benchmarks/pr-bench/abc")


def test_bench_explains_refusals(monkeypatch, capsys):
    monkeypatch.setattr(bench, "access_token", lambda: "tok")

    def request(path, body=None, token=None):
        raise auth.DashboardError("pr_already_queued")

    monkeypatch.setattr(bench, "request", request)
    assert bench.main(["123"]) == 1
    assert "already has a bench" in capsys.readouterr().err
