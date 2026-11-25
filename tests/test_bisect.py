from tperf.bisect.run import Measurement, bisect, classify

COMMITS = [f"c{i:02d}" for i in range(30)]  # good..bad, oldest first; "good" is the commit before c00


def fake_measure(effects, failing=(), noise=0.3):
    """A/B of base vs head: the sum of the effects (commit -> %) landed after base, up to head."""
    order = ["good", *COMMITS]
    calls = []

    def measure(base, head):
        calls.append((base, head))
        if head in failing:
            return None
        lo, hi = order.index(base), order.index(head)
        change = sum(effects.get(c, 0.0) for c in order[lo + 1 : hi + 1])
        return Measurement(change, (change - noise, change + noise), 100.0, 100.0 * (1 + change / 100))

    return measure, calls


def parent(sha):
    order = ["good", *COMMITS]
    return order[order.index(sha) - 1]


def test_classify():
    d = 6.0
    assert classify(Measurement(6.0, (5.4, 6.6), 0, 0), d) == "bad"
    assert classify(Measurement(0.1, (-0.5, 0.7), 0, 0), d) == "good"
    assert classify(Measurement(3.0, (2.4, 3.6), 0, 0), d) == "unclear"  # half the change: two changes?
    assert classify(Measurement(5.0, (1.0, 9.0), 0, 0), d) == "unclear"  # too noisy to call


def test_finds_the_culprit_in_log_steps():
    measure, _ = fake_measure({"c17": 6.0})
    r = bisect(COMMITS, "good", measure, 1.0, parent)
    assert r.state == "culprit" and r.culprit["sha"] == "c17"
    assert r.culprit["parent"] == "c16"
    steps = [s for s in r.steps if s.role == "step"]
    assert len(steps) <= 5  # 30 commits
    assert r.steps[0].role == "reproduce" and r.steps[-1].role == "confirm"


def test_culprit_is_the_last_commit_or_the_first():
    for c in ("c00", "c29"):
        measure, _ = fake_measure({c: 6.0})
        r = bisect(COMMITS, "good", measure, 1.0, parent)
        assert r.state == "culprit" and r.culprit["sha"] == c


def test_not_reproduced_stops_before_bisecting():
    measure, calls = fake_measure({})  # the daily chart moved, transformers did not
    r = bisect(COMMITS, "good", measure, 1.0, parent)
    assert r.state == "not_reproduced" and len(calls) == 1 and r.culprit is None


def test_an_improvement_is_not_bisected_as_a_regression():
    measure, _ = fake_measure({"c10": -6.0})
    assert bisect(COMMITS, "good", measure, 1.0, parent).state == "not_reproduced"


def test_two_changes_stop_with_a_range():
    measure, _ = fake_measure({"c05": 3.0, "c20": 3.0})
    r = bisect(COMMITS, "good", measure, 1.0, parent)
    assert r.state == "range" and r.culprit is None
    first, last = (COMMITS.index(c) for c in r.range)
    assert first <= 5 and last >= 20


def test_skips_commits_that_cannot_run():
    measure, _ = fake_measure({"c17": 6.0}, failing={"c14", "c15"})
    r = bisect(COMMITS, "good", measure, 1.0, parent)
    assert r.state == "culprit" and r.culprit["sha"] == "c17"
    assert any(s.verdict == "skipped" for s in r.steps)


def test_all_skipped_reports_the_range():
    measure, _ = fake_measure({"c17": 6.0}, failing={f"c{i:02d}" for i in range(29)})
    r = bisect(COMMITS, "good", measure, 1.0, parent)
    assert r.state == "range" and r.range == ["c00", "c29"]


def test_failed_reproduce():
    measure, _ = fake_measure({"c17": 6.0}, failing={"c29"})
    r = bisect(COMMITS, "good", measure, 1.0, parent)
    assert r.state == "failed"


def test_too_noisy_is_not_a_verdict():
    # the 2026-11-25 gemma case: +0.78% [-2.90, +4.59] against a +5.5% daily step
    def measure(base, head):
        return Measurement(0.78, (-2.90, 4.59), 5738.6, 5802.1)

    assert bisect(COMMITS, "good", measure, 1.0, parent, expected=3.0).state == "too_noisy"
    r = bisect(COMMITS, "good", measure, 1.0, parent, expected=5.5)
    assert r.state == "not_reproduced" and "rules out" in r.reason


def test_on_step_reports_every_step_as_it_lands():
    measure, _ = fake_measure({"c17": 6.0})
    seen = []
    r = bisect(COMMITS, "good", measure, 1.0, parent, on_step=seen.append)
    assert [s.role for s in seen] == [s.role for s in r.steps]
    assert seen[0].role == "reproduce" and seen[-1].role == "confirm"
