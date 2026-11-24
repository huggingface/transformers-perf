from tperf.ab.compare import analyze
from tperf.dashboard.build import DERIVED, build, value


def row(day: str, group: str, **extra) -> dict:
    return {
        "label": "daily-gemma3-1b-sliding",
        "source": "main",
        "series_key": "k",
        "group_id": group,
        "started_at": f"{day}T03:00:00+00:00",
        "success": True,
        "commit_id": "c" * 40,
        "model_id": "google/gemma-3-1b-it",
        "attn_implementation": "sdpa",
        "batch_size": 1,
        "sequence_length": 448,
        "num_tokens_to_generate": 128,
        "e2e_ms_median": 100.0,
        "startup_weights_mb": 1907.1,
        "startup_weights_dtypes": "bfloat16:1907.1,float32:0.5",
        "startup_measurement_working_set_mb": 57.5,
        "startup_measurement_peak_memory_mb": 1964.6,
        "startup_measurement_peak_reserved_mb": 2010.0,
        "startup_load_host_peak_rss_mb": 3407.6,
        **extra,
    }


def test_derived_memory_metrics():
    r = row("2026-11-24", "g1")
    assert DERIVED["fp32_mb"](r) == 0.5
    assert DERIVED["working_set_kb_per_token"](r) == 57.5 / 576 * 1024
    assert round(DERIVED["allocator_slack_mb"](r), 1) == 45.4
    assert round(DERIVED["load_rss_per_weight"](r), 2) == 1.79
    assert value(r, "derived:fp32_mb") == 0.5
    assert value({"startup_weights_dtypes": None}, "derived:fp32_mb") is None
    assert value({}, "derived:load_rss_per_weight") is None


def test_daily_points_link_their_memory_report():
    rows = [row(f"2026-11-{d:02d}", f"g{d}") for d in range(10, 25)]
    reports = {"g24": "memory/2026-11/g24", "g3": "memory/2026-11/g3"}  # g3: no such daily point
    pages = build(rows, [], reports)
    assert pages["_memory"] == {"g24": "memory/2026-11/g24"}
    points = pages["main.html"]["series"][0]["points"]
    assert points[-1]["memory"] == "memory/g24/report.html"
    assert "memory" not in points[-2]
    assert "weights" in pages["main.html"]["series"][0]["metrics"]
    assert "fp32" in pages["main.html"]["metrics"]


def test_ab_skips_non_positive_memory_values():
    def side(name: str, gpu: int, rnd: int) -> dict:
        return {"scenario": "s", "ab_round": rnd, "ab_side": name, "gpu_index": gpu, "success": True,
                "e2e_ms_median": 100.0, "startup_weights_mb": 100.0 if name == "base" else 110.0,
                "startup_measurement_working_set_mb": 0.0}  # fmt: skip

    rows = [side(n, g, r) for r in range(4) for n, g in (("base", r % 2), ("head", 1 - r % 2))]
    metrics = {m["name"]: m for m in analyze(rows)["scenarios"][0]["metrics"]}
    assert metrics["weights"]["verdict"] == "REGRESSION"
    assert "gpu ws" not in metrics  # zero: no ratio
