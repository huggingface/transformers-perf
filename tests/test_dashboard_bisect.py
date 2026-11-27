from tperf.dashboard.build import bisect_job

ROW = {
    "bench_args": "-w 5 -i 6 -b 1 -s 448 -n 128 --config-file /opt/bench/daily/scenarios/sliding-window.jsonl",
    "attn_implementation": "flash_attention_2",
    "compile_mode": None,
    "continuous_batching": False,
    "kernelize": False,
    "tp": None,
    "scenario": "w5_i6-unmonitored-b1_s448_n128-flash_attention_2-uncompiled-unkernelized-generate-no_tp",
}


def test_one_scenario_of_a_config_file_suite():
    assert bisect_job(ROW) == {"bench_args": "-w 5 -i 6 -b 1 -s 448 -n 128",
                               "config": {"attn_implementation": "flash_attention_2"}}  # fmt: skip


def test_one_scenario_of_a_level_suite():
    row = ROW | {"bench_args": "-w 5 -i 10 -b 1 -s 128 -n 64 --level 1", "attn_implementation": "flash_attention_2",
                 "continuous_batching": True, "compile_mode": "reduce-overhead"}  # fmt: skip
    assert bisect_job(row) == {
        "bench_args": "-w 5 -i 10 -b 1 -s 128 -n 64",
        "config": {"attn_implementation": "flash_attention_2", "compile_kwargs": {"mode": "reduce-overhead"},
                   "continuous_batching": True},
    }  # fmt: skip


def test_what_the_bisect_cannot_rerun():
    assert bisect_job(ROW | {"tp": 2}) is None
    assert bisect_job(ROW | {"scenario": "serve:multiturn"}) is None
    assert bisect_job(ROW | {"bench_args": None}) is None


def serve_row(day: str, ttft: float, run: int) -> dict:
    return {"label": "daily-serve-gemma3-1b", "source": "main", "series_key": "serve-key", "group_id": f"{day}-serve",
            "started_at": f"{day}T03:{run:02d}:00", "success": True, "commit_id": "c" * 40, "model_id": "google/gemma-3-1b-it",
            "gpu_name": "NVIDIA A10G", "gpu_index": 4, "instance_type": "g5.48xlarge", "scenario": "serve:chat", "n_samples": 24,
            "num_tokens_to_generate": 64, "compile_mode": None, "continuous_batching": False,
            "ttft_ms_median": ttft, "e2e_ms_median": 900.0, "bench_args": "serve chat"}  # fmt: skip


def test_daily_serve_runs_are_a_serve_series_that_cannot_be_bisected():
    from tperf.dashboard.build import main_series

    rows = [serve_row("2026-11-26", 30.0 + i, i) for i in range(3)]
    (series,) = main_series(rows)
    assert series["group"] == "Serve" and series["bisect"] is None
    assert series["instance_type"] == "g5.48xlarge"
    assert series["scenario"].startswith("transformers serve, chat")
    assert series["points"][0]["values"]["ttft"][0] == 31.0 and series["points"][0]["runs"] == 3


def test_the_daily_serve_suite_names_a_scenario_that_exists(monkeypatch):
    import importlib
    import json
    import os
    from pathlib import Path

    monkeypatch.setattr(os, "environ", dict(os.environ))  # the module sets the host's PYTHONPATH and HF_HOME
    daily = importlib.import_module("tperf.daily.main")
    names = [json.loads(line)["name"] for line in Path("scenarios/serve-multiturn.jsonl").read_text().splitlines()
             if line.startswith("{")]  # fmt: skip
    for gpu, model, name in daily.SERVE_SUITES.values():
        assert name in names and gpu not in {g for g, *_ in daily.SUITES.values()} - {"4"}
