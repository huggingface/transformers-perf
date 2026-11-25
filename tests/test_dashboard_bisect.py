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
