import json
import sys

import pytest

from tperf.noise import host_probe, summarize


def write_run(d, e2e, itl, steal):
    """One harness run directory as tperf generation run leaves it: results JSON + probe.json."""
    (d / "scen").mkdir(parents=True)
    entry = {
        "metadata": {"success": True},
        "config": {"name": "w5_i10-b1-s128-n64", "batch_size": 1, "num_tokens_to_generate": 64},
        "measurements": {
            "e2e_latency": [e2e] * 3,
            "time_to_first_token": [0.02] * 3,
            "inter_token_latency": [itl] * 3,
        },
    }
    (d / "scen" / "x_benchmark_1.json").write_text(json.dumps({"k": entry}))
    probe = {
        "cpu": {"foreign_cpu_s": 0.1, "steal_s_on_cpus": steal, "interrupts_on_cpus": 10, "runqueue_wait_ms": 1.0},
        "gpu": {"throttle_ms": {"sw_power_cap": 0.0}},
    }
    (d / "probe.json").write_text(json.dumps(probe))


def make_soak(tmp_path, name, e2es, steal=0.0):
    soak = tmp_path / name
    batch = tmp_path / f"batch-{name}"
    for i, e2e in enumerate(e2es, 1):
        write_run(batch / f"run{i}", e2e, e2e / 64, steal)
    soak.mkdir()
    meta = {"host": f"ubuntu@{name}", "label": name, "batches": [{"batch": 1, "exit": 0, "dir": str(batch)}]}
    (soak / "soak.json").write_text(json.dumps(meta))
    probe = {"metrics": {k: {"median": 1.0} for k in ("cpu_mops", "copy_gbs", "l3_ns", "dram_ns")}}
    (soak / "hostprobe.jsonl").write_text(json.dumps(probe) + "\n")
    samples = [
        {"t": 0, "busy": 0, "steal": 0, "total": 0},
        {"t": 60, "busy": 50, "steal": 6, "total": 600},
        {"t": 120, "busy": 60, "steal": 6, "total": 1200},
    ]
    (soak / "samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
    return soak


def test_describe_spread_and_drift():
    d = summarize.describe([1.0, 1.0, 2.0, 2.0])
    assert d["median"] == 1.5
    assert d["spread_pct"] == pytest.approx(100 / 1.5)
    assert d["drift_pct"] == pytest.approx(100.0)  # second half twice the first
    assert summarize.describe([]) is None


def test_summarize_reads_runs_probes_and_samples(tmp_path):
    s = summarize.summarize(make_soak(tmp_path, "new", [1.0, 1.1, 1.0, 1.2], steal=0.5))
    assert s["runs"] == 4
    (scen,) = s["bench"]
    e2e = s["bench"][scen]["e2e_ms"]
    assert e2e["n"] == 4 and e2e["median"] == pytest.approx(1050.0)
    assert e2e["drift_pct"] == pytest.approx(100 * (1.1 / 1.05 - 1))
    assert s["probe"]["steal_s"]["median"] == 0.5
    assert s["window"]["minutes"] == 2
    assert s["window"]["steal_pct"] == pytest.approx(0.5)
    assert s["window"]["worst_minute_steal_pct"] == pytest.approx(1.0)


def test_main_prints_side_by_side(tmp_path, capsys):
    a, b = make_soak(tmp_path, "old", [1.0] * 4), make_soak(tmp_path, "new", [1.0, 1.3] * 2)
    assert summarize.main([str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "old (old)" in out and "new (new)" in out and "Contention probe" in out


def test_parse_cpus():
    assert host_probe.parse_cpus("1-3,5") == [1, 2, 3, 5]


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_sample_counters():
    s = host_probe.sample([0])
    assert s["total"] >= s["busy"] >= s["steal"] >= 0 and s["irqs"] >= 0
