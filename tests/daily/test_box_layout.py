import pytest

from tperf.daily import main as daily


@pytest.fixture
def suites(monkeypatch):
    monkeypatch.setattr(daily, "SUITES", dict(daily.SUITES))
    monkeypatch.setattr(daily, "SERVE_SUITES", dict(daily.SERVE_SUITES))


def test_suite_gpus_override(suites):
    daily.box_gpus("llama2-7b=0 gemma3-1b-sliding=1 serve-qwen3-1.7b=1 llama2-7b-tp2=2,3")
    assert daily.SUITES["gemma3-1b-sliding"][0] == "1"
    assert daily.SUITES["llama2-7b-tp2"][0] == "2,3"
    assert daily.SERVE_SUITES["serve-qwen3-1.7b"][0] == "1"


def test_unknown_suite_is_refused(suites):
    with pytest.raises(SystemExit):
        daily.box_gpus("nope=1")


def test_suites_on_one_node_get_their_own_slot(suites, monkeypatch):
    monkeypatch.setattr(daily, "GPU_NUMA_MAP", [0, 0, 0, 0])
    daily.box_gpus("gemma3-1b-sliding=1")
    assert daily.slots(["llama2-7b", "gemma3-1b-sliding"], daily.SUITES) == {"llama2-7b": 0, "gemma3-1b-sliding": 1}


def test_suites_on_two_nodes_both_take_slot_0(suites, monkeypatch):
    monkeypatch.setattr(daily, "GPU_NUMA_MAP", [0, 0, 0, 0, 1, 1, 1, 1])
    assert daily.slots(["llama2-7b", "gemma3-1b-sliding"], daily.SUITES) == {"llama2-7b": 0, "gemma3-1b-sliding": 0}


def test_cpus_for_slots_do_not_overlap(monkeypatch):
    lscpu = "# CPU,CORE,NODE\n" + "".join(f"{c},{c % 24},0\n" for c in range(48))  # 24 cores, siblings at +24

    class P:
        stdout = lscpu

    monkeypatch.setattr(daily.subprocess, "run", lambda *a, **k: P())
    monkeypatch.setattr(daily, "NCORES", 8)
    assert daily.cpus_for(0, 0) == "1,2,3,4,5,6,7,8"
    assert daily.cpus_for(0, 1) == "9,10,11,12,13,14,15,16"
    assert daily.cpus_for(0, 0, 16) == ",".join(str(c) for c in range(1, 17))
    with pytest.raises(SystemExit):
        daily.cpus_for(0, 2, 8)  # 1 + 24 cores needed, 23 left after housekeeping
