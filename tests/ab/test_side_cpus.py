from tperf.ab import run


def fake_cpus_for(node: int, count: int) -> str:
    """Node n's cores, housekeeping core skipped: 1..count on node 0, 25..24+count on node 1."""
    return ",".join(str(node * 24 + i) for i in range(1, count + 1))


def test_sides_on_two_nodes_each_get_their_node(monkeypatch):
    monkeypatch.setattr(run, "cpus_for", fake_cpus_for)
    cpus = run.side_cpus(["0", "4"], {"0": 0, "4": 1}, 3)
    assert cpus == {"0": "1,2,3", "4": "25,26,27"}


def test_sides_on_one_node_share_no_core(monkeypatch):
    monkeypatch.setattr(run, "cpus_for", fake_cpus_for)
    cpus = run.side_cpus(["0", "1"], {"0": 0, "1": 0}, 3)
    assert cpus == {"0": "1,2,3", "1": "4,5,6"}


def test_tp_groups_on_one_node(monkeypatch):
    monkeypatch.setattr(run, "cpus_for", fake_cpus_for)
    cpus = run.side_cpus(["0,1", "2,3"], {"0,1": 0, "2,3": 0}, 4)
    assert cpus == {"0,1": "1,2,3,4", "2,3": "5,6,7,8"}
