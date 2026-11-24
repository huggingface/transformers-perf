"""tperf.host.prepare: the pure parts (topology, CPU lists, IRQ classification) and the state handling, on fake
sysfs/procfs trees and mocked commands. Nothing here touches the machine running the tests."""

import json
from pathlib import Path

import pytest

from tperf.host import prepare

# g5.4xlarge: 8 cores, 2 threads each (CPU n and n+8 share a core), one node
VM = "# CPU,Core,Node\n" + "".join(f"{c},{c % 8},0\n" for c in range(16))
# g5.48xlarge-like: 2 nodes, 4 cores each here, siblings at +8
BIG = "# CPU,Core,Node\n" + "".join(f"{c},{c % 8},{(c % 8) // 4}\n" for c in range(16))


def test_parse_cpulist():
    assert prepare.parse_cpulist("0-3,8-11") == {0, 1, 2, 3, 8, 9, 10, 11}
    assert prepare.parse_cpulist("5") == {5}
    assert prepare.parse_cpulist("") == set()
    assert prepare.parse_cpulist("0,48\n") == {0, 48}


def test_cpu_ranges_round_trip():
    cpus = [1, 2, 3, 5, 9, 10, 11, 12, 13, 14, 15]
    assert prepare.cpu_ranges(cpus) == "1-3,5,9-15"
    assert prepare.parse_cpulist(prepare.cpu_ranges(cpus)) == set(cpus)


def test_housekeeping_takes_the_whole_first_core():
    assert prepare.housekeeping(VM) == {"0": [0, 8]}
    assert prepare.housekeeping(BIG) == {"0": [0, 8], "1": [4, 12]}


def test_housekeeping_non_contiguous_nodes():
    topo = "# CPU,Core,Node\n0,0,0\n1,1,0\n2,2,3\n3,3,3\n"
    assert prepare.housekeeping(topo) == {"0": [0], "3": [2]}


def test_benchmark_cpus_exclude_housekeeping_siblings():
    hk = set().union(*prepare.housekeeping(VM).values())
    bench = prepare.all_cpus(VM) - hk
    assert 8 not in bench and 0 not in bench and len(bench) == 14


def test_classify_irqs_uses_sets_not_strings():
    hk = {0, 48}
    irqs = [
        ("ens5-Tx-Rx-0", "0,48", "0"),  # routed and moved: listed differently than the hk set, still fine
        ("nvidia", "0,48", "13"),  # accepted, moves at its next delivery
        ("nvme0q1", "1", "1"),  # managed, single benchmark CPU: refused
        ("nvme0q2", "1", "1"),
        ("nvme1q1", "0-3,8-11", "0"),  # managed, mask holds a housekeeping CPU: moved by isolcpus=managed_irq
        ("timer", "0-95", "0"),  # every CPU: not a device interrupt we route
    ]
    refused, pending = prepare.classify_irqs(irqs, hk, 96)
    assert refused == {"nvme0q1": 1, "nvme0q2": 1}
    assert pending == ["nvidia"]


def test_irq_affinity_routes_gpu_to_its_node(tmp_path, monkeypatch):
    """BOOT_CODE's irq_affinity with node -> cpulist: every IRQ to all housekeeping CPUs, a GPU's to its node's."""
    writes = {}
    monkeypatch.setitem(prepare._boot, "write", lambda path, value, check=True: writes.__setitem__(path, value))
    gpu = tmp_path / "0000:00:1e.0"
    (gpu / "msi_irqs").mkdir(parents=True)
    (gpu / "msi_irqs" / "61").write_text("")
    (gpu / "vendor").write_text("0x10de\n")
    (gpu / "class").write_text("0x030200\n")
    real_glob = prepare._boot["glob"].glob
    monkeypatch.setattr(
        prepare._boot["glob"],
        "glob",
        lambda pat: ["/proc/irq/5"] if pat.startswith("/proc/irq") else [str(gpu)] if "pci" in pat else real_glob(pat),
    )
    prepare._boot["irq_affinity"]({"0": "0,8", "1": "4,12"}, ["1"])
    assert writes["/proc/irq/5/smp_affinity_list"] == "0,8,4,12"
    assert writes["/proc/irq/61/smp_affinity_list"] == "4,12"


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "STATE_DIR", tmp_path)
    monkeypatch.setattr(prepare, "ORIG_STATE", tmp_path / "orig.json")
    monkeypatch.setattr(prepare, "STEPS", [])
    return tmp_path


def test_orig_state_not_recorded_from_a_tuned_host(state_dir, monkeypatch):
    monkeypatch.setattr(prepare, "output", lambda *a, **k: pytest.fail("must not read tuned values as originals"))
    prepare.log("snapshot")
    prepare.save_orig_state(first_apply=False)
    assert json.loads((state_dir / "orig.json").read_text()) == {"complete": False}
    assert "revert resets" in prepare.STEPS[-1]["notes"][-1]


def test_orig_state_written_once(state_dir, monkeypatch):
    (state_dir / "orig.json").write_text('{"complete": true, "sysctl": {"vm.stat_interval": "1"}}')
    monkeypatch.setattr(prepare, "output", lambda *a, **k: pytest.fail("snapshot must not be retaken"))
    prepare.save_orig_state(first_apply=True)
    assert json.loads((state_dir / "orig.json").read_text())["sysctl"] == {"vm.stat_interval": "1"}


def test_orig_state_first_apply_records_sysctls(state_dir, monkeypatch):
    monkeypatch.setattr(prepare, "output", lambda *a, **k: "1" if a[:2] == ("sysctl", "-n") else "")
    monkeypatch.setattr(prepare, "nvidia_ok", lambda: False)
    monkeypatch.setattr(prepare.glob, "glob", lambda pat: [])
    prepare.save_orig_state(first_apply=True)
    saved = json.loads((state_dir / "orig.json").read_text())
    assert saved["complete"] and set(saved["sysctl"]) == set(prepare.SYSCTLS)


def test_isolcpus_retries_update_grub_until_the_boot_config_has_it(tmp_path, monkeypatch):
    grub_file, grub_cfg = tmp_path / "99-bench.cfg", tmp_path / "grub.cfg"
    monkeypatch.setattr(prepare, "GRUB_FILE", grub_file)
    monkeypatch.setattr(prepare, "GRUB_CFG", grub_cfg)
    monkeypatch.setattr(prepare, "STEPS", [])

    def fake_read(path, fallback=None):
        if path == "/proc/cmdline":
            return "BOOT_IMAGE=x ro"
        return Path(path).read_text() if Path(path).exists() else fallback

    monkeypatch.setattr(prepare, "read", fake_read)
    calls = []
    monkeypatch.setattr(prepare, "run", lambda *a, **k: calls.append(a) or False)  # update-grub fails
    prepare.apply_isolcpus([1, 2, 3])
    assert grub_file.read_text().count("isolcpus=managed_irq,1-3") == 1
    assert any("update-grub" in s["blocked"][0] for s in prepare.STEPS if s["blocked"])
    # next apply: drop-in unchanged, but the boot config still lacks the argument -> update-grub runs again
    calls.clear()
    monkeypatch.setattr(
        prepare, "run", lambda *a, **k: calls.append(a) or grub_cfg.write_text("isolcpus=managed_irq,1-3") or True
    )
    prepare.apply_isolcpus([1, 2, 3])
    assert ("update-grub",) in calls
    assert any("reboot required" in b for s in prepare.STEPS for b in s["blocked"])


def test_revert_writes_sysctl_defaults_when_no_originals(state_dir, monkeypatch, tmp_path):
    """Deleting the sysctl.d file leaves the tuned values in the kernel: revert must write values back."""
    (state_dir / "units.orig").write_text("")
    monkeypatch.setattr(prepare, "UNIT_STATE", state_dir / "units.orig")
    monkeypatch.setattr(prepare, "RUNTIME_STATE", state_dir / "runtime.orig")
    for name in ("SYSCTL_FILE", "APT_FILE", "GRUB_FILE", "BOOT_SCRIPT", "OLD_BOOT_SCRIPT", "PERSISTENCED_DROPIN"):
        monkeypatch.setattr(prepare, name, tmp_path / name)
    monkeypatch.setattr(prepare, "BOOT_UNIT", "x.service")
    monkeypatch.setattr(prepare, "Path", lambda p: tmp_path / "unit" if str(p).startswith("/etc/") else Path(p))
    monkeypatch.setattr(prepare.glob, "glob", lambda pat: [])
    monkeypatch.setattr(prepare.shutil, "which", lambda cmd: None)
    calls = []
    monkeypatch.setattr(prepare, "run", lambda *a, **k: calls.append(a) or True)
    prepare.revert()
    assert ("sysctl", "-q", "-w", "kernel.randomize_va_space=2") in calls
    assert ("sysctl", "-q", "-w", "vm.stat_interval=1") in calls
