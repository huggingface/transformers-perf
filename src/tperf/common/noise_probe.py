"""Measure CPU/GPU usage and interference around a command, without sampling during the run.

Runs on the bench host. Everything is read from cumulative counters before and after the
command, so the probe itself does no work while the benchmark runs:

  - CPU: /proc/stat per-CPU busy time on the benchmark's CPUs, the command's own CPU time and
    context switches (getrusage), /proc/interrupts and /proc/schedstat run-queue wait on the
    benchmark's CPUs. "foreign CPU time"
    is busy time on the benchmark's CPUs that the command did not use (other tasks, IRQs).
  - GPU: NVML's own driver-side utilization sample buffer (read once at the end), total energy
    counter (average power), throttle (violation) counters, and max memory clock / SM clock.

    python -m tperf.common.noise_probe --cpus 1-8 --gpu 0 --out probe.json -- <command> [args...]

The command inherits the probe's CPU affinity and environment; the probe's exit code is the
command's.
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import time

try:
    import pynvml
except ImportError:  # GPU section is skipped without nvidia-ml-py
    pynvml = None


def parse_cpus(spec: str) -> list[int]:
    cpus = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        cpus.extend(range(int(lo), int(hi or lo) + 1))
    return cpus


def cpu_busy_ticks(cpus: list[int]) -> dict[int, int]:
    """Busy jiffies (everything except idle and iowait) per CPU."""
    out = {}
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu") and line[3].isdigit():
                name, *vals = line.split()
                cpu = int(name[3:])
                if cpu in cpus:
                    v = list(map(int, vals))
                    out[cpu] = sum(v) - v[3] - v[4]
    return out


def run_delay_ns(cpus: list[int]) -> int:
    """Total time tasks spent runnable but waiting for one of the given CPUs (/proc/schedstat, ns)."""
    total = 0
    with open("/proc/schedstat") as f:
        for line in f:
            if line.startswith("cpu") and line[3].isdigit():
                name, *vals = line.split()
                if int(name[3:]) in cpus:
                    total += int(vals[7])
    return total


def interrupts(cpus: list[int]) -> dict[str, int]:
    """Interrupt counts per source, summed over the given CPUs."""
    with open("/proc/interrupts") as f:
        header = f.readline().split()
        cols = [header.index(f"CPU{c}") for c in cpus if f"CPU{c}" in header]
        out = {}
        for line in f:
            parts = line.split()
            if not parts:
                continue
            name = parts[0].rstrip(":")
            counts = parts[1 : 1 + len(header)]
            total = sum(int(counts[i]) for i in cols if i < len(counts) and counts[i].isdigit())
            desc = " ".join(parts[1 + len(header) :])[-40:]
            out[f"{name} {desc}".strip()] = total
    return out


class Gpu:
    VIOLATIONS = {"power": "NVML_PERF_POLICY_POWER", "thermal": "NVML_PERF_POLICY_THERMAL"}

    def __init__(self, index: int):
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(index)  # NVML indexes ignore CUDA_VISIBLE_DEVICES
        self.t0_us = int(time.time() * 1e6)
        self.energy0 = self._energy()
        self.viol0 = self._violations()

    def _energy(self):
        try:
            return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h)  # mJ
        except pynvml.NVMLError:
            return None

    def _violations(self) -> dict[str, int]:
        out = {}
        for name, policy in self.VIOLATIONS.items():
            try:
                out[name] = pynvml.nvmlDeviceGetViolationStatus(self.h, getattr(pynvml, policy)).violationTime
            except (pynvml.NVMLError, AttributeError):
                pass
        return out

    def finish(self, elapsed_s: float) -> dict:
        res: dict = {"name": pynvml.nvmlDeviceGetName(self.h)}
        try:
            _, samples = pynvml.nvmlDeviceGetSamples(self.h, pynvml.NVML_GPU_UTILIZATION_SAMPLES, self.t0_us)
            vals = [s.sampleValue.uiVal for s in samples]
            if vals:
                res["util_samples"] = len(vals)
                res["util_mean_pct"] = sum(vals) / len(vals)
                res["util_max_pct"] = max(vals)
        except pynvml.NVMLError as e:
            res["util_error"] = str(e)
        energy1 = self._energy()
        if self.energy0 is not None and energy1 is not None and elapsed_s > 0:
            res["avg_power_w"] = (energy1 - self.energy0) / 1e3 / elapsed_s
        viol1 = self._violations()
        res["throttle_ms"] = {k: (viol1[k] - self.viol0[k]) / 1e6 for k in viol1 if k in self.viol0}
        res["sm_clock_mhz_now"] = pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_SM)
        res["mem_clock_mhz_now"] = pynvml.nvmlDeviceGetClockInfo(self.h, pynvml.NVML_CLOCK_MEM)
        pynvml.nvmlShutdown()
        return res


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cpus", required=True, help="CPUs the command is pinned to, e.g. 1-8")
    p.add_argument("--gpu", type=int, help="physical GPU index (NVML numbering)")
    p.add_argument("--out", required=True)
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    a = p.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    cpus = parse_cpus(a.cpus)

    hz = os.sysconf("SC_CLK_TCK")
    gpu = Gpu(a.gpu) if pynvml is not None and a.gpu is not None else None
    busy0, irq0, delay0 = cpu_busy_ticks(cpus), interrupts(cpus), run_delay_ns(cpus)
    ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.monotonic()

    rc = subprocess.call(cmd)

    elapsed = time.monotonic() - t0
    ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    busy1, irq1, delay1 = cpu_busy_ticks(cpus), interrupts(cpus), run_delay_ns(cpus)

    busy_s = sum(busy1[c] - busy0[c] for c in cpus) / hz
    own_s = (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)
    irq_delta = {k: irq1[k] - irq0.get(k, 0) for k in irq1 if irq1[k] - irq0.get(k, 0) > 0}
    report = {
        "cmd": cmd,
        "exit_code": rc,
        "elapsed_s": elapsed,
        "cpu": {
            "cpus": a.cpus,
            "busy_s_on_cpus": busy_s,
            "command_cpu_s": own_s,
            # Busy time on our CPUs that the command did not account for (other tasks, IRQs).
            # Approximate: the command may also run briefly on other CPUs (e.g. NVML threads).
            "foreign_cpu_s": max(0.0, busy_s - own_s),
            "voluntary_ctx_switches": ru1.ru_nvcsw - ru0.ru_nvcsw,
            "involuntary_ctx_switches": ru1.ru_nivcsw - ru0.ru_nivcsw,
            # Time tasks were ready to run on our CPUs but had to wait (includes the command's own
            # threads waiting for each other).
            "runqueue_wait_ms": (delay1 - delay0) / 1e6,
            "interrupts_on_cpus": sum(irq_delta.values()),
            "top_interrupts": dict(sorted(irq_delta.items(), key=lambda kv: -kv[1])[:8]),
        },
    }
    if gpu is not None:
        report["gpu"] = gpu.finish(elapsed)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    return rc


if __name__ == "__main__":
    sys.exit(main())
