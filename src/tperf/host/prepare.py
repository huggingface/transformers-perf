#!/usr/bin/env python3
"""Quiet an Ubuntu host so benchmarks run with minimal noise.

Usage (from your workstation; standalone, standard library only: this file is piped to `sudo python3 -`):
    tperf host tune apply    # quiet the box (idempotent); same as ssh HOST sudo python3 - apply < prepare.py
    tperf host tune status   # show current state
    tperf host tune revert   # restore original state
or on the host: sudo python3 /opt/bench/src/tperf/host/prepare.py apply

Options (environment variables; `tperf host tune` passes them on from your environment, e.g.
    KEEP_SSM=1 tperf host tune apply):
  KEEP_SSM=1         keep the AWS SSM agent running (default: stopped)
  KEEP_CSTATES=1     keep deep C-states enabled (default: states with exit latency > 10us disabled)
  NVIDIA_DRIVER=615-server-open  driver branch to install if no driver is present ("none" to skip)
  CACHE_MOUNT=/mnt/cache  where to mount a local NVMe instance-store disk, for HF_HOME ("none" to skip)
  GPU_NUMA_MAP="0 0 0 0 1 1 1 1"  GPU index -> NUMA node, used to route each GPU's interrupts
                     to its node's housekeeping CPU (the first CPU of the node)
  ISOLATE_MANAGED_IRQS=0  do not add isolcpus=managed_irq,<benchmark CPUs> to the kernel command line
                     (default: added, takes effect at the next reboot). Moves kernel-managed interrupts
                     (NVMe queues) off the benchmark CPUs when their mask also holds a housekeeping CPU;
                     a queue bound to benchmark CPUs only still fires there, on that CPU's own I/O.
  GPU_SM_CLOCK=1500  SM clock lock in MHz, "max" (max boost clock), or "off" (no lock).
                     Must be a clock the GPU can hold under its power limit, or the power cap
                     overrides the lock and clocks drift with temperature. On g5.48xlarge (A10G,
                     300 W) fp16 GEMM hits the cap at ~1620-1650 MHz; 1500 MHz holds at ~255 W
                     with no throttling. Re-check with `tperf baselines run` on other hardware.

The NVIDIA driver is installed from Ubuntu's prebuilt, kernel-matched modules (no DKMS build),
and the driver and running kernel packages are put on apt hold so they cannot drift between runs.

Persistence: masked units, sysctls, the kernel command line and apt/snap settings survive a reboot. THP, KSM, C-states
and the GPU clock lock are runtime-only; "apply" installs bench-tuning.service, which
re-applies them at boot. Re-run "apply" after changing options, so the boot unit picks them up.

The first "apply" saves the original state to /var/lib/bench-prep/ so "revert" can restore it.
"""

import argparse
import glob
import itertools
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
from pathlib import Path


STATE_DIR = Path("/var/lib/bench-prep")
UNIT_STATE = STATE_DIR / "units.orig"
RUNTIME_STATE = STATE_DIR / "runtime.orig"
REPORT_FILE = STATE_DIR / "report.txt"  # what the last "apply" did and could not do; shown by "status"
SYSCTL_FILE = Path("/etc/sysctl.d/99-bench.conf")
APT_FILE = Path("/etc/apt/apt.conf.d/99-bench-no-auto")
BOOT_UNIT = "bench-tuning.service"
BOOT_SCRIPT = Path("/usr/local/sbin/bench-tuning.py")
OLD_BOOT_SCRIPT = Path("/usr/local/sbin/bench-tuning.sh")  # written by the former prepare-bench-host.sh
CACHE_MOUNT = os.environ.get("CACHE_MOUNT") or "/mnt/cache"
CACHE_LABEL = "benchcache"
PERSISTENCED_DROPIN = Path("/etc/systemd/system/nvidia-persistenced.service.d/bench.conf")
THP = "/sys/kernel/mm/transparent_hugepage"
KSM_RUN = "/sys/kernel/mm/ksm/run"
# GPU index -> NUMA node. The VM reports numa_node=-1 for the GPUs; measured on g5.48xlarge with
# nvbandwidth host_device_latency_sm (GPUs 0-3 are ~110 ns closer to node 0, 4-7 to node 1).
GPU_NUMA_MAP = os.environ.get("GPU_NUMA_MAP") or "0 0 0 0 1 1 1 1"
KEEP_CSTATES = os.environ.get("KEEP_CSTATES", "0") == "1"
ISOLATE_MANAGED_IRQS = os.environ.get("ISOLATE_MANAGED_IRQS", "1") != "0"
GRUB_FILE = Path("/etc/default/grub.d/99-bench.cfg")

# Long-running daemons that are not needed to run benchmarks.
SERVICES = [
    "unattended-upgrades.service",  # can install packages in the middle of a run
    "packagekit.service",
    "fwupd.service",  # firmware update daemon
    "snapd.service", "snapd.socket", "snapd.seeded.service",
    "ModemManager.service",
    "udisks2.service",
    "multipathd.service", "multipathd.socket",
    "irqbalance.service",  # moves IRQs between CPUs at runtime
    "rsyslog.service",  # duplicates journald writes to disk
    "cron.service",  # runs cron.daily/weekly jobs
    "apport.service", "apport-autoreport.path", "apport-forward.socket",
    "sysstat.service",
]  # fmt: skip
if os.environ.get("KEEP_SSM", "0") != "1":
    SERVICES.append("snap.amazon-ssm-agent.amazon-ssm-agent.service")

# Periodic jobs.
TIMERS = [
    "apt-daily.timer", "apt-daily-upgrade.timer",
    "fwupd-refresh.timer",
    "motd-news.timer",
    "update-notifier-download.timer", "update-notifier-motd.timer",
    "man-db.timer",
    "dpkg-db-backup.timer",
    "e2scrub_all.timer", "xfs_scrub_all.timer",
    "fstrim.timer",
    "logrotate.timer",
    "systemd-tmpfiles-clean.timer",
    "sysstat-collect.timer", "sysstat-summary.timer", "sysstat-rotate.timer",
    "apport-autoreport.timer",
    "snapd.snap-repair.timer",
]  # fmt: skip

ALL_UNITS = SERVICES + TIMERS

# Runtime-only settings, shared with the boot script (BOOT_SCRIPT), which embeds this code as is.
# check=False: best effort, as at boot nothing should stop the remaining settings from being applied.
BOOT_CODE = '''
import glob
import os
import subprocess


def write(path, value, check=True):
    try:
        with open(path, "w") as f:
            f.write(f"{value}\\n")
    except OSError:
        if check:
            raise


def quiet_memory(check=True):
    """THP to madvise, KSM off."""
    write("/sys/kernel/mm/transparent_hugepage/enabled", "madvise", check)
    write("/sys/kernel/mm/transparent_hugepage/defrag", "madvise", check)
    if os.access("/sys/kernel/mm/ksm/run", os.W_OK):
        write("/sys/kernel/mm/ksm/run", 0, check)


def disable_deep_cstates(check=True):
    """Disable idle states with exit latency > 10us."""
    for st in glob.glob("/sys/devices/system/cpu/cpu*/cpuidle/state*"):
        try:
            latency = int(open(f"{st}/latency").read())
        except (OSError, ValueError):
            continue
        if latency > 10:
            write(f"{st}/disable", 1, check)


def irq_affinity(hk, gpu_map):
    """Route all interrupts to the housekeeping CPUs `hk` (one per NUMA node, indexed by node), and each GPU's
    interrupts to the housekeeping CPU of its node (`gpu_map`: GPU index -> node)."""
    for irq in glob.glob("/proc/irq/[0-9]*"):
        write(f"{irq}/smp_affinity_list", ",".join(hk), check=False)
    # NVIDIA display/3D controllers in PCI bus order, which is nvidia-smi's GPU index order.
    i = 0
    for dev in sorted(glob.glob("/sys/bus/pci/devices/*")):
        try:
            vendor, cls = (open(f"{dev}/{f}").read().strip() for f in ("vendor", "class"))
        except OSError:
            continue
        if vendor != "0x10de" or not cls.startswith("0x03"):
            continue
        node = int(gpu_map[i]) if i < len(gpu_map) else 0
        cpu = hk[node] if node < len(hk) else ""
        msi = f"{dev}/msi_irqs"
        for irq in sorted(os.listdir(msi)) if os.path.isdir(msi) else []:
            write(f"/proc/irq/{irq}/smp_affinity_list", cpu, check=False)
        i += 1


def best_effort(args):
    try:
        subprocess.run(args)
    except OSError:
        pass
'''
_boot: dict = {}
exec(BOOT_CODE, _boot)
write, quiet_memory, disable_deep_cstates, irq_affinity = (
    _boot[k] for k in ("write", "quiet_memory", "disable_deep_cstates", "irq_affinity")
)

DEVNULL = subprocess.DEVNULL


# The report of this run: one entry per log() step, holding the notes and warnings raised while it ran. A step
# with a warning was not (fully) done: a VM may refuse a setting, a unit may not mask, a driver may need a reboot.
STEPS: list[dict] = []


def log(msg: str) -> None:
    print(f"\033[1;34m==>\033[0m {msg}")
    STEPS.append({"step": msg, "notes": [], "blocked": []})


def note(msg: str) -> None:
    """Detail of what the current step did, for the report (not printed as it happens)."""
    if STEPS:
        STEPS[-1]["notes"].append(msg)


def warn(msg: str) -> None:
    """Something the current step could not do: printed now and listed in the report."""
    print(f"\033[1;33m!!\033[0m {msg}", file=sys.stderr)
    if not STEPS:
        STEPS.append({"step": "(setup)", "notes": [], "blocked": []})
    STEPS[-1]["blocked"].append(msg)


def report(outcome: str) -> None:
    """Print what this run did and what it could not do, and save it for "status"."""
    out = [f"tperf host tune apply: {outcome}"]
    done = [s for s in STEPS if not s["blocked"]]
    blocked = [s for s in STEPS if s["blocked"]]
    out.append(f"Done ({len(done)}):")
    for s in done:
        out.append(f"  ok  {s['step']}")
        out += [f"        {n}" for n in s["notes"]]
    out.append(f"Not done or partly done ({len(blocked)}):")
    for s in blocked:
        out.append(f"  !!  {s['step']}")
        out += [f"        {n}" for n in s["notes"]]
        out += [f"        blocked: {b}" for b in s["blocked"]]
    if not blocked:
        out.append("  (none)")
    text = "\n".join(out) + "\n"
    print("== Report")
    print(text, end="")
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_FILE.write_text(text)
    except OSError as e:
        print(f"could not save the report to {REPORT_FILE}: {e}", file=sys.stderr)


def run(*args: str, check: bool = True, quiet: bool = False, stdout=None, stderr=None) -> bool:
    """Run a command; quiet sends stdout and stderr to /dev/null. Returns True on success."""
    if quiet:
        stdout = stderr = DEVNULL
    try:
        rc = subprocess.run(args, stdout=stdout, stderr=stderr).returncode
    except OSError:
        if check:
            raise
        return False
    if check and rc:
        raise subprocess.CalledProcessError(rc, args)
    return rc == 0


def output(*args: str, fallback: str | None = None, stderr=DEVNULL) -> str:
    """Stdout of a command without trailing newlines, like $(...); `fallback` is appended if it fails."""
    try:
        p = subprocess.run(args, stdout=subprocess.PIPE, stderr=stderr, text=True)
        out, rc = p.stdout, p.returncode
    except OSError:
        out, rc = "", 127
    if rc and fallback is not None:
        out += fallback
    return out.rstrip("\n")


def read(path: str, fallback: str | None = None) -> str:
    try:
        return Path(path).read_text().rstrip("\n")
    except OSError:
        if fallback is None:
            raise
        return fallback


def lines(text: str) -> list[str]:
    """The lines of `text`, like a shell `while read` loop sees them."""
    parts = text.split("\n")
    return parts[:-1] if parts[-1] == "" else parts


def indented(text: str) -> None:
    for line in lines(text):
        print(f"   {line}")


def unit_exists(unit: str) -> bool:
    return bool(output("systemctl", "list-unit-files", "--no-legend", unit))


def nvidia_ok() -> bool:
    return bool(shutil.which("nvidia-smi")) and run("nvidia-smi", "-L", check=False, quiet=True)


def has_nvidia_gpu() -> bool:
    return "nvidia" in output("lspci").lower()


def save_unit_state() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if UNIT_STATE.is_file():  # keep the very first snapshot
        return
    for u in ALL_UNITS:
        if unit_exists(u):
            state = f"{u} {output('systemctl', 'is-enabled', u)} {output('systemctl', 'is-active', u)}\n"
            with UNIT_STATE.open("a") as f:
                f.write(state)

    # runtime knobs that are not persisted elsewhere
    thp_enabled = "".join(re.findall(r"\[([a-z]*)\]", read(f"{THP}/enabled")))
    thp_defrag = "".join(re.findall(r"\[([a-z+]*)\]", read(f"{THP}/defrag")))
    RUNTIME_STATE.write_text(f"thp_enabled {thp_enabled}\nthp_defrag {thp_defrag}\nksm_run {read(KSM_RUN, '0')}\n")


def apply_units() -> None:
    log("Stopping and masking services and timers")
    # Freeze snap auto-refresh while snapd is still reachable.
    if shutil.which("snap") and run("systemctl", "is-active", "-q", "snapd.service", check=False):
        if not run("snap", "refresh", "--hold", check=False, quiet=True):
            warn("snap refresh --hold failed")
    masked, disabled, absent = [], [], []
    for u in ALL_UNITS:
        if not unit_exists(u):
            absent.append(u)
            continue
        run("systemctl", "stop", u, check=False, stderr=DEVNULL)
        # Snap units live in /etc/systemd/system and cannot be masked: disable them instead.
        if run("systemctl", "mask", u, check=False, quiet=True):
            masked.append(u)
        elif run("systemctl", "disable", u, check=False, quiet=True):
            disabled.append(u)
        else:
            warn(f"could not mask or disable {u}")
            continue
        if run("systemctl", "is-active", "-q", u, check=False):
            warn(f"{u} is still running after stop")
    run("systemctl", "daemon-reload")
    note(f"masked {len(masked)}: {' '.join(masked) or '-'}")
    if disabled:
        note(f"disabled (cannot be masked) {len(disabled)}: {' '.join(disabled)}")
    note(f"not installed {len(absent)}: {' '.join(absent) or '-'}")


def apply_apt() -> None:
    log("Disabling apt periodic jobs")
    APT_FILE.write_text(
        "// Written by tperf/host/prepare.py\n"
        'APT::Periodic::Update-Package-Lists "0";\n'
        'APT::Periodic::Download-Upgradeable-Packages "0";\n'
        'APT::Periodic::AutocleanInterval "0";\n'
        'APT::Periodic::Unattended-Upgrade "0";\n'
    )


def apply_sysctl() -> None:
    log(f"Applying kernel sysctls ({SYSCTL_FILE})")
    SYSCTL_FILE.write_text(
        "# Written by tperf/host/prepare.py\n"
        "# No automatic NUMA page migration: pin with numactl instead.\n"
        "kernel.numa_balancing = 0\n"
        "# Deterministic memory layout across runs.\n"
        "kernel.randomize_va_space = 0\n"
        "# No soft/hard lockup watchdog timers.\n"
        "kernel.watchdog = 0\n"
        "kernel.nmi_watchdog = 0\n"
        "# Update VM stats every 10s instead of every 1s.\n"
        "vm.stat_interval = 10\n"
    )
    if run("sysctl", "-q", "-e", "-p", str(SYSCTL_FILE), check=False, stderr=DEVNULL):
        return
    # Some VMs refuse a key outright (kernel.nmi_watchdog: errno 524 on g5.4xlarge, which has no NMI watchdog to
    # stop): fine when it already holds the wanted value, an error otherwise.
    for line in SYSCTL_FILE.read_text().splitlines():
        key, sep, want = (part.strip() for part in line.partition("="))
        if not sep or key.startswith("#"):
            continue
        if run("sysctl", "-q", "-w", f"{key}={want}", check=False, stderr=DEVNULL):
            continue
        have = output("sysctl", "-n", key, fallback="n/a")
        if have != want:
            raise SystemExit(f"sysctl {key}: cannot set {want} (is {have})")
        warn(f"sysctl {key}: the kernel refuses writes (already {want}, so no effect on noise)")


def apply_runtime() -> None:
    log("Setting transparent huge pages to madvise, disabling KSM")
    try:
        quiet_memory()
    except OSError as e:
        warn(f"THP/KSM: {e}")
    if not os.access(KSM_RUN, os.W_OK):
        note("KSM: not available on this kernel")
    if not KEEP_CSTATES:
        log("Disabling idle states with exit latency > 10us")
        states = glob.glob("/sys/devices/system/cpu/cpu*/cpuidle/state*")
        if not states:
            warn("no cpuidle states exposed (the hypervisor owns idle): C-states cannot be controlled here")
        try:
            disable_deep_cstates()
        except OSError as e:
            warn(f"C-states: {e}")
    log("Dropping page cache")
    os.sync()
    try:
        write("/proc/sys/vm/drop_caches", 3)
    except OSError as e:
        warn(f"drop_caches: {e}")


def apply_storage() -> None:
    """Mount one local NVMe instance-store disk at CACHE_MOUNT for HF_HOME (model downloads).

    A single disk, not RAID0: mdadm would bring its own monitor daemon and periodic check timers.
    Instance-store data is lost on instance stop/start (not on reboot); the disk is reformatted
    by the next "apply" in that case.
    """
    if CACHE_MOUNT == "none":
        return
    if run("mountpoint", "-q", CACHE_MOUNT, check=False):
        log(f"{CACHE_MOUNT} already mounted ({output('findmnt', '-no', 'SOURCE', CACHE_MOUNT)})")
    else:
        dev = output("blkid", "-L", CACHE_LABEL)
        if not dev:
            disks = subprocess.run(["lsblk", "-dnpo", "NAME,MODEL"], stdout=subprocess.PIPE, text=True, check=True)
            dev = next((line.split()[0] for line in lines(disks.stdout) if "Instance Storage" in line), "")
            if not dev:
                warn(f"no NVMe instance-store disk found: skipping {CACHE_MOUNT}")
                return
            if any("".join(output("lsblk", "-no", col, dev, stderr=None).split()) for col in ("FSTYPE", "PARTTYPE")):
                warn(f"{dev} already has a filesystem or partitions: not formatting it, skipping {CACHE_MOUNT}")
                return
            log(f"Formatting {dev} (xfs, label {CACHE_LABEL})")
            run("mkfs.xfs", "-q", "-L", CACHE_LABEL, dev)
        os.makedirs(CACHE_MOUNT, exist_ok=True)
        log(f"Mounting {dev} at {CACHE_MOUNT}")
        run("mount", "-o", "noatime", dev, CACHE_MOUNT)
    if f"LABEL={CACHE_LABEL}" not in Path("/etc/fstab").read_text():
        with open("/etc/fstab", "a") as f:
            f.write(f"LABEL={CACHE_LABEL} {CACHE_MOUNT} xfs noatime,nofail,x-systemd.device-timeout=10s 0 2\n")
    owner = os.environ.get("SUDO_USER") or "ubuntu"
    shutil.chown(CACHE_MOUNT, owner, owner)


def apt_install(pkgs: list[str], hide_packagekit: bool = False) -> None:
    p = subprocess.run(
        ["apt-get", "install", "-y", "-q", "--no-install-recommends", *pkgs],
        stdout=DEVNULL,
        stderr=subprocess.PIPE if hide_packagekit else None,
        text=True,
    )
    if hide_packagekit:
        # packagekit is masked; its apt hook complains harmlessly.
        sys.stderr.write("".join(line for line in p.stderr.splitlines(True) if "packagekit" not in line))
    if p.returncode:
        raise subprocess.CalledProcessError(p.returncode, p.args)
    run("apt-mark", "hold", *pkgs, stdout=DEVNULL)


def install_gpu_driver() -> bool:
    """Install and load the NVIDIA driver if needed. Returns True when a reboot is required first."""
    branch = os.environ.get("NVIDIA_DRIVER") or "615-server-open"
    if not has_nvidia_gpu() or branch == "none":
        return False
    util = branch.removesuffix("-open")  # utils packages have no -open suffix
    os.environ["DEBIAN_FRONTEND"] = "noninteractive"
    if not nvidia_ok():
        kernel = os.uname().release
        pkgs = [f"linux-modules-nvidia-{branch}-{kernel}", f"nvidia-utils-{util}", f"nvidia-compute-utils-{util}"]
        log(f"Installing NVIDIA driver {branch} ({' '.join(pkgs)})")
        run("apt-get", "update", "-q", stdout=DEVNULL)
        apt_install(pkgs)
    # nvidia-persistenced needs libnvidia-cfg.so.1, which is only a "recommends" of the utils.
    cfg = f"libnvidia-cfg1-{util}"
    if not run("dpkg", "-s", cfg, check=False, quiet=True):
        log(f"Installing {cfg} (needed by nvidia-persistenced)")
        apt_install([cfg], hide_packagekit=True)
    # nouveau grabs the GPUs at boot and prevents the nvidia module from binding.
    blacklist = Path("/etc/modprobe.d/blacklist-nouveau.conf")
    if not blacklist.is_file():
        log("Blacklisting nouveau")
        blacklist.write_text("blacklist nouveau\noptions nouveau modeset=0\n")
        if not run("update-initramfs", "-u", check=False, quiet=True):
            warn("update-initramfs failed")
    # Hot-swapping nouveau for nvidia is unreliable: require a reboot instead.
    if any(line.startswith("nouveau") for line in lines(read("/proc/modules"))):
        return True
    if not run("modprobe", "nvidia", check=False):
        warn("could not load nvidia module")
    run("modprobe", "nvidia_uvm", check=False, stderr=DEVNULL)  # needed by CUDA
    # Freeze the running kernel so the matched driver modules stay valid.
    kernel = os.uname().release
    holds = [f"linux-image-{kernel}", f"linux-modules-{kernel}", "linux-aws", "linux-image-aws", "linux-headers-aws"]
    run("apt-mark", "hold", *holds, check=False, quiet=True)
    return False


def apply_gpu() -> tuple[str, str] | None:
    """Persistence mode and clock locks. Returns the (SM, memory) clocks locked, if any."""
    if not nvidia_ok():
        if has_nvidia_gpu():
            log("GPU tuning")
            warn("NVIDIA GPUs present but driver not working: skipping GPU tuning")
        return None
    log("Starting nvidia-persistenced in persistence mode (keeps GPUs initialised between runs)")
    # Ubuntu's unit is static and runs with --no-persistence-mode: override the command line.
    # bench-tuning.service Wants= it, which starts it at boot.
    PERSISTENCED_DROPIN.parent.mkdir(parents=True, exist_ok=True)
    PERSISTENCED_DROPIN.write_text(
        "# Written by tperf/host/prepare.py\n"
        "[Service]\n"
        "ExecStart=\n"
        "ExecStart=/usr/bin/nvidia-persistenced --user nvidia-persistenced --persistence-mode --verbose\n"
    )
    run("systemctl", "daemon-reload")
    run("systemctl", "unmask", "nvidia-persistenced.service", check=False, quiet=True)
    if not run("systemctl", "restart", "nvidia-persistenced.service", check=False):
        warn("nvidia-persistenced failed to start")
    modes = output("nvidia-smi", "--query-gpu=persistence_mode", "--format=csv,noheader", stderr=None)
    if any("Enabled" not in line for line in lines(modes)):
        warn("persistence mode not enabled on all GPUs")

    mode = os.environ.get("GPU_SM_CLOCK") or "1500"
    if mode == "off":
        run("nvidia-smi", "-rgc", stdout=DEVNULL)
        run("nvidia-smi", "-rmc", stdout=DEVNULL)
        log("GPU clocks unlocked")
        return None

    def max_clock(kind: str) -> str:
        args = ["nvidia-smi", "-i", "0", f"--query-gpu=clocks.max.{kind}", "--format=csv,noheader,nounits"]
        return subprocess.run(args, stdout=subprocess.PIPE, text=True, check=True).stdout.rstrip("\n")

    clk = max_clock("sm") if mode == "max" else mode
    log(f"Locking GPU SM clocks to {clk} MHz on all GPUs")
    if not run("nvidia-smi", "-lgc", f"{clk},{clk}", check=False, stdout=DEVNULL):
        warn("nvidia-smi -lgc failed")
    # Memory clock: lock to the max so it never drops to idle clocks between iterations.
    memclk = max_clock("memory")
    log(f"Locking GPU memory clocks to {memclk} MHz on all GPUs")
    if not run("nvidia-smi", "-lmc", f"{memclk},{memclk}", check=False, stdout=DEVNULL):
        warn("nvidia-smi -lmc failed")
    return clk, memclk


def apply_irqs() -> list[str]:
    """Route device interrupts to housekeeping CPUs (the first CPU of each NUMA node), away from the cores
    benchmarks are pinned to (tperf runs skip each node's first core). Each GPU's interrupts go to the
    housekeeping CPU of its own node. irqbalance is masked, so this stays put. Returns the housekeeping CPUs."""
    cpus = subprocess.run(["lscpu", "-p=CPU,NODE"], stdout=subprocess.PIPE, text=True, check=True).stdout
    hk: dict[str, str] = {}  # NUMA node -> its first CPU, in lscpu order
    for line in lines(cpus):
        if not line.startswith("#"):
            cpu, node = line.split(",")[:2]
            hk.setdefault(node, cpu)
    log(f"Routing device interrupts to housekeeping CPUs: {' '.join(hk.values())} (GPU -> node map: {GPU_NUMA_MAP})")
    irq_affinity(list(hk.values()), GPU_NUMA_MAP.split())
    report_unmoved_irqs(set(hk.values()))
    if ISOLATE_MANAGED_IRQS:
        all_cpus = [line.split(",")[0] for line in lines(cpus) if not line.startswith("#")]
        apply_isolcpus([int(c) for c in all_cpus if c not in hk.values()])
    return list(hk.values())


def cpu_ranges(cpus: list[int]) -> str:
    """[1, 2, 3, 5] -> "1-3,5"."""
    out, cpus = [], sorted(cpus)
    for k, group in itertools.groupby(enumerate(cpus), lambda ic: ic[1] - ic[0]):
        run_ = [c for _, c in group]
        out.append(f"{run_[0]}-{run_[-1]}" if len(run_) > 1 else str(run_[0]))
    return ",".join(out)


def apply_isolcpus(bench_cpus: list[int]) -> None:
    """Add isolcpus=managed_irq,<benchmark CPUs> to the kernel command line (GRUB drop-in), so the kernel puts
    managed interrupts on a housekeeping CPU whenever their mask allows it. managed_irq only: the default "domain"
    flag would take the CPUs out of the scheduler, and threads pinned to several isolated CPUs would then never
    spread. ("io_queue", which would stop the block layer from creating queues on them, is not in 7.0 kernels.)"""
    if not bench_cpus:
        return
    arg = f"isolcpus=managed_irq,{cpu_ranges(bench_cpus)}"
    log(f"Kernel command line: {arg}")
    want = f'# Written by tperf/host/prepare.py\nGRUB_CMDLINE_LINUX="$GRUB_CMDLINE_LINUX {arg}"\n'
    if read(str(GRUB_FILE), "") + "\n" != want:
        GRUB_FILE.parent.mkdir(parents=True, exist_ok=True)
        GRUB_FILE.write_text(want)
        if not run("update-grub", check=False, quiet=True):
            warn(f"update-grub failed: {arg} will not be on the next boot")
            return
        note(f"wrote {GRUB_FILE}")
    if arg in read("/proc/cmdline").split():
        note("active on the running kernel")
    else:
        warn(f"reboot required for {arg} to take effect")


def report_unmoved_irqs(hk: set[str]) -> None:
    """Report device interrupts left on the benchmark CPUs. Two cases, told apart by requested vs effective affinity:
    refused, when the kernel manages the interrupt (one per queue per CPU, e.g. NVMe queues: the write fails with
    EIO and irq_affinity ignores it); pending, when the new affinity was accepted but x86 only moves an interrupt at
    its next delivery."""
    names = {}
    for line in lines(read("/proc/interrupts")):
        irq, _, rest = line.partition(":")
        if irq.strip().isdigit() and rest.split():
            names[irq.strip()] = rest.split()[-1]
    refused: dict[str, list[str]] = {}
    pending: list[str] = []
    for irq, name in names.items():
        requested = read(f"/proc/irq/{irq}/smp_affinity_list", "")
        effective = read(f"/proc/irq/{irq}/effective_affinity_list", "")
        if requested not in hk and requested != f"0-{os.cpu_count() - 1}":  # all CPUs: not a device IRQ we route
            # Managed: the kernel picks the CPU. isolcpus=managed_irq makes it a housekeeping CPU when the mask has one
            if effective not in hk:
                refused.setdefault(name, []).append(requested)
        elif effective and effective not in hk and requested in hk:
            pending.append(name)
    if pending:
        note(f"{len(pending)} interrupts move at their next delivery: {' '.join(sorted(set(pending)))}")
    if refused:
        detail = " ".join(sorted(refused))
        warn(
            f"{sum(map(len, refused.values()))} kernel-managed interrupts stay on the benchmark CPUs: their queue is"
            f" bound to benchmark CPUs only, so they fire on that CPU's own I/O (model loading), not the timed loop."
            f" isolcpus=managed_irq cannot move them; fewer queues would need a kernel with isolcpus io_queue: {detail}"
        )


def apply_boot_unit(hk: list[str], clocks: tuple[str, str] | None) -> None:
    """Runtime-only settings (THP, KSM, C-states, GPU clock lock) are lost on reboot.
    Install a oneshot unit that re-applies them at boot, so a rebooted box is never half-tuned."""
    log(f"Installing {BOOT_UNIT} (re-applies runtime settings at boot)")
    calls = [f"irq_affinity({hk!r}, {GPU_NUMA_MAP.split()!r})", "quiet_memory(check=False)"]
    if not KEEP_CSTATES:
        calls.append("disable_deep_cstates(check=False)")
    if clocks:
        calls += [f"best_effort({['nvidia-smi', opt, f'{c},{c}']!r})" for opt, c in zip(("-lgc", "-lmc"), clocks)]
    BOOT_SCRIPT.write_text(
        "#!/usr/bin/python3\n"
        "# Generated by tperf/host/prepare.py: re-applies runtime-only benchmark settings at boot.\n"
        f"{BOOT_CODE}\n\n" + "".join(f"{c}\n" for c in calls)
    )
    BOOT_SCRIPT.chmod(0o755)
    OLD_BOOT_SCRIPT.unlink(missing_ok=True)
    Path(f"/etc/systemd/system/{BOOT_UNIT}").write_text(
        "# Generated by tperf/host/prepare.py\n"
        "[Unit]\n"
        "Description=Re-apply runtime benchmark tuning (THP, KSM, C-states, GPU clocks)\n"
        "After=nvidia-persistenced.service systemd-modules-load.service\n"
        "Wants=nvidia-persistenced.service\n"
        "\n"
        "[Service]\n"
        "Type=oneshot\n"
        "RemainAfterExit=yes\n"
        f"ExecStart={BOOT_SCRIPT}\n"
        "\n"
        "[Install]\n"
        "WantedBy=multi-user.target\n"
    )
    run("systemctl", "daemon-reload")
    if not run("systemctl", "enable", BOOT_UNIT, check=False, quiet=True):
        warn(f"could not enable {BOOT_UNIT}")


def revert() -> None:
    if not UNIT_STATE.is_file():
        sys.exit(f"no saved state in {STATE_DIR}: nothing to revert")
    log("Restoring services and timers")
    for line in lines(UNIT_STATE.read_text()):
        u, enabled, active = (line.split(maxsplit=2) + ["", ""])[:3]
        run("systemctl", "unmask", u, check=False, quiet=True)
        if enabled == "enabled":
            run("systemctl", "enable", u, check=False, quiet=True)
        if active.strip() == "active":
            run("systemctl", "start", u, check=False, stderr=DEVNULL)
    run("systemctl", "daemon-reload")
    if shutil.which("snap"):
        run("snap", "refresh", "--unhold", check=False, quiet=True)

    log("Removing sysctl and apt overrides")
    SYSCTL_FILE.unlink(missing_ok=True)
    if GRUB_FILE.is_file():
        GRUB_FILE.unlink()
        run("update-grub", check=False, quiet=True)
        log(f"Removed {GRUB_FILE}: the kernel command line is back to the default at the next reboot")
    APT_FILE.unlink(missing_ok=True)
    run("sysctl", "-q", "--system", check=False, quiet=True)

    log("Restoring runtime settings")
    if RUNTIME_STATE.is_file():
        for line in lines(RUNTIME_STATE.read_text()):
            k, v = (line.split(maxsplit=1) + ["", ""])[:2]
            if k == "thp_enabled":
                write(f"{THP}/enabled", v)
            elif k == "thp_defrag":
                write(f"{THP}/defrag", v)
            elif k == "ksm_run" and os.access(KSM_RUN, os.W_OK):
                write(KSM_RUN, v)
    for st in glob.glob("/sys/devices/system/cpu/cpu*/cpuidle/state*/disable"):
        write(st, 0)
    if shutil.which("nvidia-smi"):
        run("nvidia-smi", "-rgc", check=False, quiet=True)
        run("nvidia-smi", "-rmc", check=False, quiet=True)
    run("systemctl", "disable", BOOT_UNIT, check=False, quiet=True)
    for f in (Path(f"/etc/systemd/system/{BOOT_UNIT}"), BOOT_SCRIPT, OLD_BOOT_SCRIPT, PERSISTENCED_DROPIN):
        f.unlink(missing_ok=True)
    run("systemctl", "daemon-reload")
    # The NVIDIA driver and the package holds are left in place on purpose.

    shutil.rmtree(STATE_DIR)
    log("Reverted")


def status(last_report: bool = True) -> None:
    loadavg = " ".join(read("/proc/loadavg").split(" ")[:3])
    print(f"== Host:    {socket.gethostname()}  kernel {os.uname().release}  {len(os.sched_getaffinity(0))} CPUs")
    print(f"== Load:   {loadavg}")
    print("== Running services:")
    units = output(
        "systemctl", "list-units", "--type=service", "--state=running", "--no-pager", "--no-legend", stderr=None
    )
    for line in lines(units):
        print(f"   {(line.split() or [''])[0]}")
    print("== Active timers:")
    for line in lines(output("systemctl", "list-timers", "--no-pager", "--no-legend", stderr=None)):
        print(f"   {(line.split() or [''])[-1]}")
    print("== Kernel:")
    for k in ("kernel.numa_balancing", "kernel.randomize_va_space", "kernel.watchdog", "kernel.nmi_watchdog",
              "vm.stat_interval"):  # fmt: skip
        print(f"   {k:<28} {output('sysctl', '-n', k, fallback='n/a')}")
    print(f"   {'thp.enabled':<28} {read(f'{THP}/enabled')}")
    print(f"   {'thp.defrag':<28} {read(f'{THP}/defrag')}")
    print(f"   {'ksm.run':<28} {read(KSM_RUN, 'n/a')}")
    print(f"   {BOOT_UNIT:<28} {output('systemctl', 'is-enabled', BOOT_UNIT, fallback='missing')}")
    if last_report and REPORT_FILE.is_file():
        print("== Last apply:")
        indented(REPORT_FILE.read_text())
    print("== Idle states (cpu0):")
    for st in sorted(glob.glob("/sys/devices/system/cpu/cpu0/cpuidle/state*")):
        name, latency, disabled = (read(f"{st}/{f}") for f in ("name", "latency", "disable"))
        print(f"   {name:<6} latency={latency:<5} disabled={disabled}")
    print("== Device interrupts by effective CPU (irq count):")
    affinities = []
    for irq in glob.glob("/proc/irq/[0-9]*"):
        try:
            affinities += lines(Path(f"{irq}/effective_affinity_list").read_text())
        except OSError:
            pass
    # Like `sort -n | uniq -c`: by leading number, then by text
    affinities.sort(key=lambda s: (int(re.match(r"\s*(\d*)", s)[1] or 0), s))
    for cpus, group in itertools.groupby(affinities):
        print(f"   cpu {(cpus.split() or [''])[0]:<6} {len(list(group))}")
    print("== NUMA:")
    for line in lines(output("lscpu", stderr=None)):
        if re.match(r"NUMA node[0-9]", line):
            print(f"   {line}")
    print("== GPU:")
    if nvidia_ok():
        counters = ("sw_power_cap", "sw_thermal_slowdown", "hw_thermal_slowdown", "hw_power_brake_slowdown")
        query = (
            "index,name,driver_version,persistence_mode,clocks.sm,clocks.max.sm,power.limit,temperature.gpu,"
            "ecc.mode.current"
        )
        indented(output("nvidia-smi", f"--query-gpu={query}", "--format=csv", stderr=None))
        print("   throttle counters since driver load (us):")
        query = ",".join(["index", *(f"clocks_event_reasons_counters.{c}" for c in counters)])
        indented(output("nvidia-smi", f"--query-gpu={query}", "--format=csv", stderr=None))
        held = subprocess.run(["apt-mark", "showhold"], stdout=subprocess.PIPE, text=True).stdout.replace("\n", " ")
        print(f"   held packages: {held}")
    else:
        print("   nvidia-smi not found")
    print("== Top CPU consumers:")
    ps = output("ps", "-eo", "pid,user,pcpu,comm", "--sort=-pcpu", stderr=None)
    indented("\n".join(lines(ps)[:8]))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Quiet an Ubuntu host so benchmarks run with minimal noise.")
    p.add_argument("action", nargs="?", default="apply", choices=["apply", "status", "revert"])
    action = p.parse_args(argv).action
    # Piped over ssh, stdout is a pipe: flush each line so it interleaves with the commands' own output.
    sys.stdout.reconfigure(line_buffering=True)
    if action == "status":
        return status()
    if os.geteuid() != 0:
        sys.exit("must run as root (use sudo)")
    if action == "revert":
        try:
            return revert()
        except subprocess.CalledProcessError as e:
            sys.exit(f"command failed (exit {e.returncode}): {shlex.join(map(str, e.cmd))}")
    try:
        save_unit_state()
        reboot_required = install_gpu_driver()  # before masking units: needs apt
        apply_storage()
        apply_units()
        apply_apt()
        apply_sysctl()
        apply_runtime()
        if reboot_required:
            log("GPU tuning, interrupt routing, boot unit")
            warn("nouveau is bound to the GPUs: reboot now, then run apply again to finish GPU setup")
            report("stopped, reboot required")
            return
        clocks = apply_gpu()
        hk = apply_irqs()  # after the GPU driver has allocated its interrupts
        apply_boot_unit(hk, clocks)
    except (subprocess.CalledProcessError, SystemExit) as e:
        msg = (
            f"command failed (exit {e.returncode}): {shlex.join(map(str, e.cmd))}"
            if isinstance(e, subprocess.CalledProcessError)
            else str(e.code)
        )
        warn(msg)
        report("FAILED, the steps after this one did not run")
        sys.exit(1)
    print("== Current state:")
    status(last_report=False)
    report("complete" if not any(s["blocked"] for s in STEPS) else "complete, with steps not done (see below)")


if __name__ == "__main__":
    main()
