#!/usr/bin/env python3
"""Install pinned GPU benchmark tools on the bench host.

Usage: tperf host install        (pipes this file to `ssh HOST sudo python3 -`; standalone, standard library only)

Installs into /opt/bench:
  nvbandwidth/   NVIDIA nvbandwidth (host<->device and device<->device copy bandwidth)
  venv/          Python venv with PyTorch (cuBLAS GEMM throughput)
  transformers/  transformers at TF_COMMIT (the harness comes from this repository: tperf deploy)
  tbv2-venv/     Python venv for harness: same torch pin, transformers editable install
All versions are pinned so baselines stay comparable. Idempotent.
"""

import glob
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


NVBANDWIDTH_TAG = "v0.10.0"
TORCH_VERSION = "2.15.0"
TF_COMMIT = os.environ.get("TF_COMMIT") or "41f9d3992023970fb5a463e70d56232506c80fbc"  # transformers main, 2026-11-13
KERNELS_VERSION = "0.18.2"
CUDA_PKG = "cuda-toolkit-13-1"
PREFIX = Path("/opt/bench")
OWNER = os.environ.get("SUDO_USER") or "ubuntu"
# python3-dev: torch.compile builds C helpers
APT_PKGS = [CUDA_PKG, "cmake", "g++", "make", "libboost-program-options-dev", "numactl", "python3-venv", "python3-dev"]

DEVNULL = subprocess.DEVNULL


def log(msg: str) -> None:
    print(f"\033[1;34m==>\033[0m {msg}", flush=True)


def run(*args: object, check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run([str(a) for a in args], check=check, **kwargs)


def as_owner(*args: object, **kwargs) -> subprocess.CompletedProcess:
    return run("sudo", "-u", OWNER, *args, **kwargs)


def succeeds(*args: object) -> bool:
    return run(*args, check=False, stdout=DEVNULL, stderr=DEVNULL).returncode == 0


def install_apt_packages() -> None:
    missing = [p for p in APT_PKGS if not succeeds("dpkg", "-s", p)]
    if not missing:
        return
    log(f"Installing apt packages: {' '.join(missing)}")
    run("apt-get", "update", "-q", stdout=DEVNULL)
    p = run("apt-get", "install", "-y", "-q", "--no-install-recommends", *missing, check=False, stdout=DEVNULL,
            stderr=subprocess.PIPE, text=True)  # fmt: skip
    # packagekit is masked by tperf/host/prepare.py; its apt hook complains harmlessly.
    sys.stderr.write("".join(line for line in p.stderr.splitlines(True) if "packagekit" not in line))
    p.check_returncode()
    run("apt-mark", "hold", CUDA_PKG, stdout=DEVNULL)


def cuda_home() -> Path:
    candidates = sorted(glob.glob("/usr/local/cuda*/bin/nvcc") + glob.glob("/usr/lib/cuda/bin/nvcc"))
    nvcc = shutil.which("nvcc") or next(iter(candidates), None)
    if not nvcc:
        sys.exit("nvcc not found")
    return Path(nvcc).resolve().parent.parent


def main() -> None:
    if os.geteuid() != 0:
        sys.exit("must run as root (use sudo)")
    os.environ["DEBIAN_FRONTEND"] = "noninteractive"
    install_apt_packages()

    cuda = cuda_home()
    nvcc_version = run(cuda / "bin/nvcc", "--version", stdout=subprocess.PIPE, text=True).stdout
    log(f"CUDA toolkit: {cuda} ({' '.join(re.findall(r'release [0-9.]*', nvcc_version))})")

    PREFIX.mkdir(mode=0o755, exist_ok=True)
    shutil.chown(PREFIX, OWNER, OWNER)

    nvbw = PREFIX / "nvbandwidth"
    if not os.access(nvbw / "nvbandwidth", os.X_OK):
        log(f"Building nvbandwidth {NVBANDWIDTH_TAG}")
        shutil.rmtree(nvbw, ignore_errors=True)
        as_owner("git", "clone", "-q", "--depth", "1", "--branch", NVBANDWIDTH_TAG,
                 "https://github.com/NVIDIA/nvbandwidth.git", nvbw)  # fmt: skip
        path = f"{cuda}/bin:{os.environ.get('PATH', '')}"
        as_owner("env", f"PATH={path}", "cmake", "-S", ".", "-B", ".", "-DCMAKE_BUILD_TYPE=Release",
                 f"-DCMAKE_CUDA_COMPILER={cuda}/bin/nvcc", cwd=nvbw, stdout=DEVNULL)  # fmt: skip
        as_owner("make", f"-j{len(os.sched_getaffinity(0))}", cwd=nvbw, stdout=DEVNULL)

    venv_py = PREFIX / "venv/bin/python"
    torch_ok = f"import torch, sys; sys.exit(torch.__version__.split('+')[0] != '{TORCH_VERSION}')"
    if not os.access(venv_py, os.X_OK) or not succeeds(venv_py, "-c", torch_ok):
        log(f"Installing PyTorch {TORCH_VERSION} into {PREFIX}/venv")
        as_owner("python3", "-m", "venv", PREFIX / "venv")
        as_owner(PREFIX / "venv/bin/pip", "install", "-q", "--upgrade", "pip")
        as_owner(PREFIX / "venv/bin/pip", "install", "-q", f"torch=={TORCH_VERSION}", "numpy")

    # transformers harness, in its own venv (same torch pin as the baselines).
    tf = PREFIX / "transformers"
    head = (tf / ".git").is_dir() and run("git", "-C", tf, "rev-parse", "HEAD", check=False, stdout=subprocess.PIPE,
                                          text=True).stdout.strip()  # fmt: skip
    if head != TF_COMMIT:
        log(f"Fetching transformers {TF_COMMIT}")
        shutil.rmtree(tf, ignore_errors=True)
        as_owner("git", "init", "-q", tf)
        as_owner("git", "-C", tf, "fetch", "-q", "--depth", "1", "https://github.com/huggingface/transformers.git",
                 TF_COMMIT)  # fmt: skip
        as_owner("git", "-C", tf, "-c", "advice.detachedHead=false", "checkout", "-q", "FETCH_HEAD")
        shutil.rmtree(PREFIX / "tbv2-venv", ignore_errors=True)
    tbv2_py = PREFIX / "tbv2-venv/bin/python"
    constraints = PREFIX / "tbv2-constraints.txt"

    def pipv(*args: object) -> None:
        as_owner(PREFIX / "tbv2-venv/bin/pip", "install", "-q", "-c", constraints, *args)

    if not os.access(tbv2_py, os.X_OK):
        log(f"Installing harness environment into {PREFIX}/tbv2-venv")
        as_owner("python3", "-m", "venv", PREFIX / "tbv2-venv")
        constraints.write_text(f"torch=={TORCH_VERSION}\n")
        pipv("--upgrade", "pip")
        pipv(f"torch=={TORCH_VERSION}")
        pipv("-e", f"{tf}[torch]")
        # harness/requirements.txt without amdsmi / pyzes (AMD / Intel monitoring, unused on NVIDIA)
        pipv("numpy", "psutil", "nvidia-ml-py", "datasets", "huggingface_hub")
    # kernels: lets harness run flash_attention_2 (kernel fetched from the Hub on first use)
    # and the kernelized scenarios. transformers at TF_COMMIT requires kernels>=0.18.2,<0.19.
    kernels_ok = f"import importlib.metadata as m, sys; sys.exit(m.version('kernels') != '{KERNELS_VERSION}')"
    if not succeeds(tbv2_py, "-c", kernels_ok):
        log(f"Installing kernels {KERNELS_VERSION}")
        pipv(f"kernels=={KERNELS_VERSION}")

    log("Versions:")
    print(f"   nvbandwidth: {NVBANDWIDTH_TAG}", flush=True)
    run(venv_py, "-c", "import torch; print('   torch:', torch.__version__, 'cuda', torch.version.cuda, 'gpus', "
                       "torch.cuda.device_count())")  # fmt: skip
    run(tbv2_py, "-c", "import torch, transformers; print('   harness env: torch', torch.__version__, "
                       f"'transformers', transformers.__version__, '@ {TF_COMMIT}')")  # fmt: skip
    run(tbv2_py, "-c", "import importlib.metadata as m; print('   kernels:', m.version('kernels'))")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(f"command failed (exit {e.returncode}): {' '.join(map(str, e.cmd))}")
