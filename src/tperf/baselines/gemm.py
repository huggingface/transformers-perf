"""Repeated cuBLAS GEMM throughput measurement on one GPU, for noise baselines.

Runs on the bench host inside /opt/bench/venv (launched by `tperf baselines run`). Select the GPU with
CUDA_VISIBLE_DEVICES. Prints one JSON object per dtype on stdout.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/opt/bench/src /opt/bench/venv/bin/python -m tperf.baselines.gemm --dtype fp16 bf16
"""

import argparse
import json
import statistics
import time

import torch

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def bench(dtype_name: str, n: int, iters: int, reps: int, warmup_s: float) -> dict:
    torch.backends.cuda.matmul.allow_tf32 = False  # fp32 means real fp32
    dev = torch.device("cuda:0")
    gen = torch.Generator(device=dev).manual_seed(0)
    a = torch.randn(n, n, device=dev, dtype=DTYPES[dtype_name], generator=gen)
    b = torch.randn(n, n, device=dev, dtype=DTYPES[dtype_name], generator=gen)
    c = torch.empty(n, n, device=dev, dtype=DTYPES[dtype_name])

    # Warm up long enough to reach steady-state clocks, power and temperature.
    deadline = time.monotonic() + warmup_s
    while time.monotonic() < deadline:
        for _ in range(iters):
            torch.mm(a, b, out=c)
        torch.cuda.synchronize()

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    flop = 2 * n**3 * iters
    tflops = []
    for _ in range(reps):
        start.record()
        for _ in range(iters):
            torch.mm(a, b, out=c)
        end.record()
        end.synchronize()
        tflops.append(flop / (start.elapsed_time(end) / 1e3) / 1e12)

    mean = statistics.fmean(tflops)
    return {
        "bench": "gemm",
        "dtype": dtype_name,
        "n": n,
        "iters_per_rep": iters,
        "reps": reps,
        "tflops_median": statistics.median(tflops),
        "tflops_mean": mean,
        "tflops_min": min(tflops),
        "tflops_max": max(tflops),
        "cov_pct": 100 * statistics.stdev(tflops) / mean,
        "samples": tflops,
        "device": torch.cuda.get_device_name(dev),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", nargs="+", default=["bf16"], choices=list(DTYPES))
    p.add_argument("--n", type=int, default=8192)
    p.add_argument("--iters", type=int, default=20, help="GEMMs per timed repetition")
    p.add_argument("--reps", type=int, default=30, help="timed repetitions")
    p.add_argument("--warmup", type=float, default=30.0, help="warm-up seconds per dtype")
    args = p.parse_args(argv)
    for d in args.dtype:
        print(json.dumps(bench(d, args.n, args.iters, args.reps, args.warmup)), flush=True)


if __name__ == "__main__":
    main()
