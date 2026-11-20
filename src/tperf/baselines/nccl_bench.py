"""Time NCCL collectives the way tensor parallelism uses them: one process per GPU, torch's own NCCL.

Launched once per rank by tperf.baselines.nccl_launch (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT, BENCH_GPU set).
Rank 0 prints one JSON line per (op, size) on stdout:

    {"op": "all_reduce", "bytes": 8192, "world": 2, "gpus": "0,1", "n_ops": 20,
     "us_median": ..., "us_p10": ..., "us_p90": ..., "algbw_gbs": ..., "busbw_gbs": ...}

Each sample times n_ops back-to-back ops between CUDA events (n_ops shrinks for large messages); the latency of a
sample is the slowest rank's elapsed / n_ops. Nothing else runs on the GPUs while timing.
Ops: all_reduce (sum, bf16: what TP does after the attention and MLP output projections) for every group, the
same captured in a CUDA graph (all_reduce_graph: GPU-side cost only, as with cudagraphs in decode), and send
(rank 0 -> rank 1, one direction) for 2-GPU groups.
"""

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist


def sizes(min_bytes: int, max_bytes: int) -> list[int]:
    out, s = [], min_bytes
    while s <= max_bytes:
        out.append(s)
        s *= 4
    # bf16 decode all-reduce of a 4096-hidden model at batch 1 (Llama-2-7B): 8 KiB; 128-token prefill: 1 MiB
    for extra in (4096 * 2, 128 * 4096 * 2):
        if min_bytes <= extra <= max_bytes and extra not in out:
            out.append(extra)
    return sorted(out)


def time_op(fn, n_ops: int, samples: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    out = []
    for _ in range(samples):
        dist.barrier()
        torch.cuda.synchronize()
        start.record()
        for _ in range(n_ops):
            fn()
        end.record()
        end.synchronize()
        out.append(start.elapsed_time(end) * 1000 / n_ops)  # us per op
    # slowest rank per sample
    t = torch.tensor(out, device="cuda", dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.tolist()


def time_graph(fn, n_ops: int, samples: int, warmup: int) -> list[float]:
    """Like time_op, but the n_ops ops are captured once in a CUDA graph and each sample replays it: no per-op
    Python / ProcessGroupNCCL launch cost, as in a decode step under torch.compile(mode="reduce-overhead")."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):  # eager warm-up on a side stream before capture
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(n_ops):
            fn()
    torch.cuda.synchronize()
    us = time_op(graph.replay, 1, samples, warmup)
    return [u / n_ops for u in us]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--min-bytes", type=int, default=8)
    p.add_argument("--max-bytes", type=int, default=256 * 2**20)
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--gpus", default="", help="GPU list of the group, recorded in the output")
    args = p.parse_args(argv)

    gpu = int(os.environ["BENCH_GPU"])
    torch.cuda.set_device(gpu)
    dist.init_process_group("nccl", device_id=torch.device("cuda", gpu))
    rank, world = dist.get_rank(), dist.get_world_size()

    # First collective builds the communicator (and logs the transports with NCCL_DEBUG=INFO): keep it untimed.
    dist.all_reduce(torch.ones(1, device="cuda"))
    torch.cuda.synchronize()

    def emit(op: str, nbytes: int, n_ops: int, us: list[float], bus_factor: float) -> None:
        if rank != 0:
            return
        med = statistics.median(us)
        q = statistics.quantiles(us, n=10)
        algbw = nbytes / (med * 1e-6) / 1e9
        print(
            json.dumps(
                {
                    "op": op,
                    "bytes": nbytes,
                    "world": world,
                    "gpus": args.gpus,
                    "n_ops": n_ops,
                    "us_median": round(med, 2),
                    "us_p10": round(q[0], 2),
                    "us_p90": round(q[-1], 2),
                    "algbw_gbs": round(algbw, 3),
                    "busbw_gbs": round(algbw * bus_factor, 3),
                }
            ),
            flush=True,
        )

    for nbytes in sizes(args.min_bytes, args.max_bytes):
        n_ops = max(1, min(20, (64 * 2**20) // nbytes))
        numel = max(1, nbytes // 2)
        buf = torch.ones(numel, device="cuda", dtype=torch.bfloat16)

        us = time_op(lambda: dist.all_reduce(buf), n_ops, args.samples, args.warmup)
        emit("all_reduce", numel * 2, n_ops, us, 2 * (world - 1) / world)
        us = time_graph(lambda: dist.all_reduce(buf), n_ops, args.samples, args.warmup)
        emit("all_reduce_graph", numel * 2, n_ops, us, 2 * (world - 1) / world)

        if world == 2:
            peer = 1 - rank

            def send() -> None:
                if rank == 0:
                    dist.send(buf, peer)
                else:
                    dist.recv(buf, peer)

            us = time_op(send, n_ops, args.samples, args.warmup)
            emit("send", numel * 2, n_ops, us, 1.0)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
