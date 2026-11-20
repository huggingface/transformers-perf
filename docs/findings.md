# Findings

Measurements on the bench host (AWS g5.48xlarge, 8× A10G) that shaped how the benchmarks run.

## GPU to GPU (NCCL)

Peer-to-peer is not supported between any of the 8 GPUs (`nvidia-smi topo -m`: PHB everywhere, nvbandwidth waives its
device-to-device tests), so NCCL stages all GPU<->GPU traffic through host shared memory (`SHM/direct`).
`tperf baselines p2p` measures what tensor parallelism gets from that: `tperf.baselines.nccl_bench` times bf16
`all_reduce` (and `send` for pairs) from 8 B to 256 MiB with torch's NCCL, one process per GPU pinned to its NUMA node
(`tperf.baselines.nccl_launch`, which holds the daily lock), for a pair on one node, a pair across nodes, TP4 on each
node and TP8, and `tperf baselines summarize-p2p` prints latency, bus bandwidth, run-to-run spread and NCCL's
transports.

2026-11-19, torch 2.15 / NCCL 2.30.7, `results/*-p2p-cumem-off`:

- NCCL's default cuMem host buffers all land on NUMA node 0 (the VM hides the GPUs' node), whatever `--membind` says,
  so GPUs 4-7 got 1.8 GB/s all-reduce bus bandwidth vs 4.7 on GPUs 0-3. `NCCL_CUMEM_HOST_ENABLE=0` (the runners'
  default `NCCL_ENV`) makes them follow each rank's binding: 4-7 then match 0-3. Use it for any multi-GPU run.
- The ceiling is the GPU writing to host memory with SMs: 3.9 GB/s to its own node, 1.0 GB/s across nodes
  (nvbandwidth `device_to_host_memcpy_sm`; copy engines get 13.5 GB/s). Groups spanning both nodes pay for it:
  pair 0,4 at 1.6 GB/s, TP8 at 2.8 GB/s. Keep TP groups inside one node (0-3, 4-7).
- Small messages (decode: 8 KiB per all-reduce for a 4096-hidden model at batch 1): eager `dist.all_reduce` costs
  ~50 us whatever the size, which is CPU launch cost; in a CUDA graph it is 14 us on TP2, 21 us on TP4, 34 us on TP8.
- Within one node, run-to-run spread is under 1% except eager small messages (1-2%, about 1 us).

## Tensor parallelism through harness

`tperf tp run` (worktree of main, `torchrun --enable-tp`, one NUMA node per group, `NCCL_CUMEM_HOST_ENABLE=0`,
`scenarios/tp.jsonl`: sdpa eager and compiled). Llama-2-7B, `-w 5 -i 10 -b 1 -s 128 -n 64`, 3 runs,
`results/20261119-090823-tp`:

| | ITL compiled | ITL eager | run-to-run CoV (ITL / e2e) |
|---|---|---|---|
| TP1 (GPU 0) | 32.3 ms | 35.0 ms | < 0.1% |
| TP2 (GPUs 0-1) | 22.5 ms (1.43x) | 45.3 ms (0.77x) | < 0.7% |
| TP4 (GPUs 0-3) | 14.8 ms (2.19x) | 44.1 ms (0.79x) | < 0.3% |

Compiled TP decode is as stable as single GPU and scales; eager TP is slower than one GPU (CPU-bound). TTFT is not
stable under TP (3-11% CoV between runs).

`tperf tp profile` (`src/tperf/tp/profile.py` on the host, `tperf tp summarize`) breaks one compiled decode step
down per rank: GEMMs scale (29.7 ms TP1, 15.2 TP2, 8.0 TP4), NCCL is ~1 ms, attention 0.85. The gap to ideal is host
time between CUDA graph replays (1.4 ms TP1, 4.1 TP2/TP4), almost all of it dynamo guard evaluation (0.4 ms TP1,
2.9 ms TP2): every sharded module gets a per-instance `tp_forward` closure (guards on its cells and hook dicts) and
DTensor parameters (subclass guards, a Python lambda per param). `--class-forward` / `--local-params` test fixes (same
tokens): TP2 decode 21.97 -> 20.93 -> 19.91 ms (-9.4%), guards 3.0 -> 2.0 -> 0.96 ms. `--method-forward` (a bound
method of one shared function, kept on the instance) only reaches 21.64 ms / 2.7 ms: it is the instance-level
`forward` itself that dynamo pays for. `patches/tp-shared-forward.patch` (the class-level forward, for upstream) A/B
on TP2 (`results/ab-20261119-132843-tp-shared-forward`, 6 rounds): compiled ITL 22.51 -> 21.47 ms (-4.6%,
CI [-4.9, -4.3]), eager unchanged.
