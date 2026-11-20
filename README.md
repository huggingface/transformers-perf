# transformers-perf

Performance tracking for transformers on a dedicated GPU host: daily runs of main, side-by-side runs of PRs, and
hardware baselines, measured with as little noise as possible and published to a Hugging Face dataset and dashboard.

- Harness: [`harness/`](harness), moved here from transformers with its history.
- Results: [hf-internal-testing/transformers-perf](https://huggingface.co/datasets/hf-internal-testing/transformers-perf) (one row per run and scenario; columns in [`docs/dataset-card.md`](docs/dataset-card.md)).
- Dashboard: [hf-internal-testing/transformers-perf-dashboard](https://huggingface.co/spaces/hf-internal-testing/transformers-perf-dashboard).

## Layout

The tooling is the `tperf` package under `src/tperf/`, one subpackage per feature, with one command line, `tperf`:

| Package | |
|---|---|
| `tperf.cli` | the `tperf <command> [<subcommand>]` entry point: forwards the arguments to one module's `main` |
| `tperf.daily` | the daily main benchmark, run on the host: `main` and its systemd units |
| `tperf.ab` | side-by-side runs of a PR against main: `run`, `compare` |
| `tperf.generation` | repeated runs of main with a stability check: `run`, `compare` |
| `tperf.processors` | multimodal processors and encoders: `bench` (on the host), `run`, `summarize` |
| `tperf.tp` | tensor parallelism: `run` (harness under torchrun), `profile_run` / `profile` (per-rank decode step), `summarize` |
| `tperf.baselines` | hardware baselines: `run` and `gemm` (GEMM, host<->device bandwidth), `p2p`, `nccl_launch` and `nccl_bench` (NCCL between GPUs), `repeat`, `summarize`, `summarize_p2p`, `compare` |
| `tperf.host` | host setup and deployment: `prepare`, `install` (standalone, piped to `sudo python3 -`), `deploy` |
| `tperf.common` | shared code: `benchlib` (ssh helpers), `harness`, `record`, `noise_probe` |
| `tperf.dashboard` | static dashboard built from the dataset: `build`, `template.html` |

At the root:

| Directory | |
|---|---|
| `harness/` | the generation benchmark harness (`run_benchmarks.py`) and its tests |
| `scenarios/` | harness config files (`oneoff/`: kept for reference, not in the daily run) |
| `patches/` | transformers changes A/B-tested before their PR (`HEAD_PATCHES`) |
| `docs/` | dataset card, [findings](docs/findings.md) |

Everything is Python (see [`AGENTS.md`](AGENTS.md)), with no dependency on the workstation side:

```bash
uv tool install -e .     # or: pip install -e .   (`[dashboard]` for tperf dashboard build)
tperf --help             # every command; tperf <command> <subcommand> --help for its arguments
```

Commands run from a workstation, from the repository checkout, over SSH. The bench host and where results are
published come from `tperf.toml`, which is not committed: copy [`tperf.example.toml`](tperf.example.toml) to
`tperf.toml` (or `~/.config/tperf/config.toml`) and set `[host] ssh`. Results land in `results/` (not committed). Each command still runs as
`python -m tperf.<feature>.<module>`.

## Host

AWS g5.48xlarge: 2× AMD EPYC 7R32 (2 NUMA nodes), 8× A10G (PCIe, no P2P), Ubuntu 26.04, NVIDIA 615 server driver.
GPUs 0-3 are on NUMA node 0, GPUs 4-7 on node 1 (the VM hides it; measured with nvbandwidth latency).

```bash
# Tune the host for low noise (idempotent; re-applied at boot by bench-tuning.service). Also: status, revert
tperf host tune apply
# Install CUDA toolkit, nvbandwidth, torch, transformers (pinned commit) and harness deps under /opt/bench
tperf host install
# Gated models: log in once on the host
ssh -t $HOST /opt/bench/tbv2-venv/bin/hf auth login
# Copy the tperf package, the harness and the scenarios to the host, install the daily timer
tperf deploy
```

`apply` masks background services and timers (unattended upgrades, snapd, irqbalance, cron, …), sets kernel knobs (no
NUMA balancing, no ASLR, no watchdogs), disables deep C-states, routes all device interrupts to one housekeeping CPU
per NUMA node, enables GPU persistence mode, locks SM clocks at 1500 MHz (below the A10G power cap) and memory clocks
at max, and formats the NVMe instance store as a cache at `/mnt/cache`. `status` shows all of it. `tperf host tune`
and `tperf host install` pipe `src/tperf/host/prepare.py` and `install.py` (standard library only) to
`ssh $HOST sudo python3 -`, passing on their options from your environment (e.g. `GPU_SM_CLOCK`, `TF_COMMIT`).

On the host, the package lives in `/opt/bench/src/tperf` (rsynced by `tperf deploy`, and by every command that runs
host-side modules, so the host always runs your checkout's version) and runs as
`env PYTHONPATH=/opt/bench/src <python> -m tperf.<module>`: the system python for `tperf.daily.main` and
`tperf.baselines.nccl_launch`, the host's venvs for the others (`/opt/bench/tbv2-venv`; `/opt/bench/venv` for the
GEMM and NCCL baselines).

Every run re-applies the tuning (drops caches, re-locks clocks), is pinned to cores of its GPU's NUMA node, gets fresh
compile caches, and runs under `tperf.common.noise_probe`, which reads counters before and after only (foreign CPU
time on the pinned cores, run-queue wait, context switches, interrupts, GPU utilization, energy, throttling). Nothing samples
during measurements. The daily run holds `/tmp/bench-daily.lock`; the other commands refuse to start while it is held.

## Daily main benchmark

`tperf.daily.main` runs on the host every day at 03:00 UTC (`bench-daily.timer`: the system python, with
`PYTHONPATH=/opt/bench/src`). It benchmarks the latest main 3 times per suite with the harness `tperf deploy` installed
in `/opt/bench/harness`, publishes the rows (`source=main`, `label=daily-<suite>`) and rebuilds the dashboard.
Suites on different NUMA nodes run in parallel:

| Suite | GPU | Model | Arguments |
|---|---|---|---|
| `llama2-7b` | 0 | meta-llama/Llama-2-7b-hf | `--level 1 -w 5 -i 10 -b 1 -s 128 -n 64` |
| `gemma3-1b-sliding` | 4 | google/gemma-3-1b-it | `scenarios/sliding-window.jsonl -w 5 -i 6 -b 1 -s 448 -n 128` |
| `processors` | 0 | multimodal processors (CPU) and encoders (GPU) | `scenarios/processors.jsonl` |

```bash
tperf deploy --run-now   # redeploy and start a run now
ssh $HOST journalctl -u bench-daily -e
```

Redeploy after changing `harness/`, `src/tperf/` or `scenarios/`. Every run records the harness version
(`harness_commit`: the last commit that changed `harness/`, `-dirty` with local changes).

## Side-by-side: a PR against main

```bash
PUBLISH_REPO=hf-internal-testing/transformers-perf MODEL=google/gemma-3-1b-it SCENARIOS=scenarios/sliding-window.jsonl \
  BENCH_ARGS="-w 5 -i 6 -b 1 -s 448 -n 128" tperf ab 49542 6
```

`tperf ab <pr> [rounds]` benchmarks the PR head against its merge base with main (`BASE=<sha>` to override; a merged PR
is compared with main right before it):

- Both commits are checked out as worktrees on the host; one copy of `harness/` drives both sides, each importing
  its own transformers through `PYTHONPATH`.
- Each round runs base and head at the same time on GPUs 0 and 4, one per NUMA node, and the GPUs swap every round so
  chip-to-chip differences cancel out.
- `tperf ab compare results/ab-<id>` fits the per-round head/base ratio of each metric as PR effect + GPU effect and prints the PR's
  effect with a 95% confidence interval and a verdict (REGRESSION, improvement, no change, inconclusive). It exits 1
  on a regression.
- With `PUBLISH_REPO`, every run is published with `ab_id`, `ab_side` and `ab_round`.

Pick a model and `SCENARIOS` that exercise the code the PR changes. Variants:

- `HEAD_PATCHES=<patch> tperf ab main`: a local transformers change against the main tip.
- `HEAD_EXTRA_PATCHES=<patch> tperf ab main`: a change to the harness itself (head gets a patched copy).
- `TP=2` (or 4): each side runs under torchrun with `--enable-tp` on a GPU group inside one NUMA node (`0,1` vs
  `4,5`), with `NCCL_CUMEM_HOST_ENABLE=0` (see [findings](docs/findings.md)).

## Other runs

```bash
MODEL=meta-llama/Llama-2-7b-hf tperf generation run 3 llama2-7b  # runs of main, with a stability check
tperf processors run 3                                           # processors and encoders
tperf tp run                                                     # harness under TP, per GPU group
tperf tp profile                                                 # one TP decode step broken down per rank
tperf baselines run                                              # bf16 GEMM and host<->device bandwidth
tperf baselines repeat 3 label                                   # repeat and compare the baselines
tperf baselines p2p                                              # NCCL all-reduce/send per GPU group
tperf harness version                                            # harness version recorded with each run
```

Each prints its summary and keeps the raw results in `results/<timestamp>-<label>/`; the summaries also run on their
own (`tperf generation compare`, `tperf processors summarize`, `tperf tp summarize`, `tperf baselines summarize`,
`summarize-p2p`, `compare`).

## Recording and publishing

`tperf.common.record` (`tperf record`) runs on the host: `manifest` before each run (git identity of the transformers checkout, model,
arguments, harness version, instance, GPU clocks, driver, kernel, library versions) and `publish` after the runs (one
row per run and scenario with the latency, TTFT, ITL and throughput statistics plus the probe counters, written to
Parquet under `data/` with the raw files under `raw/`; `publish-processors` for the processor suite). Tag manual runs
with `BENCH_SOURCE` (`main`, `pr`, `manual`), `BENCH_PR` and `BENCH_BRANCH`.

The dashboard is built on the host after publishing (`tperf.dashboard.build`); `tperf dashboard build --out <dir>`
builds it locally (needs the `dashboard` extra and a token that reads the dataset).

## Development

```bash
python -m pytest   # harness tests (needs torch and transformers)
ruff check . && ruff format --check .
```
