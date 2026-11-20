# transformers-perf

Performance tracking for [transformers](https://github.com/huggingface/transformers) on a dedicated, noise-tuned GPU
host: daily runs of main, side-by-side runs of PRs, multimodal processor benchmarks and hardware baselines.

- Results: [hf-internal-testing/transformers-perf](https://huggingface.co/datasets/hf-internal-testing/transformers-perf) ([columns](docs/dataset-card.md))
- Dashboard: [hf-internal-testing/transformers-perf-dashboard](https://huggingface.co/spaces/hf-internal-testing/transformers-perf-dashboard)

## Setup

```bash
pip install -e .                       # installs the tperf command
cp tperf.example.toml tperf.toml       # set [host] ssh (tperf.toml is not committed)
tperf host install                     # CUDA, torch, transformers under /opt/bench on the host
tperf host tune apply                  # quiet the host (status / revert)
tperf deploy                           # copy tperf (harness included) and scenarios; install the daily timer
```

## Usage

```bash
MODEL=google/gemma-3-1b-it SCENARIOS=scenarios/sliding-window.jsonl tperf ab 49542   # a PR against its base
tperf processors run                   # processors and encoders
tperf baselines run                    # GEMM and host<->device bandwidth
tperf --help                           # every command
```

`tperf ab` runs base and head at the same time on two GPUs (one per NUMA node), swaps them every round, and reports
each metric's change with a 95% confidence interval. The daily run (03:00 UTC) benchmarks the latest main and
rebuilds the dashboard.

## Layout

- `src/tperf/`: the package, one subpackage per feature (`ab`, `daily`, `processors`, `tp`, `baselines`, ...)
- `src/tperf/harness/`: the generation benchmark (formerly transformers' `benchmark_v2`), the same copy for every
  commit (`tperf harness run`); its tests are in `tests/harness/`
- `scenarios/`: benchmark configs; `patches/`: transformers changes A/B-tested before their PR
- `docs/`: [dataset card](docs/dataset-card.md), [findings](docs/findings.md)

## Development

```bash
python -m pytest && ruff check . && ruff format --check .
```
