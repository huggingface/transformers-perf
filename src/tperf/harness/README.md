# Benchmarking v2

A comprehensive benchmarking framework for transformer models that supports multiple execution modes (eager, compiled, kernelized), detailed performance metrics collection, and structured output format.


## Workload and execution settings

The main runner uses exactly `--sequence-length` input tokens per request. It truncates the fixed passage for shorter inputs and repeats its tokenized contents for longer inputs. Prompts remain identical across the batch. EOS stopping is explicitly disabled (`eos_token_id=-1`) for fixed-length output measurements.

Continuous batching disables prefix sharing through `ContinuousBatchingConfig` and disables automatic switching to Flash Attention. Eager attention uses its continuous-batching variant, `paged|eager`. Each call creates and warms up a manager; that lifecycle work remains inside API-call latency. Compiled regular generation uses a static cache.

Saved measurements include `execution_settings`: actual input token lengths, attention implementation, dtype, generation settings, and the supplied continuous-batching settings. Unset continuous-batching fields remain subject to manager heuristics; this is not a dump of its resolved runtime configuration. Configuration hashes include the workload policy to distinguish these measurements from earlier runs.

The separate continuous-batching suite inherits the model's generation configuration when no override is supplied (including IFEval), preserving natural EOS stopping. Fixed-length workloads explicitly use `eos_token_id=-1`. Saved configuration summaries retain EOS lists and nested settings.

## Failed workloads

Both runners reject incomplete or failed continuous-batching results before computing measurements or accuracy. Validation checks the request IDs assigned by a fresh manager (including sampled children), prompt tokens, and terminal success status. Accuracy scoring pairs documents with request IDs, independently of dictionary order.

The main runner records setup and generation errors in `metadata.error`, continues with the remaining scenarios, and saves results before raising if none succeeded. Failed scenarios print `FAILED` with the error; empty measurements do not print zero-valued statistics. The continuous-batching suite records setup failures in each entry and includes error status in its summary.

## Quick Start

The harness is part of the `tperf` package (`pip install -e .` at the repository root, or `PYTHONPATH=src`), and runs
against whatever `import transformers` resolves to: put a checkout's `src/` first on `PYTHONPATH` to benchmark it.
On the bench host, `tperf ab`, `tperf generation run`, `tperf tp run` and the daily run launch it for you.

```bash
# The level-1 scenarios on one model (python -m tperf.harness.run_benchmarks takes the same arguments)
tperf harness run --model-id meta-llama/Llama-2-7b-hf -b 1 -s 128 -n 64 -w 5 -i 10 --output-dir my_results

# Scenarios from a config file instead of a coverage level
tperf harness run --model-id google/gemma-3-1b-it -b 1 -s 448 -n 128 --config-file scenarios/sliding-window.jsonl \
    --output-dir my_results

# Tensor parallelism: launch the module with torchrun
torchrun --standalone --nproc-per-node 2 -m tperf.harness.run_benchmarks --enable-tp --model-id <model> -b 1 -s 128 -n 64

# Push the results to a Hugging Face dataset (needs a token with write access, e.g. HF_TOKEN)
tperf harness run --model-id <model> -b 1 -s 128 -n 64 --push-result-to-dataset username/benchmark-results
```

Without `--output-dir`, results go to `benchmark_results/` in the current directory.

### Comparing Against a Baseline

`tperf harness compare` (`python -m tperf.harness.compare_results`) compares a candidate run to a baseline, scenario
by scenario, on end-to-end latency, time to first token, inter-token latency and throughput. A metric that is worse by
more than `--threshold` percent is a regression, and the command then exits with status 1, so it can gate CI.

```bash
# Baseline on main, candidate on a branch, same hardware and arguments
tperf harness run --model-id meta-llama/Llama-2-7b-hf -b 1 -s 128 -n 128 -w 5 -i 30 --output-dir results_main
tperf harness run --model-id meta-llama/Llama-2-7b-hf -b 1 -s 128 -n 128 -w 5 -i 30 --output-dir results_pr

tperf harness compare --baseline results_main --candidate results_pr --threshold 1.0
```

Each side is a results file or a directory of them: with several repeated runs, each metric is the median of the
per-run medians. The report also shows the noise for each metric (the largest within-run or between-run coefficient of
variation) and flags it when it is above the threshold: the comparison cannot resolve a change that small. Pick a
threshold above the noise of your setup, and run the baseline and the candidate on the same machine, GPU and arguments.
Scenarios that failed or only exist on one side are listed separately.

## Output Format

Results are saved as JSON files with the following structure:

```json
{
  "model_name": "llama_2_7b",
  "benchmark_scenarios": [
    {
      "scenario_name": "eager_variant",
      "metadata": {
        "timestamp": "2025-01-XX...",
        "commit_id": "abc123...",
        "hardware_info": {
          "gpu_name": "NVIDIA A100",
          "gpu_memory_total": 40960,
          "cpu_count": 64
        },
        "config": {
          "variant": "eager",
          "warmup_iterations": 3,
          "measurement_iterations": 5
        }
      },
      "measurements": {
        "latency": {
          "mean": 2.45,
          "median": 2.43,
          "std": 0.12,
          "min": 2.31,
          "max": 2.67,
          "p95": 2.61,
          "p99": 2.65
        },
        "time_to_first_token": {
          "mean": 0.15,
          "std": 0.02
        },
        "tokens_per_second": {
          "mean": 87.3,
          "unit": "tokens/sec"
        }
      },
      "gpu_metrics": {
        "gpu_utilization_mean": 85.2,
        "gpu_memory_used_mean": 12450
      }
    }
  ]
}
```

### Debug Mode

```bash
tperf harness run --log-level DEBUG ...
```

## Layout

- `run_benchmarks.py`: the runner (`tperf harness run`); `compare_results.py`: the comparison (`tperf harness compare`)
- `framework/`: scenario configs, the benchmark runner, hardware metrics, result comparison
- `benchmark_scripts/continuous_batching_overall.py`: the separate continuous-batching suite (also needs `lighteval`
  and `tabulate`), run with `python -m tperf.harness.benchmark_scripts.continuous_batching_overall`
- Tests: `tests/harness/` at the repository root
