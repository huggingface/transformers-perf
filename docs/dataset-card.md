---
configs:
- config_name: default
  data_files: data/*/*.parquet
---

# transformers performance tracking

Results of the [transformers-perf](https://github.com/huggingface/transformers-perf) benchmarks of transformers on a
dedicated, tuned host (one GPU per run, clocks locked, CPUs pinned on the GPU's NUMA node, background services
masked). Each run is published by `tperf record publish` (generation) or `tperf record publish-processors`.

## Layout

- `data/<YYYY-MM>/<run_id>.parquet`: one row per (run, scenario)
- `processors/<YYYY-MM>/<group>.parquet`: multimodal processor and encoder runs, one row per (run, scenario)
- `raw/<YYYY-MM>/<run_id>/`: the files the rows were built from (`manifest.json`, `probe.json`, harness results)

A *run* is one harness (`tperf.harness.run_benchmarks`) invocation. Runs repeated back to back share a `group_id`.

## Columns

| Column(s) | Meaning |
|---|---|
| `run_id`, `group_id`, `run_index`, `label`, `started_at` | run identity |
| `source` (`main`, `pr`, `manual`), `pr_number`, `branch`, `commit_id`, `commit_date`, `commit_subject` | code under test |
| `model_id`, `scenario`, `attn_implementation`, `compile_mode`, `kernelize`, `continuous_batching`, `tp`, `batch_size`, `sequence_length`, `num_tokens_to_generate`, `warmup_iterations`, `measurement_iterations`, `bench_args` | workload |
| `harness_commit` | harness version: the last commit of huggingface/transformers-perf that changed it (`-dirty`: local changes) |
| `success`, `n_samples` | a failed scenario has `success = false` and no metrics |
| `e2e_ms_*`, `ttft_ms_*`, `itl_ms_*`, `tokens_per_s_*` | `median`, `mean`, `min`, `max`, `p90`, `cov_pct` (coefficient of variation, %) over the measurement iterations |
| `instance_type`, `cpu_model`, `kernel`, `gpu_name`, `gpu_index`, `numa_node`, `cpus`, `driver_version`, `sm_clock_mhz`, `mem_clock_mhz`, `power_limit_w`, `gpu_temp_c_before` | hardware and clocks |
| `python_version`, `torch_version`, `torch_cuda`, `transformers_version`, `kernels_version`, `numa_balancing`, `aslr` | software |
| `probe_*` | interference over the whole run, read from counters before and after (nothing runs during measurements): CPU time on the benchmark's CPUs used by others (`foreign_cpu_s`), run-queue wait, context switches, interrupts, GPU utilization, average power, power/thermal throttling |
| `startup_weights_mb`, `startup_weights_buffers_mb`, `startup_weights_dtypes` | the model's own tensors after load (tied weights counted once); dtypes as `bfloat16:2048.0,float32:3.1` (MB) |
| `startup_load_host_peak_rss_mb`, `startup_load_host_rss_anon_mb`, `startup_load_host_rss_file_mb`, `startup_load_disk_read_mb`, `startup_load_file_read_mb`, `startup_load_device_allocated_mb` | model load: host peak RSS, anonymous (copies) versus file-backed (mmap'd safetensors) RSS, MB fetched from storage (read calls and page faults on mapped safetensors: the model's size on a cold load, near zero from the page cache) and MB through read calls (mapped files excluded), device memory after it |
| `startup_{first_generate,measurement}_{working_set_mb,peak_reserved_mb,alloc_retries}` | per phase: peak allocated above what was held when it started (KV cache, activations, workspaces), the allocator's peak reserve, and allocations it had to retry after freeing its cache |
| `startup_generate_host_peak_rss_mb`, `startup_process_gpu_mb`, `startup_non_torch_gpu_mb` | host peak RSS over the generations; after them, GPU memory NVML charges the process, and the part outside torch's allocator (CUDA context, library workspaces) |
| `series_key` | hash of model, scenario, instance type, GPU and SM clock: rows with the same key are comparable |

## Querying

```python
import duckdb

duckdb.sql("""
    SELECT started_at, source, branch, commit_id[:10] AS commit, scenario, e2e_ms_median, ttft_ms_median, e2e_ms_cov_pct
    FROM 'hf://datasets/hf-internal-testing/transformers-perf/data/*/*.parquet'
    WHERE success
    ORDER BY series_key, started_at
""").show()
```

Compare a PR with main: take the median of `e2e_ms_median` per `series_key` for `source = 'pr' AND pr_number = N` and
for `source = 'main'` at the PR's base commit, and treat differences smaller than the `cov_pct` of either side as noise.
