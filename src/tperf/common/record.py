"""Record harness runs and publish them to a Hugging Face dataset for performance tracking.

Runs on the bench host, outside the measured window: `manifest` before a run, `publish` after.

    # before the run: capture the environment (software, hardware, clocks, git identity)
    python -m tperf.common.record manifest --out RUN_DIR/manifest.json --gpu 0 --cpus 1-8 --numa-node 0 \
        --model meta-llama/Llama-2-7b-hf --bench-args "-w 5 -i 30 ..." --group G --run 1 --label L

    # after the runs: one row per (run, scenario) -> Parquet, plus the raw files, uploaded to the dataset
    python -m tperf.common.record publish RUN_DIR [RUN_DIR ...] --repo user/transformers-perf [--dry-run]

A run directory holds manifest.json, probe.json (tperf.common.noise_probe) and the harness result files.

    # tperf.processors.bench runs (one directory per run, processor_*.json): one row per (run, scenario) -> processors/
    python -m tperf.common.record publish-processors RUN_DIR [RUN_DIR ...] --repo R --group G --label L --gpu 0 --cpus 1-8

Identity of a run (all optional, for comparing main with a PR): BENCH_SOURCE (main|pr|manual, default manual),
BENCH_PR (PR number), BENCH_BRANCH (default: the checkout's branch), and for side-by-side runs BENCH_AB_ID,
BENCH_AB_SIDE (base|head) and BENCH_AB_ROUND. TRANSFORMERS_DIR is the checkout under test (git identity).
"""

from __future__ import annotations


import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
TRANSFORMERS_DIR = os.environ.get("TRANSFORMERS_DIR", "/opt/bench/transformers")


def sh(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def instance_type() -> str:
    """EC2 instance type via IMDSv2, empty elsewhere."""
    try:
        req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        token = urllib.request.urlopen(req, timeout=1).read().decode()
        req = urllib.request.Request(
            "http://169.254.169.254/latest/meta-data/instance-type", headers={"X-aws-ec2-metadata-token": token}
        )
        return urllib.request.urlopen(req, timeout=1).read().decode()
    except OSError:
        return ""


def gpu_info(index: int) -> dict:
    import pynvml

    pynvml.nvmlInit()
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(index)
        return {
            "gpu_name": pynvml.nvmlDeviceGetName(h),
            "driver_version": pynvml.nvmlSystemGetDriverVersion(),
            # Idle clocks: with clocks locked (tperf/host/prepare.py), these are the locked values
            "sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
            "mem_clock_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM),
            "power_limit_w": pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000,
            "gpu_temp_c_before": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
        }
    finally:
        pynvml.nvmlShutdown()


def package_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return ""


def transformers_version() -> str:
    """Version of the transformers that is imported (an A/B side selects its checkout with PYTHONPATH)."""
    try:
        import transformers

        return transformers.__version__
    except ImportError:
        return ""


def cmd_manifest(a: argparse.Namespace) -> None:
    import torch

    git = sh("git", "-C", TRANSFORMERS_DIR, "log", "-1", "--format=%H%n%cI%n%s").split("\n") + ["", "", ""]
    branch = os.environ.get("BENCH_BRANCH") or sh("git", "-C", TRANSFORMERS_DIR, "rev-parse", "--abbrev-ref", "HEAD")
    sysctl = {k: Path(f"/proc/sys/kernel/{k}").read_text().strip() for k in ("numa_balancing", "randomize_va_space")}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": f"{a.group}-run{a.run}",
        "group_id": a.group,
        "run_index": a.run,
        "label": a.label,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": os.environ.get("BENCH_SOURCE", "manual"),
        "pr_number": int(os.environ["BENCH_PR"]) if os.environ.get("BENCH_PR") else None,
        "branch": branch,
        "commit_id": git[0],
        "commit_date": git[1],
        "commit_subject": git[2],
        # Side-by-side runs (tperf ab): one A/B shares an id, each round runs both sides
        "ab_id": os.environ.get("BENCH_AB_ID") or None,
        "ab_side": os.environ.get("BENCH_AB_SIDE") or None,
        "ab_round": int(os.environ["BENCH_AB_ROUND"]) if os.environ.get("BENCH_AB_ROUND") else None,
        "model_id": a.model,
        "bench_args": a.bench_args,
        # harness version (tperf harness version): the last commit of this repository that changed it, -dirty if modified
        "harness_commit": os.environ.get("BENCH_HARNESS") or None,
        "instance_type": instance_type(),
        "cpu_model": next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if "model name" in line
            ),
            "",
        ),
        "kernel": platform.release(),
        "gpu_index": a.gpu,
        "numa_node": a.numa_node,
        "cpus": a.cpus,
        **gpu_info(a.gpu),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda or "",
        "transformers_version": transformers_version(),
        "kernels_version": package_version("kernels"),
        "numa_balancing": sysctl["numa_balancing"],
        "aslr": sysctl["randomize_va_space"],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(manifest, indent=2))


def stats(values: list[float], scale: float) -> dict:
    """Median/mean/min/max/p90 (scaled) and coefficient of variation (%) of a measurement series."""
    if not values:
        return {}
    v = sorted(x * scale for x in values)
    mean = statistics.fmean(v)
    return {
        "median": statistics.median(v),
        "mean": mean,
        "min": v[0],
        "max": v[-1],
        "p90": v[min(len(v) - 1, round(0.9 * (len(v) - 1)))],
        "cov_pct": 100 * statistics.stdev(v) / mean if len(v) > 1 and mean else 0.0,
    }


def split_itl(timestamps: list | None) -> tuple[list[float], list[float]]:
    """Inter-token latency over the first and second half of the generated tokens, per iteration (seconds).

    Needs the per-token timestamps (harness saves them unsummarized).
    Averaged over the requests of a batch. Choose the workload so the change of interest falls between the halves,
    e.g. a sliding window filling at the middle of the generation."""
    first, second = [], []
    for iteration in timestamps or []:
        h1, h2 = [], []
        for ts in iteration:
            n, mid = len(ts), len(ts) // 2
            if n >= 4:
                h1.append((ts[mid - 1] - ts[0]) / (mid - 1))
                h2.append((ts[-1] - ts[mid]) / (n - 1 - mid))
        if h1:
            first.append(statistics.fmean(h1))
            second.append(statistics.fmean(h2))
    return first, second


COMPILE_PHASES = ("first_generate", "warmup", "measurement")
COMPILE_COUNTERS = ("compiled_graphs", "graph_breaks", "fxgraph_cache_hits", "fxgraph_cache_misses")
STARTUP_COLUMNS = [
    "startup_load_time_s",
    "startup_first_generate_ms",
    "startup_warmup_first_ms",
    "startup_warmup_last_ms",
    "startup_first_generate_peak_memory_mb",
    "startup_measurement_peak_memory_mb",
    *(f"startup_{phase}_{counter}" for phase in COMPILE_PHASES for counter in COMPILE_COUNTERS),
]


def flat_startup(startup: dict) -> dict:
    """Startup metrics saved by harness: load time, cold first generation,
    warmup latencies, torch.compile counters per phase, peak memory."""
    if not startup:
        return {}
    ms = lambda x: x * 1e3 if x is not None else None  # noqa: E731
    warmup = startup.get("warmup_latencies") or []
    row = {
        "startup_load_time_s": startup.get("load_time"),
        "startup_first_generate_ms": ms(startup.get("first_generate_latency")),
        "startup_warmup_first_ms": ms(warmup[0]) if warmup else None,
        "startup_warmup_last_ms": ms(warmup[-1]) if warmup else None,
        "startup_first_generate_peak_memory_mb": startup.get("first_generate_peak_memory_mb"),
        "startup_measurement_peak_memory_mb": startup.get("measurement_peak_memory_mb"),
    }
    for phase in COMPILE_PHASES:
        for counter in COMPILE_COUNTERS:
            row[f"startup_{phase}_{counter}"] = startup.get(f"{phase}_{counter}")
    return row


def flat_probe(probe: dict) -> dict:
    cpu, gpu = probe.get("cpu", {}), probe.get("gpu", {})
    throttle = gpu.get("throttle_ms", {})
    return {
        "probe_exit_code": probe.get("exit_code"),
        "probe_elapsed_s": probe.get("elapsed_s"),
        "probe_command_cpu_s": cpu.get("command_cpu_s"),
        "probe_foreign_cpu_s": cpu.get("foreign_cpu_s"),
        "probe_runqueue_wait_ms": cpu.get("runqueue_wait_ms"),
        "probe_voluntary_ctx_switches": cpu.get("voluntary_ctx_switches"),
        "probe_involuntary_ctx_switches": cpu.get("involuntary_ctx_switches"),
        "probe_interrupts_on_cpus": cpu.get("interrupts_on_cpus"),
        "probe_gpu_util_mean_pct": gpu.get("util_mean_pct"),
        "probe_gpu_avg_power_w": gpu.get("avg_power_w"),
        "probe_power_throttle_ms": throttle.get("power"),
        "probe_thermal_throttle_ms": throttle.get("thermal"),
    }


def scenario_rows(run_dir: Path) -> list[dict]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    probe_path = run_dir / "probe.json"
    probe = flat_probe(json.loads(probe_path.read_text())) if probe_path.exists() else {}
    rows = []
    for file in sorted(run_dir.rglob("*_benchmark_*.json")):
        for entry in json.loads(file.read_text()).values():
            config, meas, meta = entry["config"], entry["measurements"], entry["metadata"]
            e2e = meas.get("e2e_latency") or []
            tokens = config["batch_size"] * config["num_tokens_to_generate"]
            compile_kwargs = config.get("compile_kwargs") or {}
            row = {
                **manifest,
                "scenario": config["name"],
                "attn_implementation": config.get("attn_implementation"),
                "compile_mode": compile_kwargs.get("mode") if compile_kwargs else None,
                "kernelize": bool(config.get("kernelize")),
                "continuous_batching": bool(config.get("continuous_batching")),
                "tp": bool(config.get("tp_plan")),
                "batch_size": config["batch_size"],
                "sequence_length": config["sequence_length"],
                "num_tokens_to_generate": config["num_tokens_to_generate"],
                "warmup_iterations": config["warmup_iterations"],
                "measurement_iterations": config["measurement_iterations"],
                "success": bool(meta.get("success", True) and e2e),
                "n_samples": len(e2e),
            }
            itl_halves = split_itl(meas.get("timestamps"))
            for name, values, scale in (
                ("itl_first_half_ms", itl_halves[0], 1e3),
                ("itl_second_half_ms", itl_halves[1], 1e3),
                ("e2e_ms", e2e, 1e3),
                ("ttft_ms", meas.get("time_to_first_token") or [], 1e3),
                ("itl_ms", meas.get("inter_token_latency") or [], 1e3),
                ("tokens_per_s", [tokens / x for x in e2e if x > 0], 1.0),
            ):
                for stat, value in stats(values, scale).items():
                    row[f"{name}_{stat}"] = value
            row.update(flat_startup(meas.get("startup") or {}))
            # Series of comparable measurements: same workload on the same hardware and clocks. Software versions
            # (commit, torch, driver) are what the dashboard tracks within a series.
            key = [row[k] for k in ("model_id", "scenario", "instance_type", "gpu_name", "sm_clock_mhz")]
            row["series_key"] = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:16]
            rows.append(row | probe)
    return rows


METRIC_COLUMNS = [
    f"{name}_{stat}"
    for name in ("e2e_ms", "ttft_ms", "itl_ms", "itl_first_half_ms", "itl_second_half_ms", "tokens_per_s")
    for stat in ("median", "mean", "min", "max", "p90", "cov_pct")
]
# Every row has every column, in this order, whatever the scenario's outcome
COLUMNS = [
    *("schema_version", "run_id", "group_id", "run_index", "label", "started_at"),
    *("source", "pr_number", "branch", "commit_id", "commit_date", "commit_subject", "ab_id", "ab_side", "ab_round"),
    *("model_id", "bench_args", "harness_commit", "scenario", "attn_implementation", "compile_mode", "kernelize"),
    *("continuous_batching", "tp", "batch_size", "sequence_length", "num_tokens_to_generate"),
    *("warmup_iterations", "measurement_iterations", "success", "n_samples"),
    *METRIC_COLUMNS,
    *STARTUP_COLUMNS,
    "series_key",
    *("instance_type", "cpu_model", "kernel", "gpu_index", "numa_node", "cpus", "gpu_name"),
    *("driver_version", "sm_clock_mhz", "mem_clock_mhz", "power_limit_w", "gpu_temp_c_before"),
    *("python_version", "torch_version", "torch_cuda", "transformers_version", "kernels_version"),
    *("numa_balancing", "aslr"),
    *flat_probe({}).keys(),
]


def null_column_type(name: str):
    import pyarrow as pa

    if name.endswith(COMPILE_COUNTERS):
        return pa.int64()
    if name in METRIC_COLUMNS or name.startswith(("probe_", "startup_")):
        return pa.float64()
    return pa.int64() if name in ("pr_number", "ab_round") else pa.string()


def cmd_publish(a: argparse.Namespace) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    run_dirs = [Path(d) for d in a.run_dirs]
    rows = [row for d in run_dirs for row in scenario_rows(d)]
    if not rows:
        sys.exit("no scenarios found")
    unknown = {k for r in rows for k in r} - set(COLUMNS)
    if unknown:
        sys.exit(f"columns missing from COLUMNS: {sorted(unknown)}")
    for r in rows:
        status = "ok" if r["success"] else "FAILED"
        print(
            f"{r['run_id']}  {r['scenario'][:70]:<70} {status:>6}  e2e {r.get('e2e_ms_median', float('nan')):9.2f} ms"
        )
    if a.dry_run:
        return

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
    operations = []
    for d in run_dirs:
        run_id = json.loads((d / "manifest.json").read_text())["run_id"]
        month = run_id[:4] + "-" + run_id[4:6]
        run_rows = [r for r in rows if r["run_id"] == run_id]
        table = pa.Table.from_pydict({c: [r.get(c) for r in run_rows] for c in COLUMNS})
        # Columns that are null in every row (e.g. metrics of a run where all scenarios failed) get their real type, so
        # files can be read together
        table = table.cast(
            pa.schema(
                pa.field(f.name, null_column_type(f.name)) if pa.types.is_null(f.type) else f for f in table.schema
            )
        )
        local = d / "rows.parquet"
        pq.write_table(table, local)
        operations.append(CommitOperationAdd(f"data/{month}/{run_id}.parquet", str(local)))
        for f in sorted(p for p in d.rglob("*") if p.is_file() and p.name != "rows.parquet"):
            operations.append(CommitOperationAdd(f"raw/{month}/{run_id}/{f.relative_to(d)}", str(f)))
    commit = api.create_commit(
        a.repo,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Add {len(run_dirs)} run(s): {rows[0]['group_id']}",
    )
    print(f"published {len(rows)} rows from {len(run_dirs)} run(s): {commit.commit_url}")


PROCESSOR_CONFIG = ("model", "inputs", "batch", "backend", "component", "stage", "threads")
PROCESSOR_COLUMNS = [
    *("group_id", "run_id", "run", "label", "source", "started_at", "name"),
    *("commit_id", "commit_subject", "branch"),
    *PROCESSOR_CONFIG,
    *("success", "error", "median_ms", "min_ms", "max_ms", "cov_pct", "samples_ms"),
    *("load_s", "first_call_s", "model_load_s", "output_sha256", "output_shapes", "syncs", "sync_sources", "features"),
    *("image_processor_class", "instance_type", "cpu_model", "cpus", "gpu_name", "driver_version"),
    *("python_version", "torch_version", "transformers_version", "torchvision_version", "pillow_version"),
]


def processor_rows(run_dir: Path, a: argparse.Namespace, env: dict) -> list[dict]:
    rows = []
    for f in sorted(run_dir.glob("processor_*.json")):
        r = json.loads(f.read_text())
        samples = [1e3 * s for s in r.get("samples_s", [])]
        row = {
            **env,
            "group_id": a.group,
            "run": int(run_dir.name.removeprefix("run")) if run_dir.name.startswith("run") else None,
            "run_id": f"{a.group}-{run_dir.name}",
            "label": a.label,
            "started_at": r.get("started_at"),
            "name": r["name"],
            **{k: r["config"].get(k) for k in PROCESSOR_CONFIG},
            "success": r["success"],
            "error": r.get("error"),
            "load_s": r.get("load_s"),
            "first_call_s": r.get("first_call_s"),
            "model_load_s": r.get("model_load_s"),
            "image_processor_class": r.get("image_processor_class"),
            "output_sha256": (r.get("output") or {}).get("sha256"),
            "output_shapes": json.dumps((r.get("output") or {}).get("shapes")) if r.get("output") else None,
            "features": json.dumps(r["features"]) if "features" in r else None,
            "syncs": r["syncs"]["count"] if "syncs" in r else None,
            "sync_sources": json.dumps(r["syncs"]["sources"]) if "syncs" in r else None,
            "samples_ms": samples or None,
        }
        if samples:
            row.update(
                median_ms=statistics.median(samples),
                min_ms=min(samples),
                max_ms=max(samples),
                cov_pct=100 * statistics.stdev(samples) / statistics.mean(samples) if len(samples) > 1 else 0.0,
            )
        rows.append(row)
    return rows


def cmd_publish_processors(a: argparse.Namespace) -> None:
    """tperf.processors.bench runs of one group -> processors/<month>/<group>.parquet and the raw files."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    git = lambda *args: sh("git", "-C", TRANSFORMERS_DIR, *args)  # noqa: E731
    env = {
        "source": os.environ.get("BENCH_SOURCE", "manual"),
        "commit_id": git("rev-parse", "HEAD"),
        "commit_subject": git("log", "-1", "--format=%s"),
        "branch": os.environ.get("BENCH_BRANCH", ""),
        "instance_type": instance_type(),
        "cpu_model": next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "",
        )
        if Path("/proc/cpuinfo").exists()
        else platform.processor(),
        "cpus": a.cpus,
        "python_version": platform.python_version(),
        "torch_version": package_version("torch"),
        "transformers_version": transformers_version(),
        "torchvision_version": package_version("torchvision"),
        "pillow_version": package_version("pillow"),
    }
    if a.gpu is not None:
        g = gpu_info(a.gpu)
        env.update(gpu_name=g["gpu_name"], driver_version=g["driver_version"])
    run_dirs = [Path(d) for d in a.run_dirs]
    rows = [row for d in run_dirs for row in processor_rows(d, a, env)]
    if not rows:
        sys.exit("no processor results found")
    for r in rows:
        ms = f"{r['median_ms']:9.2f} ms" if r.get("median_ms") is not None else "FAILED"
        print(f"{r['run_id']}  {r['name'][:80]:<80} {ms}")
    if a.dry_run:
        return

    from huggingface_hub import CommitOperationAdd, HfApi

    types = {
        "run": pa.int64(),
        "batch": pa.int64(),
        "threads": pa.int64(),
        "syncs": pa.int64(),
        "success": pa.bool_(),
        "samples_ms": pa.list_(pa.float64()),
    }
    floats = ("median_ms", "min_ms", "max_ms", "cov_pct", "load_s", "first_call_s", "model_load_s")
    schema = pa.schema(
        pa.field(c, types.get(c, pa.float64() if c in floats else pa.string())) for c in PROCESSOR_COLUMNS
    )
    table = pa.Table.from_pydict({c: [r.get(c) for r in rows] for c in PROCESSOR_COLUMNS}, schema=schema)
    month = a.group[:4] + "-" + a.group[4:6]
    local = run_dirs[0].parent / "processor-rows.parquet"
    pq.write_table(table, local)
    operations = [CommitOperationAdd(f"processors/{month}/{a.group}.parquet", str(local))]
    for d in run_dirs:
        for f in sorted(p for p in d.rglob("*") if p.is_file()):
            operations.append(CommitOperationAdd(f"raw/{month}/{a.group}/{d.name}/{f.relative_to(d)}", str(f)))
    api = HfApi()
    commit = api.create_commit(
        a.repo, repo_type="dataset", operations=operations, commit_message=f"Add processor runs: {a.group}"
    )
    print(f"published {len(rows)} processor rows from {len(run_dirs)} run(s): {commit.commit_url}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--out", required=True)
    m.add_argument("--gpu", type=int, required=True)
    m.add_argument("--cpus", required=True)
    m.add_argument("--numa-node", type=int, required=True)
    m.add_argument("--model", required=True)
    m.add_argument("--bench-args", required=True)
    m.add_argument("--group", required=True)
    m.add_argument("--run", type=int, required=True)
    m.add_argument("--label", default="")
    m.set_defaults(func=cmd_manifest)
    pub = sub.add_parser("publish")
    pub.add_argument("run_dirs", nargs="+")
    pub.add_argument("--repo", required=True)
    pub.add_argument("--dry-run", action="store_true")
    pub.set_defaults(func=cmd_publish)
    pp = sub.add_parser("publish-processors")
    pp.add_argument("run_dirs", nargs="+")
    pp.add_argument("--repo", required=True)
    pp.add_argument("--group", required=True, help="<UTC stamp YYYYmmdd-HHMMSS>-<label>")
    pp.add_argument("--label", required=True)
    pp.add_argument("--gpu", type=int, help="GPU of the encoder stages")
    pp.add_argument("--cpus", default="")
    pp.add_argument("--dry-run", action="store_true")
    pp.set_defaults(func=cmd_publish_processors)
    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
