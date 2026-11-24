"""Host side of `tperf memory profile`: run one generation scenario with every allocation recorded, then build the report.

    python -m tperf.memory.profile --model-id <model> --out <dir> [-b 1 -s 128 -n 64] [--attn sdpa] [--compile-mode M]
    python -m tperf.memory.profile --model-id <model> --out <dir> --config-file scenarios/x.jsonl [--scenario-index 0]

Runs the scenario twice, each time in a fresh process with one recorder on, from before the model is loaded until
the last generation:
- GPU pass: the CUDA caching allocator's history (`torch.cuda.memory._record_memory_history`), every allocation and
  free with the Python stack that made it -> gpu_summary.json, gpu_snapshot.pickle.gz (pytorch.org/memory_viz)
- host pass: memray, native stacks included -> host.memray.bin
- python pass: no recorder inside, py-spy samples the process from outside (tperf.common.pystacks) -> python.folded,
  python_summary.json: where the time goes, by function and area of code, per phase. Each phase runs in its own
  function (_tperf_phase_*), so samples are attributed by their stack, with no clock to line up
Each pass also reads counters at every phase boundary (tperf.harness.framework.memory_metrics) -> profile.json.

and then writes report.html (tperf.memory.report). Recording slows every allocation down: the timings of the GPU and
host passes mean nothing, which is why this is a separate command and never part of a measured run. The python pass
is close to real timings (py-spy hooks nothing and takes no GIL), but not a benchmark either: py-spy uses a CPU.
"""

import argparse
import json
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path


# Each phase boundary allocates and frees one small tensor from this function: the report finds the boundaries in the
# GPU trace by this name (and the phase by the tensor's size), with no clock to line up
MARKER_FUNCTION = "_tperf_phase_marker"
MARKER_UNIT = 1 << 20  # bytes per phase index


def _tperf_phase_marker(index: int, device: str) -> None:
    import torch

    torch.empty((index + 1) * MARKER_UNIT, dtype=torch.uint8, device=device)


PASSES = ("gpu", "host", "python")  # each pass is its own process: one recorder never measures the other

# The python pass's phases: each runs in its own function, so a py-spy sample belongs to the phase on its stack
PHASE_FUNCTIONS = {
    "_tperf_phase_load": "load",
    "_tperf_phase_first_generation": "first generation",
    "_tperf_phase_generations": "generations",
}


def _tperf_phase_load(runner, model_id: str, config) -> None:
    runner.setup_benchmark(model_id, config)


def _tperf_phase_first_generation(runner, config) -> None:
    runner.time_generate(config, warmup=True)


def _tperf_phase_generations(runner, config, iterations: int) -> None:
    for _ in range(iterations):
        runner.time_generate(config, warmup=False)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--batch-size", "-b", type=int, default=1)
    p.add_argument("--sequence-length", "-s", type=int, default=128)
    p.add_argument("--num-tokens-to-generate", "-n", type=int, default=64)
    p.add_argument("--attn", default="sdpa", help="attn_implementation (default sdpa)")
    p.add_argument("--compile-mode", default=None, help="torch.compile mode, e.g. default (static cache)")
    p.add_argument("--continuous-batching", action="store_true")
    p.add_argument("--config-file", type=Path, help="harness scenarios (.jsonl): profile one of them instead (-b/-s/-n"
                   " still apply)")  # fmt: skip
    p.add_argument("--scenario-index", type=int, default=0, help="which scenario of --config-file (default 0)")
    p.add_argument("--iterations", "-i", type=int, default=2, help="generations after the first one (default 2)")
    p.add_argument("--max-entries", type=int, default=2_000_000, help="GPU trace ring buffer size (events)")
    p.add_argument("--stacks", choices=["python", "all"], default="python", help="GPU stacks: all adds C++ frames")
    p.add_argument("--no-memray", action="store_true", help="skip the host pass")
    p.add_argument("--no-native", action="store_true", help="memray without native (C/C++) stacks: much faster")
    p.add_argument("--no-python", action="store_true", help="skip the python pass (py-spy)")
    p.add_argument(
        "--python-iterations",
        type=int,
        default=10,
        help="generations after the first one in the python pass (default 10: enough samples)",
    )
    p.add_argument("--python-rate", type=int, default=200, help="py-spy samples per second (default 200)")
    p.add_argument("--python-native", action="store_true", help="py-spy with native (C/C++) frames: slower sampling")
    p.add_argument("--memray-all", action="store_true", help="keep every host allocation (large file; temporal views)")
    p.add_argument("--keep-snapshot", action=argparse.BooleanOptionalAction, default=False,
                   help="also keep the raw gpu_snapshot.pickle (hundreds of MB): gpu_snapshot.pickle.gz, without the "
                   "benchmark tool's frames, is always written")  # fmt: skip
    p.add_argument("--record", choices=["both", *PASSES], default="both", help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the GPU pass and the host pass in separate processes, merge their counters, write the report.

    Recording the CUDA allocator's history keeps every event and its stack in host memory (1.4M events, more than a
    GB, for a 1B model), and memray would count that as the model's; memray's hooks would likewise slow down the GPU
    pass. Each pass's own counters are taken in the pass that does not disturb them: device counters from the GPU
    pass are unaffected by its host-side recording, and host counters come from the host pass."""
    argv = sys.argv[1:] if argv is None else list(argv)
    a = parse_args(argv)
    if a.record != "both":
        return record(a)
    a.out.mkdir(parents=True, exist_ok=True)
    errors = []
    from tperf.common import pystacks

    for name in PASSES:
        if (name == "host" and a.no_memray) or (name == "python" and a.no_python):
            continue
        command = [sys.executable, "-m", "tperf.memory.profile", *argv, "--record", name]
        if name == "python":
            pyspy = pystacks.find_pyspy(sys.executable)
            if pyspy is None:
                print("py-spy is not installed: no python pass (pip install py-spy)", file=sys.stderr)
                continue
            folded = a.out / "python.folded"
            folded.unlink(missing_ok=True)
            command = (
                pystacks.record_command(folded, rate=a.python_rate, native=a.python_native, pyspy=pyspy) + command
            )
        print(f"{name} pass", file=sys.stderr, flush=True)
        rc = subprocess.run(command).returncode
        if rc:
            errors.append(f"{name} pass exited {rc}")
    passes = {n: json.loads(f.read_text()) for n in PASSES if (f := a.out / f"profile.{n}.json").exists()}
    if not passes:
        sys.exit(f"no pass recorded anything: {'; '.join(errors)}")
    profile = dict(passes.get("host") or passes.get("gpu") or passes["python"])
    if "gpu" in passes:
        profile["gpu_marks"] = passes["gpu"]["marks"]
        for key in ("gpu_name", "gpu_memory_total_mb"):
            profile[key] = passes["gpu"].get(key)
    if "python" in passes:
        profile["python_marks"] = passes["python"]["marks"]
        write_python_summary(a, passes["python"])
    profile["passes"] = sorted(passes)
    profile["error"] = "; ".join(filter(None, [*(p.get("error") for p in passes.values()), *errors])) or None
    (a.out / "profile.json").write_text(json.dumps(profile, indent=1, default=str))

    from tperf.memory.report import build

    print(build(a.out))
    if profile["error"]:
        sys.exit(f"scenario failed: {profile['error']} (report written with what was recorded)")


def write_python_summary(a: argparse.Namespace, python_pass: dict) -> None:
    """python_summary.json from python.folded: per phase, where the time went, with the phase's wall time and the
    tokens it generated (for per-token figures)."""
    from tperf.common import pystacks

    folded = a.out / "python.folded"
    if not folded.exists():
        return
    summary = pystacks.summarize(pystacks.parse_raw(folded.read_text()), a.python_rate, phases=PHASE_FUNCTIONS)
    marks = python_pass.get("marks") or []
    walls = {cur["name"]: (cur["time_ms"] - prev["time_ms"]) / 1e3 for prev, cur in zip(marks, marks[1:])}
    per_generation = a.batch_size * a.num_tokens_to_generate
    wall_of = {"load": walls.get("loaded"), "first generation": walls.get("first generation"),
               "generations": walls.get("generations")}  # fmt: skip
    tokens_of = {"first generation": per_generation, "generations": per_generation * a.python_iterations}
    for g in summary["groups"]:
        g["wall_seconds"] = wall_of.get(g["name"])
        g["tokens"] = tokens_of.get(g["name"])
    summary["native"] = a.python_native
    summary["iterations"] = a.python_iterations
    (a.out / "python_summary.json").write_text(json.dumps(summary, indent=1))


def scenario_config(a: argparse.Namespace):
    from tperf.harness.framework.benchmark_config import BenchmarkConfig

    if a.config_file is not None:
        lines = [ln for ln in a.config_file.read_text().splitlines() if ln.strip().startswith("{")]
        base = json.loads(lines[a.scenario_index])
    else:
        base = {"attn_implementation": a.attn, "continuous_batching": a.continuous_batching}
        if a.compile_mode:
            base["compile_kwargs"] = {"mode": a.compile_mode}
    return BenchmarkConfig.from_dict({
        **base,
        "warmup_iterations": 0,
        "measurement_iterations": a.iterations,
        "batch_size": a.batch_size,
        "sequence_length": a.sequence_length,
        "num_tokens_to_generate": a.num_tokens_to_generate,
    })  # fmt: skip


def record(a: argparse.Namespace) -> None:
    """One pass: the scenario with one recorder on, counters at each phase boundary -> profile.<pass>.json."""
    import torch

    import transformers
    from tperf.harness.framework.benchmark_runner import BenchmarkRunner, get_git_revision
    from tperf.harness.framework.memory_metrics import (
        device_memory,
        host_memory,
        model_breakdown,
        process_gpu_memory_mb,
    )

    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("tperf.memory")
    a.out.mkdir(parents=True, exist_ok=True)
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"
    accelerator = torch.cuda if cuda else None
    gpu_pass = a.record == "gpu" and cuda
    config = scenario_config(a)
    iterations = a.python_iterations if a.record == "python" else a.iterations
    marks: list[dict] = []

    def mark(name: str) -> None:
        if cuda:
            torch.cuda.synchronize()
            _tperf_phase_marker(len(marks), device)
        marks.append({
            "name": name,
            "index": len(marks),
            "time_ms": time.time() * 1e3,
            "host": host_memory(),
            "device": device_memory(accelerator) if cuda else {},
            "process_gpu_mb": process_gpu_memory_mb(torch.cuda.current_device()) if cuda else None,
        })  # fmt: skip

    tracker = None
    if a.record == "host":
        import memray

        file_format = memray.FileFormat.ALL_ALLOCATIONS if a.memray_all else memray.FileFormat.AGGREGATED_ALLOCATIONS
        host_bin = a.out / "host.memray.bin"
        host_bin.unlink(missing_ok=True)
        tracker = memray.Tracker(host_bin, native_traces=not a.no_native, file_format=file_format)
        tracker.__enter__()
    if gpu_pass:
        torch.cuda.memory._record_memory_history(max_entries=a.max_entries, stacks=a.stacks)

    runner = BenchmarkRunner(logger, output_dir=str(a.out / "harness"))
    error = None
    weights: dict = {}
    try:
        with torch.no_grad():
            mark("start")
            _tperf_phase_load(runner, a.model_id, config)
            mark("loaded")
            weights = model_breakdown(runner.model)
            _tperf_phase_first_generation(runner, config)
            mark("first generation")
            _tperf_phase_generations(runner, config, iterations)
            mark("generations")
    except Exception as e:  # keep what was recorded: the report shows how far it got
        error = repr(e)
        if runner.model is not None:
            weights = model_breakdown(runner.model)
        mark("error")
    finally:
        if tracker is not None:  # before anything else allocates
            tracker.__exit__(None, None, None)
        if gpu_pass:
            snapshot = torch.cuda.memory._snapshot()
            torch.cuda.memory._record_memory_history(enabled=None)
            from tperf.memory.report import gpu_summary

            # The report needs the summary only; the full snapshot is for memory_viz
            summary = gpu_summary(snapshot, MARKER_FUNCTION, MARKER_UNIT)
            (a.out / "gpu_summary.json").write_text(json.dumps(summary, separators=(",", ":")))
            import gzip
            import pickle

            from tperf.memory.report import strip_hidden_frames

            if a.keep_snapshot:
                with open(a.out / "gpu_snapshot.pickle", "wb") as f:
                    pickle.dump(snapshot, f)
            # For PyTorch's memory viewer (served with Content-Encoding: gzip): ~25x smaller than the pickle
            with gzip.open(a.out / "gpu_snapshot.pickle.gz", "wb", compresslevel=6) as f:
                pickle.dump(strip_hidden_frames(snapshot), f)

    profile = {
        "model_id": a.model_id,
        "commit_id": get_git_revision(),
        "config": config.to_dict(),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
        "marks": marks,
        "marker_function": MARKER_FUNCTION,
        "marker_unit": MARKER_UNIT,
        "load_memory": getattr(runner, "load_memory", {}),
        "weights": weights,
        "error": error,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
        },
        "gpu_name": torch.cuda.get_device_name() if cuda else None,
        "gpu_memory_total_mb": torch.cuda.get_device_properties(0).total_memory / 2**20 if cuda else None,
        "host_memory_total_mb": _mem_total_mb(),
    }
    (a.out / f"profile.{a.record}.json").write_text(json.dumps(profile, indent=1, default=str))
    if error:
        sys.exit(1)


def _mem_total_mb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


if __name__ == "__main__":
    main()
