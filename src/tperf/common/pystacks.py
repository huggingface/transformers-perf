"""Python call profiles with py-spy: where a process spends its time, by function and by area of code.

    command = record_command(out / "python.folded", rate=200) + ["python", "-m", ...]   # launch under py-spy
    command = record_command(out / "server.folded", rate=100, pid=1234, threads=True)  # attach to a process
    summary = summarize(parse_raw(Path(out / "python.folded").read_text()), rate=200, phases={...})
    python -m tperf.common.pystacks <file.folded> --rate 200 [--by-thread]           # print one

py-spy samples the interpreter's stacks from another process: nothing is hooked or patched in the profiled one and it
does not take the GIL (--nonblocking: it does not pause the process either), so the run keeps close to its real
timings, unlike cProfile, which slows every Python call down and inflates exactly the host-side overhead we are trying
to see. Close, not equal: py-spy needs a CPU of its own; on a CPU-bound run with every core busy, 250 Hz cost about 10%
of throughput, so compare profiled runs with profiled runs. What it cannot give is call counts: a share of the time is
a number of samples, `seconds = samples / rate`. A few percent of samples fail to read (a stack changing while it is
read) and are dropped; py-spy prints how many.

Without `--native`, time spent inside a C extension (a CUDA synchronization, a torch op) is charged to the Python line
that called it, which is usually the line to look at. `--native` adds the C/C++ frames below it.

The `raw` output is the folded-stacks format (`root;...;leaf count` per line): speedscope.app and flamegraph.pl open it
as is. Standard library only: the workstation and the host's system python import this module.
"""

import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_RATE = 200  # samples per second
TOP_FUNCTIONS = 40

# py-spy frame: "name (file:line)", or "name (file)" for native frames without line information
_FRAME = re.compile(r"^(?P<name>.*?) \((?P<file>[^()]*?)(?::(?P<line>\d+))?\)$")
# Frames that only start a thread: a thread is named after the first frame below them
_THREAD_START = ("/threading.py", "/concurrent/futures/thread.py", "/asyncio/runners.py")
# Frames of the benchmark tool itself: hidden from the tables; time with one of them innermost is "(benchmark tool)"
HIDDEN_PATHS = ("/tperf/",)
# Interpreter entry frames, on every stack of a `python -m` process: they say nothing, so they are left out too
ENTRY_FILES = ("<frozen runpy>",)

# Area of a frame, first match wins: (area, substrings of the file)
AREAS = (
    ("transformers serve", ("/transformers/cli/serving/", "/transformers/cli/serve.py")),
    ("transformers generation", ("/transformers/generation/",)),
    ("transformers cache", ("/transformers/cache_utils.py",)),
    ("transformers models", ("/transformers/models/", "/transformers/modeling_", "/transformers/integrations/")),
    ("transformers, other", ("/transformers/",)),
    (
        "server framework",
        (
            "/uvicorn/",
            "/starlette/",
            "/fastapi/",
            "/anyio/",
            "/h11/",
            "/httptools/",
            "/pydantic",
            "/asyncio/",
            "/sse_starlette/",
        ),
    ),  # fmt: skip
    ("torch.compile", ("/torch/_dynamo/", "/torch/_inductor/", "/torch/fx/", "/triton/")),
    ("torch", ("/torch/",)),
    ("tokenizers", ("/tokenizers/",)),
)
NATIVE_AREAS = (
    ("CUDA runtime & driver", ("libcuda", "libcudart", "libnvidia", "libcublas", "libnccl", "libcudnn")),
    ("torch native", ("libtorch", "libc10", "_C.cpython", "libshm")),
    ("Python interpreter", ("libpython", "python3")),
)


def find_pyspy(python: str | None = None) -> str | None:
    """py-spy next to the given interpreter (a venv's bin/), else on PATH."""
    if python:
        candidate = Path(python).parent / "py-spy"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which("py-spy")


def record_command(
    output: Path | str,
    *,
    rate: int = DEFAULT_RATE,
    pid: int | None = None,
    native: bool = False,
    idle: bool = False,
    threads: bool = False,
    pyspy: str = "py-spy",
) -> list[str]:
    """`py-spy record` writing folded stacks to `output`. With `pid`, attaches to that process (needs ptrace rights:
    root, or kernel.yama.ptrace_scope=0); without, append the command to run (py-spy is then its parent, which needs
    no extra rights on Linux)."""
    args = [pyspy, "record", "--format", "raw", "--full-filenames", "--nonblocking", "--rate", str(rate),
            "--output", str(output)]  # fmt: skip
    if native:
        args.append("--native")
    if idle:
        args.append("--idle")
    if threads:
        args.append("--threads")
    return [*args, "--pid", str(pid)] if pid is not None else [*args, "--"]


def parse_raw(text: str) -> list[tuple[list[str], int]]:
    """Folded stacks: [(frames root first, sample count)]."""
    out = []
    for line in text.splitlines():
        stack, _, count = line.rstrip().rpartition(" ")
        if stack and count.isdigit():
            out.append((stack.split(";"), int(count)))
    return out


def parse_frame(frame: str) -> tuple[str, str, int | None]:
    """(function, file, line) of a py-spy frame; line is None for native frames without it."""
    m = _FRAME.match(frame)
    if not m:
        return frame, "", None
    return m["name"], m["file"], int(m["line"]) if m["line"] else None


def is_python(file: str) -> bool:
    return file.endswith(".py") or file.endswith(".pyx") or "<frozen" in file


def area_of(file: str) -> str:
    if is_python(file):
        for area, needles in AREAS:
            if any(n in file for n in needles):
                return area
        return "Python, other"
    for area, needles in NATIVE_AREAS:
        if any(n in file for n in needles):
            return area
    return "native, other"


def short_path(filename: str) -> str:
    for anchor in ("/transformers/", "/tperf/", "/torch/", "/site-packages/", "/lib/python3"):
        if anchor in filename:
            rest = filename.rsplit(anchor, 1)[1]
            if anchor == "/lib/python3":  # the standard library: drop "python3.x/"
                return rest.partition("/")[2] or rest
            head = anchor.strip("/") if anchor != "/site-packages/" else ""
            return f"{head}/{rest}" if head else rest
    return filename.rsplit("/", 1)[-1]


def _hidden(file: str) -> bool:
    return any(h in file for h in HIDDEN_PATHS)


def _entry(file: str) -> bool:
    return file in ENTRY_FILES


def thread_label(frames: list[tuple[str, str, int | None]]) -> str:
    """What a stack's thread runs: its first frame below threading's own, the interpreter's entry and the tool's."""
    for name, file, _ in frames:
        if not file or any(file.endswith(t) for t in _THREAD_START) or _entry(file) or _hidden(file):
            continue
        return f"{name} ({short_path(file)})"
    return "(unknown thread)"


def _thread_name(meta: list[str]) -> str | None:
    """py-spy's own thread frame (--threads): 'thread (27): Thread-3 (_run)' -> 'Thread-3 (_run)'."""
    for frame in meta:
        if frame.startswith("thread "):
            head, sep, name = frame.partition("): ")
            return name.strip() if sep else head
    return None


def summarize(
    samples: list[tuple[list[str], int]],
    rate: int,
    *,
    phases: dict[str, str] | None = None,
    by_thread: bool = False,
    top: int = TOP_FUNCTIONS,
) -> dict:
    """Seconds by area and the top functions, per group of samples.

    Groups: with `phases` ({function name: phase label}), a sample belongs to the phase of the outermost of those
    functions on its stack ("other" when none is: imports, start-up, teardown); with `by_thread`, to its thread
    (py-spy's thread frame, from --threads), named "<thread name>: <what it runs>", what it runs being the most common
    `thread_label` of its samples, so a truncated stack does not make a thread of its own; else everything is one
    group, "all".

    Per group:
      - by_area: self time by area of code, the area of the innermost frame (see AREAS)
      - self: the lines the samples were in (innermost visible frame), with their area
      - inclusive: functions on the stack (each counted once per sample), the time spent in them and below
    """
    groups: dict[str, dict] = defaultdict(lambda: {"samples": 0, "by_area": Counter(), "self": Counter(),
                                                   "inclusive": Counter(), "area_of": {}, "labels": Counter()})  # fmt: skip
    total = 0
    for raw_frames, count in samples:
        total += count
        meta = [f for f in raw_frames if f.startswith("process ") or f.startswith("thread ")]
        frames = [parse_frame(f) for f in raw_frames if f not in meta]
        if phases:
            group = next((phases[name] for name, _, _ in frames if name in phases), "other")
        elif by_thread:
            label = thread_label(frames)
            group = _thread_name(meta) or label
            groups[group]["labels"][label] += count
        else:
            group = "all"
        g = groups[group]
        g["samples"] += count
        shown = [f for f in frames if not _hidden(f[1]) and not _entry(f[1])]
        if not shown or _hidden(frames[-1][1]):
            g["by_area"]["(benchmark tool)"] += count
            continue
        name, file, line = shown[-1]
        area = area_of(file)
        g["by_area"][area] += count
        key = f"{short_path(file)}:{line} {name}" if line is not None else f"{short_path(file)} {name}"
        g["self"][key] += count
        g["area_of"][key] = area
        for name, file, _ in {(n, f, None) for n, f, _ in shown}:
            key = f"{name} ({short_path(file)})"
            g["inclusive"][key] += count
            g["area_of"][key] = area_of(file)

    def rows(counter: Counter, area_of: dict, n: int) -> list[dict]:
        return [{"function": k, "area": area_of.get(k, ""), "samples": c, "seconds": c / rate}
                for k, c in counter.most_common(n)]  # fmt: skip

    out_groups = []
    for name, g in sorted(groups.items(), key=lambda kv: -kv[1]["samples"]):
        if g["labels"]:
            runs = g["labels"].most_common(1)[0][0]
            name = name if name == runs else f"{name}: {runs}"
        out_groups.append({
            "name": name,
            "samples": g["samples"],
            "seconds": g["samples"] / rate,
            "by_area": {a: c / rate for a, c in g["by_area"].most_common()},
            "self": rows(g["self"], g["area_of"], top),
            "inclusive": rows(g["inclusive"], g["area_of"], top),
        })  # fmt: skip
    return {"rate": rate, "samples": total, "seconds": total / rate, "groups": out_groups}


def format_summary(summary: dict, limit: int = 15) -> str:
    """Plain-text tables of a summary, for logs and terminals."""
    lines = [f"{summary['samples']:,} samples at {summary['rate']} Hz = {summary['seconds']:.2f} s of thread time"]
    for g in summary["groups"]:
        lines.append(f"\n== {g['name']}: {g['seconds']:.2f} s")
        for area, sec in g["by_area"].items():
            lines.append(f"  {sec:8.3f} s {sec / g['seconds'] * 100 if g['seconds'] else 0:5.1f}%  {area}")
        lines.append("  self (innermost line):")
        for r in g["self"][:limit]:
            lines.append(f"  {r['seconds']:8.3f} s  {r['function']}")
        lines.append("  inclusive (function and below):")
        for r in g["inclusive"][:limit]:
            lines.append(f"  {r['seconds']:8.3f} s  {r['function']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json

    p = argparse.ArgumentParser(description="Summarize a py-spy raw (folded) profile.")
    p.add_argument("folded", type=Path)
    p.add_argument("--rate", type=int, default=DEFAULT_RATE, help="the rate it was recorded at")
    p.add_argument("--by-thread", action="store_true")
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    summary = summarize(parse_raw(a.folded.read_text()), a.rate, by_thread=a.by_thread)
    print(json.dumps(summary, indent=1) if a.json else format_summary(summary))


if __name__ == "__main__":
    sys.exit(main())
