"""tperf: one entry point for the transformers-perf tooling.

    tperf <command> [<subcommand>] [arguments...]

Each (sub)command forwards its arguments to the `main(argv)` of one module of this package, which keeps its own
arguments and environment variables (`tperf <command> <subcommand> --help`). Every module also runs on its own with
`python -m tperf.<feature>.<module>`. Modules are imported only when their command runs: some need torch or
huggingface_hub, which only the bench host has.
"""

import importlib
import sys


# command -> {subcommand: (module[:function], help)}; the subcommand "" runs when the next argument is not a subcommand
COMMANDS: dict[str, dict[str, tuple[str, str]]] = {
    "ab": {
        "": ("tperf.ab.run", "side-by-side benchmark of a PR against its base: tperf ab <pr|main> [rounds]"),
        "run": ("tperf.ab.run", "same as tperf ab <pr|main> [rounds]"),
        "compare": ("tperf.ab.compare", "compare the two sides of an A/B run: tperf ab compare results/ab-<id>"),
    },
    "generation": {
        "run": ("tperf.generation.run", "repeated harness runs of main on one GPU, with a stability check"),
        "compare": ("tperf.generation.compare", "run-to-run stability of harness run directories"),
    },
    "processors": {
        "run": ("tperf.processors.run", "multimodal processors and encoders on the host, several runs"),
        "summarize": ("tperf.processors.summarize", "summarize a processors results directory"),
    },
    "tp": {
        "run": ("tperf.tp.run", "harness under tensor parallelism, per GPU group"),
        "profile": ("tperf.tp.profile_run", "one TP decode step broken down per rank"),
        "summarize": ("tperf.tp.summarize", "print tp profile results (tp*.jsonl)"),
    },
    "memory": {
        "profile": ("tperf.memory.run", "where memory goes in one scenario: HTML report"),
        "report": ("tperf.memory.report", "rebuild report.html from a profile directory"),
    },
    "serve": {
        "run": (
            "tperf.serve.run",
            "multi-turn chat against transformers serve on the host: tperf serve run [pr|main]",
        ),
        "bench": ("tperf.serve.bench", "the same, here (any machine): start transformers serve, play conversations"),
        "summarize": ("tperf.serve.summarize", "print serve results, or compare runs turn by turn"),
        "report": ("tperf.serve.report", "a profiled serve run (--profile) as a time report: report.html, time.html"),
    },
    "baselines": {
        "run": ("tperf.baselines.run", "bf16 GEMM and host<->device bandwidth"),
        "repeat": ("tperf.baselines.repeat", "repeat the baselines and compare the runs"),
        "p2p": ("tperf.baselines.p2p", "NCCL all-reduce/send per GPU group"),
        "summarize": ("tperf.baselines.summarize", "summarize a baselines results directory"),
        "summarize-p2p": ("tperf.baselines.summarize_p2p", "summarize a p2p results directory"),
        "compare": ("tperf.baselines.compare", "run-to-run stability of baselines results directories"),
    },
    "host": {
        "tune": ("tperf.host.deploy:tune", "tune the host for low noise: tperf host tune [apply|status|revert]"),
        "install": ("tperf.host.deploy:install", "install CUDA, nvbandwidth, torch, transformers under /opt/bench"),
    },
    "deploy": {
        "": ("tperf.host.deploy", "copy the package (harness included) and scenarios to the host, install the timer")
    },
    "harness": {
        "run": (
            "tperf.harness.run_benchmarks",
            "run the generation benchmark here (python -m tperf.harness.run_benchmarks)",
        ),
        "compare": ("tperf.harness.compare_results", "compare harness results against a baseline, flag regressions"),
        "version": ("tperf.common.harness", "harness version: last commit that changed src/tperf/harness"),
    },
    "noise": {
        "soak": ("tperf.noise.soak", "hours of generation batches + contention probes + steal sampling on one host"),
        "probe": ("tperf.noise.soak:probe", "one contention probe (clock, memory bandwidth, L3/DRAM latency)"),
        "summarize": ("tperf.noise.summarize", "compare soaks side by side: tperf noise summarize <soak> <soak>"),
    },
    "dashboard": {"build": ("tperf.dashboard.build", "build (and publish) the dashboard from the dataset")},
    "record": {"": ("tperf.common.record", "host side: write a run manifest, publish runs to the dataset")},
    "daily": {"": ("tperf.daily.main", "host side: the daily main benchmark (run by bench-daily.timer)")},
}


def usage() -> str:
    lines = [__doc__.strip(), "", "commands:"]
    for name, subs in COMMANDS.items():
        for sub, (_, text) in subs.items():
            lines.append(f"  {f'{name} {sub}'.strip():<24} {text}")
    return "\n".join(lines)


def group_usage(name: str) -> str:
    subs = COMMANDS[name]
    lines = [f"usage: tperf {name} {{{','.join(s for s in subs if s)}}} ...", "", "subcommands:"]
    for sub, (_, text) in subs.items():
        lines.append(f"  {sub or '(default)':<16} {text}")
    lines.append(f"\n`tperf {name} <subcommand> --help` for its arguments.")
    return "\n".join(lines)


def run(target: str, prog: str, argv: list[str]) -> int:
    module, _, func = target.partition(":")
    sys.argv[0] = prog  # argparse names the program after it
    return getattr(importlib.import_module(module), func or "main")(argv) or 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(usage())
        return 0 if argv else 2
    name, rest = argv[0], argv[1:]
    if name not in COMMANDS:
        print(f"tperf: unknown command {name!r}\n\n{usage()}", file=sys.stderr)
        return 2
    subs = COMMANDS[name]
    if rest and rest[0] in subs and rest[0]:
        return run(subs[rest[0]][0], f"tperf {name} {rest[0]}", rest[1:])
    if "" in subs and not (len(subs) > 1 and rest[:1] in (["-h"], ["--help"])):
        return run(subs[""][0], f"tperf {name}", rest)
    print(group_usage(name), file=sys.stdout if rest[:1] in (["-h"], ["--help"]) else sys.stderr)
    return 0 if rest[:1] in (["-h"], ["--help"]) else 2


if __name__ == "__main__":
    sys.exit(main())
