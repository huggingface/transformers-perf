"""tperf baselines repeat: run the full baseline suite N times and check run-to-run stability.

Usage: tperf baselines repeat [N=3] [label=repeat]     (HOST=ubuntu@IP to override)

Re-applies tperf/host/prepare.py before each run, so every run starts from the same state
(page cache dropped, GPU clocks re-locked). Then compares all runs with `tperf baselines compare`.
"""

import argparse
import subprocess
import sys


from tperf.common.benchlib import ROOT, log, run_module, stamp, tune_host


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("n", nargs="?", type=int, default=3)
    p.add_argument("label", nargs="?", default="repeat")
    a = p.parse_args(argv)

    st = stamp()
    dirs = []
    for i in range(1, a.n + 1):
        log(f"Run {i}/{a.n}")
        try:
            tune_host("apply")
        except subprocess.CalledProcessError as e:
            print(f"host tuning failed (exit {e.returncode})", file=sys.stderr)
        run_label = f"{a.label}-{st}-run{i}"
        # Within-run noise is reported by `tperf baselines run`; keep going even if it flags something.
        run_module("baselines.run", run_label)
        dirs += sorted((ROOT / "results").glob(f"*-{run_label}"))[-1:]

    compare = run_module("baselines.compare", *dirs, text=True, stdout=subprocess.PIPE)
    print(compare.stdout, end="")
    (ROOT / "results" / f"compare-{a.label}-{st}.txt").write_text(compare.stdout)
    return compare.returncode


if __name__ == "__main__":
    sys.exit(main())
