"""Version of the harness (tperf.harness), as recorded with every run (BENCH_HARNESS, the harness_commit column).

The harness travels with the tperf package (push_package, tperf deploy). The transformers under test only comes from
PYTHONPATH (or the editable install in /opt/bench/tbv2-venv), so one harness measures any commit the same way.

Usage:
    tperf harness version   # last commit that changed src/tperf/harness, -dirty with local changes under it
"""

import argparse
import subprocess

from tperf.common.benchlib import PKG_SRC, ROOT


HARNESS_DIR = PKG_SRC / "tperf/harness"


def version() -> str:
    """Last commit of this checkout that changed the harness, -dirty with local changes; a pushed copy's VERSION."""
    if (HARNESS_DIR / "VERSION").is_file():  # a copy pushed to the host (push_package) has no git history
        return (HARNESS_DIR / "VERSION").read_text().strip()
    git = ["git", "-C", str(ROOT)]
    commit = subprocess.run(
        [*git, "log", "-1", "--format=%h", "--", str(HARNESS_DIR)], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run([*git, "diff", "--quiet", "HEAD", "--", str(HARNESS_DIR)]).returncode != 0
    return commit + ("-dirty" if dirty else "")


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    print(version())


if __name__ == "__main__":
    main()
