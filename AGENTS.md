# AGENTS.md

## Rules

- **Python, not bash.** All tooling is Python: no `.sh` files. Use `subprocess` with argument lists to call `ssh`,
  `git`, `rsync` and friends, and `argparse` for the command line.
- **One package, one subpackage per feature.** The tooling is the `tperf` package in `src/tperf/`: one subpackage per
  feature (`daily`, `ab`, `generation`, `processors`, `tp`, `baselines`, `host`, `dashboard`), holding its runner, the
  code it runs on the host and its summary. Code shared by several features goes in `tperf.common`; workstation
  commands import `tperf.common.benchlib`. Normal package imports (`from tperf.common.benchlib import ...`), no
  `sys.path` hacks. The generation benchmark is `tperf.harness` (`src/tperf/harness/`, run as
  `python -m tperf.harness.run_benchmarks`). Nothing at the top level but docs, config, `tests/`, `scenarios/` and
  `patches/`. See the layout table in `README.md`.
- **One CLI.** Every command is reached through `tperf <command> [<subcommand>]`: add it to the table in
  `src/tperf/cli.py`. Each module keeps a `main(argv=None)` and stays runnable with `python -m tperf.<feature>.<module>`,
  and keeps its environment-variable interface.
- **Host side.** Each workstation command rsyncs the package (harness included) to its own run directory on the host
  (`push_package(f"{rdir}/src")`) and runs host-side modules as `env PYTHONPATH=<rdir>/src[:...] <python> -m
  tperf.<module>` (`remote_module`). Only `tperf deploy` writes `/opt/bench/src/tperf`, the copy the daily run uses.
  Modules run by the host's system python (`tperf.daily.main`, `tperf.baselines.nccl_launch`) and `tperf.host.prepare`
  / `tperf.host.install` (piped to `sudo python3 -`, so self-contained files) use only the standard library.
  `__init__.py` files stay empty: nothing may import a third-party library at package import time, and modules import
  heavy libraries only where they need them.
- Keep benchmark scenarios as short as possible while still catching the regressions they target.
- `ruff check . && ruff format --check .` and `python -m pytest` before committing; `pip install -e .` (or
  `uv tool install -e .`) for the `tperf` command.
