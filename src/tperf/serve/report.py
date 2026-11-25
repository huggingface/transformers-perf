"""A profiled `transformers serve` run as a time report.

    python -m tperf.serve.report <serve out dir> <profile dir>

Reads results.json and server.folded of `tperf.serve.bench --profile` and writes a profile directory that
tperf.memory.report and tperf.memory.publish read: profile.json (the run's model, commit and scenario, kind "serve"),
python_summary.json (the server's time per thread, by area of code and function), python.folded, then report.html
and time.html. A serve run has no memory pass: its memory report only points at the time report.
"""

import argparse
import json
import shutil
from pathlib import Path


def convert(serve_dir: Path, profile_dir: Path) -> Path:
    serve_dir, profile_dir = Path(serve_dir), Path(profile_dir)
    results = json.loads((serve_dir / "results.json").read_text())
    summary = results.get("profile")
    if not summary:
        raise SystemExit(f"no profile in {serve_dir / 'results.json'} (run tperf.serve.bench with --profile)")
    args = results.get("args") or {}
    tokens = sum(t.get("completion_tokens") or 0 for t in results.get("turns") or [])
    summary["rate"] = args.get("profile_rate", 100)
    summary["serve"] = True
    for group in summary["groups"]:
        group["wall_seconds"] = summary.get("wall_s")
        group["tokens"] = tokens or None
    profile = {
        "model_id": results.get("model_id"),
        "commit_id": results.get("commit"),
        "kind": "serve",
        "serve": {
            "scenario": results.get("scenario"),
            **{k: args.get(k) for k in ("conversations", "turns", "max_tokens", "concurrency", "server_args")},
        },
        "error": results.get("error"),
        "marks": [],
    }
    profile_dir.mkdir(parents=True, exist_ok=True)
    (profile_dir / "profile.json").write_text(json.dumps(profile))
    (profile_dir / "python_summary.json").write_text(json.dumps(summary))
    if (serve_dir / "server.folded").exists():
        shutil.copyfile(serve_dir / "server.folded", profile_dir / "python.folded")

    from tperf.memory.report import build

    return build(profile_dir)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("serve_dir", type=Path)
    p.add_argument("profile_dir", type=Path)
    a = p.parse_args(argv)
    print(convert(a.serve_dir, a.profile_dir))


if __name__ == "__main__":
    main()
