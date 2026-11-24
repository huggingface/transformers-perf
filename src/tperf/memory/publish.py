"""Publish `tperf memory profile` directories to the dataset, next to the rows of the run they profiled.

    python -m tperf.memory.publish <profile dir> --repo <dataset> --group <group id>

Uploads the report, the summaries it is rendered from (profile.json, gpu_summary.json, host_summary.json), memray's
flamegraph and capture, and the gzipped GPU snapshot (PyTorch's memory viewer) to memory/<YYYY-MM>/<group>/ (the group
is the run's group_id, so the dashboard can link a daily point to its report).
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path


FILES = [
    *("report.html", "host-flamegraph.html", "profile.json", "gpu_summary.json", "host_summary.json"),
    *("host.memray.bin", "gpu_snapshot.pickle.gz"),
]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("profile_dir", type=Path)
    p.add_argument("--repo", required=True)
    p.add_argument("--group", required=True)
    a = p.parse_args(argv)

    from huggingface_hub import HfApi

    if not (a.profile_dir / "report.html").exists():
        raise SystemExit(f"no report.html in {a.profile_dir}")
    month = datetime.now(timezone.utc).strftime("%Y-%m")
    path = f"memory/{month}/{a.group}"
    HfApi().upload_folder(
        folder_path=str(a.profile_dir),
        path_in_repo=path,
        repo_id=a.repo,
        repo_type="dataset",
        allow_patterns=FILES,
        commit_message=f"Memory profile {a.group}",
    )
    print(f"memory profile -> {a.repo}/{path}")


if __name__ == "__main__":
    main()
