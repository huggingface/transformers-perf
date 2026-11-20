"""Build the transformers-perf dashboard from the dataset and publish it as a static HF Space.

Pages: index.html (overview), main.html (daily main series: generation, multimodal processors and encoders),
pr-<N>.html (every side-by-side run of PR N) and benchmark-changes.html (side-by-side runs of benchmark
changes on main, no PR).

The daily pages show only what runs every day: a series needs a point in the latest daily run, and a metric a value in
each of the series' points of the last DAILY_WINDOW daily runs (see `every_day`).

    tperf dashboard build --out /tmp/dash [--publish hf-internal-testing/transformers-perf-dashboard]

Runs on the bench host after each daily run. The pages are self-contained: the data is embedded at build time.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from tperf.ab.compare import analyze


HERE = Path(__file__).resolve().parent  # template.html

DATASET = "hf-internal-testing/transformers-perf"  # tperf deploy passes the configured one
# metric key -> (column prefix, label, unit, higher is better)
METRICS = {
    "e2e": ("e2e_ms", "End-to-end latency", "ms", False),
    "ttft": ("ttft_ms", "Time to first token", "ms", False),
    "itl": ("itl_ms", "Inter-token latency", "ms", False),
    "itl_h1": ("itl_first_half_ms", "ITL, first half of tokens", "ms", False),
    "itl_h2": ("itl_second_half_ms", "ITL, second half of tokens", "ms", False),
    "tps": ("tokens_per_s", "Throughput", "tok/s", True),
    "cold": ("startup_first_generate_ms", "First generation (cold start)", "ms", False),
    "mem": ("startup_measurement_peak_memory_mb", "Peak memory", "MB", False),
}

# Processor and encoder series on main.html: metric key -> (label, unit, higher is better). Their cards always show
# "time", not the generation metric picked on the page
PROCESSOR_METRICS = {
    "time": ("Call time", "ms", False),
    "syncs": ("GPU→CPU syncs per call", "syncs", False),
}
DAILY_WINDOW = 7  # daily runs


def num(x) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))


def value(row: dict, prefix: str):
    """A run's value for a metric: the median over its iterations, or the single startup value."""
    v = row.get(prefix if prefix.startswith("startup_") else f"{prefix}_median")
    return v if num(v) else None


def scenario_label(row: dict) -> str:
    parts = [row.get("attn_implementation") or "?"]
    if row.get("compile_mode") is not None:
        parts.append("compiled")
    if "_guard_filter-" in (row.get("scenario") or ""):  # patches/guard-filter-scenarios.patch
        parts.append("guard filter")
    if row.get("kernelize"):
        parts.append("kernels")
    if row.get("continuous_batching"):
        parts.append("continuous batching")
    shape = f"b{row['batch_size']} · {row['sequence_length']}+{row['num_tokens_to_generate']} tokens"
    return " + ".join(parts) + " · " + shape


def load_rows(dataset: str, folder: str = "data") -> list[dict]:
    """Rows of the dataset's <folder>/<month>/*.parquet: data/ (harness), processors/ (tperf.processors.bench)."""
    import pyarrow.dataset as ds
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    files = fs.glob(f"datasets/{dataset}/{folder}/*/*.parquet")
    tables = [ds.dataset(f, filesystem=fs, format="parquet").to_table() for f in files]
    rows = []
    for t in tables:
        rows += t.to_pylist()
    return rows


def main_series(rows: list[dict]) -> list[dict]:
    """Daily main runs, one series per (model, scenario), one point per day (group of runs)."""
    daily = [r for r in rows if (r.get("label") or "").startswith("daily-") and r.get("source") == "main"]
    by_series: dict[str, list[dict]] = defaultdict(list)
    for r in daily:
        by_series[r["series_key"]].append(r)
    series = []
    for key, srows in by_series.items():
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in srows:
            groups[r["group_id"]].append(r)
        points = []
        for gid, g in sorted(groups.items(), key=lambda kv: min(r["started_at"] for r in kv[1])):
            ok = [r for r in g if r["success"]]
            point = {
                "date": min(r["started_at"] for r in g),
                "commit": g[0]["commit_id"],
                "subject": g[0].get("commit_subject") or "",
                "runs": len(g),
                "failed": len(g) - len(ok),
                "values": {},
            }
            for m, (prefix, *_rest) in METRICS.items():
                vals = [v for v in (value(r, prefix) for r in ok) if v is not None]
                if vals:
                    point["values"][m] = [statistics.median(vals), min(vals), max(vals)]
            points.append(point)
        first = srows[0]
        series.append(
            {
                "key": key,
                "group": "Generation",
                "suite": first["label"].removeprefix("daily-"),
                "model": first["model_id"],
                "scenario": scenario_label(first),
                "gpu": f"{first.get('gpu_name')} #{first.get('gpu_index')}",
                "points": points,
            }
        )
    return every_day(sorted(series, key=lambda s: (s["model"], s["scenario"])))


def every_day(series: list[dict]) -> list[dict]:
    """Only what runs every day. A series is kept if it has a point in the latest daily run (a series whose settings
    changed gets a new key and stops getting points), with the metrics that have a value in each of its points of the
    last DAILY_WINDOW daily runs: a scenario that fails some days, or a metric a harness patch did not always record,
    is left out until it has run cleanly for DAILY_WINDOW days in a row."""
    days = sorted({p["date"][:10] for s in series for p in s["points"]})
    if not days:
        return series
    first = days[-DAILY_WINDOW:][0]
    kept = []
    for s in series:
        if s["points"][-1]["date"][:10] != days[-1]:
            continue
        recent = [p for p in s["points"] if p["date"][:10] >= first]
        metrics = set.intersection(*(set(p["values"]) for p in recent))
        if not metrics:
            continue
        for p in s["points"]:
            p["values"] = {k: v for k, v in p["values"].items() if k in metrics}
        kept.append({**s, "metrics": sorted(metrics)})
    return kept


def processor_series(rows: list[dict]) -> list[dict]:
    """Daily tperf.processors.bench runs, one series per scenario, one point per day: the median of the runs' medians and
    their min-max, the GPU -> CPU syncs per call (encoder stages) and the output hashes."""
    daily = [r for r in rows if (r.get("label") or "").startswith("daily-") and r.get("source") == "main"]
    by_name: dict[str, list[dict]] = defaultdict(list)
    for r in daily:
        by_name[r["name"]].append(r)
    series = []
    for name, srows in by_name.items():
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in srows:
            groups[r["group_id"]].append(r)
        points = []
        for gid, g in sorted(groups.items(), key=lambda kv: min(r["started_at"] or "" for r in kv[1])):
            ok = [r for r in g if r["success"] and num(r.get("median_ms"))]
            point = {
                "date": min(r["started_at"] or "" for r in g),
                "commit": g[0]["commit_id"] or "",
                "subject": g[0].get("commit_subject") or "",
                "runs": len(g),
                "failed": len(g) - len(ok),
                "values": {},
            }
            if ok:
                medians = [r["median_ms"] for r in ok]
                point["values"]["time"] = [statistics.median(medians), min(medians), max(medians)]
                syncs = [r["syncs"] for r in ok if r.get("syncs") is not None]
                if syncs:
                    point["values"]["syncs"] = [statistics.median(syncs), min(syncs), max(syncs)]
                    point["sync_sources"] = json.loads(ok[-1].get("sync_sources") or "{}")
                point["outputs"] = sorted({r["output_sha256"] for r in ok if r.get("output_sha256")})
            points.append(point)
        r0 = srows[0]
        stage = r0.get("stage") or "processor"
        parts = [r0["inputs"], f"b{r0['batch']}"]
        if r0.get("component") not in (None, "processor"):
            parts.append(r0["component"])
        if r0.get("backend"):
            parts.append(r0["backend"])
        if stage == "encoder":
            parts.append("encoder")
        elif stage == "both":
            parts.append("processor + encoder")
        where = f"{r0.get('threads')} CPU threads" + (f" · {r0.get('gpu_name')}" if stage != "processor" else "")
        series.append(
            {
                "key": name,
                "group": "Multimodal processors (CPU)" if stage == "processor" else "Multimodal encoders (GPU, bf16)",
                "metric": "time",
                "suite": "processors",
                "model": r0["model"],
                "scenario": " · ".join(parts),
                "gpu": where,
                "points": points,
            }
        )
    order = {"Multimodal processors (CPU)": 0}
    return every_day(sorted(series, key=lambda s: (order.get(s["group"], 1), s["model"], s["scenario"])))


def ab_results(rows: list[dict]) -> list[dict]:
    by_ab: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("ab_id"):
            by_ab[r["ab_id"]].append(r)
    out = []
    for ab_id, arows in by_ab.items():
        result = analyze(arows)
        base = next((r for r in arows if r["ab_side"] == "base"), {})
        head = next((r for r in arows if r["ab_side"] == "head"), {})
        out.append(
            {
                "ab_id": ab_id,
                "date": min(r["started_at"] for r in arows),
                "pr": base.get("pr_number") or head.get("pr_number"),
                "model": arows[0]["model_id"],
                "base_commit": base.get("commit_id"),
                "head_commit": head.get("commit_id"),
                "bench_args": re.sub(
                    r"--config-file \S*/",
                    "--config-file ",
                    arows[0].get("bench_args") or "",
                ),
                "scenarios": [
                    {
                        "scenario": tp_prefix(arows[0])
                        + (scenario_label(e["pairs"][0][1]["base"]) if e["pairs"] else e["scenario"]),
                        "rounds": e["rounds"],
                        "metrics": [
                            {
                                k: m[k]
                                for k in (
                                    "name",
                                    "base",
                                    "head",
                                    "change_pct",
                                    "ci",
                                    "verdict",
                                )
                            }
                            for m in e["metrics"]
                        ],
                        "counters": e["counters"],
                    }
                    for e in result["scenarios"]
                ],
                "failed": [scenario_label_from_name(s) for s in result["failed"]],
            }
        )
    return sorted(out, key=lambda a: a["date"], reverse=True)


def tp_prefix(row: dict) -> str:
    """ "TP2 · " for a tensor-parallel A/B: tperf ab records the torchrun size in bench_args (scenario names say no_tp)."""
    m = re.search(r"--nproc-per-node (\d+)", row.get("bench_args") or "")
    return f"TP{m.group(1)} · " if m else ""


def scenario_label_from_name(name: str) -> str:
    """Readable label from a harness scenario name (failed scenarios have no row fields to build one from)."""
    m = re.match(r"^w\d+_i\d+-(?:un)?monitored-b(\d+)_s(\d+)_n(\d+)-(\w+?)-(.*)$", name)
    if not m:
        return name
    b, s, n, attn, rest = m.groups()
    parts = [attn]
    if "-compiled" in "-" + rest:
        parts.append("compiled")
    if "_guard_filter-" in rest:
        parts.append("guard filter")
    if "-kernelized" in "-" + rest:
        parts.append("kernels")
    if rest.endswith("-cb-no_tp") or "-cb-" in rest:
        parts.append("continuous batching")
    return " + ".join(parts) + f" · b{b} · {s}+{n} tokens"


def pr_titles(numbers) -> dict:
    """PR titles from the public GitHub API (best effort: the pages work without them)."""
    import urllib.request

    titles = {}
    for n in numbers:
        try:
            req = urllib.request.Request(
                f"https://api.github.com/repos/huggingface/transformers/pulls/{n}",
                headers={
                    "Accept": "application/vnd.github+json",
                    "User-Agent": "transformers-perf-dashboard",
                },
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                titles[n] = json.load(r).get("title") or ""
        except Exception as e:  # rate limit, network
            print(f"warning: no title for PR #{n}: {e}", file=sys.stderr)
    return titles


VERDICT_CLASS = {
    "REGRESSION": "regression",
    "improvement": "improvement",
    "no change": "same",
    "inconclusive": "inconclusive",
}


def build(rows: list[dict], proc_rows: list[dict] = ()) -> dict[str, dict]:
    """Page file name -> page data."""
    latest = max(rows, key=lambda r: r["started_at"]) if rows else {}
    common = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "host": f"{latest.get('instance_type', '')} · {latest.get('gpu_name', '')} · SM clock {latest.get('sm_clock_mhz', '')} MHz",
        "metrics": {k: {"label": v[1], "unit": v[2], "higher_is_better": v[3]} for k, v in METRICS.items()},
    }
    series = main_series(rows)
    procs = processor_series(list(proc_rows))
    abs_ = ab_results(rows)
    by_pr: dict[int | None, list[dict]] = defaultdict(list)
    for a in abs_:
        by_pr[int(a["pr"]) if a["pr"] else None].append(a)
    titles = pr_titles(sorted(n for n in by_pr if n is not None))

    pages = {}
    prs = []
    for n, runs in sorted(by_pr.items(), key=lambda kv: max(a["date"] for a in kv[1]), reverse=True):
        file = f"pr-{n}.html" if n else "benchmark-changes.html"
        label = f"PR #{n}" if n else "Benchmark changes"
        counts: dict[str, int] = defaultdict(int)
        for sc in runs[0]["scenarios"]:  # latest run
            for m in sc["metrics"]:
                counts[next(c for k, c in VERDICT_CLASS.items() if m["verdict"].startswith(k))] += 1
        prs.append(
            {
                "file": file,
                "label": label,
                "title": titles.get(n, "") if n else "Benchmark harness changes, main vs main",
                "runs": len(runs),
                "last_date": runs[0]["date"],
                "models": sorted({a["model"] for a in runs}),
                "verdicts": counts,
            }
        )
        heading = f"PR #{n}" + (f": {titles[n]}" if titles.get(n) else "") if n else "Benchmark changes (main vs main)"
        pages[file] = {
            **common,
            "page": {
                "kind": "ab",
                "file": file,
                "nav": label,
                "heading": heading,
                "link": f"https://github.com/huggingface/transformers/pull/{n}" if n else None,
            },
            "abs": runs,
        }
    points = [p for s in series + procs for p in s["points"]]
    last = max(points, key=lambda p: p["date"]) if points else None
    shown = {m for s in series for m in s["metrics"]}
    pages["main.html"] = {
        **common,
        "metrics": {k: v for k, v in common["metrics"].items() if k in shown},
        "series_metrics": {
            k: {"label": v[0], "unit": v[1], "higher_is_better": v[2]} for k, v in PROCESSOR_METRICS.items()
        },
        "page": {
            "kind": "main",
            "file": "main.html",
            "nav": "main",
            "heading": "transformers main, daily",
        },
        "series": series + procs,
    }
    pages["index.html"] = {
        **common,
        "page": {"kind": "index", "file": "index.html", "heading": "transformers perf"},
        "main": {
            "series": len(series) + len(procs),
            "days": len({p["date"][:10] for p in points}),
            "last_date": last["date"] if last else None,
            "last_commit": last["commit"] if last else "",
        },
        "prs": prs,
    }
    for page in pages.values():  # ?page=<file> links resolve to these
        page["files"] = sorted(pages)
    return pages


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=DATASET)
    p.add_argument(
        "--out",
        required=True,
        help="directory to write index.html (and the Space README) to",
    )
    p.add_argument(
        "--publish",
        help="static Space to upload to, e.g. hf-internal-testing/transformers-perf-dashboard",
    )
    a = p.parse_args(argv)

    pages = build(load_rows(a.dataset), load_rows(a.dataset, "processors"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.html"):
        old.unlink()
    template = (HERE / "template.html").read_text(encoding="utf-8")
    for file, data in pages.items():
        payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
        title = "transformers perf" if file == "index.html" else f"{data['page']['nav']} · transformers perf"
        html = template.replace("__TITLE__", title.replace("<", "&lt;")).replace("__DATA__", payload)
        (out / file).write_text(html, encoding="utf-8")
    (out / "README.md").write_text(
        "---\ntitle: transformers perf\nemoji: 📈\ncolorFrom: blue\ncolorTo: gray\nsdk: static\npinned: false\n---\n\n"
        f"Generated from the private dataset `{a.dataset}` by `tperf.dashboard.build` "
        "(huggingface/transformers-perf). Do not edit by hand.\n",
        encoding="utf-8",
    )
    print(f"{len(pages)} pages -> {out}")
    if a.publish:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(
            a.publish,
            repo_type="space",
            space_sdk="static",
            private=False,
            exist_ok=True,
        )
        info = api.upload_folder(
            folder_path=str(out),
            repo_id=a.publish,
            repo_type="space",
            allow_patterns=["*.html", "README.md"],
            delete_patterns=["*.html"],  # pages of A/B runs that no longer exist
            commit_message=f"Update dashboard ({pages['index.html']['generated_at']})",
        )
        print(f"published: https://huggingface.co/spaces/{a.publish} ({info.oid[:10]})")


if __name__ == "__main__":
    main()
