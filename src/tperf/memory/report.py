"""Build report.html from a `tperf memory profile` directory: one static, self-contained page.

    python -m tperf.memory.report <profile dir>
    tperf memory report <profile dir>

Reads profile.json (phase counters, the model's tensors), gpu_snapshot.pickle (the CUDA allocator's history),
host.memray.bin (memray) and python_summary.json (py-spy, tperf.common.pystacks), each optional. Shows, per device:
memory over time by category with the phase boundaries, then what held memory at the peak, grouped by the code that
allocated it, with its stack; then where the Python time went, per phase. Also writes host-flamegraph.html (memray's
own) next to it when memray is installed.

Only the standard library is needed for the GPU part; the host part imports memray.
"""

import argparse
import html
import json
import pickle
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


MB = 1024**2
MAX_POINTS = 1200  # timeline points per chart
TOP_SITES = 30

# Categories of a GPU allocation, decided from its stack (first match wins)
CATEGORIES = (
    "weights & buffers",
    "KV cache",
    "compile & CUDA graphs",
    "activations & temporaries",
    "before recording",
)
COLORS = {
    "weights & buffers": "var(--c1)",
    "KV cache": "var(--c2)",
    "compile & CUDA graphs": "var(--c3)",
    "activations & temporaries": "var(--c4)",
    "before recording": "var(--c5)",
}


# ---------------------------------------------------------------------------------------------------------------- GPU


def _frames(entry: dict) -> list[dict]:
    return entry.get("frames") or []


# Frames of the benchmark tool itself (runner, phase markers): not what a reader is looking for, so stacks and
# allocation sites skip them
HIDDEN_PATHS = ("/tperf/",)


def visible(frames: list[dict]) -> list[dict]:
    return [f for f in frames if not any(h in f.get("filename", "") for h in HIDDEN_PATHS)]


def visible_host(frames: list[tuple]) -> list[tuple]:
    return [f for f in frames if not any(h in f[1] for h in HIDDEN_PATHS)]


def strip_hidden_frames(snapshot: dict) -> dict:
    """The snapshot without benchmark-tool frames (in place), for publishing."""
    for seg in snapshot.get("segments", []):
        if "frames" in seg:
            seg["frames"] = visible(seg["frames"])
        for block in seg.get("blocks", []):
            if "frames" in block:
                block["frames"] = visible(block["frames"])
    for trace in snapshot.get("device_traces", []):
        for ev in trace:
            if "frames" in ev:
                ev["frames"] = visible(ev["frames"])
    return snapshot


def short_path(filename: str) -> str:
    for anchor in ("/transformers/", "/tperf/", "/torch/", "/site-packages/"):
        if anchor in filename:
            head = anchor.strip("/") if anchor != "/site-packages/" else ""
            rest = filename.rsplit(anchor, 1)[1]
            return f"{head}/{rest}" if head else rest
    return filename.rsplit("/", 1)[-1]


def frame_label(frame: dict) -> str:
    return f"{short_path(frame.get('filename', '?'))}:{frame.get('line', '?')} {frame.get('name', '?')}"


LOAD_PHASE = 0  # allocations between the first marker ("start") and the second ("loaded")


def gpu_category(frames: list[dict], phase: int | None = None) -> str:
    """What an allocation is, from its stack and the phase it was made in. Loading runs partly in worker threads whose
    stacks do not reach from_pretrained, so everything allocated during the load phase counts as the model's."""
    files = [f.get("filename", "") for f in frames]
    names = [f.get("name", "") for f in frames]
    if not frames:
        return "before recording"
    if any(fn.endswith("cache_utils.py") or ("continuous_batching" in fn and "cache" in fn) for fn in files):
        return "KV cache"
    if phase == LOAD_PHASE or "from_pretrained" in names or any(fn.endswith("core_model_loading.py") for fn in files):
        return "weights & buffers"
    if any("/_inductor/" in fn or "cudagraph" in fn or "/_dynamo/" in fn for fn in files):
        return "compile & CUDA graphs"
    return "activations & temporaries"


def gpu_site(frames: list[dict]) -> str:
    """The line responsible for an allocation: the innermost transformers frame, else the innermost Python frame
    outside torch, else the innermost frame."""
    if not frames:
        return "(allocated before recording started)"
    frames = visible(frames)
    if not frames:
        return "(benchmark harness)"
    for f in frames:
        if "/transformers/" in f.get("filename", ""):
            return frame_label(f)
    for f in frames:
        fn = f.get("filename", "")
        if fn.endswith(".py") and "/torch/" not in fn:
            return frame_label(f)
    return frame_label(frames[0])


def _block_addresses(snapshot: dict) -> dict[int, dict]:
    """Blocks live when the snapshot was taken, by address (a block's address is its segment's plus the sizes of the
    blocks before it)."""
    live = {}
    for seg in snapshot.get("segments", []):
        addr = seg.get("address", 0)
        for block in seg.get("blocks", []):
            if block.get("state") == "active_allocated":
                live[block.get("address", addr)] = block
            addr += block.get("size", 0)
    return live


def analyze_gpu(snapshot: dict, marker_function: str, marker_unit: int, device: int = 0) -> dict:
    """Replay the allocator's trace: allocated and reserved bytes over time, by category, and what is live at the peak.

    Allocations made before recording started and still live at the end (from the snapshot's segments) are counted
    from the start as `before recording`. Phase boundaries are the marker allocations (profile.MARKER_FUNCTION).
    """
    traces = snapshot.get("device_traces") or []
    events = traces[device] if len(traces) > device else []
    end_live = _block_addresses(snapshot)
    reserved_end = sum(s.get("total_size", 0) for s in snapshot.get("segments", []) if s.get("device", 0) == device)

    def is_marker(ev: dict) -> bool:
        return any(f.get("name") == marker_function for f in _frames(ev))

    # Pass 1: which allocations of the trace are still live at its end, and the net reserve change
    live: dict[int, int] = {}
    reserved_delta = 0
    for ev in events:
        action, addr, size = ev.get("action"), ev.get("addr"), ev.get("size", 0)
        if action == "alloc":
            live[addr] = size
        elif action == "free_completed":
            live.pop(addr, None)
        elif action in ("segment_alloc", "segment_map"):
            reserved_delta += size
        elif action in ("segment_free", "segment_unmap"):
            reserved_delta -= size
    baseline = {addr: b for addr, b in end_live.items() if addr not in live}

    def initial_state() -> dict[int, tuple[int, str, list[dict]]]:
        return {addr: (b.get("size", 0), gpu_category(_frames(b)), _frames(b)) for addr, b in baseline.items()}

    phase = [-1]  # index of the last marker seen
    markers: set[int] = set()  # live marker allocations, left out of every total

    def apply(state: dict, ev: dict) -> tuple[int, str] | None:
        """Apply an alloc/free to the live set; return (signed size, category) of the change."""
        action, addr = ev.get("action"), ev.get("addr")
        if action == "alloc" and is_marker(ev):
            phase[0] = round(ev.get("size", 0) / marker_unit) - 1
            markers.add(addr)
            return None
        if action == "free_completed" and addr in markers:
            markers.discard(addr)
            return None
        if action == "alloc":
            frames = _frames(ev)
            cat = gpu_category(frames, phase[0])
            state[addr] = (ev.get("size", 0), cat, frames)
            return ev.get("size", 0), cat
        if action == "free_completed":
            item = state.pop(addr, None)
            if item is None:
                return 0, ""  # its alloc fell out of the ring buffer
            return -item[0], item[1]
        return None

    # Pass 2: totals per event, and the peak
    state = initial_state()
    phase[0] = -1
    by_cat = defaultdict(int)
    for size, cat, _ in state.values():
        by_cat[cat] += size
    allocated = sum(by_cat.values())
    reserved = max(reserved_end - reserved_delta, allocated)
    use_time = bool(events) and all("time_us" in ev for ev in events)
    t0 = events[0]["time_us"] if use_time else 0

    def x_of(i: int, ev: dict) -> float:
        return (ev["time_us"] - t0) / 1e6 if use_time else float(i)

    xs, alloc_series, reserved_series = [], [], []
    cat_series: dict[str, list[int]] = {c: [] for c in CATEGORIES}
    # One point per bin of the x axis (time, or event index without timestamps): the bin's highest allocated total,
    # so no spike is averaged away; a bin without events repeats the state the previous one ended in
    span = (x_of(len(events) - 1, events[-1]) if events else 0.0) or 1.0
    bins: list[tuple | None] = [None] * MAX_POINTS
    start_state = (allocated, reserved, dict(by_cat))
    ends: list[tuple | None] = [None] * MAX_POINTS
    peak = {"bytes": allocated, "index": -1, "x": None}
    unknown_frees = 0
    phase_x: list[tuple[int, float]] = []

    for i, ev in enumerate(events):
        action, size = ev.get("action"), ev.get("size", 0)
        if action == "alloc" and is_marker(ev):
            phase_x.append((round(size / marker_unit) - 1, x_of(i, ev)))
        change = apply(state, ev)
        if change is not None:
            if change == (0, ""):
                unknown_frees += 1
            else:
                by_cat[change[1]] += change[0]
                allocated += change[0]
        elif action in ("segment_alloc", "segment_map"):
            reserved += size
        elif action in ("segment_free", "segment_unmap"):
            reserved -= size
        if allocated > peak["bytes"]:
            peak = {"bytes": allocated, "index": i, "x": x_of(i, ev)}
        b = min(MAX_POINTS - 1, int(x_of(i, ev) / span * MAX_POINTS))
        if bins[b] is None or allocated > bins[b][0]:
            bins[b] = (allocated, reserved, dict(by_cat))
        ends[b] = (allocated, reserved, dict(by_cat))

    previous = start_state
    for b in range(MAX_POINTS if events else 0):
        point = bins[b] if bins[b] is not None else previous
        if bins[b] is not None:
            previous = ends[b]
        xs.append(b * span / MAX_POINTS)
        alloc_series.append(point[0])
        reserved_series.append(point[1])
        for c in CATEGORIES:
            cat_series[c].append(point[2].get(c, 0))

    # Pass 3: the live set at the peak
    state = initial_state()
    phase[0] = -1
    for ev in events[: peak["index"] + 1]:
        apply(state, ev)
    sites: dict[tuple[str, str], dict] = {}
    peak_by_cat = defaultdict(int)
    for size, cat, frames in state.values():
        peak_by_cat[cat] += size
        key = (gpu_site(frames), cat)
        s = sites.setdefault(key, {"site": key[0], "category": cat, "bytes": 0, "count": 0, "largest": 0, "stack": []})
        s["bytes"] += size
        s["count"] += 1
        if size > s["largest"]:
            s["largest"], s["stack"] = size, [frame_label(f) for f in visible(frames)[:25]]
    return {
        "events": len(events),
        # Frees of allocations the trace never saw: the ring buffer wrapped, and the start of the run is missing
        "truncated": unknown_frees > 0,
        "time_axis": use_time,
        "x": xs,
        "allocated": alloc_series,
        "reserved": reserved_series,
        "by_category": cat_series,
        "peak_bytes": peak["bytes"],
        "peak_x": peak.get("x"),
        "peak_by_category": dict(peak_by_cat),
        "peak_sites": sorted(sites.values(), key=lambda s: -s["bytes"]),
        "phases": [list(p) for p in phase_x],
        "baseline_bytes": sum(b.get("size", 0) for b in baseline.values()),
    }


def gpu_summary(snapshot: dict, marker_function: str, marker_unit: int) -> dict:
    """What the report needs from a snapshot, small enough to keep and publish (the snapshot is hundreds of MB)."""
    return analyze_gpu(snapshot, marker_function, marker_unit)


# --------------------------------------------------------------------------------------------------------------- host


def host_site(frames: list[tuple]) -> str:
    """memray frames are (function, file, line), innermost first."""
    frames = visible_host(frames)
    for fn, file, line in frames:
        if "/transformers/" in file:
            return f"{short_path(file)}:{line} {fn}"
    for fn, file, line in frames:
        if file.endswith(".py") and "/torch/" not in file and "/memray/" not in file:
            return f"{short_path(file)}:{line} {fn}"
    for fn, file, line in frames:
        if file.endswith(".py"):
            return f"{short_path(file)}:{line} {fn}"
    if frames:
        fn, file, line = frames[0]
        return f"{short_path(file)}:{line} {fn}"
    return "(no stack)"


def host_phase(frames: list[tuple]) -> str:
    names = {fn for fn, _, _ in frames}
    if "from_pretrained" in names:
        return "load"
    if names & {"generate", "generate_batch", "time_generate"}:
        return "generation"
    return "other"


def allocator_name(allocator) -> str:
    try:
        from memray import AllocatorType

        return AllocatorType(allocator).name.lower()
    except Exception:
        return getattr(allocator, "name", str(allocator)).lower()


def analyze_host(path: Path) -> dict:
    from memray import FileReader

    reader = FileReader(str(path))
    meta = reader.metadata
    native = getattr(meta, "has_native_traces", False)
    snaps = list(reader.get_memory_snapshots())
    step = max(1, len(snaps) // MAX_POINTS)
    pts = [max(snaps[i : i + step], key=lambda s: s.rss) for i in range(0, len(snaps), step)]
    sites: dict[tuple[str, str], dict] = {}
    by_allocator = defaultdict(int)
    by_phase = defaultdict(int)
    for rec in reader.get_high_watermark_allocation_records(merge_threads=True):
        try:
            frames = list(rec.hybrid_stack_trace(max_stacks=60) if native else rec.stack_trace(max_stacks=60))
        except Exception:
            frames = list(rec.stack_trace(max_stacks=60))
        allocator = allocator_name(rec.allocator)
        by_allocator[allocator] += rec.size
        phase = host_phase(frames)
        by_phase[phase] += rec.size
        site = host_site(frames)
        kind = "mmap" if allocator == "mmap" else "heap"
        s = sites.setdefault((site, kind), {"site": site, "kind": kind, "category": phase, "bytes": 0, "count": 0,
                                            "largest": 0, "allocators": set(), "stack": []})  # fmt: skip
        s["bytes"] += rec.size
        s["count"] += rec.n_allocations
        s["allocators"].add(allocator)
        if rec.size > s["largest"]:
            s["largest"], s["stack"] = (
                rec.size,
                [f"{short_path(f)}:{ln} {fn}" for fn, f, ln in visible_host(frames)[:25]],
            )
    for s in sites.values():
        s["allocators"] = sorted(s["allocators"])
    ordered = sorted(sites.values(), key=lambda s: -s["bytes"])
    start_ms = meta.start_time.timestamp() * 1e3
    return {
        "peak_bytes": meta.peak_memory,
        "total_allocations": meta.total_allocations,
        "native": native,
        "start_ms": start_ms,
        "x": [(p.time - start_ms) / 1e3 for p in pts],
        "rss": [p.rss for p in pts],
        "heap": [p.heap for p in pts],
        "by_allocator": dict(by_allocator),
        "by_phase": dict(by_phase),
        # mmap is mostly mapped files (safetensors) and the CUDA driver's own mappings: address space, resident only
        # where touched; the rest is the heap
        "heap_sites": [s for s in ordered if s["kind"] == "heap"],
        "mmap_sites": [s for s in ordered if s["kind"] == "mmap"],
    }


def memray_flamegraph(path: Path, out: Path) -> bool:
    out.unlink(missing_ok=True)
    rc = subprocess.run([sys.executable, "-m", "memray", "flamegraph", "-f", "-o", str(out), str(path)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode  # fmt: skip
    return rc == 0 and out.exists()


# --------------------------------------------------------------------------------------------------------------- HTML


def esc(x: object) -> str:
    return html.escape(str(x))


def fmt_mb(value: float | None, digits: int = 0) -> str:
    if value is None:
        return "–"
    if abs(value) >= 1024:
        return f"{value / 1024:,.2f} GB"
    return f"{value:,.{digits}f} MB"


def fmt_bytes(value: float | None) -> str:
    return "–" if value is None else fmt_mb(value / MB, 1 if abs(value) < 10 * MB else 0)


def _nice_max(v: float) -> float:
    if v <= 0:
        return 1.0
    import math

    exp = 10 ** math.floor(math.log10(v))
    for m in (1, 2, 2.5, 5, 10):
        if m * exp >= v:
            return m * exp
    return 10 * exp


def area_chart(xs: list[float], layers: list[tuple[str, list[float], str]], lines: list[tuple[str, list[float], str]],
               marks: list[tuple[float, str]], x_label: str, peak: tuple[float, float] | None = None) -> str:  # fmt: skip
    """Stacked areas (bytes) plus overlay lines, with vertical phase marks. Inline SVG, sized by its viewBox."""
    if not xs:
        return '<p class="muted">No samples.</p>'
    w, h, ml, mr, mt, mb = 960, 300, 64, 16, 14, 34
    pw, ph = w - ml - mr, h - mt - mb
    x0, x1 = min(xs), max([*xs, *(x for x, _ in marks)])  # a mark at the very end stays in range
    if x1 == x0:
        x1 = x0 + 1
    stacked_top = [sum(layer[1][i] for layer in layers) for i in range(len(xs))]
    ymax = _nice_max(max([*stacked_top, *(max(v) for _, v, _ in lines if v), 1]) / MB) * MB

    def sx(x: float) -> float:
        return ml + (x - x0) / (x1 - x0) * pw

    def sy(y: float) -> float:
        return mt + ph - y / ymax * ph

    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="memory over time">']
    for k in range(5):
        y = ymax * k / 4
        out.append(f'<line x1="{ml}" x2="{w - mr}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" class="grid"/>')
        out.append(f'<text x="{ml - 8}" y="{sy(y) + 4:.1f}" class="tick" text-anchor="end">{esc(fmt_bytes(y))}</text>')
    for k in range(6):
        x = x0 + (x1 - x0) * k / 5
        out.append(f'<text x="{sx(x):.1f}" y="{h - 12}" class="tick" text-anchor="middle">{x:,.1f}</text>')
    out.append(f'<text x="{w - mr}" y="{h - 1}" class="tick" text-anchor="end">{esc(x_label)}</text>')
    base = [0.0] * len(xs)
    for name, values, color in layers:
        top = [b + v for b, v in zip(base, values)]
        pts = [f"{sx(x):.1f},{sy(t):.1f}" for x, t in zip(xs, top)]
        pts += [f"{sx(x):.1f},{sy(b):.1f}" for x, b in reversed(list(zip(xs, base)))]
        out.append(f'<polygon points="{" ".join(pts)}" style="fill:{color}" class="area"><title>{esc(name)}</title>'
                   "</polygon>")  # fmt: skip
        base = top
    for name, values, color in lines:
        pts = " ".join(f"{sx(x):.1f},{sy(v):.1f}" for x, v in zip(xs, values))
        out.append(f'<polyline points="{pts}" style="stroke:{color}" class="line"><title>{esc(name)}</title>'
                   "</polyline>")  # fmt: skip
    taken: list[tuple[float, float, int]] = []  # (left, right, row) of the labels placed so far
    for x, label in sorted(marks):
        if x0 <= x <= x1:
            out.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{mt}" y2="{mt + ph}" class="mark"/>')
            width = 6.5 * len(label) + 4  # 11px text, roughly
            right = sx(x) > w - mr - width  # a label near the right edge goes on the left of its line
            left_px = sx(x) - width if right else sx(x)
            # Labels of phases close together go on the next row down instead of on top of each other
            row = 0
            while any(r == row and left_px < b and a < left_px + width for a, b, r in taken):
                row += 1
            taken.append((left_px, left_px + width, row))
            tx, anchor = (sx(x) - 4, "end") if right else (sx(x) + 4, "start")
            out.append(
                f'<text x="{tx:.1f}" y="{mt + 10 + 13 * row}" class="mark-label" text-anchor="{anchor}">'
                f"{esc(label)}</text>"
            )
    if peak is not None and peak[0] is not None:
        px, py = sx(peak[0]), sy(peak[1])
        out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" class="peak"><title>peak {esc(fmt_bytes(peak[1]))}'
                   "</title></circle>")  # fmt: skip
    out.append("</svg>")
    return "".join(out)


def legend(items: list[tuple[str, str, str]]) -> str:
    return (
        '<div class="legend">'
        + "".join(f'<span><i style="background:{c}" class="{cls}"></i>{esc(n)}</span>' for n, c, cls in items)
        + "</div>"
    )


def bars(items: dict[str, float], total: float, colors: dict[str, str] | None = None, unit_bytes: bool = True) -> str:
    rows = []
    for name, value in sorted(items.items(), key=lambda kv: -kv[1]):
        if value <= 0:
            continue
        pct = value / total * 100 if total else 0
        color = (colors or {}).get(name, "var(--accent)")
        shown = fmt_bytes(value) if unit_bytes else fmt_mb(value)
        rows.append(
            f'<div class="bar-row"><span class="bar-name">{esc(name)}</span>'
            f'<span class="bar-track"><span class="bar-fill" style="width:{pct:.1f}%;background:{color}"></span></span>'
            f'<span class="bar-val">{esc(shown)} <small>{pct:.0f}%</small></span></div>'
        )
    return "".join(rows) or '<p class="muted">Nothing recorded.</p>'


def sites_table(sites: list[dict], total: float, show_allocators: bool = False) -> str:
    rows = []
    for s in sites[:TOP_SITES]:
        stack = "\n".join(s["stack"])
        extra = f"<td>{esc(', '.join(s['allocators']))}</td>" if show_allocators else ""
        pct = s["bytes"] / total * 100 if total else 0
        rows.append(
            f"<tr><td class='num'>{esc(fmt_bytes(s['bytes']))}</td><td class='num'>{pct:.1f}%</td>"
            f"<td class='num'>{s['count']:,}</td><td>{esc(s['category'])}</td>{extra}"
            f"<td><details><summary><code>{esc(s['site'])}</code></summary><pre>{esc(stack)}</pre></details></td></tr>"
        )
    more = f"<p class='muted'>{len(sites) - TOP_SITES} smaller sites not shown.</p>" if len(sites) > TOP_SITES else ""
    head = "<th>Allocator</th>" if show_allocators else ""
    return (
        "<div class='table-wrap'><table><thead><tr><th class='num'>Live at peak</th><th class='num'>Share</th>"
        f"<th class='num'>Blocks</th><th>Category</th>{head}<th>Allocated at (click for the stack)</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>{more}"
    )


def phase_table(marks: list[dict]) -> str:
    if len(marks) < 2:
        return ""
    rows = []
    for prev, cur in zip(marks, marks[1:]):
        h0, h1 = prev.get("host") or {}, cur.get("host") or {}
        d1 = cur.get("device") or {}

        def delta(key: str) -> str:
            if h0.get(key) is None or h1.get(key) is None:
                return "–"
            return fmt_mb((h1[key] - h0[key]) / MB, 1)

        rows.append(
            f"<tr><td>{esc(prev['name'])} → {esc(cur['name'])}</td>"
            f"<td class='num'>{(cur['time_ms'] - prev['time_ms']) / 1e3:,.1f} s</td>"
            f"<td class='num'>{esc(fmt_mb(d1.get('allocated_mb')))}</td>"
            f"<td class='num'>{esc(fmt_mb(d1.get('peak_reserved_mb')))}</td>"
            f"<td class='num'>{esc(fmt_mb(cur.get('process_gpu_mb')))}</td>"
            f"<td class='num'>{esc(fmt_mb(h1.get('rss_anon_mb')))}</td>"
            f"<td class='num'>{esc(fmt_mb(h1.get('rss_file_mb')))}</td>"
            f"<td class='num'>{esc(delta('read_bytes'))}</td><td class='num'>{esc(delta('rchar'))}</td></tr>"
        )
    return (
        "<div class='table-wrap'><table><thead><tr><th>Phase</th><th class='num'>Wall time*</th>"
        "<th class='num'>GPU allocated after</th><th class='num'>GPU reserved peak†</th>"
        "<th class='num'>GPU, whole process (NVML)</th><th class='num'>Host RSS anon</th>"
        "<th class='num'>Host RSS file-backed</th><th class='num'>Read from disk</th><th class='num'>Read, any source</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
        "<p class='note'>* Recording slows allocations down: these durations are not benchmark timings. "
        "† Since recording started (peaks are not reset between phases here). "
        "<em>Read from disk</em>: bytes fetched from storage for this process, through read calls or page faults on "
        "mapped files (safetensors); about the model's size means a cold load, about zero means the page cache served "
        "it. <em>Read, any source</em> counts read calls only, so it misses mapped files. Host columns come from the "
        "host pass, device columns are unaffected by it.</p>"
    )


def tile(label: str, value: str, sub: str = "") -> str:
    return (
        f'<div class="tile"><div class="tile-label">{esc(label)}</div><div class="tile-value">{esc(value)}</div>'
        + (f'<div class="tile-sub">{esc(sub)}</div>' if sub else "")
        + "</div>"
    )


# Colours as tokens: standalone pages define them (light and dark); a host page maps them onto its own (SITE_TOKENS)
TOKENS = """
:root{--bg:#fbfbfa;--panel:#fff;--text:#1d1d1f;--muted:#6b6b70;--line:#e4e4e7;--grid:#ececef;--accent:#3b6fd8;
--c1:#3b6fd8;--c2:#d9822b;--c3:#8a5cd1;--c4:#2a9d8f;--c5:#9a9aa1;--peak:#d1495b;--code:#f3f3f5}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141416;--panel:#1c1c1f;--text:#ececef;
--muted:#9c9ca3;--line:#2e2e33;--grid:#26262b;--accent:#6e9bf0;--c1:#6e9bf0;--c2:#f0a35e;--c3:#b08ef0;--c4:#4fc3b4;
--c5:#77777f;--peak:#f0717f;--code:#232327}}
:root[data-theme="dark"]{--bg:#141416;--panel:#1c1c1f;--text:#ececef;--muted:#9c9ca3;--line:#2e2e33;--grid:#26262b;
--accent:#6e9bf0;--c1:#6e9bf0;--c2:#f0a35e;--c3:#b08ef0;--c4:#4fc3b4;--c5:#77777f;--peak:#f0717f;--code:#232327}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.mem{max-width:1100px;margin:0 auto;padding:28px 16px 64px}
"""
# Inside a page that has --bg, --panel, --fg, --muted, --line, --accent, --hover and --bad (the Transformers CI
# dashboard): mid-tone series colours that read on light and dark
SITE_TOKENS = """
.mem{--text:var(--fg);--grid:var(--line);--code:var(--hover);--peak:var(--bad);
--c1:#5b8def;--c2:#e8964a;--c3:#a07ae6;--c4:#3fb3a4;--c5:#8d8d96}
"""
RULES = """
.mem h1{font-size:22px;margin:0 0 4px}.mem h2{font-size:17px;margin:36px 0 10px}
.mem h3{font-size:14px;margin:20px 0 8px;color:var(--muted);font-weight:600}
.mem .muted,.mem .note{color:var(--muted)}.mem .note{font-size:12.5px}
.mem code,.mem pre{font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}
.mem pre{background:var(--code);padding:10px;border-radius:6px;overflow-x:auto;margin:6px 0}
.mem .meta{color:var(--muted);font-size:13px}.mem .meta code{color:var(--text)}
.mem .error{border:1px solid var(--peak);color:var(--peak);padding:8px 12px;border-radius:6px;margin:12px 0}
.mem .tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px;margin-top:18px}
.mem .tile{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.mem .tile-label{font-size:12px;color:var(--muted)}.mem .tile-value{font-size:20px;font-weight:600;margin-top:2px}
.mem .tile-sub{font-size:12px;color:var(--muted)}
.mem .card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px}
.mem .chart{width:100%;height:auto;display:block}.mem .grid{stroke:var(--grid)}
.mem .tick{fill:var(--muted);font-size:11px}.mem .area{opacity:.85}
.mem .line{fill:none;stroke-width:1.5;stroke-dasharray:4 3}.mem .mark{stroke:var(--muted);stroke-dasharray:2 3}
.mem .mark-label{fill:var(--muted);font-size:11px}.mem .peak{fill:var(--peak)}
.mem .legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:12.5px;margin:4px 0 8px}
.mem .legend i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.mem .legend i.dash{height:0;border-top:2px dashed;background:none!important;width:14px;vertical-align:3px}
.mem .bar-row{display:grid;grid-template-columns:minmax(120px,260px) 1fr 130px;gap:10px;align-items:center;
margin:4px 0;font-size:13px}.mem .bar-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mem .bar-track{background:var(--grid);border-radius:3px;height:10px;overflow:hidden}
.mem .bar-fill{display:block;height:100%}.mem .bar-val{text-align:right;font-variant-numeric:tabular-nums}
.mem .bar-val small{color:var(--muted)}.mem .cols{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media (max-width:760px){.mem .cols{grid-template-columns:1fr}.mem .bar-row{grid-template-columns:110px 1fr 100px}}
.mem .table-wrap{overflow-x:auto}.mem table{border-collapse:collapse;width:100%;font-size:13px}
.mem th,.mem td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top;
white-space:normal}.mem th{color:var(--muted);font-weight:600;font-size:12px}
.mem .num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}.mem summary{cursor:pointer}
.mem summary code{white-space:normal;word-break:break-all}.mem a{color:var(--accent)}
.mem .findings{border:1px solid var(--line);border-left:3px solid var(--c2);border-radius:8px;padding:4px 14px 8px;
margin:16px 0;background:var(--panel)}.mem .findings.ok{border-left-color:var(--c4);padding:10px 14px}
.mem .findings h2{margin:8px 0 6px}.mem .findings ul{list-style:none;margin:0;padding:0}
.mem .findings li{display:flex;gap:10px;align-items:flex-start;padding:6px 0;border-top:1px solid var(--line)}
.mem .findings li:first-child{border-top:0}.mem .badge{font-size:11px;border-radius:3px;padding:1px 6px;
margin-top:2px;white-space:nowrap;border:1px solid var(--muted);color:var(--muted)}
.mem .findings li.warn .badge{border-color:var(--c2);color:var(--c2)}
.mem .explore{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0}
.mem .explore a{border:1px solid var(--line);border-radius:6px;padding:8px 12px;text-decoration:none;
background:var(--panel)}.mem .explore small{display:block;color:var(--muted);font-size:12px}
"""


HALF_PRECISION = ("bfloat16", "float16")


def findings(
    profile: dict, gpu: dict | None, host: dict | None, python: dict | None = None
) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Things worth a look, from the counters and summaries: ([(level, title, detail)], [checks that passed]).

    Each check is a plain threshold on numbers the report shows; level is "warn" (likely a bug or a regression to
    chase) or "info" (unusual, sometimes expected). Thresholds are deliberately loose: a finding should be worth
    reading on any model."""
    found: list[tuple[str, str, str]] = []
    passed: list[str] = []
    weights = profile.get("weights") or {}
    load = profile.get("load_memory") or {}
    marks = {m["name"]: m for m in profile.get("marks") or []}
    w = weights.get("weights_mb") or 0.0

    def check(name: str, hit: tuple[str, str, str] | None) -> None:
        if hit:
            found.append(hit)
        else:
            passed.append(name)

    # Model tensors in a second dtype: an fp32 norm, router or buffer in a bf16 model doubles its size
    by_dtype = weights.get("by_dtype_mb") or {}
    main = max(by_dtype, key=by_dtype.get) if by_dtype else None
    minor = {d: mb for d, mb in by_dtype.items() if d != main and mb >= max(1.0, 0.01 * w)}
    hit = None
    if minor:
        dtype, mb = max(minor.items(), key=lambda kv: kv[1])
        modules = (weights.get("minor_dtype_modules_mb") or {}).get(dtype) or {}
        where = ", ".join(f"{g} ({fmt_mb(v, 1)})" for g, v in list(modules.items())[:4])
        level = "warn" if main in HALF_PRECISION and dtype in ("float32", "float64") else "info"
        hit = (level, f"{fmt_mb(mb, 1)} of model tensors in {dtype}, the rest in {main}",
               f"In: {where}." if where else "Upcast on load, or kept in full precision on purpose?")  # fmt: skip
    check("model tensors in one dtype", hit)

    cpu = (weights.get("by_device_mb") or {}).get("cpu", 0)
    check("weights on the GPU", ("warn", f"{fmt_mb(cpu, 1)} of model tensors left on the CPU",
          "Offloaded or never moved: every forward copies them, or runs them on the CPU.") if cpu >= 1 else None)  # fmt: skip

    # Host RAM while loading, above what the process held before: about 1× the model is a full copy in host memory
    start_rss = ((marks.get("start") or {}).get("host") or {}).get("rss_mb")
    peak = load.get("load_host_peak_rss_mb")
    hit = None
    if w and start_rss is not None and peak is not None and (peak - start_rss) > 1.5 * w:
        added = peak - start_rss
        hit = ("warn", f"Loading took {fmt_mb(added)} of host RAM at its peak, {added / w:.1f}× the model",
               "Weights materialized in host memory before moving to the GPU (a copy per tensor, or the whole state "
               "dict at once). See the mapped and heap tables below.")  # fmt: skip
    check("host RAM while loading", hit)

    disk = load.get("load_disk_read_mb")
    check("reads while loading", ("warn", f"Loading read {fmt_mb(disk)} from disk, {disk / w:.1f}× the model",
          "Files read more than once, or files the model does not need.")
          if w and disk is not None and disk > 1.5 * w else None)  # fmt: skip

    # Leaks across generations: memory_metrics flushes the cache after each generation, so what stays is held
    first = marks.get("first generation") or {}
    last = marks.get("generations") or {}
    a0, a1 = (first.get("device") or {}).get("allocated_mb"), (last.get("device") or {}).get("allocated_mb")
    check("GPU memory across generations", ("warn", f"{fmt_mb(a1 - a0, 1)} more GPU memory held after the later "
          "generations than after the first", "Something keeps tensors between generate calls (a cache, a "
          "reference in a module attribute).") if a0 is not None and a1 is not None and a1 - a0 > 8 else None)  # fmt: skip
    h0, h1 = (first.get("host") or {}).get("rss_anon_mb"), (last.get("host") or {}).get("rss_anon_mb")
    check("host memory across generations", ("warn", f"Host memory grew by {fmt_mb(h1 - h0)} across the later "
          "generations", "Python or native objects kept between generate calls.")
          if h0 is not None and h1 is not None and h1 - h0 > 64 else None)  # fmt: skip

    device = last.get("device") or (marks.get("error") or {}).get("device") or {}
    retries = device.get("alloc_retries") or 0
    check("allocator retries", ("warn", f"The caching allocator retried {retries} allocation(s)",
          "It had to free its cache to satisfy a request: memory pressure or fragmentation, and slow.")
          if retries else None)  # fmt: skip

    if gpu:
        peak_b = gpu["peak_bytes"]
        reserved = device.get("peak_reserved_mb")
        slack = reserved - peak_b / MB if reserved is not None else None
        check("reserved but unused", ("info", f"{fmt_mb(slack)} reserved but unused at the peak",
              "Fragmentation in the caching allocator: capacity lost before memory runs out "
              "(PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True often recovers it).")
              if slack is not None and slack > max(256, 0.1 * peak_b / MB) else None)  # fmt: skip
        act = [s for s in gpu["peak_sites"] if s["category"] == "activations & temporaries"]
        big = max(act, key=lambda s: s["largest"], default=None)
        check("large temporaries", ("warn", f"One temporary of {fmt_bytes(big['largest'])} at the peak",
              f"Allocated at {big['site']}: a full-size intermediate (attention scores, logits over the whole "
              "sequence, an upcast copy)?")
              if big and big["largest"] / MB > max(256, 0.1 * w) else None)  # fmt: skip
        temps = (gpu.get("peak_by_category") or {}).get("activations & temporaries", 0) / MB
        check("activations vs model size", ("warn", f"Activations and temporaries ({fmt_mb(temps)}) exceed the "
              "model at the peak", "For this batch and length, generation needs more memory than the weights.")
              if w and temps > w else None)  # fmt: skip
        compiled = (gpu.get("peak_by_category") or {}).get("compile & CUDA graphs", 0) / MB
        check("compile and CUDA graph memory", ("info", f"{fmt_mb(compiled)} held by compiled code and CUDA graphs",
              "Static inputs and graph pools; each recompile can add more.")
              if w and compiled > 0.25 * w else None)  # fmt: skip
        check("complete GPU trace", ("warn", "The GPU trace is incomplete (its ring buffer wrapped)",
              "Totals undercount early allocations: rerun with a larger --max-entries.")
              if gpu.get("truncated") else None)  # fmt: skip
    process, reserved_now = last.get("process_gpu_mb"), device.get("reserved_mb")
    outside = process - reserved_now if process is not None and reserved_now is not None else None
    check("GPU memory outside torch", ("info", f"{fmt_mb(outside)} of GPU memory outside torch's allocator",
          "Usually the CUDA context (a few hundred MB): more means library workspaces, another allocator, or a "
          "second context.") if outside is not None and outside > 1024 else None)  # fmt: skip

    gens = python_group(python, "generations")
    if gens and gens["seconds"]:
        compiling = gens["by_area"].get("torch.compile", 0)
        check("no compilation in the measured generations", ("warn", f"{compiling:.2f} s of the later generations in "
              "torch.compile code", "Recompiling or guard checks after the first generation: a dynamic shape, or a "
              "guard that changes between calls.") if compiling > 0.05 * gens["seconds"] else None)  # fmt: skip
    return found, passed


def findings_box(profile: dict, gpu: dict | None, host: dict | None, python: dict | None = None) -> str:
    found, passed = findings(profile, gpu, host, python)
    if not found:
        return ("<div class='findings ok'><strong>Nothing unusual.</strong> <span class='muted'>Checked: "
                + esc(", ".join(passed)) + ".</span></div>")  # fmt: skip
    found.sort(key=lambda f: f[0] != "warn")
    items = "".join(
        f"<li class='{esc(level)}'><span class='badge'>{'check' if level == 'warn' else 'note'}</span>"
        f"<div><strong>{esc(title)}</strong><br><span class='muted'>{esc(detail)}</span></div></li>"
        for level, title, detail in found
    )
    return (f"<div class='findings'><h2>Worth a look</h2><ul>{items}</ul>"
            f"<p class='note'>Also checked, nothing found: {esc(', '.join(passed))}.</p></div>")  # fmt: skip


# ------------------------------------------------------------------------------------------------------------- Python

PYTHON_PHASES = ("generations", "first generation", "load", "other")
TOP_FUNCTIONS = 25


def python_group(python: dict | None, name: str) -> dict | None:
    return next((g for g in (python or {}).get("groups", []) if g["name"] == name), None)


def time_bars(items: dict[str, float], total: float) -> str:
    rows = []
    for name, sec in sorted(items.items(), key=lambda kv: -kv[1]):
        pct = sec / total * 100 if total else 0
        if pct < 0.5:
            continue
        rows.append(
            f'<div class="bar-row"><span class="bar-name">{esc(name)}</span>'
            f'<span class="bar-track"><span class="bar-fill" style="width:{pct:.1f}%;background:var(--accent)"></span>'
            f'</span><span class="bar-val">{sec:,.2f} s <small>{pct:.0f}%</small></span></div>'
        )
    return "".join(rows) or '<p class="muted">Nothing recorded.</p>'


def functions_table(rows: list[dict], total: float, tokens: int | None, what: str) -> str:
    body = []
    for r in rows[:TOP_FUNCTIONS]:
        pct = r["seconds"] / total * 100 if total else 0
        per_token = f"{r['seconds'] / tokens * 1e3:,.3f}" if tokens else "–"
        body.append(f"<tr><td class='num'>{r['seconds']:,.3f} s</td><td class='num'>{pct:.1f}%</td>"
                    f"<td class='num'>{per_token}</td><td>{esc(r['area'])}</td>"
                    f"<td><code>{esc(r['function'])}</code></td></tr>")  # fmt: skip
    return (
        "<div class='table-wrap'><table><thead><tr><th class='num'>Time</th><th class='num'>Share</th>"
        f"<th class='num'>ms / token</th><th>Area</th><th>{esc(what)}</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def python_section(python: dict | None, links: dict) -> str:
    parts = ["<h2>Python time</h2>"]
    if not python:
        parts.append("<p class='muted'>No python pass (run without <code>--no-python</code>, with py-spy installed)."
                     "</p>")  # fmt: skip
        return "".join(parts)
    parts.append(
        f"<p class='note'>py-spy sampled every thread of a separate run at {python['rate']} Hz from outside the "
        "process (no hooks, no GIL taken), so these durations are close to real ones, though not benchmark timings: "
        "py-spy uses a CPU. Time inside a C call (a torch "
        "op, a CUDA synchronization) is charged to the Python line that made it"
        + (", with native frames below it" if python.get("native") else "")
        + ". Threads add up: sampled time can exceed wall time. Per token: divided by the tokens the phase generated "
        "(batch × new tokens × generations).</p>"
    )
    groups = sorted(python.get("groups", []), key=lambda g: PYTHON_PHASES.index(g["name"])
                    if g["name"] in PYTHON_PHASES else len(PYTHON_PHASES))  # fmt: skip
    for g in groups:
        tokens = g.get("tokens")
        wall = g.get("wall_seconds")
        tiles = [tile("Sampled", f"{g['seconds']:,.2f} s", f"{g['samples']:,} samples")]
        if wall:
            tiles.append(tile("Wall time", f"{wall:,.2f} s", "between the phase's boundaries"))
        if tokens and wall:
            tiles.append(tile("Per token", f"{wall / tokens * 1e3:,.2f} ms", f"{tokens:,} tokens"))
        open_attr = " open" if g["name"] == "generations" else ""
        parts.append(
            f"<details class='card' style='margin:10px 0'{open_attr}><summary><strong>{esc(g['name'])}</strong> "
            f"<span class='muted'>{g['seconds']:,.2f} s sampled</span></summary>"
            f"<div class='tiles'>{''.join(tiles)}</div>"
            "<h3>By area of code (innermost frame)</h3>"
            + time_bars(g["by_area"], g["seconds"])
            + "<h3>Self: the lines the samples were in</h3>"
            + functions_table(g["self"], g["seconds"], tokens, "Line")
            + "<h3>Inclusive: functions on the stack, and everything they called</h3>"
            + functions_table(g["inclusive"], g["seconds"], tokens, "Function")
            + "</details>"
        )
    if links.get("python_stacks"):
        parts.append(f"<p>All stacks: <a href='{esc(links['python_stacks'])}'>{esc(links['python_stacks'])}</a> "
                     "(folded stacks: open in <a href='https://www.speedscope.app'>speedscope.app</a>).</p>")  # fmt: skip
    return "".join(parts)


def title_of(profile: dict) -> str:
    return f"Memory profile: {profile.get('model_id', '?')}"


def render_body(profile: dict, gpu: dict | None, host: dict | None, links: dict, python: dict | None = None) -> str:
    """The report's content, inside <div class="mem">: every value from the data is escaped, and there is no script.

    links: "flamegraph" and "memory_viz" (URLs, or absent), "files" ([(name, what)] listed at the end)."""
    cfg = profile.get("config") or {}
    marks = profile.get("marks") or []
    weights = profile.get("weights") or {}
    load = profile.get("load_memory") or {}
    names = [m["name"] for m in marks]
    v = profile.get("versions") or {}
    title = title_of(profile)
    parts = [f"<div class='mem'><h1>{esc(title)}</h1>"]
    scenario = (f"batch {cfg.get('batch_size')} · prompt {cfg.get('sequence_length')} tokens · "
                f"{cfg.get('num_tokens_to_generate')} new tokens · {cfg.get('attn_implementation')}"
                + (f" · compile {cfg['compile_kwargs'].get('mode')}" if cfg.get("compile_kwargs") else "")
                + (" · continuous batching" if cfg.get("continuous_batching") else ""))  # fmt: skip
    parts.append(
        f"<p class='meta'>{esc(scenario)}<br>transformers <code>{esc(str(profile.get('commit_id', '?'))[:12])}</code>"
        f" ({esc(v.get('transformers'))}) · torch {esc(v.get('torch'))} · CUDA {esc(v.get('cuda'))} · "
        f"{esc(profile.get('gpu_name') or 'no GPU')}</p>"
    )
    if profile.get("error"):
        parts.append(f"<div class='error'>The scenario failed: <code>{esc(profile['error'])}</code>. "
                     "Everything below was recorded up to the failure.</div>")  # fmt: skip

    last = marks[-1] if marks else {}
    tiles = [
        tile("Model tensors", fmt_mb(weights.get("weights_mb")), load.get("weights_dtypes", "")),
        tile("GPU peak allocated", fmt_bytes(gpu["peak_bytes"]) if gpu else "–",
             f"{fmt_mb((gpu['peak_bytes'] / MB) - weights.get('weights_mb', 0))} above the model" if gpu else ""),
        tile("GPU peak reserved", fmt_mb((last.get("device") or {}).get("peak_reserved_mb")),
             "the caching allocator's high-water mark"),
        tile("GPU, whole process", fmt_mb(last.get("process_gpu_mb")), "NVML, at the end (cache emptied)"),
        tile("Host peak RSS", fmt_mb((last.get("host") or {}).get("rss_peak_mb")), "whole run (VmHWM)"),
        tile("Read during load", fmt_mb(load.get("load_file_read_mb")),
             f"{fmt_mb(load.get('load_disk_read_mb'))} from disk"),
    ]  # fmt: skip
    parts.append(findings_box(profile, gpu, host, python))
    parts.append(f"<div class='tiles'>{''.join(tiles)}</div>")

    explore = []
    if links.get("memory_viz"):
        explore.append(
            f"<a href='{esc(links['memory_viz'])}' target='_blank' rel='noopener'>Every GPU allocation ↗"
            "<small>PyTorch's memory viewer, this run's allocator trace loaded</small></a>"
        )
    if links.get("flamegraph"):
        explore.append(
            f"<a href='{esc(links['flamegraph'])}' target='_blank' rel='noopener'>Host memory flamegraph ↗"
            "<small>memray, at the host peak</small></a>"
        )
    if explore:
        parts.append(f"<div class='explore'>{''.join(explore)}</div>")

    parts.append("<h2>Phases</h2>" + phase_table(marks))

    # GPU
    parts.append("<h2>GPU memory</h2>")
    if gpu is None:
        parts.append("<p class='muted'>No GPU trace (no CUDA device, or gpu_snapshot.pickle missing).</p>")
    else:
        phase_marks = [(x, names[i] if 0 <= i < len(names) else f"#{i}") for i, x in gpu["phases"]]
        x_label = "seconds" if gpu["time_axis"] else "allocator events"
        layers = [(c, gpu["by_category"][c], COLORS[c]) for c in CATEGORIES if any(gpu["by_category"][c])]
        lines = [("reserved by the caching allocator", gpu["reserved"], "var(--muted)")]
        parts.append(
            "<div class='card'>"
            + legend([(n, c, "") for n, _, c in layers] + [("reserved (cached)", "var(--muted)", "dash")])
            + area_chart(gpu["x"], layers, lines, phase_marks, x_label, (gpu["peak_x"], gpu["peak_bytes"]))
            + (
                "<div class='error'>The trace's ring buffer wrapped: the earliest events are missing, so totals "
                "and categories undercount what was allocated before. Rerun with a larger <code>--max-entries</code>."
                "</div>"
                if gpu["truncated"]
                else ""
            )
            + f"<p class='note'>{gpu['events']:,} allocator events; each point is the highest total of its interval. "
            "Category comes from the allocating stack and phase: allocated while loading → weights &amp; buffers, "
            "<code>cache_utils.py</code> → KV cache, inductor / dynamo / CUDA graph code → compile, everything else → "
            "activations &amp; temporaries. Memory a CUDA graph replay uses lives in its private pool, allocated once "
            "at capture.</p></div>"
        )
        above = [(c, v, col) for c, v, col in layers if c not in ("weights & buffers", "before recording")]
        if above:
            parts.append(
                "<h3>Above the model: what generation adds</h3><div class='card'>"
                + legend([(n, c, "") for n, _, c in above])
                + area_chart(gpu["x"], above, [], phase_marks, x_label)
                + "<p class='note'>The same timeline without the model's weights and buffers, so the KV cache and "
                "activations are visible at their own scale.</p></div>"
            )
        parts.append("<h3>At the peak, by category</h3><div class='card'>"
                     + bars(gpu["peak_by_category"], gpu["peak_bytes"], COLORS) + "</div>")  # fmt: skip
        parts.append("<h3>At the peak, by the code that allocated it</h3>"
                     + sites_table(gpu["peak_sites"], gpu["peak_bytes"]))  # fmt: skip

    # Model tensors
    parts.append("<h2>Model tensors</h2>")
    if weights:
        total = weights.get("weights_mb") or 0
        parts.append(
            "<div class='cols'><div class='card'><h3 style='margin-top:0'>By dtype</h3>"
            + bars({k: v for k, v in (weights.get("by_dtype_mb") or {}).items()}, total, unit_bytes=False)
            + "<h3>By device</h3>" + bars(weights.get("by_device_mb") or {}, total, unit_bytes=False)
            + f"<p class='note'>Parameters {esc(fmt_mb(weights.get('parameters_mb')))}, buffers "
            f"{esc(fmt_mb(weights.get('buffers_mb')))}; tied tensors counted once.</p></div>"
            "<div class='card'><h3 style='margin-top:0'>Largest modules (layers merged)</h3>"
            + bars(weights.get("top_modules_mb") or {}, total, unit_bytes=False) + "</div></div>"
        )  # fmt: skip
    else:
        parts.append("<p class='muted'>The model was not loaded.</p>")

    # Host
    parts.append("<h2>Host memory</h2>")
    if host is None:
        parts.append("<p class='muted'>No memray trace (run without <code>--no-memray</code>, with memray installed)."
                     "</p>")  # fmt: skip
    else:
        host_marks = [((m["time_ms"] - host["start_ms"]) / 1e3, m["name"]) for m in marks]
        parts.append(
            "<div class='card'>"
            + legend([("resident (RSS)", "var(--c1)", ""), ("heap tracked by memray", "var(--c2)", "dash")])
            + area_chart(host["x"], [("RSS", host["rss"], "var(--c1)")], [("heap", host["heap"], "var(--c2)")],
                         host_marks, "seconds")
            + "<p class='note'>RSS includes file pages mapped from safetensors files; memray's heap counts what was "
            "allocated, including whole mmap'd files, whether their pages were read or not.</p></div>"
        )  # fmt: skip
        flame = links.get("flamegraph")
        parts.append(
            "<h3>At the peak</h3><div class='cols'><div class='card'>"
            + tile("memray peak", fmt_bytes(host["peak_bytes"]), f"{host['total_allocations']:,} allocations")
            + "<h3>By phase</h3>" + bars(host["by_phase"], host["peak_bytes"])
            + "</div><div class='card'><h3 style='margin-top:0'>By allocator</h3>"
            + bars(host["by_allocator"], host["peak_bytes"])
            + "<p class='note'><code>mmap</code> is mostly mapped files (safetensors): address space, resident only "
            "where read.</p></div></div>"
            + (f"<p>Full flamegraph: <a href='{esc(flame)}' target='_blank' rel='noopener'>memray flamegraph</a>"
               f"{' (native frames included)' if host['native'] else ''}.</p>" if flame else "")
            + "<h3>Heap at the peak (malloc and friends), by the code that allocated it</h3>"
            + sites_table(host["heap_sites"], host["peak_bytes"], show_allocators=True)
            + "<h3>Mapped at the peak (mmap: files, CUDA driver mappings)</h3>"
            + sites_table(host["mmap_sites"], host["peak_bytes"], show_allocators=True)
        )  # fmt: skip

    parts.append(python_section(python, links))

    if links.get("files"):
        parts.append(
            "<h2>Files</h2><ul>"
            + "".join(f"<li><code>{esc(name)}</code>: {esc(what)}</li>" for name, what in links["files"])
            + "</ul><p class='note'>To browse every GPU allocation, gunzip <code>gpu_snapshot.pickle.gz</code> and "
            "open it in <a href='https://pytorch.org/memory_viz'>pytorch.org/memory_viz</a> (it stays in the "
            "browser).</p>"
        )
    parts.append("</div>")
    return "".join(parts)


def render(profile: dict, gpu: dict | None, host: dict | None, links: dict, python: dict | None = None) -> str:
    """A standalone page."""
    return (
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
        f"content='width=device-width,initial-scale=1'><title>{esc(title_of(profile))}</title>"
        f"<style>{TOKENS}{RULES}</style></head><body>{render_body(profile, gpu, host, links, python)}</body></html>"
    )


def render_fragment(
    profile: dict, gpu: dict | None, host: dict | None, links: dict, python: dict | None = None
) -> dict:
    """For a host page with its own header and styles (the Transformers CI dashboard): {title, css, body}. The body
    has no script and escapes every value, so the host page can inline it."""
    return {
        "title": title_of(profile),
        "css": SITE_TOKENS + RULES,
        "body": render_body(profile, gpu, host, links, python),
    }


def load_summaries(profile_dir: Path) -> tuple[dict, dict | None, dict | None]:
    """profile.json, gpu_summary.json and host_summary.json of a profile directory (the last two may be absent)."""

    def read(name: str):
        path = Path(profile_dir) / name
        return json.loads(path.read_text()) if path.exists() else None

    return read("profile.json"), read("gpu_summary.json"), read("host_summary.json")


def load_python_summary(profile_dir: Path) -> dict | None:
    """python_summary.json of a profile directory (absent before the python pass existed, or without py-spy)."""
    path = Path(profile_dir) / "python_summary.json"
    return json.loads(path.read_text()) if path.exists() else None


def build(profile_dir: Path) -> Path:
    """report.html from a profile directory; also writes host_summary.json (from host.memray.bin), so the report can
    be rendered again without memray (tperf.memory.report.render_fragment, from the published summaries)."""
    profile_dir = Path(profile_dir)
    profile = json.loads((profile_dir / "profile.json").read_text())
    files = [("profile.json", "phase counters and the model's tensors")]
    links: dict = {"files": files}
    gpu = None
    snap_path = profile_dir / "gpu_snapshot.pickle"
    summary_path = profile_dir / "gpu_summary.json"
    if summary_path.exists():
        gpu = json.loads(summary_path.read_text())
        files.append(("gpu_summary.json", "the GPU trace, replayed: timeline and live set at the peak"))
    elif snap_path.exists():
        with open(snap_path, "rb") as f:
            snapshot = pickle.load(f)
        gpu = analyze_gpu(snapshot, profile.get("marker_function", "_tperf_phase_marker"),
                          profile.get("marker_unit", 1 << 20))  # fmt: skip
    if (profile_dir / "gpu_snapshot.pickle.gz").exists():
        files.append(("gpu_snapshot.pickle.gz", "the CUDA allocator's history, for pytorch.org/memory_viz"))
    host = None
    host_bin = profile_dir / "host.memray.bin"
    host_summary = profile_dir / "host_summary.json"
    if host_bin.exists():
        try:
            host = analyze_host(host_bin)
            host_summary.write_text(json.dumps(host, separators=(",", ":")))
            files.append(("host.memray.bin", "memray capture (memray flamegraph / summary / tree)"))
            flame = profile_dir / "host-flamegraph.html"
            if memray_flamegraph(host_bin, flame):
                links["flamegraph"] = flame.name
                files.append((flame.name, "memray flamegraph of the host memory at its peak"))
        except ImportError:
            print("memray is not installed: host section from host_summary.json, if any", file=sys.stderr)
    if host is None and host_summary.exists():
        host = json.loads(host_summary.read_text())
    python = load_python_summary(profile_dir)
    if python is not None:
        files.append(("python_summary.json", "the python pass: time by phase, area of code and function"))
    if (profile_dir / "python.folded").exists():
        links["python_stacks"] = "python.folded"
        files.append(("python.folded", "py-spy's stacks, folded (speedscope.app, flamegraph.pl)"))
    out = profile_dir / "report.html"
    out.write_text(render(profile, gpu, host, links, python))
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("profile_dir", type=Path)
    a = p.parse_args(argv)
    print(build(a.profile_dir))


if __name__ == "__main__":
    main()
