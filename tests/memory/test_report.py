import json
import pickle

from tperf.memory.profile import MARKER_FUNCTION, MARKER_UNIT
from tperf.memory.report import analyze_gpu, build, gpu_category, gpu_site, render

MB = 1024**2
TF = "/opt/bench/memory-work/x/src/transformers"


def frame(filename, line, name):
    return {"filename": filename, "line": line, "name": name}


LOAD = [frame("/venv/torch/nn/modules/module.py", 1, "to"), frame(f"{TF}/modeling_utils.py", 5000, "from_pretrained")]
CACHE = [frame("/venv/torch/empty.py", 1, "empty"), frame(f"{TF}/cache_utils.py", 321, "lazy_initialization")]
ACT = [frame("/venv/torch/functional.py", 9, "softmax"), frame(f"{TF}/models/llama/modeling_llama.py", 210, "forward")]
MARK = [frame("/src/tperf/memory/profile.py", 33, MARKER_FUNCTION)]


def ev(action, addr, size, frames=(), t=None):
    e = {"action": action, "addr": addr, "size": size, "frames": list(frames)}
    if t is not None:
        e["time_us"] = t
    return e


def snapshot(events, live_blocks):
    """live_blocks: (address, size, frames) still allocated at the end, in one segment at 0x1000."""
    blocks = [{"size": size, "state": "active_allocated", "frames": frames} for _, size, frames in live_blocks]
    seg = {"device": 0, "address": 0x1000, "total_size": 512 * MB, "blocks": blocks}
    return {"segments": [seg], "device_traces": [events]}


def test_categories_and_sites():
    assert gpu_category(LOAD) == "weights & buffers"
    assert gpu_category(CACHE) == "KV cache"
    assert gpu_category(ACT) == "activations & temporaries"
    assert gpu_category([frame("/venv/torch/_inductor/cudagraph_trees.py", 1, "x")]) == "compile & CUDA graphs"
    assert gpu_category([]) == "before recording"
    assert gpu_site(ACT) == "transformers/models/llama/modeling_llama.py:210 forward"
    assert gpu_site([frame("/venv/site-packages/foo/bar.py", 3, "f")]) == "foo/bar.py:3 f"


def test_replay_peak_categories_and_phases():
    weights, cache, act = 0x1000, 0x1000 + 100 * MB, 0x1000 + 150 * MB
    events = [
        ev("alloc", 0x9000_0000, 1 * MARKER_UNIT, MARK, t=0),  # phase 0
        ev("free_completed", 0x9000_0000, 1 * MARKER_UNIT, MARK, t=1),
        ev("segment_alloc", 0x1000, 512 * MB, t=2),
        ev("alloc", weights, 100 * MB, LOAD, t=1_000_000),
        ev("alloc", 0x9000_0000, 2 * MARKER_UNIT, MARK, t=1_500_000),  # phase 1
        ev("free_completed", 0x9000_0000, 2 * MARKER_UNIT, MARK, t=1_500_001),
        ev("alloc", cache, 40 * MB, CACHE, t=2_000_000),
        ev("alloc", act, 30 * MB, ACT, t=2_500_000),  # peak: 170 MB
        ev("free_completed", act, 30 * MB, ACT, t=3_000_000),
    ]
    gpu = analyze_gpu(
        snapshot(events, [(weights, 100 * MB, LOAD), (cache, 40 * MB, CACHE)]), MARKER_FUNCTION, MARKER_UNIT
    )
    assert gpu["peak_bytes"] == 170 * MB
    assert gpu["peak_x"] == 2.5
    assert gpu["peak_by_category"] == {
        "weights & buffers": 100 * MB,
        "KV cache": 40 * MB,
        "activations & temporaries": 30 * MB,
    }
    assert [s["site"] for s in gpu["peak_sites"]][0] == "transformers/modeling_utils.py:5000 from_pretrained"
    assert [i for i, _ in gpu["phases"]] == [0, 1]
    assert gpu["phases"][1][1] == 1.5
    assert gpu["time_axis"] and not gpu["truncated"]
    assert max(gpu["reserved"]) == 512 * MB


def test_allocations_before_recording_and_truncation():
    old = 0x1000
    events = [
        ev("free_completed", 0x7777, 5 * MB, ACT),  # its alloc fell out of the ring buffer
        ev("alloc", old + 64 * MB, 8 * MB, ACT),
    ]
    gpu = analyze_gpu(snapshot(events, [(old, 64 * MB, []), (old + 64 * MB, 8 * MB, ACT)]), MARKER_FUNCTION, 1)
    assert gpu["baseline_bytes"] == 64 * MB
    assert gpu["peak_by_category"]["before recording"] == 64 * MB
    assert gpu["peak_bytes"] == 72 * MB
    assert gpu["truncated"]
    assert not gpu["time_axis"]


def test_build_writes_a_self_contained_report(tmp_path):
    marks = [
        {"name": n, "index": i, "time_ms": 1000.0 * i, "host": {"rss_anon_mb": 10.0 * i, "read_bytes": i * MB},
         "device": {"allocated_mb": 100.0, "peak_reserved_mb": 512.0}, "process_gpu_mb": 900.0}
        for i, n in enumerate(["start", "loaded", "first generation", "generations"])
    ]  # fmt: skip
    profile = {
        "model_id": "org/<model>",
        "config": {"batch_size": 1, "sequence_length": 128, "num_tokens_to_generate": 64, "attn_implementation": "sdpa"},
        "marks": marks,
        "marker_function": MARKER_FUNCTION,
        "marker_unit": MARKER_UNIT,
        "load_memory": {"weights_dtypes": "bfloat16:100.0", "load_file_read_mb": 100.0},
        "weights": {"weights_mb": 100.0, "parameters_mb": 100.0, "buffers_mb": 0.0, "by_dtype_mb": {"bfloat16": 100.0},
                    "by_device_mb": {"cuda": 100.0}, "top_modules_mb": {"model.layers.*.mlp.up_proj": 40.0}},
        "versions": {"torch": "2.x", "transformers": "5.x"},
    }  # fmt: skip
    (tmp_path / "profile.json").write_text(json.dumps(profile))
    events = [ev("alloc", 0x1000, 100 * MB, LOAD), ev("alloc", 0x1000 + 100 * MB, 30 * MB, ACT)]
    with open(tmp_path / "gpu_snapshot.pickle", "wb") as f:
        pickle.dump(snapshot(events, [(0x1000, 100 * MB, LOAD), (0x1000 + 100 * MB, 30 * MB, ACT)]), f)
    page = build(tmp_path).read_text()
    assert page.startswith("<!doctype html>")
    assert "org/&lt;model&gt;" in page  # escaped
    assert "modeling_llama.py:210 forward" in page
    assert "<svg" in page and "http" not in page.split("pytorch.org/memory_viz")[0].split("<style>")[0]
    assert "No memray trace" in page


def test_render_without_traces():
    page = render({"model_id": "m", "marks": [], "error": "RuntimeError('boom')"}, None, None, {})
    assert "The scenario failed" in page and "No GPU trace" in page


def test_load_phase_allocations_are_weights_without_from_pretrained_on_the_stack():
    worker = [frame(f"{TF}/core_model_loading.py", 1465, "_materialize_copy")]
    other = [frame("/venv/threading.py", 1, "run")]  # a loader thread: no transformers frame at all
    events = [
        ev("alloc", 0x9000_0000, 1 * MARKER_UNIT, MARK),  # phase 0: load starts
        ev("free_completed", 0x9000_0000, 1 * MARKER_UNIT, MARK),
        ev("alloc", 0x1000, 10 * MB, worker),
        ev("alloc", 0x2000_0000, 5 * MB, other),
        ev("alloc", 0x9000_0000, 2 * MARKER_UNIT, MARK),  # phase 1: generation
        ev("free_completed", 0x9000_0000, 2 * MARKER_UNIT, MARK),
        ev("alloc", 0x3000_0000, 1 * MB, other),
    ]
    gpu = analyze_gpu(snapshot(events, []), MARKER_FUNCTION, MARKER_UNIT)
    assert gpu["peak_by_category"] == {"weights & buffers": 15 * MB, "activations & temporaries": 1 * MB}
    assert not gpu["truncated"]


def test_timeline_spans_the_whole_trace_with_empty_bins_carried_forward():
    from tperf.memory.report import MAX_POINTS

    events = [ev("alloc", 0x1000, 10 * MB, LOAD, t=0), ev("alloc", 0x2000_0000, 5 * MB, ACT, t=10_000_000)]
    gpu = analyze_gpu(snapshot(events, []), MARKER_FUNCTION, MARKER_UNIT)
    assert len(gpu["x"]) == MAX_POINTS and gpu["x"][0] == 0 and gpu["x"][-1] > 9.9
    assert gpu["allocated"][MAX_POINTS // 2] == 10 * MB  # no event there: the state after the first one
    assert gpu["allocated"][-1] == 15 * MB


def test_benchmark_tool_frames_are_hidden():
    from tperf.memory.report import host_site, strip_hidden_frames

    runner = frame("/opt/bench/src/tperf/harness/framework/benchmark_runner.py", 252, "setup_benchmark")
    to = frame("/venv/torch/_tensor.py", 9, "to")
    assert gpu_site([to, runner]) == "torch/_tensor.py:9 to"
    assert gpu_site([runner]) == "(benchmark harness)"
    assert gpu_site([*ACT, runner]) == "transformers/models/llama/modeling_llama.py:210 forward"
    host = [
        ("synchronize", "/venv/torch/cuda/__init__.py", 1),
        ("mark", "/opt/bench/src/tperf/memory/profile.py", 151),
    ]
    assert host_site(host) == "torch/cuda/__init__.py:1 synchronize"
    snap = snapshot([ev("alloc", 1, 1, [to, runner])], [(0x1000, 8, [runner, to])])
    strip_hidden_frames(snap)
    assert snap["device_traces"][0][0]["frames"] == [to]
    assert snap["segments"][0]["blocks"][0]["frames"] == [to]


def test_fragment_has_no_script_and_escapes_data():
    from tperf.memory.report import render_fragment

    profile = {"model_id": "org/<script>alert(1)</script>", "marks": []}
    frag = render_fragment(profile, None, None, {"memory_viz": "memory-viz.html", "flamegraph": "f.html"})
    assert "<script" not in frag["body"] and "&lt;script&gt;" in frag["body"]
    assert frag["body"].startswith("<div class='mem'>") and "memory-viz.html" in frag["body"]
    assert ":root" not in frag["css"] and ".mem h1" in frag["css"]


def test_findings_flag_what_looks_wrong_and_list_what_passed():
    from tperf.memory.report import findings, findings_box

    healthy = {
        "weights": {"weights_mb": 1000.0, "by_dtype_mb": {"bfloat16": 1000.0}, "by_device_mb": {"cuda": 1000.0}},
        "load_memory": {"load_host_peak_rss_mb": 1500.0, "load_disk_read_mb": 1000.0},
        "marks": [
            {"name": "start", "host": {"rss_mb": 1000.0}},
            {"name": "first generation", "host": {"rss_anon_mb": 900.0}, "device": {"allocated_mb": 1000.0}},
            {"name": "generations", "host": {"rss_anon_mb": 910.0}, "process_gpu_mb": 1600.0,
             "device": {"allocated_mb": 1000.0, "reserved_mb": 1100.0, "peak_reserved_mb": 1200.0}},
        ],
    }  # fmt: skip
    gpu = {"peak_bytes": 1100 * MB, "peak_sites": [], "peak_by_category": {}, "truncated": False}
    found, passed = findings(healthy, gpu, None)
    assert found == [] and "allocator retries" in passed
    assert "Nothing unusual" in findings_box(healthy, gpu, None)

    bad = json.loads(json.dumps(healthy))
    bad["weights"]["by_dtype_mb"] = {"bfloat16": 900.0, "float32": 100.0}
    bad["weights"]["minor_dtype_modules_mb"] = {"float32": {"model.layers.*.mlp.router": 100.0}}
    bad["load_memory"]["load_host_peak_rss_mb"] = 3500.0  # 2.5x the model on top of the start
    bad["marks"][2]["device"].update(allocated_mb=1100.0, alloc_retries=3)
    bad["marks"][2]["host"]["rss_anon_mb"] = 1200.0
    big = {"category": "activations & temporaries", "largest": 600 * MB, "site": "x.py:1 f", "bytes": 600 * MB}
    titles = " / ".join(t for _, t, _ in findings(bad, {**gpu, "peak_sites": [big]}, None)[0])
    for expected in ("in float32, the rest in bfloat16", "2.5× the model", "more GPU memory held",
                     "Host memory grew", "retried 3", "One temporary of 600"):  # fmt: skip
        assert expected in titles, (expected, titles)
    assert "model.layers.*.mlp.router" in findings_box(bad, gpu, None)


def test_python_section_and_compile_finding(tmp_path):
    from tperf.common.pystacks import parse_raw, summarize
    from tperf.memory.profile import PHASE_FUNCTIONS
    from tperf.memory.report import findings, render_fragment

    raw = (
        "<module> (/r/src/tperf/memory/profile.py:1);_tperf_phase_generations (/r/src/tperf/memory/profile.py:2);"
        f"generate ({TF}/generation/utils.py:2000);_fn (/venv/torch/_dynamo/eval_frame.py:600) 30\n"
        "<module> (/r/src/tperf/memory/profile.py:1);_tperf_phase_generations (/r/src/tperf/memory/profile.py:2);"
        f"generate ({TF}/generation/utils.py:2000);forward ({TF}/models/llama/modeling_llama.py:9) 70\n"
    )
    python = summarize(parse_raw(raw), rate=100, phases=PHASE_FUNCTIONS)
    python["groups"][0].update(wall_seconds=1.2, tokens=640)
    profile = {"model_id": "m", "marks": []}
    found, passed = findings(profile, None, None, python)
    assert any("torch.compile" in title for _, title, _ in found)
    body = render_fragment(profile, None, None, {"python_stacks": "python.folded"}, python)["body"]
    assert "Python time" in body and "transformers models" in body and "python.folded" in body
    assert "1.875 ms" not in body and "1.88 ms" in body  # wall per token: 1.2 s / 640
    # without a python summary the section says why, and the other findings still run
    assert "No python pass" in render_fragment(profile, None, None, {})["body"]


def test_build_reads_python_summary(tmp_path):
    (tmp_path / "profile.json").write_text(json.dumps({"model_id": "m", "marks": []}))
    (tmp_path / "python_summary.json").write_text(json.dumps(
        {"rate": 100, "samples": 10, "seconds": 0.1, "groups": [
            {"name": "generations", "samples": 10, "seconds": 0.1, "by_area": {"torch": 0.1},
             "self": [{"function": "torch/x.py:1 f", "area": "torch", "samples": 10, "seconds": 0.1}],
             "inclusive": []}]}))  # fmt: skip
    (tmp_path / "python.folded").write_text("f (/x/torch/x.py:1) 10\n")
    html = build(tmp_path).read_text()
    assert "torch/x.py:1 f" in html and "speedscope" in html
