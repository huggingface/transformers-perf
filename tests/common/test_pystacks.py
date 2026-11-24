from tperf.common.pystacks import area_of, parse_frame, parse_raw, record_command, summarize, thread_label


SP = "/opt/bench/tbv2-venv/lib/python3.12/site-packages"
TF = "/opt/bench/memory-work/x/src/transformers"
STDLIB = "/usr/lib/python3.12"

RAW = "\n".join([
    # load phase: from_pretrained under the phase function (tperf frames hidden from the tables)
    f"<module> (/r/src/tperf/memory/profile.py:300);_tperf_phase_load (/r/src/tperf/memory/profile.py:40);"
    f"from_pretrained ({TF}/modeling_utils.py:4000);load_state_dict ({SP}/safetensors/torch.py:10) 30",
    # generations: generate -> forward, and a sync charged to the calling line
    f"<module> (/r/src/tperf/memory/profile.py:300);_tperf_phase_generations (/r/src/tperf/memory/profile.py:50);"
    f"time_generate (/r/src/tperf/harness/framework/benchmark_runner.py:10);generate ({TF}/generation/utils.py:2000);"
    f"_sample ({TF}/generation/utils.py:2800);_call_impl ({SP}/torch/nn/modules/module.py:1700);"
    f"forward ({TF}/models/llama/modeling_llama.py:300) 60",
    f"<module> (/r/src/tperf/memory/profile.py:300);_tperf_phase_generations (/r/src/tperf/memory/profile.py:50);"
    f"time_generate (/r/src/tperf/harness/framework/benchmark_runner.py:10);generate ({TF}/generation/utils.py:2000);"
    f"_sample ({TF}/generation/utils.py:2850) 40",
    # outside any phase: imports
    "<module> (/r/src/tperf/memory/profile.py:1);_find_and_load (<frozen importlib._bootstrap>:1360) 10",
    "garbage line without a count",
])  # fmt: skip


def test_parse_raw_and_frame():
    samples = parse_raw(RAW)
    assert len(samples) == 4
    assert samples[0][1] == 30
    assert parse_frame(f"forward ({TF}/models/llama/modeling_llama.py:300)") == (
        "forward",
        f"{TF}/models/llama/modeling_llama.py",
        300,
    )
    assert parse_frame("cudaStreamSynchronize (libcudart.so.12)") == ("cudaStreamSynchronize", "libcudart.so.12", None)
    assert parse_frame("weird") == ("weird", "", None)


def test_areas():
    assert area_of(f"{TF}/generation/utils.py") == "transformers generation"
    assert area_of(f"{TF}/models/llama/modeling_llama.py") == "transformers models"
    assert area_of(f"{TF}/cli/serving/chat_completion.py") == "transformers serve"
    assert area_of(f"{SP}/torch/_dynamo/eval_frame.py") == "torch.compile"
    assert area_of(f"{SP}/torch/nn/modules/module.py") == "torch"
    assert area_of(f"{SP}/uvicorn/server.py") == "server framework"
    assert area_of(f"{STDLIB}/asyncio/base_events.py") == "server framework"
    assert area_of("libcudart.so.12") == "CUDA runtime & driver"
    assert area_of(f"{SP}/torch/lib/libtorch_cuda.so") == "torch native"
    assert area_of(f"{STDLIB}/json/decoder.py") == "Python, other"


def test_summarize_by_phase():
    phases = {"_tperf_phase_load": "load", "_tperf_phase_generations": "generations"}
    s = summarize(parse_raw(RAW), rate=100, phases=phases)
    assert s["samples"] == 140 and s["seconds"] == 1.4
    groups = {g["name"]: g for g in s["groups"]}
    assert set(groups) == {"load", "generations", "other"}
    gen = groups["generations"]
    assert gen["seconds"] == 1.0
    assert gen["by_area"] == {"transformers models": 0.6, "transformers generation": 0.4}
    # self: innermost line; inclusive: every function once per sample, tperf frames hidden
    assert gen["self"][0]["function"] == "transformers/models/llama/modeling_llama.py:300 forward"
    inclusive = {r["function"]: r["seconds"] for r in gen["inclusive"]}
    assert inclusive["generate (transformers/generation/utils.py)"] == 1.0
    assert inclusive["_sample (transformers/generation/utils.py)"] == 1.0  # two lines, one function
    assert not any("tperf" in f for f in inclusive)
    assert groups["load"]["by_area"] == {"Python, other": 0.3}  # safetensors/torch.py is not torch's own


def test_tool_frames_and_entry_frames():
    raw = (
        "_run_module_as_main (<frozen runpy>:198);_run_code (<frozen runpy>:88);"
        "<module> (/r/src/tperf/memory/profile.py:1);mark (/r/src/tperf/memory/profile.py:9) 5\n"
        "_run_module_as_main (<frozen runpy>:198);_run_code (<frozen runpy>:88);"
        f"<module> (/r/src/tperf/memory/profile.py:1);synchronize ({SP}/torch/cuda/__init__.py:1) 3\n"
    )
    (g,) = summarize(parse_raw(raw), rate=1)["groups"]
    assert g["by_area"] == {"(benchmark tool)": 5, "torch": 3}
    assert [r["function"] for r in g["self"]] == ["torch/cuda/__init__.py:1 synchronize"]
    assert not any("runpy" in r["function"] for r in g["inclusive"])


def test_summarize_by_thread():
    gen = f"_run ({TF}/cli/serving/utils.py:500)"
    raw = "\n".join([
        f"thread (0x1): Thread-3 (_run);_bootstrap ({STDLIB}/threading.py:1);run ({STDLIB}/threading.py:3);{gen};"
        f"generate ({TF}/generation/utils.py:2000) 70",
        # truncated (py-spy --nonblocking): still the same thread
        f"thread (0x1): Thread-3 (_run);forward ({TF}/models/llama/modeling_llama.py:116) 5",
        f"thread (0x2): MainThread;_run_module_as_main (<frozen runpy>:198);<module> (/r/src/tperf/serve/entry.py:1);"
        f"main ({TF}/cli/transformers.py:40);run_until_complete ({STDLIB}/asyncio/base_events.py:700) 20",
    ])  # fmt: skip
    s = summarize(parse_raw(raw), rate=10, by_thread=True)
    names = [(g["name"], g["seconds"]) for g in s["groups"]]
    assert names == [("Thread-3 (_run): _run (transformers/cli/serving/utils.py)", 7.5),
                     ("MainThread: main (transformers/cli/transformers.py)", 2.0)]  # fmt: skip


def test_thread_label_skips_thread_start_frames():
    frames = [("_bootstrap", f"{STDLIB}/threading.py", 1), ("_worker", f"{STDLIB}/concurrent/futures/thread.py", 9),
              ("work", "/x/app.py", 3)]  # fmt: skip
    assert thread_label(frames) == "work (app.py)"


def test_record_command():
    launch = record_command("out.folded", rate=250, pyspy="/v/bin/py-spy")
    assert launch[:2] == ["/v/bin/py-spy", "record"] and launch[-1] == "--"
    assert "--full-filenames" in launch and launch[launch.index("--rate") + 1] == "250"
    attach = record_command("out.folded", pid=42, threads=True, native=True)
    assert attach[-2:] == ["--pid", "42"] and "--threads" in attach and "--native" in attach
