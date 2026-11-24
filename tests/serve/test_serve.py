import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tperf.serve.bench import (
    by_turn,
    chat_turn,
    parse_args,
    system_prompt,
    ttft_slope,
    user_message,
)
from tperf.serve.summarize import compare, format_results


class FakeChat(BaseHTTPRequestHandler):
    """Streams `max_tokens` one-token chunks the way transformers serve does, then a usage chunk."""

    requests: list = []
    reasoning = 0  # how many leading tokens stream as `reasoning_content` (a thinking model, e.g. Qwen3)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeChat.requests.append(body)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def send(event):
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.flush()

        send({"choices": [{"delta": {"role": "assistant"}, "index": 0}]})
        for i in range(body["max_tokens"]):
            key = "reasoning_content" if i < FakeChat.reasoning else "content"
            send({"choices": [{"delta": {key: f"w{i} "}, "index": 0}]})
        prompt = sum(len(m["content"].split()) for m in body["messages"])
        send({"choices": [{"delta": {}, "finish_reason": "length", "index": 0}],
              "usage": {"prompt_tokens": prompt, "completion_tokens": body["max_tokens"]}})  # fmt: skip

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeChat)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    FakeChat.requests = []
    FakeChat.reasoning = 0
    yield server.server_address[1]
    server.shutdown()


def test_chat_turn_streams_and_times(fake_server):
    messages = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello there"}]
    r = chat_turn(fake_server, "m", messages, max_tokens=5, fixed_length=True, timeout=10)
    assert r["reply"] == "w0 w1 w2 w3 w4 "
    assert r["completion_tokens"] == 5 and r["chunks"] == 5 and r["finish_reason"] == "length"
    assert r["prompt_tokens"] == 4
    assert 0 <= r["ttft_s"] <= r["e2e_s"] and r["itl_ms"] is not None
    sent = FakeChat.requests[0]
    assert sent["stream"] is True and sent["temperature"] == 0
    assert json.loads(sent["generation_config"]) == {"max_new_tokens": 5, "min_new_tokens": 5, "do_sample": False}
    chat_turn(fake_server, "m", messages, max_tokens=5, fixed_length=False, timeout=10)
    assert "generation_config" not in FakeChat.requests[1]


def test_reasoning_deltas_count_as_streamed_tokens(fake_server):
    messages = [{"role": "user", "content": "hello"}]
    FakeChat.reasoning = 5  # all thinking: before the fix, "the server streamed no content"
    r = chat_turn(fake_server, "m", messages, max_tokens=5, fixed_length=True, timeout=10)
    assert r["chunks"] == 5 and r["reply"] == "w0 w1 w2 w3 w4 "
    FakeChat.reasoning = 2  # thinking then an answer: timings span both, the history gets the answer
    r = chat_turn(fake_server, "m", messages, max_tokens=5, fixed_length=True, timeout=10)
    assert r["chunks"] == 5 and r["reply"] == "w2 w3 w4 "


def test_conversation_resends_the_history(fake_server, tmp_path):
    from tperf.serve.bench import conversation

    a = parse_args(["--model-id", "m", "--out", str(tmp_path), "--turns", "3", "--max-tokens", "4"])
    rows = conversation(a, fake_server, 0, None)
    assert [r["turn"] for r in rows] == [0, 1, 2]
    sizes = [len(req["messages"]) for req in FakeChat.requests]
    assert sizes == [2, 4, 6]  # system + (user, assistant) * earlier turns + user
    assert FakeChat.requests[2]["messages"][2] == {"role": "assistant", "content": "w0 w1 w2 w3 "}
    assert rows[2]["prompt_tokens"] > rows[0]["prompt_tokens"]


def test_messages_are_deterministic_and_sized():
    assert user_message(0, 1, 64) == user_message(0, 1, 64) != user_message(1, 1, 64)
    assert system_prompt(256) == system_prompt(256)
    assert 40 < len(user_message(0, 0, 64).split()) < 60


def test_scenario_fields_are_defaults(tmp_path):
    scenarios = tmp_path / "s.jsonl"
    scenarios.write_text('# comment\n{"name": "a", "turns": 2}\n{"name": "b", "turns": 5, "max_tokens": 9}\n')
    a = parse_args(["--model-id", "m", "--out", "o", "--config-file", str(scenarios), "--scenario-index", "1",
                    "--turns", "3"])  # fmt: skip
    assert (a.turns, a.max_tokens, a.scenario_name) == (3, 9, "b")  # the command line wins
    scenarios.write_text('{"bogus": 1}\n')
    with pytest.raises(SystemExit):
        parse_args(["--model-id", "m", "--out", "o", "--config-file", str(scenarios)])


def rows_for(ttfts: list[float], prompts: list[int]) -> list[dict]:
    return [{"turn": t, "conversation": c, "prompt_tokens": p, "completion_tokens": 64, "ttft_s": s, "e2e_s": s + 1,
             "decode_tok_s": 60.0, "itl_ms": 16.0, "chunk_gap_p99_ms": 20.0, "chunk_gap_max_ms": 30.0,
             "server_rss_mb": None}
            for c in range(2) for t, (s, p) in enumerate(zip(ttfts, prompts))]  # fmt: skip


def test_by_turn_and_slope():
    rows = rows_for([0.10, 0.12, 0.14], [1000, 2000, 3000])
    turns = by_turn(rows)
    assert [t["n"] for t in turns] == [2, 2, 2] and turns[1]["ttft_s"] == 0.12
    assert turns[0]["server_rss_mb"] is None
    slope = ttft_slope(rows)
    assert slope["ms_per_1k_prompt_tokens"] == pytest.approx(20.0)
    assert slope["intercept_ms"] == pytest.approx(80.0)
    assert ttft_slope(rows_for([0.1], [1000])) is None


def test_summarize_and_compare():
    from tperf.serve.bench import overall

    def results(ttfts, args=None):
        rows = rows_for(ttfts, [1000, 2000])
        return {"model_id": "m", "commit": "abc", "args": args or {"turns": 2, "max_tokens": 64},
                "by_turn": by_turn(rows), "overall": overall(rows, 10.0), "turns": rows}  # fmt: skip

    base, slower = results([0.10, 0.20]), results([0.10, 0.30])
    text = format_results(base)
    assert "TTFT ms" in text and "per 1k prompt tokens" in text
    diff = compare(base, slower)
    assert "+50.0% !" in diff and "not the same" not in diff
    assert "not the same conversation" in compare(base, results([0.1, 0.2], {"turns": 3, "max_tokens": 64}))


def serve_run(tmp_path, side, rnd, gpu, ttft_s, prefix_ms=0.0):
    """A `tperf ab` side directory with SERVE: manifest.json and serve/results.json (8 requests)."""
    d = tmp_path / f"round{rnd}-{side}"
    (d / "serve").mkdir(parents=True)
    manifest = {"run_id": f"ab-r{rnd}-{side}-run1", "group_id": f"ab-r{rnd}-{side}", "started_at": f"2026-11-24T1{rnd}",
                "label": f"ab-pr1-{side}", "source": "pr", "pr_number": 1, "commit_id": side * 4, "ab_id": "ab",
                "ab_side": side, "ab_round": rnd, "model_id": "m", "bench_args": "serve chat", "gpu_index": gpu,
                "instance_type": "g5", "gpu_name": "A10G", "sm_clock_mhz": 1710}  # fmt: skip
    (d / "manifest.json").write_text(json.dumps(manifest))
    turns = [{"turn": t, "ttft_s": ttft_s + t * prefix_ms / 1e3, "e2e_s": 3.0 + ttft_s, "itl_ms": 47.0,
              "completion_tokens": 64} for t in range(8)]  # fmt: skip
    results = {"scenario": "chat", "error": None, "startup_s": 7.0, "turns": turns,
               "args": {"concurrency": 1, "system_tokens": 256, "max_tokens": 64, "warmup": 1,
                        "server_args": "--attn-implementation sdpa"}}  # fmt: skip
    (d / "serve" / "results.json").write_text(json.dumps(results))
    return d


def test_serve_run_is_one_row(tmp_path):
    from tperf.common.record import COLUMNS, scenario_rows

    (row,) = scenario_rows(serve_run(tmp_path, "base", 1, 0, 0.2, prefix_ms=40))
    assert row["scenario"] == "serve:chat" and row["success"] and row["n_samples"] == 8
    assert row["attn_implementation"] == "sdpa" and row["batch_size"] == 1 and row["num_tokens_to_generate"] == 64
    assert row["ttft_ms_min"] == 200 and abs(row["ttft_ms_max"] - (200 + 7 * 40)) < 1e-6
    assert row["itl_ms_median"] == 47.0 and row["startup_load_time_s"] == 7.0
    assert set(row) <= set(COLUMNS)  # publishable as is


def test_serve_ab_is_analyzed_and_labelled(tmp_path):
    from tperf.ab.compare import analyze
    from tperf.common.record import scenario_rows
    from tperf.dashboard.build import ab_results

    rows = []
    for rnd in range(1, 5):
        gb, gh = (0, 4) if rnd % 2 else (4, 0)
        rows += scenario_rows(serve_run(tmp_path, "base", rnd, gb, 0.30 + rnd * 1e-3))
        rows += scenario_rows(serve_run(tmp_path, "head", rnd, gh, 0.14))
    (scenario,) = analyze(rows)["scenarios"]
    ttft = next(m for m in scenario["metrics"] if m["name"] == "ttft")
    assert ttft["verdict"] == "improvement" and ttft["change_pct"] < -50
    (run,) = ab_results(rows)
    assert run["pr"] == 1 and run["scenarios"][0]["scenario"] == "transformers serve, chat · 8 requests of 64 tokens"
