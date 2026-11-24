"""Multi-turn chat against a real `transformers serve`: per-turn latency as the conversation grows, and optionally
where the server's Python time goes.

    python -m tperf.serve.bench --model-id <model> --out <dir> [--turns 8] [--server-args "--attn-implementation sdpa"]
    python -m tperf.serve.bench --model-id <model> --out <dir> --config-file scenarios/serve.jsonl --scenario-index 1
    tperf serve bench ...                 (the same, here: works on a laptop too, the server picks its device)

Starts `transformers serve <model>` (the transformers on PYTHONPATH, preloading the model), waits for /health, then
plays conversations through /v1/chat/completions with streaming, the way a chat client does: each turn sends the whole
history (a fixed system prompt, every earlier user message and the assistant's replies) plus one new user message.
The server re-reads the history every turn, so prompt length, and with it time to first token, grows turn after turn:
that growth is what this scenario is about, next to decode speed and the serving layer's own overhead.

Replies have a fixed length (`min_new_tokens = max_new_tokens`, greedy) so every run generates the same tokens;
`--natural-length` lets the model stop at EOS instead. User messages are synthetic and seeded: the same in every run.
The first `--warmup` conversations are not measured. With `--concurrency N`, N conversations run at once (continuous
batching: `--server-args --continuous-batching`).

Per turn: prompt and completion tokens (the server's own usage), time to first token, end-to-end time, decode speed
((tokens - 1) / time from first to last token), the gaps between streamed chunks (p50, p99, max: stalls), and the
server's resident memory. Written to results.json; `tperf serve summarize <dir> [<dir>]` prints and compares them.

--profile attaches py-spy (tperf.common.pystacks) to the server for the measured conversations only, so loading and
warm-up stay out of it -> server.folded, and in results.json the time per server thread (the event loop serving HTTP
and streaming, the generation thread) by area of code and function. py-spy runs outside the server and takes no GIL,
so the latencies stay meaningful, but compare profiled runs with profiled runs. Attaching needs ptrace rights: the
server is started through tperf.serve.entry, which allows any process of the same user to trace it (Linux
PR_SET_PTRACER; root is needed only where kernel.yama.ptrace_scope is 2 or more).
"""

import argparse
import http.client
import json
import os
import random
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


# Words for the synthetic user messages: ordinary English, about 1.3 tokens each with common tokenizers
WORDS = (
    "the model reads a long document about rivers mountains cities trains weather music history science gardens "
    "language markets bridges harbors forests islands deserts libraries museums kitchens recipes engines bicycles "
    "planets oceans islands castles villages festivals schools hospitals factories farms orchards vineyards and "
    "then summarizes what changed over time why it matters which parts are uncertain how people describe it today"
).split()
TOKENS_PER_WORD = 1.3
SYSTEM_PROMPT_HEAD = "You are a helpful assistant. Answer every question in plain prose. Background notes follow. "


def words(n_tokens: int, rng: random.Random) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(max(1, round(n_tokens / TOKENS_PER_WORD))))


def system_prompt(n_tokens: int) -> str:
    return SYSTEM_PROMPT_HEAD + words(n_tokens, random.Random(0))


def user_message(conversation: int, turn: int, n_tokens: int) -> str:
    rng = random.Random(conversation * 1000 + turn + 1)
    return f"Question {turn + 1}: {words(n_tokens, rng)}. Please continue the discussion."


# ------------------------------------------------------------------------------------------------------------ server


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    """`transformers serve <model>` in its own process group, started through tperf.serve.entry."""

    def __init__(self, python: str, model_id: str, port: int, args: list[str], log_path: Path):
        self.port = port
        self.command = [python, "-m", "tperf.serve.entry", "serve", model_id, "--host", "127.0.0.1",
                        "--port", str(port), *args]  # fmt: skip
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None

    def start(self, timeout: float) -> float:
        """Start and wait for /health; returns the seconds it took (imports and loading the model)."""
        env = {**os.environ, "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1")}
        self.log = open(self.log_path, "w")  # noqa: SIM115 (closed in stop)
        t0 = time.perf_counter()
        self.proc = subprocess.Popen(self.command, stdout=self.log, stderr=subprocess.STDOUT, env=env,
                                     start_new_session=True)  # fmt: skip
        while time.perf_counter() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"the server exited with {self.proc.returncode}: see {self.log_path}")
            try:
                status, _ = request(self.port, "GET", "/health", timeout=2)
                if status == 200:
                    return time.perf_counter() - t0
            except OSError:
                pass
            time.sleep(0.25)
        raise TimeoutError(f"the server did not answer /health within {timeout:.0f} s: see {self.log_path}")

    def rss_mb(self) -> float | None:
        """Resident memory of the server process (Linux)."""
        try:
            for line in Path(f"/proc/{self.proc.pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
        except (OSError, AttributeError):
            pass
        return None

    def gpu_mb(self) -> float | None:
        """GPU memory the server process holds (nvidia-smi), or None without one."""
        try:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10).stdout  # fmt: skip
        except (OSError, subprocess.TimeoutExpired):
            return None
        for line in out.splitlines():
            pid, _, mb = line.partition(",")
            if pid.strip() == str(self.proc.pid):
                return float(mb)
        return None

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGINT)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        if getattr(self, "log", None):
            self.log.close()


def request(port: int, method: str, path: str, body: dict | None = None, timeout: float = 600):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        conn.request(method, path, body=json.dumps(body) if body is not None else None,
                     headers={"Content-Type": "application/json"})  # fmt: skip
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


# -------------------------------------------------------------------------------------------------------------- client


def chat_turn(port: int, model_id: str, messages: list[dict], max_tokens: int, fixed_length: bool,
              timeout: float) -> dict:  # fmt: skip
    """One streamed chat completion: the reply and its timings, measured at the client."""
    body = {"model": model_id, "messages": messages, "stream": True, "max_tokens": max_tokens, "temperature": 0}
    if fixed_length:
        body["generation_config"] = json.dumps(
            {"max_new_tokens": max_tokens, "min_new_tokens": max_tokens, "do_sample": False}
        )
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    chunks: list[float] = []
    content: list[str] = []
    reasoning: list[str] = []  # a reasoning model's thinking (e.g. Qwen3) streams as `reasoning_content` deltas
    usage: dict = {}
    finish = None
    try:
        t0 = time.perf_counter()
        conn.request("POST", "/v1/chat/completions", body=json.dumps(body),
                     headers={"Content-Type": "application/json", "Accept": "text/event-stream"})  # fmt: skip
        resp = conn.getresponse()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:500]!r}")
        while True:
            line = resp.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            now = time.perf_counter()
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(f"server error: {event['error']}")
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                for key, parts in (("reasoning_content", reasoning), ("content", content)):
                    piece = delta.get(key)
                    if piece:
                        chunks.append(now)
                        parts.append(piece)
                finish = choice.get("finish_reason") or finish
            usage = event.get("usage") or usage
        end = time.perf_counter()
    finally:
        conn.close()
    if not chunks:
        raise RuntimeError("the server streamed no content")
    completion = usage.get("completion_tokens") or len(chunks)
    gaps = sorted((b - a) * 1e3 for a, b in zip(chunks, chunks[1:]))
    decode = chunks[-1] - chunks[0]
    return {
        # the next turn resends the reply; a reply that was all thinking goes back as text, so the history still grows
        "reply": "".join(content) or "".join(reasoning),
        "reasoning_chars": sum(map(len, reasoning)),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": completion,
        "finish_reason": finish,
        "ttft_s": chunks[0] - t0,
        "e2e_s": end - t0,
        "decode_s": decode,
        "decode_tok_s": (completion - 1) / decode if completion > 1 and decode > 0 else None,
        "itl_ms": decode * 1e3 / (completion - 1) if completion > 1 else None,
        "chunks": len(chunks),
        "chunk_gap_p50_ms": percentile(gaps, 50),
        "chunk_gap_p99_ms": percentile(gaps, 99),
        "chunk_gap_max_ms": gaps[-1] if gaps else None,
    }


def percentile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    k = (len(sorted_values) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def conversation(a: argparse.Namespace, port: int, index: int, server: Server | None) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt(a.system_tokens)}]
    rows = []
    for turn in range(a.turns):
        messages.append({"role": "user", "content": user_message(index, turn, a.user_tokens)})
        r = chat_turn(port, a.model_id, messages, a.max_tokens, not a.natural_length, a.request_timeout)
        messages.append({"role": "assistant", "content": r.pop("reply")})
        r.update(conversation=index, turn=turn, server_rss_mb=server.rss_mb() if server else None)
        rows.append(r)
    return rows


def run_conversations(a: argparse.Namespace, port: int, first: int, count: int, server: Server | None) -> list[dict]:
    if a.concurrency <= 1:
        return [row for i in range(first, first + count) for row in conversation(a, port, i, server)]
    with ThreadPoolExecutor(a.concurrency) as pool:
        futures = [pool.submit(conversation, a, port, i, server) for i in range(first, first + count)]
        return [row for f in futures for row in f.result()]


# ----------------------------------------------------------------------------------------------------------- summary


def by_turn(rows: list[dict]) -> list[dict]:
    """Medians per turn index over the measured conversations."""
    out = []
    for turn in sorted({r["turn"] for r in rows}):
        rs = [r for r in rows if r["turn"] == turn]

        def med(key: str, rs=rs):
            values = [r[key] for r in rs if r.get(key) is not None]
            return statistics.median(values) if values else None

        out.append({"turn": turn, "n": len(rs), **{k: med(k) for k in (
            "prompt_tokens", "completion_tokens", "ttft_s", "e2e_s", "decode_tok_s", "itl_ms", "chunk_gap_p99_ms",
            "chunk_gap_max_ms", "server_rss_mb")}})  # fmt: skip
    return out


def ttft_slope(rows: list[dict]) -> dict | None:
    """Least-squares fit of time to first token on prompt length: ms per 1k prompt tokens (the cost of re-reading the
    history) and the intercept (fixed per-request cost)."""
    pts = [(r["prompt_tokens"], r["ttft_s"] * 1e3) for r in rows if r.get("prompt_tokens")]
    if len({x for x, _ in pts}) < 2:
        return None
    mx, my = statistics.fmean(x for x, _ in pts), statistics.fmean(y for _, y in pts)
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    slope = sum((x - mx) * (y - my) for x, y in pts) / sxx
    return {"ms_per_1k_prompt_tokens": slope * 1e3, "intercept_ms": my - slope * mx, "points": len(pts)}


def overall(rows: list[dict], wall_s: float) -> dict:
    def med(key: str):
        values = [r[key] for r in rows if r.get(key) is not None]
        return statistics.median(values) if values else None

    tokens = sum(r["completion_tokens"] for r in rows)
    return {
        "requests": len(rows),
        "completion_tokens": tokens,
        "wall_s": wall_s,
        "throughput_tok_s": tokens / wall_s if wall_s else None,
        "ttft_s_median": med("ttft_s"),
        "decode_tok_s_median": med("decode_tok_s"),
        "itl_ms_median": med("itl_ms"),
        "chunk_gap_max_ms": max((r["chunk_gap_max_ms"] or 0) for r in rows) if rows else None,
        "ttft_vs_prompt": ttft_slope(rows),
    }


# -------------------------------------------------------------------------------------------------------------- main


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-id", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--config-file", type=Path, help="scenarios (.jsonl): take the scenario's fields as defaults")
    p.add_argument("--scenario-index", type=int, default=0)
    p.add_argument("--turns", type=int, default=8, help="turns per conversation (default 8)")
    p.add_argument("--system-tokens", type=int, default=256, help="system prompt size, about (default 256)")
    p.add_argument("--user-tokens", type=int, default=64, help="each user message, about (default 64)")
    p.add_argument("--max-tokens", type=int, default=64, help="each reply (default 64)")
    p.add_argument("--natural-length", action="store_true", help="let replies stop at EOS (not comparable runs)")
    p.add_argument("--conversations", type=int, default=3, help="measured conversations (default 3)")
    p.add_argument("--warmup", type=int, default=1, help="conversations before measuring (default 1)")
    p.add_argument("--concurrency", type=int, default=1, help="conversations at once (default 1)")
    p.add_argument("--server-args", default="", help="arguments for `transformers serve` (one string)")
    p.add_argument("--python", default=sys.executable, help="interpreter for the server (default: this one)")
    p.add_argument("--port", type=int, default=0, help="default: a free one")
    p.add_argument("--startup-timeout", type=float, default=1200)
    p.add_argument("--request-timeout", type=float, default=600)
    p.add_argument("--profile", action="store_true", help="py-spy on the server during the measured conversations")
    p.add_argument("--profile-rate", type=int, default=100, help="py-spy samples per second (default 100)")
    p.add_argument("--profile-native", action="store_true", help="py-spy with native (C/C++) frames")
    p.add_argument("--commit", default=None, help="transformers commit, for the record")
    a = p.parse_args(argv)
    if a.config_file:  # the scenario's fields are defaults: explicit arguments win
        lines = [ln for ln in a.config_file.read_text().splitlines() if ln.strip().startswith("{")]
        scenario = json.loads(lines[a.scenario_index])
        given = {arg.split("=")[0] for arg in (argv if argv is not None else sys.argv[1:]) if arg.startswith("--")}
        for key, value in scenario.items():
            dest = key.replace("-", "_")
            if dest == "name":
                continue
            if not hasattr(a, dest):
                p.error(f"unknown scenario field {key!r}")
            if f"--{dest.replace('_', '-')}" not in given:
                setattr(a, dest, value)
        a.scenario_name = scenario.get("name", f"#{a.scenario_index}")
    else:
        a.scenario_name = None
    return a


def main(argv: list[str] | None = None) -> None:
    a = parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    port = a.port or free_port()
    server_args = shlex.split(a.server_args) if isinstance(a.server_args, str) else list(a.server_args)
    server = Server(a.python, a.model_id, port, server_args, a.out / "server.log")
    results: dict = {
        "model_id": a.model_id,
        "commit": a.commit,
        "scenario": a.scenario_name,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
        "server_command": server.command,
        "error": None,
    }
    rows: list[dict] = []
    profiler = None
    try:
        print(f"starting: {shlex.join(server.command)}", file=sys.stderr, flush=True)
        results["startup_s"] = server.start(a.startup_timeout)
        results["server_rss_mb_loaded"] = server.rss_mb()
        print(f"server up in {results['startup_s']:.1f} s; warm-up", file=sys.stderr, flush=True)
        if a.warmup:
            warm = run_conversations(a, port, 10_000, a.warmup, None)
            results["warmup"] = by_turn(warm)
        if a.profile:
            profiler = start_profiler(a, server)
        print(f"measuring {a.conversations} conversation(s) of {a.turns} turns", file=sys.stderr, flush=True)
        t0 = time.perf_counter()
        rows = run_conversations(a, port, 0, a.conversations, server)
        wall = time.perf_counter() - t0
        if profiler is not None:
            results["profile"] = stop_profiler(a, profiler, wall)
            profiler = None
        results["server_gpu_mb"] = server.gpu_mb()
        results["overall"] = overall(rows, wall)
    except Exception as e:  # keep what was measured
        results["error"] = repr(e)
        print(f"error: {e!r}", file=sys.stderr)
    finally:
        if profiler is not None:
            stop_profiler(a, profiler, None)
        server.stop()
    results["turns"] = rows
    results["by_turn"] = by_turn(rows)
    (a.out / "results.json").write_text(json.dumps(results, indent=1))
    from tperf.serve.summarize import format_results

    print(format_results(results))
    if results["error"]:
        sys.exit(1)


def start_profiler(a: argparse.Namespace, server: Server) -> subprocess.Popen:
    from tperf.common import pystacks

    pyspy = pystacks.find_pyspy(a.python) or pystacks.find_pyspy(sys.executable)
    if pyspy is None:
        raise RuntimeError("--profile needs py-spy (pip install py-spy)")
    folded = a.out / "server.folded"
    folded.unlink(missing_ok=True)
    command = pystacks.record_command(folded, rate=a.profile_rate, pid=server.proc.pid, threads=True,
                                      native=a.profile_native, pyspy=pyspy)  # fmt: skip
    proc = subprocess.Popen(command, stdout=open(a.out / "py-spy.log", "w"), stderr=subprocess.STDOUT)  # noqa: SIM115
    time.sleep(1.0)  # attached before the first measured request
    if proc.poll() is not None:
        raise RuntimeError(f"py-spy exited at once ({proc.returncode}): see {a.out / 'py-spy.log'}")
    return proc


def stop_profiler(a: argparse.Namespace, proc: subprocess.Popen, wall_s: float | None) -> dict | None:
    """Stop py-spy (SIGINT makes it write its output) and summarize the server's threads."""
    from tperf.common import pystacks

    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
    folded = a.out / "server.folded"
    if wall_s is None or not folded.exists():
        return None
    summary = pystacks.summarize(pystacks.parse_raw(folded.read_text()), a.profile_rate, by_thread=True)
    for g in summary["groups"]:
        g["busy_share"] = g["seconds"] / wall_s if wall_s else None
    summary["wall_s"] = wall_s
    summary["native"] = a.profile_native
    return summary


if __name__ == "__main__":
    main()
