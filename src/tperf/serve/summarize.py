"""Print `tperf serve` results, or compare runs turn by turn.

    tperf serve summarize <dir>                 one run: per-turn medians, overall, the server's Python time
    tperf serve summarize <base dir> <dir>...   later runs against the first, as % change per turn

Each <dir> holds a results.json (tperf.serve.bench). Runs compare only when they played the same conversation
(turns, message sizes, reply length): a mismatch is printed and compared anyway. Medians over few conversations:
read changes of a few percent as noise unless they repeat.
"""

import argparse
import json
import sys
from pathlib import Path


COLUMNS = (  # key, header, format, scale
    ("prompt_tokens", "prompt", "{:>7.0f}", 1),
    ("ttft_s", "TTFT ms", "{:>9.1f}", 1e3),
    ("decode_tok_s", "decode tok/s", "{:>12.1f}", 1),
    ("itl_ms", "ITL ms", "{:>8.2f}", 1),
    ("chunk_gap_max_ms", "max gap ms", "{:>10.1f}", 1),
    ("e2e_s", "e2e ms", "{:>9.1f}", 1e3),
    ("server_rss_mb", "RSS MB", "{:>8.0f}", 1),
)
# Lower is better for these; higher for decode_tok_s
LOWER_IS_BETTER = {"ttft_s", "itl_ms", "e2e_s", "chunk_gap_max_ms", "server_rss_mb"}
SHAPE = ("turns", "system_tokens", "user_tokens", "max_tokens", "natural_length", "concurrency", "server_args")


def cell(value, fmt: str, scale: float) -> str:
    width = int(fmt.split(">")[1].split(".")[0])
    return fmt.format(value * scale) if value is not None else "–".rjust(width)


def turn_table(by_turn: list[dict]) -> str:
    head = f"{'turn':>4}  " + "  ".join(h.rjust(int(f.split(">")[1].split(".")[0])) for _, h, f, _ in COLUMNS)
    lines = [head]
    for t in by_turn:
        lines.append(f"{t['turn'] + 1:>4}  " + "  ".join(cell(t.get(k), f, s) for k, _, f, s in COLUMNS))
    return "\n".join(lines)


def format_profile(profile: dict | None) -> str:
    if not profile:
        return ""
    from tperf.common.pystacks import format_summary

    lines = [f"\nserver Python time (py-spy, {profile.get('wall_s', 0):.1f} s measured):"]
    for g in profile["groups"]:
        if g.get("busy_share") is not None:
            lines.append(f"  {g['busy_share'] * 100:5.1f}% busy  {g['name']}")
    return "\n".join(lines) + "\n" + format_summary(profile, limit=10)


def format_results(r: dict) -> str:
    a = r.get("args") or {}
    o = r.get("overall") or {}
    lines = [
        f"{r.get('model_id')} @ {str(r.get('commit') or '?')[:12]}"
        + (f" [{r['scenario']}]" if r.get("scenario") else "")
        + f": {a.get('conversations')} conversation(s) x {a.get('turns')} turns, concurrency {a.get('concurrency')},"
        f" replies of {a.get('max_tokens')} tokens, server args: {a.get('server_args') or '(none)'}",
    ]
    if r.get("startup_s") is not None:
        lines.append(f"server start (imports + load): {r['startup_s']:.1f} s, RSS {r.get('server_rss_mb_loaded')} MB")
    if r.get("error"):
        lines.append(f"ERROR: {r['error']}")
    if r.get("by_turn"):
        lines += ["", turn_table(r["by_turn"])]
    if o:
        slope = o.get("ttft_vs_prompt") or {}
        lines.append("")
        lines.append(
            f"overall: {o['requests']} requests, {o['completion_tokens']} tokens in {o['wall_s']:.1f} s = "
            f"{o['throughput_tok_s']:.1f} tok/s; median TTFT {o['ttft_s_median'] * 1e3:.1f} ms, "
            f"decode {o['decode_tok_s_median'] or 0:.1f} tok/s, max chunk gap {o['chunk_gap_max_ms'] or 0:.1f} ms"
        )
        if slope:
            lines.append(
                f"TTFT vs prompt length: +{slope['ms_per_1k_prompt_tokens']:.1f} ms per 1k prompt tokens, "
                f"{slope['intercept_ms']:.1f} ms fixed (fit over {slope['points']} requests)"
            )
        if r.get("server_gpu_mb") is not None:
            lines.append(f"server GPU memory at the end: {r['server_gpu_mb']:.0f} MB")
    lines.append(format_profile(r.get("profile")))
    return "\n".join(lines).rstrip()


def change(base, new, key: str) -> str:
    if base is None or new is None or not base:
        return "–"
    pct = (new - base) / base * 100
    worse = pct > 0 if key in LOWER_IS_BETTER else pct < 0
    flag = " !" if worse and abs(pct) >= 5 else ""
    return f"{pct:+.1f}%{flag}"


def compare(base: dict, other: dict) -> str:
    ba, oa = base.get("args") or {}, other.get("args") or {}
    lines = []
    diff = [k for k in SHAPE if ba.get(k) != oa.get(k)]
    if diff:
        lines.append("WARNING: not the same conversation: " + ", ".join(f"{k} {ba.get(k)!r} vs {oa.get(k)!r}"
                                                                         for k in diff))  # fmt: skip
    keys = [c for c in COLUMNS if c[0] != "prompt_tokens"]
    lines.append(f"{'turn':>4}  {'prompt':>7}  " + "  ".join(f"{h:>12}" for _, h, _, _ in keys))
    other_turns = {t["turn"]: t for t in other.get("by_turn") or []}
    for t in base.get("by_turn") or []:
        o = other_turns.get(t["turn"], {})
        lines.append(f"{t['turn'] + 1:>4}  {cell(t.get('prompt_tokens'), '{:>7.0f}', 1)}  "
                     + "  ".join(f"{change(t.get(k), o.get(k), k):>12}" for k, _, _, _ in keys))  # fmt: skip
    bo, oo = base.get("overall") or {}, other.get("overall") or {}
    for key, label in (("ttft_s_median", "median TTFT"), ("decode_tok_s_median", "median decode tok/s"),
                       ("throughput_tok_s", "throughput tok/s")):  # fmt: skip
        k = "ttft_s" if key == "ttft_s_median" else "decode_tok_s"
        b, o = bo.get(key), oo.get(key)
        scale = 1e3 if key == "ttft_s_median" else 1  # TTFT in ms
        shown = " -> ".join(f"{v * scale:.1f}" if v is not None else "–" for v in (b, o))
        lines.append(f"{label}{' ms' if scale != 1 else ''}: {shown} ({change(b, o, k)})")
    bs, os_ = (base.get("overall") or {}).get("ttft_vs_prompt"), (other.get("overall") or {}).get("ttft_vs_prompt")
    if bs and os_:
        lines.append(f"TTFT per 1k prompt tokens: {bs['ms_per_1k_prompt_tokens']:.1f} -> "
                     f"{os_['ms_per_1k_prompt_tokens']:.1f} ms")  # fmt: skip
    lines.append("('!': worse by 5% or more)")
    return "\n".join(lines)


def load(path: Path) -> dict:
    path = path / "results.json" if path.is_dir() else path
    return json.loads(path.read_text())


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path, help="result directories (or results.json files)")
    a = p.parse_args(argv)
    runs = [load(r) for r in a.runs]
    print(format_results(runs[0]))
    for path, run in zip(a.runs[1:], runs[1:]):
        print(f"\n=== {path} against {a.runs[0]}")
        print(compare(runs[0], run))


if __name__ == "__main__":
    sys.exit(main())
