import math

from tperf.ab.compare import analyze


def rows(gpu_of, e2e_of, rounds=6):
    out = []
    for r in range(1, rounds + 1):
        for side in ("base", "head"):
            out.append({"scenario": "s", "ab_round": r, "ab_side": side, "success": True,
                        "gpu_index": gpu_of(r, side), "e2e_ms_median": e2e_of(r, side)})  # fmt: skip
    return out


def e2e(result):
    return next(m for m in result["scenarios"][0]["metrics"] if m["name"] == "e2e")


def test_one_gpu_fits_out_which_side_ran_first():
    # Same code on both sides; whichever runs second is 2% slower, plus a little jitter
    jitter = [0.0, 0.001, -0.001, 0.0005, -0.0005, 0.0]

    def e2e_of(r, side):
        first = "base" if r % 2 else "head"
        return 100.0 * (1.02 if side != first else 1.0) * (1 + jitter[r - 1])

    m = e2e(analyze(rows(lambda r, side: 0, e2e_of)))
    assert abs(m["change_pct"]) < 0.1
    assert m["verdict"].startswith("no change")
    assert math.isclose(abs(m["gpu_pct"]), 2.0, abs_tol=0.1)  # the order gap, fitted like a GPU gap


def test_two_gpus_still_fit_the_gpu_difference():
    def gpu_of(r, side):
        return (0 if side == "base" else 1) if r % 2 else (1 if side == "base" else 0)

    m = e2e(analyze(rows(gpu_of, lambda r, side: 100.0 * (1.03 if gpu_of(r, side) == 1 else 1.0))))
    assert abs(m["change_pct"]) < 0.01
    assert m["verdict"].startswith("no change")
