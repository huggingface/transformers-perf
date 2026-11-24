"""Start the transformers CLI (`transformers serve ...`) so that py-spy can attach to it without root.

    python -m tperf.serve.entry serve <model> [options]     (the same arguments as `transformers ...`)

On Linux with Yama's ptrace_scope=1 (Ubuntu's default) a process may only be traced by its ancestors; tperf.serve.bench
starts py-spy beside the server, not above it. PR_SET_PTRACER_ANY lets any process of the same user attach, for this
process only. Elsewhere (macOS, other kernels) this does nothing and the CLI starts as usual.
"""

import sys


PR_SET_PTRACER = 0x59616D61
PR_SET_PTRACER_ANY = -1


def allow_tracing() -> bool:
    if not sys.platform.startswith("linux"):
        return False
    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        return libc.prctl(PR_SET_PTRACER, ctypes.c_ulong(PR_SET_PTRACER_ANY), 0, 0, 0) == 0
    except (OSError, AttributeError):
        return False


def main(argv: list[str] | None = None) -> None:
    allow_tracing()
    sys.argv = ["transformers", *(sys.argv[1:] if argv is None else argv)]
    from transformers.cli.transformers import main as transformers_main

    transformers_main()


if __name__ == "__main__":
    main()
