"""Print tperf.tp.profile results: tperf tp summarize results/<run>/tp*.jsonl"""

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jsonl", nargs="+", type=Path)
    paths = p.parse_args(argv).jsonl
    rows = [json.loads(line) for path in paths for line in open(path) if line.startswith("{")]
    for r in sorted(rows, key=lambda r: (r["tp"], r["rank"])):
        print(
            f"\nTP{r['tp']} rank {r['rank']}: step {r['wall_ms']:.2f} ms wall, GPU busy {r['gpu_busy_ms']:.2f}, idle {r['idle_ms']:.2f}, guards {r.get('guard_ms', 0):.2f} ms, class_forward {r.get('class_forward')}"
        )
        print(
            "   kernels ms:",
            {k: round(v, 2) for k, v in sorted(r["kernels_ms"].items(), key=lambda kv: -kv[1])},
        )
        print(
            "   kernels n: ",
            {k: round(v, 1) for k, v in sorted(r["kernels_n"].items(), key=lambda kv: -kv[1])},
        )
        print("   cuda api/step:", r["cuda_api_per_step"])
        print("   dynamo:", r["dynamo"])
        for name, ms in r["top_kernels"][:8]:
            print(f"     {ms:7.3f} ms  {name}")


if __name__ == "__main__":
    main()
