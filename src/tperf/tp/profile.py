"""Profile compiled decode steps, with or without tensor parallelism, and say where the step time goes.

Launched by `tperf tp profile` (tperf.tp.profile_run): plain python for one GPU, torchrun for TP. Builds the model the way harness does
for the compiled scenario (bf16, sdpa, static cache, CompileConfig(fullgraph=True): reduce-overhead, CUDA graphs).
After warm-up, two generate() calls are profiled, one producing 1 token (prefill) and one producing 1 + K tokens; the
difference divided by K is one decode step. Each rank prints one JSON line:

  wall_ms                    decode step, host wall clock (synchronized)
  gpu_busy_ms                union of kernel intervals per step, idle_ms = wall_ms - gpu_busy_ms
  kernels_ms{category}       kernel time per category (gemm, nccl, attention, triton, other), kernels_n{category}
  top_kernels                the largest kernels by time per step
  cuda_api_per_step{name}    runtime calls per step (cudaGraphLaunch vs cudaLaunchKernel: the latter run outside graphs)
  dynamo                     graph breaks / recompiles / unique graphs counted during warm-up and profiling
Chrome traces are written to --out (one per rank and generate call).
"""

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict

import torch
import torch.distributed as dist
from torch._dynamo.utils import counters

from transformers import AutoModelForCausalLM, GenerationConfig
from transformers.generation.configuration_utils import CompileConfig


CATEGORIES = [
    ("nccl", re.compile(r"nccl", re.I)),
    ("attention", re.compile(r"flash|fmha|attention|sdpa|efficient|softmax", re.I)),
    (
        "gemm",
        re.compile(r"gemm|gemv|cutlass|xmma|cublas|sm80_|sm86_|ampere_|matmul|splitk|dot", re.I),
    ),
    ("triton", re.compile(r"^triton_")),
]


def category(name: str) -> str:
    for cat, pattern in CATEGORIES:
        if pattern.search(name):
            return cat
    return "other"


def union_ms(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, float("-inf")
    for s, e in sorted(intervals):
        if s > end:
            total += e - s
            end = e
        elif e > end:
            total += e - end
            end = e
    return total / 1000  # us -> ms


def summarize(trace_path: str) -> dict:
    with open(trace_path) as f:
        events = json.load(f)["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    api = Counter(e["name"] for e in events if e.get("cat") == "cuda_runtime")
    by_cat_ms, by_cat_n, by_name = defaultdict(float), Counter(), defaultdict(float)
    for k in kernels:
        cat = category(k["name"])
        by_cat_ms[cat] += k["dur"] / 1000
        by_cat_n[cat] += 1
        by_name[k["name"]] += k["dur"] / 1000
    return {
        "gpu_busy_ms": union_ms([(k["ts"], k["ts"] + k["dur"]) for k in kernels]),
        "kernels_ms": dict(by_cat_ms),
        "kernels_n": dict(by_cat_n),
        "by_name": dict(by_name),
        "api": dict(api),
        "guard_ms": sum(e["dur"] for e in events if e.get("name") == "TorchDynamo Cache Lookup") / 1000,
    }


_TP_CLASSES: dict[type, type] = {}


def class_level_tp_forward(model: torch.nn.Module) -> tuple[int, int]:
    """Experiment: replace the per-instance `tp_forward` closures that TensorParallelLayer.install_forward puts on
    each sharded module by a class-level forward reading the style and mesh from module attributes. Same transforms,
    but dynamo no longer guards every closure cell (and the module hook dicts) of every sharded module on each call.
    Returns (converted, skipped)."""

    def tp_class(base: type) -> type:
        if base not in _TP_CLASSES:

            def forward(self, *args, **kwargs):
                style, mesh = self._tp_style, self._tp_mesh
                args, kwargs = style.transform_inputs_pre_forward(self, args, kwargs, mesh)
                with style.context_around_forward(self, mesh):
                    output = base.forward(self, *args, **kwargs)
                return style.transform_output_post_forward(self, output, mesh)

            _TP_CLASSES[base] = type(base.__name__, (base,), {"forward": forward})
        return _TP_CLASSES[base]

    converted = skipped = 0
    for module in model.modules():
        fn = module.__dict__.get("forward")
        if fn is None or getattr(fn, "__name__", "") != "tp_forward":
            continue
        cells = dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in fn.__closure__ or ())))
        original = cells.get("original_forward")
        if set(cells) != {"self", "module", "mesh", "original_forward"} or getattr(original, "__func__", None) is not (
            type(module).forward
        ):
            skipped += 1
            continue
        del module.forward
        module._tp_style, module._tp_mesh = cells["self"], cells["mesh"]
        module.__class__ = tp_class(type(module))
        converted += 1
    return converted, skipped


def _method_tp_forward(module, *args, **kwargs):
    style, mesh = module._tp_style, module._tp_mesh
    args, kwargs = style.transform_inputs_pre_forward(module, args, kwargs, mesh)
    with style.context_around_forward(module, mesh):
        output = type(module).forward(module, *args, **kwargs)
    return style.transform_output_post_forward(module, output, mesh)


def method_tp_forward(model: torch.nn.Module) -> tuple[int, int]:
    """Experiment: like class_level_tp_forward but without changing the module class: the instance forward becomes
    a bound method of one shared function (types.MethodType) reading the style and mesh from module attributes."""
    import types

    converted = skipped = 0
    for module in model.modules():
        fn = module.__dict__.get("forward")
        if fn is None or getattr(fn, "__name__", "") != "tp_forward":
            continue
        cells = dict(zip(fn.__code__.co_freevars, (c.cell_contents for c in fn.__closure__ or ())))
        original = cells.get("original_forward")
        if set(cells) != {"self", "module", "mesh", "original_forward"} or getattr(original, "__func__", None) is not (
            type(module).forward
        ):
            skipped += 1
            continue
        module._tp_style, module._tp_mesh = cells["self"], cells["mesh"]
        module.forward = types.MethodType(_method_tp_forward, module)
        converted += 1
    return converted, skipped


def local_tp_params(model: torch.nn.Module) -> int:
    """Experiment (after class_level_tp_forward): for modules whose TP style runs on local tensors at inference, store
    the local shard as a plain parameter instead of a DTensor. Dynamo then guards a plain tensor, not the DTensor's
    subclass metadata (a Python lambda guard on its spec, mesh, local tensor, ...). Returns the number converted."""
    from torch.distributed.tensor import DTensor

    converted = 0
    with torch.no_grad():  # should_use_local_tensors() is evaluated as at inference
        for module in model.modules():
            style = module.__dict__.get("_tp_style")
            if style is None or not style.should_use_local_tensors(module):
                continue
            for name, param in list(module._parameters.items()):
                if isinstance(param, DTensor):
                    module._parameters[name] = torch.nn.Parameter(param.to_local(), requires_grad=False)
                    converted += 1
    return converted


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--prompt-tokens", type=int, default=128)
    p.add_argument(
        "--decode-tokens",
        type=int,
        default=16,
        help="K: decode steps in the profiled window",
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--class-forward",
        action="store_true",
        help="experiment: see class_level_tp_forward",
    )
    p.add_argument(
        "--recursive-dict-tags",
        action="store_true",
        help="torch._dynamo.config.use_recursive_dict_tags_for_guards = True",
    )
    p.add_argument(
        "--compile-guard-filter",
        action="store_true",
        help="CompileConfig(fullgraph=True, guard_filter_fn=skip_guard_on_inbuilt_nn_modules_unsafe) (needs the field)",
    )
    p.add_argument(
        "--guard-filter",
        action="store_true",
        help="compile with guard_filter_fn=torch.compiler.skip_guard_on_inbuilt_nn_modules_unsafe (CUDA graphs on)",
    )
    p.add_argument(
        "--method-forward",
        action="store_true",
        help="experiment: see method_tp_forward",
    )
    p.add_argument(
        "--local-params",
        action="store_true",
        help="experiment (implies --class-forward): see local_tp_params",
    )
    args = p.parse_args(argv)

    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    tp_kwargs = {}
    if world > 1:
        from transformers.distributed import DistributedConfig

        tp_kwargs = {"distributed_config": DistributedConfig(tp_size=world)}
    else:
        tp_kwargs = {"device_map": "cuda"}

    if args.recursive_dict_tags:
        torch._dynamo.config.use_recursive_dict_tags_for_guards = True
    compile_config = CompileConfig(fullgraph=True)
    if args.guard_filter:  # torch.compile takes mode or options: reduce-overhead is triton.cudagraphs
        compile_config = CompileConfig(
            fullgraph=True,
            mode=None,
            options={
                "triton.cudagraphs": True,
                "guard_filter_fn": torch.compiler.skip_guard_on_inbuilt_nn_modules_unsafe,
            },
        )
    if args.compile_guard_filter:
        compile_config = CompileConfig(
            fullgraph=True,
            guard_filter_fn=torch.compiler.skip_guard_on_inbuilt_nn_modules_unsafe,
        )
    gen_kwargs = {
        "do_sample": False,
        "cache_implementation": "static",
        "eos_token_id": None,  # always generate exactly max_new_tokens
    }
    if args.compile_guard_filter:  # through from_pretrained, as a user would
        gen_kwargs["compile_config"] = compile_config
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        generation_config=GenerationConfig(**gen_kwargs),
        **tp_kwargs,
    ).eval()
    if not args.compile_guard_filter:  # set after loading: a guard_filter_fn in options can't be serialized
        model.generation_config.compile_config = compile_config
    converted = (
        class_level_tp_forward(model)
        if args.class_forward or args.local_params
        else method_tp_forward(model)
        if args.method_forward
        else None
    )
    local_params = local_tp_params(model) if args.local_params else None
    torch.manual_seed(0)
    input_ids = torch.randint(100, 30000, (1, args.prompt_tokens), device=model.device)
    attention_mask = torch.ones_like(input_ids)

    last_tokens: list[int] = []

    def generate(n: int) -> float:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=n,
            min_new_tokens=n,
        )
        torch.cuda.synchronize()
        assert out.shape[-1] == args.prompt_tokens + n, out.shape
        last_tokens[:] = out[0, args.prompt_tokens :].tolist()
        return (time.perf_counter() - t0) * 1000

    k = args.decode_tokens
    with torch.no_grad():
        for _ in range(args.warmup):
            generate(1 + k)
            generate(1)
        # wall clock, unprofiled: median of 5 (the first call of each n re-records, see below)
        walls = {n: sorted(generate(n) for _ in range(6))[3] for n in (1, 1 + k)}

        os.makedirs(args.out, exist_ok=True)
        summaries = {}
        for n in (1, 1 + k):
            # generate(n) sizes the static cache from n: a call with another n allocates a new cache, and the next
            # compiled decode re-records its CUDA graph. Re-record outside the profiled window.
            generate(n)
            if world > 1:
                dist.barrier()
            prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            )
            with prof:
                generate(n)
            path = f"{args.out}/tp{world}-rank{rank}-n{n}.json"
            prof.export_chrome_trace(path)
            summaries[n] = summarize(path)

    a, b = summaries[1], summaries[1 + k]
    per_step = lambda key: {  # noqa: E731
        name: round((b[key].get(name, 0) - a[key].get(name, 0)) / k, 3) for name in set(a[key]) | set(b[key])
    }
    wall = (walls[1 + k] - walls[1]) / k
    busy = (b["gpu_busy_ms"] - a["gpu_busy_ms"]) / k
    top = sorted(per_step("by_name").items(), key=lambda kv: -kv[1])[:12]
    result = {
        "tp": world,
        "rank": rank,
        "class_forward": converted,
        "local_params": local_params,
        "tokens": last_tokens,
        "wall_ms": round(wall, 3),
        "gpu_busy_ms": round(busy, 3),
        "idle_ms": round(wall - busy, 3),
        "guard_ms": round((b["guard_ms"] - a["guard_ms"]) / k, 3),
        "kernels_ms": per_step("kernels_ms"),
        "kernels_n": per_step("kernels_n"),
        "top_kernels": [(name[:90], ms) for name, ms in top],
        "cuda_api_per_step": {k_: v for k_, v in per_step("api").items() if abs(v) >= 0.5},
        "dynamo": {
            "graph_breaks": sum(counters["graph_break"].values()),
            "graph_break_reasons": list(counters["graph_break"])[:5],
            "unique_graphs": counters["stats"]["unique_graphs"],
            "recompiles": counters["recompiles"] if "recompiles" in counters else {},
        },
    }
    print(json.dumps(result), flush=True)
    if world > 1:
        dist.barrier()
    # destroy_process_group() hangs after CUDA-graph-captured collectives; nothing is left to flush
    os._exit(0)


if __name__ == "__main__":
    main()
