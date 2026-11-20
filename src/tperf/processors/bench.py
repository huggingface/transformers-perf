"""Benchmark multimodal processors: the time to turn raw images, videos or audio into model inputs.

Each scenario (one JSON line in --scenarios) loads a published checkpoint's processor and times one call on a batch of
fixed inputs, after warm-up, with gc off:

  {"model": "Qwen/Qwen2.5-VL-7B-Instruct", "inputs": "images-mixed", "batch": 8, "backend": "torchvision"}

  model      checkpoint whose processor is used (AutoProcessor)
  inputs     images-mixed: one image per sample, sizes cycling through IMAGE_SIZES
             video-16f: one 16-frame 480x640 video per sample; audio-30s: one 30 s 16 kHz clip per sample
  batch      samples per call
  backend    image processor backend, "torchvision" or "pil" (images only)
  component  "processor" (default: the full call, text and placeholders included) or "image_processor" /
             "video_processor" / "feature_extractor" alone
  threads    torch intra-op threads (default: --threads)
  stage      "processor" (default), "encoder" (the model's vision / video / audio encoder on the processed inputs, on
             GPU in bf16: get_image_features, get_video_features or the audio encoder) or "both" (processor, copy to
             the GPU and encoder, as one call). Only the encoder (and projector) goes on the GPU, the language model
             or decoder stays on CPU, unused

Inputs are generated from a fixed seed, decoded and in memory before timing: resize, normalize and feature extraction
cost the same whatever the pixel or sample values, only the shapes matter, and these shapes cover the tiling and
dynamic-resolution paths. Each scenario writes one JSON file with the per-call times (seconds), the processor load
time, and a hash of the outputs: an A/B then also shows whether a change altered what the processor returns.
"""

import argparse
import gc
import hashlib
import inspect
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from transformers import AutoImageProcessor, AutoProcessor


# (height, width): small, portrait, landscape, large, square. Tiling and dynamic resolution depend on them.
IMAGE_SIZES = [(480, 640), (1024, 768), (720, 1280), (1536, 2048), (896, 896)]


def make_inputs(kind: str, batch: int) -> dict:
    rng = np.random.default_rng(0)
    if kind == "images-mixed":
        from PIL import Image

        return {
            "images": [
                Image.fromarray(rng.integers(0, 256, (*IMAGE_SIZES[i % len(IMAGE_SIZES)], 3), dtype=np.uint8))
                for i in range(batch)
            ]
        }
    if kind == "video-16f":
        return {"videos": [rng.integers(0, 256, (16, 480, 640, 3), dtype=np.uint8) for _ in range(batch)]}
    if kind == "audio-30s":
        return {"audio": [rng.standard_normal(30 * 16000).astype(np.float32) * 0.1 for _ in range(batch)]}
    raise ValueError(f"unknown inputs {kind!r}")


def prompts(processor, inputs: dict, batch: int) -> list[str] | None:
    """One chat-formatted prompt per sample with the media placeholder, as a user of the processor would build it."""
    if not hasattr(processor, "apply_chat_template") or "audio" in inputs:
        return None
    media = "image" if "images" in inputs else "video"
    messages = [
        {
            "role": "user",
            "content": [{"type": media}, {"type": "text", "text": "Describe this."}],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    return [text] * batch


def call_fn(processor, scenario: dict, inputs: dict, batch: int):
    component = scenario.get("component", "processor")
    if component != "processor":
        target = getattr(processor, component)
        return lambda: target(**inputs, return_tensors="pt")
    kwargs = dict(inputs)
    if "audio" in kwargs:
        kwargs["sampling_rate"] = 16000
    text = prompts(processor, inputs, batch)
    if text is not None:  # prompts differ in length when the images do (one placeholder per patch)
        kwargs.update(text=text, padding=True)
    return lambda: processor(**kwargs, return_tensors="pt")


ENCODER_METHOD = {
    "images-mixed": "get_image_features",
    "video-16f": "get_video_features",
}


# Submodules the encoder stage never runs: kept on CPU, so only the encoder (and projector) takes GPU memory
OFF_GPU = {"language_model", "lm_head", "decoder", "proj_out"}


def encoder_device_map(cls, model_id: str) -> dict:
    """A device_map with the modules in OFF_GPU on CPU and everything else on GPU 0."""
    from transformers import AutoConfig

    with torch.device("meta"):
        meta = cls.from_config(AutoConfig.from_pretrained(model_id))
    device_map = {}

    def place(module, prefix):
        for name, child in module.named_children():
            if name in OFF_GPU:
                device_map[prefix + name] = "cpu"
            elif any(n.rsplit(".", 1)[-1] in OFF_GPU for n, _ in child.named_modules()):
                place(child, f"{prefix}{name}.")
            else:
                device_map[prefix + name] = 0
        for name, _ in module.named_parameters(recurse=False):
            device_map[prefix + name] = 0

    place(meta, "")
    return device_map


_loaded: dict = {}  # the last model loaded: the encoder scenarios of one checkpoint follow each other


def load_model(model_id: str, audio: bool):
    from transformers import AutoModelForImageTextToText, AutoModelForSpeechSeq2Seq

    if model_id not in _loaded:
        _loaded.clear()
        gc.collect()
        torch.cuda.empty_cache()
        cls = AutoModelForSpeechSeq2Seq if audio else AutoModelForImageTextToText
        _loaded[model_id] = cls.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=encoder_device_map(cls, model_id)
        ).eval()
    return _loaded[model_id]


def load_encoder(scenario: dict):
    """The model's encoder for the scenario's inputs: `fn(processed) -> features`, on GPU in bf16."""
    audio = scenario["inputs"] == "audio-30s"
    model = load_model(scenario["model"], audio)
    if audio:
        encoder = model.get_encoder()
        return lambda processed: encoder(input_features=processed["input_features"]).last_hidden_state
    method = getattr(model, ENCODER_METHOD[scenario["inputs"]])
    params = set(inspect.signature(method).parameters)

    def encode(processed):
        out = method(**{k: v for k, v in processed.items() if k in params})
        out = getattr(out, "pooler_output", out)  # some return a model output, some a tensor or a tuple of tensors
        return torch.cat([o.flatten(0, -2) for o in out]) if isinstance(out, (tuple, list)) else out

    return encode


def to_device(processed, dtype=torch.bfloat16) -> dict:
    """Processor output on the GPU, floating point tensors in the encoder's dtype."""
    return {
        k: (
            v.to(
                "cuda",
                dtype=dtype if v.is_floating_point() else None,
                non_blocking=True,
            )
        )
        for k, v in processed.items()
        if isinstance(v, torch.Tensor)
    }


def count_syncs(fn) -> dict:
    """GPU -> CPU synchronizations in one call (`torch.cuda.set_sync_debug_mode`), excluding the final
    synchronize of the benchmark itself: each one stalls the CPU until the GPU catches up, which in a serving engine
    breaks the overlap of the encoder with the rest of the step. Lists where they come from."""
    import traceback
    import warnings

    sources = []

    def on_warning(message, category, filename, lineno, file=None, line=None):
        if "synchroniz" in str(message):
            frames = [f for f in traceback.extract_stack()[:-1] if "/transformers/" in f.filename]
            where = f"{frames[-1].filename.split('/transformers/')[-1]}:{frames[-1].lineno}" if frames else filename
            sources.append(where)

    old = warnings.showwarning
    torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.showwarning = on_warning
            with torch.inference_mode():
                fn()
    finally:
        torch.cuda.set_sync_debug_mode("default")
        warnings.showwarning = old
    by_source: dict[str, int] = {}
    for src in sources:
        by_source[src] = by_source.get(src, 0) + 1
    return {
        "count": len(sources),
        "sources": dict(sorted(by_source.items(), key=lambda kv: -kv[1])[:10]),
    }


def features_summary(features: torch.Tensor) -> dict:
    """Shape and a rounded checksum: GPU kernels may not be bitwise reproducible, so no exact hash."""
    f = features.float()
    return {
        "shape": list(features.shape),
        "mean_abs": float(f.abs().mean()),
        "mean": float(f.mean()),
    }


def output_hash(output) -> dict:
    """Shapes and a hash of every tensor the processor returns."""
    h, shapes = hashlib.sha256(), {}
    for key in sorted(output.keys()):
        value = output[key]
        if isinstance(value, torch.Tensor):
            t = value.detach().cpu().contiguous()
            shapes[key] = list(t.shape)
            h.update(key.encode())
            h.update(t.view(torch.uint8).numpy().tobytes() if t.dtype != torch.bool else t.numpy().tobytes())
    return {"shapes": shapes, "sha256": h.hexdigest()[:16]}


def scenario_name(s: dict, threads: int) -> str:
    parts = [
        s["model"].split("/")[-1],
        s["inputs"],
        f"b{s['batch']}",
        s.get("component", "processor"),
    ]
    if s.get("stage", "processor") != "processor":
        parts.append(s["stage"])
    if "backend" in s:
        parts.append(s["backend"])
    parts.append(f"t{threads}")
    return "-".join(parts)


def run(scenario: dict, args: argparse.Namespace) -> dict:
    threads = scenario.get("threads", args.threads)
    torch.set_num_threads(threads)
    name = scenario_name(scenario, threads)
    result = {
        "name": name,
        "config": {**scenario, "threads": threads},
        "success": False,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        t0 = time.perf_counter()
        processor = AutoProcessor.from_pretrained(scenario["model"])
        if "backend" in scenario:  # the image processor's backend only: video processors have no PIL backend
            processor.image_processor = AutoImageProcessor.from_pretrained(
                scenario["model"], backend=scenario["backend"]
            )
        result["load_s"] = time.perf_counter() - t0
        if "backend" in scenario:
            result["image_processor_class"] = type(processor.image_processor).__name__
        inputs = make_inputs(scenario["inputs"], scenario["batch"])
        fn = call_fn(processor, scenario, inputs, scenario["batch"])
        stage = scenario.get("stage", "processor")
        if stage != "processor":
            process = fn
            result["output"] = output_hash(processed := process())
            t0 = time.perf_counter()
            encode = load_encoder(scenario)
            result["model_load_s"] = time.perf_counter() - t0  # ~0 when the previous scenario loaded the same model
            if stage == "encoder":  # processed once, already on the GPU
                on_gpu = to_device(processed)
                step = lambda: encode(on_gpu)  # noqa: E731
            else:
                step = lambda: encode(to_device(process()))  # noqa: E731

            def fn():
                with torch.inference_mode():
                    out = step()
                torch.cuda.synchronize()
                return out

        t0 = time.perf_counter()
        output = fn()
        result["first_call_s"] = time.perf_counter() - t0
        if stage == "processor":
            result["output"] = output_hash(output)
        else:
            result["features"] = features_summary(output)
        for _ in range(args.warmup):
            fn()
        if stage != "processor":
            result["syncs"] = count_syncs(step)
        gc.collect()
        gc.disable()
        try:
            samples = []
            for _ in range(args.iterations):
                t0 = time.perf_counter()
                fn()
                samples.append(time.perf_counter() - t0)
        finally:
            gc.enable()
        result.update(samples_s=samples, success=True)
    except Exception as e:  # one scenario failing must not stop the others
        result["error"] = f"{type(e).__name__}: {e}"[:2000]
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--scenarios",
        required=True,
        help="JSONL, one scenario per line (# comments allowed)",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--threads", type=int, default=int(os.environ.get("OMP_NUM_THREADS", 1)))
    args = p.parse_args(argv)

    scenarios = [json.loads(line) for line in Path(args.scenarios).read_text().splitlines() if line.startswith("{")]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for scenario in scenarios:
        if scenario.get("threads", args.threads) != args.threads:  # pinned to another pass of `tperf processors run`
            continue
        result = run(scenario, args)
        (out / f"processor_{result['name']}.json").write_text(json.dumps(result, indent=1))
        if result["success"]:
            s = sorted(result["samples_s"])
            print(
                f"{result['name']:<75} median {1e3 * s[len(s) // 2]:9.2f} ms  min {1e3 * s[0]:9.2f}  "
                f"max {1e3 * s[-1]:9.2f}  out {result['output']['sha256']}",
                flush=True,
            )
        else:
            print(f"{result['name']:<75} FAILED {result['error'][:200]}", flush=True)


if __name__ == "__main__":
    main()
