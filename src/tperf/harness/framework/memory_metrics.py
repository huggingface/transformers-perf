"""Memory and disk counters read at phase boundaries (after load, after the first generation, after the measurements).

Nothing here samples: each function reads counters the kernel, the CUDA caching allocator or NVML keep anyway, so
calling them between timed regions adds no work to the measurements. They answer "how much", plus a breakdown of the
model's own tensors (weights and buffers by dtype and by module) for "what"; `tperf memory profile` attributes the
rest of the memory to the code that allocated it.

Host values come from /proc (Linux); elsewhere they are absent rather than wrong.
"""

import os
import re
from collections import defaultdict
from pathlib import Path


MB = 1024**2


def _proc_kb_fields(path: str, keys: tuple[str, ...]) -> dict[str, float]:
    """`Key:   123 kB` lines of a /proc file, in MB."""
    out = {}
    try:
        for line in Path(path).read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in keys:
                out[key] = int(rest.split()[0]) / 1024
    except (OSError, ValueError, IndexError):
        return {}
    return out


def _proc_io() -> dict[str, int]:
    try:
        return {
            k: int(v) for k, _, v in (line.partition(": ") for line in Path("/proc/self/io").read_text().splitlines())
        }
    except (OSError, ValueError):
        return {}


def host_memory() -> dict[str, float | None]:
    """This process's resident memory (MB) and its read counters (bytes).

    - `rss_peak_mb` (VmHWM): highest resident set since the process started, or since `reset_host_peak()`
    - `rss_anon_mb`: memory the process wrote (heap, copies of tensors); `rss_file_mb`: file pages it mapped and touched
      (an mmap'd safetensors file), which the kernel can drop and reread
    - `read_bytes`: bytes this process caused to be fetched from storage, through read calls or page faults on mapped
      files (safetensors); about the model's size is a cold load, near zero a load from the page cache
    - `rchar`: bytes through read calls, from the page cache or not; mapped files do not count
    """
    status = _proc_kb_fields("/proc/self/status", ("VmHWM", "VmRSS", "RssAnon", "RssFile", "RssShmem"))
    if not status:
        return {}
    out = {
        "rss_peak_mb": status.get("VmHWM"),
        "rss_mb": status.get("VmRSS"),
        "rss_anon_mb": status.get("RssAnon"),
        "rss_file_mb": status.get("RssFile"),
        "rss_shmem_mb": status.get("RssShmem"),
    }
    io = _proc_io()
    out["read_bytes"] = io.get("read_bytes")
    out["rchar"] = io.get("rchar")
    return out


def reset_host_peak() -> bool:
    """Reset VmHWM to the current RSS (Linux >= 4.0), so the next `host_memory()` peak covers one phase."""
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


def device_memory(accelerator) -> dict[str, float]:
    """The caching allocator's view (MB) of the current device. `peak_*` count since the last
    `reset_peak_memory_stats()`; `alloc_retries` is cumulative (an allocation that failed, made the allocator free
    its cached blocks and try again: fragmentation or memory pressure, and slow)."""
    if not hasattr(accelerator, "memory_stats"):
        return {}
    try:
        stats = accelerator.memory_stats()
    except (RuntimeError, AssertionError):
        return {}
    if not isinstance(stats, dict) or not stats:
        return {}
    return {
        "allocated_mb": stats.get("allocated_bytes.all.current", 0) / MB,
        "reserved_mb": stats.get("reserved_bytes.all.current", 0) / MB,
        "peak_allocated_mb": stats.get("allocated_bytes.all.peak", 0) / MB,
        "peak_reserved_mb": stats.get("reserved_bytes.all.peak", 0) / MB,
        "inactive_split_mb": stats.get("inactive_split_bytes.all.current", 0) / MB,
        "alloc_retries": stats.get("num_alloc_retries", 0),
        "ooms": stats.get("num_ooms", 0),
    }


def process_gpu_memory_mb(device_index: int = 0) -> float | None:
    """GPU memory NVML charges to this process on torch's `device_index`, in MB: the caching allocator's reserve plus
    what it does not see (the CUDA context, cuBLAS/cuDNN workspaces, NCCL buffers, other allocators).

    The device is found by UUID: NVML indexes ignore CUDA_VISIBLE_DEVICES, torch's do not."""
    try:
        import pynvml
        import torch

        uuid = str(torch.cuda.get_device_properties(device_index).uuid)
        pynvml.nvmlInit()
    except Exception:
        return None
    try:
        handle = pynvml.nvmlDeviceGetHandleByUUID(uuid if uuid.startswith("GPU-") else f"GPU-{uuid}")
        for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            if proc.pid == os.getpid() and proc.usedGpuMemory is not None:
                return proc.usedGpuMemory / MB
        return None
    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


_LAYER_INDEX = re.compile(r"\.\d+(?=\.|$)")


def module_group(name: str) -> str:
    """`model.layers.12.mlp.up_proj.weight` -> `model.layers.*.mlp.up_proj`: one group per module across layers."""
    return _LAYER_INDEX.sub(".*", name.rsplit(".", 1)[0] if "." in name else name)


def model_breakdown(model) -> dict:
    """The model's own tensors: parameters and buffers, by dtype and by device type (MB), plus the largest module groups.

    A tensor is counted once per storage, so tied weights (embeddings shared with the LM head) are not counted twice.
    """
    seen: set[tuple[str, int]] = set()
    by_dtype: dict[str, float] = defaultdict(float)
    by_device: dict[str, float] = defaultdict(float)
    by_group: dict[str, float] = defaultdict(float)
    by_dtype_group: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    totals = {"parameters_mb": 0.0, "buffers_mb": 0.0}
    named = [("parameters_mb", model.named_parameters(remove_duplicate=False)), ("buffers_mb", model.named_buffers())]
    for kind, tensors in named:
        for name, tensor in tensors:
            if tensor is None or tensor.device.type == "meta":
                continue
            try:
                storage = tensor.untyped_storage()
                key = (str(tensor.device), storage.data_ptr())
                size = storage.nbytes() / MB
            except (RuntimeError, NotImplementedError):  # tensor subclasses without a plain storage
                key = (str(tensor.device), id(tensor))
                size = tensor.numel() * tensor.element_size() / MB
            if key in seen:
                continue
            seen.add(key)
            totals[kind] += size
            by_dtype[str(tensor.dtype).removeprefix("torch.")] += size
            by_device[tensor.device.type] += size
            by_group[module_group(name)] += size
            by_dtype_group[str(tensor.dtype).removeprefix("torch.")][module_group(name)] += size
    top = sorted(by_group.items(), key=lambda kv: -kv[1])[:25]
    main_dtype = max(by_dtype, key=by_dtype.get) if by_dtype else None
    # Where the other dtypes are (an fp32 norm or router in a bf16 model): the largest groups of each
    minor = {
        dtype: dict(sorted(groups.items(), key=lambda kv: -kv[1])[:10])
        for dtype, groups in by_dtype_group.items()
        if dtype != main_dtype
    }
    return {
        **totals,
        "weights_mb": totals["parameters_mb"] + totals["buffers_mb"],
        "by_dtype_mb": dict(sorted(by_dtype.items(), key=lambda kv: -kv[1])),
        "by_device_mb": dict(by_device),
        "top_modules_mb": dict(top),
        "minor_dtype_modules_mb": minor,
    }


def dtypes_string(by_dtype_mb: dict[str, float]) -> str:
    """`bfloat16:2048.0,float32:3.1`: a dtype breakdown as one dataset column."""
    return ",".join(f"{k}:{v:.1f}" for k, v in by_dtype_mb.items())
