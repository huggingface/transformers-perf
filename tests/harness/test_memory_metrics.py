import sys
from unittest.mock import Mock

import torch

from tperf.common.record import COLUMNS, MEMORY_COLUMNS, flat_startup, null_column_type
from tperf.harness.framework.memory_metrics import (
    device_memory,
    dtypes_string,
    host_memory,
    model_breakdown,
    module_group,
)

MB = 1024**2


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(256, 64, dtype=torch.bfloat16)  # 32 KiB
        self.layers = torch.nn.ModuleList(torch.nn.Linear(64, 64, bias=False, dtype=torch.bfloat16) for _ in range(2))
        self.head = torch.nn.Linear(64, 256, bias=False, dtype=torch.bfloat16)
        self.head.weight = self.embed.weight  # tied: counted once
        self.register_buffer("inv_freq", torch.ones(32, dtype=torch.float32))


def test_module_group_merges_layers():
    assert module_group("model.layers.12.mlp.up_proj.weight") == "model.layers.*.mlp.up_proj"
    assert module_group("lm_head.weight") == "lm_head"
    assert module_group("blocks.3.0.weight") == "blocks.*.*"


def test_model_breakdown_counts_tied_weights_once():
    b = model_breakdown(Tiny())
    assert b["parameters_mb"] * MB == 256 * 64 * 2 + 2 * 64 * 64 * 2
    assert b["buffers_mb"] * MB == 32 * 4
    assert set(b["by_dtype_mb"]) == {"bfloat16", "float32"}
    assert b["top_modules_mb"]["layers.*"] * MB == 2 * 64 * 64 * 2
    assert "head" not in b["top_modules_mb"]  # the tied weight was counted under its first name
    assert dtypes_string({"bfloat16": 2048.0, "float32": 3.14}) == "bfloat16:2048.0,float32:3.1"


def test_host_memory_is_absent_not_wrong_off_linux():
    mem = host_memory()
    if sys.platform.startswith("linux"):
        assert mem["rss_mb"] > 0 and mem["rss_peak_mb"] >= mem["rss_mb"] - 1
    else:
        assert mem == {}


def test_device_memory_tolerates_missing_stats():
    assert device_memory(object()) == {}
    assert device_memory(Mock()) == {}  # a Mock's memory_stats() is not a dict
    stats = {"allocated_bytes.all.current": 3 * MB, "num_alloc_retries": 2}
    assert device_memory(Mock(memory_stats=Mock(return_value=stats)))["allocated_mb"] == 3
    assert device_memory(Mock(memory_stats=Mock(return_value=stats)))["alloc_retries"] == 2


def test_memory_columns_in_rows_and_schema():
    row = flat_startup({"load_time": 1.0, "weights_mb": 10.0, "measurement_alloc_retries": 0, "weights_dtypes": "x"})
    assert row["startup_weights_mb"] == 10.0 and row["startup_measurement_alloc_retries"] == 0
    assert row["startup_process_gpu_mb"] is None
    for column in MEMORY_COLUMNS:
        assert f"startup_{column}" in COLUMNS
    import pyarrow as pa

    assert null_column_type("startup_measurement_alloc_retries") == pa.int64()
    assert null_column_type("startup_weights_dtypes") == pa.string()
    assert null_column_type("startup_weights_mb") == pa.float64()
