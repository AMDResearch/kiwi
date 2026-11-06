# Copyright © Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Data-driven runtime recommendations for the KIWI EP benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


CONFIG_DIR = Path(__file__).with_name("tuning_configs")


def gpu_arch_for(torch: Any, device_id: int) -> str:
    properties = torch.cuda.get_device_properties(device_id)
    return str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]


def nic_family_for(
    hca_name: str, sysfs_root: Path = Path("/sys/class/infiniband")
) -> str:
    if not hca_name:
        return ""
    prefix = hca_name.split("_", 1)[0].lower()
    if prefix == "ionic":
        return "pollara"
    if prefix != "mlx5":
        return prefix
    try:
        vendor = (sysfs_root / hca_name / "device/vendor").read_text().strip().lower()
        device = (sysfs_root / hca_name / "device/device").read_text().strip().lower()
    except OSError:
        return prefix
    if vendor == "0x15b3" and device == "0x1021":
        return "cx7"
    return prefix


def config_path(gpu_arch: str, nic_family: str) -> Path:
    return CONFIG_DIR / f"{gpu_arch}_{nic_family}_runtime.json"


def load_rules(gpu_arch: str, nic_family: str) -> list[dict[str, Any]]:
    path = config_path(gpu_arch, nic_family)
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    if data.get("version") != "1.0":
        return []
    if data.get("gpu_arch") != gpu_arch or data.get("nic_family") != nic_family:
        return []
    required = {"world_size", "num_tokens", "num_blocks", "rcomp_backend"}
    return [rule for rule in data.get("rules", []) if required <= rule.keys()]


def lookup_runtime_defaults(
    rules: list[dict[str, Any]], world_size: int, num_tokens: int
) -> Optional[dict[str, Any]]:
    matches = sorted(
        (rule for rule in rules if int(rule["world_size"]) == world_size),
        key=lambda rule: int(rule["num_tokens"]),
    )
    if not matches:
        return None
    selected = matches[-1]
    for rule in matches:
        if num_tokens <= int(rule["num_tokens"]):
            selected = rule
            break
    return {
        "num_blocks": int(selected["num_blocks"]),
        "rcomp_backend": str(selected["rcomp_backend"]),
    }


def recommended_runtime_defaults(
    torch: Any,
    device_id: int,
    hca_name: str,
    world_size: int,
    num_tokens: int,
) -> Optional[dict[str, Any]]:
    gpu_arch = gpu_arch_for(torch, device_id)
    nic_family = nic_family_for(hca_name)
    result = lookup_runtime_defaults(
        load_rules(gpu_arch, nic_family), world_size, num_tokens
    )
    if result is not None:
        result["source"] = config_path(gpu_arch, nic_family).name
    return result
