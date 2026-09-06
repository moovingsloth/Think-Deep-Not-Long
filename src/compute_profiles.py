"""Compute-environment profiles used by benchmark and cluster launchers."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass


@dataclass(frozen=True)
class ComputeProfile:
    name: str
    prefix_batch_size: int
    continuation_batch_size: int
    torch_backend: str
    dataset_root: str


PROFILES = {
    "local-rtx3090": ComputeProfile(
        "local-rtx3090", 8, 1, "cu121", "/home/dongwon/mnt/seraph-datasets"
    ),
    "seraph-rtx3090": ComputeProfile(
        "seraph-rtx3090", 8, 1, "cu121", "/data/dlehddnjs245/datasets"
    ),
    "dgx-spark": ComputeProfile(
        "dgx-spark", 48, 4, "cu130", "/home/dongwon/mnt/seraph-datasets"
    ),
}


def detect_compute_profile(
    gpu_name: str | None = None,
    machine: str | None = None,
    environ: dict[str, str] | None = None,
) -> str:
    """Detect a supported profile, refusing ambiguous/unknown accelerators."""
    env = os.environ if environ is None else environ
    arch = platform.machine() if machine is None else machine
    if gpu_name is None:
        try:
            import torch

            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        except (ImportError, RuntimeError):
            gpu_name = ""
    normalized = gpu_name.lower()
    if arch == "aarch64" and ("gb10" in normalized or "spark" in normalized):
        return "dgx-spark"
    if "3090" in normalized:
        return "seraph-rtx3090" if env.get("SLURM_JOB_ID") else "local-rtx3090"
    raise RuntimeError(
        f"Cannot auto-detect compute profile for GPU {gpu_name!r} on {arch}; "
        "pass --compute-profile explicitly"
    )


def resolve_compute_profile(name: str) -> ComputeProfile:
    return PROFILES[detect_compute_profile() if name == "auto" else name]
