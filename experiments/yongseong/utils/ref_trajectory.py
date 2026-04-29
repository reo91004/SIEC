from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def load_trace(path: str | Path) -> dict[str, Any]:
    raw = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        return _normalize_single(raw)
    if not isinstance(raw, list) or not raw or not isinstance(raw[0], dict):
        raise ValueError(f"unexpected trace format at {path}: {type(raw).__name__}")
    return _concat_batches(raw)


def _normalize_single(b: dict[str, Any]) -> dict[str, Any]:
    out = dict(b)
    out["num_samples"] = int(b.get("batch_size", 0))
    out["num_batches"] = 1
    if isinstance(out.get("xs_trajectory"), list) and out["xs_trajectory"]:
        out["xs_trajectory"] = torch.stack(out["xs_trajectory"], dim=0)
    if isinstance(out.get("x0_trajectory"), list) and out["x0_trajectory"]:
        out["x0_trajectory"] = torch.stack(out["x0_trajectory"], dim=0)
    return out


def _concat_batches(raw: list[dict[str, Any]]) -> dict[str, Any]:
    keys: set[str] = set()
    for b in raw:
        keys.update(b.keys())
    out: dict[str, Any] = {}
    first = raw[0]
    T = len(first.get("step_idx_per_step", []))
    for k in sorted(keys):
        v0 = first.get(k)
        if k in ("xs_trajectory", "x0_trajectory") and isinstance(v0, list) and v0:
            stacked_per_step = []
            for t in range(len(v0)):
                stacked_per_step.append(torch.cat([b[k][t] for b in raw], dim=0))
            out[k] = torch.stack(stacked_per_step, dim=0)
        elif isinstance(v0, list):
            out[k] = v0
        elif isinstance(v0, torch.Tensor):
            try:
                out[k] = torch.cat([b[k] for b in raw], dim=0)
            except RuntimeError:
                out[k] = v0
        elif isinstance(v0, (int, float, str, bool)):
            out[k] = v0
        else:
            out[k] = v0
    out["num_batches"] = len(raw)
    out["num_samples"] = sum(int(b.get("batch_size", 0)) for b in raw)
    out["T"] = T
    return out


def assert_step_aligned(ref: dict[str, Any], dep: dict[str, Any]) -> None:
    for key in ("step_idx_per_step", "t_int_per_step", "next_t_int_per_step"):
        if list(ref.get(key, [])) != list(dep.get(key, [])):
            raise AssertionError(f"step alignment mismatch on {key}")


def per_step_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """[T,N,C,H,W] pair → per-step per-sample L2 of (a-b) → [T,N]."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    diff = (a - b).reshape(a.shape[0], a.shape[1], -1)
    return diff.norm(dim=-1)


def per_step_l2_summary(ref: dict[str, Any], dep: dict[str, Any], key: str = "xs_trajectory") -> dict[str, torch.Tensor]:
    """Returns dict with per-step mean/std/max of L2(ref - dep) and ratio to ref norm."""
    assert_step_aligned(ref, dep)
    a = ref[key][:, : dep[key].shape[1]]
    b = dep[key]
    diffs = per_step_l2(a, b)              # [T, N]
    ref_norms = a.reshape(a.shape[0], a.shape[1], -1).norm(dim=-1)
    return {
        "diff_mean": diffs.mean(dim=1),
        "diff_std": diffs.std(dim=1),
        "diff_max": diffs.max(dim=1).values,
        "ref_norm_mean": ref_norms.mean(dim=1),
        "ratio_mean": (diffs / (ref_norms + 1e-9)).mean(dim=1),
    }
