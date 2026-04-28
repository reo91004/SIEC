"""Common helpers for the S-IEC = ECC × Diffusion framing experiments (real_06–real_11).

Mirrors the conventions used by `IEC/experiments/real_05_robustness.py` so that
all six framing wrappers share the same:
- conda env / GPU-visibility prefix
- entry_script resolution (mirror copy in experiments/yongseong/ takes priority)
- results directory layout (run_id, traces, logs)
- setting taxonomy (fp16, w8a8, w4a8, dc10, dc20, cachequant)
- FID parsing regex
- trace dict <-> per-sample tensor reshape

This module is intentionally read-only with respect to 1저자 코드: it imports
nothing from `IEC/mainddpm/`, `IEC/siec_core/`, `IEC/quant/`, `IEC/mainldm/`.
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

IEC_ROOT = Path(__file__).resolve().parents[2]  # IEC/  (this file is at IEC/experiments/yongseong/framing_common.py)
EXP_DIR = IEC_ROOT / "experiments/yongseong"
CALIB = IEC_ROOT / "calibration"
ERROR_DEC = IEC_ROOT / "error_dec/cifar"
REFERENCE_NPZ = IEC_ROOT / "cifar10_reference.npz"
RESULTS_BASE = EXP_DIR / "results"

NUM_STEPS = 100
DEFAULT_SEED = 1234 + 9  # matches ddim_cifar_siec.py default

RUN_DIR_RE = re.compile(r"^\d{8}_\d{6}$")
FID_RE = re.compile(r"^FID:\s*([\d.]+)\s*$", re.MULTILINE)
FID_ALT_RE = re.compile(
    r"^(?:Frechet Inception Distance|frechet_inception_distance):\s*([\d.]+)\s*$",
    re.MULTILINE,
)
SFID_RE = re.compile(r"^sFID:\s*([\d.]+)\s*$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Shell / subprocess helpers
# ---------------------------------------------------------------------------

def conda_python(args) -> list[str]:
    """Build `env CUDA_VISIBLE_DEVICES=... conda run --no-capture-output -n iec python` prefix."""
    prefix: list[str] = []
    cuda = getattr(args, "cuda_visible_devices", None) or os.environ.get(
        "CUDA_VISIBLE_DEVICES", "2"
    )
    if cuda:
        prefix = ["env", f"CUDA_VISIBLE_DEVICES={cuda}"]
    return prefix + ["conda", "run", "--no-capture-output", "-n", "iec", "python"]


def entry_script(name: str) -> str:
    """Return the experiment-copy path (preferred) or fall back to mainddpm/."""
    if (EXP_DIR / name).exists():
        return f"experiments/yongseong/{name}"
    return f"mainddpm/{name}"


def rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(IEC_ROOT))
    except ValueError:
        return str(path)


def sanitize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def current_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def resolve_results_dir(base: Path, plot_only: bool = False) -> Path:
    """If `base` already looks like a run dir, return as-is; else create a dated subdir.

    For plot-only invocations, prefer the latest existing run dir under `base`.
    """
    base = Path(base)
    if plot_only:
        if (base / "results.csv").exists():
            return base
        if base.exists():
            candidates = sorted(p for p in base.iterdir() if p.is_dir() and (p / "results.csv").exists())
            if candidates:
                return candidates[-1]
        raise FileNotFoundError(f"no existing run directory found under {base}")
    return base if RUN_DIR_RE.match(base.name) else base / current_run_id()


def png_count(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for _ in folder.glob("*.png"))


def parse_fid_log(log_path: Path) -> tuple[float | None, float | None]:
    if not log_path.exists():
        return None, None
    text = log_path.read_text(errors="ignore")
    m_fid = FID_RE.search(text) or FID_ALT_RE.search(text)
    m_sfid = SFID_RE.search(text)
    return (
        float(m_fid.group(1)) if m_fid else None,
        float(m_sfid.group(1)) if m_sfid else None,
    )


def run_cmd(cmd: list[str], log_path: Path) -> float:
    """Execute `cmd` in IEC_ROOT, redirecting stdout+stderr to `log_path`."""
    import shlex
    print(f"$ {shlex.join(cmd)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, cwd=IEC_ROOT, stdout=f, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise SystemExit(f"command failed (rc={proc.returncode}): {shlex.join(cmd)}")
    return elapsed


def pngs_to_npz(png_dir: Path, out_npz: Path) -> int:
    """Stack every PNG in `png_dir` into a (N, H, W, 3) uint8 npz at `out_npz`.

    Mirrors `real_05_robustness.pngs_to_npz` so wrappers can drive FID end-to-end.
    Returns the number of images written. No-op if `out_npz` already exists.
    """
    import numpy as np
    from PIL import Image

    if out_npz.exists():
        try:
            with np.load(out_npz) as z:
                key = "arr_0" if "arr_0" in z.files else z.files[0]
                return int(z[key].shape[0])
        except Exception:
            pass
    paths = sorted(png_dir.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"no PNGs under {png_dir}")
    arr = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths]).astype("uint8")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, arr_0=arr)
    return arr.shape[0]


def run_fid(args, sample_npz: Path, log_path: Path) -> tuple[float | None, float | None, float]:
    """Invoke `evaluator_FID.py REFERENCE_NPZ sample_npz` and parse the log.

    Returns (fid, sfid, elapsed_sec). Mirrors `real_05_robustness.run_fid`.
    """
    import shlex
    cmd = conda_python(args) + ["evaluator_FID.py", str(REFERENCE_NPZ), str(sample_npz)]
    print(f"$ {shlex.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=IEC_ROOT, capture_output=True, text=True)
    elapsed = time.time() - t0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout + proc.stderr)
    fid, sfid = parse_fid_log(log_path)
    return fid, sfid, elapsed


def write_commands_sh(commands: list[list[str]], path: Path, header: str | None = None) -> None:
    """Emit a runnable shell script. Each command is a token list (already including conda prefix).

    Uses an absolute `cd $IEC_ROOT` (mirroring real_05_robustness) so the script
    works regardless of where the run-dir ends up on disk.
    """
    import shlex
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(IEC_ROOT))}",
    ]
    if header:
        lines.extend(["# " + ln for ln in header.splitlines()])
    lines.append("")
    for cmd in commands:
        lines.append(shlex.join(cmd))
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o755)


# ---------------------------------------------------------------------------
# Setting taxonomy (mirrors real_05_robustness.setting_defs)
# ---------------------------------------------------------------------------

def setting_defs() -> dict[str, dict]:
    """Canonical definitions of the six deployment settings.

    Keep keys/values in sync with `real_05_robustness.setting_defs()`.
    """
    return {
        "fp16": dict(
            rank=1,
            description="fp16 reference (no cache reuse, no PTQ)",
            weight_bit=8,
            act_bit=8,
            replicate_interval=10,
            enable_ptq=False,
            enable_cache_reuse=False,
            required_assets=["calibration/cifar100_cache10_uni.pth"],
        ),
        "w8a8": dict(
            rank=2,
            description="W8A8 quantization only",
            weight_bit=8,
            act_bit=8,
            replicate_interval=10,
            enable_ptq=True,
            enable_cache_reuse=False,
            required_assets=[
                "calibration/cifar100_cache10_uni.pth",
                "error_dec/cifar/pre_quanterr_abCov_weight8_interval10_list_timesteps100.pth",
            ],
        ),
        "dc10": dict(
            rank=3,
            description="DeepCache only (interval=10)",
            weight_bit=8,
            act_bit=8,
            replicate_interval=10,
            enable_ptq=False,
            enable_cache_reuse=True,
            required_assets=[
                "calibration/cifar100_cache10_uni.pth",
                "error_dec/cifar/pre_cacheerr_abCov_interval10_list_timesteps100.pth",
            ],
        ),
        "w4a8": dict(
            rank=4,
            description="W4A8 quantization only",
            weight_bit=4,
            act_bit=8,
            replicate_interval=10,
            enable_ptq=True,
            enable_cache_reuse=False,
            required_assets=[
                "calibration/cifar100_cache10_uni.pth",
                "error_dec/cifar/pre_quanterr_abCov_weight4_interval10_list_timesteps100.pth",
            ],
        ),
        "dc20": dict(
            rank=5,
            description="DeepCache aggressive (interval=20)",
            weight_bit=8,
            act_bit=8,
            replicate_interval=20,
            enable_ptq=False,
            enable_cache_reuse=True,
            required_assets=[
                "calibration/cifar100_cache20_uni.pth",
                "error_dec/cifar/pre_cacheerr_abCov_interval20_list_timesteps100.pth",
            ],
        ),
        "cachequant": dict(
            rank=6,
            description="CacheQuant (W4A8 + DeepCache interval=10)",
            weight_bit=4,
            act_bit=8,
            replicate_interval=10,
            enable_ptq=True,
            enable_cache_reuse=True,
            required_assets=[
                "calibration/cifar100_cache10_uni.pth",
                "error_dec/cifar/pre_quanterr_abCov_weight4_interval10_list_timesteps100.pth",
                "error_dec/cifar/pre_cacheerr_abCov_interval10_list_timesteps100.pth",
            ],
        ),
    }


def check_assets(defs: dict[str, dict]) -> dict[str, dict]:
    out = {}
    for label, info in defs.items():
        missing = [p for p in info["required_assets"] if not (IEC_ROOT / p).exists()]
        out[label] = {**info, "missing": missing, "status": "runnable" if not missing else "missing_asset"}
    return out


def build_setting_flags(info: dict) -> list[str]:
    flags = [
        "--weight_bit", str(info["weight_bit"]),
        "--act_bit", str(info["act_bit"]),
        "--replicate_interval", str(info["replicate_interval"]),
    ]
    if not info["enable_ptq"]:
        flags.append("--no-ptq")
    if not info["enable_cache_reuse"]:
        # fp16 (rank=1) ALSO takes this branch on purpose: emitting --no-cache
        # would route through the plain `generalized_steps` sampler which has no
        # trace API, so use --disable-cache-reuse to keep the cache-aware
        # sampler (and its trace machinery) but force a fresh forward each step.
        flags.append("--disable-cache-reuse")
    return flags


def pilot_scores_path(label: str) -> Path:
    if label == "w8a8" and (CALIB / "pilot_scores_w8a8.pt").exists():
        return CALIB / "pilot_scores_w8a8.pt"
    if label == "dc10" and (CALIB / "pilot_scores_nb.pt").exists() and not (CALIB / "pilot_scores_dc10.pt").exists():
        return CALIB / "pilot_scores_nb.pt"
    p = CALIB / f"pilot_scores_{label}.pt"
    return p


def tau_schedule_path(label: str, percentile: float) -> Path:
    p = int(round(percentile))
    candidate = CALIB / f"tau_schedule_{label}_p{p}.pt"
    if candidate.exists():
        return candidate
    # Fallback: the original W8A8_DC10 baseline used unprefixed names.
    if label in ("dc10", "w8a8") and p == 80 and (CALIB / "tau_schedule_p80.pt").exists():
        return CALIB / "tau_schedule_p80.pt"
    return candidate


# ---------------------------------------------------------------------------
# Trace utilities
# ---------------------------------------------------------------------------

def load_traces(trace_path: Path):
    """Load a list of per-batch trace dicts saved by adaptive_generalized_steps_trace.

    Returns the list as-is. Each dict has keys:
      batch_size, correction_mode, x0_trajectory (list[Tensor] of shape (B,C,H,W) or []),
      xs_trajectory (same), et_per_step (same),
      syndrome_per_step (list[float] length T),
      score_values_per_step (list[list[float]] of length T, each inner length B),
      triggered_per_step, checked_per_step, nfe_per_step.
    """
    import torch
    return torch.load(trace_path, map_location="cpu", weights_only=False)


def stack_score_values(traces: list[dict]) -> "np.ndarray":
    """Stack per-batch score_values_per_step into a single (total_samples, T) array."""
    import numpy as np
    if not traces:
        return np.zeros((0, 0))
    T = len(traces[0]["score_values_per_step"])
    rows = []
    for tr in traces:
        per_step = tr["score_values_per_step"]
        # per_step[t_idx] is list[B] (or [] when not checked)
        b = tr.get("batch_size") or (len(per_step[0]) if per_step and per_step[0] else 0)
        if b == 0:
            continue
        arr = np.zeros((b, T), dtype=np.float64)
        for t_idx, vals in enumerate(per_step):
            if vals:
                arr[: len(vals), t_idx] = np.asarray(vals, dtype=np.float64)
        rows.append(arr)
    return np.concatenate(rows, axis=0) if rows else np.zeros((0, T))


def stack_xs_trajectory(traces: list[dict]) -> "torch.Tensor":
    """Stack xs_trajectory across batches → tensor of shape (total_B, T+1, C, H, W).

    Notes:
    - The sampler stores T+1 entries (x_T..x_0) but xs_trajectory typically saves
      only the per-step xt seen at the *start* of each step (length T).
    - Returns CPU float tensor.
    """
    import torch
    parts = []
    for tr in traces:
        xs = tr.get("xs_trajectory")
        if not xs:
            continue
        # xs[t] is (B, C, H, W); stack along time → (B, T, C, H, W).
        stacked = torch.stack(xs, dim=1)
        parts.append(stacked)
    if not parts:
        return torch.zeros(0)
    return torch.cat(parts, dim=0)


def stack_x0_trajectory(traces: list[dict]) -> "torch.Tensor":
    import torch
    parts = []
    for tr in traces:
        xs = tr.get("x0_trajectory")
        if not xs:
            continue
        stacked = torch.stack(xs, dim=1)
        parts.append(stacked)
    if not parts:
        return torch.zeros(0)
    return torch.cat(parts, dim=0)


def aggregate_nfe(traces: list[dict]) -> dict:
    """Sum NFE/trigger statistics across batches."""
    total_nfe_per_step = None
    total_triggers_per_step = None
    n_batches = 0
    batch_size_total = 0
    for tr in traces:
        nfe = tr.get("nfe_per_step") or []
        trig = tr.get("triggered_per_step") or []
        if not nfe:
            continue
        if total_nfe_per_step is None:
            total_nfe_per_step = list(nfe)
            total_triggers_per_step = [int(bool(x)) for x in trig]
        else:
            for i, v in enumerate(nfe):
                total_nfe_per_step[i] += v
            for i, v in enumerate(trig):
                total_triggers_per_step[i] += int(bool(v))
        n_batches += 1
        batch_size_total += int(tr.get("batch_size", 0))
    return {
        "n_batches": n_batches,
        "batch_size_total": batch_size_total,
        "nfe_per_step_sum": total_nfe_per_step or [],
        "triggers_per_step_sum": total_triggers_per_step or [],
        "nfe_per_step_mean": (
            [v / n_batches for v in total_nfe_per_step]
            if total_nfe_per_step else []
        ),
        "trigger_rate_per_step": (
            [v / n_batches for v in total_triggers_per_step]
            if total_triggers_per_step else []
        ),
        "per_sample_total_nfe_mean": (
            sum(total_nfe_per_step) if total_nfe_per_step else 0
        ) / max(1, n_batches),
        "per_sample_trigger_rate_mean": (
            (sum(total_triggers_per_step) / (len(total_triggers_per_step) * max(1, n_batches)))
            if total_triggers_per_step else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# DDPM β / α schedule (for innovation analysis)
# ---------------------------------------------------------------------------

def get_alpha_bar_schedule(num_diffusion_timesteps: int = 1000) -> "torch.Tensor":
    """Reproduce the linear β schedule used by `mainddpm/ddpm/runners/deepcache.py`.

    β_t = linspace(1e-4, 0.02, T_train), α_t = 1 - β_t, ᾱ_t = ∏ α_s.
    """
    import torch
    betas = torch.linspace(1e-4, 0.02, num_diffusion_timesteps)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return alpha_bar


def step_alpha_bars(num_steps: int, num_diffusion_timesteps: int = 1000) -> "torch.Tensor":
    """ᾱ at the timesteps actually visited by uniform-skip DDIM (length num_steps)."""
    import torch
    skip = num_diffusion_timesteps // num_steps
    seq = torch.arange(0, num_diffusion_timesteps, skip)[:num_steps]
    alpha_bar = get_alpha_bar_schedule(num_diffusion_timesteps)
    return alpha_bar[seq]


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

CORE_CSV_KEYS = [
    "experiment",
    "setting",
    "method_key",
    "method",
    "status",
    "blocked_reason",
    "tau_percentile",
    "num_samples",
    "fid",
    "sfid",
    "trigger_rate",
    "syndrome_mean",
    "per_sample_nfe",
    "nfe_total",
    "sampling_wall_clock_sec",
    "fid_wall_clock_sec",
    "trace_path",
    "source_npz",
    "source_log",
    "notes",
]


def make_row(
    experiment: str,
    setting: str,
    method_key: str,
    method: str,
    num_samples: int,
    extra: dict | None = None,
) -> dict:
    row = {k: None for k in CORE_CSV_KEYS}
    row.update({
        "experiment": experiment,
        "setting": setting,
        "method_key": method_key,
        "method": method,
        "status": "ready",
        "num_samples": num_samples,
        "notes": "",
    })
    if extra:
        row.update(extra)
    return row


def write_csv(rows: Iterable[dict], path: Path, extra_keys: list[str] | None = None) -> None:
    import csv
    keys = list(CORE_CSV_KEYS)
    if extra_keys:
        for k in extra_keys:
            if k not in keys:
                keys.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
