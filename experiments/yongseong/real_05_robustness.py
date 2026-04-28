#!/usr/bin/env python3
"""real_05_robustness — Robustness across deployment errors wrapper."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

IEC_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = IEC_ROOT / "experiments/yongseong"
DEFAULT_RESULTS_BASE = EXP_DIR / "results/real_05_robustness"
CALIB = IEC_ROOT / "calibration"
ERROR_DEC = IEC_ROOT / "error_dec/cifar"
REFERENCE_NPZ = IEC_ROOT / "cifar10_reference.npz"
RUN_DIR_RE = re.compile(r"^\d{8}_\d{6}$")
NUM_STEPS = 100
FID_RE = re.compile(r"^FID:\s*([\d.]+)\s*$", re.MULTILINE)
FID_ALT_RE = re.compile(
    r"^(?:Frechet Inception Distance|frechet_inception_distance):\s*([\d.]+)\s*$",
    re.MULTILINE,
)
SFID_RE = re.compile(r"^sFID:\s*([\d.]+)\s*$", re.MULTILINE)

CSV_KEYS = [
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
    "error_strength",
    "per_sample_nfe",
    "nfe_total",
    "sampling_wall_clock_sec",
    "fid_wall_clock_sec",
    "total_wall_clock_sec",
    "trace_path",
    "source_npz",
    "source_log",
    "notes",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["inventory", "pilot", "calibrate", "main", "fid", "plot", "all"], default="inventory")
    p.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--settings", nargs="+", default=None)
    p.add_argument("--pilot-samples", type=int, default=512)
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--sample-batch", type=int, default=500)
    p.add_argument("--percentile", type=float, default=80.0)
    p.add_argument("--siec-max-rounds", type=int, default=2)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_BASE)
    p.add_argument("--reuse-lookahead", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--siec-score-mode", choices=["raw", "mean", "calibrated"], default="raw")
    p.add_argument("--siec-stats-path", type=Path, default=None)
    p.add_argument(
        "--cuda-visible-devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "2"),
        help="GPU visibility for generated commands and subprocesses (default: 2).",
    )
    return p.parse_args()


def current_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def resolve_results_dir(base: Path, plot_only: bool) -> Path:
    if plot_only:
        if (base / "results.csv").exists():
            return base
        candidates = sorted(p for p in base.iterdir() if p.is_dir() and (p / "results.csv").exists())
        if not candidates:
            raise FileNotFoundError(f"no existing run directory found under {base}")
        return candidates[-1]
    return base if RUN_DIR_RE.match(base.name) else base / current_run_id()


def rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(IEC_ROOT))
    except ValueError:
        return str(path)


def sanitize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def conda_python(args) -> list[str]:
    prefix: list[str] = []
    if args.cuda_visible_devices:
        prefix = ["env", f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}"]
    return prefix + ["conda", "run", "--no-capture-output", "-n", "iec", "python"]


def entry_script(name: str) -> str:
    if (EXP_DIR / name).exists():
        return f"experiments/yongseong/{name}"
    return f"mainddpm/{name}"


def png_count(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for _ in folder.glob("*.png"))


def _suffix(args=None) -> str:
    return "_reuse" if getattr(args, "reuse_lookahead", False) else ""


def pilot_scores_path(label: str, args=None) -> Path:
    return CALIB / f"pilot_scores_{label}{_suffix(args)}.pt"


def tau_schedule_path(label: str, percentile: float, args=None) -> Path:
    p = int(round(percentile))
    return CALIB / f"tau_schedule_{label}{_suffix(args)}_p{p}.pt"


def setting_defs() -> dict[str, dict]:
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
            unblock_via=None,
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
            unblock_via=None,
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
            unblock_via=None,
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
            unblock_via="Generate W4A8 DEC assets for interval=10.",
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
            unblock_via="Generate cache20 calibration and DEC assets.",
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
            unblock_via="Generate W4A8 quant DEC assets first, then reuse cache10 DEC assets.",
        ),
    }


def check_assets(defs: dict[str, dict]) -> dict[str, dict]:
    report = {}
    for label, info in defs.items():
        missing = [path for path in info["required_assets"] if not (IEC_ROOT / path).exists()]
        status = "runnable" if not missing else "missing_asset"
        report[label] = {**info, "missing": missing, "status": status}
    return report


def write_inventory(report: dict[str, dict], out_md: Path) -> None:
    lines = [
        "# Experiment 5 — Setting Inventory",
        "",
        "Latest taxonomy: `fp16`, `w8a8`, `dc10`, `w4a8`, `dc20`, `cachequant`.",
        "Each runnable setting keeps four method rows: `No correction`, `TAC`, `IEC`, `S-IEC`.",
        "Only `fp16` is allowed to share one no-op sample result across all four method rows.",
        "",
        "| # | Setting | Status | W | A | Interval | PTQ | Cache reuse | Missing assets | Unblock via |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, info in sorted(report.items(), key=lambda kv: kv[1]["rank"]):
        missing = ", ".join(info["missing"]) if info["missing"] else "—"
        unblock = info["unblock_via"] or "—"
        lines.append(
            f"| {info['rank']} | `{label}` | **{info['status']}** | {info['weight_bit']} | {info['act_bit']} | "
            f"{info['replicate_interval']} | {info['enable_ptq']} | {info['enable_cache_reuse']} | {missing} | {unblock} |"
        )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))


def run_cmd(cmd: list[str], log_path: Path) -> float:
    print(f"$ {shlex.join(cmd)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, cwd=IEC_ROOT, stdout=f, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise SystemExit(f"command failed (rc={proc.returncode}): {shlex.join(cmd)}")
    return elapsed


def parse_fid_log(log_path: Path) -> tuple[float | None, float | None]:
    text = log_path.read_text(errors="ignore")
    m_fid = FID_RE.search(text) or FID_ALT_RE.search(text)
    m_sfid = SFID_RE.search(text)
    return (
        float(m_fid.group(1)) if m_fid else None,
        float(m_sfid.group(1)) if m_sfid else None,
    )


def pngs_to_npz(png_dir: Path, out_npz: Path) -> int:
    import numpy as np
    from PIL import Image

    paths = sorted(png_dir.glob("*.png"))
    arr = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths]).astype("uint8")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, arr_0=arr)
    return arr.shape[0]


def run_fid(args, sample_npz: Path, log_path: Path) -> tuple[float | None, float | None, float]:
    cmd = conda_python(args) + ["evaluator_FID.py", str(REFERENCE_NPZ), str(sample_npz)]
    print(f"$ {shlex.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=IEC_ROOT, capture_output=True, text=True)
    elapsed = time.time() - t0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout + proc.stderr)
    fid, sfid = parse_fid_log(log_path)
    return fid, sfid, elapsed


def selected_labels(args, report: dict[str, dict]) -> list[str]:
    if args.settings:
        return [label for label in args.settings if label in report]
    return [label for label, _ in sorted(report.items(), key=lambda kv: kv[1]["rank"])]


def build_setting_flags(info: dict) -> list[str]:
    flags = [
        "--weight_bit", str(info["weight_bit"]),
        "--act_bit", str(info["act_bit"]),
        "--replicate_interval", str(info["replicate_interval"]),
    ]
    if not info["enable_ptq"]:
        flags.append("--no-ptq")
    if not info["enable_cache_reuse"]:
        flags.append("--disable-cache-reuse")
    return flags


def _siec_common_flags(args) -> list[str]:
    flags = []
    if args.reuse_lookahead:
        flags.append("--reuse_lookahead")
    if args.siec_score_mode != "raw":
        flags += ["--siec_score_mode", args.siec_score_mode]
        if args.siec_stats_path is not None:
            flags += ["--siec_stats_path", rel(args.siec_stats_path)]
    return flags


def build_pilot_cmd(args, label: str, info: dict) -> list[str]:
    return conda_python(args) + [
        entry_script("ddim_cifar_siec.py"),
        "--correction-mode", "siec",
        "--num_samples", str(args.pilot_samples),
        "--sample_batch", str(args.sample_batch),
        "--siec_collect_scores",
        "--siec_scores_out", rel(pilot_scores_path(label, args)),
        "--image_folder", rel(ERROR_DEC / f"image_robust_pilot_{label}_n{args.pilot_samples}"),
        *build_setting_flags(info),
        *_siec_common_flags(args),
    ]


def build_calibrate_cmd(args, label: str) -> list[str]:
    return conda_python(args) + [
        entry_script("calibrate_tau_cifar.py"),
        "--scores_path", rel(pilot_scores_path(label, args)),
        "--percentile", str(int(round(args.percentile))),
        "--out_path", rel(tau_schedule_path(label, args.percentile, args)),
    ]


def build_main_cmd(args, label: str, info: dict, method_key: str, image_folder: Path, trace_path_: Path) -> list[str]:
    cmd = conda_python(args) + [
        entry_script("ddim_cifar_siec.py"),
        "--correction-mode", method_key if method_key in {"none", "iec", "siec", "tac"} else "none",
        "--num_samples", str(args.num_samples),
        "--sample_batch", str(args.sample_batch),
        "--siec_max_rounds", str(args.siec_max_rounds),
        "--image_folder", rel(image_folder),
        "--siec_return_trace",
        "--siec_trace_mode", method_key if method_key in {"none", "iec", "siec", "tac"} else "none",
        "--siec_trace_out", rel(trace_path_),
        *build_setting_flags(info),
    ]
    if method_key == "siec":
        cmd += [
            "--tau_path", rel(tau_schedule_path(label, args.percentile, args)),
            "--tau_percentile", str(int(round(args.percentile))),
            *_siec_common_flags(args),
        ]
    return cmd


def make_row(setting: str, method_key: str, method: str, num_samples: int) -> dict:
    return {
        "setting": setting,
        "method_key": method_key,
        "method": method,
        "status": "ready",
        "blocked_reason": None,
        "tau_percentile": None,
        "num_samples": num_samples,
        "fid": None,
        "sfid": None,
        "trigger_rate": None,
        "syndrome_mean": None,
        "error_strength": None,
        "per_sample_nfe": None,
        "nfe_total": None,
        "sampling_wall_clock_sec": None,
        "fid_wall_clock_sec": None,
        "total_wall_clock_sec": None,
        "trace_path": None,
        "source_npz": None,
        "source_log": None,
        "notes": "",
    }


def attach_runtime_targets(args, row: dict, label: str, method_key: str, info: dict, cmd_sink: list[list[str]]) -> None:
    slug = f"{label}_{method_key}"
    image_folder = ERROR_DEC / f"image_robust_{slug}_n{args.num_samples}"
    trace_path_ = args.results_dir / "traces" / f"{slug}.pt"
    npz_path = args.results_dir / f"samples_{slug}_n{args.num_samples}.npz"
    sampling_log = args.results_dir / "logs" / f"sampling_{slug}.log"
    fid_log = args.results_dir / "logs" / f"fid_{slug}.log"
    row["_slug"] = slug
    row["_image_folder"] = image_folder
    row["_trace"] = trace_path_
    row["_npz"] = npz_path
    row["_sampling_log"] = sampling_log
    row["_fid_log"] = fid_log
    row["_cmd"] = build_main_cmd(args, label, info, method_key, image_folder, trace_path_)
    row["_needs_run"] = not trace_path_.exists() or (not npz_path.exists() and png_count(image_folder) < args.num_samples)
    row["trace_path"] = rel(trace_path_)
    row["source_npz"] = rel(npz_path)
    row["source_log"] = rel(fid_log)
    if row["_needs_run"]:
        cmd_sink.append(row["_cmd"])


def aggregate_trace(trace_path_: Path) -> dict:
    import torch

    traces = torch.load(trace_path_, weights_only=False, map_location="cpu")
    if isinstance(traces, dict):
        traces = [traces]
    total_weight = 0
    total_nfe = 0.0
    total_checked = 0.0
    total_triggered = 0.0
    values: list[float] = []
    for trace in traces:
        batch_size = int(trace.get("batch_size", 1))
        total_weight += batch_size
        total_nfe += batch_size * float(sum(trace.get("nfe_per_step", [])))
        checked = trace.get("checked_per_step", [])
        triggered = trace.get("triggered_per_step", [])
        total_checked += batch_size * sum(1 for flag in checked if flag)
        total_triggered += batch_size * sum(
            1 for check, flag in zip(checked, triggered) if check and flag
        )
        for score_values in trace.get("score_values_per_step", []):
            values.extend(float(v) for v in score_values)
    return {
        "per_sample_nfe": (total_nfe / total_weight) if total_weight else None,
        "trigger_rate": (total_triggered / total_checked) if total_checked else 0.0,
        "syndrome_mean": (sum(values) / len(values)) if values else None,
    }


def compute_error_strength(label: str, args) -> float:
    if label == "fp16":
        return 0.0
    import numpy as np
    import torch

    raw_scores = torch.load(pilot_scores_path(label, args), weights_only=False, map_location="cpu")
    if isinstance(raw_scores, dict):
        scores = raw_scores.get("scores_by_t")
        if scores is None:
            raise RuntimeError(f"pilot payload missing scores_by_t: {pilot_scores_path(label, args)}")
    else:
        scores = raw_scores
    per_t = []
    for arr in scores:
        np_arr = np.asarray(arr)
        if np_arr.size and np.any(np_arr != 0):
            per_t.append(float(np_arr.mean()))
    if not per_t:
        raise RuntimeError(f"no non-zero syndrome bins found in {pilot_scores_path(label, args)}")
    return float(sum(per_t) / len(per_t))


def phase_pilot(args, report: dict[str, dict], cmd_sink: list[list[str]]) -> None:
    for label in selected_labels(args, report):
        if label == "fp16":
            continue
        info = report[label]
        if info["status"] != "runnable":
            continue
        if pilot_scores_path(label, args).exists():
            continue
        cmd = build_pilot_cmd(args, label, info)
        cmd_sink.append(cmd)
        if not args.dry_run:
            run_cmd(cmd, args.results_dir / "logs" / f"pilot_{label}.log")


def phase_calibrate(args, report: dict[str, dict], cmd_sink: list[list[str]]) -> set[str]:
    ready: set[str] = set()
    for label in selected_labels(args, report):
        if label == "fp16":
            continue
        info = report[label]
        if info["status"] != "runnable":
            continue
        tau = tau_schedule_path(label, args.percentile, args)
        if tau.exists():
            ready.add(label)
            continue
        cmd = build_calibrate_cmd(args, label)
        cmd_sink.append(cmd)
        ready.add(label)
        if not args.dry_run:
            run_cmd(cmd, args.results_dir / "logs" / f"calibrate_{label}.log")
    return ready


def blocked_row(setting: str, method_key: str, method: str, reason: str) -> dict:
    row = make_row(setting, method_key, method, num_samples=0)
    row["status"] = "blocked"
    row["blocked_reason"] = reason
    row["num_samples"] = None
    row["notes"] = reason
    return row


def missing_asset_row(setting: str, method_key: str, method: str, reason: str) -> dict:
    row = make_row(setting, method_key, method, num_samples=0)
    row["status"] = "missing_asset"
    row["blocked_reason"] = reason
    row["num_samples"] = None
    row["notes"] = reason
    return row


def phase_main(args, report: dict[str, dict], cmd_sink: list[list[str]], tau_ready: set[str] | None = None) -> list[dict]:
    tau_ready = tau_ready or set()
    rows: list[dict] = []
    for label in selected_labels(args, report):
        info = report[label]
        if info["status"] != "runnable":
            reason = f"missing assets: {', '.join(info['missing'])}"
            rows.extend([
                missing_asset_row(label, "no_correction", "No correction", reason),
                missing_asset_row(label, "tac", "TAC", reason),
                missing_asset_row(label, "iec", "IEC", reason),
                missing_asset_row(label, "siec", "S-IEC", reason),
            ])
            continue

        if label == "fp16":
            shared = make_row(label, "no_correction", "No correction", args.num_samples)
            shared["notes"] = "no-op regime; fp16 shared sample result"
            attach_runtime_targets(args, shared, label, "none", info, cmd_sink)
            rows.append(shared)
            for method_key, method in (("tac", "TAC"), ("iec", "IEC"), ("siec", "S-IEC")):
                row = make_row(label, method_key, method, args.num_samples)
                row["notes"] = "no-op regime; shares fp16 no-correction result"
                row["_share_with"] = shared["_slug"]
                rows.append(row)
            continue

        no_row = make_row(label, "no_correction", "No correction", args.num_samples)
        attach_runtime_targets(args, no_row, label, "none", info, cmd_sink)
        rows.append(no_row)

        tac_row = blocked_row(label, "tac", "TAC", "TAC not implemented")
        rows.append(tac_row)

        iec_row = make_row(label, "iec", "IEC", args.num_samples)
        attach_runtime_targets(args, iec_row, label, "iec", info, cmd_sink)
        rows.append(iec_row)

        if tau_schedule_path(label, args.percentile, args).exists() or label in tau_ready:
            siec_row = make_row(label, "siec", "S-IEC", args.num_samples)
            siec_row["tau_percentile"] = int(round(args.percentile))
            attach_runtime_targets(args, siec_row, label, "siec", info, cmd_sink)
            rows.append(siec_row)
        else:
            rows.append(blocked_row(label, "siec", "S-IEC", f"missing tau schedule: {rel(tau_schedule_path(label, args.percentile, args))}"))

    return rows


def hydrate_row_from_trace(row: dict) -> None:
    trace_path_ = row.get("_trace")
    if trace_path_ is None or not trace_path_.exists():
        return
    metrics = aggregate_trace(trace_path_)
    row["per_sample_nfe"] = metrics["per_sample_nfe"]
    row["trigger_rate"] = metrics["trigger_rate"]
    row["syndrome_mean"] = metrics["syndrome_mean"]
    if row.get("num_samples") and row.get("per_sample_nfe") is not None:
        row["nfe_total"] = row["num_samples"] * row["per_sample_nfe"]


def phase_fid(rows: list[dict], args) -> None:
    for row in rows:
        if row.get("status") in {"blocked", "missing_asset"}:
            continue
        if row.get("_share_with"):
            continue
        if not args.dry_run and row.get("_needs_run"):
            row["sampling_wall_clock_sec"] = run_cmd(row["_cmd"], row["_sampling_log"])
        hydrate_row_from_trace(row)
        npz_path = row.get("_npz")
        image_folder = row.get("_image_folder")
        if npz_path is None or image_folder is None:
            continue
        if not npz_path.exists():
            if png_count(image_folder) == 0:
                row["notes"] = ((row.get("notes") or "") + "; sampling output missing").lstrip("; ")
                continue
            pngs_to_npz(image_folder, npz_path)
        fid_log = row["_fid_log"]
        if fid_log.exists():
            row["fid"], row["sfid"] = parse_fid_log(fid_log)
            row["fid_wall_clock_sec"] = 0.0
        elif not args.dry_run:
            row["fid"], row["sfid"], row["fid_wall_clock_sec"] = run_fid(args, npz_path, fid_log)
        row["total_wall_clock_sec"] = sum(
            v for v in [row.get("sampling_wall_clock_sec"), row.get("fid_wall_clock_sec")] if v is not None
        ) or None
        if row.get("fid") is not None and row.get("per_sample_nfe") is not None:
            row["status"] = "completed"


def apply_fp16_shared_rows(rows: list[dict]) -> None:
    lookup = {row.get("_slug"): row for row in rows if row.get("_slug")}
    for row in rows:
        shared_slug = row.get("_share_with")
        if not shared_slug:
            continue
        source = lookup.get(shared_slug)
        if source is None:
            continue
        for key in (
            "fid", "sfid", "trigger_rate", "syndrome_mean", "error_strength",
            "per_sample_nfe", "nfe_total", "sampling_wall_clock_sec",
            "fid_wall_clock_sec", "total_wall_clock_sec", "trace_path",
            "source_npz", "source_log",
        ):
            row[key] = source.get(key)
        if source.get("status") == "completed":
            row["status"] = "completed"


def fill_post_hoc(rows: list[dict], report: dict[str, dict], args) -> None:
    cache: dict[str, float] = {}
    for row in rows:
        if row.get("status") in {"blocked", "missing_asset"}:
            continue
        label = row["setting"]
        if label not in cache:
            if label == "fp16" or pilot_scores_path(label, args).exists():
                cache[label] = compute_error_strength(label, args)
            else:
                cache[label] = None
        row["error_strength"] = cache[label]
        if row.get("syndrome_mean") is None:
            row["syndrome_mean"] = cache[label]
        if row.get("trigger_rate") is None:
            if row["method_key"] == "no_correction":
                row["trigger_rate"] = 0.0
            elif row["method_key"] == "iec":
                row["trigger_rate"] = 1.0
    apply_fp16_shared_rows(rows)


def save_results(rows: list[dict], results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    clean = [{k: row.get(k) for k in CSV_KEYS} for row in rows]
    with open(results_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_KEYS)
        writer.writeheader()
        writer.writerows(clean)
    with open(results_dir / "results.json", "w") as f:
        json.dump(clean, f, indent=2)


def write_commands(cmds: list[list[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("# Generated by real_05_robustness.py\n")
        f.write(f"cd {IEC_ROOT}\n\n")
        for cmd in cmds:
            f.write(shlex.join(cmd) + "\n")
    path.chmod(0o755)


def completed_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("fid") is not None and row.get("error_strength") is not None]


def plot_two_panel(rows: list[dict], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ready = completed_rows(rows)
    plt.rcParams.update({
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.dpi": 150,
    })

    colors = {
        "no_correction": "#888888",
        "tac": "#28A050",
        "iec": "#005AA0",
        "siec": "#E07814",
    }
    markers = {
        "no_correction": "X",
        "tac": "D",
        "iec": "s",
        "siec": "o",
    }
    labels = {
        "no_correction": "No correction",
        "tac": "TAC",
        "iec": "IEC",
        "siec": "S-IEC",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(alpha=0.3)

    for method_key in ("no_correction", "tac", "iec", "siec"):
        items = sorted([r for r in ready if r["method_key"] == method_key], key=lambda r: r["error_strength"])
        if not items:
            continue
        axes[0].plot(
            [r["error_strength"] for r in items],
            [r["fid"] for r in items],
            "-",
            color=colors[method_key],
            linewidth=2,
            alpha=0.8,
        )
        axes[0].scatter(
            [r["error_strength"] for r in items],
            [r["fid"] for r in items],
            s=65,
            marker=markers[method_key],
            color=colors[method_key],
            edgecolors="k",
            linewidths=0.5,
            label=labels[method_key],
            zorder=4,
        )

    axes[0].set_xlabel("Error strength")
    axes[0].set_ylabel("FID (↓ better)")
    axes[0].set_title("(a) Error Strength vs FID")
    axes[0].legend(fontsize=8, loc="upper left")

    iec_by_setting = {r["setting"]: r for r in ready if r["method_key"] == "iec"}
    tac_by_setting = {r["setting"]: r for r in ready if r["method_key"] == "tac"}
    siec_items = sorted([r for r in ready if r["method_key"] == "siec"], key=lambda r: r["error_strength"])
    if siec_items:
        iec_pairs = [(r["error_strength"], iec_by_setting[r["setting"]]["fid"] - r["fid"], r["setting"]) for r in siec_items if r["setting"] in iec_by_setting]
        xs = [x for x, _, _ in iec_pairs]
        ys = [y for _, y, _ in iec_pairs]
        if xs:
            axes[1].plot(xs, ys, "-", color=colors["siec"], linewidth=2, alpha=0.8, label="FID(IEC) - FID(S-IEC)")
            axes[1].scatter(xs, ys, s=65, marker="o", color=colors["siec"], edgecolors="k", linewidths=0.5, zorder=4)
            for x, y, label in iec_pairs:
                axes[1].annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
        xs_tac = [r["error_strength"] for r in siec_items if r["setting"] in tac_by_setting]
        ys_tac = [tac_by_setting[r["setting"]]["fid"] - r["fid"] for r in siec_items if r["setting"] in tac_by_setting]
        if xs_tac:
            axes[1].plot(xs_tac, ys_tac, "--", color=colors["tac"], linewidth=1.6, alpha=0.8, label="FID(TAC) - FID(S-IEC)")
            axes[1].scatter(xs_tac, ys_tac, s=55, marker="D", color=colors["tac"], edgecolors="k", linewidths=0.5, zorder=4)

    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_xlabel("Error strength")
    axes[1].set_ylabel("FID(IEC) - FID(S-IEC)")
    axes[1].set_title("(b) Relative Gain of S-IEC")
    axes[1].legend(fontsize=8, loc="best")

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def load_csv_rows(path: Path) -> list[dict]:
    with open(path) as f:
        raw = list(csv.DictReader(f))

    def cast(value):
        if value in ("", "None", None):
            return None
        if value == "True":
            return True
        if value == "False":
            return False
        try:
            return float(value) if "." in value else int(value)
        except ValueError:
            return value

    return [{k: cast(v) for k, v in row.items()} for row in raw]


def main() -> None:
    args = parse_args()
    args.results_dir = resolve_results_dir(args.results_dir, plot_only=args.phase == "plot")
    args.results_dir.mkdir(parents=True, exist_ok=True)

    defs = setting_defs()
    report = check_assets(defs)

    if args.phase == "plot":
        rows = load_csv_rows(args.results_dir / "results.csv")
        plot_two_panel(rows, args.results_dir / "robustness_2panel.png")
        print(f"Replotted {len(rows)} rows -> {args.results_dir}")
        return

    write_inventory(report, args.results_dir / "inventory.md")
    if args.phase == "inventory":
        runnable = [k for k, v in report.items() if v["status"] == "runnable"]
        blocked = [k for k, v in report.items() if v["status"] != "runnable"]
        print(f"Inventory written -> {args.results_dir / 'inventory.md'}")
        print(f"  runnable: {runnable}")
        print(f"  blocked:  {blocked}")
        return

    cmd_list: list[list[str]] = []
    rows: list[dict] = []
    tau_ready: set[str] = set()

    if args.phase in ("pilot", "all"):
        phase_pilot(args, report, cmd_list)
    if args.phase in ("calibrate", "all"):
        tau_ready = phase_calibrate(args, report, cmd_list)
    if args.phase in ("main", "fid", "all"):
        rows = phase_main(args, report, cmd_list, tau_ready=tau_ready)
    if args.phase in ("fid", "all"):
        phase_fid(rows, args)
        fill_post_hoc(rows, report, args)

    write_commands(cmd_list, args.results_dir / "commands.sh")
    if rows:
        save_results(rows, args.results_dir)
    if args.phase == "all" and completed_rows(rows):
        plot_two_panel(rows, args.results_dir / "robustness_2panel.png")

    if args.dry_run:
        print(f"Dry-run: {len(cmd_list)} commands -> {args.results_dir / 'commands.sh'}")
        if rows:
            print(f"         rows={len(rows)}")
        runnable = [k for k, v in report.items() if v["status"] == "runnable"]
        print(f"         runnable settings: {runnable}")
        return
    print(f"Done. Results in {args.results_dir}")


if __name__ == "__main__":
    main()
