#!/usr/bin/env python3
"""real_04_tradeoff — Compute/Quality tradeoff wrapper for Experiment 4."""
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

IEC_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = IEC_ROOT / "experiments/yongseong"
DEFAULT_RESULTS_BASE = EXP_DIR / "results/real_04_tradeoff"
PILOT_SCORES = IEC_ROOT / "calibration/pilot_scores_nb.pt"
REFERENCE_NPZ = IEC_ROOT / "cifar10_reference.npz"
CIFAR_IMAGE_DIR = IEC_ROOT / "error_dec/cifar"
NUM_STEPS = 100
SETTING_KEY = "exp4_main"
RUN_DIR_RE = re.compile(r"^\d{8}_\d{6}$")
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
    p.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--sample-batch", type=int, default=500)
    p.add_argument("--percentiles", type=int, nargs="+", default=[30, 50, 60, 70, 80, 90, 95])
    p.add_argument("--timesteps", type=int, default=100)
    p.add_argument("--siec-max-rounds", type=int, default=2)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_BASE)
    p.add_argument("--skip-tau-calibration", action="store_true")
    p.add_argument("--skip-sampling", action="store_true")
    p.add_argument("--skip-fid", action="store_true")
    p.add_argument("--plot-only", action="store_true")
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


def tau_path(percentile: int) -> Path:
    return IEC_ROOT / f"calibration/tau_schedule_p{percentile}.pt"


def png_count(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for _ in folder.glob("*.png"))


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


def ensure_tau_schedules(args, cmd_sink: list[list[str]]) -> None:
    for percentile in args.percentiles:
        if tau_path(percentile).exists():
            continue
        cmd = conda_python(args) + [
            entry_script("calibrate_tau_cifar.py"),
            "--scores_path", "./calibration/pilot_scores_nb.pt",
            "--percentile", str(percentile),
            "--out_path", f"./calibration/tau_schedule_p{percentile}.pt",
        ]
        cmd_sink.append(cmd)
        if not args.dry_run:
            run_cmd(cmd, args.results_dir / "logs" / f"calibrate_p{percentile}.log")


def pilot_trigger_rate(percentile: int) -> float | None:
    if not tau_path(percentile).exists() or not PILOT_SCORES.exists():
        return None
    try:
        import numpy as np
        import torch
    except ModuleNotFoundError:
        return None

    raw_scores = torch.load(PILOT_SCORES, weights_only=False, map_location="cpu")
    if isinstance(raw_scores, dict):
        scores_by_t = raw_scores.get("scores_by_t", [])
        batch_means_by_t = raw_scores.get("batch_score_means_by_t", [])
    else:
        scores_by_t = raw_scores
        batch_means_by_t = None
    tau = np.asarray(torch.load(tau_path(percentile), weights_only=False, map_location="cpu")).reshape(-1)
    rates = []
    for step in range(NUM_STEPS):
        arr = np.asarray(scores_by_t[step]) if step < len(scores_by_t) else np.array([])
        if arr.size == 0 or not np.any(arr != 0):
            continue
        tau_t = float(tau[step]) if step < len(tau) else float("inf")
        if batch_means_by_t is not None and step < len(batch_means_by_t):
            batch_arr = np.asarray(batch_means_by_t[step], dtype=float)
            if batch_arr.size == 0:
                continue
            rates.append(float((batch_arr > tau_t).mean()))
        else:
            rates.append(float((arr > tau_t).mean()))
    if not rates:
        return None
    return float(sum(rates) / len(rates))


def make_base_row(method_key: str, method: str, num_samples: int, notes: str = "") -> dict:
    return {
        "setting": SETTING_KEY,
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
        "notes": notes,
    }


def build_sampling_cmd(
    args,
    correction_mode: str,
    image_folder: Path,
    trace_path_: Path,
    tau_percentile: int | None = None,
    always_correct: bool = False,
    trigger_mode: str | None = None,
    trigger_prob: float | None = None,
    trigger_period: int | None = None,
) -> list[str]:
    cmd = conda_python(args) + [
        entry_script("ddim_cifar_siec.py"),
        "--correction-mode", correction_mode,
        "--num_samples", str(args.num_samples),
        "--sample_batch", str(args.sample_batch),
        "--weight_bit", "8",
        "--act_bit", "8",
        "--replicate_interval", "10",
        "--image_folder", rel(image_folder),
        "--siec_max_rounds", str(args.siec_max_rounds),
        "--siec_return_trace",
        "--siec_trace_mode", correction_mode,
        "--siec_trace_out", rel(trace_path_),
    ]
    if tau_percentile is not None:
        cmd += [
            "--tau_path", rel(tau_path(tau_percentile)),
            "--tau_percentile", str(tau_percentile),
        ]
    if always_correct:
        cmd.append("--siec_always_correct")
    if trigger_mode is not None:
        cmd += ["--trigger_mode", trigger_mode]
    if trigger_prob is not None:
        cmd += ["--trigger_prob", f"{trigger_prob:.8f}"]
    if trigger_period is not None:
        cmd += ["--trigger_period", str(trigger_period)]
    return cmd


def attach_runtime_targets(args, row: dict, slug: str, correction_mode: str, **cmd_kwargs) -> None:
    image_folder = CIFAR_IMAGE_DIR / f"image_tradeoff_{slug}_n{args.num_samples}"
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
    row["trace_path"] = rel(trace_path_)
    row["source_npz"] = rel(npz_path)
    row["source_log"] = rel(fid_log)
    row["_cmd"] = build_sampling_cmd(args, correction_mode, image_folder, trace_path_, **cmd_kwargs)
    row["_needs_run"] = not trace_path_.exists() or (not npz_path.exists() and png_count(image_folder) < args.num_samples)


def build_rows(args, cmd_sink: list[list[str]]) -> list[dict]:
    rows: list[dict] = []

    row = make_base_row("no_correction", "No correction", args.num_samples, notes="W8A8 + DeepCache deployed path; correction disabled")
    attach_runtime_targets(args, row, "no_correction", "none")
    if row["_needs_run"]:
        cmd_sink.append(row["_cmd"])
    rows.append(row)

    row = make_base_row("tac", "TAC", args.num_samples, notes="TAC skeleton only")
    row["status"] = "blocked"
    row["blocked_reason"] = "TAC not implemented"
    rows.append(row)

    row = make_base_row("iec", "IEC", args.num_samples, notes="fresh 2K IEC baseline")
    attach_runtime_targets(args, row, "iec", "iec")
    if row["_needs_run"]:
        cmd_sink.append(row["_cmd"])
    rows.append(row)

    row = make_base_row("always_on", "Naive always-on refinement", args.num_samples, notes="S-IEC always-correct ablation")
    attach_runtime_targets(args, row, "always_on", "siec", always_correct=True)
    if row["_needs_run"]:
        cmd_sink.append(row["_cmd"])
    rows.append(row)

    for percentile in args.percentiles:
        row = make_base_row("siec", f"S-IEC p{percentile}", args.num_samples, notes="tau sweep")
        row["tau_percentile"] = percentile
        attach_runtime_targets(args, row, f"siec_p{percentile}", "siec", tau_percentile=percentile)
        if row["_needs_run"]:
            cmd_sink.append(row["_cmd"])
        rows.append(row)

    match_percentile = 80
    target_rate = pilot_trigger_rate(match_percentile)
    if target_rate is None:
        random_row = make_base_row("random", f"Random trigger (matched to p{match_percentile})", args.num_samples)
        random_row["status"] = "blocked"
        random_row["blocked_reason"] = f"tau schedule missing for p{match_percentile}"
        rows.append(random_row)

        uniform_row = make_base_row("uniform", f"Uniform periodic (matched to p{match_percentile})", args.num_samples)
        uniform_row["status"] = "blocked"
        uniform_row["blocked_reason"] = f"tau schedule missing for p{match_percentile}"
        rows.append(uniform_row)
    else:
        random_row = make_base_row(
            "random",
            f"Random trigger (matched to p{match_percentile})",
            args.num_samples,
            notes=f"expected trigger rate from pilot/tau = {target_rate:.4f}",
        )
        attach_runtime_targets(
            args,
            random_row,
            f"random_matched_p{match_percentile}",
            "siec",
            trigger_mode="random",
            trigger_prob=target_rate,
            trigger_period=5,
        )
        if random_row["_needs_run"]:
            cmd_sink.append(random_row["_cmd"])
        rows.append(random_row)

        period = max(1, int(round(1.0 / max(target_rate, 1e-8))))
        realized = 1.0 / period
        uniform_row = make_base_row(
            "uniform",
            f"Uniform periodic (matched to p{match_percentile})",
            args.num_samples,
            notes=f"target rate={target_rate:.4f}, realized periodic rate={realized:.4f}",
        )
        attach_runtime_targets(
            args,
            uniform_row,
            f"uniform_matched_p{match_percentile}_period{period}",
            "siec",
            trigger_mode="uniform",
            trigger_prob=0.0,
            trigger_period=period,
        )
        if uniform_row["_needs_run"]:
            cmd_sink.append(uniform_row["_cmd"])
        rows.append(uniform_row)

    return rows


def aggregate_trace(trace_path_: Path) -> dict:
    import torch

    traces = torch.load(trace_path_, weights_only=False, map_location="cpu")
    if isinstance(traces, dict):
        traces = [traces]
    total_weight = 0
    total_nfe = 0.0
    total_triggered = 0.0
    total_checked = 0.0
    syndrome_values: list[float] = []
    step_score_values: list[list[float]] = []
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
        for step_idx, values in enumerate(trace.get("score_values_per_step", [])):
            while len(step_score_values) <= step_idx:
                step_score_values.append([])
            step_values = [float(v) for v in values]
            step_score_values[step_idx].extend(step_values)
            syndrome_values.extend(float(v) for v in values)
    trigger_rate = (total_triggered / total_checked) if total_checked else 0.0
    syndrome_mean = (sum(syndrome_values) / len(syndrome_values)) if syndrome_values else None
    return {
        "per_sample_nfe": (total_nfe / total_weight) if total_weight else None,
        "trigger_rate": trigger_rate,
        "syndrome_mean": syndrome_mean,
        "syndrome_values": syndrome_values,
        "step_score_values": step_score_values,
    }


def write_syndrome_artifacts(values: list[float], step_values: list[list[float]], results_dir: Path, slug: str) -> None:
    if not values and not any(step_values):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    artifacts = results_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    summary = {
        "count": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }
    (artifacts / f"syndrome_summary_{slug}.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(values, bins=min(40, max(10, len(values) // 20)), color="tab:blue", alpha=0.85)
    ax.set_xlabel("Syndrome score")
    ax.set_ylabel("Count")
    ax.set_title(f"Syndrome Distribution: {slug}")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(artifacts / f"syndrome_hist_{slug}.png", dpi=150)
    plt.close(fig)

    if not any(step_values):
        return

    step_summary = []
    for step_idx, vals in enumerate(step_values):
        arr = np.asarray(vals, dtype=float)
        if arr.size == 0:
            step_summary.append(
                {
                    "step": step_idx,
                    "count": 0,
                    "mean": None,
                    "median": None,
                    "p10": None,
                    "p90": None,
                    "min": None,
                    "max": None,
                }
            )
            continue
        step_summary.append(
            {
                "step": step_idx,
                "count": int(arr.size),
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "p10": float(np.percentile(arr, 10)),
                "p90": float(np.percentile(arr, 90)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        )
    (artifacts / f"syndrome_timestep_summary_{slug}.json").write_text(json.dumps(step_summary, indent=2))

    xs = [item["step"] for item in step_summary if item["count"] > 0]
    means = [item["mean"] for item in step_summary if item["count"] > 0]
    medians = [item["median"] for item in step_summary if item["count"] > 0]
    p10s = [item["p10"] for item in step_summary if item["count"] > 0]
    p90s = [item["p90"] for item in step_summary if item["count"] > 0]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.fill_between(xs, p10s, p90s, color="#9ecae1", alpha=0.35, label="p10-p90")
    ax.plot(xs, means, color="#1f77b4", linewidth=2.0, label="mean")
    ax.plot(xs, medians, color="#d62728", linewidth=1.6, linestyle="--", label="median")
    ax.set_xlabel("Reverse timestep index")
    ax.set_ylabel("Syndrome score")
    ax.set_title(f"Per-step Syndrome Distribution: {slug}")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(artifacts / f"syndrome_timestep_{slug}.png", dpi=150)
    fig.savefig(artifacts / f"syndrome_timestep_{slug}.pdf")
    plt.close(fig)


def update_row_from_trace(row: dict, results_dir: Path) -> None:
    trace_path_ = row.get("_trace")
    if trace_path_ is None or not trace_path_.exists():
        return
    metrics = aggregate_trace(trace_path_)
    row["per_sample_nfe"] = metrics["per_sample_nfe"]
    row["trigger_rate"] = metrics["trigger_rate"]
    row["syndrome_mean"] = metrics["syndrome_mean"]
    if row.get("num_samples") and row.get("per_sample_nfe") is not None:
        row["nfe_total"] = row["num_samples"] * row["per_sample_nfe"]
    write_syndrome_artifacts(metrics["syndrome_values"], metrics["step_score_values"], results_dir, row["_slug"])


def attach_loaded_trace_metadata(rows: list[dict]) -> None:
    for row in rows:
        trace_rel = row.get("trace_path")
        if not trace_rel:
            continue
        trace_abs = IEC_ROOT / str(trace_rel)
        row["_trace"] = trace_abs
        row["_slug"] = trace_abs.stem


def run_rows(rows: list[dict], args) -> None:
    for row in rows:
        if row.get("status") == "blocked" or row.get("_cmd") is None:
            continue
        if not args.skip_sampling and row.get("_needs_run"):
            row["sampling_wall_clock_sec"] = run_cmd(row["_cmd"], row["_sampling_log"])
        update_row_from_trace(row, args.results_dir)
        if args.skip_fid:
            if row.get("sampling_wall_clock_sec") is not None:
                row["total_wall_clock_sec"] = row["sampling_wall_clock_sec"]
            continue
        npz_path = row["_npz"]
        image_folder = row["_image_folder"]
        if not npz_path.exists():
            if png_count(image_folder) == 0:
                row["notes"] = ((row.get("notes") or "") + "; sampling output missing").lstrip("; ")
                continue
            pngs_to_npz(image_folder, npz_path)
        fid_log = row["_fid_log"]
        if fid_log.exists():
            fid, sfid = parse_fid_log(fid_log)
            row["fid"] = fid
            row["sfid"] = sfid
            row["fid_wall_clock_sec"] = 0.0
        else:
            fid, sfid, fid_elapsed = run_fid(args, npz_path, fid_log)
            row["fid"] = fid
            row["sfid"] = sfid
            row["fid_wall_clock_sec"] = fid_elapsed
        row["total_wall_clock_sec"] = sum(
            v for v in [row.get("sampling_wall_clock_sec"), row.get("fid_wall_clock_sec")] if v is not None
        ) or None
        if row.get("fid") is not None and row.get("per_sample_nfe") is not None:
            row["status"] = "completed"


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
        f.write("# Generated by real_04_tradeoff.py\n")
        f.write(f"cd {IEC_ROOT}\n\n")
        for cmd in cmds:
            f.write(shlex.join(cmd) + "\n")
    path.chmod(0o755)


def completed_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("fid") is not None and row.get("per_sample_nfe") is not None]


def plot_two_panel(rows: list[dict], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    ready = completed_rows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.patch.set_facecolor("white")

    for ax in axes:
        ax.set_facecolor("#fcfcfc")
        ax.grid(alpha=0.22, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    siec_rows = sorted(
        [r for r in ready if r["method_key"] == "siec" and r.get("tau_percentile") is not None],
        key=lambda r: r["tau_percentile"],
    )
    if siec_rows:
        xs = [r["per_sample_nfe"] for r in siec_rows]
        ys = [r["fid"] for r in siec_rows]
        axes[0].plot(
            xs, ys, "o-", color="#1f77b4", linewidth=2.2, markersize=7,
            markerfacecolor="white", markeredgewidth=1.8, label="S-IEC tau sweep"
        )
        axes[1].plot(
            [r["trigger_rate"] for r in siec_rows], ys, "o-", color="#1f77b4",
            linewidth=2.2, markersize=7, markerfacecolor="white",
            markeredgewidth=1.8, label="S-IEC tau sweep"
        )
        for row in siec_rows:
            axes[0].annotate(
                f"p{row['tau_percentile']}",
                (row["per_sample_nfe"], row["fid"]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
                color="#1f77b4",
            )
            axes[1].annotate(
                f"p{row['tau_percentile']}",
                (row["trigger_rate"], row["fid"]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
                color="#1f77b4",
            )

    baseline_specs = {
        "no_correction": ("D", "#7f7f7f", "No correction"),
        "iec": ("s", "#ff7f0e", "IEC"),
        "always_on": ("^", "#d62728", "Naive always-on refinement"),
    }
    drawn = set()
    baseline_lookup = {}
    for row in ready:
        spec = baseline_specs.get(row["method_key"])
        if spec is None:
            continue
        marker, color, label = spec
        legend_label = label if label not in drawn else None
        drawn.add(label)
        axes[0].scatter(
            [row["per_sample_nfe"]], [row["fid"]],
            marker=marker, s=110, color=color, edgecolors="white",
            linewidths=0.9, zorder=4, label=legend_label
        )
        axes[1].scatter(
            [row["trigger_rate"]], [row["fid"]],
            marker=marker, s=110, color=color, edgecolors="white",
            linewidths=0.9, zorder=4, label=legend_label
        )
        baseline_lookup[row["method_key"]] = row

    matched_styles = (
        ("random", "#2ca02c", "X", "Random trigger"),
        ("uniform", "#9467bd", "P", "Uniform periodic"),
    )
    for method_key, color, marker, label in matched_styles:
        items = [r for r in ready if r["method_key"] == method_key]
        if not items:
            continue
        items_by_nfe = sorted(items, key=lambda r: r["per_sample_nfe"])
        items_by_trigger = sorted(items, key=lambda r: r["trigger_rate"])
        axes[0].plot(
            [r["per_sample_nfe"] for r in items_by_nfe],
            [r["fid"] for r in items_by_nfe],
            linestyle="--", linewidth=1.4, alpha=0.65, color=color,
        )
        axes[0].scatter(
            [r["per_sample_nfe"] for r in items],
            [r["fid"] for r in items],
            color=color, marker=marker, s=70, alpha=0.9, label=label
        )
        axes[1].plot(
            [r["trigger_rate"] for r in items_by_trigger],
            [r["fid"] for r in items_by_trigger],
            linestyle="--", linewidth=1.4, alpha=0.65, color=color,
        )
        axes[1].scatter(
            [r["trigger_rate"] for r in items],
            [r["fid"] for r in items],
            color=color, marker=marker, s=70, alpha=0.9, label=label
        )

    iec_row = baseline_lookup.get("iec")
    if iec_row is not None:
        for ax in axes:
            ax.axhline(iec_row["fid"], color="#ff7f0e", linestyle=":", linewidth=1.3, alpha=0.8)
        axes[0].axvline(iec_row["per_sample_nfe"], color="#ff7f0e", linestyle=":", linewidth=1.3, alpha=0.8)
        axes[1].axvline(iec_row["trigger_rate"], color="#ff7f0e", linestyle=":", linewidth=1.3, alpha=0.8)
        axes[0].annotate(
            "IEC reference",
            (iec_row["per_sample_nfe"], iec_row["fid"]),
            textcoords="offset points",
            xytext=(8, -14),
            fontsize=9,
            color="#b35a00",
        )
        axes[1].annotate(
            "IEC reference",
            (iec_row["trigger_rate"], iec_row["fid"]),
            textcoords="offset points",
            xytext=(8, -14),
            fontsize=9,
            color="#b35a00",
        )

    if ready:
        fids = [r["fid"] for r in ready if r.get("fid") is not None]
        if fids:
            y_min, y_max = min(fids), max(fids)
            pad = max(0.15, 0.12 * (y_max - y_min))
            for ax in axes:
                ax.set_ylim(y_min - pad, y_max + pad)

    axes[0].set_xlabel("Per-sample NFE")
    axes[0].set_ylabel("FID")
    axes[0].set_title("Compute vs Quality", fontsize=13, weight="bold")
    axes[0].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

    axes[1].set_xlabel("Trigger rate")
    axes[1].set_ylabel("FID")
    axes[1].set_title("Selectivity vs Quality", fontsize=13, weight="bold")
    axes[1].xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axes[1].set_xlim(-0.03, 1.03)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    fig.suptitle("Experiment 4: Tradeoff Under W8A8 + DeepCache", fontsize=15, weight="bold", y=1.06)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def write_compute_matched(rows: list[dict], out_md: Path) -> None:
    ready = completed_rows(rows)
    iec = next(
        (
            row
            for row in ready
            if row["method_key"] == "iec" and row.get("num_samples") is not None
        ),
        None,
    )
    if iec is None:
        out_md.write_text("# Compute-Matched Row\n\nIEC row is not completed yet.\n")
        return
    candidates = [
        row for row in ready
        if row["method_key"] == "siec" and row.get("num_samples") == iec.get("num_samples")
    ]
    if not candidates:
        out_md.write_text("# Compute-Matched Row\n\nS-IEC sweep rows are not completed yet.\n")
        return
    matched = min(candidates, key=lambda row: abs(row["per_sample_nfe"] - iec["per_sample_nfe"]))

    def fmt(value, places=4):
        return "—" if value is None else f"{value:.{places}f}"

    lines = [
        "# Compute-Matched Row (real_04_tradeoff)",
        "",
        f"reference_method = `{iec['method_key']}`",
        f"num_samples = {iec['num_samples']}",
        "",
        "| Method | FID | sFID | per_sample_NFE | trigger_rate |",
        "|---|---|---|---|---|",
        f"| {iec['method']} | {fmt(iec['fid'])} | {fmt(iec['sfid'])} | {fmt(iec['per_sample_nfe'], 2)} | {fmt(iec['trigger_rate'])} |",
        f"| {matched['method']} | {fmt(matched['fid'])} | {fmt(matched['sfid'])} | {fmt(matched['per_sample_nfe'], 2)} | {fmt(matched['trigger_rate'])} |",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))


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
    args.results_dir = resolve_results_dir(args.results_dir, plot_only=args.plot_only)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        rows = load_csv_rows(args.results_dir / "results.csv")
        attach_loaded_trace_metadata(rows)
        for row in rows:
            update_row_from_trace(row, args.results_dir)
        plot_two_panel(rows, args.results_dir / "tradeoff_2panel.png")
        write_compute_matched(rows, args.results_dir / "compute_matched.md")
        print(f"Replotted {len(rows)} rows -> {args.results_dir}")
        return

    cmd_list: list[list[str]] = []
    if not args.skip_tau_calibration:
        ensure_tau_schedules(args, cmd_list)
    rows = build_rows(args, cmd_list)
    write_commands(cmd_list, args.results_dir / "commands.sh")

    if not args.dry_run:
        run_rows(rows, args)
        plot_two_panel(rows, args.results_dir / "tradeoff_2panel.png")
        write_compute_matched(rows, args.results_dir / "compute_matched.md")

    save_results(rows, args.results_dir)
    if args.dry_run:
        print(f"Dry-run: {len(cmd_list)} commands -> {args.results_dir / 'commands.sh'}")
        print(f"         rows={len(rows)}")
    else:
        print(f"Done. Results in {args.results_dir}")


if __name__ == "__main__":
    main()
