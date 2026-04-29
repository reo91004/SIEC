#!/usr/bin/env python3
"""Phase 2 sanity check for clean trajectory drift stats and score modes.

Run from S-IEC repo root after calibration is complete:
    python experiments/yongseong/verify_phase2.py \\
        --stats-path calibration/syndrome_stats_clean_sanity_n16.pt --run

Verifies:
  W1. Stats schema: version=1, kind="clean_trajectory_drift_stats",
      mu/var/std shape (T,C,H,W), count >= 1, eps stored, config attached.
  W2. End-to-end runs: sampler succeeds for score_mode in {raw, mean, calibrated}.
  W3. score_mode=mean reduces per-step score mean vs score_mode=raw on the same
      seed (deterministic comparison; not a statistical claim with small N).
  W4. score_mode=calibrated produces finite, non-degenerate scores.

Note: with small calibration N (e.g. 16) the magnitudes are noisy. This script
records the comparison for diagnostic purposes; only schema/non-degeneracy is
asserted as PASS/FAIL.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

IEC_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = IEC_ROOT / "experiments/yongseong/results/verify_phase2"
TRACE_DIR = RESULTS_DIR / "traces"
LOG_DIR = RESULTS_DIR / "logs"
IMG_DIR = RESULTS_DIR / "images"


def conda_python(cuda_visible: str | None) -> list[str]:
    prefix: list[str] = []
    if cuda_visible:
        prefix = ["env", f"CUDA_VISIBLE_DEVICES={cuda_visible}"]
    return prefix + ["conda", "run", "--no-capture-output", "-n", "iec", "python"]


def build_cmd(slug: str, score_mode: str, stats_path: Path | None,
              cuda_visible: str | None, num_samples: int, seed: int) -> tuple[list[str], Path]:
    trace_path = TRACE_DIR / f"{slug}.pt"
    img_folder = IMG_DIR / slug
    cmd = conda_python(cuda_visible) + [
        "mainddpm/ddim_cifar_siec.py",
        "--correction-mode", "siec",
        "--num_samples", str(num_samples),
        "--sample_batch", str(num_samples),
        "--seed", str(seed),
        "--weight_bit", "8",
        "--act_bit", "8",
        "--replicate_interval", "10",
        "--image_folder", str(img_folder),
        "--siec_max_rounds", "1",
        "--siec_collect_scores",
        "--siec_scores_out", str(RESULTS_DIR / f"pilot_{slug}.pt"),
        "--siec_return_trace",
        "--siec_trace_mode", "siec",
        "--siec_trace_out", str(trace_path),
        "--reuse_lookahead",
        "--siec_score_mode", score_mode,
    ]
    if stats_path is not None and score_mode != "raw":
        cmd += ["--siec_stats_path", str(stats_path)]
    return cmd, trace_path


def run_cmd(cmd: list[str], log_path: Path) -> int:
    print(f"$ {shlex.join(cmd)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, cwd=IEC_ROOT, stdout=f, stderr=subprocess.STDOUT)
    print(f"  -> rc={proc.returncode}  elapsed={time.time()-t0:.1f}s  log={log_path}")
    return proc.returncode


def fmt_pass(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def verify_schema(stats_path: Path, expected_T: int = 100,
                  expected_C: int = 3, expected_H: int = 32) -> int:
    print("\n=== W1. Stats schema ===")
    if not stats_path.exists():
        print(f"  FAIL stats path missing: {stats_path}")
        return 1
    import torch
    payload = torch.load(stats_path, map_location="cpu", weights_only=False)
    fails = 0

    def check(name, cond, detail=""):
        nonlocal fails
        if cond:
            print(f"  PASS {name}")
        else:
            print(f"  FAIL {name}  {detail}")
            fails += 1

    check("version == 1", payload.get("version") == 1, f"got {payload.get('version')}")
    check("kind == clean_trajectory_drift_stats",
          payload.get("kind") == "clean_trajectory_drift_stats",
          f"got {payload.get('kind')!r}")
    check("score_space == x0", payload.get("score_space") == "x0")

    expected_shape = (expected_T, expected_C, expected_H, expected_H)
    for k in ("mu", "var", "std"):
        v = payload.get(k)
        ok = (v is not None and tuple(v.shape) == expected_shape)
        check(f"{k} shape == {expected_shape}", ok, f"got {tuple(v.shape) if v is not None else None}")

    count = payload.get("count")
    if count is not None:
        non_final_min = int(count[:-1].min().item())
        last_count = int(count[-1].item())
        ok_count = non_final_min >= 1
        check("count >= 1 for all non-final steps", ok_count,
              f"min(steps 0..T-2)={non_final_min}; count[T-1]={last_count} (0 expected: no lookahead at next_t<0)")
    else:
        check("count present", False, "count tensor missing")

    check("eps stored", payload.get("eps") is not None)
    check("config dict present", isinstance(payload.get("config"), dict))

    var = payload.get("var")
    if var is not None:
        ok = bool(torch.all(var > 0).item())
        check("var > 0 everywhere", ok, f"min_var={float(var.min()):.4e}")
    return fails


def aggregate_step_score_mean(trace_path: Path) -> list[float]:
    import torch
    obj = torch.load(trace_path, map_location="cpu", weights_only=False)
    traces = obj if isinstance(obj, list) else [obj]
    means_per_step: list[list[float]] = []
    for trace in traces:
        for i, v in enumerate(trace.get("syndrome_per_step", [])):
            while len(means_per_step) <= i:
                means_per_step.append([])
            means_per_step[i].append(float(v))
    out = []
    for vals in means_per_step:
        out.append(sum(vals) / len(vals) if vals else float("nan"))
    return out


def verify_score_modes(raw_path: Path, mean_path: Path, cal_path: Path) -> int:
    print("\n=== W2. End-to-end runs (already executed if --run) ===")
    fails = 0
    for label, p in [("raw", raw_path), ("mean", mean_path), ("calibrated", cal_path)]:
        if not p.exists():
            print(f"  FAIL {label} trace missing: {p}")
            fails += 1
        else:
            print(f"  PASS {label} trace at {p}")

    if fails > 0:
        return fails

    print("\n=== W3. raw vs mean per-step score reduction ===")
    raw_scores = aggregate_step_score_mean(raw_path)
    mean_scores = aggregate_step_score_mean(mean_path)
    cal_scores = aggregate_step_score_mean(cal_path)
    n_compared = sum(1 for r, m in zip(raw_scores, mean_scores) if r > 0 and m >= 0)
    n_lower = sum(1 for r, m in zip(raw_scores, mean_scores) if r > 0 and m < r)
    if n_compared == 0:
        print("  WARN no checked steps (sampler trace empty?)")
    else:
        ratio = n_lower / n_compared
        print(f"  diagnostic: mean<raw on {n_lower}/{n_compared} steps ({ratio*100:.1f}%)")
        # Print first 10 and last 10 step values for quick eyeball
        idxs = list(range(min(10, len(raw_scores))))
        idxs += [len(raw_scores) - 1 - i for i in range(min(10, len(raw_scores))) if len(raw_scores) - 1 - i not in idxs]
        idxs = sorted(set(idxs))
        print("  step :     raw       mean       calibrated")
        for i in idxs:
            r = raw_scores[i] if i < len(raw_scores) else float("nan")
            m = mean_scores[i] if i < len(mean_scores) else float("nan")
            c = cal_scores[i] if i < len(cal_scores) else float("nan")
            print(f"  {i:4d} : {r:10.4e} {m:10.4e} {c:10.4e}")

    print("\n=== W4. calibrated finite & non-degenerate ===")
    import math
    bad = sum(1 for v in cal_scores if not math.isfinite(v) or v < 0)
    print(f"  {fmt_pass(bad == 0)} {bad} bad calibrated step values out of {len(cal_scores)}")
    if bad > 0:
        fails += 1
    return fails


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stats-path", type=Path, required=True)
    p.add_argument("--run", action="store_true",
                   help="Execute the 3 evaluation runs before verifying.")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=42,
                   help="Evaluation seed (must differ from calibration seed).")
    p.add_argument("--cuda-visible-devices",
                   default=os.environ.get("CUDA_VISIBLE_DEVICES", "2"))
    args = p.parse_args()

    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    fails = 0
    fails += verify_schema(args.stats_path)
    if fails > 0:
        print(f"\n=== Summary: {fails} schema failure(s); aborting ===")
        return 1

    runs = [
        ("p2_raw",        "raw",        None),
        ("p2_mean",       "mean",       args.stats_path),
        ("p2_calibrated", "calibrated", args.stats_path),
    ]
    paths: dict[str, Path] = {}
    for slug, mode, stats in runs:
        cmd, trace_path = build_cmd(slug, mode, stats,
                                    args.cuda_visible_devices,
                                    args.num_samples, args.seed)
        paths[slug] = trace_path
        if args.run:
            rc = run_cmd(cmd, LOG_DIR / f"{slug}.log")
            if rc != 0:
                print(f"!! sampler failed for {slug}; aborting verification.")
                return rc

    fails += verify_score_modes(
        raw_path=paths["p2_raw"],
        mean_path=paths["p2_mean"],
        cal_path=paths["p2_calibrated"],
    )
    print(f"\n=== Summary: {fails} failure(s) ===")
    return 1 if fails > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
