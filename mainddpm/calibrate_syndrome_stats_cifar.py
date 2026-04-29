#!/usr/bin/env python3
"""calibrate_syndrome_stats_cifar — clean trajectory drift stats for S-IEC.

Wrapper around mainddpm/ddim_cifar_siec.py that runs the sampler in
clean-trajectory mode (PTQ off, no DeepCache reuse, no correction) and
accumulates per-step diagonal mu/var/std of
    drift_t = x0_lookahead(t-1) - x0_current(t)
into a single .pt payload (v1 §3 M1 schema). Calibration vs evaluation seed
split is enforced by passing different --seed to this script vs eval runs.

Usage:
    python mainddpm/calibrate_syndrome_stats_cifar.py \\
        --num-samples 2048 --sample-batch 64 --seed 0 \\
        --out-path calibration/syndrome_stats_clean_seed0_n2048.pt

Default seed=0 keeps calibration trajectories disjoint from the evaluation
default seed (1243). Pass --seed 0 --num-samples 2048 here, then evaluate
with the unchanged default seed elsewhere.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "calibration/syndrome_stats_clean_seed0_n2048.pt"
DEFAULT_IMG = REPO / "experiments/yongseong/results/calibrate_clean_stats/images"
DEFAULT_LOG = REPO / "experiments/yongseong/results/calibrate_clean_stats/calibration.log"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-samples", type=int, default=2048,
                   help="Calibration sample count (default 2048; v1 plan recommendation).")
    p.add_argument("--sample-batch", type=int, default=64,
                   help="Per-batch size for the sampler (default 64).")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for the calibration trajectories. Use 0 for the canonical "
                        "calibration split; evaluation should use a different seed.")
    p.add_argument("--timesteps", type=int, default=100)
    p.add_argument("--replicate-interval", type=int, default=10)
    p.add_argument("--out-path", type=Path, default=DEFAULT_OUT,
                   help="Where to save the v1-format stats payload.")
    p.add_argument("--image-folder", type=Path, default=DEFAULT_IMG,
                   help="Throwaway folder for sample images during calibration.")
    p.add_argument("--log-path", type=Path, default=DEFAULT_LOG,
                   help="Sampler stdout/stderr log path.")
    p.add_argument("--cuda-visible-devices",
                   default=os.environ.get("CUDA_VISIBLE_DEVICES", "2"))
    p.add_argument("--dry-run", action="store_true",
                   help="Print the command without executing it.")
    return p.parse_args()


def conda_python(cuda_visible: str | None) -> list[str]:
    prefix: list[str] = []
    if cuda_visible:
        prefix = ["env", f"CUDA_VISIBLE_DEVICES={cuda_visible}"]
    return prefix + ["conda", "run", "--no-capture-output", "-n", "iec", "python"]


def build_cmd(args: argparse.Namespace) -> list[str]:
    return conda_python(args.cuda_visible_devices) + [
        "mainddpm/ddim_cifar_siec.py",
        "--seed", str(args.seed),
        "--num_samples", str(args.num_samples),
        "--sample_batch", str(args.sample_batch),
        "--timesteps", str(args.timesteps),
        "--replicate_interval", str(args.replicate_interval),
        "--image_folder", str(args.image_folder),
        "--collect_clean_drift_stats", str(args.out_path),
        "--no-ptq",
    ]


def main() -> int:
    args = parse_args()
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.image_folder.mkdir(parents=True, exist_ok=True)
    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(args)
    print(f"$ {shlex.join(cmd)}")
    if args.dry_run:
        return 0
    t0 = time.time()
    with open(args.log_path, "w") as f:
        proc = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    print(f"  -> rc={proc.returncode}  elapsed={elapsed/60:.1f}min  log={args.log_path}")
    if proc.returncode == 0 and args.out_path.exists():
        print(f"  saved: {args.out_path}")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
