"""
Generate tau_schedule from pilot run scores.

Usage:
    python mainddpm/calibrate_tau_cifar.py \\
        --scores_path ./calibration/pilot_scores.pt \\
        --percentile 80 \\
        --out_path ./calibration/tau_schedule_p80.pt
"""
import sys
sys.path.insert(0, '.')
import argparse
import torch
import numpy as np
from siec_core.threshold import calibrate_tau_from_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores_path", type=str, default="./calibration/pilot_scores.pt")
    ap.add_argument("--percentile", type=float, default=80.0)
    ap.add_argument("--out_path", type=str, default=None)
    args = ap.parse_args()

    if args.out_path is None:
        args.out_path = f"./calibration/tau_schedule_p{int(args.percentile)}.pt"

    print(f"Loading pilot scores from: {args.scores_path}")
    scores_by_t = torch.load(args.scores_path)
    T = len(scores_by_t)
    print(f"Loaded scores for {T} timesteps")
    print(f"  per-timestep sample counts: "
          f"min={min(len(s) for s in scores_by_t)}, "
          f"max={max(len(s) for s in scores_by_t)}")

    # Diagnostic: print score statistics
    for t in range(0, T, max(1, T // 10)):
        arr = np.asarray(scores_by_t[t])
        if len(arr) > 0:
            print(f"  t={t}: n={len(arr)}, mean={arr.mean():.6f}, "
                  f"std={arr.std():.6f}, p{int(args.percentile)}={np.percentile(arr, args.percentile):.6f}")

    tau = calibrate_tau_from_scores(scores_by_t, percentile=args.percentile)
    print(f"\nComputed tau_schedule: shape={tau.shape}, "
          f"mean={tau.mean():.6f}, min={tau.min():.6f}, max={tau.max():.6f}")

    torch.save(torch.from_numpy(tau), args.out_path)
    print(f"Saved tau schedule to: {args.out_path}")


if __name__ == "__main__":
    main()
