"""Phase 5-1: ref vs deployment trajectory analysis.

Loads two traces (PTQ-off ref, PTQ-on dep) generated with same seed,
verifies step alignment, computes per-step L2 divergence on xs/x0,
writes csv + png summary.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from utils.ref_trajectory import (  # noqa: E402
    assert_step_aligned,
    load_trace,
    per_step_l2_summary,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True)
    p.add_argument("--dep", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading ref: {args.ref}")
    ref = load_trace(args.ref)
    print(f"  num_samples={ref['num_samples']} xs_shape={tuple(ref['xs_trajectory'].shape)}")
    print(f"loading dep: {args.dep}")
    dep = load_trace(args.dep)
    print(f"  num_samples={dep['num_samples']} xs_shape={tuple(dep['xs_trajectory'].shape)}")

    assert_step_aligned(ref, dep)
    print("step alignment OK")

    n_common = min(ref["num_samples"], dep["num_samples"])
    if ref["num_samples"] != dep["num_samples"]:
        print(f"truncating to common N={n_common}")
        ref["xs_trajectory"] = ref["xs_trajectory"][:, :n_common]
        ref["x0_trajectory"] = ref["x0_trajectory"][:, :n_common]
        ref["num_samples"] = n_common
        dep["xs_trajectory"] = dep["xs_trajectory"][:, :n_common]
        dep["x0_trajectory"] = dep["x0_trajectory"][:, :n_common]
        dep["num_samples"] = n_common

    xs_summary = per_step_l2_summary(ref, dep, key="xs_trajectory")
    x0_summary = per_step_l2_summary(ref, dep, key="x0_trajectory")
    t_int = ref["t_int_per_step"]

    csv_path = out / "ref_vs_dep_per_step.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "step_idx", "t_int",
            "xs_diff_mean", "xs_diff_std", "xs_diff_max",
            "xs_ref_norm_mean", "xs_ratio_mean",
            "x0_diff_mean", "x0_diff_std", "x0_diff_max",
            "x0_ref_norm_mean", "x0_ratio_mean",
        ])
        T = len(t_int)
        for k in range(T):
            w.writerow([
                k, int(t_int[k]),
                float(xs_summary["diff_mean"][k]), float(xs_summary["diff_std"][k]), float(xs_summary["diff_max"][k]),
                float(xs_summary["ref_norm_mean"][k]), float(xs_summary["ratio_mean"][k]),
                float(x0_summary["diff_mean"][k]), float(x0_summary["diff_std"][k]), float(x0_summary["diff_max"][k]),
                float(x0_summary["ref_norm_mean"][k]), float(x0_summary["ratio_mean"][k]),
            ])
    print(f"wrote {csv_path}")

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(t_int, xs_summary["ratio_mean"].numpy(), label="xs ratio mean", color="#1F77B4")
    ax[0].fill_between(
        t_int,
        (xs_summary["ratio_mean"] - xs_summary["diff_std"] / (xs_summary["ref_norm_mean"] + 1e-9)).numpy(),
        (xs_summary["ratio_mean"] + xs_summary["diff_std"] / (xs_summary["ref_norm_mean"] + 1e-9)).numpy(),
        alpha=0.2, color="#1F77B4",
    )
    ax[0].plot(t_int, x0_summary["ratio_mean"].numpy(), label="x0 ratio mean", color="#E07814")
    ax[0].set_ylabel("L2(ref - dep) / L2(ref)")
    ax[0].set_title(f"per-step PTQ+DeepCache divergence (N={ref['num_samples']})")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(t_int, xs_summary["diff_mean"].numpy(), label="xs L2 mean", color="#1F77B4")
    ax[1].plot(t_int, x0_summary["diff_mean"].numpy(), label="x0 L2 mean", color="#E07814")
    ax[1].set_xlabel("t_int (decreasing →)")
    ax[1].set_ylabel("L2(ref - dep)")
    ax[1].invert_xaxis()
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    png = out / "ref_vs_dep_divergence.png"
    fig.savefig(png, dpi=140)
    fig.savefig(out / "ref_vs_dep_divergence.pdf")
    plt.close(fig)
    print(f"wrote {png}")

    print("\nsummary at sampled steps:")
    print("step | t_int | xs_ratio | x0_ratio")
    for k in [0, 10, 30, 50, 70, 90, 99]:
        if k >= len(t_int):
            continue
        print(f"{k:4d} | {int(t_int[k]):4d} | {float(xs_summary['ratio_mean'][k]):.4f} | {float(x0_summary['ratio_mean'][k]):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
