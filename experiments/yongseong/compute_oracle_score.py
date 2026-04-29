"""Phase 4b: per-step oracle score from ref vs dep trajectories.

Computes per-step pooled L2 of (x0_dep - x0_ref) across N samples,
saves a small artifact usable as an oracle trigger source.

Output schema:
    {
        "version": 1,
        "kind": "oracle_score_step_pooled",
        "score_space": "x0",
        "pool": "mean",            # also stores median + max for fallback
        "scores_mean": Tensor[T],
        "scores_median": Tensor[T],
        "scores_max": Tensor[T],
        "scores_xs_mean": Tensor[T],
        "score_definition": "per-step mean_n L2(x0_dep[t,n] - x0_ref[t,n])",
        "T": int,
        "num_samples": int,
        "ref_path": str,
        "dep_path": str,
        "seed": Optional[int],
    }
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from utils.ref_trajectory import (  # noqa: E402
    assert_step_aligned,
    load_trace,
    per_step_l2,
)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True)
    p.add_argument("--dep", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    print(f"loading ref: {args.ref}")
    ref = load_trace(args.ref)
    print(f"loading dep: {args.dep}")
    dep = load_trace(args.dep)
    assert_step_aligned(ref, dep)

    n_common = min(ref["num_samples"], dep["num_samples"])
    ref_xs = ref["xs_trajectory"][:, :n_common]
    dep_xs = dep["xs_trajectory"][:, :n_common]
    ref_x0 = ref["x0_trajectory"][:, :n_common]
    dep_x0 = dep["x0_trajectory"][:, :n_common]

    diffs_x0 = per_step_l2(ref_x0, dep_x0)  # [T, N]
    diffs_xs = per_step_l2(ref_xs, dep_xs)

    artifact = {
        "version": 1,
        "kind": "oracle_score_step_pooled",
        "score_space": "x0",
        "pool": "mean",
        "scores_mean": diffs_x0.mean(dim=1).clone(),
        "scores_median": diffs_x0.median(dim=1).values.clone(),
        "scores_max": diffs_x0.max(dim=1).values.clone(),
        "scores_xs_mean": diffs_xs.mean(dim=1).clone(),
        "score_definition": "per-step mean_n L2(x0_dep[t,n] - x0_ref[t,n])",
        "T": int(diffs_x0.shape[0]),
        "num_samples": int(n_common),
        "ref_path": str(args.ref),
        "dep_path": str(args.dep),
        "seed": args.seed,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, out_path)
    print(f"saved {out_path}")

    print("\nsummary (top 12 steps by scores_mean):")
    top = torch.argsort(artifact["scores_mean"], descending=True)[:12]
    print("rank | step | t_int | x0_score_mean | xs_score_mean")
    t_int = ref["t_int_per_step"]
    for rank, k in enumerate(top.tolist()):
        print(f"{rank:4d} | {k:4d} | {int(t_int[k]):4d} | "
              f"{float(artifact['scores_mean'][k]):.4f} | {float(artifact['scores_xs_mean'][k]):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
