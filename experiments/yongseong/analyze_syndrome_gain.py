"""Phase 5-3: per-step correction gain g_t and Spearman vs syndrome.

Procedure:
1. Load ref trajectory (PTQ off, no cache reuse, no correction).
2. Load dep no-correction trajectory (PTQ on, cache reuse on, no correction).
3. For each candidate step k in --steps, run sampler with
   --trigger_mode step_set --trigger_steps "k" once, then read its trace.
   Final image final_corr@k vs final_no_corr — gain G_k.
4. Read syndrome scores (raw/mean/calibrated where available) and oracle
   score from existing trace + artifact.
5. Spearman corr(syndrome_t, G_t) and corr(oracle_t, G_t); save csv + png.

Each sampler run forks ddim_cifar_siec.py via conda. Cost ≈ (#steps × ~per-run-time).
For n=16, replicate_interval=10, single-step IEC, expect ~3 min per step on a 4080-class GPU.
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
from utils.ref_trajectory import _concat_batches, load_trace  # noqa: E402


def conda_python(cuda: str) -> list[str]:
    prefix = []
    if cuda:
        prefix = ["env", f"CUDA_VISIBLE_DEVICES={cuda}"]
    return prefix + ["conda", "run", "--no-capture-output", "-n", "iec", "python"]


def run_step_set(
    out_dir: Path,
    step_idx: int,
    *,
    n: int,
    seed: int,
    cuda: str,
    extra_args: list[str],
    log_to: Path,
) -> Path:
    """Execute sampler with --trigger_mode step_set --trigger_steps "<k>" and return trace path."""
    trace_path = out_dir / f"trace_step{step_idx}_seed{seed}_n{n}.pt"
    if trace_path.exists():
        return trace_path
    image_folder = out_dir / f"images_step{step_idx}"
    image_folder.mkdir(parents=True, exist_ok=True)
    cmd = conda_python(cuda) + [
        "mainddpm/ddim_cifar_siec.py",
        "--correction-mode", "siec",
        "--num_samples", str(n),
        "--sample_batch", str(n),
        "--seed", str(seed),
        "--weight_bit", "8", "--act_bit", "8",
        "--replicate_interval", "10",
        "--image_folder", str(image_folder.resolve()),
        "--siec_return_trace",
        "--siec_trace_mode", "siec",
        "--siec_trace_out", str(trace_path.resolve()),
        "--siec_max_rounds", "2",
        "--reuse_lookahead",
        "--trace_include_xs",
        "--trigger_mode", "step_set",
        "--trigger_steps", str(step_idx),
    ] + extra_args
    log_to.parent.mkdir(parents=True, exist_ok=True)
    with log_to.open("w") as f:
        proc = subprocess.run(cmd, cwd=str(REPO), stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"step {step_idx} sampler failed; see {log_to}")
    return trace_path


def final_image(trace) -> torch.Tensor:
    """Return [N, C, H, W] of the final-step xs_trajectory."""
    xs, _ = _concat_batches(trace, key="xs_trajectory")
    if xs is None:
        raise ValueError("trace has no xs_trajectory; pass --trace_include_xs to all runs")
    return xs[-1]  # [N, C, H, W]


def per_sample_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n = min(a.shape[0], b.shape[0])
    diff = (a[:n] - b[:n]).reshape(n, -1)
    return diff.norm(dim=1)  # [N]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True, help=".pt produced by Phase 5-1 ref trace")
    p.add_argument("--dep_no_corr", required=True, help=".pt dep no-correction trace (xs included)")
    p.add_argument("--oracle", required=True, help="oracle_score artifact")
    p.add_argument("--steps", required=True,
                   help="comma list of step indices, or 'topk:K' to take top-K oracle steps "
                        "or 'all' for every step.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--seed", type=int, default=2048)
    p.add_argument("--cuda", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    p.add_argument("--use-cached", action="store_true")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs = out / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    print(f"loading ref: {args.ref}")
    ref = load_trace(args.ref)
    print(f"  num_samples={ref['num_samples']}")
    print(f"loading dep_no_corr: {args.dep_no_corr}")
    dep0 = load_trace(args.dep_no_corr)

    if dep0["xs_trajectory"].numel() == 0:
        raise SystemExit("dep_no_corr trace has empty xs_trajectory; rerun with --trace_include_xs")
    n_common = min(ref["num_samples"], dep0["num_samples"], args.n)
    ref_final = ref["xs_trajectory"][-1, :n_common]
    dep0_final = dep0["xs_trajectory"][-1, :n_common]

    e_no_corr = per_sample_l2(dep0_final, ref_final)  # [N]

    oracle = torch.load(args.oracle, map_location="cpu", weights_only=False)
    oracle_scores_mean = oracle["scores_mean"]
    T = int(oracle_scores_mean.numel())

    if args.steps == "all":
        step_list = list(range(T))
    elif args.steps.startswith("topk:"):
        k = int(args.steps.split(":", 1)[1])
        topk = torch.topk(oracle_scores_mean, k=min(k, T), largest=True).indices
        step_list = sorted(int(s) for s in topk.tolist())
    else:
        step_list = sorted(int(s.strip()) for s in args.steps.split(",") if s.strip())

    print(f"evaluating {len(step_list)} step indices: {step_list}")

    rows = []
    for k in step_list:
        log_to = logs / f"step{k}.log"
        trace_path = run_step_set(
            runs, k,
            n=n_common, seed=args.seed, cuda=args.cuda, extra_args=[],
            log_to=log_to,
        )
        tr = load_trace(trace_path)
        if tr["xs_trajectory"].numel() == 0:
            raise RuntimeError(f"step {k} trace missing xs (re-run with --trace_include_xs)")
        corr_final = tr["xs_trajectory"][-1, :n_common]
        e_corr_at_k = per_sample_l2(corr_final, ref_final)
        gain_per_sample = e_no_corr - e_corr_at_k  # [N]
        g_mean = float(gain_per_sample.mean().item())
        g_std = float(gain_per_sample.std().item())
        g_pos = float((gain_per_sample > 0).float().mean().item())
        oracle_k = float(oracle_scores_mean[k].item())
        # Syndrome at step k under no-prior-triggers: read from step-k's own trace
        # (where step k is the first trigger, so all earlier syndromes are no-correction).
        syndromes = tr.get("syndrome_per_step", None)
        syn_k = float(syndromes[k]) if (syndromes is not None and k < len(syndromes)) else float("nan")
        nfe_used = int(sum(tr["nfe_per_step"]))
        rows.append({
            "step_idx": k,
            "t_int": int(tr["t_int_per_step"][k]) if k < len(tr["t_int_per_step"]) else -1,
            "g_mean": g_mean,
            "g_std": g_std,
            "g_pos_frac": g_pos,
            "oracle_score": oracle_k,
            "syndrome_score": syn_k,
            "nfe_used": nfe_used,
            "n_samples": n_common,
        })
        print(f"  step {k:3d}: g_mean={g_mean:+.4f} std={g_std:.4f} pos={g_pos:.2f} "
              f"oracle={oracle_k:.4f} syn={syn_k:.4f} nfe={nfe_used}")

    csv_path = out / "gain_per_step.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {csv_path}")

    try:
        from scipy.stats import spearmanr
        gains = torch.tensor([r["g_mean"] for r in rows])
        oracles = torch.tensor([r["oracle_score"] for r in rows])
        syndromes = torch.tensor([r["syndrome_score"] for r in rows])
        sp_oracle = spearmanr(oracles.numpy(), gains.numpy())
        sp_syn = spearmanr(syndromes.numpy(), gains.numpy())
        with (out / "spearman.txt").open("w") as f:
            f.write(f"n={len(rows)}\n")
            f.write(f"corr(oracle, g) = {sp_oracle.correlation:.4f}, p={sp_oracle.pvalue:.4g}\n")
            f.write(f"corr(syndrome, g) = {sp_syn.correlation:.4f}, p={sp_syn.pvalue:.4g}\n")
        print(f"spearman: oracle={sp_oracle.correlation:.4f}, syndrome={sp_syn.correlation:.4f}")
    except Exception as e:
        print(f"spearman skipped: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
