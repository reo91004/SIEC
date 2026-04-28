#!/usr/bin/env python3
"""exp_B_innovation — Framing experiment **B**: Tweedie martingale (G1).

`docs/siec_ecc_framing_20260427.md` §3 의 3-순위 실험. ECC framing 의 1번째 갭
(G1 — innovation 의 분포 가정) 을 fp16 한 번의 trace 로 양적 검증한다.

이론 :
- DDIM 의 x0_t (Tweedie 추정량) 가 martingale 이라면 m_t = x0_t − x0_{t−1} 의 평균은 0,
  분산은 σ_t² − σ_{t−1}² 에 비례 (선형 회귀의 R² 가 1 에 근접).

판정 :
- |mean(m_t)| / σ_t < 0.1 그리고 R² ≥ 0.8 → G1 위반 작음 (framing 문제 없음).
- 그 외 → 명시적 위반, ICLR 본문에 언급 필요.

Phases :
    inventory       # fp16 자산 + 이전 ref-trace 재사용 가능성 확인
    ref-trace       # fp16 trace (--correction-mode none --trace_include_x0)
    analyze         # x0_trajectory → m_t, mean, var, σ² 회귀
    plot            # bar (mean) + scatter (Var vs σ²-diff)
    summary
    all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent.parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import framing_common as fc  # noqa: E402

EXPERIMENT_ID = "exp_B_innovation"
DEFAULT_RESULTS_BASE = fc.RESULTS_BASE / EXPERIMENT_ID
REF_SETTING = "fp16"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=["inventory", "ref-trace", "analyze", "plot", "summary", "all"], default="inventory")
    p.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num-samples", type=int, default=128)
    p.add_argument("--sample-batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=fc.DEFAULT_SEED)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_BASE)
    p.add_argument("--cuda-visible-devices", default=None)
    p.add_argument("--reuse-ref-from", type=Path, default=None,
                   help="Path to a prior trace.pt or its run-dir (e.g. exp_A_correlation run).")
    return p.parse_args()


def build_ref_cmd(args, run_dir: Path, trace_path: Path, image_folder: Path) -> list[str]:
    info = fc.setting_defs()[REF_SETTING]
    cmd = fc.conda_python(args) + [
        fc.entry_script("ddim_cifar_siec.py"),
        "--correction-mode", "none",
        "--num_samples", str(args.num_samples),
        "--sample_batch", str(args.sample_batch),
        "--seed", str(args.seed),
        "--image_folder", fc.rel(image_folder),
        "--siec_return_trace",
        "--siec_trace_mode", "none",
        "--siec_trace_out", fc.rel(trace_path),
        "--trace_include_x0",
        *fc.build_setting_flags(info),
    ]
    return cmd


def phase_inventory(args, run_dir: Path) -> None:
    info = fc.setting_defs()[REF_SETTING]
    missing = [p for p in info["required_assets"] if not (fc.IEC_ROOT / p).exists()]
    md = run_dir / "inventory.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(
        f"# {EXPERIMENT_ID} — Inventory\n\n"
        f"- ref setting: `{REF_SETTING}` ({info['description']})\n"
        f"- assets missing: {missing or '—'}\n"
        f"- ref-trace reuse path: `{fc.rel(Path(args.reuse_ref_from)) if args.reuse_ref_from else '—'}`\n"
    )
    print(md.read_text())
    if missing:
        raise SystemExit(f"[FAIL] missing assets: {missing}")


def phase_ref_trace(args, run_dir: Path) -> Path:
    if args.reuse_ref_from:
        candidate = Path(args.reuse_ref_from)
        if candidate.is_dir():
            for sub in ("traces/ref_fp16_trace.pt", "ref_fp16_trace.pt"):
                if (candidate / sub).exists():
                    candidate = candidate / sub
                    break
        if candidate.is_file():
            print(f"[reuse] {candidate}")
            return candidate
        print(f"[reuse-skip] {args.reuse_ref_from} has no ref_fp16_trace.pt — sampling fresh")
    trace_path = run_dir / "traces" / "ref_fp16_trace.pt"
    image_folder = fc.ERROR_DEC / f"image_expB_ref_fp16_n{args.num_samples}"
    log_path = run_dir / "logs" / "ref_fp16.log"
    cmd = build_ref_cmd(args, run_dir, trace_path, image_folder)
    fc.write_commands_sh([cmd], run_dir / "commands_ref.sh", header="exp_B — fp16 ref-trace")
    if args.dry_run:
        print(f"[dry-run] cmd → {run_dir / 'commands_ref.sh'}")
        return trace_path
    if trace_path.exists():
        print(f"[skip] {trace_path}")
        return trace_path
    fc.run_cmd(cmd, log_path)
    return trace_path


def phase_analyze(args, run_dir: Path, trace_path: Path) -> dict:
    import csv
    import numpy as np
    import torch

    if not trace_path.exists():
        raise SystemExit(f"trace not found: {trace_path}")
    traces = fc.load_traces(trace_path)
    x0 = fc.stack_x0_trajectory(traces)  # (N, T, C, H, W)
    if x0.numel() == 0:
        raise SystemExit("trace has empty x0_trajectory; ensure --trace_include_x0 was set")
    N, T = x0.shape[:2]
    flat = x0.reshape(N, T, -1)
    # innovations: m[t] = x0[t] - x0[t-1]   for t >= 1
    m = flat[:, 1:] - flat[:, :-1]  # (N, T-1, D)
    # σ_t in DDIM with cosine/linear-β schedule; we use ᾱ_t and σ_t² = (1 − ᾱ_t)/ᾱ_t in the
    # noise-conditioning convention the toy uses, OR simply σ_t² = 1 − ᾱ_t (DDPM forward).
    alpha_bar = fc.step_alpha_bars(T)  # length T
    sigma_sq = (1.0 - alpha_bar).cpu().numpy()  # (T,)
    # innovation refers to step transition t-1 → t; reference σ² difference is sigma_sq[t]-sigma_sq[t-1]
    sigma_diff = sigma_sq[1:] - sigma_sq[:-1]  # (T-1,)

    mean_per_t = m.mean(dim=(0, 2)).cpu().numpy()  # (T-1,)
    var_per_t = m.var(dim=(0, 2), unbiased=False).cpu().numpy()
    sigma_t = np.sqrt(np.maximum(sigma_sq, 1e-30))[1:]  # σ for t ∈ {1..T-1}

    rel_mean = np.abs(mean_per_t) / np.maximum(sigma_t, 1e-12)

    # Linear regression Var ~ a * σ_diff + b (allowing intercept).
    A = np.stack([sigma_diff, np.ones_like(sigma_diff)], axis=1)
    coef, residuals, rank, sv = np.linalg.lstsq(A, var_per_t, rcond=None)
    pred = A @ coef
    ss_res = float(((var_per_t - pred) ** 2).sum())
    ss_tot = float(((var_per_t - var_per_t.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    csv_path = run_dir / "innovation_table.csv"
    with open(csv_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["t", "mean_m", "var_m", "sigma_t", "sigma_diff_sq"])
        for i in range(len(mean_per_t)):
            w.writerow([i + 1,
                        float(mean_per_t[i]),
                        float(var_per_t[i]),
                        float(sigma_t[i]),
                        float(sigma_diff[i])])
    print(f"[ok] table → {csv_path}")

    return {
        "n_samples": int(N),
        "T": int(T),
        "mean_per_t": mean_per_t,
        "var_per_t": var_per_t,
        "sigma_t": sigma_t,
        "sigma_diff": sigma_diff,
        "rel_mean_max": float(rel_mean.max()),
        "rel_mean_mean": float(rel_mean.mean()),
        "regression_slope": float(coef[0]),
        "regression_intercept": float(coef[1]),
        "regression_r2": float(r2),
    }


def phase_plot(args, run_dir: Path, stats: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    t_axis = np.arange(1, len(stats["mean_per_t"]) + 1)
    axes[0].bar(t_axis, stats["mean_per_t"], color="tab:blue", alpha=0.7)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_title("mean(x0_t − x0_{t-1})")
    axes[0].set_xlabel("step t")
    axes[0].set_ylabel("mean innovation")

    sd = stats["sigma_diff"]
    var = stats["var_per_t"]
    axes[1].scatter(sd, var, s=10, alpha=0.7)
    a = stats["regression_slope"]
    b = stats["regression_intercept"]
    xs = np.linspace(sd.min(), sd.max(), 100)
    axes[1].plot(xs, a * xs + b, color="red", lw=1.0,
                 label=f"y = {a:.3f}·x + {b:.3e}\nR² = {stats['regression_r2']:.3f}")
    axes[1].set_title("Var(innovation) vs σ_t² − σ_{t-1}²")
    axes[1].set_xlabel("σ_t² − σ_{t-1}²")
    axes[1].set_ylabel("Var(m_t)")
    axes[1].legend(fontsize=8)

    fig.suptitle(f"{EXPERIMENT_ID} — Tweedie martingale (G1) — fp16, n={stats['n_samples']}")
    fig.tight_layout()
    out = run_dir / "innovation_plots.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[ok] plot → {out}")


def phase_summary(args, run_dir: Path, stats: dict) -> None:
    rel_mean_max = stats["rel_mean_max"]
    r2 = stats["regression_r2"]
    if rel_mean_max < 0.1 and r2 >= 0.8:
        verdict = "PASS — G1 위반 작음"
    elif rel_mean_max < 0.2 and r2 >= 0.5:
        verdict = "PARTIAL — 약한 위반 (mean ≠ 0 또는 R² 낮음)"
    else:
        verdict = "FAIL — G1 명시적 위반"
    md = run_dir / "summary.md"
    md.write_text(
        "\n".join([
            f"# {EXPERIMENT_ID} — Summary",
            "",
            f"- n_samples: {stats['n_samples']}",
            f"- T (sampling steps): {stats['T']}",
            "",
            f"## Verdict: {verdict}",
            "",
            f"- |mean(m_t)| / σ_t — max: **{rel_mean_max:.4f}**, mean: {stats['rel_mean_mean']:.4f}",
            f"- Var(m_t) ≈ slope · (σ_t² − σ_{{t−1}}²) + intercept",
            f"  - slope: **{stats['regression_slope']:.4f}**  (theory: 1)",
            f"  - intercept: {stats['regression_intercept']:.3e}",
            f"  - R²: **{r2:.4f}**",
            "",
        ])
    )
    print(md.read_text())


def main():
    args = parse_args()
    run_dir = fc.resolve_results_dir(args.results_dir, plot_only=(args.phase == "plot"))
    print(f"[info] run dir: {run_dir}")

    if args.phase in ("inventory", "all"):
        phase_inventory(args, run_dir)
    trace_path: Path | None = None
    if args.phase in ("ref-trace", "all"):
        trace_path = phase_ref_trace(args, run_dir)
    stats: dict | None = None
    if args.phase in ("analyze", "all"):
        if trace_path is None:
            trace_path = run_dir / "traces" / "ref_fp16_trace.pt"
            if args.reuse_ref_from:
                cand = Path(args.reuse_ref_from)
                if cand.is_dir():
                    for sub in ("traces/ref_fp16_trace.pt", "ref_fp16_trace.pt"):
                        if (cand / sub).exists():
                            trace_path = cand / sub
                            break
                elif cand.exists():
                    trace_path = cand
        stats = phase_analyze(args, run_dir, trace_path)
    if args.phase in ("plot", "all"):
        if stats is None:
            # Reload from CSV if needed.
            import csv
            import numpy as np
            csv_path = run_dir / "innovation_table.csv"
            if csv_path.exists():
                rows = list(csv.DictReader(open(csv_path)))
                m = np.array([float(r["mean_m"]) for r in rows])
                v = np.array([float(r["var_m"]) for r in rows])
                sg = np.array([float(r["sigma_t"]) for r in rows])
                sd = np.array([float(r["sigma_diff_sq"]) for r in rows])
                # quick re-regression
                A = np.stack([sd, np.ones_like(sd)], axis=1)
                coef, *_ = np.linalg.lstsq(A, v, rcond=None)
                pred = A @ coef
                ss_res = float(((v - pred) ** 2).sum())
                ss_tot = float(((v - v.mean()) ** 2).sum())
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                rel = np.abs(m) / np.maximum(sg, 1e-12)
                stats = dict(
                    n_samples=0, T=len(rows) + 1,
                    mean_per_t=m, var_per_t=v, sigma_t=sg, sigma_diff=sd,
                    rel_mean_max=float(rel.max()), rel_mean_mean=float(rel.mean()),
                    regression_slope=float(coef[0]), regression_intercept=float(coef[1]),
                    regression_r2=float(r2),
                )
        if stats is not None:
            phase_plot(args, run_dir, stats)
    if args.phase in ("summary", "all"):
        if stats is not None:
            phase_summary(args, run_dir, stats)


if __name__ == "__main__":
    main()
