#!/usr/bin/env python3
"""exp_A_correlation — Framing experiment **A**: syndrome ↔ deployment error.

`docs/siec_ecc_framing_20260427.md` §3 의 2-순위 실험. ECC framing 의 핵심 가정
"신드롬 = 노이즈 채널의 evidence" 가 CIFAR S-IEC 에서도 양적으로 성립하는지를
Spearman ρ 한 줄로 판정한다.

설계:
1. **ref-trace** (n=128, fp16 reference): `correction_mode=none`, `trace_include_xs=True`
   → xs_ref[t, i].
2. **deploy-trace** × 5 settings (w8a8, w4a8, dc10, dc20, cachequant):
   - `correction_mode=siec` 으로 호출하되, **τ 를 +inf 로 덮어쓴 임시 파일** 을 사용해
     trigger 를 한 번도 발화시키지 않는다. → xs_deploy[t, i] 는 raw deploy 궤적,
     score_values_per_step 에는 신드롬 점수가 그대로 기록된다.
   - `trace_include_xs=True` 로 xs_deploy 를 보존한다.
3. **analyze** : 각 (setting × t × i) 에 대해
   - `s_norm[t, i] = sqrt(score_values_per_step[t][i])`  (ECC 의 ‖syndrome‖₂).
   - `err_norm[t, i] = ||xs_deploy[t, i] − xs_ref[t, i]||₂ / sqrt(d)`.
   - Spearman ρ (전체 pair) + family 별 분리.

판정:
- ρ > 0.7 → ECC framing 강하게 작동 (syndrome ≈ deploy error 의 proxy).
- 0.3 ≤ ρ ≤ 0.7 → family 별 분리 검사 후 부분 작동.
- ρ < 0.3 → framing 위협 (syndrome 이 error 와 분리됨).

Phases:
    inventory     # 자산 (5 settings 의 calibration) 확인
    ref-trace     # fp16 ref 한 번 (재사용 가능)
    deploy-trace  # 5 setting × n=128
    analyze       # ρ + scatter 데이터
    plot          # scatter (s_norm vs err_norm) + ρ table
    summary
    all
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent.parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import framing_common as fc  # noqa: E402

EXPERIMENT_ID = "exp_A_correlation"
DEFAULT_RESULTS_BASE = fc.RESULTS_BASE / EXPERIMENT_ID
DEPLOY_SETTINGS = ["w8a8", "w4a8", "dc10", "dc20", "cachequant"]
REF_SETTING = "fp16"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=["inventory", "ref-trace", "deploy-trace", "analyze", "plot", "summary", "all"], default="inventory")
    p.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num-samples", type=int, default=128)
    p.add_argument("--sample-batch", type=int, default=128)
    p.add_argument("--seed", type=int, default=fc.DEFAULT_SEED)
    p.add_argument("--percentile", type=float, default=80.0)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_BASE)
    p.add_argument("--cuda-visible-devices", default=None)
    p.add_argument("--reuse-ref-from", type=Path, default=None,
                   help="Reuse ref_fp16_trace.pt produced by an earlier run dir.")
    p.add_argument("--settings", default=",".join(DEPLOY_SETTINGS),
                   help="Comma-separated deploy settings to trace.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Inf-tau trick: write a never-trigger τ schedule per setting.
# ---------------------------------------------------------------------------

def write_inf_tau(setting: str, percentile: float, run_dir: Path) -> Path:
    """Take the existing tau schedule for `setting` and replace all values with +inf
    so S-IEC never triggers a correction during the deploy trace run.
    """
    import torch
    src = fc.tau_schedule_path(setting, percentile)
    if not src.exists():
        raise SystemExit(f"missing tau schedule: {src}")
    obj = torch.load(src, map_location="cpu", weights_only=False)
    # Accept both dict-with-'tau' or raw tensor formats.
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            v = obj[key]
            if torch.is_tensor(v) and v.dtype.is_floating_point:
                obj[key] = torch.full_like(v, float("inf"))
        new_obj = obj
    elif torch.is_tensor(obj):
        new_obj = torch.full_like(obj, float("inf"))
    else:
        # last resort: list[float]
        new_obj = [float("inf")] * len(obj)
    out_dir = run_dir / "tau_inf"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tau_inf_{setting}_p{int(round(percentile))}.pt"
    torch.save(new_obj, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Phase: inventory
# ---------------------------------------------------------------------------

def phase_inventory(args, run_dir: Path) -> None:
    settings = [REF_SETTING] + args.settings.split(",")
    defs = fc.setting_defs()
    missing_total: dict[str, list[str]] = {}
    rows = []
    for s in settings:
        if s not in defs:
            raise SystemExit(f"unknown setting: {s}")
        info = defs[s]
        missing = [p for p in info["required_assets"] if not (fc.IEC_ROOT / p).exists()]
        tau = fc.tau_schedule_path(s, args.percentile) if s != REF_SETTING else None
        missing_tau = bool(tau and not tau.exists()) if s != REF_SETTING else False
        if missing or missing_tau:
            missing_total[s] = missing + ([str(tau)] if missing_tau else [])
        rows.append((s, info["description"], "OK" if not (missing or missing_tau) else "MISSING"))
    md = run_dir / "inventory.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {EXPERIMENT_ID} — Inventory",
        "",
        "| setting | description | status |",
        "|---|---|---|",
    ] + [f"| {s} | {d} | {st} |" for (s, d, st) in rows]
    md.write_text("\n".join(lines) + "\n")
    print(md.read_text())
    if missing_total:
        raise SystemExit(f"[FAIL] missing assets: {missing_total}")


# ---------------------------------------------------------------------------
# Trace command construction
# ---------------------------------------------------------------------------

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
        "--trace_include_xs",
        *fc.build_setting_flags(info),
    ]
    return cmd


def build_deploy_cmd(args, setting: str, tau_inf_path: Path, trace_path: Path, image_folder: Path) -> list[str]:
    info = fc.setting_defs()[setting]
    cmd = fc.conda_python(args) + [
        fc.entry_script("ddim_cifar_siec.py"),
        "--correction-mode", "siec",
        "--num_samples", str(args.num_samples),
        "--sample_batch", str(args.sample_batch),
        "--seed", str(args.seed),
        "--image_folder", fc.rel(image_folder),
        "--siec_return_trace",
        "--siec_trace_mode", "siec",
        "--siec_trace_out", fc.rel(trace_path),
        "--trace_include_x0",
        "--trace_include_xs",
        # never-trigger τ → pristine deploy trajectory + score recording
        "--tau_path", fc.rel(tau_inf_path),
        "--tau_percentile", str(int(round(args.percentile))),
        *fc.build_setting_flags(info),
    ]
    return cmd


# ---------------------------------------------------------------------------
# Phase: ref-trace
# ---------------------------------------------------------------------------

def phase_ref_trace(args, run_dir: Path) -> Path:
    if args.reuse_ref_from:
        candidate = Path(args.reuse_ref_from)
        if candidate.is_dir():
            candidate = candidate / "traces" / "ref_fp16_trace.pt"
        if candidate.exists():
            print(f"[reuse] {candidate} (ref-trace skipped)")
            return candidate
    trace_path = run_dir / "traces" / "ref_fp16_trace.pt"
    image_folder = fc.ERROR_DEC / f"image_expA_ref_fp16_n{args.num_samples}"
    log_path = run_dir / "logs" / "ref_fp16.log"
    cmd = build_ref_cmd(args, run_dir, trace_path, image_folder)
    fc.write_commands_sh([cmd], run_dir / "commands_ref.sh", header="exp_A — ref-trace (fp16)")
    if args.dry_run:
        print(f"[dry-run] ref-trace cmd → {run_dir / 'commands_ref.sh'}")
        return trace_path
    if trace_path.exists():
        print(f"[skip] ref-trace exists: {trace_path}")
        return trace_path
    fc.run_cmd(cmd, log_path)
    return trace_path


# ---------------------------------------------------------------------------
# Phase: deploy-trace
# ---------------------------------------------------------------------------

def phase_deploy_trace(args, run_dir: Path) -> dict[str, Path]:
    settings = args.settings.split(",")
    out: dict[str, Path] = {}
    commands = []
    for s in settings:
        tau_inf = write_inf_tau(s, args.percentile, run_dir)
        trace_path = run_dir / "traces" / f"deploy_{s}_trace.pt"
        image_folder = fc.ERROR_DEC / f"image_expA_deploy_{s}_n{args.num_samples}"
        log_path = run_dir / "logs" / f"deploy_{s}.log"
        cmd = build_deploy_cmd(args, s, tau_inf, trace_path, image_folder)
        commands.append(cmd)
        out[s] = trace_path
        if args.dry_run:
            continue
        if trace_path.exists():
            print(f"[skip] {trace_path} exists")
            continue
        fc.run_cmd(cmd, log_path)
    fc.write_commands_sh(commands, run_dir / "commands_deploy.sh", header=f"exp_A — deploy-trace ({len(settings)} settings)")
    if args.dry_run:
        print(f"[dry-run] {len(commands)} deploy commands → {run_dir / 'commands_deploy.sh'}")
    return out


# ---------------------------------------------------------------------------
# Phase: analyze
# ---------------------------------------------------------------------------

def phase_analyze(args, run_dir: Path, ref_trace: Path, deploy_traces: dict[str, Path]) -> list[dict]:
    import numpy as np
    import torch
    import csv
    from scipy.stats import spearmanr  # type: ignore

    if not ref_trace.exists():
        raise SystemExit(f"ref-trace not found: {ref_trace}")
    ref_traces = fc.load_traces(ref_trace)
    xs_ref = fc.stack_xs_trajectory(ref_traces)  # (N, T, C, H, W)
    if xs_ref.numel() == 0:
        raise SystemExit("ref-trace has empty xs_trajectory; ensure --trace_include_xs was set")

    rows: list[dict] = []
    pair_dump = run_dir / "pairs"
    pair_dump.mkdir(parents=True, exist_ok=True)
    all_s = []
    all_e = []
    all_family = []

    N_ref, T_ref = xs_ref.shape[:2]
    flat_d = float(xs_ref.shape[-3] * xs_ref.shape[-2] * xs_ref.shape[-1])

    for setting, trace_path in deploy_traces.items():
        if not trace_path.exists():
            print(f"[blocked] missing deploy trace: {trace_path}")
            rows.append({"setting": setting, "spearman_rho": None, "p_value": None,
                         "n_pairs": 0, "blocked_reason": "trace missing"})
            continue
        traces = fc.load_traces(trace_path)
        xs_dep = fc.stack_xs_trajectory(traces)
        scores = fc.stack_score_values(traces)  # (N, T)
        if xs_dep.numel() == 0:
            print(f"[blocked] {setting}: empty xs_trajectory")
            rows.append({"setting": setting, "spearman_rho": None, "p_value": None,
                         "n_pairs": 0, "blocked_reason": "no xs"})
            continue
        N = min(xs_dep.shape[0], xs_ref.shape[0])
        T = min(xs_dep.shape[1], xs_ref.shape[1], scores.shape[1])
        diff = (xs_dep[:N, :T] - xs_ref[:N, :T]).reshape(N, T, -1)
        err = diff.norm(dim=-1) / (flat_d ** 0.5)  # (N, T)
        s_norm = np.sqrt(np.maximum(scores[:N, :T], 0.0))  # (N, T)
        e = err.cpu().numpy()
        # Mask out steps where score not recorded (=0 with err >0). For exp_A
        # the inf-tau path records score every step; treat 0-score as missing
        # only when the step was never checked (rare).
        valid = (s_norm > 0)
        if valid.sum() < 32:
            valid = np.ones_like(s_norm, dtype=bool)
        s_flat = s_norm[valid]
        e_flat = e[valid]
        rho, p = spearmanr(s_flat, e_flat)
        n_pairs = len(s_flat)
        rows.append({
            "setting": setting,
            "spearman_rho": round(float(rho), 4),
            "p_value": float(p),
            "n_pairs": int(n_pairs),
            "blocked_reason": "",
            "mean_syndrome": round(float(s_flat.mean()), 6),
            "mean_err": round(float(e_flat.mean()), 6),
        })
        np.save(pair_dump / f"{setting}_pairs.npy",
                np.stack([s_flat, e_flat], axis=1).astype("float32"))
        all_s.append(s_flat)
        all_e.append(e_flat)
        all_family += [setting] * len(s_flat)
        print(f"[ok] {setting}: ρ={rho:.4f}  n={n_pairs}")

    # global ρ across all settings
    if all_s:
        S = np.concatenate(all_s)
        E = np.concatenate(all_e)
        rho_all, p_all = spearmanr(S, E)
        rows.append({
            "setting": "ALL",
            "spearman_rho": round(float(rho_all), 4),
            "p_value": float(p_all),
            "n_pairs": int(len(S)),
            "blocked_reason": "",
            "mean_syndrome": round(float(S.mean()), 6),
            "mean_err": round(float(E.mean()), 6),
        })

    csv_path = run_dir / "correlation_table.csv"
    keys = ["setting", "spearman_rho", "p_value", "n_pairs", "mean_syndrome", "mean_err", "blocked_reason"]
    with open(csv_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    print(f"[ok] table → {csv_path}")
    return rows


# ---------------------------------------------------------------------------
# Phase: plot
# ---------------------------------------------------------------------------

def phase_plot(args, run_dir: Path, rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pair_dir = run_dir / "pairs"
    settings = [r["setting"] for r in rows if r["setting"] != "ALL" and r.get("n_pairs", 0)]
    if not settings:
        print("[plot] nothing to plot (no per-setting pairs)")
        return

    fig, axes = plt.subplots(1, len(settings), figsize=(3 * len(settings), 3.2), sharey=True)
    if len(settings) == 1:
        axes = [axes]
    for ax, s in zip(axes, settings):
        path = pair_dir / f"{s}_pairs.npy"
        if not path.exists():
            ax.text(0.5, 0.5, "no data", ha="center")
            continue
        arr = np.load(path)
        rho = next((r["spearman_rho"] for r in rows if r["setting"] == s), None)
        n = arr.shape[0]
        # subsample for plotting
        idx = np.random.default_rng(0).choice(n, size=min(n, 4000), replace=False)
        ax.scatter(arr[idx, 0], arr[idx, 1], s=3, alpha=0.3)
        ax.set_title(f"{s}\nρ={rho}")
        ax.set_xlabel("‖syndrome‖")
        ax.set_xscale("log")
        ax.set_yscale("log")
    axes[0].set_ylabel("‖x_t deploy − x_t ref‖ / √d")
    fig.suptitle(f"{EXPERIMENT_ID} — syndrome ↔ deploy-error correlation")
    fig.tight_layout()
    out = run_dir / "correlation_scatter.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[ok] plot → {out}")


def phase_summary(args, run_dir: Path, rows: list[dict]) -> None:
    md_lines = [f"# {EXPERIMENT_ID} — Summary", ""]
    by_setting = {r["setting"]: r for r in rows}
    rho_all = (by_setting.get("ALL") or {}).get("spearman_rho")
    if rho_all is None:
        verdict = "INCONCLUSIVE"
    elif rho_all >= 0.7:
        verdict = "PASS — syndrome ↔ deploy-error 강한 단조 상관 (ρ ≥ 0.7)"
    elif rho_all >= 0.3:
        verdict = "PARTIAL — 약한 상관 (0.3 ≤ ρ < 0.7); family 별 분리 필요"
    else:
        verdict = "FAIL — syndrome 이 deploy-error 와 분리 (ρ < 0.3)"
    md_lines += [
        f"- n_samples per trace: {args.num_samples}",
        f"- settings: ref={REF_SETTING}, deploy={args.settings}",
        "",
        f"## Verdict: {verdict}",
        "",
        "| setting | ρ | p | n_pairs | mean ‖s‖ | mean err | note |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['setting']} | {r.get('spearman_rho')} | "
            f"{r.get('p_value'):.2e} | {r.get('n_pairs')} | "
            f"{r.get('mean_syndrome')} | {r.get('mean_err')} | "
            f"{r.get('blocked_reason') or ''} |"
        )
    md_lines.append("")
    (run_dir / "summary.md").write_text("\n".join(md_lines))
    print((run_dir / "summary.md").read_text())


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    run_dir = fc.resolve_results_dir(args.results_dir, plot_only=(args.phase == "plot"))
    print(f"[info] run dir: {run_dir}")

    ref_trace: Path | None = None
    deploy_traces: dict[str, Path] = {}
    rows: list[dict] = []

    if args.phase in ("inventory", "all"):
        phase_inventory(args, run_dir)
    if args.phase in ("ref-trace", "all"):
        ref_trace = phase_ref_trace(args, run_dir)
    if args.phase in ("deploy-trace", "all"):
        deploy_traces = phase_deploy_trace(args, run_dir)
    if args.phase in ("analyze", "all"):
        if ref_trace is None:
            ref_trace = run_dir / "traces" / "ref_fp16_trace.pt"
        if not deploy_traces:
            for s in args.settings.split(","):
                deploy_traces[s] = run_dir / "traces" / f"deploy_{s}_trace.pt"
        rows = phase_analyze(args, run_dir, ref_trace, deploy_traces)
    if args.phase in ("plot", "all"):
        if not rows:
            import csv
            csv_path = run_dir / "correlation_table.csv"
            if csv_path.exists():
                with open(csv_path) as f:
                    rows = list(csv.DictReader(f))
                for r in rows:
                    for k in ("spearman_rho", "p_value", "mean_syndrome", "mean_err"):
                        if r.get(k) not in (None, ""):
                            try:
                                r[k] = float(r[k])
                            except (TypeError, ValueError):
                                pass
                    if r.get("n_pairs") not in (None, ""):
                        r["n_pairs"] = int(float(r["n_pairs"]))
        if rows:
            phase_plot(args, run_dir, rows)
    if args.phase in ("summary", "all"):
        if rows:
            phase_summary(args, run_dir, rows)


if __name__ == "__main__":
    main()
