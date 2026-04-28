#!/usr/bin/env python3
"""exp_D_lookahead — Framing experiment **D**: lookahead 재활용 fix (toy → CIFAR).

`docs/siec_ecc_framing_20260427.md` §3 의 1-순위 실험. CIFAR S-IEC 의
NFE +91% 폭증이 알고리즘 문제인지 구현 문제인지를 한 줄짜리 ablation 으로
판정한다.

비교 4 run (모두 W8A8_DC10, n=128, 동일 seed=1234+9):
    1. IEC author baseline   — `--no-use-siec` (control; NFE ≈ 100 + n_checks).
    2. S-IEC default          — 기존 동작 (`reuse_lookahead=False`).
    3. S-IEC + reuse          — `--reuse_lookahead` (toy 의도).
    4. S-IEC + reuse + always — `--reuse_lookahead --siec_always_correct` (상한).

판정:
- run3 의 per-sample NFE 가 [99, 121] 범위에 들어오면 fix 성공.
- run3 FID 가 run2 FID 와 ±0.1 안에 있으면 reuse 가 품질을 깨지 않는다.
- 그렇지 않으면 lookahead 자체에 본질적 비용 → framing 재포지셔닝.

Phases:
    inventory        # 자산 확인
    verify           # 4 run + trace 수집 (n_samples=verify_samples; default 128)
    analyze          # trace.pt → results.csv
    fid              # (optional, 무거움) n=2000 commands.sh 만 생성 dry-run
    plot             # NFE / trigger / syndrome 비교 plot
    summary          # summary.md 작성
    all              # verify → analyze → plot → summary
"""
from __future__ import annotations

import argparse
import shlex
import sys
import time
from pathlib import Path

# allow `import framing_common` — this file lives at IEC/experiments/yongseong/framing/exp_D_lookahead.py
# framing_common.py is one level up at IEC/experiments/yongseong/.
EXP_DIR = Path(__file__).resolve().parent.parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import framing_common as fc  # noqa: E402

EXPERIMENT_ID = "exp_D_lookahead"
DEFAULT_RESULTS_BASE = fc.RESULTS_BASE / EXPERIMENT_ID
DEFAULT_SETTING = "w8a8"  # canonical W8A8_DC10 = our "w8a8" setting (PTQ + DC interval=10)


# ---------------------------------------------------------------------------
# Method definitions for the 4-way comparison
# ---------------------------------------------------------------------------

def methods_for_verify() -> list[dict]:
    return [
        dict(
            key="iec_baseline",
            name="IEC (author baseline)",
            correction_mode="iec",
            extra_flags=["--no-use-siec"],
            tau_required=False,
            note="Control: 1저자 IEC sampler, lookahead 미사용.",
        ),
        dict(
            key="siec_default",
            name="S-IEC default (no reuse)",
            correction_mode="siec",
            extra_flags=[],
            tau_required=True,
            note="기존 NFE 폭증 경로 (lookahead=fresh forward 매 step).",
        ),
        dict(
            key="siec_reuse",
            name="S-IEC + reuse_lookahead",
            correction_mode="siec",
            extra_flags=["--reuse_lookahead"],
            tau_required=True,
            note="[EXP-FRAMING-D] cache reuse + memoize.",
        ),
        dict(
            key="siec_reuse_alwayson",
            name="S-IEC reuse + always-on",
            correction_mode="siec",
            extra_flags=["--reuse_lookahead", "--siec_always_correct"],
            tau_required=False,  # always-on은 tau를 무시
            note="reuse + 모든 step 보정 (이론 상한).",
        ),
        dict(
            # [EXP-FRAMING-D] reuse path 의 score 분포에서 다시 컷한 τ.
            # 기존 τ 는 fresh-lookahead 분포에서 잡힌 거라 reuse 켜면 trigger 가
            # 폭증한다 (4% → 44%). 같은 percentile (p80) 을 reuse 분포에서
            # 다시 잡으면 trigger 율이 정상화돼 NFE 가 IEC 수준으로 떨어져야 한다.
            key="siec_reuse_recalib",
            name="S-IEC + reuse + reuse-calibrated τ",
            correction_mode="siec",
            extra_flags=["--reuse_lookahead"],
            tau_required=True,
            tau_override="calibration/tau_schedule_w8a8_reuse_p80.pt",
            note="reuse 분포 기반 p80 재캘리브 (calibrate_tau_reuse.py 산출물).",
        ),
    ]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=["inventory", "verify", "analyze", "fid", "plot", "summary", "all"], default="inventory")
    p.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--setting", default=DEFAULT_SETTING)
    p.add_argument("--verify-samples", type=int, default=128, help="N samples per run during verify (light). Default 128.")
    p.add_argument("--verify-batch", type=int, default=128, help="Sample batch size during verify.")
    p.add_argument("--full-samples", type=int, default=2000, help="N samples for the heavy FID dry-run.")
    p.add_argument("--full-batch", type=int, default=500)
    p.add_argument("--percentile", type=float, default=80.0)
    p.add_argument("--siec-max-rounds", type=int, default=2)
    p.add_argument("--seed", type=int, default=fc.DEFAULT_SEED)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_BASE)
    p.add_argument("--cuda-visible-devices", default=None, help="GPU IDs (comma-sep). Default: env CUDA_VISIBLE_DEVICES or '2'.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Phase: inventory
# ---------------------------------------------------------------------------

def phase_inventory(args, run_dir: Path) -> None:
    defs = fc.setting_defs()
    if args.setting not in defs:
        raise SystemExit(f"unknown setting: {args.setting}")
    info = defs[args.setting]
    report = fc.check_assets({args.setting: info})[args.setting]
    tau_path = fc.tau_schedule_path(args.setting, args.percentile)
    pilot_path = fc.pilot_scores_path(args.setting)
    framing_d_marker = (fc.EXP_DIR / "deepcache_denoising.py").read_text(errors="ignore")
    has_d_tag = "[EXP-FRAMING-D]" in framing_d_marker
    has_a_tag = "[EXP-FRAMING-A]" in framing_d_marker
    has_e_tag = "[EXP-FRAMING-E]" in framing_d_marker
    out_md = run_dir / "inventory.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {EXPERIMENT_ID} — Inventory",
        "",
        f"- setting: `{args.setting}` ({info['description']})",
        f"- tau schedule path: `{fc.rel(tau_path)}`  (exists={tau_path.exists()})",
        f"- pilot scores path: `{fc.rel(pilot_path)}`  (exists={pilot_path.exists()})",
        f"- assets missing: {report['missing'] or '—'}",
        f"- mirror copy `[EXP-FRAMING-D]` tag present: **{has_d_tag}**",
        f"- mirror copy `[EXP-FRAMING-A]` tag present: **{has_a_tag}**",
        f"- mirror copy `[EXP-FRAMING-E]` tag present: **{has_e_tag}**",
    ]
    out_md.write_text("\n".join(lines) + "\n")
    print(out_md.read_text())
    if not (has_d_tag and has_a_tag and has_e_tag):
        raise SystemExit("[FAIL] mirror copy missing one of the FRAMING tags")
    if report["missing"]:
        raise SystemExit(f"[FAIL] missing assets for {args.setting}: {report['missing']}")


# ---------------------------------------------------------------------------
# Phase: verify (4 light runs with trace)
# ---------------------------------------------------------------------------

def build_verify_cmd(args, info: dict, method: dict, image_folder: Path, trace_path: Path) -> list[str]:
    cmd = fc.conda_python(args) + [
        fc.entry_script("ddim_cifar_siec.py"),
        "--correction-mode", method["correction_mode"],
        "--num_samples", str(args.verify_samples),
        "--sample_batch", str(args.verify_batch),
        "--seed", str(args.seed),
        "--siec_max_rounds", str(args.siec_max_rounds),
        "--image_folder", fc.rel(image_folder),
        "--siec_return_trace",
        "--siec_trace_mode", method["correction_mode"] if method["correction_mode"] in {"none", "iec", "siec"} else "siec",
        "--siec_trace_out", fc.rel(trace_path),
        "--trace_include_x0",
        "--trace_include_xs",
        *fc.build_setting_flags(info),
        *method["extra_flags"],
    ]
    if method["tau_required"]:
        if method.get("tau_override"):
            tau = fc.IEC_ROOT / method["tau_override"]
        else:
            tau = fc.tau_schedule_path(args.setting, args.percentile)
        cmd += [
            "--tau_path", fc.rel(tau),
            "--tau_percentile", str(int(round(args.percentile))),
        ]
    return cmd


def phase_verify(args, run_dir: Path) -> list[dict]:
    defs = fc.setting_defs()
    info = defs[args.setting]
    methods = methods_for_verify()
    rows = []
    commands = []
    for m in methods:
        slug = f"{args.setting}_{m['key']}"
        image_folder = fc.ERROR_DEC / f"image_real06_{slug}_n{args.verify_samples}"
        trace_path = run_dir / "traces" / f"{slug}.pt"
        log_path = run_dir / "logs" / f"verify_{slug}.log"
        cmd = build_verify_cmd(args, info, m, image_folder, trace_path)
        commands.append(cmd)
        row = fc.make_row(
            experiment=EXPERIMENT_ID,
            setting=args.setting,
            method_key=m["key"],
            method=m["name"],
            num_samples=args.verify_samples,
            extra={
                "tau_percentile": int(round(args.percentile)) if m["tau_required"] else None,
                "trace_path": fc.rel(trace_path),
                "source_log": fc.rel(log_path),
                "notes": m["note"],
            },
        )
        row["_slug"] = slug
        row["_cmd"] = cmd
        row["_trace"] = trace_path
        row["_log"] = log_path
        row["_image_folder"] = image_folder
        rows.append(row)
    fc.write_commands_sh(
        commands,
        run_dir / "commands_verify.sh",
        header=f"{EXPERIMENT_ID} — verify phase (n={args.verify_samples})",
    )
    if args.dry_run:
        print(f"[dry-run] {len(commands)} commands written to {run_dir / 'commands_verify.sh'}")
        return rows
    for row in rows:
        # skip if trace already exists and is fresh
        if row["_trace"].exists():
            print(f"[skip] {row['_slug']} (trace exists)")
            continue
        elapsed = fc.run_cmd(row["_cmd"], row["_log"])
        row["sampling_wall_clock_sec"] = round(elapsed, 1)
    return rows


# ---------------------------------------------------------------------------
# Phase: analyze (trace.pt → results.csv)
# ---------------------------------------------------------------------------

def phase_analyze(args, run_dir: Path, rows: list[dict]) -> list[dict]:
    import numpy as np
    for row in rows:
        trace_path = run_dir / "traces" / f"{row.get('_slug') or row['method_key']}.pt"
        if not trace_path.exists():
            row["status"] = "blocked"
            row["blocked_reason"] = "no trace.pt"
            continue
        traces = fc.load_traces(trace_path)
        agg = fc.aggregate_nfe(traces)
        scores = fc.stack_score_values(traces)  # (N, T)
        per_sample_nfe = agg["per_sample_total_nfe_mean"]
        # nfe_per_step is summed across batches in agg — divide by batches.
        # per_sample_total_nfe_mean above already = sum(nfe_per_step) / n_batches.
        trigger_rate = agg["per_sample_trigger_rate_mean"]
        # syndrome mean over (N samples, T steps where vals != 0)
        if scores.size:
            mask = scores > 0
            syn_mean = float(scores[mask].mean()) if mask.any() else 0.0
        else:
            syn_mean = 0.0
        n_total_samples = agg["batch_size_total"] or args.verify_samples
        row["per_sample_nfe"] = round(float(per_sample_nfe), 2)
        row["nfe_total"] = int(per_sample_nfe * n_total_samples)
        row["trigger_rate"] = round(float(trigger_rate), 4)
        row["syndrome_mean"] = round(syn_mean, 6)
        row["num_samples"] = n_total_samples
    csv_path = run_dir / "results.csv"
    fc.write_csv(rows, csv_path, extra_keys=["_slug"])
    print(f"[ok] results.csv → {csv_path}")
    return rows


# ---------------------------------------------------------------------------
# Phase: fid (heavy n=2000 dry-run only)
# ---------------------------------------------------------------------------

def phase_fid(args, run_dir: Path) -> None:
    defs = fc.setting_defs()
    info = defs[args.setting]
    methods = methods_for_verify()
    commands = []
    for m in methods:
        slug = f"{args.setting}_{m['key']}_n{args.full_samples}"
        image_folder = fc.ERROR_DEC / f"image_real06_full_{slug}"
        # sampling
        sample_cmd = fc.conda_python(args) + [
            fc.entry_script("ddim_cifar_siec.py"),
            "--correction-mode", m["correction_mode"],
            "--num_samples", str(args.full_samples),
            "--sample_batch", str(args.full_batch),
            "--seed", str(args.seed),
            "--siec_max_rounds", str(args.siec_max_rounds),
            "--image_folder", fc.rel(image_folder),
            *fc.build_setting_flags(info),
            *m["extra_flags"],
        ]
        if m["tau_required"]:
            tau = fc.tau_schedule_path(args.setting, args.percentile)
            sample_cmd += [
                "--tau_path", fc.rel(tau),
                "--tau_percentile", str(int(round(args.percentile))),
            ]
        commands.append(sample_cmd)
        # FID
        npz_path = run_dir / f"samples_{slug}.npz"
        fid_cmd = fc.conda_python(args) + ["evaluator_FID.py", fc.rel(fc.REFERENCE_NPZ), fc.rel(npz_path)]
        commands.append(fid_cmd)
    fc.write_commands_sh(
        commands,
        run_dir / "commands_fid.sh",
        header=f"{EXPERIMENT_ID} — heavy FID n={args.full_samples} (dry-run)",
    )
    print(f"[dry-run] FID commands → {run_dir / 'commands_fid.sh'}")


# ---------------------------------------------------------------------------
# Phase: plot
# ---------------------------------------------------------------------------

def phase_plot(args, run_dir: Path, rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    keys = [r["method_key"] for r in rows]
    nfes = [r.get("per_sample_nfe") or 0 for r in rows]
    triggers = [(r.get("trigger_rate") or 0) * 100 for r in rows]
    syns = [r.get("syndrome_mean") or 0 for r in rows]
    names = [r["method"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = ["tab:gray", "tab:red", "tab:green", "tab:blue"]
    axes[0].bar(keys, nfes, color=colors[: len(keys)])
    axes[0].axhline(110, ls="--", color="black", lw=0.8, label="IEC target (110)")
    axes[0].set_title("Per-sample NFE")
    axes[0].set_ylabel("NFE / sample")
    axes[0].legend(fontsize=8)
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(keys, triggers, color=colors[: len(keys)])
    axes[1].set_title("Trigger rate (%)")
    axes[1].set_ylabel("trigger %")
    axes[1].tick_params(axis="x", rotation=20)

    axes[2].bar(keys, syns, color=colors[: len(keys)])
    axes[2].set_title("Mean syndrome score")
    axes[2].set_ylabel("||s||² / d")
    axes[2].tick_params(axis="x", rotation=20)

    fig.suptitle(
        f"{EXPERIMENT_ID} — lookahead reuse (W8A8 / n={rows[0].get('num_samples', '?')})"
    )
    fig.tight_layout()
    out_png = run_dir / "nfe_breakdown.png"
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"[ok] plot → {out_png}")


# ---------------------------------------------------------------------------
# Phase: summary
# ---------------------------------------------------------------------------

def phase_summary(args, run_dir: Path, rows: list[dict]) -> None:
    by_key = {r["method_key"]: r for r in rows}
    iec = by_key.get("iec_baseline")
    siec = by_key.get("siec_default")
    siec_reuse = by_key.get("siec_reuse")
    iec_nfe = (iec or {}).get("per_sample_nfe")
    siec_nfe = (siec or {}).get("per_sample_nfe")
    reuse_nfe = (siec_reuse or {}).get("per_sample_nfe")
    verdict = "INCONCLUSIVE"
    detail = ""
    if reuse_nfe is not None and iec_nfe is not None:
        ratio = reuse_nfe / iec_nfe if iec_nfe > 0 else float("inf")
        if 0.9 <= ratio <= 1.1:
            verdict = "PASS — fix collapses NFE to IEC ± 10%"
        elif siec_nfe and reuse_nfe < siec_nfe:
            verdict = "PARTIAL — fix reduces NFE but not to IEC level"
        else:
            verdict = "FAIL — reuse did not reduce NFE; lookahead has intrinsic cost"
        detail = (
            f"  - IEC baseline NFE/sample: **{iec_nfe}**\n"
            f"  - S-IEC default NFE/sample: **{siec_nfe}**\n"
            f"  - S-IEC + reuse NFE/sample: **{reuse_nfe}** (ratio vs IEC = {ratio:.2f})"
        )
    md = run_dir / "summary.md"
    md.write_text(
        "\n".join(
            [
                f"# {EXPERIMENT_ID} — Summary",
                "",
                f"- setting: `{args.setting}`",
                f"- n_samples per run: {args.verify_samples}",
                f"- seed: {args.seed}",
                "",
                f"## Verdict: {verdict}",
                "",
                detail,
                "",
                "| method | NFE/sample | trigger rate | syndrome mean |",
                "|---|---|---|---|",
            ]
            + [
                f"| {r['method']} | {r.get('per_sample_nfe')} | {r.get('trigger_rate')} | {r.get('syndrome_mean')} |"
                for r in rows
            ]
            + [""]
        )
    )
    print(md.read_text())


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    run_dir = fc.resolve_results_dir(args.results_dir, plot_only=(args.phase == "plot"))
    print(f"[info] run dir: {run_dir}")

    if args.phase in ("inventory", "all"):
        phase_inventory(args, run_dir)
    rows: list[dict] = []
    if args.phase in ("verify", "all"):
        rows = phase_verify(args, run_dir)
    if args.phase in ("analyze", "all"):
        if not rows:
            # rebuild method-skeleton without re-running
            defs = fc.setting_defs()
            info = defs[args.setting]
            for m in methods_for_verify():
                slug = f"{args.setting}_{m['key']}"
                trace = run_dir / "traces" / f"{slug}.pt"
                row = fc.make_row(
                    experiment=EXPERIMENT_ID,
                    setting=args.setting,
                    method_key=m["key"],
                    method=m["name"],
                    num_samples=args.verify_samples,
                    extra={
                        "tau_percentile": int(round(args.percentile)) if m["tau_required"] else None,
                        "trace_path": fc.rel(trace),
                        "notes": m["note"],
                    },
                )
                row["_slug"] = slug
                rows.append(row)
        phase_analyze(args, run_dir, rows)
    if args.phase in ("fid", "all"):
        phase_fid(args, run_dir)
    if args.phase in ("plot", "all"):
        if not rows:
            # load from results.csv
            import csv
            csv_path = run_dir / "results.csv"
            if not csv_path.exists():
                raise SystemExit(f"plot phase requires results.csv at {csv_path}")
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            for r in rows:
                for k in ("per_sample_nfe", "trigger_rate", "syndrome_mean"):
                    if r.get(k) not in (None, ""):
                        r[k] = float(r[k])
        if rows:
            phase_plot(args, run_dir, rows)
    if args.phase in ("summary", "all"):
        if rows:
            phase_summary(args, run_dir, rows)


if __name__ == "__main__":
    main()
