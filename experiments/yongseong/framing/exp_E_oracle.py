#!/usr/bin/env python3
"""exp_E_oracle — Framing experiment **E**: Oracle decoder upper bound.

`docs/siec_ecc_framing_20260427.md` §3 의 6-순위 실험. S-IEC 의 framing 이
이론적으로 도달 가능한 최선이 어디인지를 측정한다. 매 step xt 를 fp16
reference trajectory 로 직접 pull (`oracle_xt_ref`) 하면 syndrome 가 모르는
모든 noise 를 제거한 셈 — 그 결과 FID 가 fp16 와 비슷하면 framing 상한 ↑.

설계 :
1. **ref-trace**: fp16 (correction_mode=none, trace_include_xs) 1 회. 같은 seed/batch 로
   downstream 의 oracle 에 그대로 reference 가 됨.
2. **oracle-run** × {w8a8, dc10, w4a8} × pull ∈ {0.5, 1.0}:
   `correction_mode=siec_oracle --oracle_xt_ref <ref> --oracle_pull_strength <pull>`.
   n=oracle_samples (default 128, 가벼움). 무거운 FID 는 dry-run 으로 commands.sh.
3. **analyze**: oracle 결과 npz vs fp16 reference NPZ 의 FID 를 ingest (실측 후) 또는
   image_folder 의 PNG 카운트만 기록 (가벼운 모드).

Phases :
    inventory       # ref-trace 가용성 + 자산 점검
    ref-trace       # fp16 trace 생성 (또는 reuse)
    oracle-run      # 3 setting × 2 pull = 6 run
    fid             # n=2000 dry-run (heavy)
    analyze         # PNG count + (있으면) FID
    plot            # bar
    summary
    all
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent.parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import framing_common as fc  # noqa: E402

EXPERIMENT_ID = "exp_E_oracle"
DEFAULT_RESULTS_BASE = fc.RESULTS_BASE / EXPERIMENT_ID
ORACLE_SETTINGS = ["w8a8", "dc10", "w4a8"]
ORACLE_PULLS = [0.5, 1.0]
REF_SETTING = "fp16"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=["inventory", "ref-trace", "oracle-run", "fid", "analyze", "plot", "summary", "all"], default="inventory")
    p.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--oracle-samples", type=int, default=128)
    p.add_argument("--oracle-batch", type=int, default=128)
    p.add_argument("--full-samples", type=int, default=2000)
    p.add_argument("--full-batch", type=int, default=500)
    p.add_argument("--seed", type=int, default=fc.DEFAULT_SEED)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_BASE)
    p.add_argument("--reuse-ref-from", type=Path, default=None,
                   help="Reuse ref_fp16_trace.pt from another run dir (e.g. exp_A_correlation).")
    p.add_argument("--cuda-visible-devices", default=None)
    p.add_argument("--settings", default=",".join(ORACLE_SETTINGS),
                   help="Comma-separated deploy settings to oracle-test.")
    p.add_argument("--pulls", default=",".join(str(x) for x in ORACLE_PULLS),
                   help="Comma-separated pull strengths.")
    return p.parse_args()


def phase_inventory(args, run_dir: Path) -> None:
    settings = [REF_SETTING] + args.settings.split(",")
    defs = fc.setting_defs()
    rows = []
    for s in settings:
        if s not in defs:
            raise SystemExit(f"unknown setting: {s}")
        info = defs[s]
        missing = [p for p in info["required_assets"] if not (fc.IEC_ROOT / p).exists()]
        rows.append((s, info["description"], "OK" if not missing else f"MISSING:{missing}"))
    md = run_dir / "inventory.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text(
        "\n".join([
            f"# {EXPERIMENT_ID} — Inventory",
            "",
            f"- ref-trace reuse: `{fc.rel(Path(args.reuse_ref_from)) if args.reuse_ref_from else '—'}`",
            f"- pull strengths: {args.pulls}",
            "",
            "| setting | description | status |",
            "|---|---|---|",
        ] + [f"| {s} | {d} | {st} |" for (s, d, st) in rows]) + "\n"
    )
    print(md.read_text())


def build_ref_cmd(args, trace_path: Path, image_folder: Path) -> list[str]:
    info = fc.setting_defs()[REF_SETTING]
    cmd = fc.conda_python(args) + [
        fc.entry_script("ddim_cifar_siec.py"),
        "--correction-mode", "none",
        "--num_samples", str(args.oracle_samples),
        "--sample_batch", str(args.oracle_batch),
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


def phase_ref_trace(args, run_dir: Path) -> Path:
    if args.reuse_ref_from:
        cand = Path(args.reuse_ref_from)
        if cand.is_dir():
            for sub in ("traces/ref_fp16_trace.pt", "ref_fp16_trace.pt"):
                if (cand / sub).exists():
                    cand = cand / sub
                    break
        if cand.is_file():
            print(f"[reuse] {cand}")
            return cand
        print(f"[reuse-skip] {args.reuse_ref_from} has no ref_fp16_trace.pt — sampling fresh")
    trace_path = run_dir / "traces" / "ref_fp16_trace.pt"
    image_folder = fc.ERROR_DEC / f"image_expE_ref_fp16_n{args.oracle_samples}"
    log_path = run_dir / "logs" / "ref_fp16.log"
    cmd = build_ref_cmd(args, trace_path, image_folder)
    fc.write_commands_sh([cmd], run_dir / "commands_ref.sh", header="exp_E — fp16 ref-trace (oracle source)")
    if args.dry_run:
        print(f"[dry-run] cmd → {run_dir / 'commands_ref.sh'}")
        return trace_path
    if trace_path.exists():
        print(f"[skip] {trace_path}")
        return trace_path
    fc.run_cmd(cmd, log_path)
    return trace_path


def build_oracle_cmd(args, setting: str, pull: float, ref_trace: Path,
                     trace_path: Path, image_folder: Path) -> list[str]:
    info = fc.setting_defs()[setting]
    cmd = fc.conda_python(args) + [
        fc.entry_script("ddim_cifar_siec.py"),
        "--correction-mode", "siec_oracle",
        "--num_samples", str(args.oracle_samples),
        "--sample_batch", str(args.oracle_batch),
        "--seed", str(args.seed),
        "--image_folder", fc.rel(image_folder),
        "--siec_return_trace",
        "--siec_trace_mode", "siec",  # for trace path; oracle pull is applied
        "--siec_trace_out", fc.rel(trace_path),
        "--trace_include_x0",
        "--trace_include_xs",
        "--oracle_xt_ref", fc.rel(ref_trace),
        "--oracle_pull_strength", str(pull),
        *fc.build_setting_flags(info),
    ]
    return cmd


def phase_oracle_run(args, run_dir: Path, ref_trace: Path) -> list[dict]:
    settings = args.settings.split(",")
    pulls = [float(x) for x in args.pulls.split(",")]
    rows: list[dict] = []
    cmds: list[list[str]] = []
    for s in settings:
        for pull in pulls:
            slug = f"{s}_pull{int(round(pull * 100))}"
            trace_path = run_dir / "traces" / f"{slug}.pt"
            image_folder = fc.ERROR_DEC / f"image_expE_{slug}_n{args.oracle_samples}"
            log_path = run_dir / "logs" / f"oracle_{slug}.log"
            cmd = build_oracle_cmd(args, s, pull, ref_trace, trace_path, image_folder)
            cmds.append(cmd)
            row = {
                "setting": s,
                "pull_strength": pull,
                "slug": slug,
                "trace_path": fc.rel(trace_path),
                "image_folder": fc.rel(image_folder),
                "source_log": fc.rel(log_path),
            }
            rows.append(row)
            row["_cmd"] = cmd
            row["_trace"] = trace_path
            row["_log"] = log_path
            row["_image"] = image_folder
    fc.write_commands_sh(cmds, run_dir / "commands_oracle.sh", header=f"exp_E — oracle run ({len(cmds)} cells)")
    if args.dry_run:
        print(f"[dry-run] {len(cmds)} oracle commands → {run_dir / 'commands_oracle.sh'}")
        return rows
    for row in rows:
        if row["_trace"].exists():
            print(f"[skip] {row['slug']} (trace exists)")
            continue
        fc.run_cmd(row["_cmd"], row["_log"])
    return rows


def phase_fid(args, run_dir: Path) -> None:
    """Generate heavy n=2000 sampling + FID dry-run commands (oracle path)."""
    settings = args.settings.split(",")
    pulls = [float(x) for x in args.pulls.split(",")]
    cmds = []
    # Heavy oracle ref @ n=2000 first.
    info = fc.setting_defs()[REF_SETTING]
    heavy_ref = run_dir / "traces" / f"ref_fp16_n{args.full_samples}.pt"
    image_folder_ref = fc.ERROR_DEC / f"image_expE_ref_fp16_n{args.full_samples}"
    cmds.append(fc.conda_python(args) + [
        fc.entry_script("ddim_cifar_siec.py"),
        "--correction-mode", "none",
        "--num_samples", str(args.full_samples),
        "--sample_batch", str(args.full_batch),
        "--seed", str(args.seed),
        "--image_folder", fc.rel(image_folder_ref),
        "--siec_return_trace",
        "--siec_trace_mode", "none",
        "--siec_trace_out", fc.rel(heavy_ref),
        "--trace_include_xs",
        *fc.build_setting_flags(info),
    ])
    for s in settings:
        info_s = fc.setting_defs()[s]
        for pull in pulls:
            slug = f"{s}_pull{int(round(pull * 100))}_n{args.full_samples}"
            heavy_trace = run_dir / "traces" / f"{slug}.pt"
            image_folder = fc.ERROR_DEC / f"image_expE_{slug}"
            cmds.append(fc.conda_python(args) + [
                fc.entry_script("ddim_cifar_siec.py"),
                "--correction-mode", "siec_oracle",
                "--num_samples", str(args.full_samples),
                "--sample_batch", str(args.full_batch),
                "--seed", str(args.seed),
                "--image_folder", fc.rel(image_folder),
                "--siec_return_trace",
                "--siec_trace_mode", "siec",
                "--siec_trace_out", fc.rel(heavy_trace),
                "--oracle_xt_ref", fc.rel(heavy_ref),
                "--oracle_pull_strength", str(pull),
                *fc.build_setting_flags(info_s),
            ])
            npz_path = run_dir / f"samples_{slug}.npz"
            cmds.append(fc.conda_python(args) + [
                "evaluator_FID.py", fc.rel(fc.REFERENCE_NPZ), fc.rel(npz_path)
            ])
    fc.write_commands_sh(cmds, run_dir / "commands_fid.sh", header=f"exp_E — heavy FID n={args.full_samples} (dry-run)")
    print(f"[dry-run] FID commands → {run_dir / 'commands_fid.sh'}")


def phase_analyze(args, run_dir: Path, rows: list[dict]) -> list[dict]:
    """Light analyze: PNG count + (if log present) FID parse."""
    out_csv = run_dir / "oracle_results.csv"
    keys = ["setting", "pull_strength", "slug", "n_pngs", "fid", "sfid", "trace_path", "source_log"]
    for r in rows:
        img = fc.IEC_ROOT / r["image_folder"] if not Path(r["image_folder"]).is_absolute() else Path(r["image_folder"])
        r["n_pngs"] = fc.png_count(img)
        log_p = fc.IEC_ROOT / r["source_log"] if not Path(r["source_log"]).is_absolute() else Path(r["source_log"])
        fid, sfid = fc.parse_fid_log(log_p)
        r["fid"] = fid
        r["sfid"] = sfid
    with open(out_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    print(f"[ok] table → {out_csv}")
    return rows


def phase_plot(args, run_dir: Path, rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    rows_with_fid = [r for r in rows if r.get("fid") is not None]
    if not rows_with_fid:
        print("[plot] no FID values yet — execute heavy FID dry-run commands first")
        return
    settings = sorted({r["setting"] for r in rows_with_fid})
    pulls = sorted({float(r["pull_strength"]) for r in rows_with_fid})
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.35
    x_idx = np.arange(len(settings))
    for i, pull in enumerate(pulls):
        ys = []
        for s in settings:
            cell = next((r for r in rows_with_fid if r["setting"] == s and float(r["pull_strength"]) == pull), None)
            ys.append(cell["fid"] if cell else 0)
        ax.bar(x_idx + (i - 0.5) * width, ys, width, label=f"pull={pull}")
    ax.set_xticks(x_idx)
    ax.set_xticklabels(settings)
    ax.set_ylabel("FID")
    ax.set_title(f"{EXPERIMENT_ID} — Oracle decoder upper bound")
    ax.legend()
    fig.tight_layout()
    out = run_dir / "oracle_bound.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[ok] {out}")


def phase_summary(args, run_dir: Path, rows: list[dict]) -> None:
    md = run_dir / "summary.md"
    have_fid = sum(1 for r in rows if r.get("fid") is not None)
    md.write_text(
        "\n".join([
            f"# {EXPERIMENT_ID} — Summary",
            "",
            f"- oracle settings × pulls: {len(rows)}",
            f"- rows with FID: {have_fid}",
            "",
            "| setting | pull | n_pngs | FID | sFID |",
            "|---|---|---|---|---|",
        ] + [
            f"| {r['setting']} | {r['pull_strength']} | {r.get('n_pngs', 0)} | {r.get('fid')} | {r.get('sfid')} |"
            for r in rows
        ] + [""])
    )
    print(md.read_text())


def main():
    args = parse_args()
    run_dir = fc.resolve_results_dir(args.results_dir, plot_only=(args.phase == "plot"))
    print(f"[info] run dir: {run_dir}")

    ref_trace: Path | None = None
    rows: list[dict] = []

    if args.phase in ("inventory", "all"):
        phase_inventory(args, run_dir)
    if args.phase in ("ref-trace", "all"):
        ref_trace = phase_ref_trace(args, run_dir)
    if args.phase in ("oracle-run", "all"):
        if ref_trace is None:
            ref_trace = run_dir / "traces" / "ref_fp16_trace.pt"
        rows = phase_oracle_run(args, run_dir, ref_trace)
    if args.phase in ("fid", "all"):
        phase_fid(args, run_dir)
    if args.phase in ("analyze", "all"):
        if not rows:
            csv_path = run_dir / "oracle_results.csv"
            if csv_path.exists():
                rows = list(csv.DictReader(open(csv_path)))
                for r in rows:
                    if r.get("fid"):
                        try:
                            r["fid"] = float(r["fid"])
                        except ValueError:
                            pass
        if rows:
            phase_analyze(args, run_dir, rows)
    if args.phase in ("plot", "all"):
        if not rows:
            csv_path = run_dir / "oracle_results.csv"
            if csv_path.exists():
                rows = list(csv.DictReader(open(csv_path)))
                for r in rows:
                    if r.get("fid"):
                        try:
                            r["fid"] = float(r["fid"])
                        except ValueError:
                            pass
        if rows:
            phase_plot(args, run_dir, rows)
    if args.phase in ("summary", "all"):
        if rows:
            phase_summary(args, run_dir, rows)


if __name__ == "__main__":
    main()
