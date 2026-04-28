#!/usr/bin/env python3
"""exp_C_figure1 — Framing experiment **C**: Figure 1 후보 (syndrome ↔ FID).

`docs/siec_ecc_framing_20260427.md` §3 의 4-순위 실험. 6 setting × 4 method
의 (mean ‖syndrome‖, final FID) scatter 를 그려, 단일 monotone curve 가
나오면 ICLR Figure 1 으로 채택한다.

**Note** : 이 실험은 새 sampling 을 하지 않고, 기존 `real_05_robustness` 의
trace.pt + results.csv 를 ingest 한다. 누락된 (setting × method) 쌍은
heavy-FID dry-run command 만 생성한다.

Phases :
    inventory       # real_05 results.csv + trace.pt 자산 점검
    ingest          # results.csv + trace.pt → figure1_data.csv (per-row syndrome_mean)
    enrich          # 누락 row 의 sampling/FID command 만 dry-run 으로 생성
    plot            # scatter (PNG + PDF)
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

EXPERIMENT_ID = "exp_C_figure1"
DEFAULT_RESULTS_BASE = fc.RESULTS_BASE / EXPERIMENT_ID
SOURCE_DIR = fc.RESULTS_BASE / "real_05_robustness"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=["inventory", "ingest", "enrich", "plot", "summary", "all"], default="inventory")
    p.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--source", type=Path, default=None,
                   help="Specific real_05 run dir to ingest (default: latest with results.csv).")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_BASE)
    p.add_argument("--cuda-visible-devices", default=None)
    p.add_argument("--methods", default="no_correction,iec,siec",
                   help="Methods to keep on the figure (comma-separated).")
    return p.parse_args()


def find_latest_source(explicit: Path | None) -> Path:
    if explicit:
        if (explicit / "results.csv").exists():
            return explicit
        raise SystemExit(f"--source has no results.csv: {explicit}")
    cands = sorted(p for p in SOURCE_DIR.iterdir() if p.is_dir() and (p / "results.csv").exists())
    if not cands:
        raise SystemExit(f"no real_05 run-dir with results.csv under {SOURCE_DIR}")
    return cands[-1]


def phase_inventory(args, run_dir: Path) -> Path:
    src = find_latest_source(args.source)
    csv_path = src / "results.csv"
    n_rows = sum(1 for _ in open(csv_path)) - 1
    out_md = run_dir / "inventory.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(
        "\n".join([
            f"# {EXPERIMENT_ID} — Inventory",
            "",
            f"- source run dir: `{fc.rel(src)}`",
            f"- rows in results.csv: **{n_rows}**",
            f"- methods kept on plot: `{args.methods}`",
            "",
        ])
    )
    print(out_md.read_text())
    return src


def load_score_mean(trace_rel: str) -> float | None:
    """Read `trace_rel` (path relative to IEC root), compute mean √score over (N,T).

    Returns None if torch is unavailable (e.g. system Python without env activation) or
    if the trace cannot be loaded — recompute is best-effort, the source CSV's
    syndrome_mean column is the primary signal.
    """
    if not trace_rel:
        return None
    p = fc.IEC_ROOT / trace_rel
    if not p.exists():
        return None
    try:
        import numpy as np
        traces = fc.load_traces(p)
    except (ImportError, ModuleNotFoundError, RuntimeError) as e:
        print(f"[warn] cannot recompute score from {trace_rel}: {e}")
        return None
    arr = fc.stack_score_values(traces)
    if arr.size == 0:
        return None
    s = np.sqrt(np.maximum(arr, 0.0))
    mask = arr > 0
    if mask.any():
        return float(s[mask].mean())
    return float(s.mean())


def phase_ingest(args, run_dir: Path, src: Path) -> list[dict]:
    methods_keep = {m.strip() for m in args.methods.split(",")}
    rows_in: list[dict] = list(csv.DictReader(open(src / "results.csv")))
    out_rows: list[dict] = []
    for r in rows_in:
        method_key = r.get("method_key") or r.get("method")
        if method_key not in methods_keep and (r.get("method", "") not in methods_keep):
            continue
        if (r.get("status") or "").strip() in {"blocked", "skip"}:
            continue
        try:
            fid = float(r["fid"]) if r.get("fid") else None
        except ValueError:
            fid = None
        try:
            sfid = float(r["sfid"]) if r.get("sfid") else None
        except ValueError:
            sfid = None
        try:
            syn_mean = float(r["syndrome_mean"]) if r.get("syndrome_mean") else None
        except ValueError:
            syn_mean = None
        # If truly missing (None), recompute from trace.pt. A literal 0.0 is kept
        # because fp16 / no_correction rows legitimately produce zero syndrome.
        if syn_mean is None and r.get("trace_path"):
            recomputed = load_score_mean(r["trace_path"])
            if recomputed is not None:
                syn_mean = recomputed
        if fid is None or syn_mean is None:
            continue  # incomplete row
        out_rows.append({
            "setting": r.get("setting"),
            "method_key": method_key,
            "method": r.get("method"),
            "fid": fid,
            "sfid": sfid,
            "syndrome_mean": syn_mean,
            "per_sample_nfe": r.get("per_sample_nfe"),
            "trigger_rate": r.get("trigger_rate"),
            "trace_path": r.get("trace_path"),
            "source_npz": r.get("source_npz"),
        })
    out_csv = run_dir / "figure1_data.csv"
    keys = ["setting", "method_key", "method", "fid", "sfid", "syndrome_mean", "per_sample_nfe", "trigger_rate", "trace_path", "source_npz"]
    with open(out_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k) for k in keys})
    print(f"[ok] ingested {len(out_rows)} rows → {out_csv}")
    return out_rows


def phase_enrich(args, run_dir: Path, src: Path, rows: list[dict]) -> None:
    """Generate dry-run sampling/FID commands for any (setting × method) missing FID or syndrome."""
    have = {(r["setting"], r["method_key"]) for r in rows}
    target_settings = list(fc.setting_defs().keys())
    target_methods = ["no_correction", "iec", "siec"]
    cmds: list[list[str]] = []
    missing = []
    for setting in target_settings:
        for m in target_methods:
            if (setting, m) in have:
                continue
            info = fc.setting_defs()[setting]
            sample_cmd = fc.conda_python(args) + [
                fc.entry_script("ddim_cifar_siec.py"),
                "--correction-mode", {"no_correction": "none", "iec": "iec", "siec": "siec"}[m],
                "--num_samples", "2000",
                "--sample_batch", "500",
                "--seed", str(fc.DEFAULT_SEED),
                "--image_folder", f"error_dec/cifar/image_expC_{setting}_{m}_n2000",
                "--siec_return_trace",
                "--siec_trace_mode", {"no_correction": "none", "iec": "iec", "siec": "siec"}[m],
                "--siec_trace_out", fc.rel(run_dir / "traces" / f"{setting}_{m}.pt"),
                *fc.build_setting_flags(info),
            ]
            if m == "siec":
                tau = fc.tau_schedule_path(setting, 80.0)
                sample_cmd += ["--tau_path", fc.rel(tau), "--tau_percentile", "80"]
            cmds.append(sample_cmd)
            missing.append((setting, m))
    fc.write_commands_sh(cmds, run_dir / "commands_enrich.sh", header=f"exp_C — enrich (missing {len(missing)} cells)")
    print(f"[dry-run] {len(cmds)} enrichment commands → {run_dir / 'commands_enrich.sh'}")
    if missing:
        print("[info] missing cells:", missing)


def phase_plot(args, run_dir: Path, rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        print("[plot] no rows to plot")
        return
    settings_order = ["fp16", "w8a8", "dc10", "w4a8", "dc20", "cachequant"]
    setting_color = {s: c for s, c in zip(settings_order, ["#444444", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728"])}
    method_marker = {"no_correction": "o", "iec": "s", "siec": "^"}
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in rows:
        s = r["setting"]
        m = r["method_key"]
        color = setting_color.get(s, "gray")
        marker = method_marker.get(m, "x")
        x = float(r["syndrome_mean"])
        y = float(r["fid"])
        ax.scatter(x, y, s=60, color=color, marker=marker, alpha=0.85, label=f"{s} ({m})")
    ax.set_xscale("log")
    ax.set_xlabel("mean ‖syndrome‖ over (steps × samples)")
    ax.set_ylabel("Final FID")
    ax.set_title(f"{EXPERIMENT_ID} — Figure 1 candidate")
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    out_png = run_dir / "figure1.png"
    out_pdf = run_dir / "figure1.pdf"
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[ok] {out_png} + {out_pdf}")


def phase_summary(args, run_dir: Path, rows: list[dict]) -> None:
    md = run_dir / "summary.md"
    have = {(r["setting"], r["method_key"]) for r in rows}
    n_complete = len(rows)
    coverage = f"{n_complete}/{len(fc.setting_defs()) * 3}"  # 6 settings × 3 methods
    md.write_text(
        "\n".join([
            f"# {EXPERIMENT_ID} — Summary",
            "",
            f"- complete cells (setting × method): **{coverage}**",
            f"- methods kept: `{args.methods}`",
            "",
            "| setting | method | FID | mean‖s‖ |",
            "|---|---|---|---|",
        ] + [
            f"| {r['setting']} | {r['method_key']} | {r['fid']:.3f} | {r['syndrome_mean']:.4f} |"
            for r in rows
        ] + [""])
    )
    print(md.read_text())


def main():
    args = parse_args()
    run_dir = fc.resolve_results_dir(args.results_dir, plot_only=(args.phase == "plot"))
    print(f"[info] run dir: {run_dir}")

    src = phase_inventory(args, run_dir) if args.phase in ("inventory", "ingest", "enrich", "all") else None

    rows: list[dict] = []
    if args.phase in ("ingest", "all"):
        rows = phase_ingest(args, run_dir, src)
    if args.phase in ("enrich", "all"):
        if not rows:
            csv_path = run_dir / "figure1_data.csv"
            if csv_path.exists():
                rows = list(csv.DictReader(open(csv_path)))
        phase_enrich(args, run_dir, src, rows)
    if args.phase in ("plot", "all"):
        if not rows:
            csv_path = run_dir / "figure1_data.csv"
            if csv_path.exists():
                rows = list(csv.DictReader(open(csv_path)))
        if rows:
            phase_plot(args, run_dir, rows)
    if args.phase in ("summary", "all"):
        if rows:
            phase_summary(args, run_dir, rows)


if __name__ == "__main__":
    main()
