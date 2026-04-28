#!/usr/bin/env python3
"""exp_F_cross_error — Framing experiment **F**: cross-family decoder transfer.

`docs/siec_ecc_framing_20260427.md` §3 의 5-순위 실험 (dry-run only).

가정 검증 :
  family A 에서 보정 (τ, γ) 를 캘리브레이션해 family B 에 그대로 적용해도
  no-correction 대비 FID 가 개선되는가?
  - 통과 → universal decoder (ECC framing 의 transfer 보장).
  - 실패 → family-conditional decoder 가 필요.

설계 :
- 각 cross-pair (calib_setting, deploy_setting) 에 대해
  `--tau_path calibration/tau_schedule_<calib>_p80.pt` 를 deploy 환경에 적용.
- n=2000 sampling + FID 명령을 commands.sh 로만 생성 (dry-run).
- 실측 FID 가 들어오면 cross_error_grid.csv 와 heatmap 으로 시각화.

Phases :
    inventory     # 모든 family 의 tau / pilot 자산 점검
    plan          # cross-pair list + commands.sh 생성 (dry-run)
    execute-fid   # sample + pngs→npz + evaluator_FID 까지 실측 (full 모드용)
    analyze       # log → FID ingest (실측 후)
    plot          # heatmap (rows: deploy, cols: calib source)
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

EXPERIMENT_ID = "exp_F_cross_error"
DEFAULT_RESULTS_BASE = fc.RESULTS_BASE / EXPERIMENT_ID
CROSS_PAIRS = [
    ("w8a8", "dc10"),
    ("dc10", "w8a8"),
    ("w8a8", "cachequant"),
    ("cachequant", "w8a8"),
    ("dc10", "cachequant"),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=["inventory", "plan", "execute-fid", "analyze", "plot", "summary", "all"], default="inventory")
    p.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--sample-batch", type=int, default=500)
    p.add_argument("--seed", type=int, default=fc.DEFAULT_SEED)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_BASE)
    p.add_argument("--cuda-visible-devices", default=None)
    p.add_argument("--pairs", default="|".join(",".join(p) for p in CROSS_PAIRS),
                   help="Pipe-separated pairs as 'calib,deploy|calib,deploy|...'.")
    return p.parse_args()


def parse_pairs(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for chunk in text.split("|"):
        a, b = chunk.split(",")
        out.append((a.strip(), b.strip()))
    return out


def phase_inventory(args, run_dir: Path) -> None:
    pairs = parse_pairs(args.pairs)
    settings_used = sorted({s for pair in pairs for s in pair})
    rows = []
    missing_total = []
    for s in settings_used:
        info = fc.setting_defs()[s]
        miss_assets = [p for p in info["required_assets"] if not (fc.IEC_ROOT / p).exists()]
        tau = fc.tau_schedule_path(s, 80.0)
        miss_tau = bool(not tau.exists())
        if miss_assets or miss_tau:
            missing_total.append((s, miss_assets, str(tau) if miss_tau else None))
        rows.append((s, str(tau), tau.exists(), info["description"]))
    out_md = run_dir / "inventory.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(
        "\n".join([
            f"# {EXPERIMENT_ID} — Inventory",
            "",
            f"- cross-pairs (calib → deploy): `{pairs}`",
            "",
            "| setting | tau path | exists | description |",
            "|---|---|---|---|",
        ] + [f"| {s} | `{t}` | {e} | {d} |" for (s, t, e, d) in rows]) + "\n"
    )
    print(out_md.read_text())
    if missing_total:
        raise SystemExit(f"[FAIL] missing for cross-pairs: {missing_total}")


def build_cross_cmd(args, calib: str, deploy: str, run_dir: Path) -> tuple[list[str], dict]:
    info = fc.setting_defs()[deploy]
    tau_calib = fc.tau_schedule_path(calib, 80.0)
    slug = f"calib_{calib}_deploy_{deploy}"
    image_folder = fc.ERROR_DEC / f"image_expF_{slug}_n{args.num_samples}"
    npz_path = run_dir / f"samples_{slug}.npz"
    log_sample = run_dir / "logs" / f"sample_{slug}.log"
    log_fid = run_dir / "logs" / f"fid_{slug}.log"
    cmd = fc.conda_python(args) + [
        fc.entry_script("ddim_cifar_siec.py"),
        "--correction-mode", "siec",
        "--num_samples", str(args.num_samples),
        "--sample_batch", str(args.sample_batch),
        "--seed", str(args.seed),
        "--image_folder", fc.rel(image_folder),
        "--tau_path", fc.rel(tau_calib),
        "--tau_percentile", "80",
        *fc.build_setting_flags(info),
    ]
    info_dict = {
        "calib_setting": calib,
        "deploy_setting": deploy,
        "slug": slug,
        "tau_path": fc.rel(tau_calib),
        "image_folder": fc.rel(image_folder),
        "source_npz": fc.rel(npz_path),
        "source_log": fc.rel(log_sample),
        "fid_log": fc.rel(log_fid),
    }
    return cmd, info_dict


def phase_plan(args, run_dir: Path) -> list[dict]:
    pairs = parse_pairs(args.pairs)
    cmds = []
    rows = []
    for calib, deploy in pairs:
        sample_cmd, info = build_cross_cmd(args, calib, deploy, run_dir)
        # FID command (executed manually after sampling).
        fid_cmd = fc.conda_python(args) + [
            "evaluator_FID.py", fc.rel(fc.REFERENCE_NPZ), info["source_npz"],
        ]
        cmds.append(sample_cmd)
        cmds.append(fid_cmd)
        rows.append(info)
    fc.write_commands_sh(cmds, run_dir / "commands_cross.sh", header=f"exp_F — {len(rows)} cross pairs (sampling + FID)")
    csv_path = run_dir / "cross_pairs.csv"
    keys = ["calib_setting", "deploy_setting", "slug", "tau_path", "image_folder", "source_npz", "source_log", "fid_log"]
    with open(csv_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})
    print(f"[dry-run] {len(cmds)} cross commands → {run_dir / 'commands_cross.sh'}")
    print(f"[ok] cross plan → {csv_path}")
    return rows


def phase_execute_fid(args, run_dir: Path, rows: list[dict]) -> list[dict]:
    """Actually run sample + pngs→npz + evaluator_FID for every cross pair.

    Idempotent: skips a cell if its npz and fid log already exist.
    """
    for r in rows:
        slug = r["slug"]
        image_folder = fc.IEC_ROOT / r["image_folder"]
        npz_path = run_dir / f"samples_{slug}.npz"
        sample_log = run_dir / "logs" / f"sample_{slug}.log"
        fid_log = run_dir / "logs" / f"fid_{slug}.log"
        # rebuild sample command (we need it again — phase_plan only kept the dict).
        sample_cmd, _ = build_cross_cmd(args, r["calib_setting"], r["deploy_setting"], run_dir)

        # 1) sample if PNGs not present (or insufficient)
        if fc.png_count(image_folder) < args.num_samples:
            elapsed = fc.run_cmd(sample_cmd, sample_log)
            r["sampling_wall_clock_sec"] = round(elapsed, 1)
        else:
            print(f"[skip] sampling for {slug} ({fc.png_count(image_folder)} pngs already)")
        # 2) pngs → npz
        if not npz_path.exists():
            n = fc.pngs_to_npz(image_folder, npz_path)
            print(f"[ok] {slug}: stacked {n} pngs → {fc.rel(npz_path)}")
        # 3) FID
        if not fid_log.exists() or fc.parse_fid_log(fid_log) == (None, None):
            fid, sfid, elapsed = fc.run_fid(args, npz_path, fid_log)
            r["fid_wall_clock_sec"] = round(elapsed, 1)
        fid, sfid = fc.parse_fid_log(fid_log)
        r["fid"] = fid
        r["sfid"] = sfid
        r["source_npz"] = fc.rel(npz_path)
        r["fid_log"] = fc.rel(fid_log)
        r["source_log"] = fc.rel(sample_log)
        print(f"[result] {slug}: FID={fid} sFID={sfid}")
    return rows


def phase_analyze(args, run_dir: Path, rows: list[dict]) -> list[dict]:
    """Read fid logs (if present) and emit cross_error_grid.csv."""
    out: list[dict] = []
    for r in rows:
        log_path = fc.IEC_ROOT / r["fid_log"] if not Path(r["fid_log"]).is_absolute() else Path(r["fid_log"])
        fid, sfid = fc.parse_fid_log(log_path)
        out.append({
            **r,
            "fid": fid if r.get("fid") is None else r["fid"],
            "sfid": sfid if r.get("sfid") is None else r["sfid"],
        })
    csv_path = run_dir / "cross_error_grid.csv"
    keys = ["calib_setting", "deploy_setting", "slug", "fid", "sfid", "tau_path", "fid_log"]
    with open(csv_path, "w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in out:
            w.writerow({k: r.get(k) for k in keys})
    print(f"[ok] grid → {csv_path}")
    return out


def phase_plot(args, run_dir: Path, rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    have = [r for r in rows if r.get("fid") is not None]
    if not have:
        print("[plot] no FID values yet — execute commands_cross.sh first")
        return
    deploys = sorted({r["deploy_setting"] for r in have})
    calibs = sorted({r["calib_setting"] for r in have})
    grid = np.full((len(deploys), len(calibs)), np.nan)
    for r in have:
        i = deploys.index(r["deploy_setting"])
        j = calibs.index(r["calib_setting"])
        grid[i, j] = float(r["fid"])
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(grid, cmap="viridis")
    ax.set_xticks(range(len(calibs)))
    ax.set_xticklabels(calibs)
    ax.set_yticks(range(len(deploys)))
    ax.set_yticklabels(deploys)
    ax.set_xlabel("calibration source")
    ax.set_ylabel("deploy environment")
    for i in range(len(deploys)):
        for j in range(len(calibs)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, label="FID")
    ax.set_title(f"{EXPERIMENT_ID} — cross-family decoder transfer")
    fig.tight_layout()
    out = run_dir / "cross_error_heatmap.png"
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
            f"- planned cross pairs: {len(rows)}",
            f"- pairs with FID: {have_fid}",
            "",
            "| calib | deploy | FID | sFID |",
            "|---|---|---|---|",
        ] + [
            f"| {r['calib_setting']} | {r['deploy_setting']} | {r.get('fid')} | {r.get('sfid')} |"
            for r in rows
        ] + [""])
    )
    print(md.read_text())


def main():
    args = parse_args()
    run_dir = fc.resolve_results_dir(args.results_dir, plot_only=(args.phase == "plot"))
    print(f"[info] run dir: {run_dir}")
    rows: list[dict] = []
    if args.phase in ("inventory", "all"):
        phase_inventory(args, run_dir)
    if args.phase in ("plan", "execute-fid", "all"):
        rows = phase_plan(args, run_dir)
    # Run sample → npz → FID in either an explicit `execute-fid` invocation or
    # an `all` invocation that opted out of dry-run (e.g. run_all.sh full).
    if args.phase in ("execute-fid", "all") and not args.dry_run:
        rows = phase_execute_fid(args, run_dir, rows)
    if args.phase in ("analyze", "all"):
        if not rows:
            csv_path = run_dir / "cross_pairs.csv"
            if csv_path.exists():
                rows = list(csv.DictReader(open(csv_path)))
        if rows:
            rows = phase_analyze(args, run_dir, rows)
    if args.phase in ("plot", "all"):
        if not rows:
            csv_path = run_dir / "cross_error_grid.csv"
            if csv_path.exists():
                rows = list(csv.DictReader(open(csv_path)))
        if rows:
            phase_plot(args, run_dir, rows)
    if args.phase in ("summary", "all"):
        if not rows:
            csv_path = run_dir / "cross_error_grid.csv"
            if csv_path.exists():
                rows = list(csv.DictReader(open(csv_path)))
        if rows:
            phase_summary(args, run_dir, rows)


if __name__ == "__main__":
    main()
