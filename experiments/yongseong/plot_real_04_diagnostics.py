#!/usr/bin/env python3
"""Diagnostic plots for real_04_tradeoff results.

This script is intentionally separate from the main wrapper. The existing
`tradeoff_2panel.png` is a paper-style summary figure; this script produces
engineering-facing diagnostics that make failure modes obvious:

- fresh 2K runs collapsing to the same NFE
- selectivity comparisons at matched trigger rates
- seed-vs-fresh mismatches
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


IEC_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = IEC_ROOT / "experiments/yongseong/results/real_04_tradeoff"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    return p.parse_args()


def load_rows(path: Path) -> list[dict]:
    with open(path) as f:
        raw = list(csv.DictReader(f))

    def cast(v: str | None):
        if v in (None, "", "None"):
            return None
        try:
            return float(v) if "." in v else int(v)
        except ValueError:
            return v

    return [{k: cast(v) for k, v in row.items()} for row in raw]


def family(method: str) -> str:
    if method.startswith("S-IEC p") and "seed" not in method:
        return "siec_sweep"
    if method.startswith("S-IEC always-on"):
        return "always_on"
    if method.startswith("Random matched"):
        return "random_matched"
    if method.startswith("Uniform matched"):
        return "uniform_matched"
    if method.startswith("Random ("):
        return "random_grid"
    if method.startswith("Uniform ("):
        return "uniform_grid"
    if method.startswith("IEC (author)"):
        return "iec"
    if method.startswith("No correction"):
        return "no_correction"
    if method.startswith("S-IEC p80 (seed"):
        return "seed_siec"
    if method.startswith("IEC+Recon"):
        return "iec_recon"
    return "other"


def short_label(row: dict) -> str:
    method = row["method"]
    if method.startswith("S-IEC p") and row.get("tau_percentile") is not None:
        return f"p{int(row['tau_percentile'])}"
    if "period=" in method:
        return method.split("period=")[-1].rstrip(")")
    if "prob=" in method:
        return method.split("prob=")[-1].rstrip(")")
    if method.startswith("Random (p="):
        return method.split("p=")[-1].rstrip(")")
    return method


def write_summary(rows: list[dict], out_md: Path) -> None:
    fresh = [r for r in rows if r.get("num_samples") == 2000 and r.get("fid") is not None]
    fresh_nfes = sorted({r["per_sample_nfe"] for r in fresh if r.get("per_sample_nfe") is not None})
    siec_fresh = [r for r in fresh if family(r["method"]) == "siec_sweep"]
    random_matched = [r for r in fresh if family(r["method"]) == "random_matched"]
    uniform_matched = [r for r in fresh if family(r["method"]) == "uniform_matched"]
    always_on = next((r for r in fresh if family(r["method"]) == "always_on"), None)
    no_corr = next((r for r in rows if r["method"] == "No correction (never)"), None)
    iec_seed = next((r for r in rows if r["method"] == "IEC (author)"), None)
    siec_seed = next((r for r in rows if r["method"] == "S-IEC p80 (seed 50K)"), None)

    best_siec = min(siec_fresh, key=lambda r: r["fid"]) if siec_fresh else None
    best_random = min(random_matched, key=lambda r: r["fid"]) if random_matched else None
    best_uniform = min(uniform_matched, key=lambda r: r["fid"]) if uniform_matched else None

    lines = [
        "# real_04_tradeoff Diagnostic Summary",
        "",
        "This summary is diagnostic, not paper-facing.",
        "",
        "## Immediate findings",
        "",
        f"- Fresh 2K rows use these unique per-sample NFE values: {fresh_nfes}",
        "- If the list above contains only `110`, the compute-quality axis has collapsed.",
    ]
    if no_corr is not None and best_siec is not None:
        lines.append(
            f"- Best fresh S-IEC FID = {best_siec['fid']:.4f} "
            f"({best_siec['method']}) vs No correction = {no_corr['fid']:.4f}."
        )
    if always_on is not None and best_siec is not None:
        lines.append(
            f"- Always-on FID = {always_on['fid']:.4f}; "
            f"best fresh S-IEC delta = {best_siec['fid'] - always_on['fid']:+.4f}."
        )
    if best_random is not None and best_siec is not None:
        lines.append(
            f"- Best matched-random FID = {best_random['fid']:.4f}; "
            f"best fresh S-IEC delta = {best_siec['fid'] - best_random['fid']:+.4f}."
        )
    if best_uniform is not None and best_siec is not None:
        lines.append(
            f"- Best matched-uniform FID = {best_uniform['fid']:.4f}; "
            f"best fresh S-IEC delta = {best_siec['fid'] - best_uniform['fid']:+.4f}."
        )
    if iec_seed is not None and siec_seed is not None:
        lines += [
            "",
            "## Seed mismatch",
            "",
            f"- IEC 50K seed FID = {iec_seed['fid']:.4f}",
            f"- S-IEC p80 50K seed FID = {siec_seed['fid']:.4f}",
            "- These seed rows should not be mixed with fresh 2K rows when making compute-matched claims.",
        ]

    out_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.results_dir / "results.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "siec_sweep": "tab:blue",
        "always_on": "tab:red",
        "random_matched": "tab:green",
        "uniform_matched": "tab:purple",
        "random_grid": "tab:olive",
        "uniform_grid": "tab:brown",
        "iec": "tab:orange",
        "no_correction": "tab:gray",
        "seed_siec": "tab:cyan",
        "iec_recon": "tab:pink",
        "other": "black",
    }

    fresh = [r for r in rows if r.get("num_samples") == 2000 and r.get("fid") is not None]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: fresh 2K FID by family/category.
    order = [
        "no_correction",
        "siec_sweep",
        "always_on",
        "random_matched",
        "uniform_matched",
        "random_grid",
        "uniform_grid",
    ]
    xpos = {k: i for i, k in enumerate(order)}
    for row in fresh:
        fam = family(row["method"])
        if fam not in xpos:
            continue
        x = xpos[fam]
        axes[0, 0].scatter(x, row["fid"], color=colors[fam], s=55, alpha=0.9)
        axes[0, 0].annotate(
            short_label(row), (x, row["fid"]),
            textcoords="offset points", xytext=(4, 4), fontsize=8
        )
    axes[0, 0].set_xticks(range(len(order)))
    axes[0, 0].set_xticklabels(
        ["No corr", "S-IEC", "Always-on", "Rand m.", "Unif m.", "Rand grid", "Unif grid"],
        rotation=20, ha="right"
    )
    axes[0, 0].set_ylabel("FID")
    axes[0, 0].set_title("Fresh 2K FID by Family")
    axes[0, 0].grid(alpha=0.3)

    # Panel 2: NFE collapse check.
    for row in fresh:
        fam = family(row["method"])
        if row.get("per_sample_nfe") is None:
            continue
        axes[0, 1].scatter(row["per_sample_nfe"], row["fid"], color=colors[fam], s=55, alpha=0.9)
        axes[0, 1].annotate(
            short_label(row), (row["per_sample_nfe"], row["fid"]),
            textcoords="offset points", xytext=(4, 4), fontsize=8
        )
    unique_nfes = sorted({r["per_sample_nfe"] for r in fresh if r.get("per_sample_nfe") is not None})
    axes[0, 1].set_xlabel("Per-sample NFE")
    axes[0, 1].set_ylabel("FID")
    axes[0, 1].set_title(f"NFE Collapse Check (unique NFE: {unique_nfes})")
    axes[0, 1].grid(alpha=0.3)

    # Panel 3: trigger-rate selectivity among comparable methods.
    for fam in ("siec_sweep", "random_matched", "uniform_matched"):
        subset = [r for r in fresh if family(r["method"]) == fam and r.get("trigger_rate") is not None]
        subset.sort(key=lambda r: (r["trigger_rate"], r["fid"]))
        if not subset:
            continue
        axes[1, 0].plot(
            [r["trigger_rate"] for r in subset],
            [r["fid"] for r in subset],
            "o-",
            color=colors[fam],
            label={
                "siec_sweep": "S-IEC",
                "random_matched": "Random matched",
                "uniform_matched": "Uniform matched",
            }[fam],
        )
    axes[1, 0].set_xlabel("Trigger rate")
    axes[1, 0].set_ylabel("FID")
    axes[1, 0].set_title("Selectivity Check at Comparable Trigger Rates")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    # Panel 4: seed-vs-fresh mismatch.
    compare = []
    for row in rows:
        if row.get("fid") is None:
            continue
        if row["method"] in ("IEC (author)", "S-IEC p80 (seed 50K)", "No correction (never)", "S-IEC p80"):
            compare.append(row)
    for row in compare:
        fam = family(row["method"])
        axes[1, 1].scatter(row["num_samples"], row["fid"], color=colors[fam], s=70, alpha=0.95)
        axes[1, 1].annotate(
            row["method"], (row["num_samples"], row["fid"]),
            textcoords="offset points", xytext=(5, 5), fontsize=8
        )
    axes[1, 1].set_xlabel("num_samples")
    axes[1, 1].set_ylabel("FID")
    axes[1, 1].set_title("Seed vs Fresh Rows Should Not Be Merged")
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle("real_04_tradeoff Diagnostics", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_png = args.results_dir / "tradeoff_diagnostic.png"
    out_pdf = args.results_dir / "tradeoff_diagnostic.pdf"
    fig.savefig(out_png, dpi=160)
    fig.savefig(out_pdf)
    plt.close(fig)

    write_summary(rows, args.results_dir / "diagnostic_summary.md")
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {args.results_dir / 'diagnostic_summary.md'}")


if __name__ == "__main__":
    main()
