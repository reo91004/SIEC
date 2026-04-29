"""Phase 5-4: cheap predictor for expensive correction gain.

Reads gain_per_step.csv from Phase 5-3 and fits two predictors of g_t:
1. Linear: g_t ≈ a * z_t + b   (z_t = syndrome score; raw/mean/calibrated)
2. Isotonic: monotone fit g_t = f(z_t)

Outputs predictor.pt with coefficients usable downstream and a small report.

Note: trains on (syndrome, gain) pairs aggregated across step indices in the
csv. For per-step coefficients (a_t, b_t) use --per-step (requires more
samples per step than this scaffold provides).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch


def load_csv(path: Path):
    rows = []
    with path.open() as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) if v not in ("", None) else float("nan") for k, v in r.items()})
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gain-csv", required=True)
    p.add_argument("--score-key", default="syndrome_score",
                   choices=["syndrome_score", "oracle_score"])
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows = load_csv(Path(args.gain_csv))
    z = np.array([r[args.score_key] for r in rows], dtype=np.float64)
    g = np.array([r["g_mean"] for r in rows], dtype=np.float64)
    valid = ~np.isnan(z) & ~np.isnan(g)
    z = z[valid]
    g = g[valid]
    if z.size < 5:
        raise SystemExit("not enough valid rows for fitting")

    # Linear least-squares
    A = np.stack([z, np.ones_like(z)], axis=1)
    a, b = np.linalg.lstsq(A, g, rcond=None)[0]
    g_lin = a * z + b
    rss_lin = float(((g - g_lin) ** 2).sum())
    tss = float(((g - g.mean()) ** 2).sum())
    r2_lin = 1.0 - rss_lin / max(tss, 1e-12)

    # Isotonic
    try:
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip", increasing="auto")
        iso.fit(z, g)
        g_iso = iso.predict(z)
        rss_iso = float(((g - g_iso) ** 2).sum())
        r2_iso = 1.0 - rss_iso / max(tss, 1e-12)
        iso_x = iso.X_thresholds_.tolist()
        iso_y = iso.y_thresholds_.tolist()
    except Exception as e:
        iso = None
        r2_iso = float("nan")
        iso_x = iso_y = []

    # Spearman ranks for sanity
    try:
        from scipy.stats import spearmanr
        sp = spearmanr(z, g)
        sp_corr, sp_p = float(sp.correlation), float(sp.pvalue)
    except Exception:
        sp_corr = sp_p = float("nan")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "version": 1,
        "kind": "cheap_predictor",
        "score_key": args.score_key,
        "linear": {"a": float(a), "b": float(b), "r2": r2_lin},
        "isotonic": {"x_thresholds": iso_x, "y_thresholds": iso_y, "r2": r2_iso},
        "spearman": {"corr": sp_corr, "p": sp_p},
        "n_rows": int(valid.sum()),
    }
    torch.save(artifact, out)

    report_path = out.with_suffix(".json")
    with report_path.open("w") as f:
        json.dump(artifact, f, indent=2)

    print(f"saved {out}")
    print(f"report {report_path}")
    print(f"linear: a={a:.4f} b={b:.4f} R^2={r2_lin:.4f}")
    print(f"isotonic: R^2={r2_iso:.4f}  (n_pieces={len(iso_x)})")
    print(f"spearman: corr={sp_corr:.4f} p={sp_p:.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
