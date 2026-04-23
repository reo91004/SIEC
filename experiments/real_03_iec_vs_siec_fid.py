"""
real_03: IEC vs S-IEC FID Performance Comparison
=================================================

기존에 생성된 50K 샘플 NPZ 파일들을 사용하여
IEC / S-IEC / No-Correction baseline 간 FID를 비교합니다.

Usage:
    conda run -n iec python experiments/real_03_iec_vs_siec_fid.py

    # 소규모 샘플만으로 빠르게 테스트 (기본 2000장):
    conda run -n iec python experiments/real_03_iec_vs_siec_fid.py --quick

    # 특정 NPZ 추가 비교:
    conda run -n iec python experiments/real_03_iec_vs_siec_fid.py --extra my_samples.npz
"""
import os
import sys
import argparse
import warnings
import time
from pathlib import Path

import numpy as np
from scipy import linalg

# ─── paths ───────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # IEC/
sys.path.insert(0, str(ROOT))

NPZ_PATHS = {
    "Reference":        ROOT / "cifar10_reference.npz",
    "IEC":              ROOT / "iec_samples.npz",
    "IEC+Recon":        ROOT / "iec_samples_recon.npz",
    "S-IEC (p80)":      ROOT / "siec_samples.npz",
    "No Correction":    ROOT / "siec_never.npz",
}

SAVE_DIR = ROOT / "experiments"


# ═══════════════════════════════════════════════════════════════════════
# FID computation  (reuses evaluator_FID.py infrastructure)
# ═══════════════════════════════════════════════════════════════════════

def _load_inception_session():
    """TF InceptionV3 session (singleton-like)."""
    import tensorflow.compat.v1 as tf
    tf.logging.set_verbosity(tf.logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess


def compute_activations(sess, images, batch_size=64):
    """
    Extract InceptionV3 pool_3 features from uint8 NHWC images.
    Returns np.ndarray of shape (N, 2048).
    """
    from evaluator_FID import _create_feature_graph
    import tensorflow.compat.v1 as tf

    with sess.graph.as_default():
        ph = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        pool_feat, _ = _create_feature_graph(ph)

    feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size].astype(np.float32)
        feat = sess.run(pool_feat, {ph: batch})
        feats.append(feat.reshape(feat.shape[0], -1))
    return np.concatenate(feats, axis=0)


def compute_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Frechet distance between two Gaussians."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        warnings.warn("FID: singular product, adding eps to diagonal")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def stats_from_activations(acts):
    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)
    return mu, sigma


# ═══════════════════════════════════════════════════════════════════════
# Sample-level visual comparison
# ═══════════════════════════════════════════════════════════════════════

def make_comparison_grid(npz_dict, n_show=8, seed=42):
    """
    각 방법에서 동일 인덱스의 샘플을 뽑아 grid로 시각화합니다.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.RandomState(seed)
    # Reference를 제외한 방법들
    methods = [k for k in npz_dict if k != "Reference"]
    n_methods = len(methods)

    # 가장 작은 NPZ 기준으로 인덱스 범위 결정
    min_n = min(len(np.load(npz_dict[m])["arr_0"]) for m in methods)
    indices = rng.choice(min_n, size=n_show, replace=False)
    indices.sort()

    fig, axes = plt.subplots(n_methods, n_show, figsize=(n_show * 1.5, n_methods * 1.5))
    if n_methods == 1:
        axes = axes[np.newaxis, :]

    for row, method in enumerate(methods):
        imgs = np.load(npz_dict[method])["arr_0"]
        for col, idx in enumerate(indices):
            axes[row, col].imshow(imgs[idx])
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_title(method, fontsize=9, loc="left")

    plt.suptitle("Sample Comparison (same indices)", fontsize=12, y=1.01)
    plt.tight_layout()
    out_path = SAVE_DIR / "iec_vs_siec_samples.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Sample grid saved to {out_path}")


# ═══════════════════════════════════════════════════════════════════════
# FID bar chart
# ═══════════════════════════════════════════════════════════════════════

def plot_fid_comparison(results, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(results.keys())
    fids = [results[m] for m in methods]

    colors = []
    for m in methods:
        if "S-IEC" in m:
            colors.append("#2196F3")
        elif "IEC" in m and "Recon" in m:
            colors.append("#FF9800")
        elif "IEC" in m:
            colors.append("#4CAF50")
        else:
            colors.append("#9E9E9E")

    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 1.5), 5))
    bars = ax.bar(methods, fids, color=colors, edgecolor="black", linewidth=0.5)

    # 값 표시
    for bar, fid in zip(bars, fids):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{fid:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("FID (lower is better)", fontsize=12)
    ax.set_title("CIFAR-10: IEC vs S-IEC FID Comparison", fontsize=14, pad=15)
    ax.set_ylim(0, max(fids) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  FID chart saved to {save_path}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="IEC vs S-IEC FID comparison")
    parser.add_argument("--quick", action="store_true",
                        help="Use first 2000 samples only (faster, approximate FID)")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Number of samples to use (overrides --quick)")
    parser.add_argument("--extra", type=str, nargs="*", default=[],
                        help="Additional NPZ files to include (name:path or just path)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Inception feature extraction batch size")
    parser.add_argument("--skip_plot", action="store_true",
                        help="Skip visualization (text-only output)")
    args = parser.parse_args()

    # Determine sample count
    if args.n_samples is not None:
        max_n = args.n_samples
    elif args.quick:
        max_n = 2000
    else:
        max_n = None  # use all

    # Build NPZ dict (filter missing files)
    npz_dict = {}
    for name, path in NPZ_PATHS.items():
        if path.exists():
            npz_dict[name] = str(path)
        else:
            print(f"  [SKIP] {name}: {path} not found")

    # Add extra NPZ files
    for extra in args.extra:
        if ":" in extra:
            name, path = extra.split(":", 1)
        else:
            name = Path(extra).stem
            path = extra
        if os.path.exists(path):
            npz_dict[name] = path
        else:
            print(f"  [SKIP] {name}: {path} not found")

    if "Reference" not in npz_dict:
        print("ERROR: cifar10_reference.npz not found!")
        sys.exit(1)

    methods = [k for k in npz_dict if k != "Reference"]
    if not methods:
        print("ERROR: No sample NPZ files found!")
        sys.exit(1)

    print("=" * 60)
    print("  IEC vs S-IEC FID Comparison")
    print("=" * 60)
    print(f"  Reference: {npz_dict['Reference']}")
    for m in methods:
        n_total = len(np.load(npz_dict[m])["arr_0"])
        n_use = min(n_total, max_n) if max_n else n_total
        print(f"  {m}: {npz_dict[m]}  ({n_use}/{n_total} samples)")
    print()

    # ─── Load Inception ──────────────────────────────────────────────
    print("[1/4] Loading InceptionV3...")
    t0 = time.time()
    os.chdir(str(ROOT))  # for classify_image_graph_def.pb
    sess = _load_inception_session()
    print(f"  Done ({time.time() - t0:.1f}s)")

    # ─── Compute reference features ──────────────────────────────────
    print("[2/4] Computing reference features...")
    t0 = time.time()
    ref_images = np.load(npz_dict["Reference"])["arr_0"]
    if max_n and len(ref_images) > max_n:
        ref_images = ref_images[:max_n]
    ref_acts = compute_activations(sess, ref_images, args.batch_size)
    ref_mu, ref_sigma = stats_from_activations(ref_acts)
    n_ref = len(ref_images)
    del ref_images
    print(f"  Done: {n_ref} images ({time.time() - t0:.1f}s)")

    # ─── Compute FID for each method ─────────────────────────────────
    print("[3/4] Computing FID for each method...")
    results = {}
    for method in methods:
        t0 = time.time()
        images = np.load(npz_dict[method])["arr_0"]
        if max_n and len(images) > max_n:
            images = images[:max_n]
        n_imgs = len(images)

        acts = compute_activations(sess, images, args.batch_size)
        mu, sigma = stats_from_activations(acts)
        fid = compute_fid(ref_mu, ref_sigma, mu, sigma)
        results[method] = fid
        del images, acts
        print(f"  {method:20s}  FID = {fid:8.2f}  ({n_imgs} samples, {time.time() - t0:.1f}s)")

    # ─── Results table ────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  {'Method':20s}  {'FID':>10s}  {'vs IEC':>10s}")
    print("-" * 60)

    iec_fid = results.get("IEC", None)
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    for method, fid in sorted_results:
        if iec_fid is not None and method != "IEC":
            delta = fid - iec_fid
            delta_str = f"{delta:+.2f}"
        else:
            delta_str = "baseline"
        print(f"  {method:20s}  {fid:10.2f}  {delta_str:>10s}")
    print("=" * 60)

    # Best method
    best_method, best_fid = sorted_results[0]
    print(f"\n  Best: {best_method} (FID = {best_fid:.2f})")

    # ─── Visualization ────────────────────────────────────────────────
    if not args.skip_plot:
        print("\n[4/4] Generating visualizations...")
        plot_fid_comparison(results, SAVE_DIR / "iec_vs_siec_fid.png")
        make_comparison_grid(npz_dict)
    else:
        print("\n[4/4] Skipping visualization (--skip_plot)")

    # ─── Save results to text ─────────────────────────────────────────
    results_path = SAVE_DIR / "iec_vs_siec_fid_results.txt"
    with open(results_path, "w") as f:
        f.write("IEC vs S-IEC FID Comparison\n")
        f.write(f"Samples per method: {max_n or 'all (50K)'}\n\n")
        f.write(f"{'Method':20s}  {'FID':>10s}\n")
        f.write("-" * 35 + "\n")
        for method, fid in sorted_results:
            f.write(f"{method:20s}  {fid:10.2f}\n")
    print(f"  Results saved to {results_path}")

    sess.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
