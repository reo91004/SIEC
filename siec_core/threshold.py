import numpy as np
import torch


def calibrate_tau_from_scores(scores_by_t, percentile=80.0):
    """
    Convert collected syndrome scores into threshold schedule τ_t.
    
    Pilot run은 IEC repo의 sampling loop 안에서 수행됨 (CacheQuant/quant 상태
    관리 때문). 여기서는 수집된 score만 받아서 percentile 변환.
    
    Args:
        scores_by_t: list of length T, each entry is list/array of scores
                     (from pilot trajectories at that timestep)
        percentile:  0-100 scale, e.g., 80 means trigger when score is in top 20%
    Returns:
        tau: np.ndarray of shape (T,)
    """
    T = len(scores_by_t)
    tau = np.zeros(T, dtype=np.float64)
    for t in range(T):
        if len(scores_by_t[t]) > 0:
            tau[t] = np.percentile(scores_by_t[t], percentile)
    return tau


def collect_scores_from_trajectory(syndrome_scores_per_t, scores_by_t):
    """
    Update per-timestep score lists during pilot run.
    
    Args:
        syndrome_scores_per_t: list[float] of length T, scores from ONE trajectory
                               (output of `collect_scores=True` in SIEC sampling)
        scores_by_t: running accumulator, list of length T (in-place update)
    """
    for t, s in enumerate(syndrome_scores_per_t):
        scores_by_t[t].append(s)