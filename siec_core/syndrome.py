import torch


def _select_stat(stats, keys, step_idx, device, dtype):
    if stats is None:
        return None
    for key in keys:
        if key not in stats:
            continue
        value = stats[key]
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value)
        if value.ndim > 0 and step_idx is not None and value.shape[0] > step_idx:
            value = value[step_idx]
        return value.to(device=device, dtype=dtype)
    return None


def _broadcast_stat(value, target):
    if value is None:
        return None
    while value.ndim < target.ndim:
        value = value.unsqueeze(0)
    return value


def load_syndrome_stats(path, map_location="cpu"):
    """Load calibration statistics for calibrated S-IEC scoring."""
    if path is None:
        return None
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"syndrome stats must be a dict, got {type(payload)!r}")


def compute_syndrome(
    x0_current,
    x0_lookahead,
    score_mode="raw",
    stats=None,
    step_idx=None,
    eps=1e-6,
):
    """
    S-IEC syndrome.

    The returned syndrome tensor is always the raw correction direction
    x0_current - x0_lookahead. The score can be raw or calibrated. This keeps
    correction direction separate from reliability scoring.
    
    Args:
        x0_current:   (B, C, H, W) torch.Tensor - clean estimate at t
        x0_lookahead: (B, C, H, W) torch.Tensor - clean estimate at t-1
        score_mode:   "raw", "mean", or "calibrated"
        stats:        optional dict with clean trajectory stats. Expected keys:
                      mu/mean/drift_mean and var/std/q_diag/q_inv_sqrt.
                      mu is for x0_lookahead - x0_current.
        step_idx:     reverse-step index used to select timestep stats
    Returns:
        syndrome: raw correction direction, (B, C, H, W)
        score:    (B,) per-sample reliability score
    """
    syndrome = x0_current - x0_lookahead
    B = syndrome.shape[0]
    d = syndrome.numel() / B  # per-sample dimension (C*H*W)
    mode = (score_mode or "raw").lower()

    if mode == "raw":
        residual = syndrome
    else:
        # Calibration is defined on x0(t-1) - x0(t), matching the report.
        residual = x0_lookahead - x0_current
        mu = _select_stat(
            stats,
            ("mu", "mean", "drift_mean", "mean_drift"),
            step_idx,
            residual.device,
            residual.dtype,
        )
        mu = _broadcast_stat(mu, residual)
        if mu is not None:
            residual = residual - mu

        if mode == "calibrated":
            inv_sqrt = _select_stat(
                stats,
                ("q_inv_sqrt", "inv_sqrt", "precision_sqrt"),
                step_idx,
                residual.device,
                residual.dtype,
            )
            inv_sqrt = _broadcast_stat(inv_sqrt, residual)
            if inv_sqrt is not None:
                residual = residual * inv_sqrt
            else:
                var = _select_stat(
                    stats,
                    ("var", "variance", "q_diag", "diag_var"),
                    step_idx,
                    residual.device,
                    residual.dtype,
                )
                std = _select_stat(
                    stats,
                    ("std", "sigma"),
                    step_idx,
                    residual.device,
                    residual.dtype,
                )
                var = _broadcast_stat(var, residual)
                std = _broadcast_stat(std, residual)
                if std is not None:
                    residual = residual / std.clamp_min(eps)
                elif var is not None:
                    residual = residual / torch.sqrt(var.clamp_min(eps))
                elif stats is None:
                    raise ValueError("calibrated score_mode requires syndrome stats")

    score = (residual ** 2).flatten(1).sum(dim=1) / d
    return syndrome, score
