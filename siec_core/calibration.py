"""Clean trajectory drift statistics for calibrated S-IEC.

Accumulates per-step diagonal mean/variance of the drift signal
    drift_t = x0_lookahead(t-1) - x0_current(t)
across many clean (PTQ off, full-compute, no-correction) trajectories.

Uses Chan's parallel Welford formula so each batch update is O(1) batch size,
and statistics are merged across batches without revisiting raw samples.
"""
from __future__ import annotations

import torch


class DriftStatsAccumulator:
    """Per-step running mean / variance of x0_look - x0_current."""

    def __init__(
        self,
        num_steps: int,
        shape: tuple[int, int, int],
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.num_steps = int(num_steps)
        self.shape = tuple(shape)
        self.device = torch.device(device)
        self.dtype = dtype
        full_shape = (self.num_steps, *self.shape)
        self.count = torch.zeros(self.num_steps, dtype=torch.long, device=self.device)
        self.mean = torch.zeros(full_shape, dtype=self.dtype, device=self.device)
        self.M2 = torch.zeros(full_shape, dtype=self.dtype, device=self.device)

    @torch.no_grad()
    def update(self, step_idx: int, drift: torch.Tensor) -> None:
        """Merge a batch of drift samples into the running stats for step `step_idx`.

        drift: (B, C, H, W) — per-sample, per-pixel drift x0_look - x0_current.
        Uses Chan's parallel-Welford merge to combine the batch with prior counts.
        """
        if step_idx < 0 or step_idx >= self.num_steps:
            return
        x = drift.detach().to(device=self.device, dtype=self.dtype)
        B = int(x.shape[0])
        if B == 0:
            return
        if x.shape[1:] != self.shape:
            raise ValueError(
                f"drift shape {tuple(x.shape[1:])} != accumulator shape {self.shape}"
            )

        n_old = int(self.count[step_idx].item())
        n_new = n_old + B

        batch_mean = x.mean(dim=0)
        # second central moment within the batch
        batch_M2 = ((x - batch_mean) ** 2).sum(dim=0)

        if n_old == 0:
            self.mean[step_idx] = batch_mean
            self.M2[step_idx] = batch_M2
        else:
            delta = batch_mean - self.mean[step_idx]
            self.mean[step_idx] = self.mean[step_idx] + delta * (B / n_new)
            self.M2[step_idx] = self.M2[step_idx] + batch_M2 + (delta ** 2) * (n_old * B / n_new)
        self.count[step_idx] = n_new

    def finalize(self, eps: float = 1e-6) -> dict:
        """Return v1 §3 M1 payload.

        Counts with <2 samples produce variance=0 → clamped to eps.
        """
        denom = torch.clamp(self.count.float() - 1.0, min=1.0)
        var = self.M2 / denom.view(-1, *([1] * len(self.shape)))
        var = torch.clamp(var, min=float(eps))
        std = var.sqrt()
        return {
            "version": 1,
            "kind": "clean_trajectory_drift_stats",
            "score_space": "x0",
            "mu": self.mean.float().cpu(),
            "var": var.float().cpu(),
            "std": std.float().cpu(),
            "count": self.count.cpu(),
            "eps": float(eps),
        }
