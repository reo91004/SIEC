import torch

def compute_gamma(alpha_t_sq, sigma_t_sq, c=1.0):
    """
    λ_t = c · σ_t² / (α_t² + σ_t²)
    γ_t = λ_t / (1 + λ_t)
    
    DDPM convention:
        α_t² = alphas_cumprod[t]      (IEC repo의 `at`)
        σ_t² = 1 - alphas_cumprod[t]  (IEC repo의 `1 - at`)
    
    Args:
        alpha_t_sq: scalar tensor or float - α_t² (= at in DDPM code)
        sigma_t_sq: scalar tensor or float - σ_t² (= 1-at)
        c: correction strength multiplier
    Returns:
        gamma: scalar or tensor (same shape as inputs)
    """
    lam = c * sigma_t_sq / (alpha_t_sq + sigma_t_sq)
    gamma = lam / (1.0 + lam)
    return gamma


def apply_consensus_correction(x0_current, syndrome, gamma):
    """
    x̄_0 = x̂_0(t) - γ · ŝ_t
    
    Args:
        x0_current: (B, C, H, W) tensor
        syndrome:   (B, C, H, W) tensor
        gamma:      scalar tensor or (B,) per-sample
    Returns:
        x0_corrected: (B, C, H, W) tensor
    """
    if isinstance(gamma, torch.Tensor) and gamma.ndim > 0:
        # broadcast to (B, 1, 1, 1)
        gamma = gamma.view(-1, *([1] * (syndrome.ndim - 1)))
    return x0_current - gamma * syndrome