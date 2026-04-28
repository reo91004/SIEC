import torch

def compute_syndrome(x0_current, x0_lookahead):
    """
    Martingale syndrome: ŝ_t = x̂_0(t) - x̂_0(t-1)
    
    Args:
        x0_current:   (B, C, H, W) torch.Tensor - clean estimate at t
        x0_lookahead: (B, C, H, W) torch.Tensor - clean estimate at t-1
    Returns:
        syndrome: (B, C, H, W) tensor
        score:    (B,) per-sample syndrome score r_t = ||ŝ||² / d
    """
    syndrome = x0_current - x0_lookahead
    B = syndrome.shape[0]
    d = syndrome.numel() / B  # per-sample dimension (C*H*W)
    score = (syndrome ** 2).flatten(1).sum(dim=1) / d
    return syndrome, score