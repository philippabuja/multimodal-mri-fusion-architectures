import torch
import torch.nn as nn


class RelevanceGuidedLoss(nn.Module):
    """Relevance-guided (RG) loss: rewards DTD relevance that lies inside the brain mask.

    pooled_sum   -(heatmap * mask).sum() over channels and space (Tinauer et al. 2020).
                 Used for single-modality and early-fusion models.
    per_modality -mean_c( sum(relu(R_c) * mask) / sum(relu(R_c)) ): each modality is scored on its
                 own inside-fraction. Used for late/dynamic fusion: under pooled_sum, silencing a
                 whole modality branch maximises the objective, here a silent branch scores 0.

    heatmap: [B, C, D, H, W]   mask: [B, 1, D, H, W]
    """

    MODES = ("pooled_sum", "per_modality")

    def __init__(self, mode: str = "pooled_sum", eps: float = 1e-9):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.mode = mode
        self.eps = eps

    def forward(self, heatmap, mask):
        if self.mode == "pooled_sum":
            return -(heatmap * mask).sum(dim=[1, 2, 3, 4]).mean()
        pos = heatmap.clamp(min=0)
        inside = (pos * mask).sum(dim=[2, 3, 4])            # [B, C]
        total = pos.sum(dim=[2, 3, 4])
        frac = torch.where(total > self.eps, inside / (total + self.eps), torch.zeros_like(total))
        return -frac.mean()
