"""Dynamic fusion: late fusion with per-modality auxiliary heads and confidence-weighted features.

    enc_m features -+-> aux_head_m -> logits_m -> CE_m              (Gradient Blending, Wang et al. 2020)
                    +-> * w_m --> concat -> dens1 -> dens2 -> fusion logits -> CE

    loss = CE_fusion + aux_weight * sum_m CE_m + RG

1. The auxiliary heads give each encoder a label gradient that does not pass through the shared
   head, so no modality branch can be starved and die during training.
2. The fusion weight w_m of each sample is the auxiliary head's max softmax probability
   (per-sample modality reliability, after the idea of Han et al. 2022, MM-Dynamics),
   normalised to mean 1 over modalities.
3. Optional modality dropout blanks whole branches during training.

The weights are detached, so for the DTD pass they are constants and the z+ rule applies unchanged.
After each forward the module exposes aux_logits, modality_weights and bottleneck_activity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_taylor_decomposition import LinearDeepTaylorDecomposition
from model import flat_features, unflatten_shape
from model_latefusion import _Encoder


class DynFusionModel(nn.Module):
    """
    weighting:        'confidence' (per-sample weights) or 'uniform' (all 1, ablation)
    modality_dropout: per-branch dropout rate during training (0 = off)
    dropout_mode:     'independent' — each branch dropped with p;
                      'solo_scaled' — with prob. min(p*M, 0.6) exactly one branch is kept,
                      so each branch is alone ~p of the time regardless of M;
                      'auto' — solo_scaled for M > 2, otherwise independent
    """

    SOLO_RATE_MAX = 0.6

    def __init__(self, in_channels: int = 1, n_blocks: int = 5, with_dtd: bool = True,
                 width: int = 8, weighting: str = "confidence",
                 modality_dropout: float = 0.0, weight_temp: float = 1.0,
                 dropout_mode: str = "auto"):
        super().__init__()
        if weighting not in ("confidence", "uniform"):
            raise ValueError(f"weighting must be 'confidence' or 'uniform', got {weighting!r}")
        if dropout_mode == "auto":
            dropout_mode = "solo_scaled" if in_channels > 2 else "independent"
        self.with_dtd = with_dtd
        self.n_modalities = in_channels
        self.n_blocks = n_blocks
        self.width = width
        self.weighting = weighting
        self.modality_dropout = modality_dropout
        self.weight_temp = weight_temp
        self.dropout_mode = dropout_mode
        self.feat_per_mod = flat_features(width)
        self.unflatten_shape = unflatten_shape(width)

        self.encoders  = nn.ModuleList([_Encoder(n_blocks, with_dtd, width) for _ in range(in_channels)])
        self.aux_heads = nn.ModuleList([nn.Linear(self.feat_per_mod, 2) for _ in range(in_channels)])
        self.dens1      = nn.Linear(self.feat_per_mod * in_channels, 16)
        self.dens1_relu = nn.ReLU()
        self.dens2      = nn.Linear(16, 2)
        if with_dtd:
            self.dens2_dtd = LinearDeepTaylorDecomposition(self.dens2)
            self.dens1_dtd = LinearDeepTaylorDecomposition(self.dens1)

        self.aux_logits = None
        self.modality_weights = None
        self.bottleneck_activity = None

    def _dropout_mask(self, B, M, device):
        if self.dropout_mode == "independent":
            keep = (torch.rand(B, M, device=device) >= self.modality_dropout).float()
            empty = keep.sum(dim=1) == 0                 # never drop all: revive one at random
            if empty.any():
                revive = torch.randint(0, M, (int(empty.sum()),), device=device)
                keep[empty] = 0.0
                keep[empty, revive] = 1.0
            return keep
        q = min(self.modality_dropout * M, self.SOLO_RATE_MAX)
        keep = torch.ones(B, M, device=device)
        solo = torch.rand(B, device=device) < q
        if solo.any():
            pick = torch.randint(0, M, (int(solo.sum()),), device=device)
            keep[solo] = 0.0
            keep[solo, pick] = 1.0
        return keep

    def _weights(self, aux_logits, feats):
        """Per-sample fusion weights [B, M] (detached, mean 1 across modalities)."""
        B, M = feats[0].shape[0], self.n_modalities
        if self.weighting == "uniform" or M == 1:
            return feats[0].new_ones(B, M)
        conf = torch.stack([F.softmax(a / self.weight_temp, dim=1).max(dim=1).values
                            for a in aux_logits], dim=1).detach()
        if self.training and self.modality_dropout > 0 and M > 1:
            conf = conf * self._dropout_mask(B, M, conf.device)
        return conf / (conf.mean(dim=1, keepdim=True) + 1e-8)

    def forward(self, x):
        feats, cache = [], []
        for m, enc in enumerate(self.encoders):
            xm = x[:, m:m + 1]
            h, es, des = enc(xm)
            feats.append(h.flatten(1))
            cache.append((xm, es, des))

        aux_logits = [head(f) for head, f in zip(self.aux_heads, feats)]
        w = self._weights(aux_logits, feats)
        self.aux_logits = aux_logits if self.n_modalities > 1 else None
        self.modality_weights = w
        self.bottleneck_activity = torch.stack([f.abs().mean(dim=1) for f in feats], dim=1)

        f_cat  = torch.cat([f * w[:, m:m + 1] for m, f in enumerate(feats)], dim=1)
        d1     = self.dens1_relu(self.dens1(f_cat))
        logits = self.dens2(d1)
        if not self.with_dtd:
            return logits

        d2_r = self.dens2_dtd(F.softmax(logits, dim=1), d1)
        d1_r = self.dens1_dtd(d2_r, f_cat)
        heat = []
        for m, enc in enumerate(self.encoders):
            lo, hi = m * self.feat_per_mod, (m + 1) * self.feat_per_mod
            xm, es, des = cache[m]
            heat.append(enc.dtd_backward(d1_r[:, lo:hi].unflatten(1, self.unflatten_shape), xm, es, des))
        return logits, torch.cat(heat, dim=1)

    def aux_loss(self, y, weight: float = 0.3):
        """weight * sum_m CE(aux_logits_m, y); call after forward()."""
        if not self.aux_logits:
            return torch.zeros((), device=y.device)
        return weight * sum(F.cross_entropy(a, y) for a in self.aux_logits)

    def warm_start_encoders(self, ckpts: list) -> list:
        """Initialise encoder m from a single-modality checkpoint of the SAME fold (no leakage).

        ckpts: one path (or None) per modality. Returns the warm-started modality indices.
        """
        done = []
        for m, path in enumerate(ckpts):
            if path is None:
                continue
            state = torch.load(path, map_location="cpu")
            own = self.encoders[m].state_dict()
            transfer = {k: state[k] for k in own if "dtd" not in k}
            for k, v in transfer.items():
                if v.shape != own[k].shape:
                    raise ValueError(f"warm start modality {m}: shape mismatch for {k} in {path}")
            self.encoders[m].load_state_dict(transfer, strict=False)
            done.append(m)
        return done
