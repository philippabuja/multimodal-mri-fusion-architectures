"""Late (intermediate) fusion: one encoder per modality, features concatenated before the classifier.

    x[:, m] -> encoder_m -> flatten        for m = 1..M
    concat -> Linear(->16) -> ReLU -> Linear(->2)

DTD relevance is split at the concatenation and flows back through each modality's own encoder,
so heatmap channel m is the relevance of modality m. Same forward() interface as Model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from deep_taylor_decomposition import LinearDeepTaylorDecomposition, Conv3DDeepTaylorDecomposition
from model import flat_features, unflatten_shape


class _Encoder(nn.Module):
    """One modality's conv stack (1 input channel) -> bottleneck (width, 6, 7, 8)."""

    def __init__(self, n_blocks: int, with_dtd: bool, width: int = 8):
        super().__init__()
        self.n_blocks = n_blocks
        ch = width
        for i in range(1, n_blocks + 1):
            in_c = 1 if i == 1 else ch
            setattr(self, f"conv{i}",           nn.Conv3d(in_c, ch, kernel_size=3, padding=1))
            setattr(self, f"conv{i}_relu",      nn.ReLU())
            setattr(self, f"down_conv{i}",      nn.Conv3d(ch, ch, kernel_size=3, stride=2, padding=1))
            setattr(self, f"down_conv{i}_relu", nn.ReLU())
        if with_dtd:
            for i in range(1, n_blocks + 1):
                setattr(self, f"down_conv{i}_dtd",
                        Conv3DDeepTaylorDecomposition(getattr(self, f"down_conv{i}")))
                setattr(self, f"conv{i}_dtd",
                        Conv3DDeepTaylorDecomposition(getattr(self, f"conv{i}")))

    def forward(self, x):
        es, des = [], []
        h = x
        for i in range(1, self.n_blocks + 1):
            e  = getattr(self, f"conv{i}_relu")(getattr(self, f"conv{i}")(h))
            de = getattr(self, f"down_conv{i}_relu")(getattr(self, f"down_conv{i}")(e))
            es.append(e)
            des.append(de)
            h = de
        return h, es, des

    def dtd_backward(self, r, x, es, des):
        for i in range(self.n_blocks, 0, -1):
            de_r = getattr(self, f"down_conv{i}_dtd")(r, es[i - 1])
            prev = des[i - 2] if i >= 2 else x
            r    = getattr(self, f"conv{i}_dtd")(de_r, prev)
        return r


class LateFusionModel(nn.Module):

    def __init__(self, in_channels: int = 1, n_blocks: int = 5, with_dtd: bool = True,
                 width: int = 8):
        super().__init__()
        self.with_dtd = with_dtd
        self.n_modalities = in_channels
        self.n_blocks = n_blocks
        self.width = width
        self.feat_per_mod = flat_features(width)
        self.unflatten_shape = unflatten_shape(width)

        self.encoders = nn.ModuleList([_Encoder(n_blocks, with_dtd, width) for _ in range(in_channels)])
        self.dens1      = nn.Linear(self.feat_per_mod * in_channels, 16)
        self.dens1_relu = nn.ReLU()
        self.dens2      = nn.Linear(16, 2)
        if with_dtd:
            self.dens2_dtd = LinearDeepTaylorDecomposition(self.dens2)
            self.dens1_dtd = LinearDeepTaylorDecomposition(self.dens1)

        # mean |bottleneck| per branch of the last forward; an all-zero branch is "dead"
        self.bottleneck_activity = None

    def forward(self, x):
        feats, cache = [], []
        for m, enc in enumerate(self.encoders):
            xm = x[:, m:m + 1]
            h, es, des = enc(xm)
            feats.append(h.flatten(1))
            cache.append((xm, es, des))
        self.bottleneck_activity = torch.stack([f.abs().mean(dim=1) for f in feats], dim=1)

        f      = torch.cat(feats, dim=1)
        d1     = self.dens1_relu(self.dens1(f))
        logits = self.dens2(d1)
        if not self.with_dtd:
            return logits

        d2_r = self.dens2_dtd(F.softmax(logits, dim=1), d1)
        d1_r = self.dens1_dtd(d2_r, f)
        heat = []
        for m, enc in enumerate(self.encoders):
            lo, hi = m * self.feat_per_mod, (m + 1) * self.feat_per_mod
            xm, es, des = cache[m]
            heat.append(enc.dtd_backward(d1_r[:, lo:hi].unflatten(1, self.unflatten_shape), xm, es, des))
        return logits, torch.cat(heat, dim=1)
