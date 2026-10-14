import torch.nn as nn
import torch.nn.functional as F

from deep_taylor_decomposition import LinearDeepTaylorDecomposition, Conv3DDeepTaylorDecomposition

# Both input resolutions reach the same bottleneck (width, 6, 7, 8):
#   n_blocks=5: 1 mm (176,224,256)    n_blocks=4: 2 mm (88,112,128)
BOTTLENECK_HW = (6, 7, 8)


def flat_features(width: int) -> int:
    return width * BOTTLENECK_HW[0] * BOTTLENECK_HW[1] * BOTTLENECK_HW[2]


def unflatten_shape(width: int) -> tuple:
    return (width, *BOTTLENECK_HW)


def build_model(arch: str, in_channels: int, n_blocks: int, with_dtd: bool = True,
                width: int = 8, **kwargs):
    """early: modalities as input channels (Model)
    late:  one encoder per modality, joined at the classifier (LateFusionModel)
    dyn:   late + per-modality auxiliary heads + confidence-weighted fusion (DynFusionModel)
    """
    if arch == "early":
        return Model(in_channels=in_channels, n_blocks=n_blocks, with_dtd=with_dtd, width=width)
    if arch == "late":
        from model_latefusion import LateFusionModel
        return LateFusionModel(in_channels=in_channels, n_blocks=n_blocks, with_dtd=with_dtd,
                               width=width)
    if arch == "dyn":
        from model_dynfusion import DynFusionModel
        return DynFusionModel(in_channels=in_channels, n_blocks=n_blocks, with_dtd=with_dtd,
                              width=width, **kwargs)
    raise ValueError(f"unknown arch {arch!r}")


class Model(nn.Module):
    """3D CNN after Tinauer et al. (2020) with a built-in DTD relevance pass.

    n_blocks x [Conv3d(k3) + ReLU + Conv3d(k3, stride 2) + ReLU] -> Linear(->16) -> ReLU -> Linear(->2)

    with_dtd=True:  forward(x) -> (logits, heatmaps), heatmaps.shape == x.shape
    with_dtd=False: forward(x) -> logits
    """

    def __init__(self, in_channels: int = 1, n_blocks: int = 5, with_dtd: bool = True,
                 width: int = 8):
        super().__init__()
        self.with_dtd = with_dtd
        self.n_blocks = n_blocks
        self.width = width
        self.flat_features = flat_features(width)
        self.unflatten_shape = unflatten_shape(width)
        ch = width

        for i in range(1, n_blocks + 1):
            in_c = in_channels if i == 1 else ch
            setattr(self, f"conv{i}",           nn.Conv3d(in_c, ch, kernel_size=3, padding=1))
            setattr(self, f"conv{i}_relu",      nn.ReLU())
            setattr(self, f"down_conv{i}",      nn.Conv3d(ch, ch, kernel_size=3, stride=2, padding=1))
            setattr(self, f"down_conv{i}_relu", nn.ReLU())

        self.dens1      = nn.Linear(self.flat_features, 16)
        self.dens1_relu = nn.ReLU()
        self.dens2      = nn.Linear(16, 2)

        if with_dtd:
            self.dens2_dtd = LinearDeepTaylorDecomposition(self.dens2)
            self.dens1_dtd = LinearDeepTaylorDecomposition(self.dens1)
            for i in range(1, n_blocks + 1):
                setattr(self, f"down_conv{i}_dtd",
                        Conv3DDeepTaylorDecomposition(getattr(self, f"down_conv{i}")))
                setattr(self, f"conv{i}_dtd",
                        Conv3DDeepTaylorDecomposition(getattr(self, f"conv{i}")))

    def forward(self, x):
        es, des = [], []          # per-block activations, needed by the DTD pass
        h = x
        for i in range(1, self.n_blocks + 1):
            e  = getattr(self, f"conv{i}_relu")(getattr(self, f"conv{i}")(h))
            de = getattr(self, f"down_conv{i}_relu")(getattr(self, f"down_conv{i}")(e))
            es.append(e)
            des.append(de)
            h = de

        f      = h.flatten(1)
        d1     = self.dens1_relu(self.dens1(f))
        logits = self.dens2(d1)

        if not self.with_dtd:
            return logits

        d2_r = self.dens2_dtd(F.softmax(logits, dim=1), d1)
        d1_r = self.dens1_dtd(d2_r, f)
        r    = d1_r.unflatten(1, self.unflatten_shape)
        for i in range(self.n_blocks, 0, -1):
            de_r = getattr(self, f"down_conv{i}_dtd")(r, es[i - 1])
            prev = des[i - 2] if i >= 2 else x
            r    = getattr(self, f"conv{i}_dtd")(de_r, prev)
        return logits, r
