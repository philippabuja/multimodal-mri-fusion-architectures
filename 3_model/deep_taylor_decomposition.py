"""Deep Taylor Decomposition (z+ rule) for Linear and Conv3d layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearDeepTaylorDecomposition(nn.Module):
  def __init__(self, for_layer: nn.Linear) -> None:
    super().__init__()
    self.for_layer = for_layer

  def forward(self, R: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    X = X + 1e-7
    W = self.for_layer.weight.data

    Wp = W.clamp(min=1e-7)
    Zp = F.linear(X, Wp)
    S = R / Zp
    C = F.linear(S, Wp.transpose(1, 0))

    return X * C

class Conv3DDeepTaylorDecomposition(nn.Module):
  def __init__(self, for_layer: nn.Conv3d):
    super().__init__()
    self.for_layer = for_layer

  def forward(self, R: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    X = X + 1e-7
    W = self.for_layer.weight.data

    Wp = W.clamp(min=1e-7)
    Zp = F.conv3d(X, Wp,
                  bias=None,
                  stride=self.for_layer.stride,
                  padding=self.for_layer.padding,
                  dilation=self.for_layer.dilation,
                  groups=self.for_layer.groups)
    S = R / Zp
    C = F.conv_transpose3d(S, Wp,
                           bias=None,
                           stride=self.for_layer.stride,
                           padding=self.for_layer.padding,
                           groups=self.for_layer.groups,
                           dilation=self.for_layer.dilation)

    # fix output padding
    d3 = X.shape[2] - C.shape[2]
    d4 = X.shape[3] - C.shape[3]
    d5 = X.shape[4] - C.shape[4]
    C = F.pad(C, (d5 // 2, d5 - (d5 // 2), d4 // 2, d4 - (d4 // 2), d3 // 2, d3 - (d3 // 2)), mode='constant', value=1e-14)

    return X * C
