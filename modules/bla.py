import torch
from torch import nn
import math


class BLABlock(nn.Module):
    """Linear → RC-based complex activation. (1 layer)"""

    def __init__(self, in_features, out_features,
                 init_T=1.0, init_beta=0.05, init_zeta=0.1,
                 sigma=1.0, trainable=True, is_first=True):
        super().__init__()
        self.is_first = is_first
        dtype = torch.float if is_first else torch.cfloat
        self.linear = nn.Linear(in_features, out_features, dtype=dtype)

        self.T     = nn.Parameter(torch.tensor([init_T]),    requires_grad=trainable)
        self.beta  = torch.tensor([init_beta], requires_grad=False)
        self.zeta  = nn.Parameter(torch.tensor([init_zeta]), requires_grad=trainable)
        self.sigma = nn.Parameter(torch.tensor([sigma]),     requires_grad=trainable)

    def forward(self, x):
        x_proj = self.linear(x)
        T     = self.T.to(x_proj.device)
        beta  = self.beta.to(x_proj.device)
        zeta  = self.zeta.to(x_proj.device)
        sigma = self.sigma.to(x_proj.device)

        sinc        = torch.sinc(x_proj / T) / T
        cos         = torch.cos(math.pi * beta * x_proj / T)
        denom       = 1.0 - (2 * beta * x_proj / T) ** 2 + 1e-6
        rc          = sinc * cos / denom
        freq_shift  = torch.exp(2j * math.pi * zeta * x_proj)
        gauss       = torch.exp(-(x_proj ** 2) / (2 * sigma ** 2))

        out = (rc * gauss) * freq_shift
        norm = torch.abs(out) + 1e-8
        out = out / norm.clamp(min=1.0)
        return out


class _FinalLinear(nn.Linear):
    """nn.Linear that auto-takes .real of complex input. Used as the last entry of INR.net."""
    def forward(self, x):
        if x.is_complex():
            x = x.real
        return super().forward(x)


class INR(nn.Module):
    """
    BLA INR — structurally mirrors SIREN's INR:
        first BLABlock      ×1
        hidden BLABlocks    ×hidden_layers
        final Linear        ×1
        Sigmoid

    len(self.net) = hidden_layers + 2  (matches SIREN exactly).
    """

    def __init__(self, in_features, hidden_features, hidden_layers, out_features,
                 outermost_linear=True, first_omega_0=10.0, hidden_omega_0=10.0,
                 scale=1.0, pos_encode=False, sidelength=None, fn_samples=None,
                 use_nyquist=True,
                 init_T=1.0, init_beta=0.05, init_zeta=0.1,
                 use_sigmoid=True, **kwargs):
        super().__init__()
        self.pos_encode = pos_encode

        init_T    = 1.0  if init_T    is None else init_T
        init_beta = 0.05 if init_beta is None else init_beta
        init_zeta = 0.1  if init_zeta is None else init_zeta

        self.net = []
        # hidden_layers == total BLABlocks (1 first + hidden_layers-1 hidden)
        self.net.append(BLABlock(in_features, hidden_features, is_first=True,
                                 sigma=scale,
                                 init_T=init_T, init_beta=init_beta, init_zeta=init_zeta))
        for _ in range(hidden_layers - 1):
            self.net.append(BLABlock(hidden_features, hidden_features, is_first=False,
                                     sigma=scale,
                                     init_T=init_T, init_beta=init_beta, init_zeta=init_zeta))
        self.net.append(_FinalLinear(hidden_features, out_features))
        self.net = nn.Sequential(*self.net)
        self.activation = nn.Sigmoid() if use_sigmoid else nn.Identity()

    def forward(self, x):
        return self.activation(self.net(x))

    def forward_with_interm(self, x):
        intermediates = {}
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i < len(self.net) - 1:  # only BLABlocks
                intermediates[f'layer_{i}_out'] = x.real
        return intermediates
