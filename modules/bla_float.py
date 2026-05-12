"""
Real-only BLA INR — pure float, no torch.cfloat.

Freq shift is implemented via channel split:
    first half (C/2) of channels  → multiplied by cos(phase)
    second half (C/2) of channels → multiplied by sin(phase)

This keeps the layer width and parameter count identical to the cfloat BLA
(`modules/bla.py`), unlike a (re, im) per-channel pair representation that
would 4× the Linear weights.

FP16-safe: EPS, denom clamp, gauss-arg clamp.
"""

import math
import torch
import torch.nn as nn


class BLAActivation(nn.Module):
    """
    Real-only RC-Gauss activation with channel-split freq shift.

        out = (RC(x) * Gauss(x)) * freq_shift_split(x)

        RC(x)    = sinc(x/T) * cos(pi*beta*x/T) / (1 - (2*beta*x/T)^2)
        Gauss(x) = exp(-x^2 / (2*sigma^2))
        freq_shift_split: [..., :C/2] uses cos(phase), [..., C/2:] uses sin(phase)
    """

    EPS = 1e-4
    GAUSS_X_MAX = 6.0

    def __init__(self, init_T=1.0, init_beta=0.05, init_zeta=0.1,
                 sigma=1.0, trainable=True):
        super().__init__()
        self.T     = nn.Parameter(torch.tensor([init_T],    dtype=torch.float32), requires_grad=trainable)
        self.register_buffer("beta", torch.tensor([init_beta], dtype=torch.float32))
        self.zeta  = nn.Parameter(torch.tensor([init_zeta], dtype=torch.float32), requires_grad=trainable)
        self.sigma = nn.Parameter(torch.tensor([sigma],     dtype=torch.float32), requires_grad=trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, beta, zeta, sigma = self.T, self.beta, self.zeta, self.sigma

        x_over_T  = x / T
        sinc      = torch.sinc(x_over_T) / T
        cos_term  = torch.cos(math.pi * beta * x_over_T)
        denom_raw = 1.0 - (2.0 * beta * x_over_T) ** 2
        denom     = denom_raw.clamp(min=self.EPS)
        rc        = sinc * cos_term / denom

        gauss_x_max = self.GAUSS_X_MAX * sigma
        x_clamped   = x.clamp(-gauss_x_max, gauss_x_max)
        gauss       = torch.exp(-(x_clamped ** 2) / (2.0 * sigma ** 2))

        amplitude = rc * gauss
        phase     = 2.0 * math.pi * zeta * x

        C      = x.shape[-1]
        half_C = C // 2

        if half_C == 0:
            out = amplitude * torch.cos(phase)
        else:
            out_cos = amplitude[..., :half_C] * torch.cos(phase[..., :half_C])
            out_sin = amplitude[..., half_C:] * torch.sin(phase[..., half_C:])
            out = torch.cat([out_cos, out_sin], dim=-1)

        norm = torch.abs(out) + self.EPS
        out = out / norm.clamp(min=1.0)
        return out

    def extra_repr(self) -> str:
        return f"EPS={self.EPS}, GAUSS_X_MAX={self.GAUSS_X_MAX}"


class BLABlock(nn.Module):
    """Linear → BLAActivation."""

    def __init__(self, in_features, out_features,
                 init_T=1.0, init_beta=0.05, init_zeta=0.1,
                 sigma=1.0, trainable=True, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.act    = BLAActivation(
            init_T=init_T, init_beta=init_beta, init_zeta=init_zeta,
            sigma=sigma, trainable=trainable,
        )

    def forward(self, x):
        return self.act(self.linear(x))


class INR(nn.Module):
    """
    BLA-float INR — structurally mirrors SIREN's INR and bla.py:
        first BLABlock      ×1
        hidden BLABlocks    ×hidden_layers
        final Linear        ×1
        Sigmoid (optional)

    len(self.net) = hidden_layers + 2  (matches SIREN exactly).

    hidden_features must be even (channel-split freq shift).
    """

    def __init__(self, in_features, hidden_features, hidden_layers, out_features,
                 outermost_linear=True, first_omega_0=10.0, hidden_omega_0=10.0,
                 scale=1.0, pos_encode=False, sidelength=None, fn_samples=None,
                 use_nyquist=True,
                 init_T=1.0, init_beta=0.05, init_zeta=0.1,
                 use_sigmoid=True, bias=True, **kwargs):
        super().__init__()
        self.pos_encode = pos_encode

        init_T    = 1.0  if init_T    is None else init_T
        init_beta = 0.05 if init_beta is None else init_beta
        init_zeta = 0.1  if init_zeta is None else init_zeta

        if hidden_features % 2 != 0:
            raise ValueError(
                f"hidden_features must be even for channel-split freq shift, got {hidden_features}"
            )

        self.net = []
        # hidden_layers == total number of BLABlocks (1 first + hidden_layers-1 hidden)
        self.net.append(BLABlock(in_features, hidden_features,
                                 sigma=scale, bias=bias,
                                 init_T=init_T, init_beta=init_beta, init_zeta=init_zeta))
        for _ in range(hidden_layers - 1):
            self.net.append(BLABlock(hidden_features, hidden_features,
                                     sigma=scale, bias=bias,
                                     init_T=init_T, init_beta=init_beta, init_zeta=init_zeta))
        self.net.append(nn.Linear(hidden_features, out_features, bias=bias))
        self.net = nn.Sequential(*self.net)
        self.activation = nn.Sigmoid() if use_sigmoid else nn.Identity()

    def forward(self, x):
        return self.activation(self.net(x))

    def forward_with_interm(self, x):
        intermediates = {}
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i < len(self.net) - 1:
                intermediates[f'layer_{i}_out'] = x
        return intermediates
