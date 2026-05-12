import torch
import torch.nn as nn
import math
from .ChebyKANLayer import ChebyKANLayer


# =======================================
# 1. Basic Layer Definitions
# =======================================

class ChebyLayer(nn.Module):
    def __init__(self, in_features, out_features, deg, init_method):
        super(ChebyLayer, self).__init__()
        self.cheby = ChebyKANLayer(in_features, out_features, deg, init_method)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x):
        x = self.cheby(x)
        x = self.norm(x)
        return x


class ReLULayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ReLULayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        return nn.functional.relu(self.linear(input))


class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, input):
        return self.linear(input)


class LowRankReLULayer(nn.Module):
    def __init__(self, in_features, out_features, rank=128, bias=True,
                 nonlinearity='relu', linear_init_type='kaiming_uniform'):
        super(LowRankReLULayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.nonlinearity = nonlinearity

        # Low-rank factorization weights
        self.weight_left = nn.Parameter(torch.Tensor(in_features, rank))
        self.weight_right = nn.Parameter(torch.Tensor(rank, out_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters(linear_init_type)

    def reset_parameters(self, linear_init_type='kaiming_uniform'):
        if linear_init_type == 'kaiming_uniform':
            nn.init.kaiming_uniform_(self.weight_left, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.weight_right, a=math.sqrt(5))
        elif linear_init_type == 'kaiming_normal':
            nn.init.kaiming_normal_(self.weight_left, a=math.sqrt(5))
            nn.init.kaiming_normal_(self.weight_right, a=math.sqrt(5))
        elif linear_init_type == 'orthogonal':
            nn.init.orthogonal_(self.weight_left)
            nn.init.orthogonal_(self.weight_right)
        elif linear_init_type == 'uniform':
            nn.init.uniform_(self.weight_left, a=-0.5, b=0.5)
            nn.init.uniform_(self.weight_right, a=-0.5, b=0.5)
        elif linear_init_type == 'normal':
            nn.init.normal_(self.weight_left, mean=0.0, std=1 / (self.in_features * self.rank))
            nn.init.normal_(self.weight_right, mean=0.0, std=1 / (self.rank * self.out_features))
        elif linear_init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(self.weight_left)
            nn.init.xavier_uniform_(self.weight_right)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_left)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        weight = torch.matmul(self.weight_left, self.weight_right)
        output = torch.matmul(input, weight)

        if self.bias is not None:
            output += self.bias

        if self.nonlinearity == 'relu':
            return nn.functional.relu(output)
        
        elif self.nonlinearity in (None, 'none'):
            return output


# =======================================
# 2. INR Model (Cheby + LowRankReLU × L + Linear)
# =======================================

class INR(nn.Module):
    """
    INR model structure:
        ChebyLayer → (LowRankReLULayer × hidden_layers) → Linear
    """
    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 hidden_layers: int,
                 out_features: int,
                 deg: int = 256,
                 rank: int = 128,
                 nonlinearity: str = 'relu',
                 init_method: str = 'xavier_uniform',
                 linear_init_type: str = 'kaiming_uniform',
                 outermost_linear: bool = True,
                 **kwargs):
        super().__init__()

        # 1) Chebyshev feature expansion
        self.cheby = ChebyLayer(
            in_features=in_features,
            out_features=hidden_features,
            deg=deg,
            init_method=init_method
        )

        # 2) Low-rank ReLU blocks
        self.blocks = nn.ModuleList([
            LowRankReLULayer(
                in_features=hidden_features,
                out_features=hidden_features,
                rank=rank,
                nonlinearity=nonlinearity,
                linear_init_type=linear_init_type
            )
            for _ in range(hidden_layers - 1)  # +1 for ChebyLayer = hidden_layers total nonlinear
        ])

        # 3) Final linear or low-rank output layer
        if outermost_linear:
            self.final = nn.Linear(hidden_features, out_features)
        else:
            self.final = LowRankReLULayer(
                in_features=hidden_features,
                out_features=out_features,
                rank=rank,
                nonlinearity=None,
                linear_init_type=linear_init_type
            )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim == 1:
            coords = coords.unsqueeze(0)
        elif coords.ndim > 2:
            coords = coords.view(coords.size(0), -1)

        x = self.cheby(coords)
        y = x
        for blk in self.blocks:
            y = blk(x * y)  # element-wise multiply (same as einsum('ij,ij->ij', x, y))

        y = self.final(x * y)    
        # y = self.final(y)
        return y

    @torch.no_grad()
    def summary_layers(self):
        print("[0] ChebyLayer")
        for i, _ in enumerate(self.blocks, start=1):
            print(f"[{i}] LowRankReLULayer")
        print(f"[{len(self.blocks) + 1}] {self.final.__class__.__name__}")


