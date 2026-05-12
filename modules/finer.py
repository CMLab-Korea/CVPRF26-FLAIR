import torch
import torch.nn as nn
import numpy as np

class FinerLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.0, first_bias_scale=None, scale_req_grad=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.first_bias_scale = first_bias_scale
        self.scale_req_grad = scale_req_grad
        self.in_features = in_features

        self.init_weights()
        if self.is_first and self.first_bias_scale is not None:
            self.init_first_bias()

    def init_weights(self):
        with torch.no_grad():
           
            if self.is_first:
                self.linear.weight.uniform_(
                    -1 / self.in_features,
                    1 / self.in_features
                )
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0
                )
                
    def init_first_bias(self):
        with torch.no_grad():
            self.linear.bias.uniform_(-self.first_bias_scale, self.first_bias_scale)

    def generate_scale(self, x):
        if self.scale_req_grad:
            scale = torch.abs(x) + 1
        else:
            with torch.no_grad():
                scale = torch.abs(x) + 1
        return scale

    def forward(self, x):
        x = self.linear(x)
        scale = self.generate_scale(x)
        out = torch.sin(self.omega_0 * scale * x)
        return out


class INR(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features,
                 first_omega_0=30.0, hidden_omega_0=30.0, first_bias_scale=None,
                 is_first_scale=False, scale_req_grad=False, outermost_linear=True, **kwargs):
        super().__init__()
        net = []

        # First layer
        net.append(
            FinerLayer(
                in_features, hidden_features,
                is_first=True,
                omega_0=first_omega_0,
                bias=True,
                first_bias_scale=first_bias_scale,
                scale_req_grad=scale_req_grad
            )
        )

        # Hidden layers — hidden_layers == total FinerLayers (1 first + hidden_layers-1 hidden)
        for i in range(hidden_layers - 1):
            net.append(
                FinerLayer(
                    hidden_features, hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                    bias=True,
                    scale_req_grad=scale_req_grad
                )
            )

        # Outermost linear default True
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0
                )
            net.append(final_linear)
        else:
            net.append(
                FinerLayer(
                    hidden_features, out_features,
                    is_first=True,
                    omega_0=hidden_omega_0,
                    bias=True,
                    scale_req_grad=scale_req_grad
                )
            )
            
        
        self.net = nn.Sequential(*net)

    def forward(self, coords):
        return self.net(coords)
    
    