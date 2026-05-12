import os
import sys
import tqdm
import pdb

import numpy as np
import torch
from torch import nn

import torch.nn.functional as F
import math


class SincBlock(nn.Module):
    """
    Linear → Sinc-based complex activation
    """
    def __init__(self, in_features, out_features, 
                 init_T=1.0, init_zeta=0.1,
                 trainable=True, is_first=True):
        super().__init__()
        self.is_first = is_first
                
        # if self.is_first:
        #     dtype = torch.float
        # else:
        #     dtype = torch.cfloat
        dtype = torch.float
        
        self.linear = nn.Linear(in_features, out_features, dtype=dtype)
        self.T = nn.Parameter(torch.tensor([init_T]), requires_grad=trainable)
        # self.zeta = nn.Parameter(torch.tensor([init_zeta]), requires_grad=trainable)

    def forward(self, x):
        x_proj = self.linear(x)
        T = self.T.to(x_proj.device)
        # ζ = self.zeta.to(x_proj.device)
        sinc = torch.sinc(x_proj / T) / T

        # sinc = torch.sinc(x_proj) 
        
        
        # freq_shift = torch.exp(2j * math.pi * ζ * x_proj)
        
        # out = sinc * freq_shift
        out = sinc 
        # norm = torch.abs(out) + 1e-8
        # out = out / norm.clamp(min=1.0)
        return out




class SincNetwork(nn.Module):
    """
    4 layers: [Linear + Sinc complex]
    final: real part + sigmoid
    """
    def __init__(self, in_features=2, hidden_features=256, out_features=3,
                 num_layers=5, trainable=True,init_T=1.0, init_zeta=0.1,init_beta=None):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_layers - 1):
            in_dim = in_features if i == 0 else hidden_features
            is_first = (i == 0)
            self.blocks.append(
                SincBlock(
                    in_dim, hidden_features, trainable=trainable, is_first=is_first,
                    init_T=init_T, init_zeta=init_zeta
                )
            )
        self.final = nn.Linear(hidden_features, out_features)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = x.real
        x = self.final(x)
        x = self.sigmoid(x)
        return x

    def forward_with_interm(self, x):
        """Forward pass returning intermediate layer outputs"""
        intermediates = {}
        for i, block in enumerate(self.blocks):
            x = block(x)
            intermediates[f'layer_{i}_out'] = x.real
        x = x.real
        x = self.final(x)
        x = self.sigmoid(x)
        intermediates['final_out'] = x
        return intermediates


class INR(nn.Module):
    def __init__(self, in_features, 
                 hidden_features, 
                 hidden_layers, out_features,
                outermost_linear=True,
                 first_omega_0=10.0,
                 hidden_omega_0=10.0,
                 scale=1.0,
                 pos_encode=False,
                 sidelength=None,
                 fn_samples=None,
                 use_nyquist=True, 
                 init_T=1.0, init_zeta=0.1,init_beta=None):
        super().__init__()
       
        self.model = SincNetwork(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            num_layers=hidden_layers + 1,
            trainable=True,
            init_T=init_T,
            init_zeta=init_zeta 
        )
    def forward(self, x):
        return self.model(x)
