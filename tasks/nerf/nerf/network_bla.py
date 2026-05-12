import os
import sys
import torch
import torch.nn as nn

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer

# FLAIR root for modules.bla (our BLA building block)
_FLAIR_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _FLAIR_ROOT not in sys.path:
    sys.path.insert(0, _FLAIR_ROOT)
from modules.bla import BLABlock as _FlairBLABlock


class BLALayer(nn.Module):
    """
    NeRF layer with two roles:
      - is_first / hidden : FLAIR's BLA block (Linear + RC-Gauss + complex/cfloat)
      - is_last           : plain real Linear (no activation) — for sigma+RGB heads
    """
    def __init__(self, in_features, out_features,
                 init_T=1.0, init_beta=0.05, init_zeta=0.1,
                 sigma=1.0, trainable=True,
                 is_first=False, is_last=False):
        super().__init__()
        self.is_last = is_last
        if is_last:
            self.linear = nn.Linear(in_features, out_features)
        else:
            self.block = _FlairBLABlock(
                in_features, out_features,
                init_T=init_T, init_beta=init_beta, init_zeta=init_zeta,
                sigma=sigma, trainable=trainable, is_first=is_first,
            )

    def forward(self, x):
        if self.is_last:
            x = x.real if x.is_complex() else x
            return self.linear(x)
        return self.block(x)
    
  
class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_bg="hashgrid",
                 num_layers=4,
                 hidden_dim=256,
                 geo_feat_dim=256,
                 num_layers_color=4,
                 hidden_dim_color=256,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 bound=1,
                 init_T=1.0,
                 init_beta=0.05,
                 init_zeta=0.1,
                 sigma=2.0,
                 trainable=True,
                 **kwargs):
        super().__init__(bound, **kwargs)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        # no encoding
        self.encoder, self.in_dim = get_encoder(encoding='None')

        # ✅ sigma network
        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                sigma_net.append(
                    BLALayer(
                        self.in_dim, hidden_dim,
                        init_T=init_T, init_beta=init_beta, init_zeta=init_zeta,
                        sigma=sigma, trainable=trainable,
                        is_first=True
                    )
                )
            # last layer     
            elif l == num_layers - 1:
                sigma_net.append(
                    BLALayer(
                        hidden_dim, 1 + self.geo_feat_dim,
                        init_T=init_T, init_beta=init_beta, init_zeta=init_zeta,
                        sigma=sigma, trainable=trainable,
                        is_last=True
                    )
                )
            else:
                sigma_net.append(
                    BLALayer(
                        hidden_dim, hidden_dim,
                        init_T=init_T, init_beta=init_beta, init_zeta=init_zeta,
                        sigma=sigma, trainable=trainable
                    )
                )
                
                
        self.sigma_net = nn.ModuleList(sigma_net)

        # ✅ color network
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding='None')

        color_net = []
        for l in range(num_layers_color):
            if l == 0:
                color_net.append(
                    BLALayer(
                        self.in_dim_dir + self.geo_feat_dim,
                        hidden_dim_color,
                        init_T=init_T, init_beta=init_beta, init_zeta=init_zeta,
                        sigma=sigma, trainable=trainable,
                        is_first=True
                    )
                )
            elif l == num_layers_color - 1:
                color_net.append(
                    BLALayer(
                        hidden_dim_color,
                        3,
                        init_T=init_T, init_beta=init_beta, init_zeta=init_zeta,
                        sigma=sigma, trainable=trainable,
                        is_last=True
                    )
                )
            else:
                color_net.append(
                    BLALayer(
                        hidden_dim_color,
                        hidden_dim_color,
                        init_T=init_T, init_beta=init_beta, init_zeta=init_zeta,
                        sigma=sigma, trainable=trainable
                    )
                )
        self.color_net = nn.ModuleList(color_net)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg        
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(encoding_bg, input_dim=2, num_levels=4, log2_hashmap_size=19, desired_resolution=2048) # much smaller hashgrid 
            
            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.in_dim_bg + self.in_dim_dir
                else:
                    in_dim = hidden_dim_bg
                
                if l == num_layers_bg - 1:
                    out_dim = 3 # 3 rgb
                else:
                    out_dim = hidden_dim_bg
                
                bg_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.bg_net = nn.ModuleList(bg_net)
        else:
            self.bg_net = None







    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        x = self.encoder(x, bound=self.bound)

        h = x # encoded feature 
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0]) # density: first channel
        geo_feat = h[..., 1:] # geometry: remaining channels


        # color
        d = self.encoder_dir(d) # direction encoding (none)
        h = torch.cat([d, geo_feat], dim=-1) # viewing direction + geometry = color 
        
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            
            
        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        return sigma, color




    # density (geo)
    def density(self, x):
        # x: [N, 3], in [-bound, bound]

        x = self.encoder(x, bound=self.bound)
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0]) # density: first channel
        geo_feat = h[..., 1:] # geometry: remaining channels

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }



    def background(self, x, d):
        # x: [N, 2], in [-1, 1]

        h = self.encoder_bg(x) # [N, C]
        d = self.encoder_dir(d)

        h = torch.cat([d, h], dim=-1)
        for l in range(self.num_layers_bg):
            h = self.bg_net[l](h)
            if l != self.num_layers_bg - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs


    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        
        
        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device)
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)

        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype)  # fp16 --> fp32
        else:
            rgbs = h
            
        return rgbs

    def get_params(self, lr):
        params = [
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr},
        ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        return params








