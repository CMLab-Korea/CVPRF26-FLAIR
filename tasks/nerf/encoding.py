# encoding.py
import torch
import torch.nn as nn

def _make_freq_fallback(input_dim: int, multires: int,
                        log_sampling: bool = True,
                        include_input: bool = True):
    """Pure-Python Fourier Positional Encoding fallback (used if the CUDA extension fails to load)."""
    class _FallbackFreqEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_dim = input_dim
            self.include_input = include_input

            if log_sampling:
                freq_bands = 2. ** torch.linspace(0., multires - 1, multires)
            else:
                freq_bands = torch.linspace(2. ** 0., 2. ** (multires - 1), multires)

            self.register_buffer("freq_bands", freq_bands, persistent=False)

            self.output_dim = (input_dim if include_input else 0) + input_dim * multires * 2

        def forward(self, x, **kwargs):
            outs = []
            if self.include_input:
                outs.append(x)
            for f in self.freq_bands:
                outs.append(torch.sin(x * f))
                outs.append(torch.cos(x * f))
            return torch.cat(outs, dim=-1)

        def __repr__(self):
            return (f"FallbackFreqEncoder(input_dim={self.input_dim}, "
                    f"multires={multires}, output_dim={self.output_dim})")
    enc = _FallbackFreqEncoder()
    return enc, enc.output_dim


def get_encoder(encoding, input_dim=3,
                multires=6,
                degree=4,
                num_levels=16, level_dim=2, base_resolution=16,
                log2_hashmap_size=19, desired_resolution=2048,
                align_corners=False,
                **kwargs):
    """
    Returns: (encoder_module, output_dim)
      - encoder_module(x, **kwargs) -> [*, output_dim]
      - If encoding == 'None', returns (identity, input_dim).
    """

    if encoding == 'None':
        return (lambda x, **k: x), input_dim

    elif encoding == 'frequency':
        # Prefer the CUDA-backed freqencoder; fall back to Python on failure.
        try:
            from freqencoder import FreqEncoder  # (input_dim, degree)
            enc = FreqEncoder(input_dim=input_dim, degree=multires)
            return enc, enc.output_dim
        except Exception as e:
            return _make_freq_fallback(input_dim=input_dim, multires=multires)

    elif encoding == 'sphere_harmonics':
        from shencoder import SHEncoder
        enc = SHEncoder(input_dim=input_dim, degree=degree)
        return enc, enc.output_dim

    elif encoding == 'hashgrid':
        from gridencoder import GridEncoder
        enc = GridEncoder(
            input_dim=input_dim,
            num_levels=num_levels, level_dim=level_dim,
            base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size,
            desired_resolution=desired_resolution, gridtype='hash',
            align_corners=align_corners,
        )
        return enc, enc.output_dim

    elif encoding == 'tiledgrid':
        from gridencoder import GridEncoder
        enc = GridEncoder(
            input_dim=input_dim,
            num_levels=num_levels, level_dim=level_dim,
            base_resolution=base_resolution, log2_hashmap_size=log2_hashmap_size,
            desired_resolution=desired_resolution, gridtype='tiled',
            align_corners=align_corners,
        )
        return enc, enc.output_dim

    elif encoding == 'ash':
        from ashencoder import AshEncoder
        enc = AshEncoder(
            input_dim=input_dim, output_dim=16,
            log2_hashmap_size=log2_hashmap_size, resolution=desired_resolution,
        )
        return enc, enc.output_dim

    else:
        raise NotImplementedError(
            "Unknown encoding mode, choose from "
            "[None, frequency, sphere_harmonics, hashgrid, tiledgrid, ash]"
        )
