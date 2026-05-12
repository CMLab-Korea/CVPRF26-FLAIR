import os
import sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))               # tasks/sdf/
sys.path.insert(0, os.path.join(_THIS_DIR, 'lib'))                   # bacon helpers
sys.path.insert(0, os.path.dirname(os.path.dirname(_THIS_DIR)))      # FLAIR root → modules.bla

import utils
import torch
import numpy as np
from tqdm import tqdm
import mcubes
import trimesh
import dataio
import math


# -----------------------------
# Core mesh generator (single-scale)
# -----------------------------
@torch.no_grad()
def generate_mesh(model, N, return_sdf=False, num_outputs=1, model_name='model', bounds=(-0.5, 0.5)):
    """
    model: forward({'coords': [B,3]}) -> {'model_out': Tensor or [Tensor,...]}
    N: grid resolution per axis
    num_outputs: number of heads (BACON: len(output_layers), BLA: 1)
    bounds: sampling cube (min,max), set (-1,1) if you trained on [-1,1]^3
    """
    device = next(model.parameters()).device
    dt = torch.float32

    lo, hi = bounds
    if return_sdf:
        x = torch.arange(-N // 2, N // 2, dtype=dt) / N
    else:
        x = torch.linspace(lo, hi, N, dtype=dt)

    xx, yy, zz = torch.meshgrid(x, x, x, indexing='ij')  # avoid warning
    render_coords = torch.stack((xx.flatten(), yy.flatten(), zz.flatten()), dim=-1)  # CPU tensor

    # pre-alloc buffers (float32 for memory)
    sdf_values = [np.zeros((N**3, 1), dtype=np.float32) for _ in range(num_outputs)]

    # batched inference to save memory
    bsize = int(128 ** 2)
    total = render_coords.shape[0]
    for start in tqdm(range(0, total, bsize), desc=f"Raymarch {model_name}", ncols=80):
        end = min(start + bsize, total)
        coords = render_coords[start:end].to(device, non_blocking=True)
        out = model({'coords': coords})['model_out']
        if not isinstance(out, list):
            out = [out]

        for idx, sdf in enumerate(out):
            sdf_values[idx][start:end] = sdf.detach().float().cpu().numpy()

    if return_sdf:
        return [s.reshape(N, N, N) for s in sdf_values]

    # marching cubes per output
    os.makedirs('./outputs/meshes', exist_ok=True)
    for idx, s in enumerate(sdf_values):
        sdf = s.reshape(N, N, N)
        vertices, triangles = mcubes.marching_cubes(-sdf, 0)
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        mesh.vertices = (mesh.vertices / N - 0.5) + 0.5 / N
        mesh.export(f'./outputs/meshes/{model_name}_{idx+1}.obj')


# -----------------------------
# BACON adaptive helpers (only if you use model_type='bacon' and adaptive=True)
# -----------------------------
def _prepare_multi_scale(res, num_scales, device):
    def coord2ind(xyz_coord, res_):
        x, y, z = torch.split(xyz_coord, 1, dim=-1)
        flat_ind = x * (res_ ** 2) + y * res_ + z
        return flat_ind.squeeze(-1)

    shifts = torch.from_numpy(np.stack(np.mgrid[:2, :2, :2], axis=-1)).reshape(-1, 3)

    def subdiv_index(xyz_prev, next_res):
        xyz_next = xyz_prev.unsqueeze(1) * 2 + shifts  # (N^3)x8x3
        flat_ind_next = coord2ind(xyz_next, next_res)  # (N^3)*8
        return flat_ind_next

    lowest_res = res // (2 ** (num_scales - 1))
    subdiv_hash_list = []
    for i in range(num_scales - 1):
        curr_res = int(lowest_res * (2 ** i))
        xyz_ind = torch.from_numpy(
            np.stack(np.mgrid[:curr_res, :curr_res, :curr_res], axis=-1)
        ).reshape(-1, 3)
        subdiv_hash = subdiv_index(xyz_ind, curr_res * 2)
        subdiv_hash_list.append(subdiv_hash.to(device=device, dtype=torch.long))

    return subdiv_hash_list, lowest_res


@torch.no_grad()
def _compute_one_scale_bacon(model, layer_ind, render_coords, sdf_values, hash_ind, output_layers):
    assert len(render_coords) == len(hash_ind)
    device = next(model.parameters()).device
    bsize = int(128 ** 2)
    total = len(render_coords)
    for start in range(0, total, bsize):
        end = min(start + bsize, total)
        coords = render_coords[start:end, :].to(device, non_blocking=True)
        out = model({'coords': coords}, specified_layers=output_layers[layer_ind])['model_out']
        sdf_values[hash_ind[start:end]] = out[0]


@torch.no_grad()
def _compute_one_scale_adaptive_bacon(model, layer_ind, render_coords, sdf_values, hash_ind, output_layers, threshold):
    assert len(render_coords) == len(hash_ind)
    device = next(model.parameters()).device
    bsize = int(128 ** 2)
    total = len(render_coords)
    for start in range(0, total, bsize):
        end = min(start + bsize, total)
        coords = render_coords[start:end, :].to(device, non_blocking=True)
        out = model({'coords': coords}, specified_layers=2, get_feature=True)['model_out']
        sdf = out[0][0]
        if output_layers[layer_ind] > 2:
            feature = out[0][1]
            near_surf = (sdf.abs() < threshold).squeeze()
            if near_surf.any():
                coords_surf = coords[near_surf]
                feature_surf = feature[near_surf]
                out2 = model({'coords': coords_surf}, specified_layers=output_layers[layer_ind],
                             continue_layer=2, continue_feature=feature_surf)['model_out']
                sdf_near = out2[0]
                sdf[near_surf] = sdf_near
        sdf_values[hash_ind[start:end]] = sdf


@torch.no_grad()
def generate_mesh_adaptive_bacon(model, model_name, N, output_layers):
    """
    BACON-specific adaptive evaluation. Do not use with BLA.
    """
    device = next(model.parameters()).device
    num_outputs = len(output_layers)

    subdiv_hashes, lowest_res = _prepare_multi_scale(N, num_outputs, device=device)
    # Dummy index for the first stage (original code compatibility)
    subdiv_hashes = [torch.arange((N // 8) ** 3, device=device, dtype=torch.long)] + subdiv_hashes

    coords_list = [dataio.get_mgrid(lowest_res * (2 ** i), dim=3).to(device) for i in range(num_outputs)]
    sdf_out_list = [torch.zeros(((lowest_res * (2 ** i)) ** 3), 1, device=device) for i in range(num_outputs)]

    # level 0
    _compute_one_scale_bacon(model, 0, coords_list[0], sdf_out_list[0], subdiv_hashes[0], output_layers)

    # refinements
    for i in range(1, num_outputs):
        curr_res = int(lowest_res * (2 ** (i - 1)))
        pixel_len = 1.0 / curr_res
        threshold = (math.sqrt(2) * pixel_len * 0.5) * 2.0

        sdf_prev = sdf_out_list[i - 1]
        sdf_curr = sdf_out_list[i]
        hash_curr = subdiv_hashes[i]
        coords_curr = coords_list[i]
        near_surf_prev = (sdf_prev.abs() <= threshold).squeeze(-1)

        # empty
        sdf_curr[hash_curr[~near_surf_prev]] = sdf_prev[~near_surf_prev].unsqueeze(-1)

        # non-empty
        non_empty_ind = hash_curr[near_surf_prev].flatten()
        if i == num_outputs - 1:
            _compute_one_scale_adaptive_bacon(model, i, coords_curr[non_empty_ind], sdf_curr,
                                              non_empty_ind, output_layers, threshold=pixel_len * 0.5 * 2.0)
        else:
            _compute_one_scale_bacon(model, i, coords_curr[non_empty_ind], sdf_curr, non_empty_ind, output_layers)

    # MC
    sdf = sdf_curr.reshape(N, N, N).detach().float().cpu().numpy()
    vertices, triangles = mcubes.marching_cubes(-sdf, 0.0)
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    mesh.vertices = (mesh.vertices / N - 0.5) + 0.5 / N

    os.makedirs('./outputs/meshes', exist_ok=True)
    mesh.export(f'./outputs/meshes/{model_name}.obj')


# -----------------------------
# Model builder + exporter
# -----------------------------
def export_model(
    ckpt_path,
    model_name,
    N=512,
    model_type='bla',       # 'bla' or 'bacon'
    hidden_layers=3,            # BLA: hidden layer count (total = hidden_layers + 2)
    hidden_size=256,
    output_layers=None,         # BACON only: e.g. [2,4,6,8]
    return_sdf=False,
    adaptive=False,             # BACON only
    bounds=(-0.5, 0.5),         # sampling range
):
    with utils.HiddenPrint():
        if model_type == 'bla':
            ckpt = torch.load(ckpt_path, map_location='cuda')
            state = ckpt.get('state_dict', ckpt)
            if any(k.startswith('_orig_mod.') for k in state.keys()):
                state = {k[len('_orig_mod.'):]: v for k, v in state.items()}
            use_float = any('.act.' in k for k in state.keys())

            if use_float:
                from modules.bla_float import INR
            else:
                from modules.bla import INR
            import torch.nn as _nn
            class BLAforSDF(_nn.Module):
                def __init__(self):
                    super().__init__()
                    self.inr = INR(in_features=3, hidden_features=hidden_size,
                                   hidden_layers=hidden_layers, out_features=1,
                                   use_sigmoid=False)
                def forward(self, model_input):
                    return {'model_in': model_input, 'model_out': self.inr(model_input['coords'])}
            model = BLAforSDF()
            model.load_state_dict(state, strict=True)
            model = model.cuda().eval()
            generate_mesh(model, N, return_sdf, num_outputs=1, model_name=model_name, bounds=bounds)
        else:
            raise ValueError(f"Unknown model_type: {model_type} (only 'bla' supported)")


# -----------------------------
# Batch export wrapper
# -----------------------------
def export_meshes(
    ckpts,
    names,
    model_type='bla',
    N=512,
    hidden_layers=3,
    hidden_size=256,
    output_layers=None,
    adaptive=False,
    bounds=(-0.5, 0.5),
    return_sdf=False,
):
    assert len(ckpts) == len(names), "ckpts and names must have the same length."
    print(f'Exporting ({model_type})')
    for ckpt, name in tqdm(list(zip(ckpts, names)), total=len(ckpts), ncols=80):
        export_model(
            ckpt_path=ckpt,
            model_name=name,
            N=N,
            model_type=model_type,
            hidden_layers=hidden_layers,
            hidden_size=hidden_size,
            output_layers=output_layers,
            return_sdf=return_sdf,
            adaptive=adaptive,
            bounds=bounds,
        )


# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--name', type=str, required=True)
    ap.add_argument('--N', type=int, default=512)
    ap.add_argument('--hidden_layers', type=int, default=3)
    ap.add_argument('--hidden_size', type=int, default=256)
    a = ap.parse_args()

    export_meshes(
        ckpts=[a.ckpt],
        names=[a.name],
        model_type='bla',
        N=a.N,
        hidden_layers=a.hidden_layers,
        hidden_size=a.hidden_size,
        adaptive=False,
    )
