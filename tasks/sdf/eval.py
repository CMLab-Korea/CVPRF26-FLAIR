"""
Chamfer + IoU evaluation for SDF reconstructions.

Workflow (matches the BACON eval pipeline):
  1. Take the model's predicted mesh (.obj from render_sdf.py).
  2. Take the matching Stanford ground-truth mesh (.ply, see download_datasets.py --eval).
  3. Use the training .xyz point cloud to recover the scale/offset that
     `dataio.MeshSDF.normalize` applied during training, then apply the same
     transform to the .ply so it lives in the model's coordinate frame.
  4. Compute symmetric chamfer (mesh surface samples) and IoU on a 128^3 grid.

Usage:
  python eval.py --pred outputs/meshes/dragon_bla_full_1.obj --scene dragon
  python eval.py --pred outputs/meshes/armadillo_bla_fast_1.obj --scene armadillo

Add a new scene by extending SCENE_TO_PLY.
"""

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_THIS_DIR, 'lib', 'inside_mesh'))

import numpy as np
import trimesh
from metrics import compute_iou, compute_trimesh_chamfer  # bacon's inside_mesh helpers

SCENE_TO_PLY = {
    'armadillo': 'Armadillo.ply',
    'dragon':    'dragon_vrip.ply',
    'thai':      'xyzrgb_statuette.ply',
}


def normalize_params(coords, scaling=0.9):
    """Replicates dataio.MeshSDF.normalize: center -> scale into [-0.45, 0.45]^3."""
    coords = np.asarray(coords, dtype=np.float64).copy()
    cmean = np.mean(coords, axis=0, keepdims=True)
    coords -= cmean
    cmin, cmax = np.amin(coords), np.amax(coords)
    scale = scaling / (cmax - cmin)
    offset = -scaling * (cmin) / (cmax - cmin) - 0.5 * scaling - scale * cmean.squeeze()
    return scale, offset


def make_ref_mesh(xyz_path, ply_path, out_path):
    pc = np.loadtxt(xyz_path)[:, :3]
    scale, offset = normalize_params(pc)
    mesh = trimesh.load(ply_path, force='mesh', process=False, skip_materials=True)
    mesh.vertices = mesh.vertices * scale + offset
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    mesh.export(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True, help='predicted .obj mesh from render_sdf.py')
    ap.add_argument('--scene', required=True, choices=sorted(SCENE_TO_PLY.keys()))
    ap.add_argument('--xyz', default=None, help='training .xyz (default: data/gt_<scene>.xyz)')
    ap.add_argument('--ply', default=None, help='ground-truth .ply (default: data/<canonical>.ply)')
    ap.add_argument('--ref_dir', default='outputs/shapes', help='dir for normalized ref mesh')
    ap.add_argument('--iou_N', type=int, default=128)
    ap.add_argument('--cham_samples', type=int, default=300000)
    args = ap.parse_args()

    xyz_path = args.xyz or os.path.join('data', f'gt_{args.scene}.xyz')
    ply_path = args.ply or os.path.join('data', SCENE_TO_PLY[args.scene])
    ref_path = os.path.join(args.ref_dir, f'ref_{args.scene}.obj')

    for p, kind in [(args.pred, 'predicted mesh'), (xyz_path, 'training xyz'), (ply_path, 'ground-truth .ply')]:
        if not os.path.exists(p):
            raise SystemExit(f'missing {kind}: {p}')

    if not os.path.exists(ref_path):
        print(f'[ref] {ply_path} -> {ref_path} (normalized to training frame)')
        make_ref_mesh(xyz_path, ply_path, ref_path)
    else:
        print(f'[ref] reusing {ref_path}')

    mesh_pred = trimesh.load(args.pred, force='mesh', process=False, skip_materials=True)
    mesh_ref = trimesh.load(ref_path, force='mesh', process=False, skip_materials=True)

    print(f'[metrics] computing on {args.pred} vs {ref_path}')
    chamfer = compute_trimesh_chamfer(mesh_pred, mesh_ref, num_mesh_samples=args.cham_samples)
    iou = compute_iou(ref_path, args.pred, N=args.iou_N)

    print(f'  Chamfer (sym, squared L2 sum): {chamfer:.6e}')
    print(f'  IoU @ {args.iou_N}^3 grid:       {iou:.6f}')


if __name__ == '__main__':
    main()
