"""
Sample an (x, y, z, nx, ny, nz) point cloud from a watertight .ply mesh so the
training data and the evaluation reference (Stanford .ply) live in the same
coordinate frame. This replaces the BACON-distributed .xyz files for scenes
where we want chamfer/IoU eval (see tasks/sdf/eval.py).

Usage:
    python sample_xyz_from_ply.py --ply data/dragon_vrip.ply --out data/gt_dragon.xyz
    python sample_xyz_from_ply.py --ply data/Armadillo.ply  --out data/gt_armadillo.xyz
"""

import argparse
import os

import numpy as np
import trimesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ply', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_samples', type=int, default=200000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    mesh = trimesh.load(args.ply, force='mesh', process=False, skip_materials=True)
    print(f'[load] {args.ply}: {len(mesh.vertices)} verts, {len(mesh.faces)} faces')

    pts, face_idx = trimesh.sample.sample_surface(mesh, args.n_samples, seed=int(rng.integers(2**31)))
    normals = mesh.face_normals[face_idx]

    out = np.hstack([pts, normals]).astype(np.float32)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.savetxt(args.out, out, fmt='%.6f')
    print(f'[write] {args.out}: {out.shape[0]} points, bounds {pts.min(0)} -> {pts.max(0)}')


if __name__ == '__main__':
    main()
