#!/usr/bin/env python
"""
3D occupancy reconstruction task.

  python tasks/occupancy.py --expname lucy
  python tasks/occupancy.py --expname lucy --nonlin wire

Note: occupancy uses 3D coords (x, y, z) — WEGE (2D wavelet) is not applicable here.
"""

import os
import sys
import time
import copy
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
plt.gray()
from scipy import io, ndimage
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import models, utils, volutils
from modules.speed import setup_fast_env, maybe_compile, resolve_fast_nonlin

setup_fast_env()


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nonlin',          type=str,   default='bla')
    parser.add_argument('--sigma0',          type=float, default=20.0)
    parser.add_argument('--omega0',          type=float, default=10.0)
    parser.add_argument('--fomega',          type=float, default=30.0)
    parser.add_argument('--scale',           type=float, default=1.0)
    parser.add_argument('--learning_rate',   type=float, default=5e-4)
    parser.add_argument('--hidden_layers',   type=int,   default=4)
    parser.add_argument('--hidden_features', type=int,   default=256)
    parser.add_argument('--niters',          type=int,   default=500)
    parser.add_argument('--expname',         type=str,   required=True)
    parser.add_argument('--maxpoints',       type=int,   default=int(2e5))
    parser.add_argument('--mcubes_thres',    type=float, default=0.5)
    parser.add_argument('--save_interval',   type=int,   default=100)
    parser.add_argument('--init_T',    '--T', dest='init_T',    type=float, default=1.0)
    parser.add_argument('--init_beta', '--B', dest='init_beta', type=float, default=0.05)
    parser.add_argument('--init_zeta', '--Z', dest='init_zeta', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    # SL2A passthrough
    parser.add_argument('--deg', type=int, default=256)
    parser.add_argument('--rank', type=int, default=32)
    parser.add_argument('--nonlinearity', type=str, default='relu')
    parser.add_argument('--init_method', type=str, default='xavier_uniform')
    parser.add_argument('--linear_init_type', type=str, default='kaiming_uniform')
    parser.add_argument('--fast', action='store_true',
                        help='torch.compile model (recommend --nonlin bla_float)')
    args = parser.parse_args()
    args.nonlin = resolve_fast_nonlin(args.nonlin, args.fast)

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    im = io.loadmat(f'./folder/data/{args.expname}.mat')['hypercube'].astype(np.float32)
    im = ndimage.zoom(im / im.max(), [args.scale] * 3, order=0)

    occupancy = True
    if occupancy:
        hidx, widx, tidx = np.where(im > 0.99)
        im = im[hidx.min():hidx.max(), widx.min():widx.max(), tidx.min():tidx.max()]

    H, W, T = im.shape
    maxpoints = min(H * W * T, args.maxpoints)
    imten = torch.tensor(im, device=device).reshape(H * W * T, 1)
    coords = utils.get_coords(H, W, T).to(device)

    model = models.get_INR(
        nonlin=args.nonlin,
        in_features=3,
        out_features=1,
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        first_omega_0=args.omega0,
        hidden_omega_0=args.omega0,
        scale=args.sigma0,
        pos_encode=False,
        sidelength=max(H, W, T),
        init_T=args.init_T,
        init_beta=args.init_beta,
        init_zeta=args.init_zeta,
        deg=args.deg,
        rank=args.rank,
        nonlinearity=args.nonlinearity,
        init_method=args.init_method,
        linear_init_type=args.linear_init_type,
    ).to(device)
    model = maybe_compile(model, args.fast)
    print('Number of parameters:', utils.count_parameters(model))

    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = LambdaLR(optim, lambda x: 0.2 ** min(x / args.niters, 1))
    criterion = nn.MSELoss()

    mse_array  = np.zeros(args.niters)
    psnr_array = np.zeros(args.niters)
    time_array = np.zeros(args.niters)

    best_mse = float('inf'); best_img = None
    best_psnr = 0.0; best_epoch = 0

    result_root = f'results/occupancy/{args.nonlin}/{args.expname}/sigma{args.sigma0}'
    os.makedirs(result_root, exist_ok=True)

    im_estim = torch.zeros((H * W * T, 1), device=device)
    tic = time.time()
    tbar = tqdm(range(args.niters))

    for idx in tbar:
        indices = torch.randperm(H * W * T, device=device)
        train_loss = 0; nchunks = 0
        for b in range(0, H * W * T, maxpoints):
            bidx = indices[b:min(H * W * T, b + maxpoints)]
            b_coords = coords[bidx, ...]
            pred = model(b_coords[None, ...]).squeeze()[:, None]
            with torch.no_grad():
                im_estim[bidx, :] = pred
            loss = criterion(pred, imten[bidx, :])
            optim.zero_grad(); loss.backward(); optim.step()
            train_loss += loss.item(); nchunks += 1
        scheduler.step()

        lossval = train_loss / nchunks
        im_estim_vol = im_estim.reshape(H, W, T)
        mse_array[idx] = lossval
        psnr = utils.psnr(im, im_estim_vol.detach().cpu().numpy())
        psnr_array[idx] = psnr
        time_array[idx] = time.time() - tic

        if lossval < best_mse:
            best_mse = lossval
            best_img = copy.deepcopy(im_estim)
            best_psnr = psnr; best_epoch = idx + 1

        tbar.set_description(f'PSNR={psnr:.2f}dB')

        if ((idx + 1) % args.save_interval == 0) or (idx == args.niters - 1):
            best_img_np = best_img.reshape(H, W, T).detach().cpu().numpy()
            np.save(os.path.join(result_root, f'best_{args.nonlin}_{idx+1}.npy'), best_img_np)
            with open(os.path.join(result_root, 'score_history.txt'), 'a') as f:
                f.write(f"=== EPOCH {idx+1} ===\n")
                f.write(f"[Overall BEST] @ {best_epoch} | PSNR={best_psnr:.4f} MSE={best_mse:.6f}\n")

    best_img_np = best_img.reshape(H, W, T).detach().cpu().numpy()
    io.savemat(os.path.join(result_root, f'{args.nonlin}.mat'), {
        'mse_array': mse_array, 'psnr_array': psnr_array, 'time_array': time_array,
        'best_img': best_img_np, 'gt': im,
    })

    total_time = time.time() - tic
    nparams = utils.count_parameters(model)
    with open(os.path.join(result_root, 'metrics.txt'), 'w') as f:
        f.write(f'Total time (minutes): {total_time / 60:.2f}\n')
        f.write(f'Total parameters (millions): {nparams / 1e6:.2f}\n')
        if occupancy:
            iou = volutils.get_IoU(best_img_np, im, args.mcubes_thres)
            f.write(f'IoU: {iou:.6f}\n'); print('IoU:', iou)
        else:
            f.write(f'PSNR: {best_psnr:.6f}\n'); print('PSNR:', best_psnr)

    print(f'Total time {total_time / 60:.2f} minutes | params: {nparams / 1e6:.2f}M')


if __name__ == '__main__':
    main()
