#!/usr/bin/env python
"""
CT reconstruction task.

  python tasks/ct.py --image_path ./folder/data/chest.png
  python tasks/ct.py --image_path ./folder/data/chest.png --use_wege
  python tasks/ct.py --image_path ./folder/data/chest.png --nonlin wire
"""

import os
import sys
import time
import argparse

import numpy as np
import torch
import cv2
import lpips
import matplotlib.pyplot as plt
plt.gray()
from scipy import io
from skimage.metrics import structural_similarity as ssim_func
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import models, utils, lin_inverse
from modules.speed import setup_fast_env, maybe_compile, resolve_fast_nonlin
from wege import WEGEExtractor

setup_fast_env()


def build_wege_map_from_gray(imten, H, W, device, J=1, wave='db3'):
    extractor = WEGEExtractor(img_size=(H, W), J=J, wave=wave).to(device)
    with torch.no_grad():
        wb_pixel = extractor.get_pixelwise_energy_scores(imten)
        wb_np = wb_pixel.cpu().numpy().astype(np.float32)
        wb_np = cv2.bilateralFilter(wb_np, d=9, sigmaColor=0.2, sigmaSpace=7)
        wb_t = torch.tensor(wb_np, device=device)
    return WEGEExtractor.expand_energy_scores_to_coords_pixelwise(H, W, wb_t)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nonlin',          type=str,   default='bla')
    parser.add_argument('--sigma0',          type=float, default=5.0)
    parser.add_argument('--omega0',          type=float, default=10.0)
    parser.add_argument('--learning_rate',   type=float, default=5e-4)
    parser.add_argument('--hidden_layers',   type=int,   default=4)
    parser.add_argument('--hidden_features', type=int,   default=256)
    parser.add_argument('--niters',          type=int,   default=50000)
    parser.add_argument('--nmeas',           type=int,   default=100)
    parser.add_argument('--tau',             type=float, default=3e1)
    parser.add_argument('--noise_snr',       type=float, default=2.0)
    parser.add_argument('--image_path',      type=str,   required=True)
    parser.add_argument('--init_T',    '--T', dest='init_T',    type=float, default=1.0)
    parser.add_argument('--init_beta', '--B', dest='init_beta', type=float, default=0.05)
    parser.add_argument('--init_zeta', '--Z', dest='init_zeta', type=float, default=0.1)
    parser.add_argument('--use_wege',  action='store_true')
    parser.add_argument('--wege_J',    type=int, default=1)
    parser.add_argument('--wege_wave', type=str, default='db3')
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--fast', action='store_true',
                        help='torch.compile model (recommend --nonlin bla_float)')
    args = parser.parse_args()
    args.nonlin = resolve_fast_nonlin(args.nonlin, args.fast)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    thetas = torch.tensor(np.linspace(0, 180, args.nmeas, dtype=np.float32)).cuda()

    img = cv2.imread(args.image_path).astype(np.float32)[..., 1]
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    img = utils.normalize(img, True)
    H, W = img.shape
    imten = torch.tensor(img)[None, None, ...].cuda()

    posencode = (args.nonlin == 'posenc')
    nonlin = 'relu' if posencode else args.nonlin
    in_features = 3 if args.use_wege else 2

    model = models.get_INR(
        nonlin=nonlin,
        in_features=in_features,
        out_features=1,
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        first_omega_0=args.omega0,
        hidden_omega_0=args.omega0,
        scale=args.sigma0,
        pos_encode=posencode,
        sidelength=args.nmeas,
        init_T=args.init_T,
        init_beta=args.init_beta,
        init_zeta=args.init_zeta,
    ).cuda()
    model = maybe_compile(model, args.fast)

    x = torch.linspace(-1, 1, W).cuda()
    y = torch.linspace(-1, 1, H).cuda()
    X, Y = torch.meshgrid(x, y, indexing='xy')
    coords = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))

    if args.use_wege:
        wege_map = build_wege_map_from_gray(imten, H, W, device, J=args.wege_J, wave=args.wege_wave)
        coords = torch.cat([coords, wege_map], dim=1)
    coords = coords[None, ...]

    with torch.no_grad():
        sinogram = lin_inverse.radon(imten, thetas).detach().cpu().numpy()
        sinogram_ten = torch.tensor(sinogram).cuda()

    optim = torch.optim.Adam(lr=args.learning_rate, params=model.parameters())
    scheduler = LambdaLR(optim, lambda x: 0.1 ** min(x / args.niters, 1))

    mse_array   = np.zeros(args.niters)
    psnr_array  = np.zeros(args.niters)
    ssim_array  = np.zeros(args.niters)
    lpips_array = np.zeros(args.niters)
    time_array  = np.zeros(args.niters)
    lpips_model = lpips.LPIPS(net='alex').to(device).eval()

    best_mse = float('inf'); best_img = None
    best_psnr = 0.0; best_ssim = 0.0; best_lpips = 1.0; best_epoch = 0

    tag = 'wege' if args.use_wege else 'no_wege'
    result_root = (f'results/ct/{tag}/{args.nonlin}/'
                   f'sigma{args.sigma0}_T{args.init_T}_B{args.init_beta}_Z{args.init_zeta}'
                   f'_nmeas{args.nmeas}')
    os.makedirs(result_root, exist_ok=True)

    init_time = time.time()
    tbar = tqdm(range(args.niters))
    for idx in tbar:
        img_estim = model(coords).reshape(-1, H, W)[None, ...]
        sinogram_estim = lin_inverse.radon(img_estim, thetas)
        loss = ((sinogram_ten - sinogram_estim) ** 2).mean()
        optim.zero_grad(); loss.backward(); optim.step(); scheduler.step()

        with torch.no_grad():
            img_estim_cpu = img_estim.detach().cpu().squeeze().numpy()
            loss_gt = ((img_estim - imten) ** 2).mean().item()
            mse_array[idx] = loss_gt
            psnr = -10 * np.log10(loss_gt + 1e-10); psnr_array[idx] = psnr
            ssim_val = ssim_func(img, img_estim_cpu, data_range=1.0)
            ssim_array[idx] = ssim_val
            img_gt_  = torch.tensor(img).float()[None, ...].repeat(3, 1, 1)
            img_est_ = torch.tensor(img_estim_cpu).float()[None, ...].repeat(3, 1, 1)
            lpips_val = lpips_model(
                img_est_.unsqueeze(0).to(device) * 2 - 1,
                img_gt_.unsqueeze(0).to(device) * 2 - 1
            ).item()
            lpips_array[idx] = lpips_val
            time_array[idx] = time.time() - init_time

            if loss_gt < best_mse:
                best_mse = loss_gt; best_img = img_estim_cpu
                best_psnr = psnr; best_ssim = ssim_val; best_lpips = lpips_val
                best_epoch = idx + 1

            tbar.set_description(f'PSNR={psnr:.2f}dB')

        if ((idx + 1) % args.save_every == 0) or (idx == args.niters - 1):
            plt.imsave(os.path.join(result_root, f'best_{args.nonlin}_{idx+1}.png'),
                       np.clip(best_img, 0, 1))
            with open(os.path.join(result_root, 'score_history.txt'), 'a') as f:
                f.write(f"=== EPOCH {idx+1} ===\n")
                f.write(f"[Overall BEST] @ {best_epoch} | PSNR={best_psnr:.4f} SSIM={best_ssim:.4f} LPIPS={best_lpips:.4f}\n")

    io.savemat(os.path.join(result_root, f'{args.nonlin}_{args.nmeas}.mat'), {
        'rec': best_img, 'gt': img,
        'mse_array': mse_array, 'psnr_array': psnr_array,
        'ssim_array': ssim_array, 'lpips_array': lpips_array,
        'time_array': time_array, 'sinogram': sinogram,
    })
    print(f'Best PSNR: {best_psnr:.2f} dB | SSIM: {best_ssim:.4f} | LPIPS: {best_lpips:.4f}')


if __name__ == '__main__':
    main()
