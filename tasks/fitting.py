#!/usr/bin/env python
"""
Image fitting task.

  python tasks/fitting.py --image_path ./folder/data/div2k/00.png
  python tasks/fitting.py --image_path ./folder/data/div2k/00.png --use_wege

--use_wege off : in_features=2, coords = (x, y)
--use_wege on  : in_features=3, coords = (x, y, wege_score)
"""

import os
import sys
import math
import random
import argparse
import importlib

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from pytorch_msssim import ssim
import lpips

# allow running as `python tasks/fitting.py` from FLAIR root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import models, utils
from modules.speed import setup_fast_env, maybe_compile, resolve_fast_nonlin
from wege import WEGEExtractor

setup_fast_env()  # TF32 on (no-op for AMP/compile until --fast)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def soft_clip(wb, percentile=99.5, alpha=0.7):
    pval = np.percentile(wb, percentile)
    return np.where(wb > pval, pval + alpha * (wb - pval), wb)


def build_wege_map(im, H, W, device, J=1, wave='db3', radius=6, eps_gf=1e-5):
    """Compute WEGE pixel-wise score, apply guided filter + soft-clip, return tensor map (H*W, 1)."""
    extractor = WEGEExtractor(img_size=(H, W), J=J, wave=wave).to(device)
    im_torch = torch.tensor(im).permute(2, 0, 1)[None, ...].to(device)
    with torch.no_grad():
        wb_pixel = extractor.get_pixelwise_energy_scores(im_torch)
        wb_pixel_np = wb_pixel.cpu().numpy().astype(np.float32)

        guide_img = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) if im.shape[2] == 3 else im
        wb_guided = cv2.ximgproc.guidedFilter(
            guide=guide_img.astype(np.float32),
            src=wb_pixel_np,
            radius=radius,
            eps=eps_gf,
        )
        wb_guided = soft_clip(wb_guided, percentile=99.5, alpha=0)
        wb_guided = np.clip(wb_guided, 0, 1)

        wb_tensor = torch.tensor(wb_guided, device=device)
        wege_map = WEGEExtractor.expand_energy_scores_to_coords_pixelwise(H, W, wb_tensor)
    return wege_map  # (H*W, 1)


def init_weights_fn(init_type):
    def _init(m):
        if isinstance(m, nn.Linear):
            fan_in = m.weight.shape[1]
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            if init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight)
            elif init_type == 'kaiming_uniform':
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif init_type == 'kaiming_normal':
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight)
            elif init_type == 'uniform':
                bound = 1.0 / math.sqrt(fan_in)
                nn.init.uniform_(m.weight, -bound, bound)
            elif init_type == 'normal':
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
    return _init


def main():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--nonlin',          type=str,   default='bla')
    parser.add_argument('--sigma0',          type=float, default=2.0)
    parser.add_argument('--omega0',          type=float, default=8.0)
    parser.add_argument('--fomega',          type=float, default=30.0)
    parser.add_argument('--hidden_layers',   type=int,   default=4)
    parser.add_argument('--hidden_features', type=int,   default=256)
    parser.add_argument('--init_T',     '--T', dest='init_T',     type=float, default=1.0)
    parser.add_argument('--init_beta',  '--B', dest='init_beta',  type=float, default=0.05)
    parser.add_argument('--init_zeta',  '--Z', dest='init_zeta',  type=float, default=1.0)
    parser.add_argument('--init_type',  type=str, default='default',
                        choices=['default', 'xavier_uniform', 'kaiming_uniform', 'kaiming_normal',
                                 'orthogonal', 'uniform', 'normal'],
                        help="'default' = no override (each model's own init / PyTorch nn.Linear default)")
    parser.add_argument('--no_sigmoid', action='store_true',
                        help='(BLA only) disable final sigmoid — match SIREN/WIRE/Gauss output range')
    # WEGE
    parser.add_argument('--use_wege',  action='store_true',
                        help='if set, concat WEGE score as 3rd coord channel')
    parser.add_argument('--wege_J',    type=int, default=1)
    parser.add_argument('--wege_wave', type=str, default='db3')
    # data / training
    parser.add_argument('--image_path', type=str, default='./folder/data/div2k/00.png')
    parser.add_argument('--scale_im',     type=float, default=1.0)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--niters',        type=int,   default=5000)
    parser.add_argument('--seed',          type=int,   default=42)
    parser.add_argument('--save_every',    type=int,   default=500)
    parser.add_argument('--fast', action='store_true',
                        help='torch.compile model (recommend --nonlin bla_float for compile-compatible)')
    args = parser.parse_args()
    args.nonlin = resolve_fast_nonlin(args.nonlin, args.fast)

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load image
    im = utils.normalize(plt.imread(args.image_path).astype(np.float32), True)
    im = cv2.resize(im, None, fx=args.scale_im, fy=args.scale_im, interpolation=cv2.INTER_AREA)
    H, W, _ = im.shape

    posencode = (args.nonlin == 'posenc')
    nonlin = 'relu' if posencode else args.nonlin
    sidelength = int(max(H, W)) if posencode else H

    in_features = 3 if args.use_wege else 2

    model = models.get_INR(
        nonlin=nonlin,
        in_features=in_features,
        out_features=3,
        hidden_features=args.hidden_features,
        hidden_layers=args.hidden_layers,
        first_omega_0=args.fomega,
        hidden_omega_0=args.omega0,
        scale=args.sigma0,
        pos_encode=posencode,
        sidelength=sidelength,
        init_T=args.init_T,
        init_beta=args.init_beta,
        init_zeta=args.init_zeta,
        use_sigmoid=(not args.no_sigmoid),
    )
    if args.init_type != 'default':
        model.apply(init_weights_fn(args.init_type))
    model.cuda()
    model = maybe_compile(model, args.fast)

    print('Number of parameters:', utils.count_parameters(model))

    optim = torch.optim.Adam(lr=args.learning_rate, params=model.parameters())
    scheduler = LambdaLR(optim, lambda x: 0.2 ** min(x / args.niters, 1))

    # coords
    x = torch.linspace(-1, 1, W).cuda()
    y = torch.linspace(-1, 1, H).cuda()
    X, Y = torch.meshgrid(x, y, indexing='xy')
    coords = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))[None, ...]  # (1, H*W, 2)

    if args.use_wege:
        wege_map = build_wege_map(im, H, W, device, J=args.wege_J, wave=args.wege_wave)
        coords = torch.cat([coords, wege_map[None, ...].to(coords.device)], dim=2)  # (1, H*W, 3)

    gt = torch.tensor(im).cuda().reshape(H * W, 3)[None, ...]
    im_gt = gt.reshape(H, W, 3).permute(2, 0, 1)[None, ...]

    lpips_model = lpips.LPIPS(net='alex').to(device).eval()

    mse_array   = torch.zeros(args.niters, device='cuda')
    ssim_array  = torch.zeros(args.niters, device='cuda')
    lpips_array = torch.zeros(args.niters, device='cuda')

    best_mse = float('inf'); best_img = None
    best_psnr = 0.0; best_ssim = 0.0; best_lpips = 1.0; best_epoch = 0

    tag = 'wege' if args.use_wege else 'no_wege'
    sig_tag = 'no_sigmoid' if args.no_sigmoid else 'sigmoid'
    result_root = (f'results/fitting/{tag}/{sig_tag}/init={args.init_type}/'
                   f'{args.nonlin}/{args.image_path}/hf={args.hidden_features}')
    os.makedirs(result_root, exist_ok=True)

    tbar = tqdm(range(args.niters))
    for epoch in tbar:
        pred = model(coords)
        loss = ((pred - gt) ** 2).mean()
        optim.zero_grad(); loss.backward(); optim.step(); scheduler.step()

        with torch.no_grad():
            im_rec = pred.reshape(H, W, 3).permute(2, 0, 1)[None, ...]
            mse_array[epoch] = ((gt - pred) ** 2).mean().item()
            ssim_array[epoch] = ssim(im_gt, im_rec, data_range=1, size_average=True)
            lpips_array[epoch] = lpips_model(im_rec * 2 - 1, im_gt * 2 - 1).item()

        psnr     = -10 * torch.log10(mse_array[epoch]).item()
        ssim_val = ssim_array[epoch].item()
        lpips_val = lpips_array[epoch].item()
        tbar.set_description('PSNR: %.2f | SSIM: %.4f | LPIPS: %.4f' % (psnr, ssim_val, lpips_val))

        imrec = im_rec.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        if mse_array[epoch] < best_mse:
            best_mse = mse_array[epoch]
            best_img = imrec
            best_psnr = psnr; best_ssim = ssim_val; best_lpips = lpips_val
            best_epoch = epoch + 1

        if ((epoch + 1) % args.save_every == 0) or (epoch == args.niters - 1):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_mse': best_mse,
                'best_img': best_img,
                'mse_array': mse_array,
                'ssim_array': ssim_array,
                'lpips_array': lpips_array,
            }, os.path.join(result_root, f'ckpt_{args.nonlin}_{epoch+1}.pt'))

            plt.imsave(os.path.join(result_root, f'best_{args.nonlin}_{epoch+1}.png'),
                       np.clip(best_img, 0, 1))

            with open(os.path.join(result_root, f'score_history.txt'), 'a') as f:
                f.write(f"=== EPOCH {epoch+1} ===\n")
                f.write(f"[Overall BEST] @ {best_epoch} | PSNR={best_psnr:.4f} SSIM={best_ssim:.4f} LPIPS={best_lpips:.4f}\n")

            for arr, name in [(mse_array, 'psnr'), (ssim_array, 'ssim'), (lpips_array, 'lpips')]:
                vals = arr.detach().cpu().numpy()
                vals = -10 * np.log10(np.maximum(vals, 1e-12)) if name == 'psnr' else vals
                plt.plot(vals); plt.title(name.upper()); plt.xlabel('Epoch')
                plt.savefig(os.path.join(result_root, f'curve_{name}.png')); plt.clf()


if __name__ == '__main__':
    main()
