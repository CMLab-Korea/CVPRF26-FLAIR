#!/usr/bin/env python
"""
Single-image super-resolution task.

  python tasks/sr.py --image_path ./folder/data/butterfly.png
  python tasks/sr.py --image_path ./folder/data/butterfly.png --use_wege
  python tasks/sr.py --image_path ./folder/data/butterfly.png --nonlin wire

Uses cv2.ximgproc.guidedFilter for WEGE smoothing (matching tasks/fitting.py).
WEGE is computed on im_lr (the model's LR input) and upsampled to HR for concat.
Using HR for WEGE would leak GT info (SR cheating).

"""

import os
import sys
import argparse

import numpy as np
import torch
import cv2
import lpips
import matplotlib.pyplot as plt
from scipy import io
from skimage.metrics import structural_similarity as ssim_func
from torch.optim.lr_scheduler import LambdaLR
from pytorch_msssim import ssim
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import models, utils
from modules.speed import setup_fast_env, maybe_compile, resolve_fast_nonlin
from wege import WEGEExtractor

setup_fast_env()


def soft_clip(wb, percentile=99.5, alpha=0.5):
    pval = np.percentile(wb, percentile)
    return np.where(wb > pval, pval + alpha * (wb - pval), wb)


def build_wege_map_for_sr(im_lr, H, W, device,
                          J=1, wave='db3', radius=6, eps_gf=1e-3, sc_alpha=0.5):
    """
    WEGE for SR — IMPORTANT: computed on im_lr (the model's LR input), NOT on HR.
    Using HR or any downsampling of HR would leak ground-truth info into the
    coordinate input → SR cheating. The map is computed at LR resolution and
    then bicubic-upsampled to (H, W) for concat with HR coords.
    """
    Hd, Wd, _ = im_lr.shape
    extractor = WEGEExtractor(img_size=(Hd, Wd), J=J, wave=wave).to(device)
    im_t = torch.tensor(im_lr).permute(2, 0, 1)[None, ...].to(device)
    with torch.no_grad():
        wb_pixel = extractor.get_pixelwise_energy_scores(im_t)
        wb_np = wb_pixel.cpu().numpy().astype(np.float32)
        guide = cv2.cvtColor(im_lr, cv2.COLOR_RGB2GRAY) if im_lr.shape[2] == 3 else im_lr
        wb_np = cv2.ximgproc.guidedFilter(guide=guide.astype(np.float32),
                                          src=wb_np, radius=radius, eps=eps_gf)
        wb_np = np.clip(soft_clip(wb_np, 99.5, sc_alpha), 0, 1)
        wb_np_hr = cv2.resize(wb_np, (W, H), interpolation=cv2.INTER_CUBIC)
        wb_t_hr = torch.tensor(wb_np_hr, device=device)
        return WEGEExtractor.expand_energy_scores_to_coords_pixelwise(H, W, wb_t_hr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nonlin',          type=str,   default='bla')
    parser.add_argument('--sigma0',          type=float, default=2.0)
    parser.add_argument('--omega0',          type=float, default=8.0)
    parser.add_argument('--fomega',          type=float, default=30.0)
    parser.add_argument('--scale',           type=int,   default=4, help='SR upscale factor')
    parser.add_argument('--scale_im',        type=float, default=1/4, help='initial downsample for memory')
    # (removed: --down_factor — WEGE now uses im_lr, not a separate downsample of HR)
    parser.add_argument('--learning_rate',   type=float, default=5e-4)
    parser.add_argument('--hidden_layers',   type=int,   default=4)
    parser.add_argument('--hidden_features', type=int,   default=256)
    parser.add_argument('--niters',          type=int,   default=5000)
    parser.add_argument('--image_path',      type=str,   required=True)
    parser.add_argument('--init_T',    '--T', dest='init_T',    type=float, default=1.0)
    parser.add_argument('--init_beta', '--B', dest='init_beta', type=float, default=0.05)
    parser.add_argument('--init_zeta', '--Z', dest='init_zeta', type=float, default=0.1)
    parser.add_argument('--use_wege',  action='store_true')
    parser.add_argument('--wege_J',    type=int, default=1)
    parser.add_argument('--wege_wave', type=str, default='db3')
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--fast', action='store_true',
                        help='torch.compile model (recommend --nonlin bla_float)')
    args = parser.parse_args()
    args.nonlin = resolve_fast_nonlin(args.nonlin, args.fast)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    im = utils.normalize(plt.imread(args.image_path).astype(np.float32), True)
    im = cv2.resize(im, None, fx=args.scale_im, fy=args.scale_im, interpolation=cv2.INTER_AREA)
    H, W, _ = im.shape
    im = im[:args.scale * (H // args.scale), :args.scale * (W // args.scale), :]
    H, W, _ = im.shape

    im_lr = cv2.resize(im, None, fx=1/args.scale, fy=1/args.scale, interpolation=cv2.INTER_AREA)
    H2, W2, _ = im_lr.shape
    im_bi = cv2.resize(im_lr, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_LINEAR)

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
    ).cuda()
    model = maybe_compile(model, args.fast)

    optim = torch.optim.Adam(lr=args.learning_rate, params=model.parameters())
    scheduler = LambdaLR(optim, lambda x: 0.5 ** min(x / args.niters, 1))

    x_hr = torch.linspace(-1, 1, W).cuda(); y_hr = torch.linspace(-1, 1, H).cuda()
    X_hr, Y_hr = torch.meshgrid(x_hr, y_hr, indexing='xy')
    coords_hr = torch.hstack((X_hr.reshape(-1, 1), Y_hr.reshape(-1, 1)))[None, ...]

    if args.use_wege:
        # WEGE computed on im_lr (the model's actual LR input), then upsampled to HR.
        # Using HR (or any HR-derived downsampling) would leak GT info → SR cheating.
        wb_map = build_wege_map_for_sr(im_lr, H, W, device,
                                       J=args.wege_J, wave=args.wege_wave)
        coords_hr = torch.cat([coords_hr, wb_map[None, ...].to(coords_hr.device)], dim=2)

    gt    = torch.tensor(im).cuda().reshape(H * W, 3)[None, ...]
    gt_lr = torch.tensor(im_lr).cuda().reshape(H2 * W2, 3)[None, ...]
    im_gt = gt.reshape(H, W, 3).permute(2, 0, 1)[None, ...]
    im_bi_ten = torch.tensor(im_bi).cuda().permute(2, 0, 1)[None, ...]

    print(f'Bicubic baseline — PSNR: {utils.psnr(im, im_bi):.2f} | '
          f'SSIM: {ssim_func(im, im_bi, channel_axis=-1, data_range=1.0):.4f}')

    lpips_model = lpips.LPIPS(net='alex').to(device).eval()
    lpips_bicubic = lpips_model(im_bi_ten * 2 - 1,
                                torch.tensor(im).cuda().permute(2, 0, 1)[None, ...] * 2 - 1).item()
    print(f'Bicubic LPIPS: {lpips_bicubic:.4f}')

    mse_array   = torch.zeros(args.niters, device='cuda')
    ssim_array  = torch.zeros(args.niters, device='cuda')
    lpips_array = torch.zeros(args.niters, device='cuda')

    best_mse = float('inf'); best_img = None
    best_psnr = 0.0; best_ssim = 0.0; best_lpips = 1.0; best_epoch = 0

    tag = 'wege' if args.use_wege else 'no_wege'
    result_root = (f'results/sr/{tag}/{args.nonlin}/scale{args.scale}/'
                   f'sigma{args.sigma0}_T{args.init_T}_B{args.init_beta}_Z{args.init_zeta}')
    img_save_dir = os.path.join(result_root, 'images')
    os.makedirs(img_save_dir, exist_ok=True)

    plt.imsave(os.path.join(img_save_dir, 'HR_gt.png'),         np.clip(im, 0, 1))
    plt.imsave(os.path.join(img_save_dir, f'LR_x{args.scale}.png'),     np.clip(im_lr, 0, 1))
    plt.imsave(os.path.join(img_save_dir, f'Bicubic_x{args.scale}.png'), np.clip(im_bi, 0, 1))

    downsampler = torch.nn.AvgPool2d(args.scale)
    tbar = tqdm(range(args.niters))
    for epoch in tbar:
        rec_hr = model(coords_hr)
        rec_lr = downsampler(rec_hr.reshape(H, W, 3).permute(2, 0, 1)[None, ...])
        loss = ((gt_lr - rec_lr.reshape(1, 3, -1).permute(0, 2, 1)) ** 2).mean()
        optim.zero_grad(); loss.backward(); optim.step(); scheduler.step()

        with torch.no_grad():
            im_rec = rec_hr.reshape(H, W, 3).permute(2, 0, 1)[None, ...]
            mse_array[epoch]   = ((gt - rec_hr) ** 2).mean().item()
            ssim_array[epoch]  = ssim(im_gt, im_rec, data_range=1, size_average=True)
            lpips_array[epoch] = lpips_model(im_rec * 2 - 1, im_gt * 2 - 1).item()

        psnr = -10 * torch.log10(mse_array[epoch]).item()
        ssim_val = ssim_array[epoch].item()
        lpips_val = lpips_array[epoch].item()
        tbar.set_description(f'PSNR {psnr:.2f}')

        imrec = im_rec.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        if mse_array[epoch] < best_mse:
            best_mse = mse_array[epoch]; best_img = imrec
            best_psnr = psnr; best_ssim = ssim_val; best_lpips = lpips_val
            best_epoch = epoch + 1

        if ((epoch + 1) % args.save_every == 0) or (epoch == args.niters - 1):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_mse': best_mse, 'best_img': best_img,
                'mse_array': mse_array, 'ssim_array': ssim_array, 'lpips_array': lpips_array,
            }, os.path.join(result_root, f'ckpt_{args.nonlin}_{epoch+1}.pt'))
            plt.imsave(os.path.join(result_root, f'best_{args.nonlin}_{epoch+1}.png'),
                       np.clip(best_img, 0, 1))
            with open(os.path.join(result_root, 'score_history.txt'), 'a') as f:
                f.write(f"=== EPOCH {epoch+1} ===\n")
                f.write(f"[Overall BEST] @ {best_epoch} | PSNR={best_psnr:.4f} SSIM={best_ssim:.4f} LPIPS={best_lpips:.4f}\n")


if __name__ == '__main__':
    main()
