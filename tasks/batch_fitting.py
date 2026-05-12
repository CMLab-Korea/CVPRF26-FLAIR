#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import random
import argparse
import importlib
from dataclasses import dataclass
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2

from tqdm import tqdm
from pytorch_msssim import ssim as ssim_torch

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

import lpips

from modules import models
from modules import utils
from modules.speed import setup_fast_env, maybe_compile, resolve_fast_nonlin
from wege import WEGEExtractor

setup_fast_env()


# =========================
# Utils
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def soft_clip(wb, percentile=99.5, alpha=0.7):
    """
    percentile: upper percentile threshold
    alpha: soft clipping amount (0: hard, 1: linear beyond threshold)
    """
    pval = np.percentile(wb, percentile)
    return np.where(wb > pval, pval + alpha * (wb - pval), wb)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def append_line(path: str, line: str, flush: bool = True):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")
        if flush:
            f.flush()
            os.fsync(f.fileno())


def save_curves(result_root: str, nonlin: str, psnr_arr: np.ndarray, ssim_arr: np.ndarray, lpips_arr: np.ndarray):
    ensure_dir(result_root)

    plt.plot(psnr_arr)
    plt.title("PSNR Curve")
    plt.xlabel("Eval Step")
    plt.ylabel("PSNR (dB)")
    plt.savefig(os.path.join(result_root, f"curve_psnr_{nonlin}.png"))
    plt.clf()

    plt.plot(ssim_arr)
    plt.title("SSIM Curve")
    plt.xlabel("Eval Step")
    plt.ylabel("SSIM")
    plt.savefig(os.path.join(result_root, f"curve_ssim_{nonlin}.png"))
    plt.clf()

    plt.plot(lpips_arr)
    plt.title("LPIPS Curve")
    plt.xlabel("Eval Step")
    plt.ylabel("LPIPS")
    plt.savefig(os.path.join(result_root, f"curve_lpips_{nonlin}.png"))
    plt.clf()


def visualize_wavelet_score_clean(
    wb_pixel_blur_np: np.ndarray,
    title='Wavelet Energy (blurred)',
    save_path='wavelet_energy_clean.png',
    topN=50,
    show_markers: bool = False
):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(wb_pixel_blur_np, cmap='jet')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.047, pad=0.01)

    if show_markers:
        flat = wb_pixel_blur_np.flatten()
        sorted_idx = np.argsort(flat)[::-1]
        top_idx = sorted_idx[:topN]
        rows, cols = np.unravel_index(top_idx, wb_pixel_blur_np.shape)
        ax.scatter(cols, rows, s=60, edgecolors='white', facecolors='none', linewidths=1.5, marker='o')

    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path) or ".")
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


def visualize_wavelet_score_perfect_clean(
    wb_pixel_blur_np: np.ndarray,
    title='z',
    save_path='wavelet_energy_perfect_clean.png',
    topN=50,
    show_markers: bool = False
):
    H, W = wb_pixel_blur_np.shape[:2]
    ensure_dir(os.path.dirname(save_path) or ".")

    if not show_markers:
        plt.imsave(save_path, wb_pixel_blur_np, cmap='jet')
        return

    dpi = 100
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, frameon=False)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)

    ax.imshow(wb_pixel_blur_np, cmap='jet', interpolation='nearest', origin='upper')
    ax.set_aspect('equal')
    ax.axis('off')

    flat = wb_pixel_blur_np.flatten()
    sorted_idx = np.argsort(flat)[::-1]
    top_idx = sorted_idx[:topN]
    rows, cols = np.unravel_index(top_idx, wb_pixel_blur_np.shape)
    ax.scatter(cols, rows, s=60, edgecolors='white', facecolors='none', linewidths=1.5, marker='o')

    fig.savefig(save_path, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)


@torch.no_grad()
def reconstruction_and_metrics_in_chunks(
    model,
    coords: torch.Tensor,     # [1, N, C]
    gt_flat: torch.Tensor,    # [1, N, 3]
    H: int,
    W: int,
    chunk_size: int,
    im_gt: torch.Tensor,      # [1, 3, H, W]
    lpips_model,
):
    """
    Full-image reconstruction with chunked forward, then compute:
    - MSE over all pixels
    - SSIM over full image
    - LPIPS over full image
    """
    device = coords.device
    B, N, _ = coords.shape
    assert B == 1

    pred_flat = torch.empty((1, N, 3), device=device, dtype=gt_flat.dtype)

    for s in range(0, N, chunk_size):
        e = min(s + chunk_size, N)
        pred_flat[:, s:e, :] = model(coords[:, s:e, :])

    im_rec = pred_flat.reshape(H, W, 3).permute(2, 0, 1)[None, ...]  # [1,3,H,W]

    mse_val = ((gt_flat - pred_flat) ** 2).mean()
    ssim_val = ssim_torch(im_gt, im_rec, data_range=1, size_average=True)

    im_rec_norm = (im_rec * 2) - 1
    im_gt_norm = (im_gt * 2) - 1
    lpips_val = lpips_model(im_rec_norm, im_gt_norm).mean()

    return im_rec, mse_val, ssim_val, lpips_val


# =========================
# Main
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = importlib.reload(models)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--nonlin', type=str, default='bla')
    parser.add_argument('--sigma0', type=float, default=2.0)
    parser.add_argument('--omega0', type=float, default=8.0)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--scale_im', type=float, default=1)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--hidden_layers', type=int, default=4)
    parser.add_argument('--hidden_features', type=int, default=256)
    parser.add_argument('--niters', type=int, default=5000)
    parser.add_argument('--image_path', type=str, required=True)

    parser.add_argument('--T', '--init_T', dest='init_T', type=float, default=1.0)
    parser.add_argument('--B', '--init_beta', dest='init_beta', type=float, default=0.05)
    parser.add_argument('--Z', '--init_zeta', dest='init_zeta', type=float, default=1.0)

    parser.add_argument('--fomega', type=float, default=30.0)
    parser.add_argument('--seed', type=int, default=42)
    
    # ========= VRAM-safe random sampling training =========
    parser.add_argument('--num_samples', type=int, default=262144,
                        help='per-iter random sampled coords (K)')
    parser.add_argument('--metric_every', type=int, default=2000,
                        help='full-image eval period (iterations)')
    parser.add_argument('--eval_chunk', type=int, default=65536,
                        help='chunk size for full-image eval forward')

    # ========= WEGE =========
    parser.add_argument('--use_wege', action='store_true',
                        help='if set, concat WEGE wb score as 3rd input channel (default: off)')
    parser.add_argument('--J', type=int, default=1)
    parser.add_argument('--wave', type=str, default='db3')
    parser.add_argument('--gf_radius', type=int, default=6)
    parser.add_argument('--gf_eps', type=float, default=1e-5)

    # logging / saving
    parser.add_argument('--save_every', type=int, default=500, help='checkpoint & curves period (iterations)')
    parser.add_argument('--log_txt', type=str, default='metrics.txt', help='metrics text filename under result_root')

    # wavelet dump
    parser.add_argument('--dump_wb', action='store_true',
                        help='dump wavelet maps using existing visualization functions + save npy')
    parser.add_argument('--wb_out_dir', type=str, default='wb_dump',
                        help='output subdir name under result_root when --dump_wb')
    parser.add_argument('--wb_markers', action='store_true',
                        help='draw top-k markers on wavelet map dump (uses existing funcs)')
    parser.add_argument('--wb_topN', type=int, default=50,
                        help='topN for marker visualization when --wb_markers')
    parser.add_argument('--fast', action='store_true',
                        help='torch.compile model (recommend --nonlin bla_float)')

    args = parser.parse_args()
    args.nonlin = resolve_fast_nonlin(args.nonlin, args.fast)

    set_seed(args.seed)

    nonlin = args.nonlin
    sigma0 = args.sigma0
    omega0 = args.omega0
    scale_im = args.scale_im
    learning_rate = args.learning_rate
    hidden_layers = args.hidden_layers
    hidden_features = args.hidden_features
    image_path = args.image_path
    niters = int(args.niters)

    init_T = args.init_T
    init_beta = args.init_beta
    init_zeta = args.init_zeta
    first_omega = args.fomega

    num_samples = int(args.num_samples)
    metric_every = int(args.metric_every)
    eval_chunk = int(args.eval_chunk)
    save_every = int(args.save_every)

    # =========================
    # Load image
    # =========================
    im = utils.normalize(plt.imread(image_path).astype(np.float32), True)
    im = cv2.resize(im, None, fx=scale_im, fy=scale_im, interpolation=cv2.INTER_AREA)
    H, W, _ = im.shape

    if nonlin == 'posenc':
        nonlin = 'relu'
        posencode = True
        sidelength = int(max(H, W))
    else:
        posencode = False
        sidelength = H

    # =========================
    # Build model
    # =========================
    in_features = 3 if args.use_wege else 2  # (x, y[, wb])
    model = models.get_INR(
        nonlin=nonlin,
        in_features=in_features,
        out_features=3,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        first_omega_0=first_omega,
        hidden_omega_0=omega0,
        scale=sigma0,
        pos_encode=posencode,
        sidelength=sidelength,
        init_T=init_T,
        init_beta=init_beta,
        init_zeta=init_zeta
    ).to(device)
    model = maybe_compile(model, args.fast)

    optim = torch.optim.Adam(lr=learning_rate, params=model.parameters())
    scheduler = LambdaLR(optim, lambda x: 0.2 ** min(x / niters, 1))

    # =========================
    # Build full coords grid
    # =========================
    x_hr = torch.linspace(-1, 1, W, device=device)
    y_hr = torch.linspace(-1, 1, H, device=device)
    X_hr, Y_hr = torch.meshgrid(x_hr, y_hr, indexing='xy')
    coords_hr = torch.hstack((X_hr.reshape(-1, 1), Y_hr.reshape(-1, 1)))[None, ...]  # [1, N, 2]

    # =========================
    # Result dirs + txt log (create early: used by wb dump)
    # =========================
    J = int(args.J)
    wave = args.wave
    radius = int(args.gf_radius)
    eps = float(args.gf_eps)

    wege_tag = 'wege' if args.use_wege else 'no_wege'
    result_root = (
        f"results/fitting/{wege_tag}/{nonlin}/{image_path}/hidden_features={hidden_features}/"
        f"J={J}_radius={radius}_eps={eps}/sigma:{sigma0}_T:{init_T}_B:{init_beta}_Z:{init_zeta}"
    )
    ensure_dir(result_root)
    checkpoint_dir = result_root
    ensure_dir(checkpoint_dir)

    txt_path = os.path.join(result_root, args.log_txt)
    if not os.path.exists(txt_path):
        append_line(txt_path, "iter,phase,psnr,ssim,lpips,mse,lr", flush=True)

    # =========================
    # WEGE (only if --use_wege; otherwise model takes (x, y) directly)
    # =========================
    if args.use_wege:
        wavelet_energy_extractor = WEGEExtractor(
            img_size=(H, W),
            J=J,
            wave=wave
        ).to(device)

        im_torch = torch.tensor(im, device=device).permute(2, 0, 1)[None, ...]

        with torch.no_grad():
            wb_pixel = wavelet_energy_extractor.get_pixelwise_energy_scores(im_torch)  # [H, W]
            wb_pixel_np = wb_pixel.detach().cpu().numpy().astype(np.float32)

            guide_img = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) if im.shape[2] == 3 else im

            wb_pixel_guided_np = cv2.ximgproc.guidedFilter(
                guide=guide_img.astype(np.float32),
                src=wb_pixel_np,
                radius=radius,
                eps=eps
            )

            wb_pixel_guided_np = soft_clip(wb_pixel_guided_np, percentile=99.5, alpha=0)
            wb_pixel_guided_np = np.clip(wb_pixel_guided_np, 0, 1)

            # ---- dump wb maps under result_root using your existing visualization funcs ----
            if args.dump_wb:
                wb_out_dir = os.path.join(result_root, args.wb_out_dir)
                ensure_dir(wb_out_dir)

                visualize_wavelet_score_perfect_clean(
                    wb_pixel_np,
                    title=f'Wavelet Energy RAW (J={J}, Wave={wave})',
                    save_path=os.path.join(wb_out_dir, f"wb_raw_perfect_clean.png"),
                    topN=int(args.wb_topN),
                    show_markers=bool(args.wb_markers),
                )
                visualize_wavelet_score_clean(
                    wb_pixel_np,
                    title=f'Wavelet Energy RAW (J={J}, Wave={wave})',
                    save_path=os.path.join(wb_out_dir, f"wb_raw_clean.png"),
                    topN=int(args.wb_topN),
                    show_markers=bool(args.wb_markers),
                )

                visualize_wavelet_score_perfect_clean(
                    wb_pixel_guided_np,
                    title=f'Wavelet Energy GUIDED (r={radius}, eps={eps}, J={J}, Wave={wave})',
                    save_path=os.path.join(wb_out_dir, f"wb_guided_perfect_clean.png"),
                    topN=int(args.wb_topN),
                    show_markers=bool(args.wb_markers),
                )
                visualize_wavelet_score_clean(
                    wb_pixel_guided_np,
                    title=f'Wavelet Energy GUIDED (r={radius}, eps={eps}, J={J}, Wave={wave})',
                    save_path=os.path.join(wb_out_dir, f"wb_guided_clean.png"),
                    topN=int(args.wb_topN),
                    show_markers=bool(args.wb_markers),
                )

                stats_txt = os.path.join(wb_out_dir, "wb_stats.txt")
                with open(stats_txt, "w", encoding="utf-8") as f:
                    f.write(f"raw:    min={wb_pixel_np.min():.6f}, max={wb_pixel_np.max():.6f}, mean={wb_pixel_np.mean():.6f}\n")
                    f.write(f"guided: min={wb_pixel_guided_np.min():.6f}, max={wb_pixel_guided_np.max():.6f}, mean={wb_pixel_guided_np.mean():.6f}\n")

            wb_pixel_guided = torch.tensor(wb_pixel_guided_np, device=device, dtype=coords_hr.dtype)
            wb_map = WEGEExtractor.expand_energy_scores_to_coords_pixelwise(H, W, wb_pixel_guided)  # [N,1]

        coords_full = torch.cat([coords_hr, wb_map[None, ...]], dim=2).contiguous()  # [1, N, 3]
    else:
        coords_full = coords_hr.contiguous()  # [1, N, 2]

    # =========================
    # GT (full)
    # =========================
    gt = torch.tensor(im, device=device).reshape(H * W, 3)[None, ...].contiguous()  # [1,N,3]
    im_gt = gt.reshape(H, W, 3).permute(2, 0, 1)[None, ...].contiguous()  # [1,3,H,W]

    # LPIPS model (used only for eval)
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()

    # =========================
    # Checkpoint load (resume default: OFF)
    # =========================
    start_iter = 0
    best_mse = float('inf')
    best_img = None

    eval_iters = []
    eval_psnr = []
    eval_ssim = []
    eval_lpips = []

    resume = False
    if resume:
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('FIT4090_') and f.endswith('.pt')]

        def _extract_iter_from_name(name: str):
            try:
                return int(name.split('_')[-1].split('.')[0])
            except Exception:
                return -1

        checkpoint_files = sorted(checkpoint_files, key=_extract_iter_from_name)

        checkpoint_path = checkpoint_files[-1] if checkpoint_files else None
        if checkpoint_path:
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f'Loading checkpoint from {checkpoint_path} ...', flush=True)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            best_mse = checkpoint.get('best_mse', best_mse)
            best_img = checkpoint.get('best_img', best_img)
            start_iter = int(checkpoint.get('iter', checkpoint.get('epoch', -1))) + 1

            if 'eval_iters' in checkpoint:
                eval_iters = list(checkpoint['eval_iters'])
            if 'eval_psnr' in checkpoint:
                eval_psnr = list(checkpoint['eval_psnr'])
            if 'eval_ssim' in checkpoint:
                eval_ssim = list(checkpoint['eval_ssim'])
            if 'eval_lpips' in checkpoint:
                eval_lpips = list(checkpoint['eval_lpips'])

            print(f"=> Resuming from iter {start_iter}, best_mse={best_mse}", flush=True)

    # =========================
    # Train loop: uniform random sampled coords (VRAM-safe)
    # =========================
    N_total = gt.shape[1]
    tbar = tqdm(range(start_iter, niters))

    for it in tbar:
        sample_idx = torch.randint(0, N_total, (num_samples,), device=device)
        coords_sample = coords_full[:, sample_idx, :]
        gt_sample = gt[:, sample_idx, :]

        pred = model(coords_sample)
        loss = ((pred - gt_sample) ** 2).mean()

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        scheduler.step()

        # ---- train-time one-line metrics (proxy): keep everything except LPIPS ----
        with torch.no_grad():
            mse_batch = loss.detach()
            psnr_batch = -10.0 * torch.log10(mse_batch + 1e-12)

            ssim_proxy = (1.0 - mse_batch).clamp(0.0, 1.0)
            lr = optim.param_groups[0]['lr']

            tbar.set_description(
                f"Train | PSNR {psnr_batch.item():.2f} "
                f"SSIM~ {ssim_proxy.item():.4f}"
            )

            append_line(
                txt_path,
                f"{it+1},train,{psnr_batch.item():.6f},{ssim_proxy.item():.6f},nan,{mse_batch.item():.8e},{lr:.8e}",
                flush=False
            )

        # ---- full-image eval (accurate) ----
        do_eval = (it == start_iter) or ((it + 1) % metric_every == 0) or (it == niters - 1)
        if do_eval:
            im_rec, mse_val_t, ssim_val_t, lpips_val_t = reconstruction_and_metrics_in_chunks(
                model=model,
                coords=coords_full,
                gt_flat=gt,
                H=H,
                W=W,
                chunk_size=eval_chunk,
                im_gt=im_gt,
                lpips_model=lpips_model
            )

            psnr_val = -10.0 * torch.log10(mse_val_t + 1e-12)

            eval_iters.append(it + 1)
            eval_psnr.append(psnr_val.item())
            eval_ssim.append(ssim_val_t.item())
            eval_lpips.append(lpips_val_t.item())

            append_line(
                txt_path,
                f"{it+1},eval,{psnr_val.item():.6f},{ssim_val_t.item():.6f},{lpips_val_t.item():.6f},{mse_val_t.item():.8e},{optim.param_groups[0]['lr']:.8e}",
                flush=True
            )

            imrec_np = im_rec.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            if mse_val_t.item() < best_mse:
                best_mse = float(mse_val_t.item())
                best_img = imrec_np

        # ---- save checkpoint + curves ----
        do_save = ((it + 1) % save_every == 0) or (it == niters - 1)
        if do_save:
            ckpt_path = os.path.join(checkpoint_dir, f'FIT4090_j={J}_{nonlin}_{it+1}.pt')
            torch.save({
                'iter': it,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_mse': best_mse,
                'best_img': best_img,
                'eval_iters': eval_iters,
                'eval_psnr': eval_psnr,
                'eval_ssim': eval_ssim,
                'eval_lpips': eval_lpips,
            }, ckpt_path)

            if best_img is not None:
                best_img_path = os.path.join(checkpoint_dir, f"best_{nonlin}_{it+1}iter.png")
                plt.imsave(best_img_path, np.clip(best_img, 0, 1))

            if len(eval_iters) > 0:
                save_curves(
                    result_root=result_root,
                    nonlin=nonlin,
                    psnr_arr=np.array(eval_psnr, dtype=np.float32),
                    ssim_arr=np.array(eval_ssim, dtype=np.float32),
                    lpips_arr=np.array(eval_lpips, dtype=np.float32),
                )

    try:
        append_line(txt_path, f"# done niters={niters}", flush=True)
    except Exception:
        pass


if __name__ == '__main__':
    main()
