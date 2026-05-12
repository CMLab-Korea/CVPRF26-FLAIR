#!/usr/bin/env python3
# Refactored train/test entry for NeRF variants (ngp / pemlp / siren / finer / wire / gauss / bla / sl2a)
# - Checkpoint resume is disabled by default (always train from scratch)
# - Keeps original behavior/options as much as possible
# - Workspace is auto-routed by dataset/nn with _1, _2 suffix to avoid collisions
#   so existing log/ckpt is not auto-loaded from the same folder

import argparse
import os
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.optim as optim

from nerf.provider import NeRFDataset
from nerf.utils import *  # Trainer, seed_everything, PSNRMeter, LPIPSMeter, SSIMMeter, etc.
from loss import huber_loss


def str2bool(x: str) -> bool:
    return str(x).lower() in ("true", "1", "yes", "y", "t")


# -------------------------
# NEW: dataset/run parsing
# -------------------------
def parse_dataset_name(path_str: str) -> str:
    """
    Extract a stable dataset name from opt.path.
    - If path is a file, use the parent folder name.
    - Otherwise, use the last folder name.
    - Sanitize characters for filesystem safety.
    """
    p = Path(path_str)
    if p.is_file():
        p = p.parent

    name = p.name.strip()
    if not name:
        name = "dataset"

    safe = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            safe.append(ch)
        else:
            safe.append("_")
    name = "".join(safe)
    return name if name else "dataset"


def make_unique_run_dir(workspace_root: str, base_name: str) -> str:
    """
    If workspace_root/base_name already exists, create base_name_1, base_name_2, ...
    Returns the created directory path.
    """
    root = Path(workspace_root)
    root.mkdir(parents=True, exist_ok=True)

    cand = root / base_name
    if not cand.exists():
        cand.mkdir(parents=True, exist_ok=False)
        return str(cand)

    k = 1
    while True:
        cand = root / f"{base_name}_{k}"
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=False)
            return str(cand)
        k += 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("-O", action="store_true", help="equals --fp16 --cuda_ray --preload")
    parser.add_argument("--test", action="store_true", help="test mode")
    parser.add_argument("--workspace", type=str, default="workspace")
    parser.add_argument("--seed", type=int, default=0)

    # network choice
    parser.add_argument(
        "--nn",
        type=str,
        default="ngp",
        choices=["ngp", "pemlp", "siren", "finer", "wire", "gauss", "bla", "sl2a"],
        help="neural network",
    )

    # bla options
    parser.add_argument("--init_T", type=float, default=1.0)
    parser.add_argument("--init_beta", type=float, default=0.05)
    parser.add_argument("--init_zeta", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=30.0)
    parser.add_argument("--trainable", type=str2bool, default=True)

    # siren / wire style knobs
    parser.add_argument("--fw0", type=float, default=30, help="first_omega_0")
    parser.add_argument("--hw0", type=float, default=1, help="hidden_omega_0")
    parser.add_argument("--fbs", type=float, default=None, help="first_bias_scale")

    parser.add_argument("--init_method", type=str, default="Pytorch")
    parser.add_argument("--init_gain", type=float, default=1)

    # dataset sampling
    parser.add_argument("--downscale", type=int, default=1, help="downsample factor for dataloader")
    parser.add_argument("--trainskip", type=int, default=1, help="train skip for dataloader")

    # layer structure
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--geo_feat_dim", type=int, default=256)
    parser.add_argument("--num_layers_color", type=int, default=4)
    parser.add_argument("--hidden_dim_color", type=int, default=256)

    # SL2A options
    parser.add_argument("--deg", type=int, default=256, help="degree of Chebyshev in SL2A")
    parser.add_argument("--rank", type=int, default=128, help="low-rank factor in SL2A")
    parser.add_argument("--nonlinearity", type=str, default="relu", help="activation in LowRank block (relu/none) for SL2A")
    parser.add_argument(
        "--linear_init_type",
        type=str,
        default="kaiming_uniform",
        choices=["kaiming_uniform", "kaiming_normal", "orthogonal", "uniform", "normal", "xavier_uniform"],
        help="init for low-rank factors in SL2A",
    )
    parser.add_argument(
        "--cheby_init_method",
        type=str,
        default="xavier_uniform",
        choices=["xavier_uniform", "kaiming_normal", "kaiming_uniform", "orthogonal", "uniform", "normal"],
        help="init for ChebyKAN in SL2A",
    )

    # training options
    parser.add_argument("--iters", type=int, default=30000, help="training iters")
    parser.add_argument("--lr", type=float, default=1e-2, help="initial learning rate")
    parser.add_argument("--ckpt", type=str, default=None)  # keep arg for compatibility, but resume is disabled below
    parser.add_argument("--num_rays", type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument("--cuda_ray", action="store_true", help="use CUDA raymarching instead of pytorch")
    parser.add_argument("--max_steps", type=int, default=1024, help="max num steps sampled per ray (valid when --cuda_ray)")
    parser.add_argument("--num_steps", type=int, default=512, help="num steps sampled per ray (valid when NOT --cuda_ray)")
    parser.add_argument("--upsample_steps", type=int, default=0, help="upsample steps per ray (valid when NOT --cuda_ray)")
    parser.add_argument("--update_extra_interval", type=int, default=16, help="iter interval to update extra status (valid when --cuda_ray)")
    parser.add_argument("--max_ray_batch", type=int, default=4096, help="batch size of rays at inference to avoid OOM (valid when NOT --cuda_ray)")
    parser.add_argument("--patch_size", type=int, default=1, help="render patches in training for LPIPS loss. 1 disables")

    # backend options
    parser.add_argument("--fp16", action="store_true", help="use amp mixed precision training")
    parser.add_argument("--ff", action="store_true", help="use fully-fused MLP")
    parser.add_argument("--tcnn", action="store_true", help="use TCNN backend")
    parser.add_argument("--fast", action="store_true",
                        help="enable TF32 + torch.compile (compile falls back if model is cfloat)")

    # dataset options
    parser.add_argument("--color_space", type=str, default="srgb", help="Color space: (linear, srgb)")
    parser.add_argument("--preload", action="store_true", help="preload all data into GPU (fast, more VRAM)")
    parser.add_argument("--bound", type=float, default=2, help="scene bound in [-bound, bound]^3")
    parser.add_argument("--scale", type=float, default=0.33, help="scale camera location into [-bound, bound]^3")
    parser.add_argument("--offset", type=float, nargs="*", default=[0, 0, 0], help="offset of camera location")
    parser.add_argument("--dt_gamma", type=float, default=0.1, help="dt_gamma for adaptive ray marching (>=0)")
    parser.add_argument("--min_near", type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument("--density_thresh", type=float, default=10, help="threshold for density grid occupancy")
    parser.add_argument("--bg_radius", type=float, default=-1, help="if positive, background model at sphere(bg_radius)")

    # GUI options
    parser.add_argument("--gui", action="store_true", help="start a GUI")
    parser.add_argument("--W", type=int, default=1920, help="GUI width")
    parser.add_argument("--H", type=int, default=1080, help="GUI height")
    parser.add_argument("--radius", type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument("--fovy", type=float, default=50, help="default GUI camera fovy")
    parser.add_argument("--max_spp", type=int, default=64, help="GUI rendering max sample per pixel")

    # experimental
    parser.add_argument("--error_map", action="store_true", help="use error map to sample rays")
    parser.add_argument("--clip_text", type=str, default="", help="text input for CLIP guidance")
    parser.add_argument("--rand_pose", type=int, default=-1, help="<0 no rand pose, =0 only rand pose, >0 sample one rand pose every $ known poses")

    return parser


def apply_shortcuts(opt: argparse.Namespace) -> argparse.Namespace:
    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True

    if opt.patch_size > 1:
        opt.error_map = False
        assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 must divide num_rays."

    # Resume disable: ignore opt.ckpt completely
    opt.ckpt = None

    # extra safety: if upstream Trainer checks these flags, force-disable when present
    if hasattr(opt, "use_checkpoint"):
        opt.use_checkpoint = None
    if hasattr(opt, "resume"):
        opt.resume = False
    if hasattr(opt, "load"):
        opt.load = False

    return opt


def get_network_class(opt: argparse.Namespace):
    # ngp family supports ff / tcnn branches
    if opt.nn == "ngp":
        if opt.ff:
            opt.fp16 = True
            assert opt.bg_radius <= 0, "background model is not implemented for --ff"
            from nerf.network_ff import NeRFNetwork
            return NeRFNetwork
        if opt.tcnn:
            opt.fp16 = True
            assert opt.bg_radius <= 0, "background model is not implemented for --tcnn"
            from nerf.network_tcnn import NeRFNetwork
            return NeRFNetwork
        from nerf.network import NeRFNetwork
        return NeRFNetwork

    if opt.nn == "finer":
        from nerf.network_finer import NeRFNetwork
        return NeRFNetwork

    if opt.nn == "bla":
        from nerf.network_bla import NeRFNetwork
        return NeRFNetwork

    if opt.nn == "sl2a":
        from nerf.network_sl2a import SL2ANeRFNetwork
        return SL2ANeRFNetwork

    if opt.nn == "pemlp":
        from nerf.network_pemlp import NeRFNetwork
        return NeRFNetwork

    if opt.nn == "siren":
        from nerf.network_siren import NeRFNetwork
        return NeRFNetwork

    if opt.nn == "gauss":
        from nerf.network_gauss import NeRFNetwork
        return NeRFNetwork

    if opt.nn == "wire":
        from nerf.network_wire import NeRFNetwork
        return NeRFNetwork

    raise ValueError(f"Unknown --nn {opt.nn}")


def build_model(opt: argparse.Namespace):
    Net = get_network_class(opt)

    common_kwargs = dict(
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
        num_layers=opt.num_layers,
        hidden_dim=opt.hidden_dim,
        geo_feat_dim=opt.geo_feat_dim,
        num_layers_color=opt.num_layers_color,
        hidden_dim_color=opt.hidden_dim_color,
    )

    # encoding policy
    # - pure INR variants often set encoding="None"
    # - default branch uses encoding="frequency"
    if opt.nn in ("finer", "siren", "wire", "bla", "sl2a"):
        encoding = "None"
    else:
        encoding = "frequency"

    if opt.nn == "wire":
        return Net(
            encoding=encoding,
            **common_kwargs,
            omega=opt.hw0,
            sigma=opt.sigma,
        )

    if opt.nn == "bla":
        return Net(
            encoding=encoding,
            **common_kwargs,
            init_T=opt.init_T,
            init_beta=opt.init_beta,
            init_zeta=opt.init_zeta,
            sigma=opt.sigma,
            trainable=opt.trainable,
        )

    if opt.nn == "sl2a":
        # SL2A network signature differs
        return Net(
            encoding="None",
            encoding_dir="None",
            encoding_bg="hashgrid",
            bound=opt.bound,
            cuda_ray=opt.cuda_ray,
            density_scale=1,
            min_near=opt.min_near,
            density_thresh=opt.density_thresh,
            bg_radius=opt.bg_radius,
            num_layers=opt.num_layers,
            hidden_dim=opt.hidden_dim,
            geo_feat_dim=opt.geo_feat_dim,
            num_layers_color=opt.num_layers_color,
            hidden_dim_color=opt.hidden_dim_color,
            deg=opt.deg,
            rank=opt.rank,
            nonlinearity=opt.nonlinearity,
            init_method=opt.cheby_init_method,
            linear_init_type=opt.linear_init_type,
        )

    # finer / siren keep fbs
    if opt.nn in ("finer", "siren"):
        return Net(
            encoding=encoding,
            **common_kwargs,
            fbs=opt.fbs,
        )

    # ngp / pemlp / gauss default
    return Net(
        encoding=encoding,
        **common_kwargs,
    )


def build_trainer(opt: argparse.Namespace, model: torch.nn.Module, device: torch.device, criterion):
    metrics = [PSNRMeter(), LPIPSMeter(device=device), SSIMMeter(device=device)]

    # run name in Trainer can stay "ngp" or reflect opt.nn
    dataset_name = parse_dataset_name(opt.path)
    run_name = f"{dataset_name}_{opt.nn}"

    if opt.test:
        trainer = Trainer(
            run_name,
            opt,
            model,
            device=device,
            workspace=opt.workspace,
            criterion=criterion,
            fp16=opt.fp16,
            metrics=metrics,
            use_checkpoint=None,  # resume disabled
        )
        return trainer

    optimizer_fn = lambda m: torch.optim.Adam(m.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)
    scheduler_fn = lambda optz: optim.lr_scheduler.LambdaLR(optz, lambda it: 0.1 ** min(it / opt.iters, 1))

    trainer = Trainer(
        run_name,
        opt,
        model,
        device=device,
        workspace=opt.workspace,
        optimizer=optimizer_fn,
        criterion=criterion,
        ema_decay=0.95,
        fp16=opt.fp16,
        lr_scheduler=scheduler_fn,
        scheduler_update_every_step=True,
        metrics=metrics,
        use_checkpoint=None,  # resume disabled
        eval_interval=100,
    )
    return trainer


def main():
    parser = build_parser()
    opt = parser.parse_args()
    opt = apply_shortcuts(opt)

    print(opt)
    seed_everything(opt.seed)

    if getattr(opt, "fast", False):
        import sys as _sys, os as _os
        _FLAIR_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
        if _FLAIR_ROOT not in _sys.path:
            _sys.path.insert(0, _FLAIR_ROOT)
        from modules.speed import setup_fast_env
        setup_fast_env()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # NEW: unique workspace per run
    # -------------------------
    dataset_name = parse_dataset_name(opt.path)
    base_run_folder = f"{dataset_name}_{opt.nn}"
    opt.workspace = make_unique_run_dir(opt.workspace, base_run_folder)

    print(f"[run] dataset_name={dataset_name}")
    print(f"[run] run_folder={base_run_folder}")
    print(f"[run] workspace={opt.workspace}")
    # -------------------------

    model = build_model(opt)
    if getattr(opt, "fast", False):
        from modules.speed import maybe_compile
        model = maybe_compile(model, enabled=True)
    print(model)

    # criterion
    criterion = torch.nn.MSELoss(reduction="none")
    # criterion = partial(huber_loss, reduction="none")

    trainer = build_trainer(opt, model, device, criterion)

    if opt.gui:
        print("Warning: GUI mode is disabled in this script.")
        return

    if opt.test:
        test_loader = NeRFDataset(opt, device=device, type="test", downscale=opt.downscale).dataloader()
        if getattr(test_loader, "has_gt", False):
            trainer.evaluate(test_loader)
        return

    # train
    train_loader = NeRFDataset(
        opt,
        device=device,
        type="train",
        downscale=opt.downscale,
        train_skip=opt.trainskip,
    ).dataloader()
    print(f"train_loader: {train_loader}")

    valid_loader = NeRFDataset(opt, device=device, type="val", downscale=opt.downscale).dataloader()

    max_epoch = int(np.ceil(opt.iters / len(train_loader)))
    trainer.train(train_loader, valid_loader, max_epoch)


if __name__ == "__main__":
    main()
