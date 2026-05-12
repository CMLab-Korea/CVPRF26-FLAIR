import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
from torchvision.transforms.functional import to_pil_image
import numpy as np
import cv2 
import matplotlib.pyplot as plt


class WEGEExtractor(nn.Module):
    def __init__(self, img_size=(32, 32), J=3, wave='db3', mode='symmetric'):
        super().__init__()
        self.H, self.W = img_size
        self.J = J
        self.dwt = DWTForward(J=J, wave=wave, mode=mode)
        self.idwt = DWTInverse(wave=wave, mode=mode)

    def pad_to_uniform(self, x):
        B, C, H, W = x.shape
        target_H = int(np.ceil(H / (2 ** self.J)) * (2 ** self.J))
        target_W = int(np.ceil(W / (2 ** self.J)) * (2 ** self.J))
        pad_H, pad_W = target_H - H, target_W - W
        x = F.pad(x, (0, pad_W, 0, pad_H))
        return x, H, W

    def compute_H_top_global(self, x):
        x, orig_H, orig_W = self.pad_to_uniform(x)
        B, C, H, W = x.shape
        Yl, Yh = self.dwt(x)
        Yh_partial_prev = [None if i < self.J - 1 else Yh[i] for i in range(self.J)]
        D_prev = self.idwt((Yl, Yh_partial_prev))
        Yh_empty = [None for _ in range(self.J)]
        D_curr = self.idwt((Yl, Yh_empty))
        D_prev = D_prev[..., :H, :W]
        D_curr = D_curr[..., :H, :W]
        return D_prev - D_curr

    def get_pixelwise_energy_scores(self, x, eps=1e-6):
        x, H, W = self.pad_to_uniform(x)
        with torch.no_grad():
            H_top_global = self.compute_H_top_global(x)
            energy_map = (H_top_global ** 2).mean(dim=1, keepdim=False)[0][:H, :W]
            Emin, Emax = energy_map.min(), energy_map.max()
            wb_pixel = (energy_map - Emin) / (Emax - Emin + eps)
        return wb_pixel

    def get_pixelwise_energy_scores_sqrt(self, x, eps=1e-6):
        x, H, W = self.pad_to_uniform(x)
        with torch.no_grad():
            H_top_global = self.compute_H_top_global(x)
            energy_map = (H_top_global ** 2).mean(dim=1, keepdim=False).sqrt()[0][:H, :W]
            Emin, Emax = energy_map.min(), energy_map.max()
            wb_pixel = (energy_map - Emin) / (Emax - Emin + eps)
        return wb_pixel

    @staticmethod
    def expand_energy_scores_to_coords_pixelwise(H, W, wb_pixel):
        return wb_pixel.reshape(-1, 1)

    # Save LL, HF, and full reconstruction images.
    def save_wavelet_reconstructions(self, x, save_dir="wavelet_recons",
                                     gamma=0.6, contrast_alpha=2.0):
        os.makedirs(save_dir, exist_ok=True)
        x, orig_H, orig_W = self.pad_to_uniform(x)
        B, C, H, W = x.shape

        with torch.no_grad():
            # 1) DWT decomposition
            Yl, Yh = self.dwt(x)

            # 2) D_prev (LL + all HF)
            D_prev = self.idwt((Yl, Yh))[..., :H, :W]

            # 3) D_curr (LL only)
            Yh_empty = [None for _ in range(self.J)]
            D_curr = self.idwt((Yl, Yh_empty))[..., :H, :W]

            # 4) HF only (H_top_global)
            H_top_global = D_prev - D_curr

        # Inner save helpers (HF gets extra enhancement).
        def save_tensor_img(tensor, path):
            img = tensor[0].clamp(0, 1)
            to_pil_image(img).save(path)

        def save_hf_enhanced_gray(tensor, path):
            img = tensor[0].detach().cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)  # normalize to [0,1]

            # Edge emphasis while preserving background.
            img = np.power(img, gamma)

            # Contrast boost (keep gray background).
            img = (img * 255).astype(np.uint8)
            img = cv2.convertScaleAbs(img, alpha=contrast_alpha, beta=0)

            to_pil_image(torch.tensor(img / 255.0).float()).save(path)
                
        def save_wavelet_band(band, name):
            band_np = band[0].detach().cpu().numpy()

            # Normalize (C, H, W) to RGB form.
            if band_np.ndim == 3:
                if band_np.shape[0] == 3:
                    band_np = np.transpose(band_np, (1, 2, 0))  # (3, H, W) -> (H, W, 3)
                else:
                    # C != 3 -> average to single channel
                    band_np = band_np.mean(axis=0)[..., None]  # (H, W, 1)
            else:
                band_np = band_np[..., None]  # (H, W) -> (H, W, 1)

            # Replicate single-channel to 3 channels.
            if band_np.shape[-1] == 1:
                band_np = np.repeat(band_np, 3, axis=-1)  # (H, W, 1) -> (H, W, 3)

            # Normalize to [0, 1].
            band_np = (band_np - band_np.min()) / (band_np.max() - band_np.min() + 1e-6)

            plt.imshow(band_np)  # always RGB
            plt.title(name)
            plt.axis('off')
            plt.savefig(os.path.join(save_dir, f"{name}.png"))
            plt.close()

        # Save outputs.
        save_tensor_img(D_prev, os.path.join(save_dir, "D_prev_LL_HF.png"))
        save_tensor_img(D_curr, os.path.join(save_dir, "D_curr_LL_only.png"))
        save_hf_enhanced_gray(H_top_global,
                              os.path.join(save_dir, "HF_only_LH_HL_HH_enhanced_gray.png"))

        print(f"✅ Saved to {save_dir}: D_prev, D_curr, and enhanced HF images")