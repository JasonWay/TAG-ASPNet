import math

import torch
import torch.nn.functional as F


def gaussian(window_size, sigma, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype)
    coords = coords - window_size // 2
    gauss = torch.exp(-(coords**2) / (2 * sigma**2))
    return gauss / gauss.sum()


def create_window(window_size, channel, device, dtype):
    window_1d = gaussian(window_size, 1.5, device, dtype).unsqueeze(1)
    window_2d = window_1d @ window_1d.t()
    window_2d = window_2d.unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channel, 1, window_size, window_size).contiguous()


def ssim(img1, img2, window_size=11, size_average=True):
    img1 = img1.clamp(0, 1)
    img2 = img2.clamp(0, 1)
    _, channel, _, _ = img1.size()
    window = create_window(window_size, channel, img1.device, img1.dtype)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(1).mean(1).mean(1)


def psnr(pred, gt):
    pred = pred.clamp(0, 1)
    gt = gt.clamp(0, 1)
    mse = F.mse_loss(pred, gt, reduction="mean").item()
    if mse == 0:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def ms_ssim(img1, img2, levels=3):
    """Multi-scale SSIM with equal-weight averaging across scales."""
    img1 = img1.clamp(0, 1)
    img2 = img2.clamp(0, 1)
    total = None
    for i in range(levels):
        s = ssim(img1, img2)
        total = s if total is None else total + s
        if i < levels - 1:
            img1 = F.avg_pool2d(img1, 2)
            img2 = F.avg_pool2d(img2, 2)
    return total / float(levels)
