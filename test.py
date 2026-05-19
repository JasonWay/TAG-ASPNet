import argparse
import csv
import logging
import os
from datetime import datetime

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from model import TAGASPNet
from utils.metrics import psnr, ssim


def is_image_file(name):
    return name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))


class EvalDataset(Dataset):
    def __init__(self, hazy_dir, clear_dir, resize_hazy_to_clear=True):
        hazy = {f: os.path.join(hazy_dir, f) for f in os.listdir(hazy_dir) if is_image_file(f)}
        clear = {f: os.path.join(clear_dir, f) for f in os.listdir(clear_dir) if is_image_file(f)}
        self.names = sorted(set(hazy) & set(clear))
        if not self.names:
            raise ValueError("No matched image pairs found in test set. Check test/hazy and test/clear.")
        self.hazy = hazy
        self.clear = clear
        self.resize_hazy_to_clear = resize_hazy_to_clear
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        hazy_p = Image.open(self.hazy[name]).convert("RGB")
        clear_p = Image.open(self.clear[name]).convert("RGB")
        if self.resize_hazy_to_clear and hazy_p.size != clear_p.size:
            hazy_p = hazy_p.resize(clear_p.size, Image.BICUBIC)
        return {
            "name": name,
            "hazy": self.to_tensor(hazy_p),
            "clear": self.to_tensor(clear_p),
        }


def setup_logger(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file, mode="a", encoding="utf-8"), logging.StreamHandler()],
    )
    return logging.getLogger("test")


def save_image(tensor, path):
    img = transforms.ToPILImage()(tensor.detach().cpu().clamp(0, 1)[0])
    img.save(path)


def resolve_clear_dir(root, split="test"):
    split_dir = os.path.join(root, split)
    candidates = ["clear", "gt", "GT"]
    for name in candidates:
        path = os.path.join(split_dir, name)
        if os.path.exists(path):
            return path
    return os.path.join(split_dir, "clear")


def load_model(model_path, device, base_channels):
    ckpt = torch.load(model_path, map_location=device)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    model_base_channels = base_channels if base_channels > 0 else int(ckpt_args.get("base_channels", 32))
    model = TAGASPNet(base_channels=model_base_channels).to(device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model, ckpt if isinstance(ckpt, dict) else {}


def evaluate(model, loader, device, save_results=False, output_dir=None):
    rows = []
    total_psnr = 0.0
    total_ssim = 0.0
    with torch.no_grad():
        for batch in loader:
            hazy = batch["hazy"].to(device)
            clear = batch["clear"].to(device)
            names = batch["name"]
            pred, _, _ = model(hazy)
            for i in range(pred.size(0)):
                p = pred[i : i + 1].clamp(0, 1)
                g = clear[i : i + 1].clamp(0, 1)
                pi = psnr(p, g)
                si = ssim(p, g).item()
                total_psnr += pi
                total_ssim += si
                rows.append({"filename": names[i], "psnr": pi, "ssim": si})
                if save_results:
                    save_image(p, os.path.join(output_dir, names[i]))
    n = len(rows)
    return rows, total_psnr / max(n, 1), total_ssim / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="Dehazing model evaluation (PSNR/SSIM)")
    parser.add_argument("--test_hazy_dir", type=str, default=None, help="Hazy image directory (overrides dataset preset)")
    parser.add_argument("--test_clear_dir", type=str, default=None, help="Clear/GT image directory (overrides dataset preset)")
    parser.add_argument("--dataset_preset", type=str, default="nupw", choices=["nupw", "satehaze1k"])
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--tag", type=str, default="nupw")
    parser.add_argument("--model_path", type=str, default=None, help="If empty, auto-find best/latest by tag")
    parser.add_argument("--model_type", type=str, default="best", choices=["best", "latest"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=-1, help="<=0 means auto-read from checkpoint args")
    parser.add_argument("--no_resize_hazy", action="store_true")
    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--output_dir", type=str, default="test_results")
    parser.add_argument("--save_csv", action="store_true")
    parser.add_argument("--log_dir", type=str, default="logs")
    args = parser.parse_args()

    if args.dataset_preset == "satehaze1k":
        sate_root = os.environ.get("SATEHAZE1K_ROOT", "/root/autodl-tmp/SateHaze1k")
        if args.test_hazy_dir is None:
            args.test_hazy_dir = os.path.join(sate_root, "test", "hazy")
        if args.test_clear_dir is None:
            args.test_clear_dir = resolve_clear_dir(sate_root, split="test")
        if args.tag == "nupw":
            args.tag = "satehaze1k"
    else:
        nupw_root = os.environ.get("NUPW_ROOT", "/root/autodl-tmp/hazy_nupw")
        if args.test_hazy_dir is None:
            args.test_hazy_dir = os.path.join(nupw_root, "test", "hazy")
        if args.test_clear_dir is None:
            args.test_clear_dir = resolve_clear_dir(nupw_root, split="test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = os.path.join(args.log_dir, args.tag)
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(os.path.join(log_dir, f"{args.tag}_test.log"))
    logger.info("=" * 70)
    logger.info("Evaluation start time: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if args.model_path:
        model_path = args.model_path
    else:
        fname = "best_model.pth" if args.model_type == "best" else "latest_model.pth"
        model_path = os.path.join(args.checkpoint_dir, args.tag, fname)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info("tag=%s, model=%s, device=%s", args.tag, model_path, device)
    if not os.path.exists(args.test_hazy_dir) or not os.path.exists(args.test_clear_dir):
        raise FileNotFoundError("Evaluation directories do not exist. Check test_hazy_dir and test_clear_dir.")

    save_dir = os.path.join(args.output_dir, args.tag)
    if args.save_results:
        os.makedirs(save_dir, exist_ok=True)

    model, ckpt = load_model(model_path, device, args.base_channels)
    if "best_psnr" in ckpt:
        logger.info("checkpoint best_psnr=%.4f best_ssim=%.4f", ckpt.get("best_psnr", 0.0), ckpt.get("best_ssim", 0.0))

    ds = EvalDataset(args.test_hazy_dir, args.test_clear_dir, resize_hazy_to_clear=not args.no_resize_hazy)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    rows, avg_psnr, avg_ssim = evaluate(model, loader, device, save_results=args.save_results, output_dir=save_dir)

    logger.info("Number of evaluation samples: %d", len(rows))
    logger.info("Average PSNR: %.4f dB", avg_psnr)
    logger.info("Average SSIM: %.4f", avg_ssim)

    if args.save_csv:
        csv_path = os.path.join(log_dir, f"{args.tag}_test_metrics.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "psnr", "ssim"])
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Saved evaluation details to: %s", csv_path)
    if args.save_results:
        logger.info("Saved dehazed outputs to: %s", save_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

