# A Self-Supervised Physics-Aware Tri-Head Network for Remote Sensing Image Dehazing

## Overview

This repository provides **TAG-ASPNet**, a physics-aware tri-head network for remote-sensing image dehazing. The model jointly predicts:

- clean image `J`
- transmission map `t`
- atmospheric light map `A`

Outputs are constrained by the atmospheric scattering model (`I = J·t + A·(1−t)`), which improves interpretability and robustness.

This release focuses on **model inference and evaluation**. It includes the network definition, PSNR/SSIM metrics, and testing scripts. Download datasets and pre-trained weights from [Downloads](#downloads), then place checkpoints under `checkpoints/<tag>/` before running evaluation.

> **Code release plan:** Training code (supervised and self-supervised) and online haze synthesis (`haze_synthesis.py`) are **not included in this repository yet**. They will be uploaded **after the paper is published**.

## Downloads

Resources are hosted on Baidu Netdisk (百度网盘):

| Resource | Link | Extraction code |
|----------|------|-----------------|
| **HazeRS45** dataset | https://pan.baidu.com/s/1u8EKXjk1DTLT7pldPJuRmg | `bi2n` |
| **SateHaze1k** dataset | https://pan.baidu.com/s/1N2eVrgoDNIkzn27cPSokoQ | `hvpy` |
| **Pre-trained checkpoints** | https://pan.baidu.com/s/11HwSbjnLL03DbMyZ_rr62w | `sr27` |

After downloading:

1. **Datasets** — Extract and point environment variables or CLI paths to the dataset root, e.g.  
   `export SATEHAZE1K_ROOT=/path/to/SateHaze1k`  
   Expected layout: `<root>/test/hazy/` and `<root>/test/clear/` (or `gt/` / `GT/`).

2. **Checkpoints** — Extract into the project `checkpoints/` directory, preserving the per-tag subfolders (e.g. `checkpoints/satehaze1k/best_model.pth`, `checkpoints/HazeRS45/best_model.pth`).

## Project Structure

```text
our-supervised_github/
├── test.py                 # Evaluation entry (PSNR / SSIM)
├── test.sh                 # Shell wrapper for checkpoint + test data paths
├── requirements.txt
├── model/
│   ├── dehaze_net.py       # TAGASPNet definition
│   └── __init__.py
├── utils/
│   ├── metrics.py          # PSNR, SSIM, MS-SSIM
│   ├── perceptual_loss.py  # VGG perceptual loss (for training pipelines)
│   └── ema.py              # Exponential moving average helper
├── checkpoints/            # Put trained weights here (not shipped)
│   └── <tag>/
│       ├── best_model.pth
│       └── latest_model.pth
├── logs/                   # Evaluation logs
└── test_results/           # Dehazed outputs (when --save_results is set)
```

## Model

Main class: `TAGASPNet` in `model/dehaze_net.py` (alias `DehazeNet`).

| Component | Description |
|-----------|-------------|
| Encoder | Stacked `RRB` blocks with two stride-2 downsampling stages |
| Bottleneck | `TRARB` blocks + `LKDRB` + `SEBlock` |
| Decoder | Upsampling with `DSFB` skip fusion and post-refinement `RRB` blocks |
| Heads | `J` (residual on input), `t`, `A` |

Output ranges:

| Output | Range |
|--------|-------|
| `J` | `[0, 1]` (residual refinement on input) |
| `t` | `[0.1, 1.0]` |
| `A` | `[0.75, 0.95]` |

`base_channels` controls model width (commonly `32` or `48`). During evaluation, if `--base_channels <= 0`, the value is read from `checkpoint["args"]["base_channels"]` when available.

## Checkpoint Format

`test.py` accepts either:

1. A training checkpoint dict containing `model_state_dict` (and optionally `args`, `best_psnr`, `best_ssim`), or  
2. A raw `state_dict` saved directly from the model.

Expected layout:

```text
checkpoints/<tag>/best_model.pth
checkpoints/<tag>/latest_model.pth
```

## Dataset Layout

Evaluation expects **paired** hazy and clear/GT images with **matching filenames** in two directories.

Supported image extensions: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`.

### Custom directories

```text
/path/to/test/hazy/   # hazy inputs
/path/to/test/clear/  # ground truth (also accepts gt/ or GT/)
```

If hazy and clear resolutions differ, hazy images are resized to the clear size by default (disable with `--no_resize_hazy`).

### Built-in presets

| Preset | Env var | Test paths |
|--------|---------|------------|
| `nupw` | `NUPW_ROOT` | `<root>/test/hazy`, `<root>/test/clear` |
| `satehaze1k` | `SATEHAZE1K_ROOT` | `<root>/test/hazy`, `<root>/test/{clear,gt,GT}` |

For **HazeRS45** or other datasets, use custom directories (`-H` / `-G` in `test.sh`, or `--test_hazy_dir` / `--test_clear_dir` in `test.py`) and the matching checkpoint tag under `checkpoints/`.

Explicit `--test_hazy_dir` / `--test_clear_dir` override preset paths.

## Installation

```bash
pip install -r requirements.txt
```

Requirements:

- Python 3.8+
- torch==2.1.2
- torchvision==0.16.2
- numpy==1.26.4
- Pillow==10.4.0
- tqdm==4.66.4

For GPU evaluation, install a PyTorch build that matches your CUDA runtime.

## Evaluation

### Quick start with `test.sh`

Edit the default variables at the top of `test.sh`, or pass CLI flags:

```bash
chmod +x test.sh

# Explicit checkpoint + test directories
./test.sh -c checkpoints/satehaze1k/best_model.pth \
  -H /path/to/test/hazy \
  -G /path/to/test/clear \
  --save-csv -s

# Resolve checkpoint by tag (e.g. satehaze1k, HazeRS45)
./test.sh -C checkpoints -t HazeRS45 -m best \
  -H /path/to/HazeRS45/test/hazy \
  -G /path/to/HazeRS45/test/clear \
  --save-csv

# Dataset preset (uses SATEHAZE1K_ROOT or default path)
export SATEHAZE1K_ROOT=/path/to/SateHaze1k
./test.sh -c checkpoints/satehaze1k/best_model.pth -p satehaze1k --save-csv -s
```

`test.sh` options (run `./test.sh -h` for full help):

| Flag | Description |
|------|-------------|
| `-c, --checkpoint` | Path to `.pth` weight file |
| `-C, --checkpoint-dir` | Checkpoint root (with `-t`, `-m`) |
| `-t, --tag` | Experiment tag subdirectory name |
| `-m, --model-type` | `best` or `latest` |
| `-H, --hazy-dir` | Hazy image directory |
| `-G, --clear-dir` | Clear/GT directory |
| `-p, --preset` | `nupw` or `satehaze1k` |
| `-s, --save-results` | Save dehazed images |
| `--save-csv` | Write per-image metrics CSV |
| `--no-resize-hazy` | Skip resizing hazy to clear size |

### Direct use of `test.py`

```bash
# Custom paths
python test.py \
  --model_path checkpoints/satehaze1k/best_model.pth \
  --test_hazy_dir /path/to/test/hazy \
  --test_clear_dir /path/to/test/clear \
  --tag satehaze1k \
  --save_csv --save_results

# Auto-resolve checkpoint by tag
python test.py --dataset_preset nupw --tag nupw --model_type best --save_csv

python test.py --dataset_preset satehaze1k --tag satehaze1k --model_type best --save_csv
```

`test.py` arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | `None` | Checkpoint file; if omitted, uses `checkpoints/<tag>/best_model.pth` or `latest_model.pth` |
| `--test_hazy_dir` | preset | Hazy directory |
| `--test_clear_dir` | preset | Clear/GT directory |
| `--dataset_preset` | `nupw` | `nupw` or `satehaze1k` |
| `--checkpoint_dir` | `checkpoints` | Root directory for auto-resolved weights |
| `--tag` | `nupw` | Experiment tag |
| `--model_type` | `best` | `best` or `latest` when `--model_path` is not set |
| `--batch_size` | `1` | Evaluation batch size |
| `--num_workers` | `4` | DataLoader workers |
| `--base_channels` | `-1` | Model width; `<=0` reads from checkpoint |
| `--save_results` | off | Save dehazed images |
| `--output_dir` | `test_results` | Output root for dehazed images |
| `--save_csv` | off | Save per-image PSNR/SSIM |
| `--log_dir` | `logs` | Log directory |
| `--no_resize_hazy` | off | Do not resize hazy to clear resolution |

## Outputs

| Path | Content |
|------|---------|
| `logs/<tag>/<tag>_test.log` | Evaluation log (averages, checkpoint info) |
| `logs/<tag>/<tag>_test_metrics.csv` | Per-image PSNR/SSIM (with `--save_csv`) |
| `test_results/<tag>/` | Dehazed images (with `--save_results`) |

Example log line:

```text
Average PSNR: 24.7033 dB
Average SSIM: 0.9083
```

## Utilities

| Module | Purpose |
|--------|---------|
| `utils/metrics.py` | `psnr`, `ssim`, `ms_ssim` for evaluation and training |
| `utils/perceptual_loss.py` | VGG16-based perceptual loss |
| `utils/ema.py` | `ModelEMA` for weight smoothing during training |

## Acknowledgements

Parts of this project were developed with reference to the following open-source repositories:

- [DEA-Net](https://github.com/cecret3350/DEA-Net)
- [DehazeFormer](https://github.com/IDKiro/DehazeFormer)
