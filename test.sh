#!/usr/bin/env bash
# Evaluate a trained checkpoint on a specified test set.
# Wraps test.py with convenient CLI for checkpoint and data paths.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------- defaults ----------
MODEL_PATH="checkpoints/satehaze1k/best_model.pth"
HAZY_DIR="/home/jason/dataset/SateHaze1k/test/hazy"
CLEAR_DIR="/home/jason/dataset/SateHaze1k/test/clear"
CHECKPOINT_DIR="checkpoints"
TAG="satehaze1k"
MODEL_TYPE="best"          # best | latest
DATASET_PRESET=""          # empty = custom dirs; nupw | satehaze1k
BATCH_SIZE=1
NUM_WORKERS=4
BASE_CHANNELS=-1
OUTPUT_DIR="test_results"
LOG_DIR="logs"
SAVE_RESULTS=1
SAVE_CSV=0
NO_RESIZE_HAZY=0
PYTHON="${PYTHON:-python}"

usage() {
    cat <<'EOF'
Usage: ./test.sh [options]

Checkpoint (pick one):
  -c, --checkpoint PATH       Path to a .pth weight file
  -C, --checkpoint-dir DIR    Checkpoint root (used with --tag and --model-type)
  -t, --tag TAG               Experiment tag, default nupw (with -C)
  -m, --model-type TYPE       best or latest, default best (with -C)

Test data (pick one):
  -H, --hazy-dir PATH         Hazy image directory
  -G, --clear-dir PATH        Clear/GT directory (filenames must match hazy)
  -p, --preset PRESET         Dataset preset: nupw | satehaze1k (uses NUPW_ROOT / SATEHAZE1K_ROOT)

Optional:
  -b, --batch-size N          Batch size, default 1
  -w, --num-workers N         DataLoader workers, default 4
  --base-channels N           Model base_channels; <=0 reads from checkpoint, default -1
  -o, --output-dir DIR        Dehazed output directory, default test_results
  -l, --log-dir DIR           Log directory, default logs
  -s, --save-results          Save dehazed images
  --save-csv                  Save per-image PSNR/SSIM to CSV
  --no-resize-hazy            Do not resize hazy to clear resolution
  -h, --help                  Show this help

Environment:
  NUPW_ROOT          NUPW dataset root (default /root/autodl-tmp/hazy_nupw)
  SATEHAZE1K_ROOT    SateHaze1k root (default /root/autodl-tmp/SateHaze1k)

Examples:
  # Explicit checkpoint and test directories
  ./test.sh -c checkpoints/nupw/best_model.pth \
    -H /data/test/hazy -G /data/test/clear --save-csv

  # Resolve best model by tag
  ./test.sh -C checkpoints -t nupw -m best \
    -H /data/test/hazy -G /data/test/clear -s

  # Dataset preset (set NUPW_ROOT or use default paths)
  ./test.sh -c checkpoints/nupw/best_model.pth -p nupw --save-csv

  ./test.sh -C checkpoints -t satehaze1k -p satehaze1k --save-csv -s
EOF
}

die() {
    echo "错误: $*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--checkpoint)       MODEL_PATH="$2"; shift 2 ;;
        -C|--checkpoint-dir)  CHECKPOINT_DIR="$2"; shift 2 ;;
        -t|--tag)              TAG="$2"; shift 2 ;;
        -m|--model-type)       MODEL_TYPE="$2"; shift 2 ;;
        -H|--hazy-dir)         HAZY_DIR="$2"; shift 2 ;;
        -G|--clear-dir)        CLEAR_DIR="$2"; shift 2 ;;
        -p|--preset)           DATASET_PRESET="$2"; shift 2 ;;
        -b|--batch-size)       BATCH_SIZE="$2"; shift 2 ;;
        -w|--num-workers)      NUM_WORKERS="$2"; shift 2 ;;
        --base-channels)       BASE_CHANNELS="$2"; shift 2 ;;
        -o|--output-dir)       OUTPUT_DIR="$2"; shift 2 ;;
        -l|--log-dir)          LOG_DIR="$2"; shift 2 ;;
        -s|--save-results)     SAVE_RESULTS=1; shift ;;
        --save-csv)            SAVE_CSV=1; shift ;;
        --no-resize-hazy)      NO_RESIZE_HAZY=1; shift ;;
        -h|--help)             usage; exit 0 ;;
        *) die "未知参数: $1（使用 -h 查看帮助）" ;;
    esac
done

# ---------- validate ----------
if [[ -z "$MODEL_PATH" ]]; then
    fname="best_model.pth"
    [[ "$MODEL_TYPE" == "latest" ]] && fname="latest_model.pth"
    MODEL_PATH="${CHECKPOINT_DIR}/${TAG}/${fname}"
fi

[[ -f "$MODEL_PATH" ]] || die "找不到模型文件: $MODEL_PATH"

USE_PRESET=0
if [[ -n "$DATASET_PRESET" ]]; then
    USE_PRESET=1
    [[ "$DATASET_PRESET" == "nupw" || "$DATASET_PRESET" == "satehaze1k" ]] \
        || die "--preset 只能是 nupw 或 satehaze1k"
fi

if [[ $USE_PRESET -eq 0 ]]; then
    [[ -n "$HAZY_DIR" && -n "$CLEAR_DIR" ]] \
        || die "请同时指定 -H/--hazy-dir 与 -G/--clear-dir，或使用 -p/--preset"
    [[ -d "$HAZY_DIR" ]] || die "有雾目录不存在: $HAZY_DIR"
    [[ -d "$CLEAR_DIR" ]] || die "清晰目录不存在: $CLEAR_DIR"
fi

# ---------- build command ----------
CMD=(
    "$PYTHON" test.py
    --model_path "$MODEL_PATH"
    --tag "$TAG"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --base_channels "$BASE_CHANNELS"
    --output_dir "$OUTPUT_DIR"
    --log_dir "$LOG_DIR"
)

if [[ $USE_PRESET -eq 1 ]]; then
    CMD+=(--dataset_preset "$DATASET_PRESET")
else
    CMD+=(--test_hazy_dir "$HAZY_DIR" --test_clear_dir "$CLEAR_DIR")
fi

[[ $SAVE_RESULTS -eq 1 ]] && CMD+=(--save_results)
[[ $SAVE_CSV -eq 1 ]] && CMD+=(--save_csv)
[[ $NO_RESIZE_HAZY -eq 1 ]] && CMD+=(--no_resize_hazy)

echo ">>> 运行评测"
echo "    模型:     $MODEL_PATH"
if [[ $USE_PRESET -eq 1 ]]; then
    echo "    数据集:   preset=$DATASET_PRESET"
else
    echo "    有雾目录: $HAZY_DIR"
    echo "    清晰目录: $CLEAR_DIR"
fi
echo "    命令:     ${CMD[*]}"
echo

exec "${CMD[@]}"
