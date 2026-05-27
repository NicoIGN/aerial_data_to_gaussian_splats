#!/bin/bash
#SBATCH --job-name=colmap-video
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --partition=jean-zellou
#SBATCH -o /mnt/common/hdd/slurm/logs/colmap-video-%j.out
#SBATCH -e /mnt/common/hdd/slurm/logs/colmap-video-%j.err

set -e

# =========================
# REQUIRED ENV
# =========================
: "${GIT_ROOT:?❌ GIT_ROOT is not set}"
: "${SCENE_ROOT:?❌ SCENE_ROOT is not set (example: /mnt/common/hdd/home/NBellaiche/data/statue)}"

# Optional overrides
FPS="${FPS:-10}"
DURATION="${DURATION:-20}"
OUTPUT_PATH="${OUTPUT_PATH:-$SCENE_ROOT/exports/video.mov}"

# =========================
# SLURM / ENV
# =========================
export SRUN_CPUS_PER_TASK=4
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

export HTTP_PROXY
export HTTPS_PROXY
export http_proxy
export https_proxy
export NO_PROXY
export MAX_JOBS

export MPLCONFIGDIR="$HOME_SLURM/.config/matplotlib"
mkdir -p "$MPLCONFIGDIR"

# =========================
# PATH CHECKS
# =========================
SCRIPT_PATH="$GIT_ROOT/scripts/rendering/make_colmap_video.py"
EXPORT_DIR="$(dirname "$OUTPUT_PATH")"

[ -d "$GIT_ROOT" ] || { echo "❌ GIT_ROOT not found: $GIT_ROOT"; exit 1; }
[ -d "$SCENE_ROOT" ] || { echo "❌ SCENE_ROOT not found: $SCENE_ROOT"; exit 1; }
[ -f "$SCRIPT_PATH" ] || { echo "❌ Script not found: $SCRIPT_PATH"; exit 1; }

mkdir -p "$EXPORT_DIR"

cd "$GIT_ROOT"

# =========================
# CONDA ENV
# =========================
CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
if [ -z "$CONDA_BASE" ]; then
    echo "❌ conda not found"
    exit 1
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate gsplat

# =========================
# AUTO-DETECT INPUTS
# =========================

# 1) COLMAP dir
COLMAP_DIR=""

if [ -d "$SCENE_ROOT/ori/colmap" ]; then
    COLMAP_DIR="$SCENE_ROOT/ori/colmap"
elif [ -d "$SCENE_ROOT/colmap" ]; then
    COLMAP_DIR="$SCENE_ROOT/colmap"
else
    COLMAP_DIR=$(find "$SCENE_ROOT" -type d -name colmap 2>/dev/null | head -n 1 || true)
fi

if [ -z "$COLMAP_DIR" ] || [ ! -d "$COLMAP_DIR" ]; then
    echo "❌ Could not find COLMAP directory under $SCENE_ROOT"
    exit 1
fi

# 2) latest splatfacto run
MODEL_BASE="$SCENE_ROOT/model3d/splatfacto"
[ -d "$MODEL_BASE" ] || { echo "❌ Model base not found: $MODEL_BASE"; exit 1; }

LATEST_MODEL_DIR=$(find "$MODEL_BASE" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)

if [ -z "$LATEST_MODEL_DIR" ] || [ ! -d "$LATEST_MODEL_DIR" ]; then
    echo "❌ Could not find latest model directory in $MODEL_BASE"
    exit 1
fi

LOAD_CONFIG="$LATEST_MODEL_DIR/config.yml"
[ -f "$LOAD_CONFIG" ] || { echo "❌ Missing config.yml: $LOAD_CONFIG"; exit 1; }

# =========================
# DEBUG INFO
# =========================
echo "========================"
echo "🎬 COLMAP VIDEO RUN ENV"
echo "========================"
echo "job_id        : ${SLURM_JOB_ID:-N/A}"
echo "hostname      : $(hostname)"
echo "user          : $(whoami)"
echo "pwd           : $(pwd)"
echo "python        : $(which python)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "GIT_ROOT      : $GIT_ROOT"
echo "SCENE_ROOT    : $SCENE_ROOT"
echo "SCRIPT_PATH   : $SCRIPT_PATH"
echo "COLMAP_DIR    : $COLMAP_DIR"
echo "LATEST_MODEL  : $LATEST_MODEL_DIR"
echo "LOAD_CONFIG   : $LOAD_CONFIG"
echo "FPS           : $FPS"
echo "DURATION      : $DURATION"
echo "OUTPUT_PATH   : $OUTPUT_PATH"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"gpu[{i}]: {p.name} | total_vram={p.total_memory/1024**3:.2f} GiB")
PY

echo
echo "========================"
echo "💾 RAM INFO"
echo "========================"
free -h || true

echo
echo "========================"
echo "🎮 GPU INFO"
echo "========================"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
    echo
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv || true
    echo
    nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory --format=csv || true
else
    echo "⚠️ nvidia-smi not available"
fi

# =========================
# OPTIONAL BACKGROUND MONITOR
# =========================
(
  while true; do
    echo
    echo "========================"
    echo "⏱ RESOURCE SNAPSHOT $(date)"
    echo "========================"
    free -h || true
    echo
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv || true
        echo
        nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory --format=csv || true
    fi
    sleep 120
  done
) &
MONITOR_PID=$!

cleanup() {
    kill "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT

# =========================
# RUN RENDER
# =========================
srun -v python "$SCRIPT_PATH" \
  --colmap "$COLMAP_DIR" \
  --load-config "$LOAD_CONFIG" \
  --fps "$FPS" \
  --duration "$DURATION" \
  --output "$OUTPUT_PATH"

echo "✅ Video render complete"
