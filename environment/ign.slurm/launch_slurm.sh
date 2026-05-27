#!/bin/bash
#SBATCH --job-name=gsplat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=jean-zellou
#SBATCH -o /mnt/common/hdd/slurm/logs/gsplat-%j.out
#SBATCH -e /mnt/common/hdd/slurm/logs/gsplat-%j.err

set -e
# set -x

# =========================
# LOAD USER CONFIG
# =========================
source config.sh

# Expected variables in config.sh:
#   ROOTDIR
#   BASENAME
#   GSPLAT_PROFILE
#   COLMAP_DIR
#   IMAGE_DIR

# =========================
# SRUN / SLURM SETTINGS
# =========================
export SRUN_CPUS_PER_TASK=8
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

# =========================
# PROXY / ENV
# =========================
export HTTP_PROXY
export HTTPS_PROXY
export http_proxy
export https_proxy
export NO_PROXY
export MAX_JOBS

# =========================
# MATPLOTLIB CACHE
# =========================
export MPLCONFIGDIR="$HOME_SLURM/.config/matplotlib"
mkdir -p "$MPLCONFIGDIR"

# =========================
# PROJECT
# =========================
cd "$HOME_SLURM/video_to_ply"

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
# INPUT VALIDATION
# =========================
if [ -z "${ROOTDIR:-}" ]; then
    echo "❌ ROOTDIR is not set in config.sh"
    exit 1
fi

if [ -z "${BASENAME:-}" ]; then
    echo "❌ BASENAME is not set in config.sh"
    exit 1
fi

if [ -z "${GSPLAT_PROFILE:-}" ]; then
    echo "❌ GSPLAT_PROFILE is not set in config.sh"
    exit 1
fi

if [ -z "${COLMAP_DIR:-}" ]; then
    echo "❌ COLMAP_DIR is not set in config.sh"
    exit 1
fi

if [ -z "${IMAGE_DIR:-}" ]; then
    echo "❌ IMAGE_DIR is not set in config.sh"
    exit 1
fi

if [ ! -d "$COLMAP_DIR" ]; then
    echo "❌ COLMAP_DIR does not exist: $COLMAP_DIR"
    exit 1
fi

if [ ! -d "$IMAGE_DIR" ]; then
    echo "❌ IMAGE_DIR does not exist: $IMAGE_DIR"
    exit 1
fi

# =========================
# DEBUG INFO
# =========================
echo "========================"
echo "🚀 GSPLAT RUN ENV"
echo "========================"
echo "job_id: ${SLURM_JOB_ID:-N/A}"
echo "hostname: $(hostname)"
echo "user: $(whoami)"
echo "pwd: $(pwd)"
echo "python: $(which python)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "ROOTDIR=$ROOTDIR"
echo "BASENAME=$BASENAME"
echo "GSPLAT_PROFILE=$GSPLAT_PROFILE"
echo "COLMAP_DIR=$COLMAP_DIR"
echo "IMAGE_DIR=$IMAGE_DIR"

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

echo
echo "========================"
echo "📦 TOP MEMORY PROCESSES"
echo "========================"
ps -eo pid,user,comm,%mem,%cpu,rss,vsz --sort=-rss | head -30 || true

echo
echo "========================"
echo "🧾 SLURM JOB INFO"
echo "========================"
scontrol show job "${SLURM_JOB_ID}" || true

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
# RUN TRAINING
# =========================
srun -v bash run.sh \
  --root "$ROOTDIR" \
  --name "$BASENAME" \
  --colmap-dir "$COLMAP_DIR" \
  --image-dir "$IMAGE_DIR" \
  --skip-conda \
  --gsplat-profile "$GSPLAT_PROFILE"

echo "✅ Job complete"
