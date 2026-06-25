#!/bin/bash
set -euo pipefail

# =====================================================
# TWO-STAGE NERFSTUDIO TRAINING (COARSE -> FULL RES)
# =====================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ======================
# UTILS
# ======================

add_arg() {
  local -n arr="$1"
  local flag="$2"
  local value="${3-}"
  [[ -z "$value" ]] && return
  arr+=("$flag" "$value")
}

add_bool_arg() {
  local -n arr="$1"
  local flag="$2"
  local value="${3-}"
  [[ -z "$value" ]] && return

  case "$value" in
    True|true|1)   value="True" ;;
    False|false|0) value="False" ;;
    *) echo "⚠️ Invalid boolean value for $flag: $value"; return ;;
  esac

  arr+=("$flag" "$value")
}

add_multi_arg() {
  local -n arr="$1"
  local flag="$2"
  shift 2
  [[ $# -eq 0 ]] && return
  arr+=("$flag")
  for arg in "$@"; do
    arr+=("$arg")
  done
}

bool_true() {
  local v="${1:-}"
  [[ "$v" =~ ^(True|true|1|on|ON|yes|YES)$ ]]
}

# ======================
# SAFETY CHECKS
# ======================

if [[ -z "${DATA:-}" ]]; then
  echo "❌ DATA is empty (check run.sh)"
  exit 1
fi

if [[ ! -f "$DATA/transforms.json" ]]; then
  echo "❌ Missing transforms.json in $DATA"
  exit 1
fi

# ======================
# DEFAULTS (GLOBAL)
# ======================

VERBOSE=${VERBOSE:-False}
TRAIN_VIS_MODE=${TRAIN_VIS_MODE:-tensorboard}
STEPS_PER_SAVE=${STEPS_PER_SAVE:-5000}
STEPS_PER_LOG=${STEPS_PER_LOG:-250}
STEPS_PER_EVAL_ALL_IMAGES=${STEPS_PER_EVAL_ALL_IMAGES:-3000}

# Device
DEVICE=${DEVICE:-gpu}
MIXED_PRECISION=${MIXED_PRECISION:-True}
USE_GRAD_SCALER=${USE_GRAD_SCALER:-True}
MAX_JOBS=${MAX_JOBS:-4}
MODEL=${MODEL:-splatfacto}
MODEL_IMPLEMENTATION=${MODEL_IMPLEMENTATION:-tcnn}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-exp_two_stage}
OUTPUTDIR=${OUTPUTDIR:-./outputs}

# Splat defaults (fallbacks)
DENSIFY_GRAD_THRESH=${DENSIFY_GRAD_THRESH:-0.00045}
CULL_ALPHA_THRESH=${CULL_ALPHA_THRESH:-0.12}
CULL_SCREEN_SIZE=${CULL_SCREEN_SIZE:-0.25}
SPLIT_SCREEN_SIZE=${SPLIT_SCREEN_SIZE:-0.02}
REFINE_EVERY=${REFINE_EVERY:-300}
MAX_GAUSS_RATIO=${MAX_GAUSS_RATIO:-4.0}
STOP_SPLIT_AT=${STOP_SPLIT_AT:-6000}
CULL_SCALE_THRESH=${CULL_SCALE_THRESH:-0.5}
RESET_ALPHA_EVERY=${RESET_ALPHA_EVERY:-40}
SSIM_LAMBDA=${SSIM_LAMBDA:-0.25}
USE_BILATERAL_GRID=${USE_BILATERAL_GRID:-True}
USE_SCALE_REGULARIZATION=${USE_SCALE_REGULARIZATION:-False}

ENABLE_COLLIDER=${ENABLE_COLLIDER:-False}
COLLIDER_NEAR=${COLLIDER_NEAR:-}
COLLIDER_FAR=${COLLIDER_FAR:-}

# IMPORTANT: use DOWNSCALE_FACTOR (not NUM_DOWNSCALES)
# Stage-specific defaults
COARSE_MAX_ITER=${COARSE_MAX_ITER:-5000}
COARSE_CAMERA_RES_SCALE_FACTOR=${COARSE_CAMERA_RES_SCALE_FACTOR:-0.5}
COARSE_DOWNSCALE_FACTOR=${COARSE_DOWNSCALE_FACTOR:-2}
COARSE_TRAIN_RAYS_PER_BATCH=${COARSE_TRAIN_RAYS_PER_BATCH:-512}
COARSE_REFINE_EVERY=${COARSE_REFINE_EVERY:-300}
COARSE_DENSIFY_GRAD_THRESH=${COARSE_DENSIFY_GRAD_THRESH:-0.00045}
COARSE_STOP_SPLIT_AT=${COARSE_STOP_SPLIT_AT:-4000}

FULL_MAX_ITER=${FULL_MAX_ITER:-3000}
FULL_CAMERA_RES_SCALE_FACTOR=${FULL_CAMERA_RES_SCALE_FACTOR:-1.0}
FULL_DOWNSCALE_FACTOR=${FULL_DOWNSCALE_FACTOR:-1}
FULL_TRAIN_RAYS_PER_BATCH=${FULL_TRAIN_RAYS_PER_BATCH:-256}
FULL_REFINE_EVERY=${FULL_REFINE_EVERY:-500}
FULL_DENSIFY_GRAD_THRESH=${FULL_DENSIFY_GRAD_THRESH:-0.00060}
FULL_STOP_SPLIT_AT=${FULL_STOP_SPLIT_AT:-1500}

# Optional two-step full-res ramp (0.85 then 1.0) if you want:
# FULL_CAMERA_RES_SCALE_FACTOR=0.85 first, then rerun full stage with 1.0

# ======================
# DEVICE CONFIG
# ======================

if [[ "$DEVICE" == "gpu" ]]; then
  MACHINE_DEVICE_TYPE="cuda"
elif [[ "$DEVICE" == "cpu" ]]; then
  MACHINE_DEVICE_TYPE="cpu"
  export TORCHDYNAMO_DISABLE=1
  export OMP_NUM_THREADS=1
else
  echo "❌ CONFIGURATION ERROR: DEVICE must be cpu or gpu"
  exit 1
fi

export MACHINE_DEVICE_TYPE
export MODEL_IMPLEMENTATION
export MAX_JOBS
export NUM_DEVICES=1
export NUM_MACHINES=1
export CMAKE_BUILD_PARALLEL_LEVEL="$MAX_JOBS"

export TORCH_DISABLE_ADDR2LINE=1
export TORCHINDUCTOR_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export TORCH_USE_CUDA_DSA=1
export TORCH_SHOW_CPP_STACKTRACES=1

# ======================
# CUDA ARCH AUTO-DETECTION
# ======================

if command -v python3 >/dev/null 2>&1; then
  if python3 -c "import torch" >/dev/null 2>&1; then
    export TORCH_CUDA_ARCH_LIST=$(
      python3 - << 'EOF'
import torch
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    print(f"{cap[0]}.{cap[1]}")
EOF
    )
    echo "⚙️ TORCH_CUDA_ARCH_LIST auto-set to: ${TORCH_CUDA_ARCH_LIST:-<empty>}"
  else
    echo "⚠️ torch not available in python, skipping TORCH_CUDA_ARCH_LIST"
  fi
else
  echo "⚠️ python3 not found, skipping TORCH_CUDA_ARCH_LIST"
fi

# ======================
# PIL LARGE IMAGE PATCH
# ======================

PIL_PATCH_DIR="$SCRIPT_DIR/.python_patches"
mkdir -p "$PIL_PATCH_DIR"

cat > "$PIL_PATCH_DIR/sitecustomize.py" <<'PYEOF'
from PIL import Image
import warnings
Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter("ignore", Image.DecompressionBombWarning)
PYEOF

export PYTHONPATH="$PIL_PATCH_DIR${PYTHONPATH:+:$PYTHONPATH}"

# ======================
# LOGGING SETUP
# ======================

LOG_DIR="$OUTPUTDIR/logs"
mkdir -p "$LOG_DIR"

if bool_true "$VERBOSE"; then
  export LOGLEVEL=DEBUG
  export PYTHONUNBUFFERED=1
  export NCCL_DEBUG=INFO
else
  export LOGLEVEL=INFO
fi

# ======================
# CHECKPOINT HELPERS
# ======================

get_model_dir_name() {
  if [[ "$MODEL" == *splat* ]]; then
    echo "splatfacto"
  else
    echo "nerfacto"
  fi
}

find_latest_checkpoint_dir() {
  local base_dir="$1"
  local found=""
  if [[ -d "$base_dir/nerfstudio_models" ]]; then
    found="$base_dir/nerfstudio_models"
  fi
  if [[ -d "$base_dir" ]]; then
    local last_run
    last_run=$(ls -td "$base_dir"/*/nerfstudio_models 2>/dev/null | head -n 1 || true)
    if [[ -n "$last_run" ]]; then
      found="$last_run"
    fi
  fi
  echo "$found"
}

# ======================
# STAGE RUNNER
# ======================

run_stage() {
  local stage_name="$1"                 # coarse / full
  local max_iter="$2"
  local camera_res_scale_factor="$3"
  local downscale_factor="$4"
  local train_rays_per_batch="$5"
  local stage_refine_every="$6"
  local stage_densify_grad_thresh="$7"
  local stage_stop_split_at="$8"
  local load_dir="$9"

  local train_log="$LOG_DIR/ns_train_${stage_name}.log"
  local heartbeat_log="$LOG_DIR/ns_train_${stage_name}_heartbeat.log"

  echo "────────────────────────────────────────────"
  echo "🚀 STAGE: $stage_name"
  echo "────────────────────────────────────────────"
  echo "📁 DATA                       : $DATA"
  echo "📁 OUTPUTDIR                  : $OUTPUTDIR"
  echo "🧪 MODEL                      : $MODEL"
  echo "⚙️ DEVICE                     : $DEVICE"
  echo "🔁 MAX ITER                   : $max_iter"
  echo "🖼️ CAMERA_RES_SCALE_FACTOR    : $camera_res_scale_factor"
  echo "🧱 DOWNSCALE_FACTOR           : $downscale_factor"
  echo "🎯 TRAIN_RAYS_PER_BATCH       : $train_rays_per_batch"
  echo "✨ REFINE_EVERY               : $stage_refine_every"
  echo "✨ DENSIFY_GRAD_THRESH        : $stage_densify_grad_thresh"
  echo "✂️ STOP_SPLIT_AT              : $stage_stop_split_at"
  echo "♻️ LOAD_DIR                   : ${load_dir:-<none>}"
  echo "📝 LOG                        : $train_log"

  # heartbeat
  (
    while true; do
      sleep 60
      echo "$(date '+%F %T') ns-train [$stage_name] still active" >> "$heartbeat_log"
    done
  ) &
  local heartbeat_pid=$!

  COMMON_ARGS=()
  PERF_ARGS=()
  MODEL_ARGS=()
  LOGGING_ARGS=()

  # Common args
  add_arg COMMON_ARGS --output-dir "$OUTPUTDIR"
  add_arg COMMON_ARGS --experiment-name "$EXPERIMENT_NAME"
  add_arg COMMON_ARGS --steps-per-save "$STEPS_PER_SAVE"
  add_arg COMMON_ARGS --vis "$TRAIN_VIS_MODE"
  add_arg COMMON_ARGS --logging.steps-per-log "$STEPS_PER_LOG"

  add_bool_arg COMMON_ARGS --save-only-latest-checkpoint True
  add_bool_arg COMMON_ARGS --mixed_precision "$MIXED_PRECISION"
  add_bool_arg COMMON_ARGS --use_grad_scaler "$USE_GRAD_SCALER"
  add_bool_arg COMMON_ARGS --logging.local-writer.enable True
  add_bool_arg COMMON_ARGS --viewer.quit-on-train-completion True
  add_arg COMMON_ARGS --load-dir "$load_dir"

  # Perf args
  add_arg PERF_ARGS --machine.device-type "$MACHINE_DEVICE_TYPE"
  add_arg PERF_ARGS --machine.num-devices "${NUM_DEVICES:-1}"
  add_arg PERF_ARGS --machine.num-machines "${NUM_MACHINES:-1}"
  add_arg PERF_ARGS --max-num-iterations "$max_iter"
  add_arg PERF_ARGS --steps-per-eval-all-images "$STEPS_PER_EVAL_ALL_IMAGES"
  add_bool_arg PERF_ARGS --mixed-precision "$MIXED_PRECISION"
  add_bool_arg PERF_ARGS --use-grad-scaler "$USE_GRAD_SCALER"

  if bool_true "$VERBOSE"; then
    add_arg LOGGING_ARGS --logging.local-writer.max-log-size 0
  fi

  # Model args
  if [[ "$DEVICE" == "gpu" ]]; then
    add_arg MODEL_ARGS       --pipeline.datamanager.train-num-rays-per-batch "$train_rays_per_batch"
    add_arg MODEL_ARGS       --pipeline.datamanager.camera-res-scale-factor "$camera_res_scale_factor"
    add_arg MODEL_ARGS       --pipeline.datamanager.dataparser.downscale-factor "$downscale_factor"

    add_arg MODEL_ARGS       --pipeline.datamanager.cache-images cpu
    add_bool_arg MODEL_ARGS  --pipeline.datamanager.images-on-gpu False
    add_bool_arg MODEL_ARGS  --pipeline.datamanager.masks-on-gpu False

    add_arg MODEL_ARGS       --pipeline.model.densify-grad-thresh "$stage_densify_grad_thresh"
    add_arg MODEL_ARGS       --pipeline.model.cull-alpha-thresh "$CULL_ALPHA_THRESH"
    add_arg MODEL_ARGS       --pipeline.model.cull-screen-size "$CULL_SCREEN_SIZE"
    add_arg MODEL_ARGS       --pipeline.model.split-screen-size "$SPLIT_SCREEN_SIZE"
    add_arg MODEL_ARGS       --pipeline.model.refine-every "$stage_refine_every"
    add_bool_arg MODEL_ARGS  --pipeline.model.use-bilateral-grid "$USE_BILATERAL_GRID"
    add_bool_arg MODEL_ARGS  --pipeline.model.use-scale-regularization "$USE_SCALE_REGULARIZATION"
    add_arg MODEL_ARGS       --pipeline.model.max-gauss-ratio "$MAX_GAUSS_RATIO"
    add_arg MODEL_ARGS       --pipeline.model.stop-split-at "$stage_stop_split_at"
    add_arg MODEL_ARGS       --pipeline.model.cull-scale-thresh "$CULL_SCALE_THRESH"
    add_arg MODEL_ARGS       --pipeline.model.reset-alpha-every "$RESET_ALPHA_EVERY"
    add_arg MODEL_ARGS       --pipeline.model.ssim-lambda "$SSIM_LAMBDA"
    add_bool_arg MODEL_ARGS  --pipeline.model.enable-collider "$ENABLE_COLLIDER"

    if [[ "$ENABLE_COLLIDER" == "True" ]]; then
      ARGS=()
      [[ -n "${COLLIDER_NEAR:-}" ]] && ARGS+=("near_plane" "$COLLIDER_NEAR")
      [[ -n "${COLLIDER_FAR:-}"  ]] && ARGS+=("far_plane"  "$COLLIDER_FAR")
      if [[ ${#ARGS[@]} -gt 0 ]]; then
        add_multi_arg MODEL_ARGS --pipeline.model.collider-params "${ARGS[@]}"
      fi
    fi

    if [[ "$stage_stop_split_at" -eq 0 ]]; then
      add_arg MODEL_ARGS --optimizers.means.optimizer.lr 0.0001
      add_arg MODEL_ARGS --pipeline.model.camera-optimizer.mode off
    fi
  else
    # CPU fallback
    add_arg MODEL_ARGS --pipeline.datamanager.camera-res-scale-factor "$camera_res_scale_factor"
    add_arg MODEL_ARGS --pipeline.model.implementation "$MODEL_IMPLEMENTATION"
    add_arg MODEL_ARGS --pipeline.model.max-res "${MAX_RES:-1024}"
  fi

  set +e
  ns-train \
    "$MODEL" \
    "${COMMON_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    nerfstudio-data \
    --data "$DATA" \
    > >(tee -a "$train_log") \
    2> >(tee -a "$train_log" >&2)
  local status=$?
  set -e

  kill "$heartbeat_pid" 2>/dev/null || true

  if [[ "$status" -ne 0 ]]; then
    echo "❌ Stage '$stage_name' crashed (exit code: $status)"
    tail -50 "$train_log" || true
    exit "$status"
  fi

  if ! tail -n 20 "$train_log" | grep -q "Training Finished"; then
    echo "⚠️ Stage '$stage_name' may have stopped unexpectedly"
    tail -50 "$train_log" || true
  fi

  echo "✅ Stage '$stage_name' completed"
}

# ======================
# MAIN FLOW (2 STAGES)
# ======================

MODEL_DIR="$(get_model_dir_name)"
BASE_DIR="$OUTPUTDIR/$EXPERIMENT_NAME/$MODEL_DIR"

echo "────────────────────────────────────────────"
echo "🔍 PRE-CHECKPOINT"
echo "────────────────────────────────────────────"
echo "📂 BASE_DIR: $BASE_DIR"

# Stage 1: COARSE (fresh start by default)
COARSE_LOAD_DIR=""
echo "➡️ Launching COARSE stage..."
run_stage \
  "coarse" \
  "$COARSE_MAX_ITER" \
  "$COARSE_CAMERA_RES_SCALE_FACTOR" \
  "$COARSE_DOWNSCALE_FACTOR" \
  "$COARSE_TRAIN_RAYS_PER_BATCH" \
  "$COARSE_REFINE_EVERY" \
  "$COARSE_DENSIFY_GRAD_THRESH" \
  "$COARSE_STOP_SPLIT_AT" \
  "$COARSE_LOAD_DIR"

# Find checkpoint after stage 1
FULL_LOAD_DIR="$(find_latest_checkpoint_dir "$BASE_DIR")"
if [[ -z "$FULL_LOAD_DIR" ]]; then
  echo "❌ No checkpoint found after coarse stage; cannot start full-res stage."
  exit 1
fi

echo "➡️ Launching FULL stage from: $FULL_LOAD_DIR"
run_stage \
  "full" \
  "$FULL_MAX_ITER" \
  "$FULL_CAMERA_RES_SCALE_FACTOR" \
  "$FULL_DOWNSCALE_FACTOR" \
  "$FULL_TRAIN_RAYS_PER_BATCH" \
  "$FULL_REFINE_EVERY" \
  "$FULL_DENSIFY_GRAD_THRESH" \
  "$FULL_STOP_SPLIT_AT" \
  "$FULL_LOAD_DIR"

# Final validation
CKPT_DIR="$(find_latest_checkpoint_dir "$BASE_DIR")"
if [[ -z "$CKPT_DIR" ]]; then
  echo "⚠️ Training finished but no checkpoint found"
  exit 1
fi

echo "────────────────────────────────────────────"
echo "✅ TWO-STAGE TRAINING COMPLETE"
echo "📦 Final checkpoint directory: $CKPT_DIR"
echo "📝 Logs:"
echo "   - $LOG_DIR/ns_train_coarse.log"
echo "   - $LOG_DIR/ns_train_full.log"
echo "────────────────────────────────────────────"
