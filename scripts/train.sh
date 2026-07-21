#!/bin/bash
set -e

# ======================
# LOAD CONFIG (CRITICAL)
# ======================

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
# DEFAULTS
# ======================

VERBOSE=${VERBOSE:-False}

TRAIN_VIS_MODE=${TRAIN_VIS_MODE:-tensorboard}

STEPS_PER_SAVE=${STEPS_PER_SAVE:-5000}

STEPS_PER_EVAL_ALL_IMAGES=${STEPS_PER_EVAL_ALL_IMAGES:-1000}

REFINE_EVERY=${REFINE_EVERY:-500}

STEPS_PER_LOG=${STEPS_PER_LOG:-250}

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

    echo "⚙️ TORCH_CUDA_ARCH_LIST auto-set to: $TORCH_CUDA_ARCH_LIST"

  else
    echo "⚠️ torch not available in python, skipping TORCH_CUDA_ARCH_LIST"
  fi

else
  echo "⚠️ python3 not found, skipping TORCH_CUDA_ARCH_LIST"
fi

# ======================
# SUMMARY
# ======================

echo "────────────────────────────────────────────"
echo "🚀 TRAINING CONFIG SUMMARY"
echo "────────────────────────────────────────────"

echo "📁 DATA                     : $DATA"
echo "📁 OUTPUTDIR                : $OUTPUTDIR"
echo "🧪 MODEL                    : $MODEL"
echo "🧪 MODEL_IMPLEMENTATION     : $MODEL_IMPLEMENTATION"
echo "🧪 EXPERIMENT_NAME          : $EXPERIMENT_NAME"

echo "⚙️ DEVICE                   : $DEVICE"
echo "⚙️ MIXED PRECISION          : $MIXED_PRECISION"
echo "⚙️ USE GRAD SCALER          : $USE_GRAD_SCALER"
echo "⚙️ USE DEFAULTS             : $USE_DEFAULTS"
echo "⚙️ RELOAD FROM CHECKPOINT   : $RELOAD_FROM_CHECKPOINT"

echo "🔁 MAX ITERATIONS           : $MAX_ITER"
echo "🔁 MAX JOBS                 : $MAX_JOBS"
echo "💾 STEPS PER SAVE           : $STEPS_PER_SAVE"
echo "🖼️ STEPS PER EVAL ALL IMG   : $STEPS_PER_EVAL_ALL_IMAGES"
echo "🧪 REFINE EVERY             : $REFINE_EVERY"

echo "📊 VIS MODE                 : $TRAIN_VIS_MODE"

echo "────────────────────────────────────────────"
echo "🧠 DATA PIPELINE"

echo "  - train rays per batch    : $TRAIN_RAYS_PER_BATCH"
echo "  - camera resolution scale : $CAMERA_RES_SCALE_FACTOR"

echo "────────────────────────────────────────────"
echo "🧠 MODEL CONFIG"

echo "  - implementation          : $MODEL_IMPLEMENTATION"
echo "  - max iterations          : $MAX_ITER"

echo "  - NeRF samples per ray    : $NUM_NERF_SAMPLES_PER_RAY"
echo "  - proposal samples/ray    : $NUM_PROPOSAL_SAMPLES_PER_RAY"

echo "  - max resolution          : $MAX_RES"
echo "  - max gaussians           : $MAX_GAUSSIANS"

echo "────────────────────────────────────────────"
echo "✨ GAUSSIAN SPLATTING"

echo "  - refine every            : $REFINE_EVERY"

echo "  - densify grad threshold  : $DENSIFY_GRAD_THRESH"

echo "  - cull alpha threshold    : $CULL_ALPHA_THRESH"
echo "  - cull scale threshold    : $CULL_SCALE_THRESH"
echo "  - cull screen size        : $CULL_SCREEN_SIZE"

echo "  - split screen size       : $SPLIT_SCREEN_SIZE"
echo "  - stop split at           : $STOP_SPLIT_AT"

echo "  - reset alpha every       : $RESET_ALPHA_EVERY"

echo "  - max gauss ratio         : $MAX_GAUSS_RATIO"

echo "  - scale regularization    : $USE_SCALE_REGULARIZATION"

echo "  - bilateral grid          : $USE_BILATERAL_GRID"

echo "  - ssim lambda             : $SSIM_LAMBDA"

echo "────────────────────────────────────────────"
echo "🧱 COLLIDER"

echo "  - enable collider         : $ENABLE_COLLIDER"
echo "  - near plane              : $COLLIDER_NEAR"
echo "  - far plane               : $COLLIDER_FAR"

echo "────────────────────────────────────────────"
echo "🔥 STARTING TRAINING..."
echo "────────────────────────────────────────────"

# ======================
# CHECKPOINT AUTO-RESUME
# ======================

echo "────────────────────────────────────────────"
echo "🔍 CHECKPOINT AUTO-RESUME"
echo "────────────────────────────────────────────"

LOAD_DIR=""

if [[ "$MODEL" == *splat* ]]; then
  MODEL_DIR="splatfacto"
else
  MODEL_DIR="nerfacto"
fi

BASE_DIR="$OUTPUTDIR/$EXPERIMENT_NAME/$MODEL_DIR"

echo "📂 BASE_DIR : $BASE_DIR"

if [[ -d "$BASE_DIR/nerfstudio_models" ]]; then

  LOAD_DIR="$BASE_DIR/nerfstudio_models"

  echo "✅ Direct checkpoint found"
  echo "📦 $LOAD_DIR"

fi

if [[ -d "$BASE_DIR" ]]; then

  LAST_RUN=$(ls -td "$BASE_DIR"/*/nerfstudio_models 2>/dev/null | head -n 1)

  if [[ -n "$LAST_RUN" ]]; then

    LOAD_DIR="$LAST_RUN"

    echo "✅ Latest checkpoint found"
    echo "📦 $LOAD_DIR"

  fi
fi

LOAD_STEP=""

if [[ -z "${RELOAD_FROM_CHECKPOINT:-}" || "${RELOAD_FROM_CHECKPOINT}" =~ ^(false|False|FALSE|0|off|OFF|no|NO)$ ]]; then
    LOAD_DIR=""
else
    if [[ -n "$LOAD_DIR" ]]; then
      # prend le dernier step-XXXXXXXX.ckpt du load_dir
      LAST_CKPT=$(ls -1 "$LOAD_DIR"/step-*.ckpt 2>/dev/null | sort | tail -n 1)
      if [[ -n "$LAST_CKPT" ]]; then
        LOAD_STEP=$(basename "$LAST_CKPT" | sed -E 's/^step-0*([0-9]+)\.ckpt$/\1/')
        echo "🔢 LOAD_STEP resolved to: $LOAD_STEP"
      fi
  fi
fi



# ======================
# RUN TIMESTAMP
# ======================

if [[ -n "$LOAD_DIR" ]]; then

  RUN_TIMESTAMP="$(basename "$(dirname "$LOAD_DIR")")"

  echo "♻️ Resuming existing run"
  echo "🕒 TIMESTAMP : $RUN_TIMESTAMP"

else

  RUN_TIMESTAMP="$(date +%Y-%m-%d_%H%M%S)"

  echo "🆕 Starting new run"
  echo "🕒 TIMESTAMP : $RUN_TIMESTAMP"

fi

# ======================
# COMMON ARGS
# ======================

COMMON_ARGS=()

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

add_arg COMMON_ARGS --load-dir "$LOAD_DIR"
if [[ -n "${LOAD_STEP:-}" ]]; then
  add_arg COMMON_ARGS --load-step "$LOAD_STEP"
else
  echo "ℹ️ LOAD_STEP empty -> starting from iteration 0"
fi
# ======================
# PERFORMANCE ARGS
# ======================

PERF_ARGS=()

add_arg PERF_ARGS --machine.device-type "$MACHINE_DEVICE_TYPE"
add_arg PERF_ARGS --machine.num-devices "${NUM_DEVICES:-1}"
add_arg PERF_ARGS --machine.num-machines "${NUM_MACHINES:-1}"
add_arg PERF_ARGS --max-num-iterations "$MAX_ITER"
add_arg PERF_ARGS --steps-per-eval-all-images "$STEPS_PER_EVAL_ALL_IMAGES"
add_bool_arg PERF_ARGS --mixed-precision "$MIXED_PRECISION"
add_bool_arg PERF_ARGS --use-grad-scaler "$USE_GRAD_SCALER"

# ======================
# MODEL ARGS
# ======================

MODEL_ARGS=()

#unset DENSIFY_GRAD_THRESH
#unset CULL_ALPHA_THRESH
#unset CULL_SCREEN_SIZE
#unset SPLIT_SCREEN_SIZE
  
if [[ "$DEVICE" == "gpu" ]]; then
        add_arg MODEL_ARGS       --pipeline.datamanager.camera-res-scale-factor "$CAMERA_RES_SCALE_FACTOR"
        add_arg MODEL_ARGS       --pipeline.model.num-downscales "$NUM_DOWNSCALES"
        add_arg MODEL_ARGS       --pipeline.datamanager.cache-images cpu
        add_bool_arg MODEL_ARGS  --pipeline.datamanager.images-on-gpu False
        add_bool_arg MODEL_ARGS  --pipeline.datamanager.masks-on-gpu False
        add_arg MODEL_ARGS       --pipeline.model.densify-grad-thresh "$DENSIFY_GRAD_THRESH"
        add_arg MODEL_ARGS       --pipeline.model.cull-alpha-thresh "$CULL_ALPHA_THRESH"
        add_arg MODEL_ARGS       --pipeline.model.cull-screen-size "$CULL_SCREEN_SIZE"
        add_arg MODEL_ARGS       --pipeline.model.split-screen-size "$SPLIT_SCREEN_SIZE"
        add_arg MODEL_ARGS       --pipeline.model.refine-every "$REFINE_EVERY"
        add_bool_arg MODEL_ARGS  --pipeline.model.use-bilateral-grid "$USE_BILATERAL_GRID"
        add_bool_arg MODEL_ARGS  --pipeline.model.use-scale-regularization "$USE_SCALE_REGULARIZATION"
        add_arg MODEL_ARGS       --pipeline.model.max-gauss-ratio "$MAX_GAUSS_RATIO"
        add_arg MODEL_ARGS       --pipeline.model.stop-split-at "$STOP_SPLIT_AT"
        add_arg MODEL_ARGS       --pipeline.model.cull-scale-thresh "$CULL_SCALE_THRESH"
        add_arg MODEL_ARGS       --pipeline.model.reset-alpha-every "$RESET_ALPHA_EVERY"
        add_arg MODEL_ARGS       --pipeline.model.ssim-lambda "$SSIM_LAMBDA"
        add_bool_arg MODEL_ARGS  --pipeline.model.enable-collider "$ENABLE_COLLIDER"
        add_arg MODEL_ARGS       --pipeline.model.resolution-schedule "1000"

#        add_bool_arg MODEL_ARGS  --pipeline.model.continue_cull_post_densification "False"

        if [[ "$ENABLE_COLLIDER" == "True" ]]; then

          ARGS=()

          [[ -n "${COLLIDER_NEAR:-}" ]] && ARGS+=("near_plane" "$COLLIDER_NEAR")
          [[ -n "${COLLIDER_FAR:-}"  ]] && ARGS+=("far_plane"  "$COLLIDER_FAR")

          if [[ ${#ARGS[@]} -gt 0 ]]; then
            add_multi_arg MODEL_ARGS --pipeline.model.collider-params "${ARGS[@]}"
          fi

        fi
        
        if [[ "$STOP_SPLIT_AT" -eq 0 ]]; then
            add_arg MODEL_ARGS --optimizers.means.optimizer.lr 0.0001
            add_arg MODEL_ARGS --pipeline.model.camera-optimizer.mode off
        else
            add_arg MODEL_ARGS --pipeline.model.camera-optimizer.mode "SO3xR3"
        fi

elif [[ "$DEVICE" == "cpu" ]]; then
    add_arg MODEL_ARGS --pipeline.datamanager.camera-res-scale-factor "$CAMERA_RES_SCALE_FACTOR"
    add_arg MODEL_ARGS --pipeline.model.implementation "$MODEL_IMPLEMENTATION"
    add_arg MODEL_ARGS --pipeline.model.num-nerf-samples-per-ray "$NUM_NERF_SAMPLES_PER_RAY"
    add_arg MODEL_ARGS --pipeline.model.max-res "$MAX_RES"
    add_bool_arg MODEL_ARGS --pipeline.model.predict-normals True

    if [[ -n "${NUM_PROPOSAL_SAMPLES_PER_RAY:-}" ]]; then
        add_multi_arg MODEL_ARGS --pipeline.model.num-proposal-samples-per-ray $NUM_PROPOSAL_SAMPLES_PER_RAY
    fi
fi


# ======================
# LOGGING ARGS
# ======================
LOGGING_ARGS=()

if [[ "$VERBOSE" == "True" || "$VERBOSE" == "true" || "$VERBOSE" == "1" ]]; then
  echo "🔎 VERBOSE mode enabled"

  add_arg LOGGING_ARGS --logging.local-writer.max-log-size 0

  export LOGLEVEL=DEBUG
  export PYTHONUNBUFFERED=1
  export NCCL_DEBUG=INFO
else
  export LOGLEVEL=INFO
fi



# ======================
# LOGGING
# ======================

LOG_DIR="$OUTPUTDIR/logs"

mkdir -p "$LOG_DIR"

TRAIN_LOG="$LOG_DIR/ns_train.log"

HEARTBEAT_LOG="$LOG_DIR/ns_train_heartbeat.log"

echo "📝 Full training log : $TRAIN_LOG"
echo "💓 Heartbeat log     : $HEARTBEAT_LOG"

(
  while true; do
    sleep 60
    echo "$(date '+%F %T') ns-train still active" >> "$HEARTBEAT_LOG"
  done
) &

HEARTBEAT_PID=$!

# ======================
# PIL LARGE IMAGE PATCH
# ======================

PIL_PATCH_DIR="$SCRIPT_DIR/.python_patches"
mkdir -p "$PIL_PATCH_DIR"

cat > "$PIL_PATCH_DIR/sitecustomize.py" <<'PYEOF'
from PIL import Image
import warnings

# Autorise les très grandes images
Image.MAX_IMAGE_PIXELS = None

# Optionnel : masque le warning si jamais Pillow le déclenche encore ailleurs
warnings.simplefilter("ignore", Image.DecompressionBombWarning)
PYEOF

export PYTHONPATH="$PIL_PATCH_DIR${PYTHONPATH:+:$PYTHONPATH}"

# ======================
# RUN TRAINING
# ======================

export LOGLEVEL=DEBUG

export TORCH_SHOW_CPP_STACKTRACES=1

USE_DEFAULTS=${USE_DEFAULTS:-False}

if [[ "$USE_DEFAULTS" == "True" ]]; then
  MODEL_ARGS=()
  PERF_ARGS=()
  echo "USE_DEFAULTS: $USE_DEFAULTS"
  echo "*** Reset parameters to defaults ***"
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
  > >(tee -a "$TRAIN_LOG") \
  2> >(tee -a "$TRAIN_LOG" >&2)

STATUS=$?

kill "$HEARTBEAT_PID" 2>/dev/null || true

set -e

# ======================
# CHECK RESULT
# ======================

if [[ "$STATUS" -ne 0 ]]; then

  echo "❌ ns-train crashed (exit code: $STATUS)"

  tail -50 "$TRAIN_LOG"

  exit "$STATUS"

fi

# ======================
# SILENT FAILURE DETECTION
# ======================

LAST_LOG_LINE=$(tail -n 20 "$TRAIN_LOG")

if ! echo "$LAST_LOG_LINE" | grep -q "Training Finished"; then

  echo "⚠️ ns-train may have stopped unexpectedly"

  tail -50 "$TRAIN_LOG"

fi

# ======================
# CHECKPOINT VALIDATION
# ======================

CKPT_DIR=$(find "$OUTPUTDIR" -type d -name "nerfstudio_models" 2>/dev/null | head -n 1)

if [[ -z "$CKPT_DIR" ]]; then

  echo "⚠️ Training finished but no checkpoint found"

  exit 1

fi

echo "✅ TRAINING COMPLETE"

echo "📦 Checkpoint directory: $CKPT_DIR"
