#!/bin/bash
set -e

# ======================
# TIMING UTILS
# ======================

SCRIPT_START=$(date +%s)

format_duration() {
  local seconds=$1

  local h=$((seconds / 3600))
  local m=$(((seconds % 3600) / 60))
  local s=$((seconds % 60))

  if [ $h -gt 0 ]; then
    printf "%02dh %02dm %02ds" "$h" "$m" "$s"
  elif [ $m -gt 0 ]; then
    printf "%02dm %02ds" "$m" "$s"
  else
    printf "%02ds" "$s"
  fi
}

print_step_time() {
  local label="$1"
  local start_ts="$2"

  local end_ts=$(date +%s)
  local elapsed=$((end_ts - start_ts))

  echo ""
  echo "⏱️  ${label} completed in $(format_duration "$elapsed")"
  echo ""
}

# ======================
# DEFAULTS
# ======================
DEVICE="cpu"
ROOT_DIR="runs/default"
SKIP_CONDA=false
IGNORE_PROXY=false
MAX_JOBS=2
SKIP_TRAINING=false
SKIP_EXPORT=false
TWO_STAGES=false


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GSPLAT_PROFILE=""
COLMAP_DIR=""
IMAGE_DIR=""

# ======================
# HELP
# ======================
show_help() {
  cat << EOF
Usage:
  ./run.sh --colmap-dir <dir> --image-dir <dir> --root <dir> [options]

Required:
  --colmap-dir <dir>         Existing COLMAP directory containing sparse/0/...
  --image-dir <dir>          Directory containing source images used by transforms.json

Options:
  --root <dir>               Root output directory (default: runs/default)
  --name <name>              Base name of outputs (default: gsplat_<timestamp>)
  --gsplat-profile <name>    fast | balanced | quality | quality_plus
  --max-jobs <int>           Number of parallelizable jobs
  --two-stages              Use two-stage training (coarse -> full resolution)

  --skip-conda               Skip conda environment setup
  --no-proxy                 Disable proxy configuration
  --skip-training
  --skip-export

Example:
  ./run.sh \
    --colmap-dir /path/to/ori/colmap \
    --image-dir /path/to/ori/images \
    --root runs/statue \
    --gsplat-profile quality
EOF
}

# ======================
# ARG PARSING
# ======================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --colmap-dir) COLMAP_DIR="$2"; shift 2 ;;
    --image-dir) IMAGE_DIR="$2"; shift 2 ;;
    --gsplat-profile) GSPLAT_PROFILE="$2"; shift 2 ;;
    --max-jobs) MAX_JOBS="$2"; shift 2 ;;
    --name) BASENAME="$2"; shift 2 ;;
    --root) ROOT_DIR="$2"; shift 2 ;;
    --skip-conda) SKIP_CONDA=true; shift ;;
    --no-proxy) IGNORE_PROXY=true; shift ;;
    --skip-training) SKIP_TRAINING=true; shift ;;
    --skip-export) SKIP_EXPORT=true; shift ;;
    --two-stages) TWO_STAGES=true; shift ;;
    --help) show_help; exit 0 ;;
    *) echo "❌ Unknown param: $1"; show_help; exit 1 ;;
  esac
done

NO_PROXY="$IGNORE_PROXY" \
MAX_JOBS="$MAX_JOBS" \
SKIP_TRAINING="$SKIP_TRAINING" \
source "$SCRIPT_DIR/config/config.sh"

# ======================
# SETUP VALIDATION
# ======================
BASENAME=${BASENAME:-gsplat_$(date +%Y%m%d_%H%M%S)}

if [ -z "$COLMAP_DIR" ]; then
  echo "❌ --colmap-dir is required"
  show_help
  exit 1
fi

if [ -z "$IMAGE_DIR" ]; then
  echo "❌ --image-dir is required"
  show_help
  exit 1
fi

if [ ! -d "$COLMAP_DIR" ]; then
  echo "❌ COLMAP directory not found: $COLMAP_DIR"
  exit 1
fi

if [ ! -d "$IMAGE_DIR" ]; then
  echo "❌ Image directory not found: $IMAGE_DIR"
  exit 1
fi

if [ ! -d "$COLMAP_DIR/colmap/sparse/0" ]; then
  echo "❌ Missing COLMAP sparse model: $COLMAP_DIR/colmap/sparse/0"
  exit 1
fi

IMAGE_COUNT=$(find "$IMAGE_DIR" -maxdepth 1 \
  \( -type f -o -xtype f \) \
  \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) \
  | wc -l | tr -d ' ')

if [ "$IMAGE_COUNT" -lt 2 ]; then
  echo "❌ At least 2 images are required in image-dir $IMAGE_DIR"
  echo "   Found: $IMAGE_COUNT"
  exit 1
fi

if [ -z "$GSPLAT_PROFILE" ]; then
  echo "⚠️  no gsplat profile loaded"
else
  if [ -f "config/profiles/${GSPLAT_PROFILE}.sh" ]; then
    source "config/profiles/${GSPLAT_PROFILE}.sh"
    echo "👉 using profile: ${GSPLAT_PROFILE}"
  else
    echo "❌ profile ${GSPLAT_PROFILE} not found"
    echo "❌ use profile fast|quality|balanced|best"
    exit 1
  fi
fi

# ======================
# CONDA ENVIRONMENT
# ======================
if [ "$SKIP_CONDA" = true ]; then
    echo "⏩ Skipping conda setup (--skip-conda enabled)"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"

    if [ "$IGNORE_PROXY" != true ]; then
      echo "IGNORE_PROXY: $IGNORE_PROXY"

      if [ -n "$HTTP_PROXY" ]; then
        export HTTP_PROXY="$HTTP_PROXY"
        export http_proxy="$HTTP_PROXY"
        echo "🌐 HTTP proxy enabled"
      fi

      if [ -n "$HTTPS_PROXY" ]; then
        export HTTPS_PROXY="$HTTPS_PROXY"
        export https_proxy="$HTTPS_PROXY"
        echo "🌐 HTTPS proxy enabled"
      fi

      conda config --set proxy_servers.http "$HTTP_PROXY" 2>/dev/null || true
      conda config --set proxy_servers.https "$HTTPS_PROXY" 2>/dev/null || true
    else
      echo "🚫 Proxy disabled via IGNORE_PROXY=true"
      conda config --remove-key proxy_servers.http 2>/dev/null || true
      conda config --remove-key proxy_servers.https 2>/dev/null || true
    fi

    set +e

    if conda env list | awk '{print $1}' | grep -qw "$CONDA_ENV_NAME"; then
        echo "🔁 Updating env: $CONDA_ENV_NAME"
        CONDA_CMD="conda env update -n $CONDA_ENV_NAME -f $CONDA_ENV_FILE --prune"
    else
        echo "🆕 Creating env: $CONDA_ENV_NAME"
        conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
        conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
        CONDA_CMD="conda env create -n $CONDA_ENV_NAME -f $CONDA_ENV_FILE"
    fi

    echo "⚙️ Running: $CONDA_CMD"

    $CONDA_CMD
    STATUS=$?

    if [ $STATUS -ne 0 ]; then
        echo "❌ Conda failed ($CONDA_CMD)"
        echo "👉 Run manually for debug"
        exit 1
    fi

    set -e

    echo "🔌 Activating env: $CONDA_ENV_NAME"
    conda activate "$CONDA_ENV_NAME"

    bash scripts/check_torch_stack.sh
fi

# ======================
# PYTHON VERSION CHECK
# ======================
PY_VER=$(python --version 2>&1)

if [[ "$PY_VER" != *"3.10"* && "$PY_VER" != *"3.11"* ]]; then
  echo "❌ ERROR: Python 3.10 or 3.11 required but found: $PY_VER"
  echo "👉 Supported versions: 3.10.x, 3.11.x"
  echo "👉 Aborting execution"
  exit 1
fi

echo "✅ Using python: $PY_VER"

# ======================
# MODEL VALIDATION
# ======================
CUDA_AVAILABLE=false
if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi >/dev/null 2>&1; then
    CUDA_AVAILABLE=true
  fi
fi

if [ "$DEVICE" == "gpu" ] && [ "$CUDA_AVAILABLE" = false ]; then
  echo "❌ ERROR: GPU requested but CUDA is not available."
  exit 1
fi

if [ "$DEVICE" == "cpu" ]; then
  case "$MODEL" in
    nerfacto|nerf|kplanes|tensorf)
      echo "🧠 CPU model OK: $MODEL"
      ;;
    *)
      echo "❌ MODEL '$MODEL' not supported on CPU"
      exit 1
      ;;
  esac
fi

if [ "$DEVICE" == "gpu" ]; then
  case "$MODEL" in
    splatfacto|splatfacto-big|splatfacto-w|instant-ngp|zip-nerf|pynerf|feature-splatting)
      echo "🚀 GPU model OK: $MODEL"
      ;;
    *)
      echo "⚠️ MODEL '$MODEL' is not GPU-optimized (will run but may be slow)"
      ;;
  esac
fi

# ======================
# DATA STRUCTURE
# ======================

OUTPUT_DIR="$ROOT_DIR/model3d"
EXPORT_DIR="$ROOT_DIR/exports"
TRAIN_DIR="$ROOT_DIR"

mkdir -p "$OUTPUT_DIR" "$EXPORT_DIR" "$TRAIN_DIR"

echo "📦 ROOT: $ROOT_DIR"
echo "📦 COLMAP INPUT: $COLMAP_DIR"
echo "📦 IMAGE INPUT:  $IMAGE_DIR"

# ======================
# PREPARE WORKING ORI DIR
# ======================
echo ""

if [ ! -f "$COLMAP_DIR/transforms.json" ]; then
  echo "❌ Missing transforms.json in $COLMAP_DIR"
  echo "👉 This training pipeline expects $COLMAP_DIR/transforms.json to exist."
  echo "👉 Copy or generate it before running."
  exit 1
fi

# ----------------------
# TRAIN
# ----------------------
if [ "$SKIP_TRAINING" = true ]; then
  echo "⏩ Skipping training (config)"
else
    COLMAP_SPARSE_DIR="$COLMAP_DIR/colmap/sparse/0"

    if [ ! -d "$COLMAP_SPARSE_DIR" ]; then
      echo "❌ COLMAP directory not found: $COLMAP_SPARSE_DIR"
      exit 1
    fi

    echo ""
    echo "📏 Estimating near/far planes from COLMAP..."

    ESTIMATE_SCRIPT="$SCRIPT_DIR/python/estimate_planes.py"

    if [ ! -f "$ESTIMATE_SCRIPT" ]; then
      echo "❌ Missing script: $ESTIMATE_SCRIPT"
      exit 1
    fi

    EST_OUTPUT=$(python3 "$ESTIMATE_SCRIPT" --input "$COLMAP_SPARSE_DIR")

    NEAR=$(echo "$EST_OUTPUT" | grep NEAR | cut -d= -f2)
    FAR=$(echo "$EST_OUTPUT" | grep FAR  | cut -d= -f2)

    if [[ -z "$NEAR" || -z "$FAR" || "$NEAR" == "nan" || "$FAR" == "nan" ]]; then
      echo "❌ Invalid near/far values"
      echo "$EST_OUTPUT"
      exit 1
    fi

    echo "✅ Estimated:"
    echo "   near = $NEAR"
    echo "   far  = $FAR"

    export COLLIDER_NEAR="$NEAR"
    export COLLIDER_FAR="$FAR"
    export ENABLE_COLLIDER="True"

    LATEST_RUN=$(ls -td "$OUTPUT_DIR"/ori/$MODEL/* 2>/dev/null | head -n 1 || true)

    if [ -n "$LATEST_RUN" ] && [ -d "$LATEST_RUN/nerfstudio_models" ]; then
        echo "⏩ Skipping training"
    else
        echo ""
        echo "🧠 Training..."

        STEP_START=$(date +%s)

        if [ "$TWO_STAGES" = true ]; then
          echo "👉 two-stage mode enabled (coarse -> full)"

          # Snapshot des valeurs profil d'origine
          ORIG_MAX_ITER="$MAX_ITER"
          ORIG_CAMERA_RES_SCALE_FACTOR="$CAMERA_RES_SCALE_FACTOR"
          ORIG_NUM_DOWNSCALES="${NUM_DOWNSCALES:-}"
          ORIG_REFINE_EVERY="$REFINE_EVERY"
          ORIG_STOP_SPLIT_AT="$STOP_SPLIT_AT"
          ORIG_RELOAD_FROM_CHECKPOINT="${RELOAD_FROM_CHECKPOINT:-False}"

          # -----------------------------
          # Stage 1: COARSE
          # Profil inchangé + résolution capée
          # -----------------------------
          CAMERA_RES_SCALE_FACTOR="0.5"
          NUM_DOWNSCALES="2"
          RELOAD_FROM_CHECKPOINT="False"

          echo "🧩 Stage coarse: SCALE=$CAMERA_RES_SCALE_FACTOR DOWNSCALE=$NUM_DOWNSCALES"

          HTTP_PROXY="$HTTP_PROXY" \
          HTTPS_PROXY="$HTTPS_PROXY" \
          NO_PROXY="$NO_PROXY" \
          VERBOSE="True" \
          RELOAD_FROM_CHECKPOINT="False" \
          MODEL="$MODEL" \
          MODEL_IMPLEMENTATION="$MODEL_IMPLEMENTATION" \
          DEVICE="$DEVICE" \
          MAX_ITER="$MAX_ITER" \
          REFINE_EVERY="$REFINE_EVERY" \
          MAX_JOBS="$MAX_JOBS" \
          STEPS_PER_SAVE="$STEPS_PER_SAVE" \
          STEPS_PER_EVAL_ALL_IMAGES="$STEPS_PER_EVAL_ALL_IMAGES" \
          DATA="$COLMAP_DIR" \
          EXPERIMENT_NAME="$EXPERIMENT_NAME" \
          OUTPUTDIR="$TRAIN_DIR" \
          TRAIN_RAYS_PER_BATCH="$TRAIN_RAYS_PER_BATCH" \
          CAMERA_RES_SCALE_FACTOR="$CAMERA_RES_SCALE_FACTOR" \
          NUM_DOWNSCALES="$NUM_DOWNSCALES" \
          NUM_NERF_SAMPLES_PER_RAY="$NUM_NERF_SAMPLES_PER_RAY" \
          NUM_PROPOSAL_SAMPLES_PER_RAY="$NUM_PROPOSAL_SAMPLES_PER_RAY" \
          MAX_RES="$MAX_RES" \
          MAX_GAUSS_RATIO="$MAX_GAUSS_RATIO" \
          DENSIFY_GRAD_THRESH="$DENSIFY_GRAD_THRESH" \
          CULL_ALPHA_THRESH="$CULL_ALPHA_THRESH" \
          CULL_SCREEN_SIZE="$CULL_SCREEN_SIZE" \
          SPLIT_SCREEN_SIZE="$SPLIT_SCREEN_SIZE" \
          STOP_SPLIT_AT="$STOP_SPLIT_AT" \
          CULL_SCALE_THRESH="$CULL_SCALE_THRESH" \
          RESET_ALPHA_EVERY="$RESET_ALPHA_EVERY" \
          USE_SCALE_REGULARIZATION="$USE_SCALE_REGULARIZATION" \
          SSIM_LAMBDA="$SSIM_LAMBDA" \
          MAX_GAUSSIANS="$MAX_GAUSSIANS" \
          COLLIDER_NEAR="$COLLIDER_NEAR" \
          COLLIDER_FAR="$COLLIDER_FAR" \
          ENABLE_COLLIDER="$ENABLE_COLLIDER" \
          USE_BILATERAL_GRID="$USE_BILATERAL_GRID" \
          USE_DEFAULTS="False" \
          MIXED_PRECISION="$MIXED_PRECISION" \
          USE_GRAD_SCALER="$USE_GRAD_SCALER" \
          bash scripts/train.sh

          # -----------------------------
          # Stage 2: FULL RES (1000 iters, no refine/split, resume)
          # -----------------------------
          MAX_ITER=$((ORIG_MAX_ITER + 1000))
          CAMERA_RES_SCALE_FACTOR="$ORIG_CAMERA_RES_SCALE_FACTOR"
          NUM_DOWNSCALES="$ORIG_NUM_DOWNSCALES"
          REFINE_EVERY="999999"
          STOP_SPLIT_AT="0"
          RELOAD_FROM_CHECKPOINT="True"

          echo "🧩 Stage full: SCALE=$CAMERA_RES_SCALE_FACTOR DOWNSCALE=${NUM_DOWNSCALES:-<unset>} MAX_ITER=$MAX_ITER REFINE_EVERY=$REFINE_EVERY STOP_SPLIT_AT=$STOP_SPLIT_AT (resume)"

          HTTP_PROXY="$HTTP_PROXY" \
          HTTPS_PROXY="$HTTPS_PROXY" \
          NO_PROXY="$NO_PROXY" \
          VERBOSE="True" \
          RELOAD_FROM_CHECKPOINT="True" \
          MODEL="$MODEL" \
          MODEL_IMPLEMENTATION="$MODEL_IMPLEMENTATION" \
          DEVICE="$DEVICE" \
          MAX_ITER="$MAX_ITER" \
          REFINE_EVERY="$REFINE_EVERY" \
          MAX_JOBS="$MAX_JOBS" \
          STEPS_PER_SAVE="$STEPS_PER_SAVE" \
          STEPS_PER_EVAL_ALL_IMAGES="$STEPS_PER_EVAL_ALL_IMAGES" \
          DATA="$COLMAP_DIR" \
          EXPERIMENT_NAME="$EXPERIMENT_NAME" \
          OUTPUTDIR="$TRAIN_DIR" \
          TRAIN_RAYS_PER_BATCH="$TRAIN_RAYS_PER_BATCH" \
          CAMERA_RES_SCALE_FACTOR="$CAMERA_RES_SCALE_FACTOR" \
          NUM_DOWNSCALES="$NUM_DOWNSCALES" \
          NUM_NERF_SAMPLES_PER_RAY="$NUM_NERF_SAMPLES_PER_RAY" \
          NUM_PROPOSAL_SAMPLES_PER_RAY="$NUM_PROPOSAL_SAMPLES_PER_RAY" \
          MAX_RES="$MAX_RES" \
          MAX_GAUSS_RATIO="$MAX_GAUSS_RATIO" \
          DENSIFY_GRAD_THRESH="$DENSIFY_GRAD_THRESH" \
          CULL_ALPHA_THRESH="$CULL_ALPHA_THRESH" \
          CULL_SCREEN_SIZE="$CULL_SCREEN_SIZE" \
          SPLIT_SCREEN_SIZE="$SPLIT_SCREEN_SIZE" \
          STOP_SPLIT_AT="$STOP_SPLIT_AT" \
          CULL_SCALE_THRESH="$CULL_SCALE_THRESH" \
          RESET_ALPHA_EVERY="$RESET_ALPHA_EVERY" \
          USE_SCALE_REGULARIZATION="$USE_SCALE_REGULARIZATION" \
          SSIM_LAMBDA="$SSIM_LAMBDA" \
          MAX_GAUSSIANS="$MAX_GAUSSIANS" \
          COLLIDER_NEAR="$COLLIDER_NEAR" \
          COLLIDER_FAR="$COLLIDER_FAR" \
          ENABLE_COLLIDER="$ENABLE_COLLIDER" \
          USE_BILATERAL_GRID="$USE_BILATERAL_GRID" \
          USE_DEFAULTS="False" \
          MIXED_PRECISION="$MIXED_PRECISION" \
          USE_GRAD_SCALER="$USE_GRAD_SCALER" \
          bash scripts/train.sh

          # restore (optionnel)
          MAX_ITER="$ORIG_MAX_ITER"
          CAMERA_RES_SCALE_FACTOR="$ORIG_CAMERA_RES_SCALE_FACTOR"
          NUM_DOWNSCALES="$ORIG_NUM_DOWNSCALES"
          REFINE_EVERY="$ORIG_REFINE_EVERY"
          STOP_SPLIT_AT="$ORIG_STOP_SPLIT_AT"
          RELOAD_FROM_CHECKPOINT="$ORIG_RELOAD_FROM_CHECKPOINT"

        else
          HTTP_PROXY="$HTTP_PROXY" \
          HTTPS_PROXY="$HTTPS_PROXY" \
          NO_PROXY="$NO_PROXY" \
          VERBOSE="True" \
          RELOAD_FROM_CHECKPOINT="False" \
          MODEL="$MODEL" \
          MODEL_IMPLEMENTATION="$MODEL_IMPLEMENTATION" \
          DEVICE="$DEVICE" \
          MAX_ITER="$MAX_ITER" \
          REFINE_EVERY="$REFINE_EVERY" \
          MAX_JOBS="$MAX_JOBS" \
          STEPS_PER_SAVE="$STEPS_PER_SAVE" \
          STEPS_PER_EVAL_ALL_IMAGES="$STEPS_PER_EVAL_ALL_IMAGES" \
          DATA="$COLMAP_DIR" \
          EXPERIMENT_NAME="$EXPERIMENT_NAME" \
          OUTPUTDIR="$TRAIN_DIR" \
          TRAIN_RAYS_PER_BATCH="$TRAIN_RAYS_PER_BATCH" \
          CAMERA_RES_SCALE_FACTOR="$CAMERA_RES_SCALE_FACTOR" \
          NUM_NERF_SAMPLES_PER_RAY="$NUM_NERF_SAMPLES_PER_RAY" \
          NUM_PROPOSAL_SAMPLES_PER_RAY="$NUM_PROPOSAL_SAMPLES_PER_RAY" \
          MAX_RES="$MAX_RES" \
          MAX_GAUSS_RATIO="$MAX_GAUSS_RATIO" \
          DENSIFY_GRAD_THRESH="$DENSIFY_GRAD_THRESH" \
          CULL_ALPHA_THRESH="$CULL_ALPHA_THRESH" \
          CULL_SCREEN_SIZE="$CULL_SCREEN_SIZE" \
          SPLIT_SCREEN_SIZE="$SPLIT_SCREEN_SIZE" \
          STOP_SPLIT_AT="$STOP_SPLIT_AT" \
          CULL_SCALE_THRESH="$CULL_SCALE_THRESH" \
          RESET_ALPHA_EVERY="$RESET_ALPHA_EVERY" \
          USE_SCALE_REGULARIZATION="$USE_SCALE_REGULARIZATION" \
          SSIM_LAMBDA="$SSIM_LAMBDA" \
          MAX_GAUSSIANS="$MAX_GAUSSIANS" \
          COLLIDER_NEAR="$COLLIDER_NEAR" \
          COLLIDER_FAR="$COLLIDER_FAR" \
          ENABLE_COLLIDER="$ENABLE_COLLIDER" \
          USE_BILATERAL_GRID="$USE_BILATERAL_GRID" \
          USE_DEFAULTS="False" \
          MIXED_PRECISION="$MIXED_PRECISION" \
          USE_GRAD_SCALER="$USE_GRAD_SCALER" \
          bash scripts/train.sh
        fi

        print_step_time "TRAINING" "$STEP_START"
    fi
fi

# ----------------------
# EXPORT
# ----------------------
PLY_FILE=""

if [[ "$SKIP_EXPORT" == "true" ]]; then
  echo "⏩ Skipping export (config)"
else
  PLY_FOUND=$(find "$OUTPUT_DIR" -type f -name "*.ply" | head -n 1 || true)

  if [[ -n "$PLY_FOUND" ]]; then
    echo "📦 Existing PLY found: $PLY_FOUND, backing it up into ${PLY_FOUND}.bkp"
    cp "$PLY_FOUND" "${PLY_FOUND}.bkp"
  fi

  echo "🚀 Exporting model in $EXPORT_DIR..."
  STEP_START=$(date +%s)

  OUTPUT_DIR="$TRAIN_DIR/$EXPERIMENT_NAME" \
  EXPORT_DIR="$OUTPUT_DIR" \
  ZIP_RUN=0 \
  bash scripts/export_splat_to_ply.sh

  PLY_FILE=$(find "$OUTPUT_DIR" -type f -name "*.ply" | head -n 1 || true)

  if [[ -f "$PLY_FILE" ]]; then
    echo "✅ PLY export successful: $PLY_FILE"
    print_step_time "EXPORT" "$STEP_START"
  else
    echo "❌ PLY export failed"
    exit 1
  fi

  LATEST_ZIP=$(ls -t "$OUTPUT_DIR"/*.zip 2>/dev/null | head -n 1 || true)

  if [ -z "$LATEST_ZIP" ]; then
    echo "⚠️ No .zip file found in $OUTPUT_DIR"
  else
    mv "$LATEST_ZIP" "$EXPORT_DIR/" && \
    if [ -f "$EXPORT_DIR/$(basename "$LATEST_ZIP")" ]; then
      echo "📦 Moved $(basename "$LATEST_ZIP") to $EXPORT_DIR"
    else
      echo "❌ Failed to move $(basename "$LATEST_ZIP")" >&2
    fi
  fi
fi

# ======================
# CLEAN PLY
# ======================
echo "🧹 Removing filtered Gaussian Splat files..."

FILES_TO_DELETE=($(find "$EXPORT_DIR" -type f -name "${BASENAME}_*.ply" | sort))

if [[ ${#FILES_TO_DELETE[@]} -eq 0 ]]; then
    echo "⚠️ No filtered files to remove in $EXPORT_DIR for basename: $BASENAME"
else
    echo "📦 Found ${#FILES_TO_DELETE[@]} file(s) to delete"

    for FILE in "${FILES_TO_DELETE[@]}"; do
        echo "🗑️ Deleting: $(basename "$FILE")"
        rm -f "$FILE"
    done

    echo "✅ Cleanup complete"
fi

if [[ -z "$PLY_FILE" ]]; then
  PLY_FILE=$(find "$OUTPUT_DIR" -type f -name "*.ply" | head -n 1 || true)
fi

echo "📦 Source PLY: $PLY_FILE"

STEP_START=$(date +%s)

# ======================
# RUN RESOLUTION
# ======================
if [[ "$MODEL" == *splat* ]]; then
  MODEL_DIR="splatfacto"
else
  MODEL_DIR="nerfacto"
fi

RUN_ROOT="$OUTPUT_DIR/$MODEL_DIR"
LATEST_RUN=$(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)

if [ -z "$LATEST_RUN" ] || [ ! -d "$LATEST_RUN" ]; then
  echo "[ERROR] No run found in $RUN_ROOT"
  exit 1
fi

TRANSFORM_FILE="$LATEST_RUN/dataparser_transforms.json"

if [ ! -f "$TRANSFORM_FILE" ]; then
  echo "[ERROR] Transform file not found: $TRANSFORM_FILE"
  exit 1
fi

# ======================
# INPUT VALIDATION
# ======================
if [[ ! -f "$PLY_FILE" ]]; then
  echo "❌ PLY not found: $PLY_FILE"
  exit 1
fi

COLMAP_POINTS="$COLMAP_DIR/colmap/sparse/0/points3D.bin"

if [[ ! -f "$COLMAP_POINTS" ]]; then
  echo "❌ COLMAP points not found: $COLMAP_POINTS"
  exit 1
fi

# ======================
# CLEANING STEP
# ======================
python python/clean-ply.py \
  --in-ply "$PLY_FILE" \
  --points "$COLMAP_POINTS" \
  --out-ply "$EXPORT_DIR/${BASENAME}.ply" \
  --transform "$TRANSFORM_FILE"

print_step_time "CLEAN PLY" "$STEP_START"
