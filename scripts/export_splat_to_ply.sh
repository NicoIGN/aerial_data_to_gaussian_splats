#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


# ======================
# CHECKS
# ======================
if [[ -z "$EXPORT_DIR" || -z "$OUTPUT_DIR" ]]; then
  echo "❌ Missing required environment variables:"
  echo "   EXPORT_DIR=$EXPORT_DIR"
  echo "   OUTPUT_DIR=$OUTPUT_DIR"
  exit 1
fi

if [[ ! -d "$EXPORT_DIR" ]]; then
  echo "❌ Export directory does not exist: $EXPORT_DIR"
  exit 1
fi

if [[ ! -d "$OUTPUT_DIR" ]]; then
  echo "❌ Output directory does not exist: $OUTPUT_DIR"
  exit 1
fi


SPLAT_ROOT="$OUTPUT_DIR/splatfacto"

if [ ! -d "$SPLAT_ROOT" ]; then
  echo "❌ splatfacto folder not found: $SPLAT_ROOT"
  exit 1
fi

# ======================
# FIND LATEST RUN
# ======================
RUN_DIR=$(ls -dt "$SPLAT_ROOT"/* 2>/dev/null | head -n 1)

if [ -z "$RUN_DIR" ]; then
  echo "❌ No splatfacto runs found in $SPLAT_ROOT"
  exit 1
fi

RUN_NAME=$(basename "$RUN_DIR")

# ======================
# CONFIG
# ======================
CONFIG="$RUN_DIR/config.yml"

if [ ! -f "$CONFIG" ]; then
  echo "❌ config.yml not found:"
  echo "$RUN_DIR"
  exit 1
fi

mkdir -p "$EXPORT_DIR"

# ======================
# ZIP RUN DIRECTORY
# ======================
ZIP_PATH="$EXPORT_DIR/${RUN_NAME}.zip"

if [ -f "$ZIP_PATH" ]; then
  echo "⏭️ Zip already exists, skipping: $ZIP_PATH"
else
  echo "📦 Zipping run directory directly to export: $RUN_DIR"

  (
    cd "$(dirname "$RUN_DIR")" && \
    zip -r "$ZIP_PATH" "$RUN_NAME" > /dev/null
  )

  if [ -f "$ZIP_PATH" ]; then
    echo "✅ Archive created: $ZIP_PATH"
  else
    echo "❌ Failed to create zip archive in export dir"
    exit 1
  fi
fi

echo "✅ Archive created: $ZIP_PATH"

echo "────────────────────────────────────────────"
echo "📦 SPLAT ROOT      : $SPLAT_ROOT"
echo "📦 RUN DIR        : $RUN_DIR"
echo "📄 CONFIG         : $CONFIG"
echo "📁 EXPORT DIR     : $EXPORT_DIR"
echo "────────────────────────────────────────────"

# ======================
# ENV FLAGS
# ======================
# export TORCHDYNAMO_DISABLE=1
# export OMP_NUM_THREADS=1

# ======================
# DEBUG MODEL
# ======================
echo "📊 Checking checkpoint..."
ls "$RUN_DIR/nerfstudio_models" || echo "⚠️ No models folder"

# ======================
# EXPORT PLY
# ======================
echo "🚀 Exporting Gaussian Splat (.ply)..."

ns-export gaussian-splat \
  --load-config "$CONFIG" \
  --output-dir "$EXPORT_DIR"

PLY_FILE=$(find "$EXPORT_DIR" -name "*.ply" | head -n 1)

if [ -z "$PLY_FILE" ]; then
  echo "❌ Gaussian PLY export failed"
  exit 1
fi

echo "✅ PLY exported:"
echo "   $PLY_FILE"

