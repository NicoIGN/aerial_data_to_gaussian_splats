#!/bin/bash
set -euo pipefail

VERBOSE="${VERBOSE:-false}"

# Usage:
# GIT_ROOT=/path/to/video_to_ply ./submit.sh

: "${GIT_ROOT:?❌ GIT_ROOT is not set. Example: GIT_ROOT=/path/to/video_to_ply ./submit.sh}"

LAUNCH_SLURM="$GIT_ROOT/environment/ign.slurm/launch_slurm.sh"
RUN_SH="$GIT_ROOT/run.sh"
CONFIG_SH="./config.sh"

LOG_DIR="/mnt/common/hdd/slurm/logs"
SUBMIT_LOG="$LOG_DIR/submit.log"

mkdir -p "$LOG_DIR"

# Log terminal + fichier
exec > >(tee -a "$SUBMIT_LOG") 2>&1

log() {
  echo "$@"
}

verbose_log() {
  [ "$VERBOSE" = "true" ] && echo "$@"
}

log "========================"
log "🚀 PRE-SUBMISSION CHECK"
log "========================"
log "date         : $(date)"
log "host         : $(hostname)"
log "user         : $(whoami)"
log "pwd          : $(pwd)"
log "GIT_ROOT     : $GIT_ROOT"
log "launch.slurm : $LAUNCH_SLURM"
log "run.sh       : $RUN_SH"
log "config.sh    : $CONFIG_SH"
log "log dir      : $LOG_DIR"
log "submit log   : $SUBMIT_LOG"
log "verbose      : $VERBOSE"

[ -d "$GIT_ROOT" ] || { log "❌ GIT_ROOT not found: $GIT_ROOT"; exit 1; }
[ -f "$LAUNCH_SLURM" ] || { log "❌ launch.slurm missing: $LAUNCH_SLURM"; exit 1; }
[ -f "$RUN_SH" ] || { log "❌ run.sh missing: $RUN_SH"; exit 1; }
[ -f "$CONFIG_SH" ] || { log "❌ config.sh missing: $CONFIG_SH"; exit 1; }

log
log "========================"
log "📤 SUBMITTING JOB"
log "========================"

OUT=$(sbatch "$LAUNCH_SLURM")
log "$OUT"

JOB_ID=$(echo "$OUT" | awk '{print $4}')

if [ -z "${JOB_ID:-}" ]; then
  log "❌ Could not parse job ID from sbatch output"
  exit 1
fi

STDOUT_LOG="$LOG_DIR/gsplat-$JOB_ID.out"
STDERR_LOG="$LOG_DIR/gsplat-$JOB_ID.err"

log "✅ Submitted job $JOB_ID"
log "📄 stdout: $STDOUT_LOG"
log "📄 stderr: $STDERR_LOG"
log "🔎 queue: squeue -j $JOB_ID"

log
log "========================"
log "⏳ WAITING FOR LOG FILES"
log "========================"

for _ in $(seq 1 60); do
  if [ -f "$STDOUT_LOG" ] || [ -f "$STDERR_LOG" ]; then
    break
  fi
  sleep 1
done

touch "$STDOUT_LOG" "$STDERR_LOG"

log
log "========================"
log "📡 STREAMING JOB LOGS"
log "========================"

if [ "$VERBOSE" = "true" ]; then
  log "Verbose mode enabled → full logs"
  tail -n +1 -f "$STDOUT_LOG" "$STDERR_LOG" &
else
  log "Verbose mode disabled → filtered logs"

  tail -n +1 -f "$STDOUT_LOG" "$STDERR_LOG" 2>/dev/null \
    | grep -vE \
'RESOURCE SNAPSHOT|memory\.total|memory\.used|memory\.free|utilization\.gpu|used_gpu_memory|^Mem:|^Swap:|^pid, process_name|^index, name|^\[INFO\]|^\[DEBUG\]|^INFO:|^DEBUG:|it/s|step=|epoch=|loss=|Loading|Saving|Caching|Downloading|Analyzing|Processing|Rendering|Iteration|Checkpoint|Progress|^==> .* <==$' \
    || true &
fi

TAIL_PID=$!

cleanup() {
  kill "$TAIL_PID" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

# Attendre la fin du job Slurm
while squeue -j "$JOB_ID" -h | grep -q .; do
  sleep 2
done

kill "$TAIL_PID" 2>/dev/null || true
wait "$TAIL_PID" 2>/dev/null || true

log
log "========================"
log "🏁 JOB FINISHED"
log "========================"

log "--- Last lines of stdout ---"
tail -n 30 "$STDOUT_LOG" || true

log
log "--- Last lines of stderr ---"
tail -n 30 "$STDERR_LOG" || true

log

FINAL_STATE=$(
  sacct -j "$JOB_ID" --format=State --noheader 2>/dev/null \
    | awk 'NF {print $1; exit}'
)

EXIT_CODE=$(
  sacct -j "$JOB_ID" --format=ExitCode --noheader 2>/dev/null \
    | awk 'NF {print $1; exit}'
)

log "Final state: ${FINAL_STATE:-unknown}"
log "Exit code  : ${EXIT_CODE:-unknown}"

case "${FINAL_STATE:-}" in
  COMPLETED)
    exit 0
    ;;
  *)
    exit 1
    ;;
esac
