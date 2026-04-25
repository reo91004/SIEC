#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

LOG_DIR="experiments/yongseong/results/real_05_robustness/manual_logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/exp5_refresh_${STAMP}.log"
BACKUP_DIR="calibration/exp5_refresh_backup_${STAMP}"

exec > >(tee -a "$LOG_FILE") 2>&1

SETTINGS=(fp16 w8a8 dc10 w4a8 dc20 cachequant)

run() {
  echo
  echo "==> $*"
  "$@"
}

backup_if_exists() {
  local path="$1"
  if [[ -e "$path" ]]; then
    mkdir -p "$BACKUP_DIR"
    local base
    base="$(basename "$path")"
    echo "[exp5-refresh] backing up $path -> $BACKUP_DIR/$base"
    mv "$path" "$BACKUP_DIR/$base"
  fi
}

echo "[exp5-refresh] script_dir=$SCRIPT_DIR"
echo "[exp5-refresh] root_dir=$ROOT_DIR"
echo "[exp5-refresh] cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "[exp5-refresh] log_file=$LOG_FILE"

echo
echo "[0/5] Inventory before refresh"
run conda run --no-capture-output -n iec \
  python experiments/real_05_robustness.py --phase inventory

echo
echo "[1/5] Generating blocked deployment-error assets"
run conda run --no-capture-output -n iec \
  python mainddpm/ddim_cifar_cali.py --replicate_interval 20 --timesteps 100

run conda run --no-capture-output -n iec \
  python mainddpm/ddim_cifar_predadd.py --replicate_interval 20 --timesteps 100

run conda run --no-capture-output -n iec \
  python error_dec/cifar/cifar_dec.py --error cache --replicate_interval 20 --timesteps 100

run conda run --no-capture-output -n iec \
  python mainddpm/ddim_cifar_params.py --weight_bit 4 --act_bit 8 --replicate_interval 10 --timesteps 100

run conda run --no-capture-output -n iec \
  python error_dec/cifar/cifar_dec.py --error quant --weight_bit 4 --replicate_interval 10 --timesteps 100

echo
echo "[2/5] Backing up stale pilot/tau artifacts so exp5 recalibrates from scratch"
for setting in "${SETTINGS[@]}"; do
  backup_if_exists "calibration/pilot_scores_${setting}.pt"
  backup_if_exists "calibration/tau_schedule_${setting}_p80.pt"
done

echo
echo "[3/5] Inventory after asset generation"
run conda run --no-capture-output -n iec \
  python experiments/real_05_robustness.py --phase inventory

echo
echo "[4/5] Fresh exp5 run with per-setting pilot and recalibration"
run conda run --no-capture-output -n iec \
  python experiments/real_05_robustness.py --phase all --no-dry-run

echo
echo "[5/5] Latest results"
ls -td experiments/yongseong/results/real_05_robustness/* | head -n 3

if [[ -d "$BACKUP_DIR" ]]; then
  echo "[exp5-refresh] stale calibration backup saved at $BACKUP_DIR"
fi
