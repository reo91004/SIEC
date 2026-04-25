#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

LOG_DIR="experiments/yongseong/results/real_04_tradeoff/manual_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/exp4_refresh_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG_FILE") 2>&1

PERCENTILES=(30 50 60 70 80 90 95)

run() {
  echo
  echo "==> $*"
  "$@"
}

echo "[exp4-refresh] script_dir=$SCRIPT_DIR"
echo "[exp4-refresh] root_dir=$ROOT_DIR"
echo "[exp4-refresh] cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "[exp4-refresh] log_file=$LOG_FILE"

if pgrep -af "ddim_cifar_siec.py.*pilot_scores_nb.pt" >/dev/null; then
  echo "[exp4-refresh] existing pilot process detected:"
  pgrep -af "ddim_cifar_siec.py.*pilot_scores_nb.pt"
  echo "[exp4-refresh] stop the existing pilot first, then rerun this script."
  exit 1
fi

echo
echo "[1/3] Regenerating S-IEC pilot scores"
run conda run --no-capture-output -n iec \
  python experiments/yongseong/ddim_cifar_siec.py \
  --correction-mode siec \
  --num_samples 512 \
  --sample_batch 500 \
  --weight_bit 8 \
  --act_bit 8 \
  --replicate_interval 10 \
  --image_folder error_dec/cifar/image_tradeoff_pilot_nb_n512 \
  --siec_collect_scores \
  --siec_scores_out calibration/pilot_scores_nb.pt

echo
echo "[2/3] Regenerating tau schedules"
for percentile in "${PERCENTILES[@]}"; do
  run conda run --no-capture-output -n iec \
    python mainddpm/calibrate_tau_cifar.py \
    --scores_path ./calibration/pilot_scores_nb.pt \
    --percentile "$percentile" \
    --out_path "./calibration/tau_schedule_p${percentile}.pt"
done

echo
echo "[3/3] Running exp4 tradeoff wrapper"
run conda run --no-capture-output -n iec \
  python experiments/real_04_tradeoff.py --no-dry-run

echo
echo "[exp4-refresh] completed successfully"
echo "[exp4-refresh] latest results:"
ls -td experiments/yongseong/results/real_04_tradeoff/* | head -n 3
