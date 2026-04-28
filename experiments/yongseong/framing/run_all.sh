#!/usr/bin/env bash
# run_all.sh — orchestrate all 6 framing experiments (S-IEC = ECC × Diffusion).
#
# Usage:
#   ./run_all.sh inventory   # smoke test all 6 wrappers (no GPU calls)
#   ./run_all.sh light       # light phases for D, A, B (GPU); dry-run for C, E, F
#   ./run_all.sh full        # everything end-to-end (overnight; ~3–4 h)
#   ./run_all.sh plot        # regenerate plots/summaries from existing run dirs
#
# `full` mode runs every experiment, including exp_F's heavy n=2000 cross-FID,
# and writes a top-level `framing_summary_<stamp>.md` aggregating each
# experiment's verdict so the user can review everything in one place.
#
# Defaults to "inventory" if no argument given.
#
# `set -e` is intentionally OFF so a single failing experiment does not abort
# the rest of the chain — full mode is meant to leave behind whatever results
# completed before any failure.
set -uo pipefail
cd "$(dirname "$0")/../../.."  # → IEC/

PYTHON_PREFIX="${PYTHON_PREFIX:-env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} conda run --no-capture-output -n iec python}"
WRAPPERS_DIR="experiments/yongseong/framing"
RESULTS_DIR="experiments/yongseong/results"
PHASE="${1:-inventory}"

# Print the most recent date-stamped subdir of $1 (empty if none exist).
latest_run_dir() {
    local base="$1"
    [[ -d "${base}" ]] || return 0
    ls -td "${base}"/*/ 2>/dev/null | head -1 | sed 's:/$::'
}

run_inventory() {
    for w in exp_D_lookahead exp_A_correlation exp_B_innovation exp_C_figure1 exp_E_oracle exp_F_cross_error; do
        echo "=== ${w} :: inventory ==="
        ${PYTHON_PREFIX} "${WRAPPERS_DIR}/${w}.py" --phase inventory --dry-run
        echo
    done
}

run_light() {
    # Heaviest first: exp_D (4 short runs) → exp_B (1 fp16 run) → exp_A (1 ref + 5 deploy)
    echo "=== exp_D_lookahead :: verify (n=128 × 4) ==="
    ${PYTHON_PREFIX} "${WRAPPERS_DIR}/exp_D_lookahead.py" --phase all --no-dry-run

    echo "=== exp_B_innovation :: ref-trace + analyze (fp16 n=128) ==="
    ${PYTHON_PREFIX} "${WRAPPERS_DIR}/exp_B_innovation.py" --phase all --no-dry-run \
        --reuse-ref-from "$(latest_run_dir "${RESULTS_DIR}/exp_A_correlation")"

    echo "=== exp_A_correlation :: ref-trace + 5 deploy traces ==="
    ${PYTHON_PREFIX} "${WRAPPERS_DIR}/exp_A_correlation.py" --phase all --no-dry-run

    echo "=== exp_C_figure1 :: ingest + plot (no GPU) ==="
    ${PYTHON_PREFIX} "${WRAPPERS_DIR}/exp_C_figure1.py" --phase all --dry-run

    echo "=== exp_E_oracle :: dry-run (commands.sh) ==="
    ${PYTHON_PREFIX} "${WRAPPERS_DIR}/exp_E_oracle.py" --phase all --dry-run

    echo "=== exp_F_cross_error :: dry-run (commands.sh) ==="
    ${PYTHON_PREFIX} "${WRAPPERS_DIR}/exp_F_cross_error.py" --phase all --dry-run
}

# `full` mode: end-to-end run for unattended overnight execution.
#
# Order is chosen to maximise reuse and put the heaviest job last:
#   1. exp_A   — fp16 ref + 5 deploy traces (n=128). Produces ref_fp16_trace.pt
#                that exp_B and exp_E can reuse instead of re-tracing fp16.
#   2. exp_B   — Tweedie martingale check, reuses exp_A's ref-trace.
#   3. exp_E   — Oracle decoder, n=128 single-batch (multi-batch alignment is
#                a known limitation), reuses exp_A's ref-trace.
#   4. exp_D   — 4-way lookahead reuse comparison (n=128).
#   5. exp_C   — Figure 1 ingest from existing real_05 results (no GPU).
#   6. exp_F   — Cross-family decoder transfer, n=2000 sample + FID per pair.
#                This is the longest single phase (~1–2 h).
#
# Failures inside any single experiment are logged but do not abort the chain.
run_full() {
    local stamp
    stamp=$(date +%Y%m%d_%H%M%S)
    local LOG="${WRAPPERS_DIR}/run_full_${stamp}.log"
    mkdir -p "$(dirname "${LOG}")"
    echo "[full] start ${stamp}"
    echo "[full] log file: ${LOG}"
    echo "[full] tail -f ${LOG}  # to monitor"
    echo

    {
        echo "=== exp_A_correlation :: full (fp16 ref + 5 deploy n=128) ==="
        ${PYTHON_PREFIX} "${WRAPPERS_DIR}/exp_A_correlation.py" --phase all --no-dry-run \
            || echo "[fail] exp_A_correlation rc=$?"
    } 2>&1 | tee -a "${LOG}"

    local A_RUN_DIR
    A_RUN_DIR=$(latest_run_dir "${RESULTS_DIR}/exp_A_correlation")
    echo "[full] exp_A run dir: ${A_RUN_DIR:-<none>}" | tee -a "${LOG}"

    {
        echo "=== exp_B_innovation :: full (reuse exp_A ref-trace) ==="
        if [[ -n "${A_RUN_DIR}" ]]; then
            ${PYTHON_PREFIX} "${WRAPPERS_DIR}/exp_B_innovation.py" --phase all --no-dry-run \
                --reuse-ref-from "${A_RUN_DIR}" \
                || echo "[fail] exp_B_innovation rc=$?"
        else
            ${PYTHON_PREFIX} "${WRAPPERS_DIR}/exp_B_innovation.py" --phase all --no-dry-run \
                || echo "[fail] exp_B_innovation rc=$?"
        fi
    } 2>&1 | tee -a "${LOG}"

    {
        echo "=== exp_E_oracle :: full (n=128 oracle, reuse exp_A ref-trace) ==="
        if [[ -n "${A_RUN_DIR}" ]]; then
            ${PYTHON_PREFIX} "${WRAPPERS_DIR}/exp_E_oracle.py" --phase all --no-dry-run \
                --reuse-ref-from "${A_RUN_DIR}" \
                || echo "[fail] exp_E_oracle rc=$?"
        else
            ${PYTHON_PREFIX} "${WRAPPERS_DIR}/exp_E_oracle.py" --phase all --no-dry-run \
                || echo "[fail] exp_E_oracle rc=$?"
        fi
    } 2>&1 | tee -a "${LOG}"

    {
        echo "=== exp_D_lookahead :: full (4 verify runs n=128) ==="
        ${PYTHON_PREFIX} "${WRAPPERS_DIR}/exp_D_lookahead.py" --phase all --no-dry-run \
            || echo "[fail] exp_D_lookahead rc=$?"
    } 2>&1 | tee -a "${LOG}"

    {
        echo "=== exp_C_figure1 :: full (ingest from real_05; no GPU) ==="
        ${PYTHON_PREFIX} "${WRAPPERS_DIR}/exp_C_figure1.py" --phase all --dry-run \
            || echo "[fail] exp_C_figure1 rc=$?"
    } 2>&1 | tee -a "${LOG}"

    {
        echo "=== exp_F_cross_error :: full (execute-fid n=2000 × 5 pairs) ==="
        ${PYTHON_PREFIX} "${WRAPPERS_DIR}/exp_F_cross_error.py" --phase all --no-dry-run \
            || echo "[fail] exp_F_cross_error rc=$?"
    } 2>&1 | tee -a "${LOG}"

    aggregate_summary "${stamp}" 2>&1 | tee -a "${LOG}"
    echo "[full] done ${stamp}" | tee -a "${LOG}"
}

# Walk each experiment's latest run dir and concatenate its summary.md into a
# single top-level framing_summary_<stamp>.md the user can read at a glance.
aggregate_summary() {
    local stamp="$1"
    local OUT="${WRAPPERS_DIR}/framing_summary_${stamp}.md"
    {
        echo "# S-IEC Framing — Aggregate Summary"
        echo ""
        echo "_run timestamp: ${stamp}_"
        echo ""
        echo "Each section below is the per-experiment \`summary.md\` from the"
        echo "latest run dir under \`${RESULTS_DIR}/<exp>/\`."
        for exp in exp_A_correlation exp_B_innovation exp_D_lookahead exp_C_figure1 exp_E_oracle exp_F_cross_error; do
            echo ""
            echo "---"
            echo ""
            echo "## ${exp}"
            echo ""
            local latest
            latest=$(latest_run_dir "${RESULTS_DIR}/${exp}")
            if [[ -n "${latest}" && -f "${latest}/summary.md" ]]; then
                echo "_run dir: \`${latest}\`_"
                echo ""
                cat "${latest}/summary.md"
            elif [[ -n "${latest}" ]]; then
                echo "_run dir: \`${latest}\`  (no summary.md found — analyze/summary phase may have failed)_"
            else
                echo "_no run dir found under \`${RESULTS_DIR}/${exp}\`_"
            fi
        done
    } > "${OUT}"
    echo "[aggregate] wrote ${OUT}"
}

run_plot() {
    for w in exp_D_lookahead exp_A_correlation exp_B_innovation exp_C_figure1 exp_E_oracle exp_F_cross_error; do
        echo "=== ${w} :: plot+summary (latest run) ==="
        ${PYTHON_PREFIX} "${WRAPPERS_DIR}/${w}.py" --phase plot --dry-run || echo "  [skip] no completed run yet"
        ${PYTHON_PREFIX} "${WRAPPERS_DIR}/${w}.py" --phase summary --dry-run || echo "  [skip] no completed run yet"
        echo
    done
}

case "${PHASE}" in
    inventory) run_inventory ;;
    light)     run_light ;;
    full)      run_full ;;
    plot)      run_plot ;;
    *)
        echo "Unknown phase: ${PHASE}"
        echo "Usage: $0 [inventory|light|full|plot]"
        exit 1
        ;;
esac
