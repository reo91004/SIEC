#!/usr/bin/env python3
"""real_04_tradeoff — S-IEC Compute-Quality Tradeoff wrapper (실험 4).

원본 1저자 코드는 건드리지 않는다. 모든 결과/로그/수정 복사본은
`experiments/yongseong/` 아래에 모인다. 실행은 `conda run -n iec python ...`.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

IEC_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = IEC_ROOT / "experiments/yongseong"
DEFAULT_RESULTS = EXP_DIR / "results/real_04_tradeoff"
PILOT_SCORES = IEC_ROOT / "calibration/pilot_scores_nb.pt"
REFERENCE_NPZ = IEC_ROOT / "cifar10_reference.npz"
CIFAR_IMAGE_DIR = IEC_ROOT / "error_dec/cifar"
LOGS_DIR = IEC_ROOT / "logs"

FID_RE = re.compile(r"^FID:\s*([\d.]+)\s*$", re.MULTILINE)
FID_ALT_RE = re.compile(
    r"^(?:Frechet Inception Distance|frechet_inception_distance):\s*([\d.]+)\s*$",
    re.MULTILINE,
)
SFID_RE = re.compile(r"^sFID:\s*([\d.]+)\s*$", re.MULTILINE)
NUM_STEPS = 100

CSV_KEYS = [
    "method", "tau_percentile", "num_samples",
    "fid", "sfid", "trigger_rate", "per_sample_nfe", "nfe_total",
    "wall_clock_sec", "source_log", "source_npz", "notes",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--sample-batch", type=int, default=500)
    p.add_argument("--percentiles", type=int, nargs="+",
                   default=[30, 50, 60, 70, 80, 90, 95])
    p.add_argument("--timesteps", type=int, default=100)
    p.add_argument("--siec-max-rounds", type=int, default=1)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--skip-tau-calibration", action="store_true")
    p.add_argument("--skip-sampling", action="store_true")
    p.add_argument("--skip-fid", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--use-experiment-copy", action="store_true",
                   help="Candidate C1-C4 승인 시 실험 복사본 호출 (기본 off)")
    p.add_argument("--trigger-probs", type=float, nargs="+",
                   default=[0.1, 0.2, 0.4],
                   help="[C1] 랜덤 트리거의 Bernoulli 확률들 (실험 복사본 전용)")
    p.add_argument("--trigger-periods", type=int, nargs="+",
                   default=[3, 5, 10],
                   help="[C1] 균등 트리거의 주기 (실험 복사본 전용)")
    p.add_argument("--trigger-period-default", type=int, default=5,
                   help="[C1] random 모드에서 sampler가 사용하지 않는 placeholder 주기")
    return p.parse_args()


def tau_path(p: int) -> Path:
    return IEC_ROOT / f"calibration/tau_schedule_p{p}.pt"


def image_folder(p: int, n: int) -> Path:
    return CIFAR_IMAGE_DIR / f"image_siec_p{p}_n{n}"


def always_image_folder(n: int) -> Path:
    return CIFAR_IMAGE_DIR / f"image_siec_always_n{n}"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(IEC_ROOT))
    except ValueError:
        return str(path)


def conda_python(args) -> list[str]:
    return ["conda", "run", "--no-capture-output", "-n", "iec", "python"]


def entry_script(args, name: str) -> str:
    if args.use_experiment_copy and (EXP_DIR / name).exists():
        return f"experiments/yongseong/{name}"
    return f"mainddpm/{name}"


def png_count(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for _ in folder.glob("*.png"))


_N_CHECKS_CACHE: dict = {}


def n_checks_default() -> int:
    """pilot_scores_nb.pt 에서 **실제 syndrome 측정이 이뤄진** timestep 개수.

    pilot_scores 는 length==NUM_STEPS 로 pre-allocate 되고 interval_seq 위치에서만
    score 가 기록되며 나머지 bin 은 0 으로 채워진다. 즉 len(bin)>0 기준으로는
    모든 bin 이 True → 항상 100. 대신 "어떤 값이든 non-zero 인 bin" 을 세면
    정확히 |interval_seq| (DC10 기준 10) 가 나온다. 파일/환경이 없으면
    논문용 수치가 정의되지 않으므로 즉시 실패시킨다.
    """
    if "v" in _N_CHECKS_CACHE:
        return _N_CHECKS_CACHE["v"]
    import numpy as np
    import torch
    scores = torch.load(PILOT_SCORES, weights_only=False, map_location="cpu")
    v = 0
    for s in scores:
        arr = np.asarray(s)
        if arr.size and np.any(arr != 0):
            v += 1
    if v == 0:
        raise RuntimeError(f"no non-zero syndrome bins found in {PILOT_SCORES}")
    _N_CHECKS_CACHE["v"] = v
    return v


def run_cmd(cmd: list[str], log_path: Path | None = None) -> float:
    print(f"$ {shlex.join(cmd)}")
    t0 = time.time()
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            proc = subprocess.run(cmd, cwd=IEC_ROOT, stdout=f, stderr=subprocess.STDOUT)
    else:
        proc = subprocess.run(cmd, cwd=IEC_ROOT)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        raise SystemExit(f"Command failed (rc={proc.returncode}): {shlex.join(cmd)}")
    return elapsed


def ensure_tau_schedules(args, cmd_sink: list[list[str]]) -> None:
    for p in args.percentiles:
        if tau_path(p).exists():
            continue
        cmd = conda_python(args) + [
            entry_script(args, "calibrate_tau_cifar.py"),
            "--scores_path", "./calibration/pilot_scores_nb.pt",
            "--percentile", str(p),
            "--out_path", f"./calibration/tau_schedule_p{p}.pt",
        ]
        cmd_sink.append(cmd)
        if not args.dry_run:
            run_cmd(cmd)


def build_siec_cmd(args, p: int) -> list[str]:
    return conda_python(args) + [
        entry_script(args, "ddim_cifar_siec.py"),
        "--tau_path", f"./calibration/tau_schedule_p{p}.pt",
        "--tau_percentile", str(p),
        "--use_siec",
        "--num_samples", str(args.num_samples),
        "--sample_batch", str(args.sample_batch),
        "--weight_bit", "8", "--act_bit", "8",
        "--replicate_interval", "10",
        "--image_folder", f"./error_dec/cifar/image_siec_p{p}_n{args.num_samples}",
        "--siec_max_rounds", str(args.siec_max_rounds),
    ]


def build_always_cmd(args) -> list[str]:
    return conda_python(args) + [
        entry_script(args, "ddim_cifar_siec.py"),
        "--use_siec", "--siec_always_correct",
        "--num_samples", str(args.num_samples),
        "--sample_batch", str(args.sample_batch),
        "--weight_bit", "8", "--act_bit", "8",
        "--replicate_interval", "10",
        "--image_folder", f"./error_dec/cifar/image_siec_always_n{args.num_samples}",
        "--siec_max_rounds", str(args.siec_max_rounds),
    ]


def build_trigger_cmd(args, mode: str, prob: float, period: int) -> list[str]:
    """랜덤/균등 트리거 베이스라인 (C1). --use-experiment-copy 필요."""
    tag = f"{mode}{int(round(prob*100)) if mode == 'random' else period}"
    return conda_python(args) + [
        entry_script(args, "ddim_cifar_siec.py"),
        "--use_siec",
        "--trigger_mode", mode,
        "--trigger_prob", str(prob),
        "--trigger_period", str(period),
        "--tau_path", "./calibration/tau_schedule_never.pt",
        "--num_samples", str(args.num_samples),
        "--sample_batch", str(args.sample_batch),
        "--weight_bit", "8", "--act_bit", "8",
        "--replicate_interval", "10",
        "--image_folder", f"./error_dec/cifar/image_siec_{tag}_n{args.num_samples}",
        "--siec_max_rounds", str(args.siec_max_rounds),
    ]


def build_sweep_rows(args, cmd_sink: list[list[str]]) -> list[dict]:
    rows = []
    for p in args.percentiles:
        folder = image_folder(p, args.num_samples)
        npz = args.results_dir / f"samples_p{p}_n{args.num_samples}.npz"
        needs_run = not (npz.exists() or png_count(folder) >= args.num_samples)
        cmd = build_siec_cmd(args, p)
        if needs_run:
            cmd_sink.append(cmd)
        rows.append({
            "method": f"S-IEC p{p}",
            "tau_percentile": p,
            "num_samples": args.num_samples,
            "fid": None, "sfid": None, "trigger_rate": None,
            "per_sample_nfe": None, "nfe_total": None, "wall_clock_sec": None,
            "source_log": None, "source_npz": rel(npz),
            "notes": "sweep",
            "_image_folder": folder, "_npz": npz, "_cmd": cmd, "_needs_run": needs_run,
        })
    folder = always_image_folder(args.num_samples)
    npz = args.results_dir / f"samples_always_n{args.num_samples}.npz"
    needs_run = not (npz.exists() or png_count(folder) >= args.num_samples)
    cmd = build_always_cmd(args)
    if needs_run:
        cmd_sink.append(cmd)
    n_checks = n_checks_default()
    # always-on: 모든 interval 에서 trigger. n_triggered = n_checks.
    # per_sample_nfe = num_steps + n_checks + (rounds-1) * n_checks
    always_nfe = NUM_STEPS + n_checks + max(0, args.siec_max_rounds - 1) * n_checks
    rows.append({
        "method": "S-IEC always-on",
        "tau_percentile": None,
        "num_samples": args.num_samples,
        "fid": None, "sfid": None, "trigger_rate": 1.0,
        "per_sample_nfe": always_nfe,
        "nfe_total": args.num_samples * always_nfe,
        "wall_clock_sec": None,
        "source_log": None, "source_npz": rel(npz),
        "notes": f"always-on ablation (n_checks={n_checks})",
        "_image_folder": folder, "_npz": npz, "_cmd": cmd, "_needs_run": needs_run,
    })

    # [C1] 랜덤/균등 트리거 베이스라인 — 실험 복사본이 있을 때만 emit.
    # NFE 공식은 S-IEC sampler (adaptive_generalized_steps_siec) 기준:
    #   per_sample_nfe = num_steps + n_checks + (rounds - 1) * n_triggered
    # n_triggered 는 random 은 prob*n_checks, uniform 은 (1/period)*n_checks 기댓값.
    if args.use_experiment_copy and (EXP_DIR / "ddim_cifar_siec.py").exists():
        n_checks = n_checks_default()
        rounds_factor = max(0, args.siec_max_rounds - 1)

        # [C1-matched] 논문 주력 비교: 각 S-IEC percentile p 에 대해 측정된
        # trigger rate 와 동일한 rate 를 갖는 Random(prob=rate) / Uniform(period=round(1/rate))
        # 를 emit. build_sweep_rows 는 commands.sh 생성만 담당하므로, 이 블록은
        # tau_path(p) 가 이미 존재할 때만 postmortem 을 돌려 매칭 rate 를 구한다.
        for p in args.percentiles:
            if not tau_path(p).exists():
                continue
            p80_rate, _nfe, _n = postmortem(p, args.timesteps, args.siec_max_rounds)
            if p80_rate <= 0.0:
                raise RuntimeError(
                    f"cannot build compute-matched baselines for p{p}: "
                    f"postmortem trigger rate is {p80_rate}"
                )
            # Matched Random (prob = mean_rate)
            tag = f"random_matched_p{p}"
            folder = CIFAR_IMAGE_DIR / f"image_siec_{tag}_n{args.num_samples}"
            npz = args.results_dir / f"samples_{tag}_n{args.num_samples}.npz"
            needs_run = not (npz.exists() or png_count(folder) >= args.num_samples)
            cmd = build_trigger_cmd(args, "random", p80_rate, args.trigger_period_default)
            # override image_folder tag so the subprocess knows where to write
            for i, tok in enumerate(cmd):
                if tok.endswith(f"image_siec_random{int(round(p80_rate*100))}_n{args.num_samples}"):
                    cmd[i] = f"./error_dec/cifar/image_siec_{tag}_n{args.num_samples}"
                    break
            if needs_run:
                cmd_sink.append(cmd)
            random_nfe = NUM_STEPS + n_checks + rounds_factor * p80_rate * n_checks
            rows.append({
                "method": f"Random matched to p{p} (prob={p80_rate:.3f})",
                "tau_percentile": None,
                "num_samples": args.num_samples,
                "fid": None, "sfid": None,
                "trigger_rate": float(p80_rate),
                "per_sample_nfe": random_nfe,
                "nfe_total": args.num_samples * random_nfe,
                "wall_clock_sec": None,
                "source_log": None, "source_npz": rel(npz),
                "notes": (f"C1 compute-matched to S-IEC p{p} (rate={p80_rate:.4f}, "
                          f"n_checks={n_checks}); paper primary baseline"),
                "_image_folder": folder, "_npz": npz, "_cmd": cmd, "_needs_run": needs_run,
            })
            # Matched Uniform (period = round(1/rate))
            period_match = max(1, int(round(1.0 / p80_rate)))
            tag = f"uniform_matched_p{p}_period{period_match}"
            folder = CIFAR_IMAGE_DIR / f"image_siec_{tag}_n{args.num_samples}"
            npz = args.results_dir / f"samples_{tag}_n{args.num_samples}.npz"
            needs_run = not (npz.exists() or png_count(folder) >= args.num_samples)
            cmd = build_trigger_cmd(args, "uniform", 0.0, period_match)
            for i, tok in enumerate(cmd):
                if tok.endswith(f"image_siec_uniform{period_match}_n{args.num_samples}"):
                    cmd[i] = f"./error_dec/cifar/image_siec_{tag}_n{args.num_samples}"
                    break
            if needs_run:
                cmd_sink.append(cmd)
            expected_rate = 1.0 / period_match
            uniform_nfe = NUM_STEPS + n_checks + rounds_factor * expected_rate * n_checks
            rows.append({
                "method": f"Uniform matched to p{p} (period={period_match})",
                "tau_percentile": None,
                "num_samples": args.num_samples,
                "fid": None, "sfid": None,
                "trigger_rate": expected_rate,
                "per_sample_nfe": uniform_nfe,
                "nfe_total": args.num_samples * uniform_nfe,
                "wall_clock_sec": None,
                "source_log": None, "source_npz": rel(npz),
                "notes": (f"C1 compute-matched to S-IEC p{p} "
                          f"(target rate={p80_rate:.4f}, realized={expected_rate:.4f}, "
                          f"n_checks={n_checks}); paper primary baseline"),
                "_image_folder": folder, "_npz": npz, "_cmd": cmd, "_needs_run": needs_run,
            })

        for prob in args.trigger_probs:
            tag = f"random{int(round(prob*100))}"
            folder = CIFAR_IMAGE_DIR / f"image_siec_{tag}_n{args.num_samples}"
            npz = args.results_dir / f"samples_{tag}_n{args.num_samples}.npz"
            needs_run = not (npz.exists() or png_count(folder) >= args.num_samples)
            cmd = build_trigger_cmd(args, "random", prob, args.trigger_period_default)
            if needs_run:
                cmd_sink.append(cmd)
            per_sample_nfe = NUM_STEPS + n_checks + rounds_factor * prob * n_checks
            rows.append({
                "method": f"Random (p={prob:.2f})",
                "tau_percentile": None,
                "num_samples": args.num_samples,
                "fid": None, "sfid": None,
                "trigger_rate": float(prob),
                "per_sample_nfe": per_sample_nfe,
                "nfe_total": args.num_samples * per_sample_nfe,
                "wall_clock_sec": None,
                "source_log": None, "source_npz": rel(npz),
                "notes": (f"C1 grid (exploratory, not compute-matched): "
                          f"random trigger (실험 복사본, n_checks={n_checks})"),
                "_image_folder": folder, "_npz": npz, "_cmd": cmd, "_needs_run": needs_run,
            })
        for period in args.trigger_periods:
            tag = f"uniform{period}"
            folder = CIFAR_IMAGE_DIR / f"image_siec_{tag}_n{args.num_samples}"
            npz = args.results_dir / f"samples_{tag}_n{args.num_samples}.npz"
            needs_run = not (npz.exists() or png_count(folder) >= args.num_samples)
            cmd = build_trigger_cmd(args, "uniform", 0.0, period)
            if needs_run:
                cmd_sink.append(cmd)
            expected_rate = 1.0 / max(1, period)
            per_sample_nfe = NUM_STEPS + n_checks + rounds_factor * expected_rate * n_checks
            rows.append({
                "method": f"Uniform (period={period})",
                "tau_percentile": None,
                "num_samples": args.num_samples,
                "fid": None, "sfid": None,
                "trigger_rate": expected_rate,
                "per_sample_nfe": per_sample_nfe,
                "nfe_total": args.num_samples * per_sample_nfe,
                "wall_clock_sec": None,
                "source_log": None, "source_npz": rel(npz),
                "notes": (f"C1 grid (exploratory, not compute-matched): "
                          f"uniform trigger (실험 복사본, n_checks={n_checks})"),
                "_image_folder": folder, "_npz": npz, "_cmd": cmd, "_needs_run": needs_run,
            })
    return rows


def parse_fid_log(log_path: Path) -> tuple[float | None, float | None]:
    if not log_path.exists():
        raise FileNotFoundError(f"required FID log not found: {log_path}")
    text = log_path.read_text(errors="ignore")
    m_fid = FID_RE.search(text) or FID_ALT_RE.search(text)
    m_sfid = SFID_RE.search(text)
    if not m_fid:
        raise ValueError(f"FID metric not found in {log_path}")
    return (
        float(m_fid.group(1)),
        float(m_sfid.group(1)) if m_sfid else None,
    )


def seed_rows() -> list[dict]:
    fid_siec, sfid_siec = parse_fid_log(LOGS_DIR / "siec_fid.log")
    fid_never, sfid_never = parse_fid_log(LOGS_DIR / "siec_fid_never.log")
    fid_official, sfid_official = parse_fid_log(LOGS_DIR / "stage6_fid_result.txt")
    fid_recon, sfid_recon = parse_fid_log(LOGS_DIR / "stage6_fid_recon.log")
    n_checks = n_checks_default()
    # IEC author (`adaptive_generalized_steps_3`) 는 interval_seq 위치에서 max_iter=2
    # 로 2 forwards 를 항상 수행 (residual<tol break 은 max_iter=2 에서 no-op).
    # → NFE = 100 + n_checks. 기존 코드의 100 은 잘못된 하한이었음.
    iec_nfe = NUM_STEPS + n_checks
    # "No correction (never)" 는 siec sampler 를 tau=never 로 돌려 생성 → 여전히
    # interval_seq 위치에서 unconditional lookahead (step_nfe += 1) 가 발생 → 110.
    never_nfe = NUM_STEPS + n_checks
    return [
        {
            "method": "IEC (author)", "tau_percentile": None, "num_samples": 50000,
            "fid": fid_official,
            "sfid": sfid_official,
            "trigger_rate": 0.0,
            "per_sample_nfe": iec_nfe, "nfe_total": 50000 * iec_nfe,
            "wall_clock_sec": None,
            "source_log": "logs/stage6_fid_result.txt",
            "source_npz": "iec_samples.npz",
            "notes": f"50K seed; IEC author NFE = 100 + n_checks ({n_checks})",
        },
        {
            "method": "IEC+Recon", "tau_percentile": None, "num_samples": 50000,
            "fid": fid_recon,
            "sfid": sfid_recon,
            "trigger_rate": None, "per_sample_nfe": None, "nfe_total": None,
            "wall_clock_sec": None,
            "source_log": "logs/stage6_fid_recon.log",
            "source_npz": "iec_samples_recon.npz",
            "notes": "50K seed; reconstruction IEC variant",
        },
        {
            "method": "S-IEC p80 (seed 50K)", "tau_percentile": 80, "num_samples": 50000,
            "fid": fid_siec,
            "sfid": sfid_siec,
            "trigger_rate": None, "per_sample_nfe": None, "nfe_total": None,
            "wall_clock_sec": None,
            "source_log": "logs/siec_fid.log",
            "source_npz": "siec_samples.npz",
            "notes": "50K seed; trigger_rate/NFE via postmortem",
        },
        {
            "method": "No correction (never)", "tau_percentile": None, "num_samples": 2000,
            "fid": fid_never, "sfid": sfid_never,
            "trigger_rate": 0.0,
            "per_sample_nfe": never_nfe, "nfe_total": 2000 * never_nfe,
            "wall_clock_sec": None,
            "source_log": "logs/siec_fid_never.log",
            "source_npz": "siec_never.npz",
            "notes": (f"siec_never.npz=2K samples (NOT 50K); NFE = 100 + n_checks "
                      f"({n_checks}) because SIEC sampler 의 unconditional lookahead 포함; "
                      "plan Risks #2 참조"),
        },
        {
            "method": "Random trigger (placeholder)", "tau_percentile": None, "num_samples": None,
            "fid": None, "sfid": None,
            "trigger_rate": None, "per_sample_nfe": None, "nfe_total": None,
            "wall_clock_sec": None, "source_log": None, "source_npz": None,
            "notes": ("placeholder; see 'Random matched to p{P}' and 'Random (p=*)' rows "
                      "generated when --use-experiment-copy"),
        },
        {
            "method": "Uniform periodic (placeholder)", "tau_percentile": None, "num_samples": None,
            "fid": None, "sfid": None,
            "trigger_rate": None, "per_sample_nfe": None, "nfe_total": None,
            "wall_clock_sec": None, "source_log": None, "source_npz": None,
            "notes": ("placeholder; see 'Uniform matched to p{P}' and 'Uniform (period=*)' rows "
                      "generated when --use-experiment-copy"),
        },
    ]


def postmortem(percentile: int, num_steps: int, rounds: int):
    """pilot scores + tau 로 per-sample NFE / trigger rate 를 추정.

    실제 S-IEC NFE 구조 (adaptive_generalized_steps_siec):
      - base step 당 main forward: num_steps
      - interval_seq 위치마다 **trigger 여부와 무관한** lookahead forward 1회
      - siec_max_rounds >= 2 일 때 trigger 된 interval 에서 추가 lookahead (rounds-1)

    → per_sample_nfe = num_steps + N_check + (rounds - 1) * N_trigger
      where N_check = |interval_seq|, N_trigger = sum(trigger_rate) over interval.

    pilot_scores 는 interval_seq 위치에서만 데이터가 쌓이므로, 빈 배열이
    아닌 t 의 개수를 N_check 로 empirically 추정한다.
    """
    import numpy as np
    import torch

    scores = torch.load(PILOT_SCORES, weights_only=False, map_location="cpu")
    tau = torch.load(tau_path(percentile), weights_only=False, map_location="cpu")
    tau_np = np.asarray(tau).reshape(-1)
    rates = []
    n_checks = 0
    for t in range(num_steps):
        s = np.asarray(scores[t]) if t < len(scores) else np.array([])
        # pilot_scores 는 전 길이가 allocate 되지만 interval_seq 가 아닌 위치는
        # 0 으로 패딩. 실제 syndrome 측정이 있었던 bin 만 interval position 으로 카운트.
        if s.size == 0 or not np.any(s != 0):
            rates.append(0.0)
            continue
        n_checks += 1
        tau_t = float(tau_np[t]) if t < len(tau_np) else float("inf")
        rates.append(float((s > tau_t).mean()))
    sum_rate = float(np.sum(rates))
    mean_rate = (sum_rate / n_checks) if n_checks > 0 else 0.0
    per_sample_nfe = num_steps + n_checks + max(0, rounds - 1) * sum_rate
    return mean_rate, per_sample_nfe, n_checks


def fill_postmortem(rows: list[dict], args) -> None:
    for row in rows:
        p = row.get("tau_percentile")
        if p is None or row.get("trigger_rate") is not None and row.get("per_sample_nfe") is not None:
            continue
        if not tau_path(p).exists():
            row["notes"] = (row.get("notes") or "") + f"; tau_schedule_p{p}.pt missing"
            continue
        tr, nfe, n_checks = postmortem(p, args.timesteps, args.siec_max_rounds)
        row["trigger_rate"] = tr
        row["per_sample_nfe"] = nfe
        row["notes"] = ((row.get("notes") or "") + f"; n_checks={n_checks}").lstrip("; ")
        if row.get("num_samples"):
            row["nfe_total"] = row["num_samples"] * nfe


def pngs_to_npz(png_dir: Path, out_npz: Path) -> int:
    import numpy as np
    from PIL import Image

    paths = sorted(png_dir.glob("*.png"))
    arr = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths]).astype("uint8")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, arr_0=arr)
    return arr.shape[0]


def run_fid(sample_npz: Path, log_path: Path) -> tuple[float | None, float | None]:
    cmd = ["conda", "run", "--no-capture-output", "-n", "iec",
           "python", "evaluator_FID.py", str(REFERENCE_NPZ), str(sample_npz)]
    print(f"$ {shlex.join(cmd)}")
    proc = subprocess.run(cmd, cwd=IEC_ROOT, capture_output=True, text=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(proc.stdout + proc.stderr)
    m_fid = FID_RE.search(proc.stdout)
    m_sfid = SFID_RE.search(proc.stdout)
    return (
        float(m_fid.group(1)) if m_fid else None,
        float(m_sfid.group(1)) if m_sfid else None,
    )


def execute_sweep(rows: list[dict], args) -> None:
    if not args.skip_sampling:
        for row in rows:
            if not row.get("_needs_run"):
                continue
            log = args.results_dir / "logs" / f"sampling_{row['method'].replace(' ', '_')}.log"
            row["wall_clock_sec"] = run_cmd(row["_cmd"], log)
    if args.skip_fid:
        return
    for row in rows:
        folder = row["_image_folder"]
        npz = row["_npz"]
        if not npz.exists():
            if png_count(folder) == 0:
                row["notes"] = (row.get("notes") or "") + "; sampling output missing"
                continue
            pngs_to_npz(folder, npz)
        fid_log = args.results_dir / "logs" / f"fid_{row['method'].replace(' ', '_')}.log"
        fid, sfid = parse_fid_log(fid_log) if fid_log.exists() else run_fid(npz, fid_log)
        row["fid"] = fid
        row["sfid"] = sfid
        row["source_log"] = rel(fid_log)
        row["source_npz"] = rel(npz)


def save_results(rows: list[dict], results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    clean = [{k: r.get(k) for k in CSV_KEYS} for r in rows]
    with open(results_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_KEYS)
        writer.writeheader()
        writer.writerows(clean)
    with open(results_dir / "results.json", "w") as f:
        json.dump(clean, f, indent=2)


def write_commands(cmds: list[list[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("# Generated by real_04_tradeoff.py — review before running.\n")
        f.write(f"cd {IEC_ROOT}\n\n")
        for cmd in cmds:
            f.write(shlex.join(cmd) + "\n")
    path.chmod(0o755)


def plot_two_panel(rows: list[dict], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    siec = sorted(
        [r for r in rows if r.get("method", "").startswith("S-IEC p")
         and r.get("fid") is not None and r.get("per_sample_nfe") is not None
         and r.get("tau_percentile") is not None],
        key=lambda r: r["tau_percentile"],
    )
    if siec:
        xs = [r["per_sample_nfe"] for r in siec]
        ys = [r["fid"] for r in siec]
        axes[0].plot(xs, ys, "o-", color="tab:blue", label="S-IEC τ sweep")
        for x, y, p in zip(xs, ys, [r["tau_percentile"] for r in siec]):
            axes[0].annotate(f"p{p}", (x, y), textcoords="offset points",
                             xytext=(5, 5), fontsize=8)

    markers = {
        "IEC (author)": ("s", "tab:orange", "IEC"),
        "IEC+Recon": ("D", "tab:brown", "IEC+Recon"),
        "S-IEC always-on": ("^", "tab:red", "Always-on"),
        "No correction (never)": ("v", "tab:gray", "No correction"),
        "S-IEC p80 (seed 50K)": ("*", "tab:green", "S-IEC p80 (50K seed)"),
    }
    for r in rows:
        spec = markers.get(r["method"])
        if spec is None or r.get("fid") is None or r.get("per_sample_nfe") is None:
            continue
        marker, color, label = spec
        axes[0].scatter([r["per_sample_nfe"]], [r["fid"]],
                        marker=marker, s=90, color=color, label=label)
    axes[0].set_xlabel("Per-sample NFE")
    axes[0].set_ylabel("FID")
    axes[0].set_title("Compute-Quality Tradeoff")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    trig = [r for r in rows if r.get("method", "").startswith("S-IEC p")
            and r.get("trigger_rate") is not None and r.get("fid") is not None]
    if trig:
        xs = [r["trigger_rate"] for r in trig]
        ys = [r["fid"] for r in trig]
        axes[1].scatter(xs, ys, color="tab:blue")
        for x, y, p in zip(xs, ys, [r["tau_percentile"] for r in trig]):
            if p is None:
                continue
            axes[1].annotate(f"p{p}", (x, y), textcoords="offset points",
                             xytext=(5, 5), fontsize=8)
    axes[1].set_xlabel("Mean trigger rate")
    axes[1].set_ylabel("FID")
    axes[1].set_title("Trigger Rate vs Quality (τ as control knob)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def write_compute_matched(rows: list[dict], out_md: Path) -> None:
    iec = next((r for r in rows if r["method"] == "IEC (author)"), None)
    siec = [r for r in rows if r.get("method", "").startswith("S-IEC p")
            and r.get("per_sample_nfe") is not None and r.get("fid") is not None]
    if iec is None or iec.get("per_sample_nfe") is None or not siec:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("# Compute-Matched Row\n\nNot enough data yet.\n")
        return
    matched = min(siec, key=lambda r: abs(r["per_sample_nfe"] - iec["per_sample_nfe"]))

    def fmt(v, places=4):
        return "—" if v is None else f"{v:.{places}f}"

    lines = [
        "# Compute-Matched Row (real_04_tradeoff)",
        "",
        f"IEC per-sample NFE = {iec['per_sample_nfe']:.2f}. Nearest S-IEC point:",
        "",
        "| Method | FID | sFID | per_sample_NFE | trigger_rate |",
        "|---|---|---|---|---|",
        f"| {iec['method']} | {fmt(iec['fid'])} | {fmt(iec['sfid'])} "
        f"| {iec['per_sample_nfe']:.2f} | 0.0000 |",
        f"| {matched['method']} | {fmt(matched['fid'])} | {fmt(matched['sfid'])} "
        f"| {matched['per_sample_nfe']:.2f} | {fmt(matched['trigger_rate'])} |",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))


def load_csv_rows(path: Path) -> list[dict]:
    with open(path) as f:
        raw = list(csv.DictReader(f))

    def cast(v):
        if v in (None, "", "None"):
            return None
        try:
            return float(v) if "." in v else int(v)
        except ValueError:
            return v

    rows = []
    for r in raw:
        rows.append({k: cast(v) for k, v in r.items()})
    return rows


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        rows = load_csv_rows(args.results_dir / "results.csv")
        plot_two_panel(rows, args.results_dir / "tradeoff_2panel.png")
        write_compute_matched(rows, args.results_dir / "compute_matched.md")
        print(f"Replotted {len(rows)} rows → {args.results_dir}")
        return

    cmd_list: list[list[str]] = []
    if not args.skip_tau_calibration:
        ensure_tau_schedules(args, cmd_list)
    sweep = build_sweep_rows(args, cmd_list)

    all_rows = seed_rows() + sweep
    fill_postmortem(all_rows, args)
    write_commands(cmd_list, args.results_dir / "commands.sh")

    if args.dry_run:
        save_results(all_rows, args.results_dir)
        print(f"Dry-run: {len(cmd_list)} commands → {args.results_dir/'commands.sh'}")
        print(f"         rows={len(all_rows)} (csv/json written for inspection)")
        missing_tau = [p for p in args.percentiles if not tau_path(p).exists()]
        if missing_tau:
            print(f"         missing tau schedules: {missing_tau}")
        return

    execute_sweep(sweep, args)
    save_results(all_rows, args.results_dir)
    plot_two_panel(all_rows, args.results_dir / "tradeoff_2panel.png")
    write_compute_matched(all_rows, args.results_dir / "compute_matched.md")
    print(f"Done. Results in {args.results_dir}")


if __name__ == "__main__":
    main()
