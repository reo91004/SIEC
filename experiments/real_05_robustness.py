#!/usr/bin/env python3
"""real_05_robustness — Robustness Across Deployment Errors wrapper (실험 5).

Six deployment-error settings (fp16 → W8A8 → DeepCache → W4A8 → CacheQuant).
Default phase is `inventory` so no sampling runs until the user reviews which
settings are actually unblocked. 원본 1저자 코드는 건드리지 않는다.
모든 결과/로그/수정 복사본은 `experiments/yongseong/` 아래.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import subprocess
import time
from pathlib import Path

IEC_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = IEC_ROOT / "experiments/yongseong"
DEFAULT_RESULTS = EXP_DIR / "results/real_05_robustness"
CALIB = IEC_ROOT / "calibration"
ERROR_DEC = IEC_ROOT / "error_dec/cifar"
REFERENCE_NPZ = IEC_ROOT / "cifar10_reference.npz"
LOGS_DIR = IEC_ROOT / "logs"

FID_RE = re.compile(r"^FID:\s*([\d.]+)\s*$", re.MULTILINE)
FID_ALT_RE = re.compile(
    r"^(?:Frechet Inception Distance|frechet_inception_distance):\s*([\d.]+)\s*$",
    re.MULTILINE,
)
SFID_RE = re.compile(r"^sFID:\s*([\d.]+)\s*$", re.MULTILINE)
NUM_STEPS = 100

CSV_KEYS = [
    "setting", "error_strength", "method", "tau_percentile", "num_samples",
    "fid", "sfid", "trigger_rate", "per_sample_nfe", "nfe_total",
    "wall_clock_sec", "source_npz", "source_log", "notes",
]


def setting_defs() -> dict[str, dict]:
    """Ordered by increasing error strength (hypothesis-level)."""
    return {
        "fp16": dict(
            weight_bit=None, act_bit=None, replicate_interval=None,
            rank=1, description="no quant + no cache (fp16 reference)",
            required_assets=[],
            unblock_via=("Candidate C4: ddim_cifar_siec.py + deepcache.py "
                         "실험 복사본에서 --no-ptq/--no-cache 플래그 제공."),
            runnable=False,
            needs_core_flag=True,
        ),
        "W8A8_DC10": dict(
            weight_bit=8, act_bit=8, replicate_interval=10,
            rank=2, description="W8A8 quant + DeepCache interval=10",
            required_assets=[
                "error_dec/cifar/pre_quanterr_abCov_weight8_interval10_list_timesteps100.pth",
                "error_dec/cifar/pre_cacheerr_abCov_interval10_list_timesteps100.pth",
                "error_dec/cifar/weight_params_W8_cache10_timesteps100.pth",
            ],
            unblock_via=None,
            runnable=True,
            needs_core_flag=False,
        ),
        "W8A8_DC20": dict(
            weight_bit=8, act_bit=8, replicate_interval=20,
            rank=3, description="W8A8 quant + DeepCache interval=20",
            required_assets=[
                "error_dec/cifar/pre_quanterr_abCov_weight8_interval20_list_timesteps100.pth",
                "error_dec/cifar/pre_cacheerr_abCov_interval20_list_timesteps100.pth",
            ],
            unblock_via=("Re-run mainddpm/ddim_cifar_cali.py with "
                         "--replicate_interval 20 to generate DEC params (no core edit)."),
            runnable=False,
            needs_core_flag=False,
        ),
        "W4A8_DC10": dict(
            weight_bit=4, act_bit=8, replicate_interval=10,
            rank=4, description="W4A8 quant + DeepCache interval=10",
            required_assets=[
                "error_dec/cifar/pre_quanterr_abCov_weight4_interval10_list_timesteps100.pth",
                "error_dec/cifar/weight_params_W4_cache10_timesteps100.pth",
            ],
            unblock_via=("Re-run mainddpm/ddim_cifar_params.py with --weight_bit 4, "
                         "then regenerate DEC params (heavy GPU work)."),
            runnable=False,
            needs_core_flag=False,
        ),
        "W8A8_DC50": dict(
            weight_bit=8, act_bit=8, replicate_interval=50,
            rank=5, description="W8A8 quant + DeepCache interval=50",
            required_assets=[
                "error_dec/cifar/pre_quanterr_abCov_weight8_interval50_list_timesteps100.pth",
                "error_dec/cifar/pre_cacheerr_abCov_interval50_list_timesteps100.pth",
            ],
            unblock_via=("Re-run mainddpm/ddim_cifar_cali.py with "
                         "--replicate_interval 50 (no core edit)."),
            runnable=False,
            needs_core_flag=False,
        ),
        "CacheQuant": dict(
            weight_bit=4, act_bit=8, replicate_interval=10,
            rank=6, description="W4A8 + DeepCache (CacheQuant joint regime)",
            required_assets=[
                "error_dec/cifar/pre_quanterr_abCov_weight4_interval10_list_timesteps100.pth",
                "error_dec/cifar/pre_cacheerr_abCov_interval10_list_timesteps100.pth",
                "error_dec/cifar/weight_params_W4_cache10_timesteps100.pth",
            ],
            unblock_via="After Setting 4 (W4A8_DC10) is unblocked; combined regime.",
            runnable=False,
            needs_core_flag=False,
        ),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["inventory", "pilot", "calibrate",
                                        "main", "fid", "plot", "all"],
                   default="inventory")
    p.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--settings", nargs="+", default=None,
                   help="subset of setting labels (default: all)")
    p.add_argument("--pilot-samples", type=int, default=512)
    p.add_argument("--num-samples", type=int, default=2000)
    p.add_argument("--sample-batch", type=int, default=500)
    p.add_argument("--percentile", type=float, default=80.0,
                   help="single tau percentile for S-IEC per setting")
    p.add_argument("--siec-max-rounds", type=int, default=1)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--use-experiment-copy", action="store_true",
                   help="Candidate C1-C4 승인 시 실험 복사본 호출 (기본 off)")
    return p.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(IEC_ROOT))
    except ValueError:
        return str(path)


def conda_python() -> list[str]:
    return ["conda", "run", "--no-capture-output", "-n", "iec", "python"]


def entry_script(args, name: str) -> str:
    if args.use_experiment_copy and (EXP_DIR / name).exists():
        return f"experiments/yongseong/{name}"
    return f"mainddpm/{name}"


def png_count(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for _ in folder.glob("*.png"))


def pilot_scores_path(label: str) -> Path:
    """Setting 별 pilot scores 파일 경로.

    W8A8_DC10 은 canonical setting 이라 기존 `pilot_scores_nb.pt` 를 그대로 사용.
    다른 라벨은 `pilot_scores_{label}.pt` 로 저장된다. 이렇게 하면 wrapper 가
    symlink 를 만들 필요 없이 commands.sh 가 실제 존재하는 파일을 가리킨다.
    """
    if label == "W8A8_DC10":
        return CALIB / "pilot_scores_nb.pt"
    return CALIB / f"pilot_scores_{label}.pt"


def tau_schedule_path(label: str, percentile: float) -> Path:
    """Setting 별 tau schedule 파일 경로. W8A8_DC10 은 canonical 파일 재사용."""
    p = int(round(percentile))
    if label == "W8A8_DC10":
        return CALIB / f"tau_schedule_p{p}.pt"
    return CALIB / f"tau_schedule_{label}_p{p}.pt"


_N_CHECKS_CACHE: dict = {}


def n_checks_for_setting(label: str) -> int:
    """pilot scores 의 실제 syndrome 측정 bin 수 (= |interval_seq|).

    pilot_scores 는 길이 NUM_STEPS 전체로 pre-allocate 되고 interval_seq 위치에서만
    값이 기록되며 나머지는 0 패딩. 따라서 `len(x)>0` 기준으로 세면 항상 100.
    "non-zero 가 한 번이라도 있는 bin" 을 세면 정확히 |interval_seq| 를 얻는다.
    pilot 이 없거나 환경이 맞지 않으면 논문용 NFE 가 정의되지 않으므로 즉시 실패시킨다.
    """
    key = label
    if key in _N_CHECKS_CACHE:
        return _N_CHECKS_CACHE[key]
    p = pilot_scores_path(label)
    import numpy as np
    import torch
    s = torch.load(p, weights_only=False, map_location="cpu")
    v = 0
    for x in s:
        arr = np.asarray(x)
        if arr.size and np.any(arr != 0):
            v += 1
    if v == 0:
        raise RuntimeError(f"no non-zero syndrome bins found in {p}")
    _N_CHECKS_CACHE[key] = v
    return v


def image_folder_main(label: str, method: str, n: int) -> Path:
    return ERROR_DEC / f"image_robust_{label}_{method}_n{n}"


def image_folder_pilot(label: str, n: int) -> Path:
    return ERROR_DEC / f"image_robust_pilot_{label}_n{n}"


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


def check_assets(defs: dict[str, dict], args=None) -> dict[str, dict]:
    """{label: {status, missing: [...], runnable: bool, unblock_via, ...}} 반환."""
    # [C4] fp16 은 --no-ptq/--no-cache 를 노출하는 실험 복사본이 있을 때만 runnable.
    # 복사본이 없으면 core CLI에 해당 플래그가 없어 그대로 blocked 유지.
    exp_available = bool(
        args is not None
        and getattr(args, "use_experiment_copy", False)
        and (EXP_DIR / "ddim_cifar_siec.py").exists()
    )
    report = {}
    for label, d in defs.items():
        d = dict(d)
        if label == "fp16" and exp_available:
            d["needs_core_flag"] = False
            d["runnable"] = True
        missing = [a for a in d["required_assets"] if not (IEC_ROOT / a).exists()]
        assets_ok = not missing and not d["needs_core_flag"]
        status = "runnable" if (d["runnable"] and assets_ok) else "blocked"
        report[label] = {
            **d,
            "missing": missing,
            "status": status,
        }
    return report


def write_inventory(report: dict[str, dict], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Experiment 5 — Setting Inventory",
        "",
        "오늘 기준으로 실험 5의 6개 setting이 실행 가능한지 검사한 표.",
        "`runnable`만 wrapper가 바로 돌릴 수 있고, `blocked`는 unblock_via에",
        "적힌 절차를 수행해야 runnable이 된다. 어떤 경우에도 1저자 core 코드를",
        "수정하지 않는다 — core 수정이 필요한 setting은 Candidate 플래그로 표시.",
        "",
        "| # | Setting | Status | W | A | Interval | Missing assets | Unblock via |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, info in sorted(report.items(), key=lambda kv: kv[1]["rank"]):
        w = info["weight_bit"] if info["weight_bit"] is not None else "—"
        a = info["act_bit"] if info["act_bit"] is not None else "—"
        iv = info["replicate_interval"] if info["replicate_interval"] is not None else "—"
        missing = ", ".join(info["missing"]) if info["missing"] else "—"
        if info["needs_core_flag"]:
            missing = (missing + "; " if missing != "—" else "") + "CLI flag (core)"
        unblock = info["unblock_via"] or "—"
        lines.append(
            f"| {info['rank']} | `{label}` | **{info['status']}** | {w} | {a} | {iv} | "
            f"{missing} | {unblock} |"
        )
    lines += [
        "",
        "## Method rows per setting",
        "",
        "양자화/캐시 regime (W8A8_DC10 등) 은 세 줄: `no-correction`, `IEC (author)`, `S-IEC`.",
        "**fp16** 은 과학적으로 IEC/S-IEC 가 모두 no-op (양자화 에러가 없어 수정할 것이 없음) "
        "이므로 단일 `fp16 reference` row 로 축소. pilot/calibrate 단계도 생략.",
        "",
        "- `fp16 reference`: `--no-use-siec` 로 plain DDIM 호출 (C4 실험 복사본의 `--no-ptq/--no-cache`).",
        "- `no-correction`: `--tau_path calibration/tau_schedule_never.pt --use_siec` "
        "(모든 step trigger off; S-IEC sampler 의 unconditional lookahead 는 남아 NFE=110).",
        "- `IEC (author)`: `W8A8_DC10` 은 기존 `iec_samples.npz` (50K seed) 재사용. "
        "다른 setting 에서는 Candidate C2 (`--no-use-siec`) 실험 복사본에서 fresh run 가능. "
        "NFE = 100 + n_checks (max_iter=2 at interval_seq).",
        "- `S-IEC`: `--tau_path tau_schedule_{label}_p{P}.pt` (W8A8_DC10 은 canonical "
        "`tau_schedule_p{P}.pt` 직접 참조). fp16 에는 적용 불가.",
        "",
        "## Error-strength axis",
        "",
        "`error_strength(label) = mean_over_t(mean(pilot_scores_{label}[t]))` 만 사용.",
        "fp16 은 syndrome score 가 정의되지 않는 reference regime 이므로 0.0 으로 둔다.",
        "그 외 setting 에 pilot score 가 없으면 robustness axis 를 만들 수 없으므로 실패시킨다.",
    ]
    out_md.write_text("\n".join(lines))


def build_pilot_cmd(args, label: str, info: dict) -> list[str]:
    # [C4] fp16: weight/act/interval 값이 None 이라 더미 8/8/10 을 넘기고,
    # 실제 양자화/캐시 경로는 --no-ptq/--no-cache 로 실험 복사본에서 끈다.
    is_fp16 = label == "fp16"
    w = info["weight_bit"] if info["weight_bit"] is not None else 8
    a = info["act_bit"] if info["act_bit"] is not None else 8
    iv = info["replicate_interval"] if info["replicate_interval"] is not None else 10
    cmd = conda_python() + [
        entry_script(args, "ddim_cifar_siec.py"),
        "--num_samples", str(args.pilot_samples),
        "--sample_batch", str(args.sample_batch),
        "--weight_bit", str(w),
        "--act_bit", str(a),
        "--replicate_interval", str(iv),
        "--use_siec", "--siec_collect_scores",
        "--siec_scores_out", str(pilot_scores_path(label).relative_to(IEC_ROOT)),
        "--image_folder", str(image_folder_pilot(label, args.pilot_samples).relative_to(IEC_ROOT)),
    ]
    if is_fp16:
        cmd += ["--no-ptq", "--no-cache"]
    return cmd


def build_calibrate_cmd(args, label: str) -> list[str]:
    return conda_python() + [
        entry_script(args, "calibrate_tau_cifar.py"),
        "--scores_path", str(pilot_scores_path(label).relative_to(IEC_ROOT)),
        "--percentile", str(int(round(args.percentile))),
        "--out_path", str(tau_schedule_path(label, args.percentile).relative_to(IEC_ROOT)),
    ]


def build_main_cmd(args, label: str, info: dict, method: str) -> list[str]:
    folder = image_folder_main(label, method, args.num_samples)
    # [C4] fp16: 더미 양자화/캐시 값을 넘기고 --no-ptq/--no-cache 를 추가.
    is_fp16 = label == "fp16"
    w = info["weight_bit"] if info["weight_bit"] is not None else 8
    a = info["act_bit"] if info["act_bit"] is not None else 8
    iv = info["replicate_interval"] if info["replicate_interval"] is not None else 10
    common = conda_python() + [
        entry_script(args, "ddim_cifar_siec.py"),
        "--num_samples", str(args.num_samples),
        "--sample_batch", str(args.sample_batch),
        "--weight_bit", str(w),
        "--act_bit", str(a),
        "--replicate_interval", str(iv),
        "--image_folder", str(folder.relative_to(IEC_ROOT)),
        "--siec_max_rounds", str(args.siec_max_rounds),
    ]
    if is_fp16:
        common += ["--no-ptq", "--no-cache"]
    exp_available = (
        getattr(args, "use_experiment_copy", False)
        and (EXP_DIR / "ddim_cifar_siec.py").exists()
    )
    if method == "fp16_ref":
        # [C4] fp16 reference: plain DDIM. interval_seq=None 이라 S-IEC lookahead 없음.
        # --no-use-siec 로 adaptive_generalized_steps_3 호출 시 max_iter=1 분기 = plain DDIM.
        return common + ["--no-use-siec"]
    if method == "no_correction":
        return common + ["--use_siec", "--tau_path", "./calibration/tau_schedule_never.pt"]
    if method == "siec":
        tau = tau_schedule_path(label, args.percentile).relative_to(IEC_ROOT)
        return common + [
            "--use_siec",
            "--tau_path", str(tau),
            "--tau_percentile", str(int(round(args.percentile))),
        ]
    if method == "iec":
        # [C2] --no-use-siec 는 진짜 IEC-only fresh run 을 만든다.
        if exp_available:
            return common + ["--no-use-siec"]
        raise RuntimeError("IEC fresh run requires --use-experiment-copy (C2 --no-use-siec)")
    raise ValueError(f"unknown method {method}")


def seed_row_w8a8_dc10_iec() -> dict:
    """Re-use existing iec_samples.npz (50K) as the IEC baseline for W8A8_DC10.

    IEC author (`adaptive_generalized_steps_3`) 는 interval_seq 위치에서 max_iter=2
    로 2 개의 forward 를 수행 (residual early-exit 은 max_iter=2 에서 no-op).
    → NFE = NUM_STEPS + n_checks.
    """
    log = LOGS_DIR / "stage6_fid_result.txt"
    fid, sfid = parse_required_fid_log(log)
    n_checks = n_checks_for_setting("W8A8_DC10")
    nfe = NUM_STEPS + n_checks
    return {
        "setting": "W8A8_DC10", "error_strength": None,
        "method": "IEC (author)", "tau_percentile": None, "num_samples": 50000,
        "fid": fid, "sfid": sfid,
        "trigger_rate": 0.0,
        "per_sample_nfe": nfe, "nfe_total": 50000 * nfe,
        "wall_clock_sec": None,
        "source_npz": "iec_samples.npz",
        "source_log": "logs/stage6_fid_result.txt",
        "notes": f"50K seed; IEC author NFE = 100 + n_checks ({n_checks}); trigger_rate=0",
    }


def seed_row_w8a8_dc10_siec_p80() -> dict:
    log = LOGS_DIR / "siec_fid.log"
    fid, sfid = parse_required_fid_log(log)
    return {
        "setting": "W8A8_DC10", "error_strength": None,
        "method": "S-IEC (p80 seed 50K)", "tau_percentile": 80, "num_samples": 50000,
        "fid": fid,
        "sfid": sfid,
        "trigger_rate": None, "per_sample_nfe": None, "nfe_total": None,
        "wall_clock_sec": None,
        "source_npz": "siec_samples.npz",
        "source_log": "logs/siec_fid.log",
        "notes": "50K seed; S-IEC p80 on W8A8_DC10 (main setting)",
    }


def parse_required_fid_log(log_path: Path) -> tuple[float, float | None]:
    if not log_path.exists():
        raise FileNotFoundError(f"required FID log not found: {log_path}")
    text = log_path.read_text(errors="ignore")
    m_fid = FID_RE.search(text) or FID_ALT_RE.search(text)
    m_sfid = SFID_RE.search(text)
    if not m_fid:
        raise ValueError(f"FID metric not found in {log_path}")
    return float(m_fid.group(1)), float(m_sfid.group(1)) if m_sfid else None


def compute_error_strength(label: str) -> float:
    """Mean over timesteps of mean syndrome score. Requires pilot scores.

    pilot_scores_path 가 W8A8_DC10 에 대해 canonical `pilot_scores_nb.pt` 를
    반환하므로 별도 alias 가 필요 없다.
    """
    if label == "fp16":
        return 0.0
    p = pilot_scores_path(label)
    import numpy as np
    import torch

    scores = torch.load(p, weights_only=False, map_location="cpu")
    per_t = []
    for s in scores:
        arr = np.asarray(s) if s is not None else np.array([])
        if arr.size and np.any(arr != 0):
            per_t.append(float(arr.mean()))
    if not per_t:
        raise RuntimeError(f"no non-zero syndrome bins found in {p}")
    return float(sum(per_t) / len(per_t))


def postmortem_trigger_nfe(label: str, percentile: float, rounds: int) -> tuple[float, float, int]:
    """Return (mean_trigger_rate, per_sample_nfe, n_checks).

    NFE 공식은 `adaptive_generalized_steps_siec` 기준:
      per_sample_nfe = num_steps + n_checks + max(0, rounds - 1) * sum_rate
    n_checks 는 pilot scores 의 비어있지 않은 bin 수 (= |interval_seq|).
    """
    scores = pilot_scores_path(label)
    tau = tau_schedule_path(label, percentile)
    import numpy as np
    import torch

    s = torch.load(scores, weights_only=False, map_location="cpu")
    t_np = np.asarray(torch.load(tau, weights_only=False, map_location="cpu")).reshape(-1)
    rates: list[float] = []
    n_checks = 0
    for t in range(NUM_STEPS):
        arr = np.asarray(s[t]) if t < len(s) else np.array([])
        # pilot_scores 는 interval_seq 밖 위치가 0 패딩이므로 len(arr)>0 로는
        # 전체가 interval 로 잡힌다. non-zero 존재 여부로 실제 측정 bin 만 카운트.
        if arr.size == 0 or not np.any(arr != 0):
            rates.append(0.0)
            continue
        n_checks += 1
        tau_t = float(t_np[t]) if t < len(t_np) else float("inf")
        rates.append(float((arr > tau_t).mean()))
    sum_rate = float(np.sum(rates))
    mean_rate = (sum_rate / n_checks) if n_checks > 0 else 0.0
    if n_checks == 0:
        raise RuntimeError(f"no non-zero syndrome bins found in {scores}")
    per_sample_nfe = NUM_STEPS + n_checks + max(0, rounds - 1) * sum_rate
    return mean_rate, per_sample_nfe, n_checks


def pngs_to_npz(png_dir: Path, out_npz: Path) -> int:
    import numpy as np
    from PIL import Image

    paths = sorted(png_dir.glob("*.png"))
    arr = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in paths]).astype("uint8")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, arr_0=arr)
    return arr.shape[0]


def run_fid(sample_npz: Path, log_path: Path) -> tuple[float | None, float | None]:
    cmd = conda_python() + ["evaluator_FID.py", str(REFERENCE_NPZ), str(sample_npz)]
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


def selected_labels(args, report: dict[str, dict]) -> list[str]:
    if args.settings:
        return [s for s in args.settings if s in report]
    return [k for k, v in sorted(report.items(), key=lambda kv: kv[1]["rank"])]


def supports_siec(label: str) -> bool:
    """S-IEC 파이프라인이 실제로 동작하는 setting 인지.

    fp16 (no-cache) 은 interval_seq=None 이라 S-IEC lookahead 가 정의되지 않는다.
    → fp16 은 no_correction + IEC 만 지원하고 pilot/calibrate/siec 는 skip.
    """
    return label != "fp16"


def phase_pilot(args, report: dict[str, dict], cmd_sink: list[list[str]]) -> None:
    # W8A8_DC10 은 pilot_scores_path() 가 canonical `pilot_scores_nb.pt` 를 반환하므로
    # 이미 존재할 경우 run 을 skip. 다른 라벨은 label-keyed 경로에 생성.
    for label in selected_labels(args, report):
        info = report[label]
        if info["status"] != "runnable":
            continue
        if not supports_siec(label):
            continue
        if pilot_scores_path(label).exists():
            continue
        cmd = build_pilot_cmd(args, label, info)
        cmd_sink.append(cmd)
        if not args.dry_run:
            log = args.results_dir / "logs" / f"pilot_{label}.log"
            run_cmd(cmd, log)


def phase_calibrate(args, report: dict[str, dict], cmd_sink: list[list[str]]) -> None:
    # W8A8_DC10 은 tau_schedule_path() 가 canonical `tau_schedule_p{P}.pt` 를 반환하므로
    # 이미 존재할 경우 run 을 skip. 다른 라벨은 label-keyed 경로에 생성.
    for label in selected_labels(args, report):
        info = report[label]
        if info["status"] != "runnable":
            continue
        if not supports_siec(label):
            continue
        tau = tau_schedule_path(label, args.percentile)
        if tau.exists():
            continue
        cmd = build_calibrate_cmd(args, label)
        cmd_sink.append(cmd)
        if not args.dry_run:
            log = args.results_dir / "logs" / f"calibrate_{label}.log"
            run_cmd(cmd, log)


def phase_main(args, report: dict[str, dict],
               cmd_sink: list[list[str]]) -> list[dict]:
    rows: list[dict] = []
    for label in selected_labels(args, report):
        info = report[label]
        if info["status"] != "runnable":
            rows.append({
                "setting": label, "error_strength": None,
                "method": "(all)", "tau_percentile": None, "num_samples": None,
                "fid": None, "sfid": None, "trigger_rate": None,
                "per_sample_nfe": None, "nfe_total": None, "wall_clock_sec": None,
                "source_npz": None, "source_log": None,
                "notes": f"blocked: {info['unblock_via']}",
            })
            continue
        # [C2] --use-experiment-copy 시 ddim_cifar_siec.py 복사본이
        # --no-use-siec 를 노출하므로 IEC-only fresh run 가능.
        exp_available = (
            getattr(args, "use_experiment_copy", False)
            and (EXP_DIR / "ddim_cifar_siec.py").exists()
        )
        # [C4] fp16 (no-cache, interval_seq=None) 은 과학적으로 S-IEC/IEC 양쪽 모두 no-op 이다.
        # - S-IEC: lookahead 위치가 없어 정의 불가 (deepcache.py:261 에서 NotImplementedError).
        # - IEC author: max_iter=2 gate 가 `cur_i in interval_seq` 인데 interval_seq 가 비어
        #   항상 max_iter=1 로 plain DDIM 과 동치.
        # → fp16 은 단일 "fp16 reference" row 로 축소 (`--no-use-siec` 로 plain DDIM 호출).
        if label == "fp16":
            if not exp_available:
                rows.append({
                    "setting": label, "error_strength": None,
                    "method": "fp16 reference (blocked)",
                    "tau_percentile": None, "num_samples": None,
                    "fid": None, "sfid": None, "trigger_rate": 0.0,
                    "per_sample_nfe": NUM_STEPS, "nfe_total": None,
                    "wall_clock_sec": None, "source_npz": None, "source_log": None,
                    "notes": "fp16 reference requires --use-experiment-copy (C4 실험 복사본)",
                })
                continue
            method = "fp16_ref"
            folder = image_folder_main(label, method, args.num_samples)
            npz = args.results_dir / f"samples_{label}_{method}_n{args.num_samples}.npz"
            needs_run = not (npz.exists() or png_count(folder) >= args.num_samples)
            cmd = build_main_cmd(args, label, info, method)
            if needs_run:
                cmd_sink.append(cmd)
            rows.append({
                "setting": label, "error_strength": None,
                "method": "fp16 reference",
                "tau_percentile": None,
                "num_samples": args.num_samples,
                "fid": None, "sfid": None, "trigger_rate": 0.0,
                "per_sample_nfe": NUM_STEPS, "nfe_total": args.num_samples * NUM_STEPS,
                "wall_clock_sec": None,
                "source_npz": rel(npz), "source_log": None,
                "notes": ("fp16 reference (plain DDIM). IEC/S-IEC 는 no quantization "
                          "regime 에서 no-op 이므로 단일 row 로 보고."),
                "_image_folder": folder, "_npz": npz, "_cmd": cmd,
                "_needs_run": needs_run, "_method_key": method,
            })
            continue
        methods = ["no_correction"]
        if supports_siec(label):
            methods.append("siec")
        if exp_available and label != "W8A8_DC10":
            methods.append("iec")
        for method in methods:
            folder = image_folder_main(label, method, args.num_samples)
            npz = args.results_dir / f"samples_{label}_{method}_n{args.num_samples}.npz"
            needs_run = not (npz.exists() or png_count(folder) >= args.num_samples)
            cmd = build_main_cmd(args, label, info, method)
            if needs_run:
                cmd_sink.append(cmd)
            method_name = {
                "no_correction": "No correction (never)",
                "siec": f"S-IEC p{int(round(args.percentile))}",
                "iec": "IEC (fresh, C2)",
            }[method]
            rows.append({
                "setting": label, "error_strength": None,
                "method": method_name,
                "tau_percentile": (int(round(args.percentile))
                                   if method == "siec" else None),
                "num_samples": args.num_samples,
                "fid": None, "sfid": None, "trigger_rate": None,
                "per_sample_nfe": None, "nfe_total": None, "wall_clock_sec": None,
                "source_npz": rel(npz), "source_log": None,
                "notes": "runnable",
                "_image_folder": folder, "_npz": npz, "_cmd": cmd,
                "_needs_run": needs_run, "_method_key": method,
            })
        # IEC baseline: W8A8_DC10 은 seed row 사용. 다른 runnable 은
        # C2(실험 복사본 --no-use-siec) 승인 시 위에서 이미 처리됨.
        if label == "W8A8_DC10":
            rows.append(seed_row_w8a8_dc10_iec())
            rows.append(seed_row_w8a8_dc10_siec_p80())
        elif not exp_available:
            n_checks = n_checks_for_setting(label)
            rows.append({
                "setting": label, "error_strength": None,
                "method": "IEC (author)", "tau_percentile": None,
                "num_samples": None,
                "fid": None, "sfid": None, "trigger_rate": 0.0,
                "per_sample_nfe": NUM_STEPS + n_checks, "nfe_total": None,
                "wall_clock_sec": None, "source_npz": None, "source_log": None,
                "notes": (f"IEC author NFE = 100 + n_checks ({n_checks}); "
                          "fresh run 은 Candidate C2 (실험 복사본 --no-use-siec) 필요, 미배선."),
            })

    if not args.dry_run:
        for row in rows:
            if not row.get("_needs_run"):
                continue
            log = args.results_dir / "logs" / (
                f"sampling_{row['setting']}_{row['_method_key']}.log")
            row["wall_clock_sec"] = run_cmd(row["_cmd"], log)
    return rows


def phase_fid(rows: list[dict], args) -> None:
    for row in rows:
        folder = row.get("_image_folder")
        npz = row.get("_npz")
        if folder is None or npz is None:
            continue
        if row.get("fid") is not None:
            continue
        if not npz.exists():
            if png_count(folder) == 0:
                row["notes"] = (row.get("notes") or "") + "; sampling output missing"
                continue
            pngs_to_npz(folder, npz)
        log = args.results_dir / "logs" / (
            f"fid_{row['setting']}_{row['_method_key']}.log")
        fid, sfid = (None, None)
        if log.exists():
            text = log.read_text(errors="ignore")
            m_fid = FID_RE.search(text)
            m_sfid = SFID_RE.search(text)
            fid = float(m_fid.group(1)) if m_fid else None
            sfid = float(m_sfid.group(1)) if m_sfid else None
        if fid is None and not args.dry_run:
            fid, sfid = run_fid(npz, log)
        row["fid"] = fid
        row["sfid"] = sfid
        row["source_log"] = rel(log)


def fill_post_hoc(rows: list[dict], args) -> None:
    for row in rows:
        if row.get("error_strength") is None:
            if row.get("method") == "(all)":
                continue
            row["error_strength"] = compute_error_strength(row["setting"])
        method = row.get("method", "")
        if method.startswith("S-IEC p") and row.get("trigger_rate") is None:
            p = row.get("tau_percentile")
            if p is not None:
                tr, nfe, n_checks = postmortem_trigger_nfe(
                    row["setting"], float(p), args.siec_max_rounds
                )
                row["trigger_rate"] = tr
                row["per_sample_nfe"] = nfe
                row["notes"] = ((row.get("notes") or "")
                                + f"; n_checks={n_checks}").lstrip("; ")
                if row.get("num_samples"):
                    row["nfe_total"] = row["num_samples"] * nfe


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
        f.write("# Generated by real_05_robustness.py — review before running.\n")
        f.write(f"cd {IEC_ROOT}\n\n")
        for cmd in cmds:
            f.write(shlex.join(cmd) + "\n")
    path.chmod(0o755)


def plot_two_panel(rows: list[dict], out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_method: dict[str, list[dict]] = {}
    for r in rows:
        if r.get("fid") is None or r.get("error_strength") is None:
            continue
        m = r["method"]
        key = ("No correction" if m.startswith("No correction")
               else "IEC" if m.startswith("IEC")
               else "S-IEC" if m.startswith("S-IEC")
               else m)
        by_method.setdefault(key, []).append(r)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"No correction": "tab:gray", "IEC": "tab:orange", "S-IEC": "tab:blue"}

    for key in ("No correction", "IEC", "S-IEC"):
        items = sorted(by_method.get(key, []), key=lambda r: r["error_strength"])
        if not items:
            continue
        xs = [r["error_strength"] for r in items]
        ys = [r["fid"] for r in items]
        axes[0].plot(xs, ys, "o-", label=key, color=colors[key])

    axes[0].set_xlabel("Error strength")
    axes[0].set_ylabel("FID")
    axes[0].set_title("FID vs Error Strength")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    iec_by_setting = {r["setting"]: r["fid"] for r in by_method.get("IEC", [])
                      if r.get("fid") is not None}
    gains = []
    for r in by_method.get("S-IEC", []):
        base = iec_by_setting.get(r["setting"])
        if base is None:
            continue
        gains.append((r["error_strength"], base - r["fid"], r["setting"]))
    gains.sort()
    if gains:
        xs, ys, labels = zip(*gains)
        axes[1].plot(xs, ys, "o-", color="tab:purple")
        for x, y, lbl in gains:
            axes[1].annotate(lbl, (x, y), textcoords="offset points",
                             xytext=(5, 5), fontsize=8)
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].set_xlabel("Error strength")
    axes[1].set_ylabel("FID(IEC) − FID(S-IEC)")
    axes[1].set_title("Relative gain of S-IEC vs IEC")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


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

    return [{k: cast(v) for k, v in r.items()} for r in raw]


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    defs = setting_defs()
    report = check_assets(defs, args)

    if args.phase == "plot":
        rows = load_csv_rows(args.results_dir / "results.csv")
        plot_two_panel(rows, args.results_dir / "robustness_2panel.png")
        print(f"Replotted {len(rows)} rows → {args.results_dir}")
        return

    write_inventory(report, args.results_dir / "inventory.md")
    if args.phase == "inventory":
        runnable = [k for k, v in report.items() if v["status"] == "runnable"]
        blocked = [k for k, v in report.items() if v["status"] != "runnable"]
        print(f"Inventory written → {args.results_dir/'inventory.md'}")
        print(f"  runnable: {runnable}")
        print(f"  blocked:  {blocked}")
        return

    cmd_list: list[list[str]] = []
    rows: list[dict] = []

    if args.phase in ("pilot", "all"):
        phase_pilot(args, report, cmd_list)
    if args.phase in ("calibrate", "all"):
        phase_calibrate(args, report, cmd_list)
    if args.phase in ("main", "all", "fid"):
        rows = phase_main(args, report, cmd_list)
    if args.phase in ("fid", "all") and not args.dry_run:
        phase_fid(rows, args)

    if rows:
        fill_post_hoc(rows, args)
        save_results(rows, args.results_dir)
    write_commands(cmd_list, args.results_dir / "commands.sh")

    if args.phase == "all" and rows:
        plot_two_panel(rows, args.results_dir / "robustness_2panel.png")

    if args.dry_run:
        print(f"Dry-run: {len(cmd_list)} commands → {args.results_dir/'commands.sh'}")
        if rows:
            print(f"         rows={len(rows)} (csv/json written for inspection)")
        runnable = [k for k, v in report.items() if v["status"] == "runnable"]
        print(f"         runnable settings: {runnable}")
        return
    print(f"Done. Results in {args.results_dir}")


if __name__ == "__main__":
    main()
