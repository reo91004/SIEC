#!/usr/bin/env python3
"""calibrate_tau_reuse — reuse_lookahead 전용 τ schedule 생성.

기존 `calibration/tau_schedule_<setting>_p80.pt` 는 `reuse_lookahead=False`
(=fresh lookahead) 의 syndrome score 분포에서 p80 으로 컷한 것이라, reuse
path 에 그대로 쓰면 cache approximation 이 더한 노이즈 때문에 점수 분포가
1.5–2x 우측으로 옮겨가 trigger 율이 폭증한다 (exp_D 의 측정치 4% → 44%).

이 스크립트는 reuse 가 켜진 상태에서 측정된 trace.pt 의 step 별 score
분포에서 다시 percentile 컷을 잡아 새 τ 파일을 만든다. exp_D 의 5번째
방법 (`siec_reuse_recalib`) 이 이 파일을 참조한다.

Usage:
    python experiments/yongseong/framing/calibrate_tau_reuse.py \
        --trace experiments/yongseong/results/exp_D_lookahead/<run>/traces/w8a8_siec_reuse.pt \
        --out calibration/tau_schedule_w8a8_reuse_p80.pt \
        --percentile 80
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent.parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import framing_common as fc  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trace", type=Path, required=True, help="reuse_lookahead 가 켜진 상태로 만든 trace.pt")
    p.add_argument("--out", type=Path, required=True, help="새 τ schedule 출력 경로")
    p.add_argument("--percentile", type=float, default=80.0)
    p.add_argument("--reference-tau", type=Path, default=None,
                   help="기존 τ 파일 (shape/dtype 맞추기용; 없으면 trace 분포 길이 그대로).")
    p.add_argument("--floor", type=float, default=0.0,
                   help="τ 의 하한 (마지막 step 처럼 score=0 인 곳을 0 으로 두면 항상 trigger 해버림).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import numpy as np
    import torch

    if not args.trace.exists():
        raise SystemExit(f"trace not found: {args.trace}")
    traces = fc.load_traces(args.trace)
    scores = fc.stack_score_values(traces)  # (N, T)
    if scores.size == 0:
        raise SystemExit("trace has no score_values_per_step")
    N, T = scores.shape
    tau = np.percentile(scores, args.percentile, axis=0)  # (T,)

    if args.reference_tau and args.reference_tau.exists():
        ref = torch.load(args.reference_tau, map_location="cpu", weights_only=False)
        if torch.is_tensor(ref):
            ref_len = int(ref.numel())
            if ref_len != T:
                # The original schedules drop the very last step (next_t < 0),
                # giving length T-1 = 99 for the canonical 100-step config.
                tau = tau[:ref_len]
                print(f"[info] truncated τ from {T} to {ref_len} to match {args.reference_tau}")
            tau = np.asarray(tau, dtype=np.float64 if ref.dtype == torch.float64 else np.float32)
            tau_t = torch.tensor(tau, dtype=ref.dtype)
        else:
            raise SystemExit(f"unsupported reference τ format: {type(ref)}")
    else:
        tau_t = torch.tensor(tau, dtype=torch.float64)

    if args.floor > 0:
        tau_t = torch.clamp(tau_t, min=args.floor)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tau_t, args.out)
    print(f"[ok] {args.out}  shape={tuple(tau_t.shape)} dtype={tau_t.dtype}")
    print(f"[ok] τ first 5: {tau_t[:5].tolist()}")
    print(f"[ok] τ last 5:  {tau_t[-5:].tolist()}")
    print(f"[ok] source trace: {fc.rel(args.trace)}  (n_batches={len(traces)} N={N} T={T})")


if __name__ == "__main__":
    main()
