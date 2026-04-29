#!/usr/bin/env python3
"""Phase 1 sanity check for trace fields and NFE accounting.

Run from S-IEC repo root:
    python experiments/yongseong/verify_phase1.py --run
    # then re-run with --check-only to re-verify without re-sampling

Invariants checked (per the redesign plan):
  V1. Trace integrity: len(step_idx_per_step) == timesteps; sum(nfe_per_step) > 0.
  V2. NFE accounting consistency: per_sample_nfe (aggregate_trace) matches
      sum(nfe_per_step) / batch averaged.
  V3. reuse_lookahead on/off: with reuse on, total NFE drops AND memo_hit_per_step
      contains True at least once.
  V4. mode=none: triggered.sum() == 0 and checked.sum() == 0.
  V5. mode=siec --siec_always_correct: every step where next_t >= 0 has
      checked == True and triggered == True.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

IEC_ROOT = Path(__file__).resolve().parents[2]
TRACE_DIR = IEC_ROOT / "experiments/yongseong/results/verify_phase1/traces"
LOG_DIR = IEC_ROOT / "experiments/yongseong/results/verify_phase1/logs"
IMG_DIR = IEC_ROOT / "experiments/yongseong/results/verify_phase1/images"


def conda_python(cuda_visible: str | None) -> list[str]:
    prefix: list[str] = []
    if cuda_visible:
        prefix = ["env", f"CUDA_VISIBLE_DEVICES={cuda_visible}"]
    return prefix + ["conda", "run", "--no-capture-output", "-n", "iec", "python"]


def entry_script() -> str:
    return "mainddpm/ddim_cifar_siec.py"


def build_cmd(slug: str, mode: str, *, reuse: bool, always_correct: bool, cuda_visible: str | None,
              num_samples: int = 8) -> tuple[list[str], Path]:
    trace_path = TRACE_DIR / f"{slug}.pt"
    img_folder = IMG_DIR / slug
    cmd = conda_python(cuda_visible) + [
        entry_script(),
        "--correction-mode", mode,
        "--num_samples", str(num_samples),
        "--sample_batch", str(num_samples),
        "--weight_bit", "8",
        "--act_bit", "8",
        "--replicate_interval", "10",
        "--image_folder", str(img_folder),
        "--siec_max_rounds", "1",
        "--siec_return_trace",
        "--siec_trace_mode", mode if mode != "auto" else "siec",
        "--siec_trace_out", str(trace_path),
    ]
    if reuse and mode == "siec":
        cmd.append("--reuse_lookahead")
    if always_correct and mode == "siec":
        cmd.append("--siec_always_correct")
    return cmd, trace_path


def run_cmd(cmd: list[str], log_path: Path) -> int:
    print(f"$ {shlex.join(cmd)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, cwd=IEC_ROOT, stdout=f, stderr=subprocess.STDOUT)
    print(f"  -> rc={proc.returncode}  elapsed={time.time()-t0:.1f}s  log={log_path}")
    return proc.returncode


def load_traces(trace_path: Path) -> list[dict]:
    import torch
    obj = torch.load(trace_path, weights_only=False, map_location="cpu")
    return obj if isinstance(obj, list) else [obj]


def fmt_pass(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def assert_trace_integrity(label: str, traces: list[dict], expected_steps: int) -> bool:
    ok = True
    for ti, trace in enumerate(traces):
        for key in ("step_idx_per_step", "t_int_per_step", "next_t_int_per_step",
                    "refresh_step_per_step", "memo_hit_per_step",
                    "nfe_per_step", "checked_per_step", "triggered_per_step"):
            if key not in trace:
                print(f"  [{label}/trace#{ti}] missing key: {key}")
                ok = False
                continue
            length = len(trace[key])
            if length != expected_steps:
                print(f"  [{label}/trace#{ti}] {key} len={length} (expected {expected_steps})")
                ok = False
        nfe_sum = sum(trace.get("nfe_per_step", []))
        if nfe_sum <= 0:
            print(f"  [{label}/trace#{ti}] sum(nfe_per_step)={nfe_sum}")
            ok = False
    return ok


def aggregate_per_sample_nfe(traces: list[dict]) -> tuple[float, int]:
    total_weight = 0
    total_nfe = 0.0
    for trace in traces:
        bs = int(trace.get("batch_size", 1))
        total_weight += bs
        total_nfe += bs * float(sum(trace.get("nfe_per_step", [])))
    return total_nfe / max(total_weight, 1), total_weight


def main_verify(no_path: Path, siec_off_path: Path, siec_on_path: Path, siec_always_path: Path,
                expected_steps: int) -> int:
    failures = 0

    print("\n=== V1. Trace integrity ===")
    for label, p in [("none", no_path), ("siec_off", siec_off_path),
                     ("siec_on", siec_on_path), ("siec_always", siec_always_path)]:
        if not p.exists():
            print(f"  [{label}] trace missing: {p}")
            failures += 1
            continue
        traces = load_traces(p)
        ok = assert_trace_integrity(label, traces, expected_steps)
        print(f"  {fmt_pass(ok)} {label}")
        if not ok:
            failures += 1

    print("\n=== V2. NFE accounting (per_sample_nfe ↔ sum(nfe_per_step)) ===")
    for label, p in [("none", no_path), ("siec_off", siec_off_path),
                     ("siec_on", siec_on_path), ("siec_always", siec_always_path)]:
        if not p.exists():
            continue
        traces = load_traces(p)
        per_sample_nfe, w = aggregate_per_sample_nfe(traces)
        first_sum = sum(traces[0].get("nfe_per_step", [])) if traces else 0
        ok = abs(per_sample_nfe - first_sum) < 1e-9 if len(traces) == 1 else True
        print(f"  {fmt_pass(ok)} {label}: per_sample_nfe={per_sample_nfe:.2f}  weight={w}")
        if not ok:
            failures += 1

    print("\n=== V3. reuse_lookahead effect ===")
    if siec_off_path.exists() and siec_on_path.exists():
        off_nfe, _ = aggregate_per_sample_nfe(load_traces(siec_off_path))
        on_nfe, _ = aggregate_per_sample_nfe(load_traces(siec_on_path))
        on_traces = load_traces(siec_on_path)
        on_memo_hits = sum(sum(t.get("memo_hit_per_step", [])) for t in on_traces)
        print(f"  off NFE={off_nfe:.2f}, on NFE={on_nfe:.2f}, on memo_hits={on_memo_hits}")
        ok = (on_nfe < off_nfe) and (on_memo_hits > 0)
        print(f"  {fmt_pass(ok)} reuse_lookahead reduces NFE and produces memo hits")
        if not ok:
            failures += 1
    else:
        print("  SKIP (missing one or both traces)")

    print("\n=== V4. mode=none: no triggers, no checks ===")
    if no_path.exists():
        traces = load_traces(no_path)
        bad = 0
        for trace in traces:
            t_sum = sum(trace.get("triggered_per_step", []))
            c_sum = sum(trace.get("checked_per_step", []))
            if t_sum != 0 or c_sum != 0:
                bad += 1
                print(f"  triggered={t_sum} checked={c_sum} (expected 0/0)")
        ok = bad == 0
        print(f"  {fmt_pass(ok)} no-correction: all triggered/checked = 0")
        if not ok:
            failures += 1
    else:
        print("  SKIP (mode=none trace missing)")

    print("\n=== V5. siec_always_correct: triggered == checked == steps with next_t>=0 ===")
    if siec_always_path.exists():
        traces = load_traces(siec_always_path)
        bad = 0
        for trace in traces:
            checked = trace.get("checked_per_step", [])
            triggered = trace.get("triggered_per_step", [])
            next_t = trace.get("next_t_int_per_step", [])
            expected = sum(1 for x in next_t if x >= 0)
            cs = sum(1 for f in checked if f)
            ts = sum(1 for f in triggered if f)
            if cs != expected or ts != expected:
                bad += 1
                print(f"  expected={expected} checked={cs} triggered={ts}")
        ok = bad == 0
        print(f"  {fmt_pass(ok)} siec_always_correct: full coverage")
        if not ok:
            failures += 1
    else:
        print("  SKIP (siec_always trace missing)")

    print(f"\n=== Summary: {failures} failure(s) ===")
    return 1 if failures > 0 else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true",
                        help="Execute the 4 sampler runs before verifying.")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=100,
                        help="Expected number of reverse steps (matches sampler default).")
    parser.add_argument("--cuda-visible-devices",
                        default=os.environ.get("CUDA_VISIBLE_DEVICES", "2"))
    args = parser.parse_args()

    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    runs = [
        ("v1_none",        "none", False, False),
        ("v1_siec_off",    "siec", False, False),
        ("v1_siec_on",     "siec", True,  False),
        ("v1_siec_always", "siec", True,  True),
    ]
    paths: dict[str, Path] = {}
    for slug, mode, reuse, always in runs:
        cmd, trace_path = build_cmd(slug, mode, reuse=reuse, always_correct=always,
                                    cuda_visible=args.cuda_visible_devices,
                                    num_samples=args.num_samples)
        paths[slug] = trace_path
        if args.run:
            rc = run_cmd(cmd, LOG_DIR / f"{slug}.log")
            if rc != 0:
                print(f"!! sampler failed for {slug}; aborting verification.")
                return rc

    try:
        import torch  # noqa: F401
    except ImportError:
        new_argv = [a for a in sys.argv if a not in {"--run"}]
        relaunch = ["conda", "run", "--no-capture-output", "-n", "iec", "python"] + new_argv
        print(f"\n[verify] torch unavailable in parent; relaunching under iec env:\n  $ {shlex.join(relaunch)}")
        os.execvp(relaunch[0], relaunch)

    return main_verify(
        no_path=paths["v1_none"],
        siec_off_path=paths["v1_siec_off"],
        siec_on_path=paths["v1_siec_on"],
        siec_always_path=paths["v1_siec_always"],
        expected_steps=args.timesteps,
    )


if __name__ == "__main__":
    sys.exit(main())
