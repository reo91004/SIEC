#!/usr/bin/env python3
"""Phase 3 (M2) sanity check for speculative lookahead commit/discard logic.

Run from S-IEC repo root:
    python experiments/yongseong/verify_phase3.py --run

Invariants checked (per the redesign plan §Phase 3, with cur_f propagation):
  Y1. New trace fields exist:
      memo_committed_per_step, memo_discarded_per_step, lookahead_cache_mode_per_step.
  Y2. NFE accounting: reuse-on total NFE < reuse-off total NFE on the same trigger
      pattern, AND reuse-on memo_hit count > 0.
  Y3. Causality (no false hits):
      For every step k, if memo_hit[k] == True then memo_committed[k-1] == True.
  Y4. Discard correctness:
      For every step k, if memo_discarded[k] == True then memo_hit[k+1] == False
      (the speculative prediction must not be reused after correction).
  Y5. always_correct ⇒ no commits:
      With --siec_always_correct, every checked step has memo_committed == False.
  Y6. always_correct ⇒ all eligible discards:
      With --siec_always_correct AND --reuse_lookahead, every checked step has
      memo_discarded == True (commits at refresh boundaries are now allowed too,
      so eligibility = checked).
  Y7. lookahead_cache_mode is sane:
      reuse_on:  "full" iff next_refresh, "quick" iff (checked & ¬next_refresh),
                  "none" iff ¬checked.
      reuse_off: every checked step is "full".
  Y8. cur_f propagation:
      Refresh-step memo_hit must have been committed at the predecessor step with
      lookahead_cache_mode == "full" (so prv_f_after_commit is a fresh full-path
      anchor). Equivalent: ∀k. (memo_hit[k] & refresh_step[k]) ⇒ lookahead_cache_mode[k-1]=="full".

Auto re-execs under conda when torch is unavailable in the parent shell.
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
TRACE_DIR = IEC_ROOT / "experiments/yongseong/results/verify_phase3/traces"
LOG_DIR = IEC_ROOT / "experiments/yongseong/results/verify_phase3/logs"
IMG_DIR = IEC_ROOT / "experiments/yongseong/results/verify_phase3/images"


def conda_python(cuda_visible: str | None) -> list[str]:
    prefix: list[str] = []
    if cuda_visible:
        prefix = ["env", f"CUDA_VISIBLE_DEVICES={cuda_visible}"]
    return prefix + ["conda", "run", "--no-capture-output", "-n", "iec", "python"]


def build_cmd(slug: str, *, reuse: bool, always_correct: bool,
              cuda_visible: str | None, num_samples: int = 8,
              seed: int = 42) -> tuple[list[str], Path]:
    trace_path = TRACE_DIR / f"{slug}.pt"
    img_folder = IMG_DIR / slug
    cmd = conda_python(cuda_visible) + [
        "mainddpm/ddim_cifar_siec.py",
        "--correction-mode", "siec",
        "--num_samples", str(num_samples),
        "--sample_batch", str(num_samples),
        "--seed", str(seed),
        "--weight_bit", "8",
        "--act_bit", "8",
        "--replicate_interval", "10",
        "--image_folder", str(img_folder),
        "--siec_max_rounds", "1",
        "--siec_return_trace",
        "--siec_trace_mode", "siec",
        "--siec_trace_out", str(trace_path),
    ]
    if reuse:
        cmd.append("--reuse_lookahead")
    if always_correct:
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


def fmt_pass(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def load_traces(path):
    import torch
    obj = torch.load(path, weights_only=False, map_location="cpu")
    return obj if isinstance(obj, list) else [obj]


# ---------------------------------------------------------------------------

def check_fields(label: str, traces: list[dict]) -> int:
    needed = ("memo_committed_per_step", "memo_discarded_per_step",
              "lookahead_cache_mode_per_step")
    fails = 0
    for ti, t in enumerate(traces):
        for key in needed:
            if key not in t:
                print(f"  FAIL [{label}/trace#{ti}] missing key: {key}")
                fails += 1
        T = len(t.get("step_idx_per_step", []))
        for key in needed:
            if key in t and len(t[key]) != T:
                print(f"  FAIL [{label}/trace#{ti}] {key} len={len(t[key])} (expected {T})")
                fails += 1
    if fails == 0:
        print(f"  PASS {label} has all 3 new fields with correct length")
    return fails


def total_nfe(traces: list[dict]) -> int:
    return sum(sum(t.get("nfe_per_step", [])) for t in traces)


def total_memo_hits(traces: list[dict]) -> int:
    return sum(sum(t.get("memo_hit_per_step", [])) for t in traces)


def causality_no_false_hits(traces: list[dict]) -> tuple[int, int]:
    """For all k>=1, memo_hit[k] => memo_committed[k-1]. Returns (violations, total_hits)."""
    viol = 0
    total = 0
    for t in traces:
        h = t.get("memo_hit_per_step", [])
        c = t.get("memo_committed_per_step", [])
        for k in range(1, len(h)):
            if h[k]:
                total += 1
                if k - 1 >= len(c) or not c[k - 1]:
                    viol += 1
    return viol, total


def discard_correctness(traces: list[dict]) -> tuple[int, int]:
    """For all k<T-1, memo_discarded[k] => memo_hit[k+1] == False. Returns (violations, total_discards)."""
    viol = 0
    total = 0
    for t in traces:
        d = t.get("memo_discarded_per_step", [])
        h = t.get("memo_hit_per_step", [])
        for k in range(len(d) - 1):
            if d[k]:
                total += 1
                if k + 1 < len(h) and h[k + 1]:
                    viol += 1
    return viol, total


def always_correct_invariants(traces: list[dict]) -> tuple[int, int, int, int]:
    """For always_correct + reuse_on runs: returns (commits, discards, eligible, T_total).

    Eligibility for discard = checked (after Phase 3 cur_f propagation, refresh
    boundaries are now committable too).
    """
    commits = 0
    discards = 0
    eligible = 0
    T_total = 0
    for t in traces:
        ck = t.get("checked_per_step", [])
        com = t.get("memo_committed_per_step", [])
        dis = t.get("memo_discarded_per_step", [])
        T = len(ck)
        T_total += T
        for k in range(T):
            commits += int(bool(com[k]))
            discards += int(bool(dis[k]))
            if ck[k]:
                eligible += 1
    return commits, discards, eligible, T_total


def cur_f_propagation_invariant(traces: list[dict]) -> tuple[int, int]:
    """Y8: refresh-step memo_hit ⇒ predecessor lookahead_cache_mode == 'full'.

    Returns (violations, refresh_hits)."""
    viol = 0
    rh = 0
    for t in traces:
        h = t.get("memo_hit_per_step", [])
        rs = t.get("refresh_step_per_step", [])
        lcm = t.get("lookahead_cache_mode_per_step", [])
        T = len(h)
        for k in range(1, T):
            if h[k] and rs[k]:
                rh += 1
                if k - 1 >= len(lcm) or lcm[k - 1] != "full":
                    viol += 1
    return viol, rh


def cache_mode_sanity(traces: list[dict], reuse: bool) -> tuple[int, dict]:
    """Verify lookahead_cache_mode against the run's reuse setting.

    Reuse-on:  'full' only when next_refresh; 'quick' otherwise (when checked).
    Reuse-off: every checked step must be 'full' (no quick path possible).
    Both:      'none' when not checked.
    Returns (violations, mode_counts).
    """
    viol = 0
    counts = {"full": 0, "quick": 0, "none": 0, "other": 0}
    for t in traces:
        lcm = t.get("lookahead_cache_mode_per_step", [])
        rs = t.get("refresh_step_per_step", [])
        ck = t.get("checked_per_step", [])
        T = len(lcm)
        for k in range(T):
            mode = lcm[k]
            counts[mode if mode in counts else "other"] += 1
            next_refresh = (k + 1 < T and bool(rs[k + 1])) or (k + 1 >= T)
            if not ck[k]:
                if mode != "none":
                    viol += 1
                continue
            if reuse:
                if next_refresh and mode != "full":
                    viol += 1
                if (not next_refresh) and mode != "quick":
                    viol += 1
            else:
                if mode != "full":
                    viol += 1
    return viol, counts


# ---------------------------------------------------------------------------

def verify_all(off_path: Path, on_path: Path, always_path: Path) -> int:
    fails = 0

    print("\n=== Y1. New trace fields exist (3 required) ===")
    for label, p in [("siec_off", off_path), ("siec_on", on_path), ("siec_always", always_path)]:
        if not p.exists():
            print(f"  FAIL {label} trace missing: {p}")
            fails += 1
            continue
        traces = load_traces(p)
        fails += check_fields(label, traces)

    print("\n=== Y2. NFE accounting: reuse-on < reuse-off, memo_hits > 0 ===")
    if off_path.exists() and on_path.exists():
        off_traces = load_traces(off_path)
        on_traces = load_traces(on_path)
        nfe_off = total_nfe(off_traces)
        nfe_on = total_nfe(on_traces)
        hits_on = total_memo_hits(on_traces)
        ok_nfe = nfe_on < nfe_off
        ok_hit = hits_on > 0
        print(f"  {fmt_pass(ok_nfe)} nfe_on={nfe_on} < nfe_off={nfe_off}")
        print(f"  {fmt_pass(ok_hit)} memo_hits_on={hits_on} > 0")
        if not ok_nfe:
            fails += 1
        if not ok_hit:
            fails += 1

    print("\n=== Y3. Causality: memo_hit[k] ⇒ memo_committed[k-1] ===")
    for label, p in [("siec_on", on_path), ("siec_always", always_path)]:
        if not p.exists():
            continue
        traces = load_traces(p)
        viol, total = causality_no_false_hits(traces)
        ok = viol == 0
        print(f"  {fmt_pass(ok)} {label}: violations={viol}/{total}")
        if not ok:
            fails += 1

    print("\n=== Y4. Discard correctness: memo_discarded[k] ⇒ memo_hit[k+1]==False ===")
    for label, p in [("siec_on", on_path), ("siec_always", always_path)]:
        if not p.exists():
            continue
        traces = load_traces(p)
        viol, total = discard_correctness(traces)
        ok = viol == 0
        print(f"  {fmt_pass(ok)} {label}: violations={viol}/{total}")
        if not ok:
            fails += 1

    print("\n=== Y5/Y6. always_correct: commits=0, discards == eligible ===")
    if always_path.exists():
        traces = load_traces(always_path)
        commits, discards, eligible, T = always_correct_invariants(traces)
        ok5 = commits == 0
        ok6 = discards == eligible
        print(f"  {fmt_pass(ok5)} commits={commits} (expected 0)  T_total={T}")
        print(f"  {fmt_pass(ok6)} discards={discards} == eligible(checked)={eligible}")
        if not ok5:
            fails += 1
        if not ok6:
            fails += 1

    print("\n=== Y8. cur_f propagation: refresh memo_hit ⇒ predecessor 'full' lookahead ===")
    for label, p in [("siec_on", on_path)]:
        if not p.exists():
            continue
        traces = load_traces(p)
        viol, rh = cur_f_propagation_invariant(traces)
        ok = viol == 0
        print(f"  {fmt_pass(ok)} {label}: violations={viol}/{rh} refresh_hits")
        if not ok:
            fails += 1

    print("\n=== Y7. lookahead_cache_mode sanity ===")
    for label, p, reuse in [("siec_off", off_path, False), ("siec_on", on_path, True),
                             ("siec_always", always_path, True)]:
        if not p.exists():
            continue
        traces = load_traces(p)
        viol, counts = cache_mode_sanity(traces, reuse=reuse)
        ok = viol == 0
        print(f"  {fmt_pass(ok)} {label} (reuse={reuse}): counts={counts} violations={viol}")
        if not ok:
            fails += 1

    return fails


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="store_true",
                   help="Execute the 3 sampler runs before verifying.")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cuda-visible-devices",
                   default=os.environ.get("CUDA_VISIBLE_DEVICES", "2"))
    args = p.parse_args()

    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    runs = [
        ("p3_siec_off",    False, False),
        ("p3_siec_on",     True,  False),
        ("p3_siec_always", True,  True),
    ]
    paths: dict[str, Path] = {}
    for slug, reuse, always in runs:
        cmd, tp = build_cmd(slug, reuse=reuse, always_correct=always,
                            cuda_visible=args.cuda_visible_devices,
                            num_samples=args.num_samples, seed=args.seed)
        paths[slug] = tp
        if args.run:
            rc = run_cmd(cmd, LOG_DIR / f"{slug}.log")
            if rc != 0:
                print(f"!! sampler failed for {slug}; aborting.")
                return rc

    # Re-exec under conda if torch missing in current interpreter (after sampling).
    try:
        import torch  # noqa: F401
    except ImportError:
        new_argv = [a for a in sys.argv if a != "--run"]
        relaunch = conda_python(args.cuda_visible_devices) + new_argv
        os.execvp(relaunch[0], relaunch)

    fails = verify_all(paths["p3_siec_off"], paths["p3_siec_on"], paths["p3_siec_always"])
    print(f"\n=== Summary: {fails} failure(s) ===")
    return 1 if fails > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
