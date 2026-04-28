# 실험용 수정 로그 (`Semantic/S-IEC/`)

이 폴더는 **실험 4/5 전용 공간 + S-IEC 재설계 작업 영역**이다.

> **2026-04-28 KST — 폴더 이전 / 정책 갱신**
>
> 이 로그는 원래 `IEC/experiments/yongseong/` 안의 flat 미러 변경분을 추적하기 위해 만들어졌다.
> 2026-04-28에 `Semantic/S-IEC/` 로 작업 영역이 이전되었다 (하단 `[EXP-RELOC]` 항목 참조).
>
> **갱신된 정책**
> - `IEC/{mainddpm, siec_core, quant, mainldm}/` 1저자 원본은 **여전히 한 글자도 수정하지 않는다** — IEC repo 는 frozen.
> - `Semantic/S-IEC/{mainddpm, siec_core, quant}/` 는 1저자 코드의 **사본** 이며, 이 사본 안에서는 자유롭게 재설계가 가능하다.
> - 1저자에게 이식할 때는 `S-IEC/experiments/yongseong/` 폴더를 제외한 나머지를 그대로 전달한다 — 폴더 위치 자체가 "이건 baseline / 이건 용성 실험" 의 신호.
> - 변경 라인은 계속 `# [EXP-CN]`, `# [EXP-FRAMING-X]` 같은 태그로 마킹해 1저자가 diff 로 추적할 수 있게 한다.

## 철학

1저자 core 코드(`IEC/mainddpm/`, `IEC/siec_core/`, `IEC/quant/`, `IEC/mainldm/`)는 **한 글자도 수정하지 않는다**. 수정이 필요한 경우 해당 파일을 이 폴더에 복사한 뒤 복사본만 수정한다. 따라서 IEC/ 의 변경은 어느 경우에도 발생하지 않는다.

## 폴더 구조

```
experiments/yongseong/
├── CHANGES.md               (이 파일; append-only)
├── real_04_tradeoff.py      (실험 4 wrapper)
├── real_05_robustness.py    (실험 5 wrapper)
├── <복사본 파일들>.py        (수정이 필요할 때만 생성)
└── results/
    ├── real_04_tradeoff/
    └── real_05_robustness/
```

수정 대상이 다른 파일을 import해야 한다면 그 import 체인의 최소 부분 집합만 복사한다. 복사본은 상단에 다음 헤더 주석을 둔다.

```python
# [EXP] <원본 절대 경로>의 실험용 복사본.
# 원본은 수정하지 않음. 아래 수정 부분은 # [EXP-CN] 태그로 표시.
# sys.path shim 으로 이 실험 내에서만 원본을 가린다.
```

## 기록 형식

복사본 파일을 하나 만들거나 수정할 때마다 이 파일 하단 "항목" 섹션에 아래 블록을 append한다. 롤백해도 항목을 지우지 말고 `> reverted on YYYY-MM-DD KST — 이유` 한 줄을 덧붙인다.

```
### YYYY-MM-DD KST — <실험 ID: EXP-4 | EXP-5> — <experiments/yongseong/ 내부 상대경로>

- **원본**: IEC/.../<원본 경로>
- **변경 요지**: (1–3줄)
- **이유**: (wrapper로 대체 불가한 이유)
- **주석 태그**: `# [EXP-4]` 또는 `# [EXP-5-ROBUSTNESS]`
- **import 영향**: 이 복사본이 shadow하는 모듈 + 상위에서 이 복사본을 호출하는 파일
- **롤백**: 이 복사본을 삭제하면 완료 (원본은 건드리지 않음)
- **관련 Candidate**: C1 / C2 / C3 / C4 (plan 파일 참조)
```

## 항목

### 2026-04-23 KST — EXP-4 / EXP-5 — `ddim_cifar_siec.py`

- **원본**: `IEC/mainddpm/ddim_cifar_siec.py`
- **변경 요지**:
  - `sys.path.insert(0, <이 폴더>)` 추가로 `from deepcache import Diffusion` 가 이 폴더의 flat 복사본을 import 하게 함.
  - C1: `--trigger_mode {syndrome,random,uniform}`, `--trigger_prob`, `--trigger_period` CLI 추가.
  - C2: `--no-use-siec` (dest=`use_siec`, action=`store_false`).
  - C3: `--siec_return_trace`, `--siec_trace_out`, `--siec_trace_mode {iec,siec}` + 실행 후 trace dict를 `.pt`로 저장.
  - C4: `--no-cache`, `--no-ptq` CLI 추가. `--no-cache` 시 DeepCache calibration/a_list/b_list load skip, `interval_seq=None`.
- **이유**: (C1) plan v1에 따라 Random/Uniform trigger baseline이 논문 §8에서 요구됨; (C2) IEC-only fresh run에 필요; (C3) post-mortem NFE 추정을 실측으로 대체; (C4) 실험 5 Setting 1 (fp16).
- **주석 태그**: `# [EXP]` (헤더/shim/import), `# [EXP-C1]`, `# [EXP-C2]`, `# [EXP-C3]`, `# [EXP-C4]`.
- **import 영향**: `mainddpm/ddim_cifar_siec.py` 는 shadow 대상이 아님. 이 복사본은 wrapper의 `--use-experiment-copy` 플래그가 켜진 경우에만 subprocess 호출됨. `from ddpm.runners.deepcache import Diffusion` → `from deepcache import Diffusion` (이 폴더 flat).
- **롤백**: `experiments/yongseong/ddim_cifar_siec.py` 삭제. 원본은 무변경.
- **관련 Candidate**: C1 / C2 / C3 / C4.

### 2026-04-23 KST — EXP-4 / EXP-5 — `deepcache.py`

- **원본**: `IEC/mainddpm/ddpm/runners/deepcache.py`
- **변경 요지**:
  - 상대 import (`from ..models.ema import EMAHelper` 등)를 절대 import (`from ddpm.models.ema import EMAHelper`)로 재작성.
  - `from ..functions.deepcache_denoising import X` → `from deepcache_denoising import X` (이 폴더 flat 라우팅).
  - C1: `adaptive_generalized_steps_siec` 호출 시 `args.trigger_mode/prob/period` 전달.
  - C3: `args.siec_return_trace` 일 때 `adaptive_generalized_steps_trace` 호출 및 `args._siec_traces` 누적.
  - C4: `creat_model`에서 `args.cache=False` 허용 → `ddpm.models.diffusion.Model` 로드.
- **이유**: import 경로 재작성은 flat 복사본이 shadow로 동작하기 위해 필수. C1/C3/C4는 해당 플래그를 sampling 함수까지 전달하기 위함.
- **주석 태그**: `# [EXP]`, `# [EXP-C1]`, `# [EXP-C3]`, `# [EXP-C4]`.
- **import 영향**: 실험 복사본 `ddim_cifar_siec.py` 가 `from deepcache import Diffusion` 으로만 이 파일을 import. `mainddpm/ddpm/runners/deepcache.py` 의 다른 caller는 영향 없음.
- **롤백**: `experiments/yongseong/deepcache.py` 삭제.
- **관련 Candidate**: C1 / C3 / C4.

### 2026-04-23 KST — EXP-4 — `deepcache_denoising.py`

- **원본**: `IEC/mainddpm/ddpm/functions/deepcache_denoising.py`
- **변경 요지**:
  - `adaptive_generalized_steps_siec` 시그니처에 `trigger_mode`, `trigger_prob`, `trigger_period` 추가.
  - C1: syndrome-check gate 안에서 `trigger_mode=="random"/"uniform"` 일 때 threshold 결정을 대체 (per-step NFE 구조는 유지).
  - 기존 `adaptive_generalized_steps_trace` 함수는 수정 없음 (C3는 상위 CLI만 노출).
- **이유**: plan §8 Random/Uniform trigger baseline.
- **주석 태그**: `# [EXP]`, `# [EXP-C1]`.
- **import 영향**: `experiments/yongseong/deepcache.py` 가 `from deepcache_denoising import ...` 로 이 파일을 flat import. 원본은 그대로 `mainddpm/ddpm/functions/deepcache_denoising.py` 경로로 남아 IEC baseline 호출에 영향 없음.
- **롤백**: `experiments/yongseong/deepcache_denoising.py` 삭제.
- **관련 Candidate**: C1 (C3는 이 파일에서 기존 함수를 그대로 사용).

### 2026-04-23 KST — EXP-4 / EXP-5 — bug-fix pass (외부 리뷰 대응)

외부 5-point 리뷰에서 지적된 정합성 문제 3건을 fix. 파일별 변경은 아래 세 블록.

#### (bug-1) 실험 5 tau 경로 불일치 — `real_05_robustness.py`

- **원본**: (wrapper 자체; `IEC/mainddpm/` 건드리지 않음)
- **변경 요지**:
  - `_ensure_fallback_link(target, fallback, ...)` 헬퍼 추가. `W8A8_DC10` 라벨에 대해
    `pilot_scores_W8A8_DC10.pt` ← `pilot_scores_nb.pt`,
    `tau_schedule_W8A8_DC10_p{P}.pt` ← `tau_schedule_p{P}.pt` symlink 생성 시 `build_main_cmd` 의 `--tau_path` 와 일치.
  - `supports_siec(label)` 헬퍼로 fp16 에서 S-IEC/pilot/calibrate phase 를 skip (bug-2 와 함께 반영).
- **이유**: 기존 로직은 기존 tau/pilot 파일이 있으면 calibrate phase 가 skip 되는데, `build_main_cmd` 는 라벨 prefixed 경로를 요구하여 실제 실행 시 `FileNotFoundError` 를 일으켰다. 이름이 맞도록 symlink.
- **주석 태그**: wrapper 내부이므로 `# [EXP]` 계열 태그는 미사용 (원본 코드 수정 아님).
- **import 영향**: 없음.
- **롤백**: `_ensure_fallback_link` / `supports_siec` 함수 제거 + 호출부 복원.
- **관련 Candidate**: 해당 없음 (wrapper 버그).

#### (bug-2) fp16 경로에서 S-IEC silent fallback 방지 — `deepcache.py`

- **원본**: `IEC/mainddpm/ddpm/runners/deepcache.py` (실험 복사본)
- **변경 요지**:
  - `interval_seq is None` (C4 `--no-cache`) 분기에서 `args.use_siec=True` 이면 `NotImplementedError` raise.
  - 이유 메시지: fp16 / no-cache path 는 lookahead 위치(`interval_seq`)를 갖지 않아 S-IEC 가 정의되지 않음.
- **이유**: fp16 runnable 로 승격하면서 S-IEC 를 요구하는 run 이 오면 silent 하게 IEC 경로로 흘러가던 버그. 명시적 실패로 변경하고 wrapper 에서는 `supports_siec(label)==False` 일 때 해당 phase 를 애초에 건너뛴다.
- **주석 태그**: `# [EXP-C4]`.
- **import 영향**: 원본 경로는 영향 없음. wrapper 가 `--use-experiment-copy` 일 때만 이 파일 shadow.
- **롤백**: 해당 raise 블록 제거.
- **관련 Candidate**: C4.

#### (bug-3) post-mortem NFE 공식 정정 — `real_04_tradeoff.py`

- **원본**: (wrapper 자체)
- **변경 요지**:
  - `postmortem()` 의 반환식을 `num_steps + rounds * sum_rate` 에서
    `num_steps + n_checks + max(0, rounds - 1) * sum_rate` 로 정정.
    추가로 n_checks (|interval_seq|) 도 반환하여 `fill_postmortem` 이 notes 에 기록.
  - `n_checks_default()` 헬퍼 추가: pilot_scores 의 비어있지 않은 bin 수를 센다.
  - `build_sweep_rows` 의 always-on / random(C1) / uniform(C1) per_sample_nfe 공식을 위 정정식으로 통일.
- **이유**: S-IEC sampler (`adaptive_generalized_steps_siec`) 는 interval_seq 에서 trigger 와 무관한 lookahead 1회를 수행하고 (`step_nfe += 1` at `deepcache_denoising.py:436`), 추가 correction forward 는 `siec_max_rounds - 1` round 부터만 발생 (`:483`). 기존 공식은 rounds=1 에서도 trigger 에 비례해 NFE 가 증가한다고 잘못 계산.
- **주석 태그**: wrapper 내부.
- **import 영향**: 없음.
- **롤백**: 해당 함수/호출부 되돌리기.
- **관련 Candidate**: C1 (random/uniform), C3 (post-mortem 정확도).

### 2026-04-23 KST — EXP-4 / EXP-5 — 외부 리뷰 2차 대응 (P1-P5)

5 개 보강 포인트를 "논문 신뢰 결과" 방향으로 정리. wrapper 내부가 중심이고 실험 복사본 (`experiments/yongseong/*.py`) 는 건드리지 않았다.

#### (P1) symlink 제거 — canonical path resolver 도입

- **원본**: wrapper (`real_05_robustness.py`)
- **변경 요지**:
  - `_ensure_fallback_link` 헬퍼 삭제.
  - `pilot_scores_path(label)` 이 `W8A8_DC10` 에 대해 `pilot_scores_nb.pt` 를, 다른 라벨에는 `pilot_scores_{label}.pt` 를 반환.
  - `tau_schedule_path(label, p)` 이 `W8A8_DC10` 에 대해 `tau_schedule_p{p}.pt` 를, 다른 라벨에는 `tau_schedule_{label}_p{p}.pt` 를 반환.
  - `build_pilot_cmd` / `build_calibrate_cmd` 가 이 헬퍼가 반환한 실제 경로를 emit → commands.sh 가 단독 실행 가능.
- **이유**: 리뷰 지적 "commands.sh 가 symlink 를 전제하면 재현성이 깨진다". `W8A8_DC10` 이 canonical setting 이라는 사실을 파일명 자체에 반영하여 alias 없이 직접 참조.
- **영향**: `pilot_scores_W8A8_DC10.pt` / `tau_schedule_W8A8_DC10_p80.pt` 경로는 더 이상 emit 되지 않음.

#### (P2) fp16 단일 reference row — no_correction/iec 분기 제거

- **원본**: wrapper (`real_05_robustness.py`)
- **변경 요지**:
  - `phase_main` 이 `label == "fp16"` 일 때 3 methods 루프를 생략하고 단일 `fp16 reference` row (`method="fp16_ref"`) 만 emit.
  - `build_main_cmd` 에 `fp16_ref` 분기 추가 (`common + ["--no-use-siec"]`) — plain DDIM 호출.
  - 실험 복사본이 없으면 blocked row (NFE=100, 미실행) 로 표시.
- **이유**:
  - fp16 은 interval_seq=None 이라 S-IEC lookahead 가 정의되지 않는다 (이전 패치에서 NotImplementedError 로 막음).
  - IEC author (`adaptive_generalized_steps_3`) 는 `enable_implicit = cur_i in interval_seq` 인데 interval_seq=[] 이므로 항상 max_iter=1 → plain DDIM 과 수학적으로 동치.
  - 즉 fp16 에서는 no_correction/iec 를 구분할 의미가 없다. 두 row 를 독립 결과로 보고하면 리뷰어가 "왜 FID 가 동일한가?" 로 지적한다. 단일 reference 로 축소하는 것이 paper-defensible.
- **영향**: fp16 commands.sh 는 2 줄 → 1 줄. results.csv 의 fp16 row 도 3 개 → 1 개.
- **관련 Candidate**: C4 (fp16 지원 경로). C4 자체는 유지되고, 의미론만 reference-only 로 재정의.

#### (P3) real_05 postmortem NFE 공식 동기화

- **원본**: wrapper (`real_05_robustness.py`) line 432 (patch 전) / 해당 영역
- **변경 요지**:
  - `postmortem_trigger_nfe` 가 `(mean_rate, per_sample_nfe, n_checks)` 세 값을 반환.
  - 공식을 `NUM_STEPS + rounds * sum(rates)` → `NUM_STEPS + n_checks + max(0, rounds-1) * sum_rate` 로 정정 (real_04 와 동일).
  - `fill_post_hoc` 가 n_checks 를 row notes 에 기록.
- **이유**: real_04 의 정정 공식과 맞춤. S-IEC sampler 의 per-step NFE 구조 (interval_seq 에서 unconditional lookahead + (rounds-1) 개의 trigger-conditional lookahead) 를 정확히 반영.

#### (P4) IEC author NFE = 100 + n_checks (= 110)

- **원본**: wrapper 2 개 (`real_04_tradeoff.py`, `real_05_robustness.py`)
- **변경 요지**:
  - real_04 `seed_rows()` 의 `IEC (author)` NFE: `NUM_STEPS` → `NUM_STEPS + n_checks_default()`.
  - real_04 `seed_rows()` 의 `No correction (never)` NFE 도 `NUM_STEPS + n_checks_default()` — siec_never.npz 가 S-IEC sampler 로 생성되어 unconditional lookahead 포함.
  - real_05 `seed_row_w8a8_dc10_iec()` 의 NFE: `NUM_STEPS` → `NUM_STEPS + n_checks_for_setting("W8A8_DC10")`.
  - real_05 `phase_main` 의 non-C2 IEC fallback row NFE 도 동일하게 정정.
- **이유**: `adaptive_generalized_steps_3` (IEC author) 는 interval_seq 위치에서 `max_iter=2` 반복문을 돌며 iter=0, iter=1 두 forward 를 항상 수행. `residual<tol` break 은 `max_iter=2` 에서 다음 iter 가 없어 **실질적으로 no-op**. 따라서 IEC author NFE 는 정확히 `100 + n_checks`. 기존 100 은 compute-matched row 계산을 왜곡시켰음.
- **영향**: `compute_matched.md` 의 IEC baseline 이 110 기준으로 바뀌어 S-IEC 110 점과 동일 NFE 비교가 정상화.

#### (P5) compute-matched Random/Uniform baseline 자동 생성

- **원본**: wrapper (`real_04_tradeoff.py`)
- **변경 요지**:
  - `build_sweep_rows` 가 `args.use_experiment_copy` 일 때 각 percentile p 에 대해 `postmortem(p, ...)` 로 mean trigger rate 를 계산.
  - 매칭 rate 로:
    - `Random matched to p{P} (prob=rate)` row + command
    - `Uniform matched to p{P} (period=round(1/rate))` row + command
  - notes 에 `paper primary baseline` + `realized vs target rate` 를 기록.
  - 기존 `--trigger-probs` / `--trigger-periods` grid 는 `C1 grid (exploratory, not compute-matched)` 로 라벨 변경.
- **이유**: "Is syndrome selection better than random at the same compute?" 는 compute-matched 일 때만 의미 있다. 기존 grid 는 compute 가 S-IEC 와 달라서 FID 차이가 trigger quality 때문인지 compute 차이 때문인지 분리 불가.
- **영향**: commands.sh 에 매칭 baseline 2 개/percentile 이 추가. results.csv 에 `Random matched`, `Uniform matched` 행이 추가되며 논문 figure 의 주력 비교 대상.
- **관련 Candidate**: C1 (random/uniform trigger).

### 2026-04-23 KST — EXP-4 / EXP-5 — wrapper 검증 엄격화

P1-P5 dry-run 검증 후 남은 wrapper-level 불명확성을 정리. 실험 복사본 sampler 는 건드리지 않았다.

#### real_04_tradeoff.py

- **원본**: wrapper 자체.
- **변경 요지**:
  - dry-run 경로에서도 `fill_postmortem()` 을 실행해 S-IEC sweep row 의 `trigger_rate`, `per_sample_nfe`, `nfe_total` 을 채운다.
  - `n_checks_default()` 에서 pilot score 파일/torch import 실패 시 `NUM_STEPS//10` 으로 대체하던 fallback 제거.
  - compute-matched Random/Uniform 생성 중 postmortem 예외를 삼키던 `try/except` 제거.
  - seed FID 는 hard-coded fallback 대신 실제 로그를 파싱한다. IEC official seed 는 `logs/stage6_fid_result.txt` 의 `Frechet Inception Distance` 를 source 로 사용.
- **이유**: 논문용 결과표에서 조용한 fallback 값이나 빈 postmortem row 가 섞이면 compute-matched 비교의 의미가 깨진다.
- **롤백**: 이 항목 이후 `real_04_tradeoff.py` 변경분 되돌리기.

#### real_05_robustness.py

- **원본**: wrapper 자체.
- **변경 요지**:
  - `n_checks_for_setting()`, `compute_error_strength()`, `postmortem_trigger_nfe()` 의 silent `None`/replicate-interval fallback 제거.
  - fp16 은 syndrome score 가 정의되지 않는 reference regime 으로 명시해 `error_strength=0.0` 만 사용.
  - non-fp16 setting 의 error strength / postmortem NFE 는 pilot score 와 tau schedule 이 실제로 있어야 계산된다.
  - IEC fresh run command 에서 `--no-use-siec` 가 없을 때 `--siec_always_correct` 로 대체하던 fallback 제거.
  - W8A8_DC10 IEC seed FID 는 `logs/stage6_fid_result.txt` 에서 직접 파싱한다.
- **이유**: robustness table 의 error-strength axis 와 NFE 는 paper-critical 값이라 placeholder/fallback 으로 채우지 않는다.
- **롤백**: 이 항목 이후 `real_05_robustness.py` 변경분 되돌리기.

### 2026-04-27 KST — EXP-FRAMING (D/E/A) — `deepcache_denoising.py`

- **원본**: `IEC/mainddpm/ddpm/functions/deepcache_denoising.py`
- **변경 요지**:
  - **[EXP-FRAMING-D]** `_adaptive_generalized_core()` 에 `reuse_lookahead: bool = False` 옵션 추가.
    - True 일 때: lookahead `_call_model()` 이 `prv_f=prv_f, allow_cache_reuse=True` 로 cache 재활용.
    - 추가로 `triggered=False` 인 step 의 lookahead 결과 `(et_look, x0_look, xt_next_hat)` 을 `lookahead_memo` 에 저장하고, 다음 step 진입 시 `t == memo.step_t_int` 이면 첫 forward 를 skip 하고 memo 사용 → toy `siec_sim/core/siec.py:97-99` 의 "net 1 NFE" 의도를 CIFAR 로 이식.
    - invalidate 조건: triggered=True / 마지막 step / refresh_step / xt mismatch.
  - **[EXP-FRAMING-E]** `correction_mode="siec_oracle"` 새 모드 + `oracle_xt_ref: list[Tensor] | None`, `oracle_pull_strength: float = 1.0` 옵션. 매 step 끝에서 `xt_next_hat ← (1-pull) xt_next_hat + pull * oracle_xt_ref[cur_i+1]`. syndrome score 는 측정만 (보정에는 미사용).
    - 신규 entry: `adaptive_generalized_steps_oracle()`.
  - **[EXP-FRAMING-A]** `trace_include_xs: bool = False` 옵션 + trace dict 에 `xs_trajectory`, `et_per_step` 추가. 실험 A (deploy vs ref) 와 실험 E (reference 확보) 의 입력으로 사용.
  - `adaptive_generalized_steps_siec()`, `adaptive_generalized_steps_trace()` 시그니처에 위 3 옵션 추가 (모두 default off → 기존 호출 무영향).
- **이유**: `docs/siec_ecc_framing_20260427.md` §3 의 실험 D/A/E 구현. D 는 NFE +91% 폭증의 원인이 알고리즘이 아니라 lookahead 재활용 차단 (`allow_cache_reuse=False`) 임을 검증. E 는 framing 의 이론 상한 측정. A 는 syndrome–error 상관성 측정.
- **주석 태그**: `# [EXP-FRAMING-D]`, `# [EXP-FRAMING-E]`, `# [EXP-FRAMING-A]`.
- **import 영향**: 모든 옵션 default off → 기존 IEC/S-IEC/trace 호출 동일. `experiments/yongseong/deepcache.py` 가 이 옵션을 forwarding.
- **롤백**: 위 태그가 달린 라인 / 함수 (`adaptive_generalized_steps_oracle`) 제거.
- **관련 Candidate**: 신규 (FRAMING-D / FRAMING-E / FRAMING-A).

### 2026-04-27 KST — EXP-FRAMING (D/E/A) — `deepcache.py`

- **원본**: `IEC/mainddpm/ddpm/runners/deepcache.py` (실험 복사본)
- **변경 요지**:
  - `adaptive_generalized_steps_trace` 호출에 `trace_include_xs`, `reuse_lookahead`, `oracle_xt_ref`, `oracle_pull_strength` forwarding.
  - `correction_mode="siec_oracle"` 분기 추가: `adaptive_generalized_steps_oracle` 호출 + `args.oracle_xt_ref` 파일 lazy-load (list[Tensor] or dict["xs_trajectory"]).
  - `correction_mode="siec"` 분기에 `reuse_lookahead` forwarding.
- **이유**: 새 옵션을 sampling 함수까지 전달.
- **주석 태그**: `# [EXP-FRAMING-A]`, `# [EXP-FRAMING-D]`, `# [EXP-FRAMING-E]`.
- **import 영향**: 원본 경로 영향 없음.
- **롤백**: 해당 태그 라인 / `siec_oracle` 분기 제거.
- **관련 Candidate**: FRAMING-D / E / A.

### 2026-04-27 KST — EXP-FRAMING (D/E/A) — `ddim_cifar_siec.py`

- **원본**: `IEC/mainddpm/ddim_cifar_siec.py` (실험 복사본)
- **변경 요지**: CLI 4 개 신규 — `--trace_include_xs`, `--reuse_lookahead`, `--oracle_xt_ref`, `--oracle_pull_strength`, `--correction_mode {auto,none,iec,siec,siec_oracle}`. argparse 에만 추가; 기존 동작 무변경.
- **이유**: wrapper 들이 새 옵션을 subprocess CLI 로 넘길 수 있도록 노출.
- **주석 태그**: `# [EXP-FRAMING-A]`, `# [EXP-FRAMING-D]`, `# [EXP-FRAMING-E]`.
- **import 영향**: 없음.
- **롤백**: 해당 add_argument 라인 제거.
- **관련 Candidate**: FRAMING-D / E / A.

### 2026-04-27 KST — EXP-FRAMING (CLI consolidation) — `ddim_cifar_siec.py`
- **원본**: `IEC/mainddpm/ddim_cifar_siec.py` (실험 복사본)
- **변경 요지**: 직전 패치에서 추가됐던 두 번째 `--correction_mode` (underscore 형) add_argument 를 제거하고, 기존 `--correction-mode` 의 choices 에 `siec_oracle` 만 합류. argparse 가 두 형태 (hyphen/underscore) 를 같은 dest 에 매핑해 마지막 argv 가 이기는 footgun 회피.
- **이유**: smoke test 중 dest 충돌로 인해 옵션의 효과가 argv 순서에 의존하는 문제 발견.
- **롤백**: choices 에서 `siec_oracle` 만 제거.
- **관련 Candidate**: FRAMING-E (영향 없음 — 외부 노출 인터페이스 동일).

### 2026-04-27 KST — EXP-FRAMING (folder relocation) — wrappers
- **신규 폴더**: `IEC/experiments/yongseong/framing/`
- **변경 요지**: framing 실험 6 개 wrapper 를 `IEC/experiments/real_06_framing_*.py` 대신 `IEC/experiments/yongseong/framing/exp_{D,A,B,C,E,F}_*.py` 에 둔다. 사용자가 06–11 번호를 혼동해 폴더 단위로 분리해 달라는 요청을 반영.
- **공용 helper**: `IEC/experiments/yongseong/framing_common.py` (yongseong/ 직속) — wrapper 들은 한 단계 위 (`Path(__file__).resolve().parent.parent`) 를 sys.path 에 추가해 import.
- **롤백**: 폴더 삭제.
- **관련 Candidate**: 모든 FRAMING 실험.

### 2026-04-28 KST — EXP-RELOC — `IEC/experiments/yongseong/` → `Semantic/S-IEC/`

- **변경 요지**: 작업 영역을 `IEC/experiments/yongseong/` 에서 `Semantic/S-IEC/` 로 이전.
  - `IEC/{mainddpm, siec_core, quant}/` 와 `IEC/evaluator_FID.py` 를 `S-IEC/` 루트에 사본화 (1저자 원본은 IEC/ 에 그대로 frozen).
  - yongseong 미러 변경분 3 파일 (`ddim_cifar_siec.py`, `ddpm/runners/deepcache.py`, `ddpm/functions/deepcache_denoising.py`) 을 사본의 대응 위치에 적용 → 재설계 출발점.
  - wrapper / framing / scripts 는 `S-IEC/experiments/yongseong/` 하위로 이동 (1저자 이식 시 이 폴더만 제외하면 baseline 깨끗).
  - `Semantic/docs/` 는 `S-IEC/docs/{postmortems, critique, plan, paper}/` 로 분류 이동 (Semantic/docs/ 폴더 자체는 삭제).
  - `S-IEC/.gitignore` 신규 작성 (npz/pt/log/results 등 통째 ignore).
  - IEC repo 안의 yongseong/ 원본 + IEC/experiments/real_0{4,5}.py 는 사용자 검증 후 별도 단계에서 삭제 예정.
- **이유**: S-IEC 논리에 구조적 문제(deployment-conditional pilot, dead iterative correction, manifold framing inversion — `S-IEC/docs/critique/` 참조)가 발견되어 단발 patch 가 아니라 **모듈 단위 재설계** 필요. flat 미러 + sys.path shim 구조로는 한계가 있었고, 1저자 이식성을 위해 파일명/디렉토리 구조를 그대로 보존하는 통째 사본 방식으로 전환.
- **주석 태그**: `# [EXP-RELOC]` (이 폴더 이전 자체와 관련된 라인만 사용; 이후 재설계는 새 태그 체계로).
- **import 영향**:
  - 기존 yongseong/ flat shim (`sys.path.insert(0, parent)`) 은 사본 안에서도 그대로 작동하지만 사실상 redundant — `S-IEC/` 가 자체 sys.path root 역할.
  - wrapper(`real_04_tradeoff.py`, `real_05_robustness.py`) 의 `EXP_DIR = IEC_ROOT / "experiments/yongseong"` 와 `f"experiments/yongseong/{name}"` 는 아직 옛 경로를 가리킴 — 재설계 첫 단계에서 정리 예정.
  - 데이터/모델 파일 (`cifar10_reference.npz`, `classify_image_graph_def.pb`, `calibration/`, `data_cifar10/`) 은 옮기지 않음. 실행 시 `cwd=IEC_ROOT` 로 두는 방식으로 1저자 코드 path 문자열 무수정 유지.
- **롤백**: `Semantic/S-IEC/` 통째 삭제 + IEC repo 의 yongseong/ 변경 commit 복원.
- **관련 Candidate**: 모든 candidate 의 새로운 home.

### 2026-04-28 KST — EXP-RELOC (post-cleanup) — results/ 정리 + mainddpm 보충

- **결과 폴더 (results/) 분류 이동**: `IEC/experiments/yongseong/results/` (4.2GB) → `S-IEC/results/`.
  - **보존 (S-IEC 로 이동)**:
    - exp_A/exp_C/exp_D/exp_E/exp_F latest run dir (총 8.0GB; tau_inf, traces, samples_*.npz, logs, figure1.{pdf,png}, oracle_results.csv, cross_error_grid.csv 등).
    - exp_B_innovation/20260428_104717 (innovation_plots.png + innovation_table.csv).
    - real_04_tradeoff/{20260424_180904 (random/uniform_matched 전 percentile sweep, 140M), 20260425_103931, 20260425_104700} + dir-level metadata (commands.sh, results.csv, results.json, tradeoff_2panel.{pdf,png}, tradeoff_diagnostic.{pdf,png}, diagnostic_summary.md, manual_logs/).
    - real_05_robustness/{20260424_235643, 20260425_000009, 20260425_103931 (history results.csv/json), 20260425_153134 (latest)} + dir-level metadata.
  - **삭제 (S-IEC latest 가 superset)**:
    - 24 개 빈/metadata-only run dir (8K~120K, inventory.md / commands.sh 만 보유).
    - exp_A_correlation/20260428_004253 (2.2GB old traces; latest 가 더 광범위).
    - exp_D_lookahead/20260427_235329 (1.8GB old traces).
    - real_04_tradeoff/20260424_235939 (85M; latest 의 직전 dup).
  - 총 회수 4.1GB. `IEC/experiments/yongseong/results/` 자체는 빈 트리만 남음.
- **mainddpm/ 사본 누락 보충 (이후 정정 — 아래 EXP-RELOC (baseline-clean) 참조)**: 직전 EXP-RELOC step 에서 `deepcache.py`, `deepcache_denoising.py` 가 `S-IEC/mainddpm/` 직속 사본에 빠져있던 걸 발견하고 yongseong 사본을 카피. — 그러나 baseline 다른 ddim_cifar_*.py 들이 모두 `from ddpm.runners.deepcache import Diffusion` (절대 import + relative import baseline) 패턴을 쓰는 걸 발견, 이 sys.path shim 은 옛 mirror-copy 시절 잔재로 판명됨. 다음 entry 에서 되돌림.
- **잔여 (별도 단계 예정)**: `IEC/experiments/yongseong/` 폴더 자체 (CHANGES.md, ddim/deepcache mirror, framing/, framing_common.py, plot_real_04_diagnostics.py, run_exp{4,5}_refresh.sh — 모두 S-IEC 에 매핑 완료) + `IEC/experiments/real_0{4,5}.py` 원본은 사용자 최종 검증 후 삭제.
- **롤백**: 삭제된 dir 은 복구 불가 (40GB 절약 우선); mainddpm/ 사본은 두 파일 rm 으로 복귀.

### 2026-04-28 KST — EXP-RELOC (baseline-clean) — 1저자 baseline 으로 정렬

- **결정 (사용자)**: "재설계를 좀 할거니까 1저자분이랑 똑같이 두자. 나의 수정사항을 이식할 때 같은 파일·같은 위치면 되도록."
- **변경 요지**: yongseong [EXP-CN] / [EXP-FRAMING] 패치를 모두 되돌리고 `S-IEC/mainddpm/` 의 3 파일을 `IEC/mainddpm/` baseline 으로 복원.
  - `S-IEC/mainddpm/ddpm/runners/deepcache.py` ← baseline (325 줄, relative import 원형)
  - `S-IEC/mainddpm/ddpm/functions/deepcache_denoising.py` ← baseline (688 줄)
  - `S-IEC/mainddpm/ddim_cifar_siec.py` ← baseline (291 줄, sys.path shim 없음, `from ddpm.runners.deepcache import Diffusion`)
  - `S-IEC/mainddpm/deepcache.py`, `S-IEC/mainddpm/deepcache_denoising.py` (mainddpm/ 직속 yongseong 사본) **삭제**.
- **이유**:
  - 1저자 다른 baseline 파일 (`ddim_cifar_{cali,sampling,predadd,quant,quant_noquant,quant_nocache,params}.py`) 도 모두 `from ddpm.runners.deepcache import Diffusion` 한 줄로 잘 작동. 별도 shim 불필요.
  - 재설계 단계에서 [EXP-CN] / [EXP-FRAMING] 패치를 새 구조에 맞춰 다시 작성할 예정이므로, 출발점은 깨끗한 baseline 사본이어야 diff/patch 분리가 명확.
  - "같은 파일·같은 위치" 원칙: 재설계 후 변경사항은 baseline 의 같은 path 에 직접 적용되어 1저자 흡수 시 path 단위 patch 로 추출 가능.
- **현재 상태**: `diff -q IEC/mainddpm/{ddpm/runners/deepcache.py, ddpm/functions/deepcache_denoising.py, ddim_cifar_siec.py} S-IEC/mainddpm/{...}` 모두 일치 (출력 없음).
- **결과로 깨진 것 (재설계 시 다시 작성 예정)**:
  - `S-IEC/experiments/yongseong/framing/exp_{A,D,E,...}.py` 가 의존하던 CLI 옵션 (`--reuse_lookahead`, `--oracle_xt_ref`, `--oracle_pull_strength`, `--correction-mode siec_oracle`, `--trigger_mode`, `--no-cache`, `--no-ptq`, `--no-use-siec` 등) 이 baseline ddim_cifar_siec.py 에는 없음 → wrapper 들은 현재 실행 시 argparse 에러.
  - `siec_core/{utils,corrector_oracle}.py` 에서 정의되었던 oracle decoder / [EXP-CN] entry 도 baseline 에 부재 → S-IEC/siec_core/ 도 baseline 사본 상태인지 별도 확인 필요 (다음 step).
- **이식 가이드**: 재설계 시 변경분은 다음 위치에만 두고 별도 사본/shim 없이 진행:
  - `mainddpm/ddim_cifar_siec.py` (CLI 추가 + correction_mode)
  - `mainddpm/ddpm/runners/deepcache.py` (model creation + cache 토글)
  - `mainddpm/ddpm/functions/deepcache_denoising.py` (correction backend dispatch)
  - `siec_core/` (oracle / 트리거 모드 등 corrector 측)
- **롤백**: `git diff` 가 가능한 상태였다면 revert. 지금은 `IEC/experiments/yongseong/{deepcache.py, deepcache_denoising.py}` 와 `IEC/mainddpm/ddim_cifar_siec.py` 의 yongseong 사본 (= 이전에 있던 mirror) 으로부터 다시 복원 가능.

### 2026-04-28 KST — EXP-RELOC (final cleanup) — IEC repo 의 yongseong 흔적 제거

- **삭제**: 사용자 인가 후 다음을 IEC repo 에서 제거.
  - `IEC/experiments/yongseong/` (통째) — CHANGES.md, ddim_cifar_siec.py, deepcache.py, deepcache_denoising.py, framing/, framing_common.py, plot_real_04_diagnostics.py, run_exp{4,5}_refresh.sh. 모두 `S-IEC/` 에 매핑 완료.
  - `IEC/experiments/real_04_tradeoff.py` (32 KB)
  - `IEC/experiments/real_05_robustness.py` (31 KB)
- **보존**: `IEC/experiments/real_03_iec_vs_siec_fid.py` 는 유지 (사용자 지시 — 4, 5 만 삭제).
- **결과**: IEC repo 의 1저자 코드 영역 (`mainddpm/`, `siec_core/`, `quant/`, `evaluator_FID.py`) 은 baseline 그대로, `experiments/` 는 yongseong 흔적 제거로 다시 깨끗. yongseong 작업은 모두 `Semantic/S-IEC/` 에 격리됨.
- **롤백**: 삭제는 비가역. 필요 시 `Semantic/S-IEC/` 의 사본에서 IEC repo 위치로 다시 카피하면 복원 가능.
