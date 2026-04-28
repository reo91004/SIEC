# S-IEC 실험 4/5 검증 및 보강 구현 가이드

마지막 검증: 2026-04-25 KST

최신 exp4 결과 분석은 별도 문서로 정리했다.

- [exp4_postmortem_20260425.md](/home/user/jowithu/Semantic/docs/exp4_postmortem_20260425.md)

최신 exp5 결과 분석도 별도 문서로 정리했다.

- [exp5_postmortem_20260425.md](/home/user/jowithu/Semantic/docs/exp5_postmortem_20260425.md)

이 문서는 구현 관점의 validation 문서로 유지하고, 최신 수치 기반 해석과 다음 액션은 위 postmortem 문서를 우선 참고한다.

이 문서는 현재 `IEC/` 코드베이스가 용성님 실험 4/5 지시사항을 얼마나 충족하는지 검증하고, 무엇을 어떻게 보강해야 하는지 구현자 관점에서 정리한 문서다. 이 문서만 읽어도 다음이 가능해야 한다.

1. 최신 지시사항이 무엇인지 이해할 수 있어야 한다.
2. 현재 구현이 어디까지 맞고 어디서 어긋나는지 알 수 있어야 한다.
3. 어떤 파일을 어떤 순서로 보강해야 하는지 알 수 있어야 한다.
4. TAC는 뼈대만 남기고 실제 알고리즘 구현은 일단 스킵할 수 있어야 한다.

이 문서는 기존의 간단한 피드백 메모를 대체한다. 현재 실험 운영의 source of truth는 아래 우선순위를 따른다.

## 0. Source Of Truth

최신 지시사항 우선순위는 다음과 같다.

1. 사용자에게 최근에 전달된 지시사항
2. 그 이전의 재설계 메모
3. 저장소 내 보조 문서 (`docs/YONGSEONG_TODO.md`, `docs/research_summary.md`, `docs/YONGSEONG_RUN_COMMANDS.md`)
4. 현재 wrapper 구현

따라서 현재 구현이 문서와 충돌하면 구현이 아니라 최신 지시사항을 기준으로 고쳐야 한다.

## 1. 최신 요구사항 정리

### 1.1 실험 4. Compute-Quality Tradeoff

현재 최우선 요구사항은 아래 조합이다.

- 메인 데이터셋/모델: `CIFAR-10`, `W8A8 + DeepCache`, `100 DDIM steps`
- 확장: `LSUN`, `SD on COCO`는 CIFAR 패턴 확인 후
- 필수 baseline:
  - `No correction`
  - `TAC`
  - `IEC`
  - `Naive always-on refinement`
  - `Random trigger`
  - `Uniform periodic`
  - `S-IEC`
- 구현 대응:
  - `Naive always-on refinement`는 `S-IEC always-correct`와 등가이며, 현재 코드에서는 `--siec_always_correct`로 대응된다.
- 지표:
  - `FID`, `sFID`
  - `total NFE`
  - `wall-clock time`
  - `trigger rate`
  - `per-step syndrome score distribution`
- plot:
  - 왼쪽: `NFE vs FID`
  - 오른쪽: `trigger rate vs FID`
- 성공 조건:
  - `NFE saving`: `NFE < IEC` 이면서 `FID <= IEC + margin`
  - `Selectivity`: 같은 compute에서 `S-IEC > Random/Uniform`
  - `Spectrum`: `tau` 변화에 따라 점들이 부드럽게 이동
- 필수 산출물:
  - `tau` 7점 sweep: `30, 50, 60, 70, 80, 90, 95`
  - `compute-matched` 표: IEC와 NFE가 가장 가까운 S-IEC 행 비교
  - pilot 1회 후 percentile만 바꿔 재사용

### 1.2 실험 5. Robustness Across Deployment Errors

현재 최우선 요구사항은 아래 조합이다.

- 메인 데이터셋/모델: `CIFAR-10`, `DDIM 100 steps`
- 고정 조건: 모델과 스케줄은 고정하고 deployment error만 바꾼다.
- 지시된 error family:
  - `fp16`
  - `W8A8 quantization`
  - `DeepCache only (interval=10)`
  - `W4A8 quantization`
  - `DeepCache aggressive (interval=20 or 50)`
  - `CacheQuant = W4A8 + DeepCache`
- 필수 baseline:
  - `No correction`
  - `TAC`
  - `IEC`
  - `S-IEC`
- 지표:
  - `FID`, `sFID`
  - `per-step syndrome score mean`
  - `trigger rate`
  - `total NFE`
  - `wall-clock time`
- plot:
  - 왼쪽: `error strength vs FID`, 네 곡선 (`No correction`, `TAC`, `IEC`, `S-IEC`)
  - 오른쪽: 최신 지시사항까지 반영하면 `TAC`도 포함되어야 한다.
  - 가장 보수적으로는 오른쪽에 최소 `FID_IEC - FID_SIEC`를 유지하고, 가능하면 `FID_TAC - FID_SIEC`도 함께 그린다.
- 필수 정책:
  - error strength 축은 단순 라벨 나열이 아니라 수치화해야 한다.
  - 각 setting마다 `pilot -> tau recalibration`을 다시 수행한다.

### 1.3 TAC 정책

현재 TAC는 다음 전제로 다룬다.

- 논문: `TAC-Diffusion` (ECCV 2024), 사용자 제공 링크 `https://arxiv.org/abs/2407.03917`
- 로컬 저장소에는 TAC 구현이 없다.
- 빠른 확인 기준으로 공식 코드 공개 여부도 검증되지 않았다.
- 따라서 지금 단계의 원칙은 다음과 같다.
  - wrapper/CSV/plot에 `TAC` 행과 뼈대는 반드시 넣는다.
  - 실제 알고리즘 계산은 일단 `blocked / not implemented`로 남길 수 있다.
  - TAC 결과를 임의로 채우거나 다른 방법 결과를 복사해 넣으면 안 된다.

## 2. 최종 판정

현재 코드는 최신 지시사항을 완전히 반영하지 못했다.

- 실험 4: `부분 구현 + 핵심 결함 존재`
- 실험 5: `부분 구현 + setting taxonomy 자체가 어긋남`
- TAC: `미구현`

즉 현재 `real_04_tradeoff.py`와 `real_05_robustness.py`는 진단용 wrapper로는 쓸 수 있지만, 지금 상태 그대로는 최종 제출용 실험 wrapper가 아니다.

## 3. 실험 4 검증 결과

### 3.1 요구사항 대비 상태 표

| 항목 | 상태 | 판정 | 근거 | 조치 |
|---|---|---|---|---|
| CIFAR-10 / W8A8 + DeepCache / 100 steps | 구현됨 | 대체로 맞음 | `IEC/experiments/real_04_tradeoff.py` | 유지 |
| `tau = {30,50,60,70,80,90,95}` sweep | 구현됨 | 맞음 | `parse_args`, `build_sweep_rows` | 유지 |
| `Random trigger`, `Uniform periodic` | 구현됨 | 맞음 | `build_trigger_cmd`, matched/grid rows | 유지 |
| `IEC` baseline | 구현됨 | 부분적으로 맞음 | seed row는 존재하나 50K seed 중심 | 2K/동일 샘플수 기준 fresh run 정책 정리 필요 |
| `Naive always-on refinement` | 구현됨 | 부분적으로 맞음 | 현재 `S-IEC always-on` row가 존재 | baseline 이름/표기 통일 필요 |
| `No correction` baseline | 구현됐지만 잘못됨 | 치명적 | 현재 `tau_schedule_never` 기반 S-IEC path라 unconditional lookahead 비용이 남음 | 진짜 deployed no-correction path를 새로 구현 |
| `TAC` baseline | 없음 | 치명적 | wrapper/CSV/plot 어디에도 없음 | TAC skeleton 추가 |
| `NFE vs FID` curve | 형태만 있음 | 치명적 | 기본값 `--siec-max-rounds 1`이면 sweep/random/uniform NFE가 모두 110으로 수렴 | NFE semantics와 실행 설정 수정 |
| `trigger rate vs FID` panel | 구현됨 | 맞음 | `plot_two_panel` | 유지 |
| `compute-matched` 행 | 구현돼 있으나 무력화 | 치명적 | NFE가 전부 같아서 의미 퇴색 | 실제 NFE variation 확보 후 재사용 |
| `wall-clock time` | 부분 구현 | 불충분 | sampling 시간만 기록, FID 시간은 별도 반영 안 됨 | total wall-clock 집계로 변경 |
| `sFID` | evaluator 지원 | 부분 구현 | FID log에 따라 채워지나 coverage 불완전 | 결과 채움 확인 필요 |
| syndrome distribution artifact | 없음 | 부족 | trace는 있으나 산출물 없음 | histogram/json/png 생성 추가 |

### 3.2 핵심 문제 1: 현재 실험 4는 Pareto frontier를 제대로 증명하지 못한다

현재 `real_04_tradeoff.py`는 S-IEC 계열 NFE를 다음처럼 계산한다.

```text
per_sample_nfe = num_steps + n_checks + (rounds - 1) * n_triggered
```

그런데 wrapper 기본값은 `--siec-max-rounds 1`이다. 그러면:

```text
rounds - 1 = 0
=> trigger_rate가 달라도 추가 NFE가 0
=> S-IEC p30~p95, random, uniform, always-on이 사실상 같은 NFE
```

현재 실제 결과도 이 구조를 보여준다.

- `IEC/experiments/yongseong/results/real_04_tradeoff/results.csv`
- 현재 sweep/random/uniform row들의 `per_sample_nfe`가 전부 `110.0`

이 상태에서는 다음 주장을 할 수 없다.

- `S-IEC가 IEC보다 더 적은 NFE에서 같은 FID를 달성한다`
- `S-IEC curve가 Pareto frontier 위에 있다`

현재 할 수 있는 것은 많아야 아래 정도다.

- `tau -> trigger rate`가 단조롭게 움직이는지
- 같은 NFE처럼 보이는 조건에서 random/uniform보다 선택성이 나은지

즉 지금까지 돌아간 실험 4 결과는 `diagnostic`이지 `final evidence`가 아니다.

### 3.3 핵심 문제 2: `No correction`이 진짜 lower bound가 아니다

현재 `No correction (never)`는 진짜 deployed no-correction sampler가 아니라 다음 방식이다.

- S-IEC sampler를 켠 채
- `tau_schedule_never`를 써서 trigger만 영원히 꺼둔다.

하지만 이 경로는 interval check 시점마다 lookahead를 수행한다. 따라서 NFE가 여전히 `100 + n_checks = 110`이다. 즉:

- 이름은 `No correction`
- 실제 비용은 `no correction`보다 비싸다

이 row는 최신 지시사항의 `No correction (하한 reference)`를 만족하지 못한다.

### 3.4 핵심 문제 3: `compute-matched` 표가 현재 무의미하다

`compute-matched`는 IEC와 가장 가까운 NFE의 S-IEC row를 뽑아 비교해야 한다. 그런데 현재는 IEC도 110, S-IEC sweep도 110이어서 표가 사실상 같은 x축 점끼리의 FID 비교가 된다.

따라서 현재 `compute_matched.md`는 최종 산출물로 간주하면 안 된다.

### 3.5 실험 4에서 유지해도 되는 것

아래는 방향이 맞다.

- CIFAR-10 메인 세팅
- pilot 1회 + 7개 percentile
- random/uniform matched baseline
- 오른쪽 panel에서 `trigger rate vs FID`
- 실험 복사본(`IEC/experiments/yongseong/`) 중심 운영

## 4. 실험 5 검증 결과

### 4.1 요구사항 대비 상태 표

| 항목 | 상태 | 판정 | 근거 | 조치 |
|---|---|---|---|---|
| wrapper 존재 | 구현됨 | 맞음 | `IEC/experiments/real_05_robustness.py` | 유지 |
| per-setting pilot/calibration 구조 | 구현됨 | 대체로 맞음 | `phase_pilot`, `phase_calibrate` | 유지 |
| error strength 수치화 | 구현됨 | 맞음 | `compute_error_strength` | 유지 |
| `No correction`, `IEC`, `S-IEC` | 구현됨 | 부분적으로 맞음 | method rows 있음 | TAC 추가, fp16 정책 수정 |
| `TAC` baseline | 없음 | 치명적 | wrapper/plot 어디에도 없음 | TAC skeleton 추가 |
| 최신 setting taxonomy 반영 | 안 됨 | 치명적 | 현재는 `W8A8_DC10`, `W8A8_DC20`, `W4A8_DC10`, `W8A8_DC50`, `CacheQuant` 중심 | pure quant / pure deepcache 축으로 재구성 |
| fp16 처리 | 구현됨 | 최신 지시와 불일치 | 현재는 `fp16 reference` 1행으로 축소 | plot completeness를 위해 4행 유지 필요 |
| robustness left plot | 구현됨 | 부분적으로 맞음 | 현재는 `No correction`, `IEC`, `S-IEC` 3곡선 | TAC 추가 |
| robustness right plot | 구현됨 | 부분적으로 맞음 | 현재는 `IEC-SIEC`만 표시 | TAC 관련 gain도 반영 고려 |
| `wall-clock time` | 부분 구현 | 불충분 | sampling만 기록 | total wall-clock으로 변경 |

### 4.2 핵심 문제 1: setting taxonomy가 최신 지시사항과 다르다

현재 `real_05_robustness.py`는 다음 세팅을 쓴다.

- `fp16`
- `W8A8_DC10`
- `W8A8_DC20`
- `W4A8_DC10`
- `W8A8_DC50`
- `CacheQuant`

하지만 최신 지시사항은 최소한 아래 축을 명시적으로 포함하라고 한다.

- `fp16`
- `W8A8 quantization` only
- `DeepCache only (10)`
- `W4A8 quantization` only
- `DeepCache aggressive (20 or 50)` only
- `CacheQuant`

즉 현재 구현은 hybrid setting 위주라서 다음 질문에 답하지 못한다.

- `quantization만 강해질 때` S-IEC가 어떻게 변하는가
- `DeepCache만 강해질 때` S-IEC가 어떻게 변하는가

현재 wrapper는 deployment error family를 분리해 분석하는 구조가 아니다.

### 4.3 핵심 문제 2: fp16을 단일 reference row로 축약한 것은 최신 지시와 충돌한다

현재 wrapper는 fp16에서 IEC/S-IEC가 no-op이라고 보고 `fp16 reference` 단일 행만 만든다. 이 판단 자체는 기술적으로 타당하다. 하지만 최신 지시사항은 robustness plot에 아래 네 방법이 모두 보여야 한다.

- `No correction`
- `TAC`
- `IEC`
- `S-IEC`

따라서 fp16에서도 최소한 출력 row와 plot legend는 네 방법을 유지해야 한다. 구현 비용을 줄이려면 하나의 fp16 샘플 결과를 네 row가 공유하고, `notes=no-op regime`를 남기는 방식이 맞다.

### 4.4 핵심 문제 3: TAC가 완전히 빠져 있다

최신 지시사항 기준으로 실험 5 plot에서 TAC는 선택이 아니라 필수다. 현재는 전혀 반영되어 있지 않다.

## 5. 추가 정합성 주의사항

### 5.1 toy `perturbations.py`와 real diffusion taxonomy를 혼동하면 안 된다

사용자 지시에는 "교수님 코드의 perturbations.py 4가지 유형"이라는 표현이 있었지만, 실제 로컬 파일 `siec_sim/core/perturbations.py`는 다음 클래스를 정의한다.

- `IsotropicPerturbation`
- `DirectionalPerturbation`
- `QuantizationSimulation`
- `SNRDependentPerturbation`

즉 이 파일이 `fp16`, `DeepCache`, `CacheQuant`를 직접 정의하는 것은 아니다.

따라서 real diffusion 실험 5 구현에서는 toy 파일의 클래스 이름을 억지로 맞추려 하지 말고, 최신 지시사항의 실제 deployment error taxonomy를 기준으로 wrapper를 다시 짜야 한다.

## 6. 구현 보강 가이드

이 절은 실제로 어떤 파일을 어떻게 고쳐야 하는지 정리한다. 핵심 원칙은 기존 1저자 core를 건드리지 않고 `IEC/experiments/` 및 `IEC/experiments/yongseong/` 복사본에서만 해결하는 것이다.

### 6.1 공통 설계 원칙

모든 wrapper와 experimental copy는 method를 문자열 하나로 명시적으로 구분해야 한다.

권장 `method_key`:

```text
no_correction
tac
iec
random
uniform
siec
always_on
```

권장 `setting_key`:

```text
exp4_main
fp16
w8a8
dc10
w4a8
dc20
dc50
cachequant
```

`dc20`와 `dc50`는 둘 다 지원해도 되지만, 최종 plot에서는 최소 하나의 aggressive DeepCache point가 들어가야 한다.

**결과 저장 디렉토리 원칙**:
모든 실험 결과는 과거 실행 기록을 덮어쓰지 않도록 타임스탬프 기반의 하위 폴더에 격리하여 저장해야 한다.
- 경로 예시: `IEC/experiments/yongseong/results/real_04_tradeoff/YYYYMMDD_HHMMSS/`
- 경로 예시: `IEC/experiments/yongseong/results/real_05_robustness/YYYYMMDD_HHMMSS/`

### 6.2 `IEC/experiments/real_04_tradeoff.py` 보강

필수 수정 사항:

1. baseline registry를 최신 합의 기준으로 다시 정의한다.
   - 필수 표시 대상: `No correction`, `TAC`, `IEC`, `Naive always-on refinement`, `Random trigger`, `Uniform periodic`, `S-IEC`
   - `Naive always-on refinement`의 표기와 현재 코드의 `S-IEC always-on` 표기를 통일한다.
2. `TAC` row를 추가한다.
   - 아직 미구현이면 `status=blocked`, `notes=TAC not implemented`로 남긴다.
3. `No correction` row를 진짜 deployed no-correction path로 바꾼다.
   - 현재의 `tau_schedule_never` 방식은 폐기한다.
   - same deployment error (`W8A8 + DeepCache`)는 유지하되, correction만 완전히 꺼진 sampler가 필요하다.
4. `NFE`를 실제 trace 기반으로 집계한다.
   - 현재 postmortem 추정치는 보조 지표로만 쓰고, 최종 CSV는 실행 trace에서 직접 계산한 NFE를 사용한다.
5. 기본 설정에서 NFE variation이 실제로 생기게 한다.
   - 최소 `siec_max_rounds >= 2`가 필요하다.
   - 그렇지 않으면 S-IEC sweep의 x축이 전부 겹친다.
6. `compute_matched.md`는 같은 `num_samples` 집합 안에서만 비교한다.
   - 2K sweep와 50K seed를 섞어 nearest-match를 잡으면 안 된다.
7. `wall_clock_sec`는 최소 아래 둘로 분리하거나 total로 재정의한다.
   - `sampling_wall_clock_sec`
   - `fid_wall_clock_sec`
   - `total_wall_clock_sec`
8. syndrome 분석 artifact를 생성한다.
   - 예: `syndrome_hist_{method}.png`
   - 예: `syndrome_summary_{method}.json`

권장 구현 방식:

- `plot_two_panel()`는 다음 시그니처를 만족해야 한다.
  - 왼쪽: `S-IEC tau sweep` curve + `No correction`, `TAC`, `IEC`, `Naive always-on`, `Random`, `Uniform`
  - 오른쪽: `trigger_rate vs FID`
- `write_compute_matched()`는 `method_key == siec`만 대상으로 가장 가까운 NFE를 찾고, `reference_method == iec`를 명시한다.

### 6.3 `IEC/experiments/real_05_robustness.py` 보강

필수 수정 사항:

1. `setting_defs()`를 최신 taxonomy 기준으로 재작성한다.
   - `fp16`
   - `W8A8` pure quantization
   - `DC10` pure DeepCache
   - `W4A8` pure quantization
   - `DC20` 또는 `DC50` pure DeepCache aggressive
   - `CacheQuant`
2. method rows를 네 개로 고정한다.
   - `No correction`
   - `TAC`
   - `IEC`
   - `S-IEC`
3. `fp16`도 네 row를 만든다.
   - 같은 결과를 공유하더라도 row는 네 개를 유지한다.
   - `notes=no-op regime`를 남긴다.
4. plot legend와 color mapping에 `TAC`를 추가한다.
5. 오른쪽 panel은 최소 `FID_IEC - FID_SIEC`를 유지하고, 가능하면 `FID_TAC - FID_SIEC`도 추가한다.
6. setting별 `pilot -> tau recalibration`은 계속 유지한다.
7. pure quant / pure DeepCache setting이 실제로 돌 수 있도록 experimental copy를 확장한다.

권장 구현 방식:

- `error_strength`는 계속 `mean_over_t(mean(pilot_scores[t]))`를 주축으로 사용한다.
- `fp16`은 `0.0` 고정이 가능하다.
- plot 상에서 x축 정렬은 수치형 `error_strength` 기준으로 하되, legend/annotation에 setting label을 남긴다.

### 6.4 `IEC/experiments/yongseong/ddim_cifar_siec.py` 보강

현재 이 파일은 다음이 가능하다.

- `--no-cache`
- `--no-ptq`
- `--no-use-siec`
- `--siec_always_correct`
- `--trigger_mode`
- `--siec_return_trace`

하지만 아직 method semantics가 혼란스럽다. 권장 보강 방향은 아래 둘 중 하나다.

#### 선택지 A. 현재 플래그 체계를 유지하며 필요한 mode를 추가

- `--use-tac`
- `--no-correction`
- `--tac-stats-path`

#### 선택지 B. method를 단일 enum으로 통합

```text
--correction-mode {none,tac,iec,siec}
```

권장안은 `선택지 B`다. 이유는 wrapper에서 method selection이 명확해지고 `--use_siec`, `--no-use-siec`, `--siec_always_correct` 조합 실수를 줄일 수 있기 때문이다.

주의:

- pure quantization (`--no-cache --ptq`)는 현재 그대로는 안전하지 않다.
- 현재 no-cache path에서는 `all_cali_data=[]`라서 PTQ branch의 `torch.cat(all_cali_data)`가 실패할 수 있다.
- 따라서 pure quantization은 별도 experimental copy entrypoint 또는 no-cache 전용 calibration 로드 로직이 필요하다.

### 6.5 `IEC/experiments/yongseong/deepcache.py` 보강

현재 routing은 대략 다음과 같다.

- `interval_seq is None`이면 generalized/no-cache path
- `interval_seq is not None` + `use_siec=True`면 S-IEC
- `interval_seq is not None` + `use_siec=False`면 IEC

여기에 다음 모드를 명시적으로 추가해야 한다.

- `none`: deployed model with error, but correction disabled
- `tac`: TAC correction
- `iec`
- `siec`

즉 sample routing을 `correction_mode` 기준으로 분기하는 구조가 필요하다.

### 6.6 `IEC/experiments/yongseong/deepcache_denoising.py` 보강

이 파일은 실제 NFE semantics와 correction logic의 핵심이다.

필수 추가/수정:

1. `adaptive_generalized_steps_none` 또는 동등한 no-correction sampler 추가
   - deployed error는 유지
   - correction loop는 완전히 없음
   - trace에서 실제 `nfe_per_step`를 기록
2. `adaptive_generalized_steps_tac` 뼈대 추가
   - 초기 버전은 `NotImplementedError` 또는 `blocked` 반환이어도 됨
3. trace API를 모든 모드에서 공통으로 쓰게 한다.
   - `none`, `iec`, `siec`, `tac`
4. 최종 NFE는 trace의 `nfe_per_step` 합으로 계산한다.
   - wrapper의 analytic 추정치보다 trace를 우선한다.
5. syndrome distribution summary를 trace에서 직접 산출한다.

### 6.7 새로 추가해도 되는 파일

아래는 추가를 권장한다.

- `IEC/experiments/yongseong/tac_correction.py`
  - TAC 관련 correction factor 로직용 placeholder
- `IEC/experiments/tac_calibrate_cifar.py`
  - TAC calibration artifact 생성용 placeholder
- `IEC/experiments/yongseong/ddim_cifar_quant_nocache.py`
  - pure quantization no-cache experimental copy
- `IEC/experiments/yongseong/ddim_cifar_noquant_cache.py`
  - pure DeepCache no-PTQ experimental copy

이 네 파일은 전부 optional이지만, pure quant/pure DeepCache를 깔끔하게 분리하려면 사실상 필요해질 가능성이 높다.

## 7. TAC skeleton 설계안

TAC는 당장 구현하지 않아도 되지만, 코드 구조는 지금 넣어두는 것이 맞다.

### 7.1 반드시 남길 것

- wrapper에 `TAC` method row
- CSV/JSON에 `method=TAC`
- plot legend에 `TAC`
- inventory나 notes에 `blocked: TAC not implemented`
- 향후 calibration file 경로 placeholder
  - 예: `calibration/tac_stats_{setting}.pt`

### 7.2 지금은 하지 말아야 할 것

- TAC FID를 빈칸 대신 다른 baseline 값으로 복사
- TAC row를 조용히 삭제하고 plot에서 숨김
- TAC를 IEC 또는 S-IEC와 같은 구현으로 위장

### 7.3 TAC 구현 메모

논문 요약 수준에서 TAC는 quantized diffusion correction을 위해 timestep-dependent correction factor를 쓰는 방식으로 이해하면 된다. 실제 구현 단계에서는 다음이 필요할 가능성이 높다.

- timestep별 correction factor 저장 파일
- quantized vs full-precision 통계 calibration
- 샘플러에서 quantized noise estimate/output에 correction 적용

하지만 지금은 여기까지 들어가지 않는다. 뼈대만 넣고 `blocked`로 남긴다.

## 8. 결과 파일 스키마 권장안

현재 CSV 키는 최소 수준이라 이후 구현자가 다시 헤매기 쉽다. 아래 필드를 권장한다.

```text
setting
method_key
method
status
blocked_reason
tau_percentile
num_samples
fid
sfid
trigger_rate
error_strength
per_sample_nfe
nfe_total
sampling_wall_clock_sec
fid_wall_clock_sec
total_wall_clock_sec
trace_path
source_npz
source_log
notes
```

이 중 `status`는 다음처럼 쓴다.

- `ready`
- `completed`
- `blocked`
- `missing_asset`
- `not_implemented`

## 9. 실행 순서

지금부터의 권장 작업 순서는 아래다.

1. 이 문서 기준으로 wrapper 사양을 정리한다.
2. `real_04_tradeoff.py`를 먼저 고친다.
   - 이유: 실험 4는 현재 NFE 축이 붕괴해 있어 가장 치명적이다.
3. experimental copy에 `true no-correction`과 `trace-first NFE`를 넣는다.
4. `real_05_robustness.py`의 setting taxonomy를 pure quant / pure DeepCache 기준으로 다시 짠다.
5. TAC skeleton을 Exp4/Exp5 모두에 넣는다.
6. 2K sanity run을 다시 수행한다.
7. Exp4 plot/CSV가 정상인지 확인한 뒤에만 Exp5 본실험으로 간다.
8. 50K는 마지막에 핵심 점만 돌린다.

## 10. 50K 정책

현재 단계에서는 full sweep 50K를 돌리면 안 된다.

정책은 아래와 같다.

- 2K로 wrapper 정합성과 curve shape를 먼저 확인
- 그 다음 50K는 핵심 점만
  - `IEC`
  - `S-IEC` 대표 1~2점
  - 필요하면 `No correction`
  - TAC가 구현되면 `TAC`

즉 `실험 4 전체 7점 sweep + random/uniform 전부 50K`는 현재 목표가 아니다.

## 11. 현재 돌아간 결과의 해석 가이드

### 11.1 실험 4 현재 결과

현재 `real_04_tradeoff` 결과는 다음 용도로만 쓸 수 있다.

- trigger rate가 tau에 따라 움직이는지
- random/uniform보다 selectivity 경향이 있는지

현재 결과로는 다음을 주장하면 안 된다.

- S-IEC가 Pareto frontier 위에 있다
- S-IEC가 IEC보다 더 적은 NFE로 같은 FID를 달성한다
- compute-matched table이 최종본이다
- TAC 포함 비교가 끝났다

### 11.2 실험 5 현재 결과

현재 `real_05_robustness` 결과는 taxonomy가 최신 지시와 다르므로 최종 robustness evidence가 아니다.

지금 결과는 많아야 아래 정도다.

- 현재 hybrid setting들에서 S-IEC 동작 sanity check
- pilot/calibration mechanics 점검

## 12. 구현 완료 체크리스트

### 12.1 실험 4 완료 조건

- [ ] baseline 목록이 최신 지시사항과 일치한다.
- [ ] `TAC` row가 존재한다. 미구현이면 `blocked`로 표시된다.
- [ ] `Naive always-on refinement` row가 필수 baseline으로 포함된다.
- [ ] `No correction`이 진짜 deployed lower-bound path다.
- [ ] S-IEC tau sweep의 `per_sample_nfe`가 실제로 서로 다르다.
- [ ] `compute_matched.md`가 같은 sample budget 안에서 채워진다.
- [ ] plot 왼쪽에 `No correction`, `TAC`, `IEC`, `Naive always-on`, `Random`, `Uniform`, `S-IEC`가 모두 반영된다.
- [ ] plot 오른쪽은 `trigger rate vs FID`를 정상적으로 보여준다.
- [ ] wall-clock과 trace artifact가 저장된다.
- [ ] 실행 시마다 타임스탬프 폴더(`YYYYMMDD_HHMMSS`)가 생성되어 이전 실행 결과를 덮어쓰지 않는다.

### 12.2 실험 5 완료 조건

- [ ] setting taxonomy가 `fp16 / pure quant / pure DeepCache / CacheQuant` 기준으로 정리된다.
- [ ] 모든 setting에서 `No correction`, `TAC`, `IEC`, `S-IEC` 네 row가 존재한다.
- [ ] `TAC` row는 미구현이면 `blocked`로 표시된다.
- [ ] 각 setting의 `pilot -> tau recalibration`이 동작한다.
- [ ] error strength가 수치형으로 저장된다.
- [ ] left plot에 네 곡선이 모두 나온다.
- [ ] right plot에 최소 `IEC-SIEC` gain이 나오고, 가능하면 `TAC-SIEC`도 나온다.
- [ ] 실행 시마다 타임스탬프 폴더(`YYYYMMDD_HHMMSS`)가 생성되어 이전 실행 결과를 덮어쓰지 않는다.

## 13. 최종 결론

현재 코드의 핵심 결론은 단순하다.

1. 실험 4는 `wrapper는 있으나`, 현재 기본 설정으로는 NFE 축이 붕괴해 Pareto claim을 지지하지 못한다.
2. 실험 4의 `No correction`은 실제 lower bound가 아니므로 다시 구현해야 한다.
3. 실험 5는 `wrapper는 있으나`, 최신 지시사항의 error taxonomy와 TAC baseline을 반영하지 못한다.
4. TAC는 현재 미구현이므로, 지금 당장 해야 할 일은 `구현 위장`이 아니라 `skeleton + blocked 상태 명시`다.
5. 다음 실험은 이 문서 기준으로 wrapper를 먼저 보강한 뒤에만 진행해야 한다.
