# 용성 TODO: S-IEC 실험 4/5 정리 및 실행 계획

마지막 업데이트: 2026-04-25 KST

## 0. 이 문서를 다시 볼 때 가장 먼저 기억할 것

- 1저자 코드로 보이는 `IEC/mainddpm/`, `IEC/siec_core/`, `IEC/quant/`, `IEC/mainldm/` 내부 로직은 함부로 수정하지 않는다.
- 특히 sampling loop, DeepCache runner, S-IEC core 로직을 직접 바꾸는 방식은 우선 피한다.
- 필요한 구현은 가능하면 `IEC/experiments/` 아래의 실험용 wrapper/script/notebook에서 처리한다.
- 이미 생성된 샘플, 로그, tau schedule, FID 결과를 최대한 재사용한다.
- 새 실험을 돌릴 때도 먼저 작은 샘플로 dry run/pilot을 하고, 50K는 마지막에 돌린다.

## 0.1 최신 기준선과 해석 원칙

- 이 문서의 최신 source of truth는 `2026-04-25` 기준으로 다시 전달된 원저자 지시사항이다.
- 오래된 TODO 메모와 충돌할 때는 이 문서의 `8`, `9`절과 최신 postmortem, 그리고 현재 코드 구현을 우선한다.
- 현재 exp4/exp5의 실제 실행 경로는 원본 `mainddpm` 엔트리가 아니라 실험용 copy인 아래 파일들이다.
  - `IEC/experiments/yongseong/ddim_cifar_siec.py`
  - `IEC/experiments/yongseong/deepcache.py`
  - `IEC/experiments/yongseong/deepcache_denoising.py`
- TAC는 아직 구현하지 않아도 된다. 다만 최신 지시사항상 `row/CSV/plot` 비교 대상에는 반드시 이름이 남아 있어야 한다.
- S-IEC semantics의 최신 기준은 다음이다.
  - 예전의 `interval_seq` refresh point에서만 check하는 제한형이 아니라
  - 원래 S-IEC 의도와 맞게 syndrome score가 큰 timestep에서 correction이 일어나는 형태로 해석한다.
  - 현재 실험용 copy는 실제로 every-timestep lookahead check를 수행한다.

## 0.2 2026-04-25 코드 기준 체크리스트

### 실험 4

- [x] 메인 세팅 `CIFAR-10 / W8A8 + DeepCache / 100 steps`
- [x] `No correction`
- [ ] `TAC` 알고리즘
- [x] `IEC`
- [x] `Naive always-on refinement`
- [x] `Random trigger`
- [x] `Uniform periodic`
- [x] `S-IEC` 7점 sweep (`30, 50, 60, 70, 80, 90, 95`)
- [x] `FID`, `sFID`, `NFE`, `wall-clock time`, `trigger_rate`
- [x] `compute_matched.md`
- [x] pilot 1회 후 7개 `tau`
- [x] per-step syndrome plot artifact (`png/pdf/json`)
- [ ] exp4 2-panel plot에서 `TAC` 표시

### 실험 5

- [x] 메인 세팅 `CIFAR-10 / DDIM 100 steps`
- [x] 동일 모델/스케줄 고정, deployment error family만 변경
- [x] taxonomy `fp16 / w8a8 / dc10 / w4a8 / dc20 / cachequant`
- [x] setting별 pilot 재수행
- [x] setting별 `tau` 재캘리브레이션
- [x] `No correction`
- [ ] `TAC` 알고리즘
- [x] `IEC`
- [x] `S-IEC`
- [x] `FID`, `sFID`, `NFE`, `wall-clock time`, `trigger_rate`
- [x] error strength 정량화 (`per-step syndrome score 평균`)
- [x] robustness 2-panel plot 생성
- [~] `TAC` row는 존재하지만, 현재는 non-`fp16` setting에서 `blocked`이고 `fp16`은 no-op shared row만 존재

## 1. 지금 이 프로젝트가 하는 일

이 프로젝트는 IEC baseline 위에 S-IEC를 얹어서 논문 실험을 준비하는 작업이다.

핵심 아이디어는 다음과 같다.

- IEC는 efficient diffusion model에서 test-time iterative error correction을 수행한다.
- S-IEC는 syndrome score를 이용해서 correction이 필요한 시점만 선택적으로 보정하려는 방법이다.
- 목표는 단순히 모든 설정에서 FID를 더 낮추는 것이 아니라, 같은 품질에서 더 적은 compute를 쓰거나, 같은 compute에서 random/periodic trigger보다 좋은 품질을 얻는 것이다.
- 따라서 용성 담당 실험의 핵심은 `Compute-Quality Tradeoff`와 `Robustness Across Deployment Errors`를 실험적으로 보여주는 것이다.

## 2. 마감과 일정

- 초록 제출: 2026-05-04 AOE
- 한국시간 기준 예상: 2026-05-05 20:59 KST 전후
- Full paper 제출: 2026-05-06 AOE
- 한국시간 기준 예상: 2026-05-07 20:59 KST 전후
- 오늘 기준 날짜: 2026-04-23 KST
- 용성/호은에게 배정된 논문 또는 PPT 대본 훑기는 2026-04-22부터 2026-04-23까지로 잡혀 있었다.

## 3. 공유받은 자료

- Overleaf: https://www.overleaf.com/3432114893kcfdqgynzngw#fce581
- IEC baseline code: https://github.com/zysxmu/IEC
- 읽어볼 논문: https://ieeexplore.ieee.org/document/10540315
- 로컬 IEC 코드 위치: `IEC/`
- 교수님 toy/S-IEC simulation 코드 위치: `siec_sim/`

## 4. 서버와 실행 환경

실험은 반드시 4090 서버에서 수행한다. Pro6000은 Blackwell이라 기존 baseline이 잘 안 돌 수 있다고 전달받았다.

SSH 정보:

```ssh-config
Host 10.150.20.205
    HostName 10.150.29.188
    User user
```

환경:

```bash
conda activate iec
```

GPU 확인:

```bash
watch -d -n 1 nvidia-smi
```

긴 실험은 `tmux` 안에서 실행한다. SSH가 끊겨도 실험이 계속 돌게 하기 위해서다.

## 5. 팀별 역할

### 조은

- IEC baseline 재현
- 논문 Table 3과 비슷한지 확인
- IEC 저자 코드에 교수님 `siec.py` 로직을 sampling loop에 주입
- S-IEC sanity 및 IEC vs S-IEC 비교 진행

### 호은

- 교수님 toy 코드 기반 실험 담당
- `toy_01_observability`
- `toy_02_selective_stabilization`
- `toy_03_syndrome_mmse_kl`
- 선택적으로 real diffusion proxy 실험
  - timestep별 syndrome score와 FID 상관관계
  - W8에서 W4로 양자화 강도 증가 시 syndrome energy 증가 여부

### 용성

- 실험 4: Compute-Quality Tradeoff
- 실험 5: Robustness Across Deployment Errors
- 우선순위는 실험 4 CIFAR-10을 먼저 완성하는 것이다.

## 6. 현재 코드베이스 진행 상황

루트 `/home/user/jowithu/Semantic` 자체는 git repo가 아니고, `IEC/` 내부가 git repo다.

현재 확인된 주요 파일:

- `IEC/mainddpm/ddim_cifar_siec.py`
  - CIFAR-10 S-IEC sampling entrypoint
  - `--tau_path`, `--tau_percentile`, `--siec_always_correct`, `--siec_collect_scores`, `--siec_scores_out`, `--siec_max_rounds` 등이 있음
- `IEC/mainddpm/calibrate_tau_cifar.py`
  - pilot score에서 tau schedule을 만드는 스크립트
- `IEC/siec_core/syndrome.py`
  - syndrome score 계산
- `IEC/siec_core/correction.py`
  - consensus correction 계산
- `IEC/mainddpm/ddpm/runners/deepcache.py`
  - DeepCache runner
  - 이미 S-IEC branch가 들어가 있음
- `IEC/mainddpm/ddpm/functions/deepcache_denoising.py`
  - 실제 S-IEC sampling 함수가 있음
- `IEC/experiments/real_01_baseline_reproduction.ipynb`
  - baseline reproduction 관련 notebook
- `IEC/experiments/real_02_siec_sanity.ipynb`
  - S-IEC sanity notebook
- `IEC/experiments/real_03_iec_vs_siec_fid.py`
  - 기존 sample NPZ들의 FID 비교 스크립트
- `siec_sim/run_all.py`
  - 교수님 toy 실험 suite
- `siec_sim/core/perturbations.py`
  - 교수님 toy perturbation 정의

현재 없는 파일:

- `IEC/experiments/real_04_tradeoff.ipynb`
- `IEC/experiments/real_04_tradeoff.py`
- `IEC/experiments/real_05_robustness.ipynb`
- `IEC/experiments/real_05_robustness.py`

단, 앞으로는 1저자 core code를 수정하지 않는 방향이므로, 이 파일들은 실험 orchestration과 결과 정리용으로만 만드는 것이 안전하다.

## 7. 현재까지 나온 결과

### Baseline reproduction

공유받은 상태:

- 기존 논문 Table 3: `FID 6.47`
- 현재 재현 결과: `FID 약 7.2`
- GPU 차이나 실행 환경 차이를 고려하면 baseline은 대체로 맞게 재현된 것으로 판단됨

로컬 로그에서 확인된 값:

- CIFAR-10 W8A8 IEC 50K FID: `7.2048`

### 기존 50K 비교 결과

파일:

- `IEC/experiments/iec_vs_siec_fid_results.txt`

결과:

| Method | FID |
| --- | ---: |
| IEC | 7.20 |
| IEC+Recon | 7.85 |
| S-IEC p80 | 8.79 |
| No Correction | 22.56 |

추가 로그에서 확인된 값:

| Method | FID | sFID |
| --- | ---: | ---: |
| IEC+Recon | 7.8545 | 6.8136 |
| S-IEC p80 | 8.7873 | 5.8383 |
| No Correction | 22.5610 | 49.4525 |

### 공유받은 quick-mode 결과

2,000 samples:

| Method | FID |
| --- | ---: |
| IEC | 30.65 |
| S-IEC p80 | 31.53 |

해석:

- 현재 S-IEC p80은 FID만 보면 IEC보다 조금 나쁘다.
- 따라서 논문에서 이길 수 있는 포인트는 `raw FID 개선`이 아니라 `compute 절약`, `trigger selectivity`, `tau에 따른 smooth tradeoff`다.

## 8. 용성이 맡은 실험 4: Compute-Quality Tradeoff

### 목표

실제 diffusion model에서 S-IEC가 compute-quality tradeoff를 만든다는 것을 보인다.

즉 다음 중 하나라도 보여야 한다.

- NFE 절약 승리: `NFE < IEC` 이면서 `FID <= IEC + margin`인 S-IEC 점이 있다.
- Selectivity 승리: 같은 NFE에서 S-IEC가 Random trigger와 Uniform periodic보다 FID가 좋다.
- Spectrum 승리 (Corollary 6.9 검증): tau percentile을 바꾸면 NFE-FID curve가 부드럽고 단조롭게 움직인다.

### 메인 설정

- Dataset: CIFAR-10
- Model/error setting: W8A8 + DeepCache
- Sampling steps: 100
- LSUN, SD on COCO는 CIFAR-10에서 패턴이 보인 뒤 확장한다.

### 비교 대상

원저자 최신 지시사항 기준으로 반드시 포함해야 하는 방법:

- No correction
- TAC
- IEC 저자 구현
- Naive always-on refinement
  - 현재 코드에서는 S-IEC always-correct와 대응
  - `--siec_always_correct`
- Random trigger
  - S-IEC p80의 실측 trigger rate와 맞춰야 함
- Uniform periodic
  - 같은 NFE budget에 맞춰야 함
- S-IEC tau sweep
  - tau percentile: `{30, 50, 60, 70, 80, 90, 95}`

### 코드 이행 상태 (2026-04-25 기준)

- [x] `No correction`
- [ ] `TAC` 알고리즘
- [x] `IEC`
- [x] `Naive always-on refinement`
- [x] `Random trigger`
- [x] `Uniform periodic`
- [x] `S-IEC` 7점 sweep
- [x] `compute-matched` row
- [x] pilot 1회 후 7개 `tau`
- [x] per-step syndrome score distribution artifact
- [ ] exp4 plot에 `TAC` 표시는 아직 없음

주의:

- 최신 실험용 copy에서 S-IEC는 `interval_seq` refresh point에서만 check하지 않는다.
- 현재 구현은 거의 모든 reverse timestep에서 lookahead check를 수행하므로, exp4에서 `NFE saving victory`는 구조적으로 매우 어렵다.
- 따라서 exp4의 현재 실패 원인은 DeepCache 오구현이라기보다 현재 S-IEC check semantics와 operating point에 더 가깝다.

### 지표

반드시 기록:

- FID
- sFID
- total NFE
- wall-clock time
- average trigger rate
- per-step syndrome score distribution

### plot

2-panel plot:

- 왼쪽: NFE vs FID scatter
  - S-IEC 7개 tau 점을 선으로 연결
  - baseline 점들을 같이 표시
- 오른쪽: Trigger rate vs FID
  - tau가 control knob 역할을 하는지 보여야 함

### table

반드시 compute-matched row를 포함한다.

- IEC와 NFE가 가장 가까운 S-IEC 점을 찾는다.
- IEC vs compute-matched S-IEC를 Table 1 스타일로 명시 비교한다.

### 비용 절약 원칙

- Pilot run은 한 번만 한다.
- 같은 pilot score에서 percentile만 바꿔 7개 tau schedule을 만든다.
- 같은 샘플 수와 seed 설정을 최대한 맞춘다.

## 9. 용성이 맡은 실험 5: Robustness Across Deployment Errors

### 목표

S-IEC가 여러 deployment error 유형에서도 일관되게 작동하며, error 강도가 커질수록 IEC 대비 상대 이득이 커진다는 것을 보인다.

즉 다음 중 하나라도 보여야 한다.

- Trend 승리 (주 목표): error 강도가 커질수록 `(FID_IEC − FID_S-IEC)` 격차가 단조 증가한다. 논문 Figure 4의 real diffusion 버전 재현.
- Regime 승리: 약한 error 세팅(fp16, W8A8)에서는 S-IEC가 IEC와 비슷하거나 약간 뒤져도, 강한 세팅(W4A8, CacheQuant)에서는 명확히 이긴다.
- Consistency 승리 (약한 목표): 모든 세팅에서 S-IEC가 No correction보다 일관되게 개선된다. 최소한 퇴보는 없다.

### 메인 설정

- Dataset: CIFAR-10
- Sampling: DDIM, 100 steps
- 모델·스케줄은 동일하게 고정하고, deployment error 유형만 바꾼다.

LSUN, SD on COCO는 CIFAR-10에서 패턴이 보인 뒤 확장한다.

### Deployment error 설정 (error 강도 순)

아래 순서로 정렬해서 "error 강도 ↑ vs 이득 ↑" 관계를 시각화한다.

1. fp16 — 가장 약한 error
2. W8A8 quantization
3. DeepCache 단독, `replicate_interval=10`
4. W4A8 quantization
5. DeepCache 공격적, `replicate_interval=20` 또는 `50`
6. CacheQuant = W4A8 + DeepCache — 가장 강한 error

### 비교 대상 (각 세팅마다)

- No correction
- TAC
- IEC 저자 구현
- S-IEC — 실험 4에서 결정된 고정 τ percentile 사용

주의:

- 원저자 최신 지시사항상 exp5의 비교 프레임에는 `TAC`가 반드시 이름으로 남아 있어야 한다.
- 다만 현재 TAC 알고리즘은 미구현이므로, row/CSV/plot 레벨의 skeleton만 유지하는 것이 맞다.

### 코드 이행 상태 (2026-04-25 기준)

- [x] `fp16 / w8a8 / dc10 / w4a8 / dc20 / cachequant`
- [x] setting별 pilot 재수행
- [x] setting별 `tau` 재캘리브레이션
- [x] `No correction`
- [ ] `TAC` 알고리즘
- [x] `IEC`
- [x] `S-IEC`
- [x] error strength 정량화
- [x] robustness 2-panel plot
- [~] `TAC` row는 남아 있으나, non-`fp16`은 `blocked`이고 `fp16`은 no-op shared row만 존재

해석 주의:

- `fp16`의 shared row는 no-op regime에서 plotting/CSV alignment를 위한 편의 처리다.
- 이것을 TAC 구현 완료의 증거로 해석하면 안 된다.
- strict policy상 TAC는 여전히 미구현이다.

### 지표

- FID, sFID
- per-step syndrome score 평균 (error 강도의 proxy)
- trigger rate (S-IEC의 실제 발동 빈도)
- 총 NFE, wall-clock time

### plot

2-panel plot:

- 왼쪽: X축 = deployment error 강도(유형 순서 또는 per-step syndrome score 평균), Y축 = FID. 최신 지시사항 기준으로는 `No correction`, `TAC`, `IEC`, `S-IEC`를 비교 대상으로 둔다.
- 오른쪽: X축 = error 강도, Y축 = `FID_IEC − FID_S-IEC`를 기본으로 둔다. 가능하면 `FID_TAC − FID_S-IEC`도 함께 본다. 논문 Figure 4의 real 버전.

### 필수 포함

- Error 강도 축 정량화: 단순 유형 나열이 아니라, 각 세팅의 per-step syndrome score 평균 또는 No correction 대비 FID 증가폭을 가로축으로 사용해 `error 심각도 vs 이득` 관계를 수치로 제시한다.
- 각 세팅별 τ 재캘리브레이션: error 분포가 세팅마다 다르므로 pilot을 각 세팅별로 재수행한다. pilot은 수백 trajectory면 충분하므로 비용은 크지 않다.

### 리스크와 현재 가능성

- CIFAR 쪽 W8A8 관련 파일은 확보되어 있다.
- W4A8, `dc20`, `cachequant`까지 현재 최신 run에서는 runnable 상태로 풀린 적이 있다.
- `fp16` path도 실험용 copy에서 no-op regime으로 정리돼 있다.
- `real_05_robustness.py`는 이미 inventory, pilot, recalibration, main run, plot 생성까지 수행한 상태다.
- 남은 핵심 리스크는 asset 부재가 아니라 `TAC` 미구현과 robustness claim 미성립이다.

따라서 실험 5의 다음 단계는 inventory 작성이 아니라, 현재 결과를 바탕으로 `p80` 고정 운영점과 family별 반응 차이를 어떻게 해석/보강할지 결정하는 것이다.

## 10. 앞으로 할 일: 안전한 실행 순서

### 1단계: 1저자 코드 변경 없이 현재 산출물 정리

해야 할 일:

- `IEC/experiments/iec_vs_siec_fid_results.txt` 결과를 표로 정리한다.
- 기존 로그에서 FID/sFID를 다시 모은다.
- 현재 있는 tau schedule을 확인한다.
  - `tau_schedule_p50.pt`
  - `tau_schedule_p70.pt`
  - `tau_schedule_p80.pt`
  - `tau_schedule_p90.pt`
  - `pilot_scores_nb.pt`
- 없는 tau schedule을 확인한다.
  - p30
  - p60
  - p95

주의:

- 이 단계에서는 `IEC/mainddpm` 내부를 수정하지 않는다.

### 2단계: missing tau schedule 생성

가능하면 기존 `pilot_scores_nb.pt`를 사용한다.

예상 명령:

```bash
cd IEC
conda run -n iec python mainddpm/calibrate_tau_cifar.py \
  --scores_path ./calibration/pilot_scores_nb.pt \
  --percentile 30 \
  --out_path ./calibration/tau_schedule_p30.pt

conda run -n iec python mainddpm/calibrate_tau_cifar.py \
  --scores_path ./calibration/pilot_scores_nb.pt \
  --percentile 60 \
  --out_path ./calibration/tau_schedule_p60.pt

conda run -n iec python mainddpm/calibrate_tau_cifar.py \
  --scores_path ./calibration/pilot_scores_nb.pt \
  --percentile 95 \
  --out_path ./calibration/tau_schedule_p95.pt
```

주의:

- 이건 1저자 core code 수정이 아니라 기존 calibration script 실행이다.
- 그래도 먼저 작은 테스트로 파일이 잘 로드되는지 확인한다.

### 3단계: 실험 4용 wrapper만 작성

새 파일 후보:

- `IEC/experiments/real_04_tradeoff.py`

이 파일은 core sampling logic을 바꾸지 말고 다음만 담당한다.

- 실행할 command 목록 생성
- 결과 파일 경로 관리
- 이미 있는 결과는 재사용
- 새 결과가 없으면 사용자가 실행할 명령 출력
- FID/sFID 로그 parsing
- tau별 결과 CSV/JSON 작성
- plot 생성

처음부터 자동으로 50K를 전부 돌리는 스크립트보다는, `--dry-run`을 기본값으로 두는 것이 안전하다.

필요 기능:

- `--dry-run`
- `--num-samples`
- `--sample-batch`
- `--percentiles 30 50 60 70 80 90 95`
- `--results-dir experiments/results/real_04_tradeoff`
- 기존 결과 파일 parsing
- 누락된 run 명령 출력

### 4단계: NFE와 trigger rate 집계 상태

현재는 실험용 copy에서 trace가 저장되고 wrapper가 이를 직접 집계한다.

- `exp4`: `real_04_tradeoff.py`가 trace에서 `per_sample_nfe`, `trigger_rate`, `syndrome_mean`을 집계한다.
- `exp5`: `real_05_robustness.py`가 setting별 trace에서 동일 지표를 집계한다.
- 따라서 현재 문서와 결과 표에서는 proxy가 아니라 wrapper가 저장한 trace 기반 수치를 우선 사용한다.

### 5단계: Random/Uniform baseline 처리

Random trigger와 Uniform periodic은 현재 원본 core가 아니라 `IEC/experiments/yongseong/` 실험용 copy에 구현되어 있다.

- `deepcache_denoising.py`의 experiment copy가 `trigger_mode=random|uniform`을 지원한다.
- exp4 wrapper는 이 실험용 copy를 호출해 matched baseline을 실제로 생성한다.
- 따라서 이 둘은 더 이상 TODO가 아니라 `실험용 copy 범위에서 구현 완료` 상태다.

### 6단계: 실험 5용 wrapper 작성

새 파일 후보:

- `IEC/experiments/real_05_robustness.py`

역할:

- 6개 error 강도 세팅(fp16, W8A8, DeepCache interval=10, W4A8, DeepCache interval=20/50, CacheQuant=W4A8+DeepCache)별로 필요한 파일이 있는지 검사
- 실행 가능한 command를 출력
- 빠진 prerequisite을 명시
- 결과 로그 parsing (FID, sFID, per-step syndrome score 평균, trigger rate, NFE, wall-clock)
- 각 세팅별 τ 재캘리브레이션 job 목록 생성
- robustness summary table 생성 (error 강도 축 정량화 포함)
- 2-panel plot 생성 (error 강도 vs FID 3곡선, error 강도 vs `FID_IEC − FID_S-IEC`)

현재 상태:

- inventory 기능은 이미 구현되어 있다.
- per-setting pilot, tau recalibration, main run, plot 생성까지 한 번 돌아간 상태다.
- 따라서 다음 작업은 wrapper 작성이 아니라 문서 정합화와 후속 ablation 설계다.

반드시 확인할 파일/경로:

- fp16 CIFAR sampling 경로
- W8A8 관련 CIFAR 파일 (이미 확보됨)
- W4A8 관련 CIFAR 파일 (`pre_quanterr_*weight4*`, quantizer params W4 등)
- `pre_cacheerr_*` (DeepCache)
- DeepCache `replicate_interval=10/20/50` 각 설정 실행 가능 여부
- CacheQuant(W4A8 + DeepCache) 조합 스크립트/파이프라인
- noquant/nocache script
- 각 세팅별 pilot 실행 후 tau schedule 저장 경로 규약

### 7단계: 실제 실행 우선순위

가장 먼저 돌릴 것 (실험 4 메인):

1. CIFAR-10, W8A8 + DeepCache, 100 steps, small sample
2. tau p80 기존 결과 재확인
3. tau p50/p70/p90 기존 schedule로 small sample
4. p30/p60/p95 schedule 생성 후 small sample
5. 결과가 말이 되면 2K sample
6. 2K에서 경향이 있으면 50K run

그 다음 (실험 5, error 강도 순):

1. fp16 pilot → τ 재캘리브레이션 → small sample
2. W8A8 (이미 데이터 있음) 재정리
3. DeepCache interval=10 단독
4. W4A8 pilot → τ 재캘리브레이션 → small sample
5. DeepCache interval=20 또는 50 공격적
6. CacheQuant (W4A8 + DeepCache)

마지막에 돌릴 것:

1. LSUN
2. SD on COCO

## 11. 다음 Codex 세션에서 해야 할 구체 작업

다음에 이 문서를 보고 이어갈 때는 아래 순서로 진행한다.

1. `git -C IEC status --short`로 상태 확인
2. `IEC/mainddpm` core 파일은 수정하지 말라는 제약 재확인
3. `docs/exp4_postmortem_20260425.md`, `docs/exp5_postmortem_20260425.md`, 이 TODO의 정합성부터 확인
4. `IEC/calibration/`의 tau schedule과 setting별 pilot 파일 목록 확인
5. exp4 후속은 `tau / c_siec / siec_max_rounds` 조정으로 `Selectivity` 또는 `Spectrum`을 살릴 수 있는지 본다
6. exp4에서 `NFE saving`을 다시 노릴 경우에는 `check sparsification`이 별도 설계 항목임을 유지한다
7. exp5 후속은 `p70 / p80 / p90` sweep, family별 분리 해석, error-strength proxy 추가를 우선 검토한다
8. TAC는 계속 `blocked / not implemented`로 유지한다
9. heavy GPU run은 사용자가 명시적으로 요청하거나 tmux/4090 서버 조건이 확인된 뒤 실행

## 12. 현재 롤백 상태 메모

이 문서를 작성하기 직전에 다음 요청이 있었다.

- `IEC/mainddpm/ddpm/functions/deepcache_denoising.py`에 추가했던 random/uniform trigger 및 trace return 관련 수정은 롤백한다.
- `IEC/mainddpm/ddpm/runners/deepcache.py`에 추가하려던 diagnostics 저장 수정도 유지하지 않는다.
- 이유: 1저자 코드를 직접 수정하는 것은 부담스럽기 때문에, 우선 core code는 건드리지 않는 방향으로 간다.

따라서 앞으로는 실험 관리/결과 정리/plot 생성 중심으로 `IEC/experiments/` 아래에서 작업한다.
