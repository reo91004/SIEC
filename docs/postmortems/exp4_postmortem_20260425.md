# 실험 4 Postmortem (2026-04-25)

기준 run:

- `/home/user/jowithu/Semantic/IEC/experiments/yongseong/results/real_04_tradeoff/20260425_104700`

이 문서는 최신 exp4 run이 재설계 지시사항을 얼마나 충족했는지, 무엇이 실제로 실패했는지, 다음에 어떤 파라미터를 바꿔야 하는지를 결과 중심으로 정리한다.

## 1. 실행 완료 여부

실험은 정상 완료됐다.

- `results.csv`, `results.json`, `compute_matched.md`, `tradeoff_2panel.png/.pdf` 생성 완료
- `artifacts/` 아래 syndrome histogram/json 생성 완료
- row 상태: `12 completed`, `1 blocked`
- blocked 1개는 의도된 `TAC not implemented`

즉, 실행 실패가 아니라 결과 해석 단계다.

## 2. 지시사항 이행 여부

## 2.0 quick checklist

아래는 최신 재설계 지시사항을 항목별로 다시 체크한 표다.

| 항목 | 판정 | 메모 |
|---|---|---|
| `CIFAR-10`, `W8A8 + DeepCache`, `100 steps` | 충족 | 최신 run이 해당 세팅으로 완료됨 |
| `No correction` baseline | 충족 | deployed path로 `NFE=100` 확인 |
| `TAC` baseline row | 부분 충족 | row/CSV skeleton은 있으나 수치 결과 없음. 현재 exp4 plot에는 TAC marker가 없다 |
| `IEC` baseline | 충족 | fresh 2K run 완료 |
| `Naive always-on refinement` | 충족 | `--siec_always_correct` row 완료 |
| `Random trigger` | 충족 | `p80` 실측 trigger rate 기준으로 다시 매칭 완료 |
| `Uniform periodic` | 충족 | `p80`와 동일 예산 근사로 다시 매칭 완료 |
| `S-IEC` 7점 sweep | 충족 | `30,50,60,70,80,90,95` 완료 |
| `FID`, `sFID` | 충족 | 전 row 수치 채워짐 (`TAC` 제외) |
| `total NFE`, `wall-clock time` | 충족 | trace/FID 시간 집계 포함 |
| 평균 `trigger_rate` | 충족 | 전 row 수치 채워짐 (`TAC` 제외) |
| `compute-matched` 행 | 충족 | 생성됨 |
| 2-panel plot | 부분 충족 | 파일은 생성되지만, 현재 exp4 plot에는 TAC marker가 없다 |
| syndrome distribution analysis | 충족 | raw trace와 method별 histogram/json 생성 |
| per-step syndrome score distribution plot | 충족 | method별 `syndrome_timestep_{slug}.png/.pdf` 자동 생성 |
| pilot 1회 후 7개 tau 생성 | 충족 | `run_exp4_refresh.sh` 경로로 정합하게 재생성됨 |

요약:

- `TAC` 알고리즘은 여전히 없다.
- exp4 plot에는 아직 `TAC`가 표시되지 않는다.
- 그 외 구현/실행 구조는 거의 다 맞다.

## 2.1 요구 baseline

지시사항의 baseline 요구는 아래였다.

- `No correction`
- `TAC`
- `IEC`
- `Naive always-on refinement`
- `Random trigger`
- `Uniform periodic`
- `S-IEC`

최신 run 기준 판정:

| 항목 | 상태 | 판정 |
|---|---|---|
| `No correction` | 수치 결과 있음 | 충족 |
| `TAC` | row 존재, `blocked` | row/CSV는 있으나 plot에는 아직 표시되지 않음 |
| `IEC` | 수치 결과 있음 | 충족 |
| `Naive always-on refinement` | 수치 결과 있음 | 충족 |
| `Random trigger` | p80 실측 trigger rate 기준으로 다시 매칭 | 충족 |
| `Uniform periodic` | p80와 동일 NFE 예산 근사로 다시 매칭 | 충족 |
| `S-IEC` 7점 sweep | `p30,50,60,70,80,90,95` 전부 완료 | 충족 |

strict하게 보면 `TAC`가 수치 결과가 없고 exp4 plot에도 아직 표시되지 않으므로 "완벽히" 충족했다고는 못 한다.

`TAC`를 제외하면 baseline 구성은 현재 요구사항과 맞다.

## 2.2 필수 구성 요소

| 항목 | 상태 | 판정 |
|---|---|---|
| `CIFAR-10`, `W8A8 + DeepCache`, `100 DDIM steps` | 맞음 | 충족 |
| pilot 1회 후 7개 tau 생성 | 맞음 | 충족 |
| `FID`, `sFID` | 채워짐 | 충족 |
| `total NFE`, `wall-clock time` | 채워짐 | 충족 |
| 평균 `trigger_rate` | 채워짐 | 충족 |
| `compute_matched.md` | 생성됨 | 충족 |
| 2-panel plot | 생성됨 | 충족 |
| syndrome 분석 artifact | 생성됨 | 충족 |

즉, 최신 exp4는 `TAC` 미구현과 exp4 plot의 `TAC` 미표시를 제외하면 재설계 요구사항을 거의 다 이행했다.

## 3. 최신 수치 요약

핵심 row만 추리면 아래와 같다.

| Method | FID | sFID | trigger_rate | per_sample_NFE |
|---|---|---|---|---|
| No correction | 46.9833 | 55.8912 | 0.0000 | 100.00 |
| IEC | 44.2872 | 55.1975 | 1.0000 | 110.00 |
| Always-on | 46.1661 | 55.7665 | 1.0000 | 298.00 |
| S-IEC p30 | 46.1853 | 55.7543 | 1.0000 | 298.00 |
| S-IEC p50 | 46.2896 | 55.7966 | 0.9066 | 288.75 |
| S-IEC p60 | 47.3508 | 55.9785 | 0.1667 | 215.50 |
| S-IEC p70 | 47.0564 | 55.9031 | 0.0253 | 201.50 |
| S-IEC p80 | 46.9470 | 55.8818 | 0.0202 | 201.00 |
| S-IEC p90 | 46.9977 | 55.8885 | 0.0202 | 201.00 |
| S-IEC p95 | 46.9479 | 55.8816 | 0.0202 | 201.00 |
| Random matched to p80 | 46.9063 | 55.8735 | 0.0177 | 200.75 |
| Uniform matched to p80 | 46.9618 | 55.8727 | 0.0303 | 202.00 |

`compute_matched.md`도 불리하다.

- `IEC`: `44.2872 @ 110.00 NFE`
- closest `S-IEC`: `p80`, `46.9470 @ 201.00 NFE`

## 4. 무엇이 정확히 실패했는가

## 4.1 NFE 절약 승리 실패

지시사항의 첫 번째 승리 조건은:

- `NFE < IEC`
- 동시에 `FID <= IEC + margin`

이었지만, 현재 S-IEC의 최소 NFE는 약 `201`이고 IEC는 `110`이다.

즉, 최신 구현에서는 어떤 tau를 골라도 S-IEC가 IEC보다 싸지지 않았다.

이건 단순히 tau가 나빠서가 아니라 현재 NFE accounting 구조 때문이기도 하다.

- `IEC`는 `100 + refresh 재평가 비용` 수준이라 `110`
- 현재 `S-IEC`는 거의 모든 reverse timestep에서 lookahead check를 수행하므로 기본 바닥이 대략 `199`
- `max_rounds=2`에서는 trigger가 걸릴 때마다 추가 re-lookahead가 붙어서 `200+`가 된다

결과적으로 현재 구현/계수 방식에서는 `NFE saving victory`가 구조적으로 나오기 어렵다.

## 4.2 Quality parity 실패

좋은 S-IEC 점은 IEC FID 근처로 가야 하는데, 실제론 그렇지 않았다.

- `IEC`: `44.2872`
- 가장 좋은 S-IEC: `p30 = 46.1853`
- `p80 = 46.9470`

즉 S-IEC는 `No correction`보다는 조금 낫지만, IEC 수준으로 회복하지 못했다.

## 4.3 Selectivity 승리 실패

현재 fairness는 맞췄다.

- `S-IEC p80`: `trigger_rate 0.0202`, `NFE 201.00`, `FID 46.9470`
- `Random matched`: `0.0177`, `200.75`, `46.9063`
- `Uniform matched`: `0.0303`, `202.00`, `46.9618`

즉 이제는 같은 예산에서 비교하고 있다.

그런데 결과는:

- S-IEC가 random을 확실히 이기지 못함
- uniform과도 사실상 비슷함

즉 syndrome-based selectivity가 blind trigger보다 의미 있게 낫다고 말하기 어렵다.

## 4.4 Spectrum 승리도 약함

지시사항의 세 번째 승리 조건은 tau에 따라 곡선이 단조롭고 부드럽게 움직이는 것이었다.

이번 run에서는:

- trigger rate는 대체로 내려간다
- NFE도 대체로 내려간다
- 하지만 FID는 부드럽고 단조롭게 좋아지지 않는다

예:

- `p50`: `46.2896`
- `p60`: `47.3508`
- `p70`: `47.0564`
- `p80`: `46.9470`

즉 control knob 자체는 존재하지만, quality 쪽 반응이 매끈한 tradeoff curve라고 보긴 어렵다.

## 5. 왜 이런 결과가 나왔는가

실제 원인은 크게 두 가지다.

### 5.1 check 비용이 너무 비싸다

현재 S-IEC는 correction이 거의 안 일어나도, lookahead syndrome check 자체가 비싸다.

그래서:

- `p80`, `p90`, `p95`처럼 거의 안 고쳐도 `NFE ~= 201`
- 이미 여기서 `IEC=110`을 크게 넘는다

즉 현재 semantics를 유지하면 `tau`만으로는 IEC보다 싼 점을 만들 수 없다.

### 5.2 correction이 품질을 충분히 회복시키지 못한다

낮은 tau:

- correction이 너무 자주 걸림
- NFE는 크게 증가
- FID는 IEC만큼 좋아지지 않음

높은 tau:

- correction이 거의 안 걸림
- NFE는 여전히 높음
- FID는 no-correction 근처에 머묾

즉 현재 `tau` sweep은 "많이 고쳐도 별 이득 없고, 적게 고쳐도 기본 check 비용이 너무 큰" 상태다.

## 6. 다음에 어떤 파라미터를 바꿔야 claim 가능성이 생기는가

먼저 결론:

- 현재 구현 그대로라면 `NFE saving victory`는 파라미터만으로 만들기 어렵다.
- 현재 구현에서 노려볼 수 있는 것은 `Selectivity` 또는 `Spectrum`이다.

즉, 아래 제안은 두 층으로 나뉜다.

### 6.1 현재 구현을 유지한 채 먼저 해볼 파라미터

#### A. `tau percentile`을 높은 구간에 집중

현재 7점은 `30,50,60,70,80,90,95`인데, 유의미한 비교 구간은 사실상 `70+`다.

다음 후보:

- `70, 80, 85, 90, 95, 97, 99`

이유:

- `30,50`은 거의 always-on이라 baseline 의미가 약하다
- `80~99` 구간에서만 low-trigger regime selectivity를 좀 더 볼 수 있다

#### B. `c_siec` sweep

현재 correction gain `c_siec=1.0` 고정인데, correction이 약하거나 과할 수 있다.

다음 후보:

- `0.25, 0.5, 1.0, 2.0`

목표:

- 같은 trigger budget에서 FID가 실제로 내려가는 gain 구간 찾기

#### C. `siec_max_rounds` 비교

현재 `2`인데, 추가 round가 품질 개선보다 NFE 증가가 더 클 수 있다.

비교 후보:

- `1`
- `2`

주의:

- `1`로 줄여도 check 바닥 비용은 그대로라 `NFE saving victory`는 여전히 어렵다
- 다만 selectivity 비교에서는 더 깔끔해질 수 있다

#### D. per-step syndrome plot은 이제 구현됨

현재 wrapper는 아래 산출물을 자동 생성한다.

- `syndrome_timestep_{slug}.png`
- `syndrome_timestep_{slug}.pdf`
- `syndrome_timestep_summary_{slug}.json`

요약 통계:

- mean
- median
- p10/p90 band

이 산출물은 claim을 직접 살려주진 않지만, "control knob가 어느 step에서 작동하는가"를 설명하는 분석 근거로 유용하다.

### 6.2 NFE saving claim을 정말 원하면 필요한 것

이건 단순 tau 조정이 아니라 사실상 알고리즘/계수 방식 변화다.

현재 구현은 "거의 모든 timestep에서 check"를 수행하므로 `NFE < IEC`가 사실상 불가능하다.

따라서 NFE-saving claim을 다시 열려면 다음 류의 새 파라미터가 필요하다.

- `check_period = k`: 매 `k` step마다만 syndrome check
- `check_topk_steps`: pilot에서 syndrome이 큰 timestep subset만 check
- `check_interval_seq_only`: DeepCache refresh step에서만 check

즉, 지금 상태에서 `tau`만 바꿔서는 첫 번째 승리 조건을 충족시키기 어렵다.

## 7. 최종 판정

strict하게 보면 최신 exp4는 아직 "완벽 이행"이 아니다.

이유:

- `TAC`는 여전히 blocked skeleton이며 exp4 plot에는 아직 표시되지 않음
- 무엇보다 승리 조건 3개를 모두 만족하지 못함

하지만 `TAC`를 제외한 구현/실행 구조 자체는 거의 다 맞다.

즉 현재 상태를 가장 정확히 표현하면:

- 구현 구조: 대체로 완성
- 실행 정합성: 확보
- 결과 claim: 실패

따라서 다음 단계는 "버그 수정"보다는 아래 둘 중 하나다.

1. 현재 구현 안에서 `tau / c_siec / max_rounds`를 다시 스윕해서 `Selectivity` 또는 `Spectrum`이라도 살릴지 결정
2. 정말 `NFE saving victory`가 목표라면, check sparsification을 새 파라미터로 도입할지 결정
3. per-step syndrome 전용 plot을 해석에 실제로 활용할지 결정

## 8. 다음 액션 계획

기준 run:

- `/home/user/jowithu/Semantic/IEC/experiments/yongseong/results/real_04_tradeoff/20260425_104700`

이 섹션은 최신 exp4 결과를 바탕으로, 다음에 무엇을 바꿔서 어떤 claim 가능성을 볼 것인지 정리한다.

### 8.1 현재 판정

현재 exp4는 아래 상태다.

- 실행 정합성: 확보
- matched baseline fairness: 확보
- `NFE saving victory`: 실패
- `Selectivity victory`: 실패
- `Spectrum victory`: 약함

즉 다음 액션은 "버그 수정"보다 "어떤 종류의 claim을 아직 노릴 수 있는가"를 정하는 문제다.

### 8.2 우선순위

우선순위는 아래처럼 두 층으로 나눈다.

#### A. 현재 구현 유지

목표:

- `Selectivity`
- `Spectrum`

이 층에서는 구조를 거의 안 바꾸고 아래 파라미터만 다시 본다.

#### B. 구조 변경

목표:

- `NFE saving`

이 층은 사실상 새 실험이다. 현재처럼 매 timestep check를 유지하면 `IEC=110`보다 낮은 NFE 점을 만들기 어렵기 때문이다.

### 8.3 즉시 해볼 ablation

#### 8.3.1 tau high-percentile sweep 재설계

현재 7점:

- `30, 50, 60, 70, 80, 90, 95`

다음 후보:

- `70, 80, 85, 90, 95, 97, 99`

이유:

- `30`, `50`은 거의 always-on에 가까워 정보량이 적다
- 현재 의미 있는 구간은 low-trigger regime이다
- matched random/uniform도 `p80` 근처에서 비교하므로 그 주변을 더 조밀하게 보는 편이 낫다

예상:

- quality가 크게 좋아질 가능성은 낮다
- 대신 `Spectrum`의 매끈함을 좀 더 정확히 확인할 수 있다

#### 8.3.2 `c_siec` sweep

현재:

- `c_siec = 1.0`

다음 후보:

- `0.25`
- `0.5`
- `1.0`
- `2.0`

이유:

- correction이 약해서 효과가 없는지
- correction이 과해서 오히려 품질을 망치는지

를 분리해서 봐야 한다.

추천 방법:

- 먼저 `p80` 고정
- 그 다음 `c_siec`만 스윕
- 가장 나은 `c_siec`를 찾은 뒤 tau sweep을 다시 수행

#### 8.3.3 `siec_max_rounds` 비교

현재:

- `siec_max_rounds = 2`

다음 비교:

- `1`
- `2`

이유:

- 현재 결과에서는 extra round가 품질을 충분히 끌어올리지 못하는 반면 NFE는 확실히 늘린다
- `max_rounds=1`이 selectivity 비교에는 더 깔끔할 수 있다

주의:

- 이 변경만으로 `NFE saving`은 열리지 않는다
- 하지만 `Random/Uniform` 대비 우위는 더 선명해질 수 있다

### 8.4 per-step syndrome plot 해석 활용

자동 생성 자체는 이미 들어갔다.

현재 산출물:

- `syndrome_timestep_{slug}.png/.pdf`
- `syndrome_timestep_summary_{slug}.json`

다음 단계는 생성이 아니라 해석이다.

권장 row:

- `S-IEC p80`
- `Random matched`
- `Uniform matched`
- `Always-on`

볼 것:

- 어떤 timestep 구간에서 syndrome이 크게 튀는가
- low-trigger tau가 어느 구간을 사실상 무시하는가
- random/uniform과 비교해 S-IEC가 특정 구간을 더 잘 고르는가

### 8.5 구조 변경 없이는 어려운 claim

현재 구현에서는 S-IEC가 거의 모든 step에서 check를 한다.

그래서 trigger가 거의 없더라도:

- `S-IEC p80 ~= 201 NFE`
- `IEC = 110 NFE`

즉 `NFE saving victory`는 구조적으로 나오기 어렵다.

따라서 정말 그 claim을 원하면 새 파라미터가 필요하다.

후보:

- `check_period = k`
- `check_topk_steps`
- `check_interval_seq_only`

이건 현재 exp4 후속 ablation이 아니라 거의 exp4-b 또는 exp4-redesign에 가깝다.

### 8.6 추천 실행 순서

가장 현실적인 순서는 이렇다.

1. `p80` 고정으로 `c_siec` sweep
2. 가장 나은 `c_siec`에서 `siec_max_rounds 1 vs 2` 비교
3. 그 조합으로 high-percentile tau sweep (`70~99`)
4. per-step syndrome plot을 실제 해석에 반영
5. 그래도 여전히 `IEC`보다 멀면, `NFE saving`은 포기하고 `Selectivity/Spectrum` claim만 노릴지 결정
6. `NFE saving`을 계속 원하면 sparse-check 파라미터를 별도 설계

### 8.7 한 줄 결론

현재 exp4의 다음 단계는 두 갈래다.

- 지금 구현 안에서 `Selectivity/Spectrum`을 살릴 파라미터 탐색
- 아니면 sparse-check 도입으로 아예 `NFE saving` 구조를 다시 설계

## 9. Per-step Syndrome Plot 정량 분석 및 해석

최신 실행(`20260425_104700`)에서 생성된 `syndrome_timestep_summary_*.json` 결과 데이터들을 파싱하여 분석한 결과, S-IEC가 직면한 한계의 근본적 원인이 정량적 데이터로 확인되었습니다.

### 9.1 정량적 데이터 분석 결과 (S-IEC p80 기준)

전체 100 스텝 중 생성 초반(Step 0~10)에 신드롬 점수가 압도적으로 집중되어 있으며, 뒤로 갈수록 급격히 0에 수렴하는 현상이 나타났습니다.

**[상위 10개 스텝 평균 신드롬]**
- Step 0: 0.1099
- Step 1: 0.0814
- Step 2: 0.0624
- Step 3: 0.0493
- Step 4: 0.0412
- Step 5: 0.0309
- Step 6: 0.0240
- Step 7: 0.0203
- Step 8: 0.0176
- Step 9: 0.0150

**[구간별 평균 신드롬 추이]**
- Steps 0-18: 0.027099
- Steps 19-37: 0.001557
- Steps 38-56: 0.000330
- Steps 57-75: 0.000095
- Steps 76-94: 0.000018

### 9.2 확산 모델(Diffusion)에서 초반에 에러가 쏠리는 이유

위 결과에서 '초반(Step 0 근처)'은 확산 모델의 역방향 생성(Reverse process) 과정에서 **순수 노이즈 상태($t \approx T$)**를 의미합니다. 신드롬(에러)이 초반에 쏠리는 이유는 다음과 같습니다.

1. **Global Structure 형성기**: 생성 초반에는 완전한 무작위 노이즈에서 이미지의 전체적인 윤곽과 뼈대를 잡아내야 합니다. 이 시기에는 모델의 예측값이 매 타임스텝마다 매우 크고 역동적으로 변합니다. 따라서 약간의 근사 연산(예: DeepCache에 의한 이전 특성맵 재사용이나 W8A8 양자화)만 개입되어도 예측 궤적이 크게 틀어지게 됩니다. 즉, 근사로 인한 '에러(Syndrome)'가 가장 크게 발생합니다.
2. **Fine Detail 정제기**: 반면 후반부 스텝(예: Step 50 이후, $t \to 0$)으로 갈수록 이미지는 거의 형태를 갖추고 있으며, 미세한 질감이나 디테일만 추가합니다. 이때는 캐싱이나 양자화를 하더라도 이전 스텝과의 차이가 미미하여 모델이 체감하는 에러가 거의 0에 가깝습니다.

### 9.3 실험 결과에 대한 최종 해석

1. **Selectivity 승리 실패의 원인**:
   신드롬 수치가 특정 샘플마다 다르게 튀는 것이 아니라, **모든 샘플이 공통적으로 극초반 스텝에서만 높은 에러를 발생**시킵니다. 따라서 `p80` 처럼 임계값을 주게 되면, S-IEC는 동적으로 위험 구간을 찾는 것이 아니라 "모든 샘플에 대해 초반 스텝만 집중적으로 보정"하는 **고정된 스케줄(Deterministic schedule)**처럼 작동하게 됩니다. 이로 인해 위험 구간을 지능적으로 탐지한다는 Selectivity의 이점이 퇴색되어, 단순 Random이나 Uniform에 비해 압도적인 품질(FID) 향상을 끌어내지 못했습니다.

2. **NFE 절약 실패의 근본 원인**:
   후반부 스텝(Step 40~99)은 신드롬이 사실상 0이라서 `p80` 세팅에서는 보정(Correction)이 거의 일어나지 않습니다. **하지만 현재 구현은 보정 여부를 결정하기 위한 '신드롬 검사(Lookahead check)'를 매 스텝마다 수행하고 있습니다.** 어차피 고치지 않을 후반부 60여 개 스텝에서도 쓸데없이 검사 비용(NFE)을 낭비하고 있기 때문에, 보정 빈도를 극단적으로 낮춰도 기본 NFE가 200 밑으로 내려갈 수 없는 병목이 발생했습니다.

**결론적으로 NFE saving 주장을 달성하기 위해서는, 에러가 없는 후반부 스텝의 신드롬 체크 자체를 생략하거나 특정 주기로만 수행하도록 하는 'Sparse-check' 구조 도입이 논리적으로 필수적임이 데이터로 증명되었습니다.**
