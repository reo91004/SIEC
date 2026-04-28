# 실험 5 Postmortem (2026-04-25)

기준 run:

- `/home/user/jowithu/Semantic/IEC/experiments/yongseong/results/real_05_robustness/20260425_153134`

이 문서는 최신 exp5 run이 재설계 지시사항을 얼마나 충족했는지, 실제 결과가 무엇을 의미하는지, 다음에 무엇을 바꿔야 하는지를 결과 중심으로 정리한다.

## 1. 실행 완료 여부

실험은 정상 완료됐다.

- `inventory.md`, `results.csv`, `results.json`, `robustness_2panel.png/.pdf` 생성 완료
- 여섯 setting 모두 `runnable`로 풀린 상태에서 full run 수행
- non-`fp16` setting에서는 `TAC`가 여전히 `blocked`
- `fp16`에서는 no-op regime 정렬을 위해 `TAC/IEC/S-IEC`가 `No correction` 결과를 공유함

즉, 실행 실패가 아니라 결과 해석 단계다.

## 2. 지시사항 이행 여부

## 2.0 quick checklist

아래는 최신 재설계 지시사항을 코드 기준으로 다시 체크한 표다.

| 항목 | 판정 | 메모 |
|---|---|---|
| `CIFAR-10`, `100 steps`, `DDIM` | 충족 | 최신 run이 해당 조건으로 완료됨 |
| 동일 모델/스케줄 고정, deployment error만 변화 | 충족 | wrapper taxonomy가 그 구조로 재정리됨 |
| `fp16 / w8a8 / dc10 / w4a8 / dc20 / cachequant` taxonomy | 충족 | 여섯 setting 모두 runnable 상태로 완료 |
| 각 setting의 `No correction / IEC / S-IEC` | 충족 | 수치 결과 존재 |
| `TAC` row | 부분 충족 | non-`fp16` setting은 `blocked`, `fp16`은 no-op shared row만 존재. TAC 알고리즘 자체는 여전히 미구현 |
| 고정 `tau percentile` 사용 | 충족 | 현재 `p80` 고정 |
| `FID`, `sFID` | 충족 | 전 setting 채워짐 (`TAC` 제외) |
| `trigger_rate` | 충족 | 전 setting 채워짐 (`TAC` 제외) |
| `total NFE`, `wall-clock time` | 충족 | 전 setting 채워짐 (`TAC` 제외) |
| error strength 정량화 | 충족 | `per-step syndrome score 평균` 사용 |
| setting별 pilot 재수행 | 충족 | 이번 run에서 재생성 |
| setting별 tau 재캘리브레이션 | 충족 | 이번 run에서 재생성 |
| 2-panel robustness plot | 부분 충족 | 파일은 생성되지만, `TAC`는 실질적 곡선이 아니라 `fp16` no-op shared point만 존재 |

요약:

- `TAC` 알고리즘은 여전히 없다.
- non-`fp16` 기준으로 보면 구현/실행 구조는 지시사항과 맞다.
- `fp16`의 `TAC` row는 no-op regime 정렬용 shared convenience row일 뿐, TAC 구현 증거가 아니다.

## 2.1 taxonomy와 baseline

최신 요구사항의 핵심은 아래였다.

- `fp16`
- `W8A8 quantization`
- `DeepCache only (interval=10)`
- `W4A8 quantization`
- `DeepCache aggressive (interval=20 또는 50)`
- `CacheQuant = W4A8 + DeepCache`

최신 run은 이를 아래처럼 구현했다.

- `fp16`
- `w8a8`
- `dc10`
- `w4a8`
- `dc20`
- `cachequant`

`dc20`를 썼기 때문에 "interval=20 또는 50" 요구도 충족한다.

각 setting에는 아래 row가 있다.

- `No correction`
- `TAC`
- `IEC`
- `S-IEC`

## 2.2 지표와 plot

요구 지표:

- `FID`, `sFID`
- `Per-step syndrome score 평균`
- `Trigger rate`
- `총 NFE`, `wall-clock time`

최신 run은 위 지표를 모두 결과 표에 기록한다.

plot도 지시사항과 맞다.

- 왼쪽: `error_strength vs FID`
- 오른쪽: `FID(IEC) - FID(S-IEC)`

단, `TAC`는 non-`fp16` setting에서 `blocked`다. `fp16`에는 shared no-op point가 존재하지만, 이것을 TAC gain curve로 해석하면 안 된다.

## 3. 최신 수치 요약

핵심 수치는 아래와 같다.

| Setting | error_strength | No correction FID | IEC FID | S-IEC FID | IEC-S-IEC | NoCorr-S-IEC | S-IEC trigger |
|---|---|---|---|---|---|---|---|
| `fp16` | `0.000000000` | 17.3192 | 17.3192 | 17.3192 | 0.0000 | 0.0000 | 0.0000 |
| `w8a8` | `0.005515371` | 52.2845 | 53.0272 | 52.3267 | 0.7006 | -0.0422 | 0.0404 |
| `dc10` | `0.000119379` | 17.5521 | 17.5107 | 17.6861 | -0.1754 | -0.1340 | 0.0076 |
| `w4a8` | `0.006425759` | 104.4366 | 105.0213 | 104.4266 | 0.5947 | 0.0101 | 0.0404 |
| `dc20` | `0.000283790` | 17.7685 | 18.0229 | 18.2371 | -0.2142 | -0.4686 | 0.0051 |
| `cachequant` | `0.006526918` | 99.8473 | 96.2755 | 100.1967 | -3.9211 | -0.3494 | 0.0202 |

해석:

- `w8a8`, `w4a8`에서는 S-IEC가 IEC보다 좋다
- `dc10`, `dc20`, `cachequant`에서는 S-IEC가 IEC보다 나쁘다
- `No correction`보다도 나쁜 setting이 여러 개 있다

## 4. 승리 조건 판정

지시사항의 승리 조건은 아래 셋 중 하나였다.

1. Trend 승리:
   error 강도가 커질수록 `FID_IEC - FID_SIEC`가 단조 증가
2. Regime 승리:
   약한 error에서는 비슷하거나 약간 뒤져도, 강한 error에서 S-IEC가 명확히 이김
3. Consistency 승리:
   모든 setting에서 S-IEC가 `No correction`보다 일관되게 개선

최신 run 판정:

### 4.1 Trend 승리 실패

`IEC-S-IEC`가 error strength에 따라 단조 증가하지 않는다.

예:

- `w8a8`: `+0.7006`
- `dc10`: `-0.1754`
- `w4a8`: `+0.5947`
- `dc20`: `-0.2142`
- `cachequant`: `-3.9211`

즉 오른쪽 panel이 논문 Figure 4 스타일의 상승 곡선을 보여주지 못한다.

### 4.2 Regime 승리 실패

강한 세팅에서 S-IEC가 유리해야 하는데, 가장 강한 후보인 `cachequant`에서 크게 진다.

- `cachequant`: `IEC-S-IEC = -3.9211`

즉 "강한 error에서 S-IEC가 명확히 이긴다"는 문장을 쓸 수 없다.

### 4.3 Consistency 승리 실패

S-IEC가 `No correction`보다 나은지 보면:

- `w8a8`: `-0.0422`
- `dc10`: `-0.1340`
- `dc20`: `-0.4686`
- `cachequant`: `-0.3494`

즉 여러 setting에서 S-IEC가 오히려 `No correction`보다 나쁘다.

## 5. 왜 이런 결과가 나왔는가

실제 원인은 크게 두 가지로 해석할 수 있다.

## 5.1 deployment error family별로 S-IEC 효과가 일관되지 않다

현재 결과를 보면 quantization 계열과 cache 계열이 다르게 반응한다.

- quantization 계열:
  - `w8a8`, `w4a8`에서는 S-IEC가 IEC보다 조금 낫다
- cache 계열:
  - `dc10`, `dc20`에서는 S-IEC가 IEC보다 나쁘다
- 혼합 계열:
  - `cachequant`에서는 크게 나쁘다

즉 "error가 강해질수록 S-IEC가 더 좋아진다"라기보다, "오류 유형에 따라 반응이 달라진다"에 가깝다.

## 5.2 고정 `p80`가 모든 setting에 잘 맞지 않는다

설계상 각 setting마다 pilot과 tau recalibration을 하긴 했지만, 최종 percentile은 전부 `p80` 하나로 고정했다.

현재 trigger rate도 setting마다 꽤 다르다.

- `w8a8`: `0.0404`
- `dc10`: `0.0076`
- `w4a8`: `0.0404`
- `dc20`: `0.0051`
- `cachequant`: `0.0202`

즉 setting별로 실질 동작점이 꽤 다르고, `p80`이 robustness 공통 operating point로 적절하지 않을 수 있다.

## 5.3 error_strength proxy와 직관적 severity 순서가 다르다

현재 X축은 `per-step syndrome score 평균`을 사용한다.

그 결과:

- `dc10`, `dc20`은 매우 작은 값
- `w8a8`, `w4a8`, `cachequant`는 큰 값

즉 사용자가 기대한 "deployment error 유형 순서"와 syndrome-based 수치 순서가 다르다.

이건 요구 위반은 아니다.

- 지시사항이 "유형 순서 또는 per-step syndrome score 평균"을 허용했기 때문이다.

하지만 해석상으로는 다음을 뜻한다.

- 현재 결과는 "cache error가 약하다"가 아니라
- "현재 syndrome proxy가 cache family를 약하게 본다"

일 가능성도 있다.

## 6. 다음에 무엇을 바꿔야 하는가

현재 상태에서 가장 실무적인 다음 액션은 세 가지다.

### 6.1 exp4에서 고른 `p80` 하나만 쓰지 말고 percentile sweep 재도입

현재 exp5는 `p80` 고정이라 setting별 최적 operating point를 놓칠 수 있다.

다음 후보:

- `p70`
- `p80`
- `p90`

최소 3점만 봐도 setting별 민감도 차이를 확인할 수 있다.

### 6.2 `error_strength` proxy를 한 가지 더 추가

현재는 syndrome mean만 쓴다.

추가 후보:

- `No correction 대비 FID 증가폭`
- `pilot syndrome p90`
- `pilot syndrome mean + trigger_rate` 복합 지표

목표:

- `dc10`, `dc20`가 정말 약한 regime인지
- 아니면 current proxy가 잘못 ordering하는지

를 분리해서 보는 것

### 6.3 cache 계열과 quant 계열을 분리 해석

현재는 여섯 setting을 한 plot에 넣지만, 실제로는 family별 반응이 다르다.

다음 분석 후보:

- quant family: `fp16 -> w8a8 -> w4a8`
- cache family: `fp16 -> dc10 -> dc20`
- mixed family: `cachequant`

이렇게 보면 "한 개의 monotone robustness law"가 아니라 "family-specific behavior"일 가능성을 더 정확히 볼 수 있다.

## 7. 최종 판정

strict하게 보면 최신 exp5도 "완벽 이행"은 아니다.

이유:

- `TAC` 알고리즘은 여전히 없다
- non-`fp16` setting에서는 `TAC`가 `blocked`
- `fp16`의 `TAC` row는 no-op shared convenience row일 뿐, TAC 구현 증거가 아니다

하지만 non-`fp16` 기준으로 보면 구현과 실행 구조는 재설계 지시사항과 맞는다.

즉 현재 상태를 가장 정확히 표현하면:

- 구현 구조: 충족
- setting별 recalibration: 충족
- 실행 정합성: 충족
- 결과 claim: 실패

한 줄 결론:

- exp5는 `TAC` 제외 기준으로는 설계와 실행이 맞게 끝났다.
- 하지만 최신 결과는 `Trend`, `Regime`, `Consistency` 어느 승리 조건도 만족하지 못한다.
