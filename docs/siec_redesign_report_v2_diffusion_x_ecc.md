# S-IEC 재설계 보고서 v2
## Diffusion × ECC 프레임으로 확장한 최종 구현 방향

작성일: 2026-04-29 KST

---

## 0. 핵심 결론

기존 `siec_redesign_report_v0.md`와 `siec_redesign_execution_plan_v1.md`는 **S-IEC를 IEC correction budget controller로 재정의**하는 데 초점이 있었다. 이 방향은 단기 구현과 실험에는 맞다. 하지만 당초 목표가 “IEC 개선”이 아니라 **Diffusion에 ECC 이론을 결합해 연산량과 품질을 동시에 개선하는 것**이라면, 최종 프레임은 더 넓어져야 한다.

가장 중요한 결론은 다음이다.

**기존 구현 목표가 틀린 것은 아니다. 다만 그것은 Diffusion × ECC의 전체 프레임 중 첫 번째 instance, 즉 `S-IEC for IEC scheduling`에 해당한다. 더 큰 주장을 하려면 S-IEC를 IEC 전용 알고리즘이 아니라, diffusion trajectory의 syndrome으로 timestep reliability를 추정하고, 그 reliability에 따라 correction / recomputation / cache refresh / precision escalation / timestep refinement를 선택하는 일반적인 ECC-inspired controller로 재정의해야 한다.**

즉 계층은 다음과 같다.

\[
\text{Diffusion} \times \text{ECC framework}
\quad \supset \quad
\text{S-IEC as IEC scheduler}
\]

따라서 구현은 더 큰 틀로 “벗어나는” 것이 아니라, 기존 M0–M5는 그대로 foundation으로 유지하고, M6 이후의 action space를 IEC 밖으로 확장하면 된다.

---

## 1. 기존 v1 실행 계획의 위치

`S-IEC Redesign Execution Plan v1`의 최종 목표는 다음이었다.

> Diffusion의 `x0_hat`/feature trajectory를 중복성을 가진 analog codeword로 보고, DeepCache/PTQ/CacheQuant를 noisy channel로 해석한다. S-IEC는 calibrated trajectory syndrome으로 timestep reliability를 추정하고 IEC correction budget을 더 필요한 곳에 배분한다.

이 목표는 정합하다. 특히 다음 원칙들은 v2에서도 그대로 유지된다.

1. `data manifold point = codeword` 주장은 버린다.
2. `trajectory = analog codeword`를 기본 프레임으로 둔다.
3. `linear learned parity is sufficient`라고 주장하지 않는다.
4. 모든 비교는 NFE-matched로 한다.
5. calibration seed와 evaluation seed를 분리한다.
6. 실험 결과가 없으면 claim을 낮춘다.

다만 v1은 마지막 문장이 좁다.

\[
\text{S-IEC} \Rightarrow \text{IEC correction budget allocation}
\]

v2에서는 이를 다음처럼 넓힌다.

\[
\text{S-IEC} \Rightarrow \text{diffusion compute/correction budget allocation}
\]

여기서 IEC는 여러 action 중 하나다.

---

## 2. 왜 IEC 전용으로 두면 좁은가

IEC는 efficient diffusion sampler에서 생기는 approximation error를 줄이는 좋은 correction operator다. 따라서 S-IEC를 처음 검증할 때 IEC scheduling에 붙이는 것은 합리적이다.

하지만 S-IEC의 본질은 IEC가 아니다. S-IEC의 본질은 다음 세 가지다.

\[
\text{trajectory redundancy}
\]

\[
\text{syndrome / parity residual}
\]

\[
\text{reliability-guided compute allocation}
\]

IEC는 이 중 세 번째 항목에서 선택할 수 있는 하나의 action이다.

만약 S-IEC를 IEC에만 묶으면, 논문 contribution은 다음으로 좁아진다.

> IEC를 더 잘 schedule하는 방법.

반대로 Diffusion × ECC로 확장하면 contribution은 다음이 된다.

> Diffusion sampling trajectory를 analog code로 보고, syndrome으로 timestep reliability를 추정해 연산과 보정 예산을 동적으로 배분하는 방법.

후자가 당초 목표에 더 가깝다.

---

## 3. Diffusion × ECC 프레임

### 3.1 Codeword

기존의 위험한 정의는 다음이었다.

\[
\text{codeword} = x \in \mathcal{M}_{data}
\]

이 정의는 버린다. data manifold의 한 점을 codeword로 두면, PTQ/DeepCache가 만든 on-manifold drift를 검출하기 어렵다. 또한 현재 syndrome인 \(\hat{x}_0(t)-\hat{x}_0(t-1)\)는 data manifold normal projection이 아니다.

v2에서 codeword는 diffusion trajectory다.

\[
\mathbf{c}
=
(y_T,y_{T-1},\ldots,y_0)
\]

여기서 \(y_t\)는 여러 선택지가 가능하다.

\[
y_t = \hat{x}_0(t)
\]

\[
y_t = \epsilon_\theta(x_t,t)
\]

\[
y_t = h_t
\quad
\text{U-Net intermediate feature}
\]

\[
y_t = (x_t,\hat{x}_0(t),h_t)
\]

즉 S-IEC의 codeword는 **clean/full-compute diffusion trajectory**다.

### 3.2 Channel

channel은 efficient deployment method 전체다.

\[
\text{channel}
\in
\{
\text{DeepCache},
\text{PTQ},
\text{CacheQuant},
\text{fast sampler},
\text{pruning},
\text{distillation},
\text{feature reuse}
\}
\]

각 channel은 clean trajectory를 다른 방식으로 오염시킨다.

- DeepCache는 feature reuse로 cached-step error를 만든다.
- PTQ는 low-precision quantization error를 만든다.
- fast sampler는 coarse integration error를 만든다.
- distillation은 teacher trajectory와 student trajectory 사이의 systematic drift를 만든다.

### 3.3 Syndrome

syndrome은 trajectory가 clean/full-compute code에서 얼마나 벗어났는지 측정하는 statistical parity residual이다.

가장 단순한 syndrome은 현재 구현과 같다.

\[
s_t^{raw}
=
\hat{x}_0(t)-\hat{x}_0(t-1)
\]

그러나 raw syndrome은 clean natural drift, model bias, discretization gap, deployment error를 섞어 본다. 따라서 v2의 기본 syndrome은 calibrated trajectory syndrome이다.

\[
s_t^{cal}
=
Q_t^{-1/2}
\left(
 y_{t-1}-y_t-\mu_t
\right)
\]

또는 learned parity를 쓰면 다음이다.

\[
s_t^{A}
=
Q_t^{-1/2}
\left(
 y_{t-1}-A_t y_t-b_t
\right)
\]

여기서 linear parity가 충분하다고 가정하지 않는다. learned parity는 ablation 대상이다.

\[
\text{raw}
\rightarrow
\text{mean drift}
\rightarrow
\text{diagonal affine}
\rightarrow
\text{low-rank linear}
\rightarrow
\text{piecewise / nonlinear}
\rightarrow
\text{feature-level parity}
\]

### 3.4 Reliability

syndrome score는 timestep reliability로 변환된다.

\[
z_t = \|s_t\|^2
\]

\[
\rho_t
=
F(z_t,\text{cache\_age}_t,\text{refresh\_step}_t,\text{timestep\_region}_t,\text{recent score pattern})
\]

여기서 \(\rho_t\)는 “이 timestep 또는 interval이 얼마나 신뢰하기 어려운가”를 나타낸다.

### 3.5 Decoder action

v1에서는 decoder action이 사실상 IEC였다.

v2에서는 action space를 확장한다.

\[
\mathcal{A}
=
\{
\text{skip},
\text{defer},
\text{IEC},
\text{light correction},
\text{early refresh},
\text{full recompute},
\text{precision escalation},
\text{timestep subdivision},
\text{feature correction}
\}
\]

이제 IEC는 필수 구성요소가 아니라 action 중 하나다.

---

## 4. 기존 구현 계획과 v2의 관계

중요한 점은 기존 M0–M5를 버리지 않는다는 것이다.

| 기존 milestone | v2에서의 역할 | 유지 여부 |
|---|---|---|
| M0 trace/NFE 정리 | 모든 action 비교의 기반 | 유지 |
| M1 clean trajectory stats | trajectory code calibration | 유지 |
| M2 speculative lookahead reuse | low-overhead syndrome 계산 | 유지 |
| M3 calibrated trigger baseline | first syndrome validation | 유지 |
| M4 teacher syndrome/gain | reliability가 실제 gain을 예측하는지 검증 | 유지 |
| M5 learned parity ablation | parity model 선택 | 유지 |
| M6 controller | IEC action을 넘는 general controller로 확장 | 확장 |

즉 v2 구현은 기존 계획을 폐기하는 것이 아니라 M6 이후를 더 넓히는 것이다.

v1의 핵심 구조는 다음이었다.

\[
z_t \rightarrow \text{IEC or skip}
\]

v2의 핵심 구조는 다음이다.

\[
z_t \rightarrow \rho_t \rightarrow a_t \in \mathcal{A}
\]

---

## 5. v2에서 추가해야 할 구현 목표

## M6'. General reliability controller

기존 M6는 `skip`, `defer`, `light_siec`, `iec`, `early_refresh` 정도의 action enum을 제안했다. v2에서는 이를 일반화한다.

### Action enum

```python
Action = Enum(
    "skip",
    "defer",
    "light_correction",
    "iec",
    "early_refresh",
    "full_recompute",
    "precision_escalate",
    "timestep_subdivide",
    "feature_correct",
)
```

### 입력 feature

```text
z_t
cache_age_t
refresh_step
recent_scores = [z_{t-2}, z_{t-1}, z_t]
timestep_region = early / mid / late
score_trend = spike / drift / stable
deployment_mode = deepcache / ptq / cachequant / sampler
```

### 기본 deterministic policy

처음부터 learned policy를 만들 필요는 없다.

```text
rho_t < tau1
    -> skip

tau1 <= rho_t < tau2
    -> defer or keep cached path

tau2 <= rho_t < tau3
    -> light correction or IEC at next anchor

rho_t >= tau3 and deployment == DeepCache
    -> early_refresh or full_recompute

rho_t >= tau3 and deployment == PTQ
    -> precision_escalate

rho_t >= tau3 and deployment == fast_sampler
    -> timestep_subdivide
```

이 구조가 들어가야 S-IEC가 IEC scheduler를 넘어선다.

---

## M7. DeepCache refresh controller

### 목적

IEC 없이도 S-IEC가 quality/NFE를 개선할 수 있음을 보인다.

### 핵심 아이디어

DeepCache에서는 full-compute step이 high-reliability anchor이고 cached step이 noisy symbol이다.

\[
\text{full-compute step} = \text{anchor symbol}
\]

\[
\text{cached step} = \text{noisy symbol}
\]

Syndrome이 커지면 IEC를 호출하는 대신 full-compute refresh를 앞당긴다.

\[
z_t > \tau
\Rightarrow
\text{early full-compute refresh}
\]

### 비교군

| Method | 의미 |
|---|---|
| DeepCache fixed interval | 기존 baseline |
| DeepCache shorter interval | 단순 비용 증가 baseline |
| DeepCache random early refresh | random control |
| S-IEC early refresh | syndrome-guided refresh |
| S-IEC early refresh + IEC | combined upper variant |

### 성공 기준

같은 NFE에서 S-IEC early refresh가 fixed/random refresh보다 FID가 좋아야 한다.

이 실험이 성공하면 S-IEC는 IEC 없이도 작동한다.

---

## M8. PTQ precision controller

### 목적

S-IEC가 quantization setting에서도 general compute allocator로 작동함을 보인다.

### 핵심 아이디어

Low precision을 기본으로 쓰되, syndrome이 큰 timestep만 high precision으로 올린다.

\[
z_t \le \tau
\Rightarrow
\text{W4A8 or W8A8}
\]

\[
z_t > \tau
\Rightarrow
\text{fp16 or higher precision}
\]

### 비교군

| Method | 의미 |
|---|---|
| all fp16 | reference |
| all low precision | deployed baseline |
| periodic high precision | schedule baseline |
| random high precision | random control |
| S-IEC precision escalation | proposed |

### 성공 기준

같은 high-precision step 수에서 S-IEC가 periodic/random보다 FID 또는 CLIP이 좋아야 한다.

이 실험은 S-IEC가 IEC에 특화되지 않았음을 가장 분명하게 보여준다.

---

## M9. Fast sampler timestep controller

### 목적

S-IEC를 caching/quantization뿐 아니라 sampler-level acceleration에도 확장한다.

### 핵심 아이디어

Fast sampler는 큰 step size 때문에 integration error를 만들 수 있다. Syndrome이 큰 구간은 timestep subdivision을 수행한다.

\[
z_t > \tau
\Rightarrow
[t,t-1] \text{ interval을 둘로 나눔}
\]

\[
z_t \le \tau
\Rightarrow
\text{기존 coarse step 유지}
\]

### 비교군

| Method | 의미 |
|---|---|
| fixed steps | baseline |
| more uniform steps | cost-matched schedule |
| random subdivision | random control |
| S-IEC subdivision | syndrome-guided adaptive sampler |

### 성공 기준

같은 NFE에서 uniform/random subdivision보다 S-IEC subdivision이 좋아야 한다.

이 실험은 장기 목표에 해당한다. 단기 마감에서는 optional이다.

---

## M10. Feature-level syndrome

### 목적

DeepCache는 feature reuse 기법이므로, image-space \(\hat{x}_0\) syndrome보다 feature-space syndrome이 더 직접적인지 검증한다.

### 정의

\[
s_t^{feat}
=
Q_t^{-1/2}
\left(
 h_t^{cached}-\hat{h}_t^{anchor}
\right)
\]

또는 인접 feature parity를 쓸 수 있다.

\[
s_t^{feat}
=
Q_t^{-1/2}
\left(
 h_{t-1}-A_t h_t-b_t
\right)
\]

### 비교군

| Syndrome space | 목적 |
|---|---|
| x0-space | 현재 S-IEC |
| epsilon-space | score/noise prediction consistency |
| feature-space | DeepCache error 직접 감지 |
| hybrid | reliability fusion |

### 성공 기준

feature-level syndrome이 DeepCache error, early refresh benefit, IEC benefit 중 하나를 x0-space보다 잘 예측해야 한다.

---

## 6. v2 실험 구조

v2에서는 실험을 두 계층으로 나눈다.

## Layer 1. IEC instance

이 계층은 기존 실행 계획과 거의 같다.

| Method | 목적 |
|---|---|
| DeepCache only | deployed baseline |
| DeepCache + IEC author | IEC baseline |
| random IEC same-NFE | trigger control |
| periodic IEC same-NFE | schedule control |
| S-IEC raw | old syndrome |
| S-IEC calibrated | improved syndrome |
| S-IEC learned parity | stronger syndrome |
| oracle IEC | upper bound |

여기서 보일 것은 다음이다.

\[
\text{S-IEC improves IEC scheduling.}
\]

## Layer 2. Non-IEC action instance

이 계층을 반드시 하나 이상 넣어야 Diffusion × ECC claim이 산다.

### 추천 1순위: DeepCache early refresh

| Method | 목적 |
|---|---|
| fixed DeepCache interval | baseline |
| shorter fixed interval | more compute baseline |
| random early refresh | random control |
| S-IEC early refresh | proposed non-IEC action |

보일 것은 다음이다.

\[
\text{S-IEC improves compute allocation even without IEC.}
\]

### 추천 2순위: PTQ precision escalation

| Method | 목적 |
|---|---|
| all low precision | baseline |
| periodic high precision | schedule baseline |
| random high precision | random control |
| S-IEC high precision | proposed non-IEC action |

보일 것은 다음이다.

\[
\text{S-IEC generalizes from correction scheduling to precision allocation.}
\]

---

## 7. 최종 논문 claim v2

## 7.1 너무 좁은 claim

다음 claim은 안전하지만 좁다.

> S-IEC improves IEC by allocating correction budget to unreliable timesteps.

이 claim은 “IEC scheduler”로 읽힌다.

## 7.2 권장 claim

다음 claim이 더 좋다.

> Diffusion sampling contains temporal redundancy in clean estimates, scores, and intermediate features. Efficient samplers and deployment accelerators corrupt this redundant trajectory through caching, quantization, or coarse integration. S-IEC introduces an ECC-inspired syndrome mechanism that estimates timestep reliability from calibrated trajectory parity residuals and uses this reliability to allocate computation and correction budget. IEC is one correction operator in this framework; the same syndrome can also drive cache refresh, precision escalation, and adaptive timestep refinement.

한국어로는 다음과 같다.

**Diffusion sampling은 \(\hat{x}_0\), score, U-Net feature의 시간축 중복성을 갖는다. DeepCache, PTQ, fast sampler 같은 효율화 기법은 이 중복 trajectory를 각기 다른 방식으로 오염시킨다. S-IEC는 ECC의 syndrome/reliability 개념을 이용해 timestep별 신뢰도를 추정하고, 그 신뢰도에 따라 IEC, full recompute, cache refresh, precision escalation, timestep refinement 같은 연산·보정 budget을 배분한다.**

## 7.3 논문 구조

논문은 다음 순서로 구성하는 것이 가장 좋다.

1. Diffusion trajectory has redundancy.
2. Efficient deployment corrupts this redundancy.
3. Data manifold codeword framing is insufficient.
4. We define an analog trajectory code.
5. We define calibrated trajectory syndrome.
6. Syndrome estimates timestep reliability.
7. Reliability controls decoder actions.
8. IEC scheduling is the first instance.
9. DeepCache early refresh or PTQ precision control shows generality.
10. Same-NFE Pareto improves over random/periodic/fixed baselines.

---

## 8. 안전한 표현과 피해야 할 표현

## 써도 되는 표현

- `Diffusion × ECC`
- `analog trajectory code`
- `statistical parity residual`
- `trajectory syndrome`
- `timestep reliability`
- `ECC-inspired reliability controller`
- `compute/correction budget allocation`
- `syndrome-guided cache refresh`
- `syndrome-guided precision escalation`
- `S-IEC as a general efficient diffusion controller`

## 조심해야 할 표현

- `true ECC`
- `Hc=0 exactly holds`
- `data manifold parity check`
- `off-manifold ECC decoder`
- `linear parity is sufficient`
- `IEC is the decoder` as a universal statement

정확한 표현은 다음이다.

\[
\text{IEC is one decoder action, not the whole S-IEC framework.}
\]

---

## 9. 구현 우선순위 v2

마감이 있는 상황에서는 다음 순서가 가장 안전하다.

### Phase 1. Foundation

1. M0 trace/NFE 정리
2. true no-correction path 검증
3. M1 clean trajectory stats calibration
4. M2 speculative lookahead reuse

이 단계는 v1과 동일하다.

### Phase 2. IEC instance

5. calibrated syndrome sanity check
6. `z_t -> Delta_t`, `z_t -> g_t` 분석
7. NFE-matched random/periodic/IEC comparison
8. learned parity ablation

여기서 S-IEC가 IEC scheduling을 개선하는지 본다.

### Phase 3. Generalization beyond IEC

9. DeepCache early refresh controller
10. random/fixed refresh same-NFE 비교
11. 가능하면 PTQ precision escalation 추가

이 단계가 Diffusion × ECC claim을 살린다.

### Phase 4. Stronger ECC-like controller

12. gate-only vs multi-action controller 비교
13. syndrome pattern 기반 spike/drift 분류
14. feature-level syndrome ablation

이 단계는 anomaly detection 비판을 줄인다.

---

## 10. 성공 기준 v2

Positive claim을 하려면 최소 다음이 필요하다.

### 필수 조건

1. calibrated or learned syndrome이 raw보다 clean drift를 잘 제거한다.
2. same-NFE에서 S-IEC가 random/periodic IEC scheduling보다 낫다.
3. `reuse_lookahead`가 NFE를 줄이면서 FID를 크게 해치지 않는다.

### Diffusion × ECC claim을 위한 추가 조건

4. IEC가 아닌 action 하나 이상에서 S-IEC가 random/fixed baseline보다 낫다.
   - 예: DeepCache early refresh
   - 예: PTQ precision escalation
5. controller가 gate-only보다 낫거나, 최소한 다른 error pattern에 다른 action을 배분해 해석 가능한 결과를 만든다.

즉 v2에서 positive claim의 최소 형태는 다음이다.

\[
\text{S-IEC improves IEC scheduling}
\]

그리고 더 강한 형태는 다음이다.

\[
\text{S-IEC improves general diffusion compute allocation}
\]

후자를 주장하려면 반드시 non-IEC action 실험이 하나 이상 필요하다.

---

## 11. 최종 판정

기존 개선 구현 목표는 잘못된 것이 아니다. 다만 그것은 큰 프레임의 일부다.

**S-IEC를 Diffusion × ECC로 보려면 구현이 완전히 다른 방향으로 벗어나는 것이 아니다. 기존 M0–M5는 그대로 필요하고, M6 controller를 IEC 전용 gate에서 general reliability controller로 확장하면 된다. 차이는 action space에 있다. 기존에는 syndrome이 IEC를 호출할지 말지를 결정했다. v2에서는 syndrome이 IEC뿐 아니라 early refresh, full recompute, precision escalation, timestep subdivision까지 결정한다.**

따라서 최종 방향은 다음이다.

\[
\text{S-IEC v1}
=
\text{syndrome-guided IEC scheduling}
\]

\[
\text{S-IEC v2}
=
\text{syndrome-guided efficient diffusion control}
\]

그리고 논문에서 가장 강한 문장은 다음이다.

**S-IEC is not merely an IEC scheduler. It is an ECC-inspired reliability controller for efficient diffusion sampling. It treats clean/full-compute diffusion trajectories as analog codewords, detects trajectory corruption through calibrated syndromes, and allocates compute or correction actions to unreliable timesteps. IEC is one action in this controller, while cache refresh, recomputation, precision escalation, and timestep refinement provide broader Diffusion × ECC instantiations.**
