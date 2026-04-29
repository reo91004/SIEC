# S-IEC 재설계 보고서 v2.1
## Diffusion × ECC 프레임으로 확장한 최종 구현 방향 — 정합성 검증 반영본

작성일: 2026-04-29 KST  
기반 문서: `siec_redesign_report_v2_diffusion_x_ecc.md`, `siec_redesign_execution_plan_v1.md`  
반영 문서: `v2 문서 정합성 평가`, `v2 검증 및 개선 제안`

---

## 0. 핵심 결론

기존 `siec_redesign_report.md`와 `siec_redesign_execution_plan_v1.md`는 **S-IEC를 IEC correction budget controller로 재정의**하는 데 초점이 있었다. 이 방향은 단기 구현과 실험에는 맞다. 하지만 당초 목표가 “IEC 개선”이 아니라 **Diffusion에 ECC 이론을 결합해 연산량과 품질을 동시에 개선하는 것**이라면, 최종 프레임은 더 넓어져야 한다.

가장 중요한 결론은 다음이다.

**기존 구현 목표가 틀린 것은 아니다. 다만 그것은 Diffusion × ECC의 전체 프레임 중 첫 번째 instance, 즉 `S-IEC for IEC scheduling`에 해당한다. 더 큰 주장을 하려면 S-IEC를 IEC 전용 알고리즘이 아니라, diffusion trajectory의 syndrome으로 timestep reliability를 추정하고, 그 reliability에 따라 correction / recomputation / cache refresh / precision escalation / timestep refinement를 선택하는 일반적인 ECC-inspired controller로 재정의해야 한다.**

즉 계층은 다음과 같다.

\[
\text{Diffusion} \times \text{ECC framework}
\quad \supset \quad
\text{S-IEC as IEC scheduler}
\]

따라서 구현은 더 큰 틀로 “벗어나는” 것이 아니라, 기존 M0–M5는 그대로 foundation으로 유지하고, M6 이후의 action space를 IEC 밖으로 확장하면 된다.

다만 v2 정합성 검증 결과, 더 큰 프레임으로 확장하면서 새로 생긴 위험이 있다. 본 v2.1은 기존 v2의 맥락과 내용을 유지하면서 다음 네 가지를 명시적으로 보강한다.

1. **Speculative rollback 정의 명확화**  
   DDIM sampling은 autoregressive이므로 이미 committed된 과거 timestep으로 실제 rollback하지 않는다. syndrome trigger가 발생하면 speculative lookahead output을 폐기하고, 현재 step의 tentative state에 correction / recompute / early refresh action을 적용한다. 즉 정확한 표현은 `rollback`이 아니라 **discard speculative lookahead after trigger**다.

2. **Per-channel calibration 전략 추가**  
   DeepCache, PTQ, CacheQuant, fast sampler는 서로 다른 오류 구조를 만든다. 따라서 calibration은 deployment channel별로 분리해야 한다. channel 간 transfer는 기본 가정이 아니라 ablation 대상이다.

3. **Phase 2 → Phase 3 진입 게이트 추가**  
   syndrome이 IEC benefit 또는 action-specific benefit을 예측하지 못하면, 같은 syndrome으로 early refresh, precision escalation, timestep subdivision을 해도 실패할 가능성이 높다. 따라서 broader Diffusion × ECC claim으로 넘어가기 전에 signal quality gate를 통과해야 한다.

4. **Precision escalation의 배포 전제 명시**  
   PTQ-only 배포 환경에서 원본 fp16 weight가 없으면 “W4A8에서 fp16으로 동적 상승”은 불가능하거나 배포 동기와 충돌한다. 따라서 precision escalation은 **mixed-precision deployment**, 즉 higher precision weight/path가 함께 존재하는 환경에서만 optional extension으로 둔다.

---

## 1. 기존 v1 실행 계획의 위치

`S-IEC Redesign Execution Plan v1`의 최종 목표는 다음이었다.

> Diffusion의 `x0_hat`/feature trajectory를 중복성을 가진 analog codeword로 보고, DeepCache/PTQ/CacheQuant를 noisy channel로 해석한다. S-IEC는 calibrated trajectory syndrome으로 timestep reliability를 추정하고 IEC correction budget을 더 필요한 곳에 배분한다.

이 목표는 정합하다. 특히 다음 원칙들은 v2.1에서도 그대로 유지된다.

1. `data manifold point = codeword` 주장은 버린다.
2. `trajectory = analog codeword`를 기본 프레임으로 둔다.
3. `linear learned parity is sufficient`라고 주장하지 않는다.
4. 모든 비교는 NFE-matched로 한다.
5. calibration seed와 evaluation seed를 분리한다.
6. 실험 결과가 없으면 claim을 낮춘다.
7. `IEC/` 원본은 수정하지 않고, 구현은 `S-IEC/` 안에서 수행한다.
8. NFE는 공식 추정이 아니라 trace의 `nfe_per_step` 합으로 계산한다.
9. `rollback`이라는 표현은 조심한다. 실제 구현은 과거 state로 되돌아가는 것이 아니라, speculative prediction을 폐기하고 현재 step에서 action을 재결정하는 것이다.

다만 v1은 마지막 문장이 좁다.

\[
\text{S-IEC} \Rightarrow \text{IEC correction budget allocation}
\]

v2.1에서는 이를 다음처럼 넓힌다.

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

v2.1에서 codeword는 diffusion trajectory다.

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
- CacheQuant는 cache stale error와 quantization error를 동시에 만든다.
- fast sampler는 coarse integration error를 만든다.
- distillation은 teacher trajectory와 student trajectory 사이의 systematic drift를 만든다.

여기서 반드시 보강해야 할 점이 있다. **general channel claim은 per-channel calibration 없이는 성립하지 않는다.** DeepCache에서 calibration한 residual statistics를 PTQ에 그대로 쓰면 false positive 또는 false negative가 커질 수 있다. DeepCache의 error는 feature stale/reuse 성격이고, PTQ error는 weight/activation quantization bias 성격이며, fast sampler error는 numerical integration 성격이기 때문이다.

따라서 v2.1의 calibration 원칙은 다음이다.

\[
\text{stats} = \text{stats}[\text{model},\text{sampler},\text{channel},\text{score space},\text{schedule}]
\]

각 deployment channel마다 별도의 calibration을 수행한다.

\[
\text{DeepCache stats} \neq \text{PTQ stats} \neq \text{fast-sampler stats}
\]

channel 간 calibration transfer는 다음과 같은 ablation으로만 다룬다.

| Calibration source | Evaluation channel | 목적 |
|---|---|---|
| DeepCache | DeepCache | in-channel baseline |
| PTQ | PTQ | in-channel baseline |
| DeepCache | PTQ | transfer 가능성 확인 |
| PTQ | DeepCache | transfer 가능성 확인 |
| clean/full-compute only | all channels | universal baseline |

논문에서 “general controller”라고 말하려면 최소한 channel-specific calibration이 가능하고, non-IEC action 하나 이상에서 같은 channel 내 random/fixed baseline을 이긴다는 것을 보여야 한다. channel transfer는 더 강한 주장이지 기본 전제가 아니다.

### 3.3 Syndrome

syndrome은 trajectory가 clean/full-compute code에서 얼마나 벗어났는지 측정하는 statistical parity residual이다.

가장 단순한 syndrome은 현재 구현과 같다.

\[
s_t^{raw}
=
\hat{x}_0(t)-\hat{x}_0(t-1)
\]

그러나 raw syndrome은 clean natural drift, model bias, discretization gap, deployment error를 섞어 본다. 따라서 v2.1의 기본 syndrome은 calibrated trajectory syndrome이다.

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

추가로, calibration이 이미지 내용에 무관하게 일반화된다는 것도 가정하면 안 된다. CIFAR-class, prompt family, semantic complexity, timestep region에 따라 clean drift distribution이 다를 수 있다. 따라서 calibration은 seed split뿐 아니라 가능하면 content split 또는 class/prompt split에서도 점검한다.

최소 검증은 다음이다.

| 검증 | 의미 |
|---|---|
| calibration seed → evaluation seed | seed leakage 방지 |
| class/prompt holdout | image-content generalization 확인 |
| early/mid/late timestep split | dynamics region별 residual 안정성 확인 |
| channel-specific stats | deployment error 구조 차이 반영 |

이 검증을 통과하지 못하면 calibrated syndrome은 “보편적 parity residual”이 아니라 특정 split에 과적합된 anomaly score일 수 있다.

### 3.4 Reliability

syndrome score는 timestep reliability로 변환된다.

\[
z_t = \|s_t\|^2
\]

\[
\rho_t
=
F(z_t,\text{cache\_age}_t,\text{refresh\_step}_t,\text{timestep\_region}_t,\text{recent score pattern},\text{deployment channel})
\]

여기서 \(\rho_t\)는 “이 timestep 또는 interval이 얼마나 신뢰하기 어려운가”를 나타낸다.

정합성 검증에서 지적된 대로, action space를 넓히면 policy 설계 복잡도가 빠르게 증가한다. 따라서 처음부터 하나의 거대한 multi-action learned policy를 만들면 안 된다. v2.1에서는 reliability를 먼저 검증하고, action은 channel별로 하나씩 binary 또는 small action set으로 검증한다.

예를 들어 다음 순서가 안전하다.

\[
\text{DeepCache}: \rho_t \rightarrow \{\text{keep cache},\text{early refresh}\}
\]

\[
\text{PTQ}: \rho_t \rightarrow \{\text{low precision},\text{higher precision path}\}
\]

\[
\text{IEC}: \rho_t \rightarrow \{\text{skip},\text{IEC}\}
\]

그 뒤에만 multi-action controller로 확장한다.

### 3.5 Decoder action

v1에서는 decoder action이 사실상 IEC였다.

v2.1에서는 action space를 확장한다.

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

다만 각 action에는 적용 전제가 있다.

| Action | 적용 channel | 전제 | 단기 우선순위 |
|---|---|---|---|
| IEC | DeepCache, PTQ, CacheQuant | IEC 구현 및 NFE-matched baseline | 높음 |
| Early refresh | DeepCache | cache schedule를 동적으로 바꿀 수 있음 | 높음 |
| Full recompute | DeepCache / cached sampler | full path 호출 가능 | 중간 |
| Precision escalation | PTQ / mixed precision | higher precision path 또는 fp16 weight가 존재 | 중간-낮음 |
| Timestep subdivision | fast sampler | 동적 schedule 삽입과 cache state 처리 가능 | 낮음 |
| Feature correction | DeepCache | feature residual 정의와 correction operator 필요 | 장기 |

---

## 4. 기존 구현 계획과 v2.1의 관계

중요한 점은 기존 M0–M5를 버리지 않는다는 것이다.

| 기존 milestone | v2.1에서의 역할 | 유지 여부 |
|---|---|---|
| M0 trace/NFE 정리 | 모든 action 비교의 기반 | 유지 |
| M1 clean trajectory stats | trajectory code calibration | 유지, per-channel / content-generalization 검증 추가 |
| M2 speculative lookahead reuse | low-overhead syndrome 계산 | 유지, rollback 정의 명확화 |
| M3 calibrated trigger baseline | first syndrome validation | 유지 |
| M4 teacher syndrome/gain | reliability가 실제 gain을 예측하는지 검증 | 유지, Phase 3 gate로 사용 |
| M5 learned parity ablation | parity model 선택 | 유지 |
| M6 controller | IEC action을 넘는 general controller로 확장 | 확장 |

즉 v2.1 구현은 기존 계획을 폐기하는 것이 아니라 M6 이후를 더 넓히는 것이다.

v1의 핵심 구조는 다음이었다.

\[
z_t \rightarrow \text{IEC or skip}
\]

v2.1의 핵심 구조는 다음이다.

\[
z_t \rightarrow \rho_t \rightarrow a_t \in \mathcal{A}
\]

---

## 5. v2.1에서 추가해야 할 구현 목표

## M6'. General reliability controller

기존 M6는 `skip`, `defer`, `light_siec`, `iec`, `early_refresh` 정도의 action enum을 제안했다. v2.1에서는 이를 일반화하되, 처음부터 모든 action을 하나의 policy로 섞지 않는다.

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
channel_calibration_id
```

### 기본 deterministic policy

처음부터 learned policy를 만들 필요는 없다. 다만 action space가 커질수록 threshold tuning이 복잡해지므로, v2.1에서는 action-specific binary policy부터 검증한다.

```text
DeepCache refresh controller:
    rho_t < tau        -> keep cached path
    rho_t >= tau       -> early refresh or full recompute

IEC scheduler:
    rho_t < tau        -> skip IEC
    rho_t >= tau       -> IEC at current/next valid correction point

PTQ precision controller:
    rho_t < tau        -> low precision path
    rho_t >= tau       -> higher precision path, only if mixed-precision deployment allows it

Fast sampler controller:
    rho_t < tau        -> keep coarse step
    rho_t >= tau       -> timestep subdivision, optional long-term action
```

그 뒤에만 multi-action policy를 검증한다.

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
    -> precision_escalate only in mixed-precision setting

rho_t >= tau3 and deployment == fast_sampler
    -> timestep_subdivide only if dynamic schedule is implemented
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

### speculative execution 의미

여기서 “rollback”이라고 쓰면 안 된다. 실제 동작은 다음이다.

1. 현재 step \(x_t\)에서 tentative \(x_{t-1}^{tent}\)와 speculative lookahead prediction을 만든다.
2. syndrome이 낮으면 speculative prediction과 cache state를 commit한다.
3. syndrome이 높으면 speculative lookahead output을 폐기한다.
4. 과거 \(x_t\)로 역추적하는 것이 아니라, 현재 tentative transition에 대해 early refresh / full recompute / IEC action을 적용해 \(x_{t-1}\)를 다시 산출한다.
5. trace에는 `memo_discarded`, `action`, `extra_nfe`, `committed_state`를 기록한다.

즉 표현은 다음이 정확하다.

```text
discard speculative lookahead after trigger
```

다음 표현은 피한다.

```text
rollback to previous timestep
```

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

S-IEC가 quantization setting에서도 general compute allocator로 작동할 수 있는지 탐색한다.

### 핵심 아이디어

Low precision을 기본으로 쓰되, syndrome이 큰 timestep만 higher precision path로 올린다.

\[
z_t \le \tau
\Rightarrow
\text{W4A8 or W8A8}
\]

\[
z_t > \tau
\Rightarrow
\text{higher precision path}
\]

### 중요한 전제 제한

이 action은 standard PTQ-only deployment에 바로 적용되는 것이 아니다. 만약 배포 환경에 low-bit weight만 있고 원본 fp16 weight 또는 higher precision path가 없다면, timestep별 fp16 escalation은 불가능하거나 PTQ의 메모리 절약 목적과 충돌한다.

따라서 v2.1에서 precision escalation은 다음 조건 중 하나를 만족할 때만 실험한다.

1. fp16 또는 bf16 weight가 함께 저장된 mixed-precision deployment.
2. 일부 layer만 higher precision으로 남겨둔 hybrid quantization deployment.
3. dequantized higher-bit path가 runtime에서 허용되는 서버/고메모리 환경.
4. activation precision만 올리는 방식처럼 weight memory를 크게 늘리지 않는 제한적 precision escalation.

논문 표현도 다음처럼 제한해야 한다.

> Precision escalation is an optional mixed-precision instantiation, not an assumption of standard PTQ-only deployment.

### 비교군

| Method | 의미 |
|---|---|
| all fp16 | reference, 단 mixed-precision setting에서만 가능 |
| all low precision | deployed baseline |
| periodic high precision | schedule baseline |
| random high precision | random control |
| S-IEC precision escalation | proposed optional extension |

### 성공 기준

같은 high-precision step 수 또는 같은 memory/latency budget에서 S-IEC가 periodic/random보다 FID 또는 CLIP이 좋아야 한다.

이 실험은 S-IEC가 IEC에 특화되지 않았음을 보여줄 수 있지만, 배포 전제가 강하므로 main contribution보다는 optional extension으로 두는 것이 안전하다.

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

### 기술적 주의

이 action은 장기 목표다. DDIM schedule에서 timestep은 사전에 정해진 sequence를 따르므로, 중간 step을 동적으로 삽입하려면 noise schedule의 중간값을 정의해야 한다.

\[
\bar{\alpha}_{t+1/2}
\]

같은 값을 어떻게 보간할지, solver order가 있는 sampler에서는 중간 state와 history를 어떻게 다룰지, DeepCache interval 안에서 subdivision이 발생할 때 anchor freshness와 feature cache를 어떻게 처리할지 명시해야 한다.

따라서 M9는 단기 main experiment가 아니라 optional long-term extension이다. 논문 contribution에 넣으려면 다음 세부사항이 필요하다.

1. intermediate timestep schedule 정의.
2. subdivision이 NFE에 반영되는 방식.
3. DeepCache / cache state와의 상호작용.
4. uniform more-steps baseline과 NFE-matched 비교.

### 비교군

| Method | 의미 |
|---|---|
| fixed steps | baseline |
| more uniform steps | cost-matched schedule |
| random subdivision | random control |
| S-IEC subdivision | syndrome-guided adaptive sampler |

### 성공 기준

같은 NFE에서 uniform/random subdivision보다 S-IEC subdivision이 좋아야 한다.

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

## 6. v2.1 실험 구조

v2.1에서는 실험을 세 계층으로 나눈다.

## Layer 0. Signal validity gate

Layer 1과 Layer 2로 가기 전에 syndrome이 실제 benefit을 예측하는지 먼저 확인한다.

| Signal | Target | 목적 |
|---|---|---|
| raw \(r_t\) | \(g_t^{IEC}\) | 기존 syndrome 검증 |
| calibrated \(z_t^{cal}\) | \(g_t^{IEC}\) | clean drift 제거 효과 |
| learned parity \(z_t^A\) | \(g_t^{IEC}\) | parity 개선 효과 |
| calibrated / learned score | early refresh gain | non-IEC action 가능성 |
| calibrated / learned score | precision escalation gain | PTQ optional action 가능성 |

여기서 \(g_t^{IEC}\)는 IEC를 적용했을 때 reference error가 줄어드는 양이고, early refresh gain은 같은 timestep에서 full-compute refresh를 앞당겼을 때 error 또는 final metric이 개선되는 정도다.

Phase gate는 다음이다.

```text
If corr(z_t, g_t^IEC) is low:
    Do not claim S-IEC improves IEC scheduling.

If corr(z_t, early_refresh_gain) is low:
    Do not run or claim DeepCache early refresh as a positive result.

If only clean/deploy AUROC is high but gain correlation is low:
    Treat S-IEC as anomaly detector, not decoder/controller.
```

즉 Layer 2로의 진입은 Layer 0/1 결과에 의존한다. 다만 action-specific gain을 별도로 측정해 해당 action에서 상관이 높다면, IEC benefit correlation이 낮아도 그 action에 대해서는 별도 검증을 진행할 수 있다.

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

이 실험은 optional mixed-precision extension이다.

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

단, 이 claim은 higher precision path가 존재하는 deployment setting에서만 유효하다.

---

## 7. 최종 논문 claim v2.1

## 7.1 너무 좁은 claim

다음 claim은 안전하지만 좁다.

> S-IEC improves IEC by allocating correction budget to unreliable timesteps.

이 claim은 “IEC scheduler”로 읽힌다.

## 7.2 권장 claim

다음 claim이 더 좋다.

> Diffusion sampling contains temporal redundancy in clean estimates, scores, and intermediate features. Efficient samplers and deployment accelerators corrupt this redundant trajectory through caching, quantization, or coarse integration. S-IEC introduces an ECC-inspired syndrome mechanism that estimates timestep reliability from calibrated trajectory parity residuals and uses this reliability to allocate computation and correction budget. IEC is one correction operator in this framework; the same syndrome can also drive cache refresh, recomputation, mixed-precision escalation, and adaptive timestep refinement when the corresponding action is available.

한국어로는 다음과 같다.

**Diffusion sampling은 \(\hat{x}_0\), score, U-Net feature의 시간축 중복성을 갖는다. DeepCache, PTQ, fast sampler 같은 효율화 기법은 이 중복 trajectory를 각기 다른 방식으로 오염시킨다. S-IEC는 ECC의 syndrome/reliability 개념을 이용해 timestep별 신뢰도를 추정하고, 그 신뢰도에 따라 IEC, full recompute, cache refresh, mixed-precision escalation, timestep refinement 같은 연산·보정 budget을 배분한다.**

여기서 mixed-precision escalation은 standard PTQ-only deployment의 기본 action이 아니라, higher precision path가 존재하는 환경에서의 optional instantiation이다.

## 7.3 논문 구조

논문은 다음 순서로 구성하는 것이 가장 좋다.

1. Diffusion trajectory has redundancy.
2. Efficient deployment corrupts this redundancy.
3. Data manifold codeword framing is insufficient.
4. We define an analog trajectory code.
5. We define calibrated trajectory syndrome.
6. Syndrome estimates timestep reliability.
7. Reliability controls decoder / compute actions.
8. IEC scheduling is the first instance.
9. DeepCache early refresh shows non-IEC generality.
10. PTQ precision control or fast sampler timestep subdivision is optional, subject to deployment feasibility.
11. Same-NFE Pareto improves over random/periodic/fixed baselines.

---

## 8. 안전한 표현과 피해야 할 표현

## 써도 되는 표현

- `Diffusion × ECC`
- `analog trajectory code`
- `statistical parity residual`
- `trajectory syndrome`
- `diagonal-whitened trajectory syndrome`
- `channel-specific calibration`
- `timestep reliability`
- `ECC-inspired reliability controller`
- `compute/correction budget allocation`
- `syndrome-guided cache refresh`
- `syndrome-guided mixed-precision escalation`
- `discard speculative lookahead after trigger`
- `S-IEC as a general efficient diffusion controller`

## 조심해야 할 표현

- `true ECC`
- `Hc=0 exactly holds`
- `data manifold parity check`
- `off-manifold ECC decoder`
- `linear parity is sufficient`
- `IEC is the decoder` as a universal statement
- `rollback to previous timestep`
- `precision escalation in PTQ-only deployment` without stating assumptions
- `universal channel calibration` without per-channel evidence

정확한 표현은 다음이다.

\[
\text{IEC is one decoder action, not the whole S-IEC framework.}
\]

\[
\text{Precision escalation is optional and deployment-dependent.}
\]

\[
\text{Calibration is channel-specific unless transfer is empirically validated.}
\]

---

## 9. 구현 우선순위 v2.1

마감이 있는 상황에서는 다음 순서가 가장 안전하다.

### Phase 1. Foundation

1. M0 trace/NFE 정리
2. true no-correction path 검증
3. M1 clean trajectory stats calibration
4. channel-specific stats schema 추가
5. M2 speculative lookahead reuse
6. speculative discard semantics trace 추가

이 단계는 v1과 동일하되, per-channel calibration과 rollback 표현 정리가 추가된다.

### Phase 2. IEC instance

7. calibrated syndrome sanity check
8. `z_t -> Delta_t`, `z_t -> g_t^IEC` 분석
9. NFE-matched random/periodic/IEC comparison
10. learned parity ablation

여기서 S-IEC가 IEC scheduling을 개선하는지 본다.

### Phase 2.5. 진입 게이트

다음 조건 중 하나 이상을 통과해야 Phase 3 positive experiment로 넘어간다.

1. calibrated / learned syndrome이 raw보다 `corr(z_t, g_t^IEC)`를 개선한다.
2. S-IEC가 same-NFE random/periodic IEC scheduling보다 낫다.
3. IEC benefit correlation은 낮지만, action-specific pilot에서 `corr(z_t, early_refresh_gain)`이 높다.

이 조건을 통과하지 못하면 broader Diffusion × ECC claim은 positive result로 쓰지 않는다. 대신 negative/diagnostic contribution으로 낮춘다.

### Phase 3. Generalization beyond IEC

11. DeepCache early refresh controller
12. random/fixed refresh same-NFE 비교
13. 가능하면 PTQ mixed-precision escalation 추가

이 단계가 Diffusion × ECC claim을 살린다. 단, precision escalation은 mixed-precision deployment setting으로 제한한다.

### Phase 4. Stronger ECC-like controller

14. gate-only vs multi-action controller 비교
15. syndrome pattern 기반 spike/drift 분류
16. feature-level syndrome ablation
17. timestep subdivision은 optional long-term extension으로 유지

이 단계는 anomaly detection 비판을 줄인다.

---

## 10. 성공 기준 v2.1

Positive claim을 하려면 최소 다음이 필요하다.

### 필수 조건

1. calibrated or learned syndrome이 raw보다 clean drift를 잘 제거한다.
2. calibrated or learned syndrome이 action gain과 상관된다.
3. same-NFE에서 S-IEC가 random/periodic IEC scheduling보다 낫다.
4. `reuse_lookahead`가 NFE를 줄이면서 FID를 크게 해치지 않는다.

### Diffusion × ECC claim을 위한 추가 조건

5. IEC가 아닌 action 하나 이상에서 S-IEC가 random/fixed baseline보다 낫다.
   - 예: DeepCache early refresh
   - 예: mixed-precision escalation, 단 higher precision path가 존재하는 setting
6. controller가 gate-only보다 낫거나, 최소한 다른 error pattern에 다른 action을 배분해 해석 가능한 결과를 만든다.
7. calibration이 channel-specific split에서 정상 작동하고, channel transfer를 주장하려면 transfer ablation에서 성능이 유지된다.

즉 v2.1에서 positive claim의 최소 형태는 다음이다.

\[
\text{S-IEC improves IEC scheduling}
\]

그리고 더 강한 형태는 다음이다.

\[
\text{S-IEC improves general diffusion compute allocation}
\]

후자를 주장하려면 반드시 non-IEC action 실험이 하나 이상 필요하다.

### 실패 시 claim downgrade

조건이 충족되지 않으면 논문 claim은 다음처럼 낮춘다.

> We identify why raw trajectory syndromes fail as ECC-style decoders in CIFAR DDPM, and show that clean-drift calibration, channel-specific statistics, and speculative reuse are necessary conditions for ECC-inspired reliability control in efficient diffusion.

이 경우 논문은 positive method paper가 아니라 diagnostic / design principle paper에 가까워진다.

---

## 11. 최종 판정

기존 개선 구현 목표는 잘못된 것이 아니다. 다만 그것은 큰 프레임의 일부다.

**S-IEC를 Diffusion × ECC로 보려면 구현이 완전히 다른 방향으로 벗어나는 것이 아니다. 기존 M0–M5는 그대로 필요하고, M6 controller를 IEC 전용 gate에서 general reliability controller로 확장하면 된다. 차이는 action space에 있다. 기존에는 syndrome이 IEC를 호출할지 말지를 결정했다. v2.1에서는 syndrome이 IEC뿐 아니라 early refresh, full recompute, mixed-precision escalation, timestep subdivision까지 결정한다.**

다만 v2.1은 v2보다 더 엄격하다. broader claim을 하려면 다음 조건을 반드시 지켜야 한다.

1. rollback이 아니라 speculative lookahead discard로 구현한다.
2. calibration은 channel-specific으로 수행한다.
3. Phase 2 signal-quality gate를 통과해야 Phase 3 positive claim을 한다.
4. precision escalation은 mixed-precision deployment에서만 optional action으로 주장한다.
5. timestep subdivision은 schedule interpolation과 cache interaction을 구현하기 전까지 long-term optional action으로 둔다.

따라서 최종 방향은 다음이다.

\[
\text{S-IEC v1}
=
\text{syndrome-guided IEC scheduling}
\]

\[
\text{S-IEC v2.1}
=
\text{syndrome-guided efficient diffusion control}
\]

그리고 논문에서 가장 강한 문장은 다음이다.

**S-IEC is not merely an IEC scheduler. It is an ECC-inspired reliability controller for efficient diffusion sampling. It treats clean/full-compute diffusion trajectories as analog codewords, detects trajectory corruption through calibrated and channel-specific syndromes, and allocates compute or correction actions to unreliable timesteps. IEC is one action in this controller, while cache refresh, recomputation, mixed-precision escalation, and timestep refinement provide broader Diffusion × ECC instantiations when their deployment assumptions are satisfied.**
