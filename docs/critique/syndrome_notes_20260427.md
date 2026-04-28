# S-IEC Syndrome: 정의, 기하학, 일반화

작성일: 2026-04-27

---

## 1. Syndrome 정의

### 수식

```
ŝ_t = x̂_0(t) − x̂_0(t−1)

r_t = ‖ŝ_t‖² / d          (syndrome score, scalar)
```

- `x̂_0(t)` : 현재 상태 `xt`에서 posterior mean으로 구한 clean image 추정 (Tweedie estimator)
- `x̂_0(t−1)` : `xt`를 한 스텝 DDIM으로 진행한 `xt_tent`에서 구한 clean image 추정 (lookahead)
- `d` : 차원 수 (정규화용)

### 직관

> **"현재 스텝과 다음 스텝에서 바라본 clean image 추정이 얼마나 불일치하는가."**

네비게이션 비유:

```
"지금 위치에서 보면 목적지가 A다"      ← x̂_0(t)
              ↕  차이 = syndrome ŝ_t
"한 걸음 더 가서 보면 목적지가 B다"    ← x̂_0(t−1)

‖A − B‖이 작다  → 경로가 일관됨 (정상)
‖A − B‖이 크다  → 뭔가 틀렸다  → correction trigger
```

### ECC와의 대응

| ECC | S-IEC |
|---|---|
| `s = H·r` (parity check) | `ŝ_t = x̂_0(t) − x̂_0(t−1)` |
| valid codeword → `s = 0` | 올바른 trajectory → `ŝ_t ≈ 0` |
| `s ≠ 0` → 오류 검출 | `r_t > τ_t` → correction trigger |
| parity check matrix H | consecutive Tweedie consistency check |

**명칭이 "syndrome"인 이유**: ECC에서 따온 것. 의미도 동일하다 — "이 수신 신호(diffusion 상태)가 valid code(올바른 궤적) 위에 있는가?"를 check하는 값.

---

## 2. Tangent / Normal 기하학

### 코드의 용어 정의 (gaussian_model.py 기준)

이 코드베이스는 **ECC/신호처리 관점**의 naming을 사용하며, 표준 미분기하학 관례와 반대다.

| 이름 | 수학적 정의 | 의미 |
|---|---|---|
| **Normal space** | `span(U)` | 데이터가 놓인 방향 (data manifold) |
| **Tangent space** | `span(U)^⊥` | 데이터와 수직인 방향 (noise-only direction) |

```
Normal space = span(U)       : U의 k개 열(column)이 span하는 k차원 부분공간
                               = 실제 이미지 데이터가 있는 방향
Tangent space = span(U)^⊥    : 나머지 (d−k)차원
                               = 데이터가 전혀 없는 방향, 순수 노이즈 공간
```

> **주의**: 표준 미분기하학에서는 반대다 — "tangent to manifold" = 데이터 방향, "normal to manifold" = 수직 방향. 이 코드의 naming은 내부적으로 일관되지만 외부 문헌과 혼동할 수 있음.

### 투영 연산자 (코드 구현)

```python
def project_normal(self, v):
    """span(U) 위로 투영: P_U · v = U (U^T v)"""
    return self.U @ (self.UT @ v)          # 1D case

def project_tangent(self, v):
    """span(U)^⊥ 위로 투영: (I − P_U) · v"""
    return v - self.project_normal(v)
```

**수학적으로 맞다.** `U^T U = I_k`이면:
- `P_U = U U^T` : orthogonal projector onto span(U) ✓
- `I − P_U` : orthogonal projector onto span(U)^⊥ ✓

**Batch case `(v @ U) @ UT`도 맞다:**
```
(n,d) @ (d,k) → (n,k)
(n,k) @ (k,d) → (n,d)   ← 각 행 i에 대해 P_U·v_i를 수행한 것과 동일 ✓
```

**Docstring 표기 주의**: `project_normal`의 docstring에 `P^⊥`로 적혀 있는데, 이 코드에서 ⊥는 "normal space = span(U)"를 가리키는 것으로, 표준 ⊥(orthogonal complement) 표기와 다르다. 계산은 맞다.

---

## 3. Toy Gaussian에서 Syndrome이 감지하는 것

### 모델 설정

```
X_0 = U Z,  Z ~ N(0, Λ),  U ∈ R^{d×k},  U^T U = I_k
posterior mean: x̂_0(t) = U · D_t · U^T · xt
D_t = diag(α_t² λ_i / (α_t² λ_i + σ_t²))
```

### Case 1: Normal space (span(U)) 방향 오류

```
xt = xt* + ε·u,   u ∈ span(U)
U^T·u ≠ 0

x̂_0(t)   = x̂_0*(t) + ε·U·D_t·(U^T·u)
x̂_0(t−1) = x̂_0*(t−1) + ε·U·D_{t-1}·G·(U^T·u)

D_t ≠ D_{t-1}  (timestep마다 shrinkage 계수 다름)
→ syndrome ŝ_t ∝ ε    ← syndrome이 오류 크기에 비례
```

**→ syndrome이 Normal space 오류를 감지한다. (fig1 plot (a)의 선형 상관)**

### Case 2: Tangent space (span(U)^⊥) 방향 오류

```
xt = xt* + ε·v,   v ⊥ span(U)
U^T·v = 0     ← 핵심: U^T가 이 성분을 완전히 소거

x̂_0(t)   = U·D_t·U^T·(xt* + ε·v) = U·D_t·U^T·xt* = x̂_0*(t)
x̂_0(t−1) = x̂_0*(t−1)                            (마찬가지로 unaffected)

syndrome ŝ_t = x̂_0*(t) − x̂_0*(t−1)   ← ε에 무관 (baseline 값만)
```

**→ syndrome이 Tangent space 오류를 감지하지 못한다. (fig1 plot (b)의 flat)**

### 이게 왜 올바른 설계인가?

**핵심 구분: S-IEC가 수정하는 것은 "데이터 노이즈"가 아니다**

"데이터 방향 변동"에는 두 종류가 있다:

```
A. 자연 다양성: 서로 다른 xT → 서로 다른 이상적 궤적 → 서로 다른 이미지
   → 각 trajectory 내부적으로 consistent → syndrome ≈ 0 → S-IEC 건드리지 않음

B. Deployment error: 같은 xT인데 PTQ/DeepCache가 x̂_0(t)를 오염
   → x̂_0(t)와 x̂_0(t-1)이 불일치 → syndrome 큼 → S-IEC 교정
```

S-IEC가 수정하는 것은 B뿐이다. Syndrome은 "trajectory 내부의 inconsistency"를 측정하지, 어떤 이미지로 수렴하는지를 판단하지 않는다.

**왜 deployment error가 Normal space(span(U)) 방향으로 나타나는가?**

Posterior mean은 항상 span(U) 안에 값을 가진다:
```
x̂_0(t) = U · D_t · U^T · xt  →  항상 ∈ span(U)
```
따라서 PTQ, DeepCache 등이 x̂_0(t)를 오염시켜도 그 오염은 span(U) 안에서 일어난다. Syndrome은 이 span(U) 방향의 불일치를 감지하는 것이다.

**Tangent space(span(U)^⊥) 오류는 DDIM에서 자연 감쇠된다:**

```
xt_prev = α_{t-1}·x̂_0 + (σ_{t-1}/σ_t)·(xt − α_t·x̂_0)

span(U)^⊥ 성분 ε·v:
  x̂_0 = U D_t U^T (xt* + ε·v) = x̂_0*(t)   ← U^T·v = 0 으로 소거
  xt_prev = xt_prev* + ε·v·(σ_{t-1}/σ_t)   ← 오직 이 채널로만 전파

σ_{t-1} < σ_t (σ_t 단조 감소) → σ_{t-1}/σ_t < 1 → 매 스텝 감쇠 → t=0에서 소멸
```

따라서:
- **Normal space 오류** (span(U)): deployment error가 x̂_0를 오염 → posterior mean 채널로 계속 전파 → syndrome 감지 → S-IEC 교정 ✓
- **Tangent space 오류** (span(U)^⊥): x̂_0가 이를 소거 → 오직 σ_{t-1}/σ_t 채널로만 전파 → 자연 감쇠 → 교정 불필요 ✓

이것이 S-IEC의 selective stabilization 설계 의도다.

---

## 4. 일반 DDPM에서 Syndrome: 무엇이 같고 무엇이 다른가

### 4.1 수식은 동일하다

```python
# Toy Gaussian
syndrome = x0_current - x0_lookahead

# CIFAR DDPM (IEC/siec_core/syndrome.py)
syndrome = x0_current - x0_lookahead
score = (syndrome ** 2).flatten(1).sum(dim=1) / d
```

공식은 완전히 같다. 달라지는 것은 `x̂_0`를 어떻게 구하느냐다.

| | Toy Gaussian | General DDPM |
|---|---|---|
| `x̂_0(t)` 계산 | closed-form posterior mean | UNet `f_θ(xt, t)` |
| 데이터 다양체 | explicit: `span(U)` | implicit: 학습된 분포 |
| Normal/Tangent 분해 | `U^T` 투영으로 해석적 계산 | **분석적으로 불가능** |
| Clean sampler syndrome | = 0 (수학적 보장) | ≈ 0 (통계적 평균, 보장 없음) |
| Tweedie martingale | 정확히 성립 | 근사적으로만 성립 |

### 4.2 무엇이 깨지는가

**G1: Clean sampler에서도 syndrome ≠ 0**

UNet `f_θ`는 posterior mean을 근사할 뿐이다. 오류가 전혀 없는 clean deployment에서도:

```
ŝ_t = f_θ(xt, t) − f_θ(xt_tent, t−1) ≠ 0
```

이 non-zero baseline은 세 가지가 뒤섞인 것이다:
1. **모델 bias**: UNet이 posterior mean을 정확히 구현하지 못함
2. **이산화 오차**: DDIM step의 numerical error
3. **실제 deployment error**: PTQ, DeepCache 등이 유발하는 오류

세 가지를 syndrome score 하나로 구분할 수 없다.

**G2: Normal/Tangent 분해 불가**

CIFAR에서는 "데이터 다양체"가 명시적이지 않아 `span(U)^⊥` 개념이 없다. 따라서 "syndrome이 Normal space 오류를 감지하고 Tangent space 오류를 무시한다"는 이론적 보장이 직접 적용되지 않는다.

### 4.3 그럼에도 syndrome이 작동하려면: 구현 요건

**요건 1: Threshold calibration (필수)**

Baseline syndrome을 제거하기 위해 threshold `τ_t`를 clean deployment에서 교정해야 한다:

```python
# SIEC.calibrate_thresholds (siec_sim/core/siec.py)
# clean deployment로 n_calib 샘플 실행 → syndrome score 분포 측정
# τ_t = percentile(scores_at_t, p)

# CIFAR 구현 (IEC)에서도 동일한 방식
tau = calibrate_tau(clean_scores, percentile=90)
```

절대값 threshold가 아니라 **상대적 percentile threshold**를 쓰는 이유가 바로 이것이다.

**요건 2: 경험적 상관 검증 (실험 A)**

일반 DDPM에서 syndrome이 실제 deployment error와 상관관계를 갖는지는 **이론적으로 보장되지 않고 실험으로 확인해야 한다.** 구체적으로:

```
검증해야 할 것: r_t(deployed) > r_t(clean) 가 통계적으로 유의한가?

실험 방법:
  1. Clean DDPM으로 N개 샘플 → syndrome score 분포 baseline
  2. PTQ'd DDPM으로 N개 샘플 → syndrome score 분포
  3. 두 분포의 분리 정도 측정 (AUROC 또는 KL)
```

**요건 3: UNet 호출 방식 (NFE 고려)**

`x̂_0(t−1)`을 구하려면 추가 UNet forward pass가 필요하다:

```python
# Step t에서 syndrome 계산
x0_t    = unet(xt, t)          # 1 NFE
xt_tent = ddim_step(xt, x0_t, t)
x0_t1   = unet(xt_tent, t-1)   # 1 NFE (추가)

syndrome = x0_t - x0_t1
score = (syndrome**2).mean()

# Correction이 trigger되지 않으면: x0_t1을 다음 스텝에 재사용
# (lookahead reuse trick → 실질 NFE overhead ≈ 0 when correction rate is low)
```

### 4.4 일반 DDPM에서 syndrome 구현 요약

```python
def compute_syndrome(unet, xt, t, schedule):
    """
    Returns: syndrome vector, syndrome score
    NFE: 2 (x0_t + x0_t1). x0_t1은 다음 스텝에 재사용 가능.
    """
    x0_t = unet(xt, t)                        # Tweedie at t
    xt_tent = ddim_step(xt, x0_t, t, schedule)
    x0_t1 = unet(xt_tent, t - 1)             # Tweedie at t−1 (lookahead)

    syndrome = x0_t - x0_t1
    score = (syndrome ** 2).mean()            # per-sample scalar
    return syndrome, score

def should_correct(score, tau_t):
    """tau_t는 clean deployment에서 calibrate된 percentile threshold."""
    return score > tau_t
```

---

## 5. IEC vs S-IEC — 정확한 구분

### 5.1 실제 IEC (p_sample_ddim_implicit_2, README 기준 공식 구현)

IEC는 syndrome을 쓰지 않는다. 메커니즘은 **xt_prev에 대한 fixed-point iteration**이다:

```python
# Step 1: 초기 DDIM step
x_prev_hat = DDIM(xt, t)          # 1 NFE

# Step 2: Fixed-point iteration
for iter in range(max_iter - 1):  # 최대 2회 추가
    e_t = model(x_prev_hat, t)    # 같은 t로 x_prev_hat을 다시 넣음
    x_prev_new = DDIM_formula(e_t)

    delta = ||x_prev_new - x_prev_hat|| / ||x_prev_hat||
    if delta < tol: break

    x_prev_hat += (gamma^iter) × (x_prev_new - x_prev_hat)  # 댐핑 업데이트
```

"x_prev_hat를 다시 모델에 입력했을 때 출력이 자기 자신과 일관되는가"를 체크한다. Syndrome(x̂_0 차이)과는 무관하다.

### 5.2 세 방법의 정확한 구분

| | 실제 IEC | NaiveRefinement (baselines.py) | S-IEC |
|---|---|---|---|
| 메커니즘 | fixed-point iteration on xt_prev | syndrome → 항상 consensus correction | syndrome → r_t > τ_t 일 때만 correction |
| Syndrome 계산 | **없음** | 있음 (항상 적용) | 있음 (선택 적용) |
| NFE overhead | 최대 max_iter/step | 항상 2/step | trigger 시 2, 아니면 1 (lookahead reuse) |
| 코드 위치 | `mainldm/ldm/models/diffusion/ddim.py` | `siec_sim/core/baselines.py` | `siec_sim/core/siec.py` |

**주의**: `baselines.py`의 `NaiveRefinement`는 "IEC-style"로 레이블되어 있지만 실제 IEC와 메커니즘이 다르다. NaiveRefinement는 "항상 syndrome 교정을 적용하면 어떻게 되는가"를 보여주는 기준선이지, IEC 구현체가 아니다.

### 5.3 exp4와의 관계

`siec_sim/run_all.py`의 Experiment 4 (Pareto 비교)에서:
- "Naive (always)" = NaiveRefinement = 항상 syndrome 교정
- 실제 IEC(fixed-point)는 exp4에 없음
- 실제 IEC vs S-IEC 비교는 `IEC/experiments/real_03_iec_vs_siec_fid.py`에서 NPZ 파일로 수행

따라서 exp4의 NFE 오버헤드 차이는 "항상 교정 vs 선택 교정"의 차이이며, 실제 IEC의 오버헤드와는 별개다.

### 5.4 IEC도 on-manifold (tangent/span(U)) 오차를 교정하는가?

**결론: 그렇다. IEC도 같은 방향을 교정한다. 메커니즘만 다르다.**

#### 왜 IEC도 on-manifold 교정인가?

Fixed-point iteration의 핵심 연산:

```
x̂_0 = (xt − et·σ_t) / α_t       ← Tweedie estimate
                                    항상 data manifold(span(U)) 근방
x_prev_hat += γ^iter · (x_prev_new − x_prev_hat)
```

`x̂_0`는 항상 span(U) 방향 값만 가진다 (Toy Gaussian에서는 수학적으로, 일반 DDPM에서는 UNet이 data distribution 안에서 출력). Fixed-point iteration은 이 `x̂_0`의 **동일 timestep 내 자기일관성(self-consistency)**을 강제하므로, 교정 방향은 on-manifold다.

span(U)^⊥ (off-manifold, normal) 방향 오차는 어떤가?
- `U^T·v = 0` → `x̂_0`에 영향 없음 → fixed-point iteration이 볼 수 없음
- σ_{t-1}/σ_t 감쇠로 자연 소멸 → IEC도 개입하지 않음

#### IEC vs S-IEC 교정 대상 비교

| | IEC | S-IEC |
|---|---|---|
| **교정 대상** | span(U) on-manifold 오차 | span(U) on-manifold 오차 |
| **교정 안 하는 것** | span(U)^⊥ off-manifold (DDIM이 처리) | span(U)^⊥ off-manifold (DDIM이 처리) |
| **메커니즘** | 같은 t에서 fixed-point iteration | 연속 timestep 간 syndrome 임계값 비교 |
| **일관성 체크 방향** | **timestep 내부** (xt_prev의 self-consistency) | **timestep 간** (x̂_0(t) vs x̂_0(t−1)) |
| **트리거 조건** | DeepCache refresh step마다 항상 | score_t > τ_t 일 때만 |
| **NFE 오버헤드** | 항상 추가 NFE (최대 max_iter−1회) | 트리거될 때만 +1 NFE (아니면 lookahead reuse) |

#### 한 문장 요약

> IEC는 **"지금 이 step에서 x̂_0이 수렴할 때까지 반복"**으로 on-manifold 오차를 교정하고,  
> S-IEC는 **"직전 step과 현재 step의 x̂_0이 너무 다를 때만 선택적으로 평균"**으로 on-manifold 오차를 교정한다.  
> 두 방법 모두 span(U)^⊥ (off-manifold) 오차는 건드리지 않는다 — DDIM의 σ_{t-1}/σ_t 감쇠가 처리한다.

---

## 6. 요약 대조표

| 속성 | Toy Gaussian | General DDPM |
|---|---|---|
| Syndrome 수식 | `x̂_0(t) − x̂_0(t−1)` | **동일** |
| `x̂_0` 계산 | closed-form | UNet forward pass |
| Clean syndrome = 0 | 수학적 보장 | 보장 없음 → calibration 필요 |
| Normal space 오류 감지 | 이론적 보장 (Theorem 6.1) | 경험적 검증 필요 |
| Tangent space 오류 무시 | 수학적 보장 (U^T·v = 0) | 동일한 현상 기대되나 보장 없음 |
| Threshold τ_t | 이론적으로 유도 가능 | Percentile calibration 필수 |
| Correction 효과 | Theorem 6.2-6.4로 보장 | 실험 B (impact test)로 확인 필요 |

**IEC / S-IEC / DDIM 역할 분담 요약**

| 오차 유형 | 방향 | 처리 주체 |
|---|---|---|
| Deployment error (PTQ, DeepCache) | span(U) on-manifold | IEC (fixed-point) 또는 S-IEC (syndrome 선택) |
| 자연 다양성 (서로 다른 xT) | span(U) on-manifold | 건드리지 않음 (syndrome ≈ 0) |
| Off-manifold noise | span(U)^⊥ | DDIM σ_{t-1}/σ_t 감쇠 (자동) |

---

## 7. Code Checklist

| 항목 | 파일 | 맞는가 |
|---|---|---|
| `project_normal` 구현 | `gaussian_model.py:74-79` | ✓ (U U^T 투영, batch도 맞음) |
| `project_tangent` 구현 | `gaussian_model.py:81-83` | ✓ (I − P_U 투영) |
| Docstring의 `P^⊥` 표기 | `gaussian_model.py:75,82` | ⚠ 비표준 (계산은 맞음, 표기가 표준 ⊥ 의미와 다름) |
| Toy syndrome 계산 | `siec_sim/core/siec.py:56-57` | ✓ |
| CIFAR syndrome 계산 | `IEC/siec_core/syndrome.py` | ✓ |
| Lookahead reuse | `siec_sim/core/siec.py:99` | ✓ (net 1 NFE when not triggered) |
