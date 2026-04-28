---
title: "S-IEC manifold 기하학 — SDDM 정의 차이와 후속 실험(G/H/I)"
date: 2026-04-27
author: 분석 보조 (Claude)
related:
  - docs/siec_critique_20260427.md
  - docs/siec_ecc_framing_20260427.md
  - siec_sim/core/gaussian_model.py
  - siec_sim/core/perturbations.py
  - siec_sim/run_all.py
---

# S-IEC manifold 기하학 — SDDM 정의 차이와 후속 실험

> 한 줄 요약. 사용자(용성)가 짚은 두 가지 의문은 모두 정확하다. (1) SDDM의 tangent/normal과 S-IEC의 normal/tangent error는 다른 기하 객체다. (2) **toy code의 라벨링이 표준 미분기하와 정반대**라서, fig1_observability.png의 "syndrome은 normal에만 민감" 결과는 표준 컨벤션으로는 "syndrome은 **standard-tangent (along-manifold)** 에만 민감"이며, 이는 ECC-with-manifold-codeword가 기대하는 정반대 방향이다. 즉 **toy는 'data manifold = codeword' 가설을 입증하는 게 아니라 반증한다**. 실제 codeword는 data manifold가 아니라 "Tweedie 시간 일관성을 만족하는 trajectory"이고, toy에서는 posterior mean의 closed-form 사영 때문에 우연히 span(U)와 align될 뿐이다. 이 정정은 §3, §4의 framing을 다시 쓰게 만든다.

> **정정 노트 (2026-04-28)**: 이 문서 v1은 §3에서 코드의 inverted-label을 그대로 써서 "syndrome은 off-manifold에만 민감"이라고 적었다. §2에서 inversion을 인지했음에도 §3.1·§3.3·§4의 가설/판정이 모두 그 inverted-label 위에 세워져 자체 모순이었다. v2(이 버전)에서 §3, §4.1, §4.2, §4.3을 표준 컨벤션으로 다시 쓴다.

이 문서는 다음 네 가지를 다룬다.

1. SDDM ↔ S-IEC 정의 차이 (왜 critique §5의 SDDM 인용이 부정확한가)
2. 코드 측 명명 컨벤션 이슈 (`siec_sim/core/gaussian_model.py`)
3. toy가 작동하는 진짜 이유와 CIFAR에서 깨지는 이유의 manifold-기하학적 설명
4. 후속 실험 G / H / I 설계 + 마감 1주 우선순위

`docs/siec_critique_20260427.md`(결과 기준 비판), `docs/siec_ecc_framing_20260427.md`(ECC × Diffusion framing + 실험 A–F)와 상호보완이다. 본 문서는 전자 §5의 SDDM 인용에 대한 명시적 정정과 후자 실험 A–F의 manifold-aware 확장을 다룬다.

---

## 1. SDDM ↔ S-IEC — 같은 단어, 다른 기하

### 1.1 두 정의의 비교

| 항목 | S-IEC (사용자 정의 — 표준 미분기하) | SDDM (ICML'23, 2308.02154) |
|---|---|---|
| 기준 manifold | **데이터 manifold M** (clean data 분포의 support, = M_0) | **noise level별 manifold {M_t}** (각 t별 다른 level set) |
| Tangent 의미 | M 위에서 움직임 (on-manifold drift) | M_t 안에서 image 다듬기 ("refinement") |
| Normal 의미 | M에서 벗어남 (off-manifold) | M_t → M_{t-1} 이동 ("denoising") |
| 원인이 되는 기하 객체 | T_x M, N_x M (data 분포 자체) | T_{x_t} M_t, N_{x_t} M_t (noise-perturbed level set) |

SDDM 본문 인용 (web search 기반):

> "components of gradients are restricted on the tangent space T_{x_t} M_t as 'refinement' parts, while components pointing to the next manifold M_{t-1} are obtained by restricting … on the N_{x_t} M_t"
>
> "s(x) = s_r(x) + s_d(x), where s_r = tangent (refinement on manifold), s_d = normal (denoising)"

→ SDDM의 "normal"은 "다음 noise level로 이동하는 방향"이고, "tangent"는 "같은 noise level 안에서 이미지 내용을 다듬는 방향"이다. 데이터 분포 support의 직교 분해가 아니다.

S-IEC의 정의 (사용자 메시지):

- ε^⊥ (normal error) : data manifold M 에서 벗어나는 방향의 error
- ε^∥ (tangent error) : data manifold M 위에서 움직이는 방향의 error
- 여기서 M = "전체 데이터 분포가 사는 곳"

→ S-IEC는 error를 데이터 manifold 자체의 직교 분해로 본다. SDDM과는 **다른 기하 객체**를 가리킨다.

### 1.2 critique §5의 SDDM 인용에 대한 정정

`docs/siec_critique_20260427.md` §5 표에는 SDDM이 "Score를 tangent+normal로 decompose, 학습 단계에 통합"으로 적혀 있다. 사실 자체는 맞으나, 이 분해가 **S-IEC가 의도하는 normal/tangent와 다른 기하**임을 명시하지 않은 채 인용했다. 그대로 paper에 들어가면 reviewer가 "왜 SDDM에서는 되는데 너희는 안 되는가" 라는 질문을 던질 수 있고, 답은 "분해 방향 자체가 다른 framework"가 되어야 한다.

S-IEC framing의 진짜 reference 후보(데이터 manifold tangent/normal을 다루는 흐름):

| 논문 | framework |
|---|---|
| **Be Tangential to Manifold** (2510.05509) | score Jacobian의 spectral structure로 **data manifold의 tangent/normal**을 분리. training-free, SD2.1 실증. **S-IEC와 같은 framework**. |
| **Niso-DM / Tango-DM** (2505.09922) | "Manifold Data" 가정 아래 score function singularity 완화. 데이터 manifold framework. 학습 단계 변형이 필요. |
| **What's Inside Your Diffusion Model?** (2505.11128) | score 기반 Riemannian metric. 데이터 manifold framework. CIFAR/CelebA, training-free. |

**권고**: critique §5의 SDDM row를 "다른 framework — direct comparison 불가" 로 표기하고, 실험·discussion에서의 primary reference를 위 세 논문으로 옮긴다. SDDM은 contrast로만 인용("we instead decompose deployment error along the data manifold's tangent/normal directions, in the spirit of [2510.05509, 2505.11128]; SDDM's decomposition is along the noise-evolution direction and is not a directly comparable baseline.").

---

## 2. 코드 측 명명 컨벤션 이슈

`siec_sim/core/gaussian_model.py:74-83`:

```python
def project_normal(self, v: np.ndarray) -> np.ndarray:
    """P^⊥ v = U (U^T v).  Normal space = span(U)."""
    ...

def project_tangent(self, v: np.ndarray) -> np.ndarray:
    """P^∥ v = v - P^⊥ v.  Tangent space = span(U)^⊥."""
```

low-rank Gaussian 모델 `X_0 = U Z, U ∈ R^{d×k}` 에서 데이터가 사는 곳은 span(U). **표준 미분기하학**에서 linear manifold의 경우 모든 점이 같은 tangent space를 공유하고, 이 tangent space가 곧 manifold가 사는 subspace = span(U). 즉:

- 표준 정의: tangent = span(U), normal = span(U)^⊥
- 코드 정의: normal = span(U), tangent = span(U)^⊥

코드의 라벨이 표준과 정반대다. 사용자 메시지의 정의("ε^⊥ = manifold에서 벗어나는") 도 표준에 따른 것이므로, 코드 라벨과 사용자 정의가 정반대인 셈.

`siec_sim/run_all.py:140`:

```python
ax.set_xlabel(r'$\|e_t^\perp\|$ (normal error)')
# perturbation 생성: noise = model.project_normal(noise) → span(U)에 사영
```

이 figure에서 "normal error"로 표기된 축은 **데이터가 사는 방향**(코드 정의의 normal = span(U))의 perturbation이다. 표준/사용자 정의로는 **tangent error**.

### 2.1 영향과 권고

- 코드 자체는 1저자 영역이 아니지만 (`siec_sim/`은 toy sim), `feedback_modify_protocol.md`의 "1저자 미수정" 원칙 바깥이므로 수정 가능. 다만 변경하면 figure 재생성·기존 plot 재해석이 필요.
- **paper 단계에서 가장 안전한 처리**: 코드는 그대로 두되, paper의 수식·figure에서는 표준 컨벤션으로 라벨링. 즉:
  - paper: ‖e^⊥‖ = off-manifold (data가 안 사는 방향) error
  - 코드 변수 `e_normal` → paper 라벨 `‖e^∥‖` (tangent in standard convention)
  - 코드 변수 `e_tangent` → paper 라벨 `‖e^⊥‖` (normal in standard convention)
- 더 깨끗한 처리는 `siec_sim/core/gaussian_model.py`의 두 함수명을 swap하고 docstring/주석을 맞추는 것. 다만 `run_all.py`, `experiments/`, `runs/`의 기존 결과 호환성을 점검해야 한다.

---

## 3. toy가 보여주는 진짜 결과와 그것이 manifold framing을 반증하는 이유

### 3.1 toy의 비대칭성 — 메커니즘

`siec_sim/run_all.py:65-186` Experiment 1(Theorem 6.1 — Syndrome Observability):

- 같은 ‖perturbation‖로 코드 라벨 normal-only / tangent-only 두 종류 inject
- syndrome score 측정 → Fig 1a (vs ‖e^⊥‖, 코드 normal = span(U)), Fig 1b (vs ‖e^∥‖, 코드 tangent = span(U)^⊥), Fig 1c (κ/β)

수학적 메커니즘:

1. data가 사는 곳은 명시적인 linear k차원 subspace span(U). 표준 미분기하로 **span(U) = manifold-tangent (along-manifold), span(U)^⊥ = manifold-normal (off-manifold)**.
2. `gaussian_model.py:46`의 posterior mean은 closed-form
   `m_t(x_t) = U Λ α_t (α²Λ + σ²I)⁻¹ U^T x_t`
   → **m_t의 image가 span(U)에 갇힘**, 또한 m_t는 x_t를 오직 `U^T x_t`로만 본다 (kernel = span(U)^⊥).
3. lookahead clean-estimate `m_{t-1}(x_t_tent)`도 같은 사영 구조라 span(U) 안.
4. 따라서 `syndrome = x̂_0(t) − x̂_0(t-1) ∈ span(U)`.
5. **표준 manifold-normal (span(U)^⊥) perturbation**: U^T가 0으로 사영 → m_t 변화 없음 → syndrome 변화 없음 (Fig 1b 평평).
6. **표준 manifold-tangent (span(U)) perturbation**: U^T 비영 → m_t 응답 → syndrome 변함 (Fig 1a 증가).

### 3.2 toy 결과를 표준 라벨로 다시 읽기

| 코드 라벨 (figure 축) | 표준 미분기하 라벨 | toy 결과 |
|---|---|---|
| ‖e^⊥‖ ("normal error") = span(U) 방향 | manifold-**tangent** (along-manifold) | syndrome **강한 응답** |
| ‖e^∥‖ ("tangent error") = span(U)^⊥ 방향 | manifold-**normal** (off-manifold) | syndrome **거의 무반응** |

표준 컨벤션으로 옮긴 toy의 핵심 결과:

> **toy는 "syndrome이 along-manifold (standard-tangent) error에 민감하고 off-manifold (standard-normal) error에 둔감"임을 보여준다.**

### 3.3 이게 ECC-with-manifold-codeword를 반증하는 이유

ECC framing에서 codeword=data manifold M라면 syndrome decoding의 표준 기대:

| 방향 | codeword 성질 | syndrome 기대 |
|---|---|---|
| along-manifold (tangent to M) | 여전히 valid codeword | s = 0 (둔감해야) |
| off-manifold (normal to M) | invalid (out of code) | s ≠ 0 (민감해야) |

toy 결과는 둘 다 **정반대**:
- along-manifold에 민감 (codeword 안 움직임을 detect한다는 모순)
- off-manifold에 둔감 (codeword 밖으로 나간 걸 못 본다는 모순)

→ **"codeword = data manifold" 가설은 toy 자체로 반증된다.**

### 3.4 toy가 작동하는 진짜 이유 — implicit codeword

코드의 syndrome 정의 `s = x̂_0(t) − x̂_0(t-1)`이 0이 되는 조건은:

$$\hat{x}_0(t) = \hat{x}_0(t-1) \iff \text{Tweedie 추정이 인접 step에서 일치}$$

이게 진짜 implicit codeword:

$$\mathcal{C}_{\text{implicit}} = \{ \text{trajectory} : \text{model의 posterior mean이 인접 step에서 일관} \}$$

**이건 ambient state space의 부분집합이 아니라 trajectory space의 부분집합**이라 data manifold M ⊂ ℝ^d와는 카테고리 자체가 다른 객체.

toy에서 우연히 align되는 이유:
- m_t의 image가 span(U)에 갇힘 → x̂_0(t), x̂_0(t-1) 둘 다 span(U) 안
- syndrome도 span(U) 안 → "Tweedie 일관성 위반"이 정확히 span(U) 안의 mismatch로 표현됨
- → toy에서 implicit codeword와 span(U)가 align됨 (manifold와 무관)

이건 **모델이 가진 사영 구조의 결과**이지 manifold 기하의 결과가 아니다. 일반 모델(예: 학습된 UNet)에서는 이 align이 보장되지 않는다.

### 3.5 CIFAR DDPM에서의 의미

`IEC/experiments/yongseong/deepcache_denoising.py:386-401`의 syndrome:
- 학습된 UNet `s_θ`의 image는 명시적 부분공간에 갇히지 않음.
- syndrome=0 조건은 여전히 "Tweedie 시간 일관성"이지만, 이게 어떤 ambient 부분공간과 align된다는 보장이 없음.
- 따라서 syndrome이 standard manifold-tangent / manifold-normal 어느 쪽과도 자동으로 align되지 않는다.

`siec_critique_20260427.md` §3의 "syndrome score가 random과 차이 없음"은 이 implicit codeword가 deployment error의 주요 성분과 정렬되어 있지 않다는 증거로 해석해야 한다 (manifold framing의 implicit 문제가 아니라 codeword 자체의 정의 문제).

---

## 4. 후속 실험 G / H / I

`siec_ecc_framing_20260427.md`의 실험 A–F를 manifold-aware로 확장한다. 핵심은 "**데이터 manifold tangent/normal을 명시적으로 추정해 syndrome 분해 능력을 측정**"하는 것.

### 4.1 실험 G — Manifold-Aware Syndrome Sensitivity (≈1일)

toy의 Theorem 6.1을 CIFAR로 옮기는 first-pass evidence.

**Manifold tangent estimator** — 비용·정확도 trade-off:

| 방법 | 비용 | 정확도 |
|---|---|---|
| (a) **PCA-based** | 가장 가벼움. fp16 trajectory에서 (x_t, x̂_0(t)) 모아 sample covariance의 top-k eigenvector를 tangent로. | 매우 거칠지만 first-pass에는 충분 |
| (b) **Score Jacobian** ([Be Tangential to Manifold](https://arxiv.org/abs/2510.05509) 방식) | UNet `s_θ`의 Jacobian-vector product. K개 forward 추가. | 가장 정확. paper-grade |
| (c) **Pretrained autoencoder/VAE encoder Jacobian** | encoder Jacobian의 column space를 tangent로. | 중간. CIFAR DDPM에 잘 맞는 VAE 가용 여부 확인 필요 |

권장: **(a) PCA-based로 먼저 1일 진행 → 결과 살아남으면 (b) 로 강화.**

**세팅**:

1. fp16 reference trajectory에서 step별 (x_t, x̂_0(t)) 수집.
2. tangent space 추정 (PCA, k = 64 또는 128 추천).
3. synthetic perturbation 주입:
   - `δ_normal = P^⊥ δ_iso` (off-manifold, 표준 정의)
   - `δ_tangent = P^∥ δ_iso` (on-manifold, 표준 정의)
   - 같은 ‖δ‖로 정규화.
4. step별 syndrome score 측정.

**Plot/판정 (표준 라벨로 정정)**:

- ‖s_t‖ vs ‖δ^⊥‖ (off-manifold, standard-normal), ‖s_t‖ vs ‖δ^∥‖ (on-manifold, standard-tangent) 두 곡선.
- 비율 ρ := slope(‖s‖ vs ‖δ^⊥‖) / slope(‖s‖ vs ‖δ^∥‖).
- **ρ > 2 (off-manifold 5–10× 더 민감)**: ECC-with-manifold-codeword framing 살아남음 → ICLR Figure 1 후보. (이는 toy와 정반대 패턴이고, 학습된 UNet이 toy의 closed-form 사영보다 더 ECC-적 codeword를 정의하고 있음을 의미)
- **ρ ≈ 1**: framing 미입증 — syndrome이 manifold 분해를 따르지 않음. critique §7-4의 limitation figure로 전환.
- **ρ < 1 (on-manifold에 더 민감, toy와 같은 패턴)**: ECC-with-manifold-codeword framing **반증**. paper의 ECC narrative 자체를 다시 짜야 함. 대안 codeword(예: Tweedie 시간 일관성, low-frequency content, …)를 명시적으로 제안해야 한다.

### 4.2 실험 H — Real Deployment Error Decomposition (≈1일)

paper의 가장 강한 figure 후보. exp5의 family-specific behavior(quant family는 IEC<S-IEC, cache family는 IEC>S-IEC)를 manifold 기하학으로 직접 설명.

**세팅**:

- synthetic이 아니라 **실제** deployment error δ_real = x_t^deploy − x_t^ref를 step·family별로 측정.
- 실험 G에서 추정한 tangent space로 δ_real을 P^⊥ / P^∥ 로 분해.
- family별 (`fp16 / w8a8 / dc10 / w4a8 / dc20 / cachequant`) ‖δ^⊥‖ / ‖δ^∥‖ 비율을 plot.

**예상 결과 (가설) — §3 정정 후 재서술**:

기존 v1의 가설은 "syndrome은 off-manifold에 민감"이라는 inverted-label framing 위에 세워져 있어 무효. v2에서는 두 가능성을 병렬로 제시한다:

(i) **manifold-codeword framing이 살아있는 시나리오** (실험 G에서 ρ > 2):

| family | 예상 ‖δ^⊥‖ / ‖δ^∥‖ | 의미 |
|---|---|---|
| quantization (`w8a8`, `w4a8`) | ≫ 1 (off-manifold dominant) | rounding이 manifold 외부로 튀게 만듦 → S-IEC가 잡아냄 |
| cache (`dc10`, `dc20`) | ≤ 1 (on-manifold drift) | feature 재활용은 manifold 위에서 클래스/모드를 옮기는 형태 → S-IEC가 못 잡음 |
| `cachequant` | 두 성분 혼재, 큰 ‖δ^∥‖이 dominant | exp5에서 가장 안 좋은 setting인 이유 |

(ii) **manifold-codeword framing이 깨진 시나리오** (실험 G에서 ρ ≤ 1, toy와 같은 패턴):

이 경우 family-specific behavior는 manifold 기하로 설명되지 않는다. 대신 **"deployment error가 model의 implicit codeword (Tweedie 시간 일관성) 와 얼마나 정렬되는가"** 로 다시 framing해야 한다. 구체적으로:
- quant family에서 IEC<S-IEC: quant error가 Tweedie 일관성과 잘 정렬 → syndrome이 잡음.
- cache family에서 IEC>S-IEC: cache error가 Tweedie 일관성과 정렬 안 됨 → syndrome 못 잡음, IEC의 fixed-point가 더 효과적.

실험 G가 (ii)로 나오면 paper의 ECC narrative를 **"implicit codeword = Tweedie 시간 일관성"** 로 다시 써야 한다 (§3.4 참조). 이 plot은 결과와 무관하게 figure로 가치 있다 — 어느 쪽이든 family-specific behavior의 진짜 출처를 가른다.

### 4.3 실험 I — Manifold-Aware S-IEC Trigger (≈1.5일, 실험 G가 ρ > 2 시나리오로 살았을 때만)

기존 S-IEC의 syndrome score 정의를 manifold-aware로 교체.

**기존**: `score = ‖x̂_0(t) − x̂_0(t-1)‖² / d`

**제안 (표준 라벨로 정정)**: `score_⊥ = ‖P^⊥_M(x̂_0(t) − x̂_0(t-1))‖² / d_⊥`
- 여기서 `P^⊥_M`은 **표준 manifold-normal** (off-manifold) 사영. 실험 G에서 추정한 tangent space U에 대해 `P^⊥_M = I − U U^T`.
- 즉 syndrome의 **off-manifold 성분만** trigger 신호로 사용.
- 가설: deployment error의 off-manifold 성분이 dominant한 family(예상: quant)에서는 더 깨끗한 trigger; on-manifold dominant family(예상: cache)에서는 trigger rate 감소.

**중요한 caveat**: 실험 G가 ρ ≤ 1 (toy와 같은 패턴)으로 나오면 이 제안은 무의미하다. toy의 syndrome은 이미 span(U) 안에만 살아서 `P^⊥_M`을 적용하면 거의 0이 된다. 학습된 UNet에서도 같은 결과면 score_⊥는 trigger 신호로 못 씀. 이 경우 대신:
- `score_∥ = ‖P^∥_M(syndrome)‖² / d_∥` (on-manifold 성분만) — toy의 패턴을 그대로 따라가는 형식
- 또는 syndrome의 정의 자체를 "Tweedie 시간 일관성" 직접 측정으로 재설계

**실험**: exp4·exp5와 같은 6 setting에서 score_⊥ (또는 score_∥) 기반 S-IEC를 돌려 FID·trigger rate 측정.

**판정**:
- score_⊥ trigger가 실험 G의 (i) 시나리오 + cache family에서 No-correction 대비 개선 → success, manifold framework 강화.
- score_∥ trigger가 (ii) 시나리오에서 quant family에서 개선 → "syndrome ≈ implicit codeword" framing 강화.
- 둘 다 차이 없음 → critique §7-4 limitation path로 전환.

### 4.4 실험 J — paper Table 5 selective IEC와의 직접 비교 (≈0.5일)

`siec_critique_20260427.md` §2가 지적한 위협: paper의 ±1/10 selective IEC는 9.58 FID @ +2.8% overhead로 본 S-IEC(8.79 FID @ +91% NFE)를 dominate.

실험 D + I 가 살아남은 후에만 의미. 같은 NFE에서 paper Table 5와 본 S-IEC + manifold-aware trigger의 Pareto 비교 plot.

---

## 5. 마감 1주(2026-05-06) 우선순위

| 순위 | 실험 | 이유 | 예상 |
|---|---|---|---|
| 1 | **G** (PCA-based manifold sensitivity) | 가장 가볍게 framing 입증 가능. toy Theorem 6.1의 CIFAR 일반화 first-pass. | 1일 |
| 2 | **H** (real deployment error decomposition) | paper의 가장 강한 figure 후보. exp5의 family 비대칭을 직접 설명. 결과와 무관하게 paper-defensible. | 1일 |
| 3 | **D** (lookahead reuse fix, ecc_framing 문서) | NFE 비교 공정성 회복. cache reuse 설계 때문에 0.5일이 아니라 1일 잡는 게 안전. | 1일 |
| 4 | **I** (manifold-aware trigger) | G·H 살았을 때만 진행. paper contribution을 "manifold-aware syndrome decoder"로 재정의. | 1.5일 |
| 5 | **J** (paper Table 5와 Pareto 비교) | I 결과를 Pareto plot으로 정리. | 0.5일 |
| 6 | paper writing + ablation polish | | 2-3일 |

총 4–5일 + 글쓰기 2–3일. G가 깨지면 critique §7-4의 "negative limitation figure" path로 즉시 전환 (실험 H는 이때도 단독으로 figure 가치).

---

## 6. 기존 두 문서에 대한 수정 권고

### 6.1 `siec_critique_20260427.md`

- §5 표의 SDDM row를 다음으로 정정:
  > **SDDM** (ICML'23): *Different framework* — score를 noise-level direction(refinement vs denoising)으로 분해. 본 작업의 data-manifold tangent/normal과 다른 기하 객체. direct comparison 불가.
- §5 본문에 "S-IEC framing의 진짜 reference는 data-manifold geometry를 다루는 [Be Tangential to Manifold (2510.05509), Niso-DM (2505.09922), What's Inside (2505.11128)]." 한 줄 추가.

### 6.2 `siec_ecc_framing_20260427.md`

- §3 실험 A–F 위에 "manifold-aware extensions G/H/I — see siec_manifold_geometry_20260427.md" 한 줄.
- §2.2 G2 ("다양체가 명시적이지 않다") 에 "이 갭을 명시적 tangent estimation(실험 G)으로 메우려는 시도가 본 문서의 후속 실험" 한 줄.

### 6.3 (선택) 코드

- `siec_sim/core/gaussian_model.py`의 `project_normal` / `project_tangent` 명명 표준화. 다만 `run_all.py`·기존 결과 호환성 점검 필요. **paper 단계에서는 코드 그대로 두고 figure 라벨만 표준 컨벤션으로 출력하는 게 더 안전**.

---

## 7. Sources

### Manifold framework 비교
- [SDDM: Score-Decomposed Diffusion Models on Manifolds (ICML 2023, 2308.02154)](https://arxiv.org/abs/2308.02154) ([proceedings PDF](https://proceedings.mlr.press/v202/sun23n/sun23n.pdf))
- [Be Tangential to Manifold: Discovering Riemannian Metric for Diffusion Models (2510.05509)](https://arxiv.org/abs/2510.05509)
- [What's Inside Your Diffusion Model? A Score-Based Riemannian Metric (2505.11128)](https://arxiv.org/abs/2505.11128)
- [Improving the Euclidean Diffusion Generation of Manifold Data (Niso-DM, NeurIPS'25, 2505.09922)](https://arxiv.org/abs/2505.09922)

### 본 프로젝트
- `siec_sim/core/gaussian_model.py` (toy posterior mean + tangent/normal projector)
- `siec_sim/core/perturbations.py` (DirectionalPerturbation 'tangent'/'normal'/'mixed')
- `siec_sim/run_all.py:65-186` (Experiment 1: Theorem 6.1 syndrome observability)
- `IEC/experiments/yongseong/deepcache_denoising.py:386-401` (CIFAR S-IEC syndrome 계산)
- `docs/siec_critique_20260427.md` §5 (SDDM 인용 — 본 문서가 정정)
- `docs/siec_ecc_framing_20260427.md` §2.2 G2/G3 (manifold implicit 문제 — 본 문서가 manifold-기하학으로 통합 설명)
- `docs/exp5_postmortem_20260425.md` §5.1 (family-specific behavior — 실험 H의 직접적 동기)
