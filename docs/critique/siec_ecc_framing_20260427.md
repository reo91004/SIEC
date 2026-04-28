---
title: "S-IEC = ECC × Diffusion 가설 검증 및 실험 설계"
date: 2026-04-27
author: 분석 보조 (Claude)
related:
  - docs/siec_critique_20260427.md
  - IEC/siec_core/syndrome.py
  - IEC/siec_core/correction.py
  - IEC/experiments/yongseong/deepcache_denoising.py
  - siec_sim/core/siec.py
  - siec_sim/core/gaussian_model.py
---

# S-IEC = ECC × Diffusion 가설 검증

> 한 줄 요약. 교수님의 toy 코드(`IEC/siec_core/`)와 toy sim(`siec_sim/`)을 같이 읽으면, S-IEC의 진짜 주장은 "selective IEC"가 아니라 **"오류정정 부호화(ECC) 이론을 diffusion에 결합해 더 좋은 quality/NFE를 달성한다"** 쪽이 맞다. 다만 현재 CIFAR 구현은 그 주장을 절반밖에 못 살리고 있다.

이 문서는 다음 세 가지를 다룬다.

1. S-IEC의 진짜 주장 재구성 — ECC × Diffusion mapping
2. 그 주장이 toy/CIFAR 각각에서 얼마나 입증되어 있는지
3. 그 주장을 ICLR 2026 마감 안에 입증하기 위한 실험 6개와 우선순위

같은 날 작성된 `docs/siec_critique_20260427.md`는 **현재 결과 기준의 비판/대안**을 다루고, 본 문서는 **이론적 framing 기준의 검증/실험 설계**를 다룬다. 둘은 상호보완이다.

---

## 1. S-IEC의 진짜 주장 재구성

### 1.1 ECC × Diffusion mapping

`IEC/siec_core/syndrome.py`, `IEC/siec_core/correction.py`, 그리고 `siec_sim/core/`를 같이 보면, S-IEC가 노리는 건 단순한 "더 자주 보정하는 IEC"가 아니라 아래 mapping을 통한 통합이다.

| ECC 개념 | Diffusion 대응물 | toy/CIFAR 위치 |
|---|---|---|
| 코드워드 (codeword space) | 데이터 다양체 / posterior mean의 일관 궤적 | `siec_sim/core/gaussian_model.py:posterior_mean` |
| 채널 (noisy channel) | 배포 환경 sampler (DDIM + DeepCache + PTQ + …) | `IEC/experiments/yongseong/deepcache_denoising.py` |
| 패리티 검사 (parity check) | posterior 일관성: 인접 step의 $\hat{x}_0$ 가 동일 궤적에 있어야 한다 | `siec_core/syndrome.py` |
| 신드롬 (syndrome) | $s_t = \hat{x}_0(t) - \hat{x}_0(t-1)$ 의 norm/score | `compute_syndrome()` |
| 디코더 (decoder) | consensus correction: $x_t \leftarrow (1-\gamma_t)\,x_t^{\text{tent}} + \gamma_t\,x_t^{\text{look}}$ | `correction.py:compute_gamma` |
| 코딩 이득 (coding gain) | $\gamma_t = \lambda_t/(1+\lambda_t)$, $\lambda_t = c\sigma_t^2/(\alpha_t^2+\sigma_t^2)$ | `correction.py` |

이 mapping이 의도대로 동작하면, "deployment에서 발생한 오류"는 codeword 바깥으로 튄 신호로 해석되고, 신드롬이 그 이탈을 측정하며, 디코더는 codeword 위로 사영하는 역할을 한다. **즉 S-IEC = "diffusion sampler 위에 얹는 syndrome decoder."**

### 1.2 prior art와의 차이

이 framing 자체는 신기하지만 완전한 신규는 아니다. 인접한 두 흐름이 있다.

- **DDECC** (`arXiv:2209.13533`): diffusion으로 ECC decoding을 수행. **방향이 반대**다 (S-IEC는 ECC로 diffusion을 도와주는 쪽).
- **Random Walks with Tweedie** (`arXiv:2411.18702`): Tweedie 추정량의 martingale 성질을 이론적으로 정리. S-IEC의 "$\hat{x}_0(t)$ 가 step 변화에 안정해야 한다"는 가정에 정확히 대응하는 이론적 기반. 즉 toy 가정은 이미 알려진 성질에 올라타 있다.

따라서 S-IEC의 차별점은 "ECC framing 자체"가 아니라 **"deployment 환경에서 발생한 비-Gaussian, non-stationary 오차에 대해 syndrome 기반 디코더가 실제로 작동함을 보이는 것"**에 있다. 이게 실험으로 입증되어야 ICLR 2026의 contribution이 살아난다.

### 1.3 toy와 CIFAR이 같은 주장을 하고 있는가

같은 주장을 *목표*로는 한다. 다만 toy에서는 이 framing이 **closed form으로 깔끔히 떨어지고**, CIFAR에서는 **여러 가정을 추가로 빌리는** 구조다.

- toy (`siec_sim/`):
  - 가우시안 + 저차원 PCA. posterior mean이 닫힌 형태로 풀린다.
  - syndrome ↔ tangent/normal 분해가 정확히 일치한다.
  - $\gamma_t$, $\lambda_t$ 가 closed form. 디코더 이득을 수학적으로 증명할 수 있다.
- CIFAR (`IEC/experiments/yongseong/deepcache_denoising.py`):
  - posterior mean을 학습된 UNet으로 근사 (Tweedie).
  - 다양체가 명시적이지 않다.
  - syndrome이 normal 성분만이 아니라 **UNet bias + timestep 이산화 + stochasticity**를 모두 담는다.
  - $\gamma_t$ 는 toy와 같은 식을 그대로 쓰지만, 그 식의 도출 가정은 CIFAR에서는 약하다.

즉 toy에서의 깔끔한 ECC mapping을 CIFAR에서 그대로 쓰면 **G1–G4 (다음 절)의 4가지 갭**이 생긴다.

---

## 2. 입증 정도 검증

### 2.1 toy에서 입증된 부분 (강함)

- `gaussian_model.py:posterior_mean` 에서 closed-form posterior mean 구현. 코드 자체가 ECC 디코더의 정의식이다.
- `compute_gamma()` 의 $\gamma_t = \lambda_t/(1+\lambda_t)$ 는 가우시안 가정에서 MAP 디코딩 가중치로 정확히 유도된다.
- toy에서 syndrome score는 deployment error 강도와 단조 증가. **mapping이 toy 안에서는 살아 있다.**
- `siec_sim/core/siec.py:97-99` 의 주석 `# net 1 NFE (lookahead reused)` — 알고리즘 의도상 추가 NFE는 0 또는 매우 작다.

### 2.2 CIFAR DDPM에서 어긋나는 4가지 갭

#### G1. Martingale 성질 (Tweedie 일관성)

- **가정**: $\mathbb{E}[\hat{x}_0(t-1)\,|\,\hat{x}_0(t)] = \hat{x}_0(t)$.
- **현실**: 학습된 UNet은 마진 적합/유한 시간 이산화 때문에 이 등식을 정확히 만족시키지 않는다 (`arXiv:2411.18702` 논의).
- **영향**: $s_t = \hat{x}_0(t) - \hat{x}_0(t-1)$ 의 0이 아닌 평균이 **모델 자체에서** 나오므로, 신드롬은 deployment error만의 함수가 아니다.

#### G2. 다양체가 명시적이지 않다

- **가정**: codeword space = data manifold. syndrome = manifold 외 성분.
- **현실**: CIFAR DDPM의 manifold는 명시적 형태가 없다. tangent/normal 분해가 정의되지 않는다.
- **영향**: 디코더가 normal 성분을 줄이는 게 아니라 **모든 차원의 score-weighted blending**을 한다. 본래 ECC framing의 "사영" 의미가 흐려진다.

#### G3. in-manifold error도 일어난다

- **가정**: deployment error는 manifold 바깥으로 튄다.
- **현실**: PTQ/DeepCache의 오차는 manifold "위에서도" 클래스/모드를 살짝 옮기는 형태로 나타난다 (e.g., 4-bit 양자화에서 색감 shift, cache 재활용에서 디테일 단순화).
- **영향**: 이 종류의 오차에서는 syndrome이 작아도 **품질은 떨어진다**. 신드롬-품질 상관성이 깨진다. exp5의 `dc10/dc20` 결과가 정확히 이 상황과 일치한다 (`docs/exp5_postmortem_20260425.md` 5.3 절).

#### G4. lookahead 도메인 시프트

- **가정**: lookahead step에서 산출된 $\hat{x}_0$ 는 같은 데이터 분포의 신뢰 가능한 두 번째 추정.
- **현실**: lookahead step은 cache 미사용/추가 forward pass — **실제 배포 sampler와 다른 통계**를 낼 수 있다 (e.g., `allow_cache_reuse=False` 면 cache가 깨진 상태의 출력).
- **영향**: 두 추정이 같은 채널의 두 관측이 아니라 **다른 채널의 관측**일 수 있고, 이 경우 consensus는 평균 내려는 두 분포가 어긋나서 오히려 품질을 해칠 수 있다.

### 2.3 결정적 발견 — 알고리즘 결함이 아니라 구현 결함

`siec_sim/core/siec.py:97-99`:

```python
if not triggered:
    return xt_tent, x0_current, syn_score, False, 1  # net 1 NFE (lookahead reused)
```

→ toy는 lookahead 결과를 **재활용**하도록 의도되어 있다. 즉 trigger되지 않으면 추가 NFE는 사실상 0.

`IEC/experiments/yongseong/deepcache_denoising.py:386-401`:

```python
if correction_mode == "siec":
    checked = (next_t.long()[0].item() >= 0)
    if checked:
        et_look, _ = _call_model(model, xt_next_hat, next_t,
                                 prv_f=None, allow_cache_reuse=False)
        step_nfe += 1
```

→ CIFAR에서는 **매 step lookahead 재계산**, cache reuse 차단. 결과적으로 NFE가 IEC 110 → S-IEC p80 ≈ 201 (`+91%`) 로 폭증.

**결론**: 현재 CIFAR S-IEC의 +91% NFE는 ECC framing의 본질적 비용이 아니라 **lookahead 재활용을 끄고 매 step 재계산하기 때문**에 발생한다. 이 한 줄이 framing의 가장 큰 약점을 만들고 있다 — 같은 비용으로 "그냥 IEC를 더 자주 호출"하는 것과 비교 우위가 사라지기 때문.

---

## 3. 입증을 위한 실험 6종 (A–F)

각 실험은 ECC framing의 어느 부분을 살리거나 부수는지를 명확하게 한다.

### 실험 A. 신드롬-오차 상관성 (correlation test)

- **목적**: G1, G3 점검. syndrome score $\|s_t\|_2$ 가 실제 per-step deployment error와 단조 상관인가.
- **세팅**: `fp16` 기준 trajectory에 대해, 각 step에서
  - $\|s_t\|_2$ (syndrome score)
  - $\|x_t^{\text{deploy}} - x_t^{\text{ref}}\|_2$ (deployment 오차의 ground truth)
  - 두 값을 step별로 기록.
- **분석**: Spearman $\rho$, scatter plot, 그리고 family별 (`w8a8 / w4a8 / dc10 / dc20 / cachequant`) 분리.
- **판정**:
  - $\rho > 0.7$: ECC framing 작동. syndrome이 실제 오차의 proxy.
  - $\rho < 0.3$: framing 위협. exp5의 `dc10/dc20` 비일관 결과를 설명할 수 있는 직접 증거.

### 실험 B. Innovation 분해 (martingale 검증)

- **목적**: G1 직접 측정.
- **세팅**: `fp16` 깨끗한 sampler에서 $\hat{x}_0(t)$ trajectory를 모은다. 이로부터
  - $m_t = \hat{x}_0(t) - \hat{x}_0(t-1)$ (innovation)
  - $\bar{m} = \mathbb{E}_t[m_t]$, $\text{Var}_t(m_t)$ 계산.
- **판정**: Tweedie 일관성 가정 하에 $\bar{m} \approx 0$, $\text{Var}(m_t) \propto \sigma_t^2 - \sigma_{t-1}^2$ 비례성이 보여야 한다. 안 보이면 G1이 강한 가정 위반이다.

### 실험 C. CIFAR Figure 1 재현 — 신드롬-품질 곡선

- **목적**: 단일 그림으로 framing의 설득력 확보.
- **세팅**: 6 setting × 100 step에서 step별 $\|s_t\|_2$ 평균과 final FID를 산점도로 그린다. family별 색상 구분.
- **판정**: 
  - 단조 우상향: ICLR Figure 1 후보.
  - family별 분리된 곡선: G3 증거이지만, 동시에 family-specific decoder를 정당화하는 발판이 된다.

### 실험 D. lookahead 재활용 fix (구현 결함 제거)

- **목적**: G4 완화 + 비용 비교 공정성 확보. **가장 즉시 임팩트 큰 실험.**
- **세팅**: `deepcache_denoising.py:386-401` 경로에서, lookahead 호출 시 가능한 한 cache feature를 재활용하고 다음 step에서 `prv_f` 로 회수. toy의 "net 1 NFE" 의도를 CIFAR에 옮긴다.
- **검증**: 같은 trigger rate에서 NFE가 IEC 110 ± 5 수준으로 떨어지면 fix 성공.
- **판정**: NFE가 IEC 수준 ± 10% 안으로 들어오면 framing 비교가 의미 있어진다. 들어오지 않으면 알고리즘 자체에 lookahead 본질 비용이 있다는 뜻이고, 그 경우 framing을 "더 비싼 보정"으로 다시 포지셔닝해야 한다.

### 실험 E. Oracle decoder bound

- **목적**: framing의 **이론 상한** 측정.
- **세팅**: 각 step에서 ground-truth $x_t^{\text{ref}}$ 를 알 수 있다고 가정 (fp16 동일 seed trajectory를 reference로). 디코더가 매 step마다 $x_t \leftarrow x_t^{\text{ref}}$ 로 끌어당기는 경우의 FID를 측정.
- **판정**: 이 oracle FID가 fp16 FID에 충분히 가까우면, "syndrome으로 oracle을 잘 흉내내면 끝나는 문제"로 정의된다. 멀면 framing 자체의 상한이 낮다는 신호.

### 실험 F. Cross-error robustness

- **목적**: G3 일반화 점검. 한 family에서 캘리브레이션한 decoder가 다른 family에서 살아남는가.
- **세팅**: `w8a8` 에서 캘리브레이션한 $\tau$, $\gamma$ 를 그대로 `dc10`, `cachequant` 에 적용. 반대 방향도.
- **판정**: family-cross 적용에서도 No-correction 대비 개선이 유지되면 universal decoder. 안 되면 **family-conditional decoder**가 본 framing의 자연스러운 일반화이다 (논문 contribution을 "selective family-aware decoder"로 다시 잡을 근거).

### 3.1 일주일 우선순위

ICLR 2026 마감 (abstract 2026-05-04) 까지 약 1주이므로, 비용 대비 결과 기준으로 다음 순서를 권장한다.

| 순위 | 실험 | 이유 | 예상 소요 |
|---|---|---|---|
| 1 | D (lookahead 재활용 fix) | 한 줄 변경으로 비용 비교의 공정성 회복. NFE 폭증 문제가 알고리즘인지 구현인지가 결판난다. | 0.5 일 |
| 2 | A (correlation test) | 가장 작은 분석 비용으로 framing의 핵심 가정 검증. exp5 비일관 원인 직접 진단. | 0.5 일 |
| 3 | B (innovation baseline) | A의 결과 해석에 필요한 reference. fp16 한 번이면 충분. | 0.5 일 |
| 4 | C (Figure 1 재현) | 위 셋이 살아 있을 때 한 그림으로 정리. ICLR Figure 1 후보. | 1 일 |
| 5 | F (cross-error) | family-conditional decoder 정당화 근거. paper contribution 재정의에 필요. | 1 일 |
| 6 | E (oracle bound) | 이론 상한 — 시간 남으면 수행. discussion 섹션 보강. | 1 일 |

---

## 4. 결론

- S-IEC의 진짜 주장은 "selective IEC"가 아니라 **ECC × Diffusion**으로 보는 게 더 정확하다.
- 그 framing은 toy에서는 closed form으로 깔끔히 살아 있고, CIFAR에서는 G1–G4 4갭을 추가로 빌린다.
- **현재 CIFAR S-IEC가 IEC 대비 NFE 폭증을 겪는 핵심 원인은 알고리즘이 아니라 lookahead 재활용을 끈 구현(`deepcache_denoising.py:386-401`)이다.** 이 한 줄을 toy 의도(`siec_sim/core/siec.py:99`)대로 되돌리는 게 가장 큰 즉시 이득이다.
- 실험 D → A → B → C 순서로 1주 안에 framing의 작동 여부를 이진 판정할 수 있다. 결과에 따라 논문 contribution 문구를 "syndrome-guided decoder for deployment-error-induced sampling drift"로 다시 잡는 게 자연스럽다.
- 실험 결과가 가설을 살리지 못하면, `docs/siec_critique_20260427.md` 의 대안 (TAC / CTEC / ERTACache / PTQD 방향) 으로 빠르게 피벗하는 게 합리적이다.
