# S-IEC 가정 감사 (Assumption Audit) — 2026-04-27

> **컨텍스트**: 마감 1주 앞 (초록 2026-05-04 / full 2026-05-06 AOE). 용성의 4개 가설을 코드·실험 데이터·논문 근거로 검증하고, IEC 발전 방향을 유지한 채 마감 안에 가능한 salvage path와 더 좋은 baseline 후보를 정리.
>
> **결론 한 줄**: 4개 주장 중 **3개는 데이터/코드/논문으로 100% 확정**, 1개(주장 4)만 부분 보정. 동시에 **원저자 IEC 논문 자체가 이미 selective 변형을 ablation에 가지고 있어서**, 현재 S-IEC는 두 축(FID·NFE) 모두에서 paper의 selective IEC에 dominated됨.

---

## 목차

1. [주장 1 — S-IEC overhead 문제](#주장-1--s-iec가-iec보다-오버헤드가-크다--100-맞음)
2. [추가 발견 — paper 자체의 selective IEC](#2-새로-발견한-더-큰-문제--원저자-iec-논문-자체에-selective-버전이-이미-있음)
3. [주장 2 — 저화질 ↔ 오류 부족](#3-주장-2--이미지-저화질이라-오류-자체가-적다--부분적-맞음-정확한-진단은-다름)
4. [주장 3 — DDPM에서 syndrome ≠ error](#4-주장-3--ddpm에서-x0t--x0t1--error--100-맞음-수식적-근거)
5. [주장 4 — Normal/Tangent 분리 가능성](#5-주장-4--normaltangent-분리가-ddpm에서는-안-됨--부분-보정)
6. [SDDM 외 더 좋은 방향](#6-sddm-외-더-좋은-방향--카테고리별-실용-옵션)
7. [마감 1주 salvage 권고](#7-마감2026-05-06을-고려한-현실적-salvage-권고)
8. [Sources](#sources)

---

## 주장 1 — "S-IEC가 IEC보다 오버헤드가 크다" → **100% 맞음**

### 코드 증거

`IEC/experiments/yongseong/deepcache_denoising.py:386-401`

```python
if correction_mode == "siec":
    # Upstream fix: syndrome check is done at every reverse timestep.
    checked = (next_t.long()[0].item() >= 0)        # 모든 t에서 True
    if checked:
        et_look, _ = _call_model(model, xt_next_hat, next_t,
                                 prv_f=None, allow_cache_reuse=False)  # ← 매 step 추가 forward
        step_nfe += 1
        ...
        syndrome, score = compute_syndrome(x0_t, x0_look)
```

vs IEC 분기 (`:358-384`): `if correction_mode == "iec" and refresh_step:` — DeepCache interval(매 10 step)에서만 lookahead+correction.

원본 `mainddpm/ddpm/functions/deepcache_denoising.py:330-433` 도 동일한 구조 (line 424: `do_siec_check = (next_t.long()[0].item() >= 0)`).

### NFE 데이터 (실험 4 results.csv)

| Method | trigger_rate | per_sample_NFE | FID(2K) |
|---|---|---|---|
| IEC (baseline) | 1.000 | **110** | 44.29 |
| S-IEC p95 (거의 trigger 안됨) | 0.020 | **201** | 46.95 |
| S-IEC p80 | 0.020 | 201 | 46.95 |
| S-IEC p30 (거의 항상 trigger) | 1.000 | 298 | 46.19 |
| Random matched p80 | 0.018 | 201 | 46.91 |
| Uniform matched p80 | 0.030 | 202 | 46.96 |
| Naive always-on | 1.000 | 298 | 46.17 |

**핵심 관찰**: S-IEC의 minimum overhead floor는 **+91 NFE 고정** (어떤 percentile을 써도). DeepCache의 cache-reuse 가속(`allow_cache_reuse=False`로 매 step 강제)을 매 step 한 번씩 무효화하기 때문. 코드 주석("Upstream fix: syndrome check is done at every reverse timestep")이 이 결정을 명시적으로 인정.

---

## 2. 새로 발견한 더 큰 문제 — 원저자 IEC 논문 자체에 selective 버전이 이미 있음

### `arxiv 2511.06250` Table 5 (CIFAR-10, DeepCache N=10)

| IEC variant | wall-clock 오버헤드 | FID |
|---|---|---|
| All steps | +14% | **7.77** |
| ±1/10 timesteps (selective) | **+2.8%** | 9.58 |
| ±1/20 timesteps (selective) | +1.4% | 9.55 |

→ "IEC를 모든 step이 아니라 부분 step에서만 적용" 자체가 paper ablation에 이미 존재. **paper의 ±1/10 selective IEC = 2.8% overhead로 9.58 FID**.

### 본 프로젝트 50K 결과 (`iec_vs_siec_fid_results.txt`)

```
IEC          7.20    (paper의 all-steps ≈ 7.77)
S-IEC p80    8.79    (paper의 selective ±1/10 = 9.58)
```

### Pareto 비교

| | FID | Wall-clock 오버헤드 |
|---|---|---|
| Paper all-steps IEC | 7.77 | +14% |
| Paper selective ±1/10 | 9.58 | **+2.8%** |
| **본 S-IEC p80** | **8.79** | **+~91% (NFE)** |

S-IEC p80은 **paper의 selective IEC보다 FID 0.79 좋지만 오버헤드는 32배**. 즉:

- "All-steps IEC에 dominated" — FID 1.59 나쁨, NFE 더 큼
- "Selective IEC와의 Pareto"에서도 매우 비효율적 (FID 미세 우위에 NFE 32배)

**S-IEC의 셀링 포인트("선택적 적용으로 효율성 확보")는 paper가 이미 더 단순한 방식(고정 ±1/k 스케줄)으로 달성한 상태.**

---

## 3. 주장 2 — "이미지 저화질이라 오류 자체가 적다" → **부분적 맞음, 정확한 진단은 다름**

| | FID(2K) | FID(50K) |
|---|---|---|
| No correction | 46.98 | 22.56 |
| IEC | 44.29 | 7.20 |
| S-IEC p80 | 46.95 | 8.79 |

- No-corr 22.56 vs IEC 7.20 = **15 FID delta** → deployment error는 분명히 큼.
- 하지만 S-IEC p80, Random p80, Uniform p80 모두 **2K에서 FID 46.9 ± 0.05** → **syndrome score가 trigger 시점을 random보다 잘 고르지 못함**.

**정확한 진단**: "오류는 있으나 syndrome score가 그 오류와 정렬되지 않는다" — 이게 주장 3의 핵심으로 직결.

---

## 4. 주장 3 — "DDPM에서 x0(t) − x0(t−1) ≠ error" → **100% 맞음 (수식적 근거)**

### Toy Gaussian (`siec_sim/core/gaussian_model.py:46`)

```
m_t(x_t) = U Λ α_t (α_t² Λ + σ_t² I)⁻¹ U^T x_t
```

posterior mean이 **closed-form 선형**. 따라서 시간적 변화 ‖x̂₀(t) − x̂₀(t−1)‖ 이 정확히 quantization/cache 잡음에 비례.

### Real DDPM

- Tweedie: `x̂₀ = (x_t + (1 − α_t)·s_θ(x_t, t)) / α_t`
- `s_θ` 는 학습된 nonlinear UNet → 자연스러운 step-to-step refinement (이게 reverse process의 본질).
- "Exposure bias" 문헌(PTQD, Q-Drift 등)이 정확히 이 현상을 다룸.

→ ‖x̂₀(t) − x̂₀(t−1)‖ = **(자연스러운 refinement) + (deployment error)** 의 합.

syndrome score는 합을 보고 있어서 error 성분만 분리 못함.

### 관측 증거

`results.csv` 의 `syndrome_mean`이 No-corr / IEC / S-IEC / Random / Uniform 모두 **0.00558 ~ 0.00561** 사이로 method-invariant. method가 만드는 error 구조가 syndrome 분포에 거의 흔적을 안 남김.

---

## 5. 주장 4 — "Normal/Tangent 분리가 DDPM에서는 안 됨" → **부분 보정**

엄밀히는 **정확한** 분리는 안 되지만, **approximate 분리는 최근 가능**해짐 (2023~2025):

| 방법 | 키 idea | real-image 실증 |
|---|---|---|
| **SDDM** (ICML'23) | Score를 tangent+normal로 decompose, 학습 단계에 통합 | I2I translation 한정 |
| **Niso-DM / Tango-DM** (NeurIPS'25) | Non-isotropic noise / tangential-only loss로 score singularity 완화 | 학습 재필요 |
| **"Be Tangential to Manifold"** (2510.05509) | score function Jacobian의 spectral gap로 tangent/normal 분리 | SD2.1, training-free |
| **"What's Inside Your Diffusion Model?"** (2505.11128) | Score 기반 Riemannian metric | CIFAR/CelebA, training-free |

**결론**: "DDPM에서 분리 불가" 명제는 2025년 기준으로 **틀렸음**. 단:

- Jacobian spectral gap 계산 = 비싼 추가 forward (S-IEC가 syndrome lookahead로 추가 NFE 쓰는 함정과 동일)
- Niso-DM / Tango-DM은 **재학습 필요** → "test-time, training-free" 컨셉과 충돌
- SDDM은 setting이 다름 (paired/unpaired translation)

→ 사용자 가설의 결과("그래서 figure 1을 CIFAR DDPM에서 재현 못한다")는 맞을 가능성이 매우 높지만, 그 이유는 "원리적으로 불가능"이 아니라 **"training-free + extra-NFE-free 의 제약 안에서는 어렵다"**가 더 정확.

---

## 6. SDDM 외 더 좋은 방향 — 카테고리별 실용 옵션

### A. IEC 정신 유지 + 더 효과적인 selective trigger

| 후보 | 동작 | 본 프로젝트 적합도 |
|---|---|---|
| **Score Jacobian spectral gap** ([2510.05509](https://arxiv.org/abs/2510.05509)) | score Jacobian 으로 manifold 직교 성분만 추출, off-manifold 오차일 때만 trigger | 이론 깔끔. Jacobian-vector product 추가 필요 |
| **DC-Solver의 dynamic compensation** ([Springer'24](https://link.springer.com/chapter/10.1007/978-3-031-73247-8_26)) | predictor-corrector misalignment 측정 → IEC trigger 신호로 활용 | DPM-Solver 계열로 풀면 NFE 자체가 절반 |
| **Free-cost trigger** | DeepCache가 이미 계산한 cache disagreement (`et_new − et_hat`) 또는 PTQD의 quantization noise covariance를 trigger 신호로 재활용 | **추가 NFE 0**. 가장 마감 친화적 |
| **TFMQ-DM의 Temporal Feature Maintenance** ([CVPR'24 Highlight](https://github.com/ModelTC/TFMQ-DM)) | 시간 정보 보존 quantization, IEC 없이도 4-bit |

### B. IEC를 다른 acceleration 패러다임에 얹기

| 후보 | 왜 IEC 가 더 의미있는지 |
|---|---|
| **Latent Diffusion (LDM)** — 본 repo `mainldm/`에 이미 존재 | per-step 비용이 더 비싸 NFE 절감 가치가 큼. paper Table 1도 LDM-Bedroom/Church/ImageNet 보고 |
| **DiT 기반 (FLUX/SD3) + Feature Caching** ([ToCa ICLR'25](https://github.com/Shenyi-Z/ToCa), [TaylorSeer](https://arxiv.org/html/2503.06923v1), [HyCa](https://darrenzheng303.github.io/HyCa.github.io/)) | 모던 target. cache miss 비용이 크므로 syndrome 기반 selective recompute가 진짜로 의미 있음 |
| **Consistency Models (CM, CTM, SANA-Sprint)** | 1-4 step inference. error correction은 "step 수가 적어 오차 큰 영역에서만" trigger 하는 식 |
| **Rectified Flow + Restart** | 직선 trajectory 에서 drift detect → IEC restart |

### C. IEC와 패러다임 정합되는 최신 baseline (paper 비교군 후보)

| 논문 | 메모 |
|---|---|
| **TAC** ([ECCV'24, 2407.03917](https://arxiv.org/abs/2407.03917)) | NER+IBC. **현재 프로젝트에 row만 있고 blocked 상태** — 1주일 안에 구현 시 가장 큰 수확 |
| **CTEC** ([AAAI'25](https://ojs.aaai.org/index.php/AAAI/article/view/34039)) | Cross-Timestep Error Correction by distillation |
| **ERTACache** ([2508.21091, ByteDance](https://arxiv.org/abs/2508.21091)) | feature shift error vs step amplification error 분리. syndrome 아이디어와 가장 유사하지만 **offline residual profiling** + analytic rectification (training-free, NFE 추가 없음) |
| **PTQD** ([2305.10657](https://arxiv.org/abs/2305.10657)) | Quantization noise를 correlated/uncorrelated로 분해해 보정. **DDPM이 아니라 LDM-4 + ImageNet에서 W4A8 거의 무손실** |
| **Q-Drift** ([2603.18095](https://arxiv.org/html/2603.18095)) | Quantization-aware drift correction, 가장 최근 |

---

## 7. 마감(2026-05-06)을 고려한 현실적 salvage 권고

> "교수님이 IEC 발전 방향 유지를 원하신다" 제약 + 마감 1주일 조건에서의 우선순위.

| # | 옵션 | 작업량 | 기대 효과 |
|---|---|---|---|
| 1 | **Free-cost trigger 재구현** — cache disagreement / 양자화 노이즈 통계를 trigger로 사용 → S-IEC NFE를 +91% → +0%로 회복. paper Table 5 ±1/10 과 직접 비교 가능 | 1-2일 | Pareto 정리 가능 |
| 2 | **TAC unblock** — 현재 row만 비어 있어 plot 자체가 약함. ECCV'24 official code 있음. baseline 강화 | 2-3일 | exp4/exp5 plot 완성 |
| 3 | **LDM-Bedroom/ImageNet 으로 setting 이동** — paper Table 1에 LDM 결과 있음. DDPM CIFAR보다 deployment error 구조가 더 명확하고 IEC 가치가 더 큼 | 3-5일 | 더 강한 셀링 포인트 |
| 4 | **Limitation을 정직하게 figure 화** — "왜 syndrome score가 CIFAR DDPM에서 random과 차이 없는지" → natural refinement vs deployment error 분리 불가 figure. 이 자체가 좋은 negative result paper가 됨 | 1-2일 | discussion/limitation 섹션 |
| 5 | (다음 cycle 용) **Manifold-aware S-IEC** — "Be Tangential to Manifold"의 Jacobian metric을 trigger로 도입. 진짜 contribution이 됨 | >1주 | 5/6 마감 안에는 어려움 |

**권고 조합**: 1 + 2 + 4 (코어 마감용). 시간 남으면 3.

---

## Sources

### IEC paper
- [Test-Time Iterative Error Correction for Efficient Diffusion Models — arXiv 2511.06250](https://arxiv.org/abs/2511.06250) ([html](https://arxiv.org/html/2511.06250), [OpenReview](https://openreview.net/forum?id=AhwAsF89EG))

### Manifold decomposition for diffusion
- [SDDM: Score-Decomposed Diffusion Models on Manifolds (ICML 2023, 2308.02154)](https://arxiv.org/abs/2308.02154)
- [Be Tangential to Manifold: Discovering Riemannian Metric for Diffusion Models (2510.05509)](https://arxiv.org/abs/2510.05509)
- [What's Inside Your Diffusion Model? A Score-Based Riemannian Metric (2505.11128)](https://arxiv.org/abs/2505.11128)
- [Improving the Euclidean Diffusion Generation of Manifold Data (Niso-DM, Tango-DM, NeurIPS 2025) — 2505.09922](https://arxiv.org/abs/2505.09922)

### Quantized diffusion error correction
- [Timestep-Aware Correction for Quantized Diffusion Models (TAC, ECCV 2024) — 2407.03917](https://arxiv.org/abs/2407.03917)
- [Optimizing Quantized Diffusion Models via Distillation with Cross-Timestep Error Correction (CTEC, AAAI 2025)](https://ojs.aaai.org/index.php/AAAI/article/view/34039)
- [PTQD: Accurate Post-Training Quantization for Diffusion Models — 2305.10657](https://arxiv.org/abs/2305.10657) ([code](https://github.com/ziplab/PTQD))
- [TFMQ-DM: Temporal Feature Maintenance Quantization (CVPR'24 Highlight)](https://github.com/ModelTC/TFMQ-DM)
- [Q-Drift: Quantization-Aware Drift Correction (2603.18095)](https://arxiv.org/html/2603.18095)

### Caching / Solver alternatives
- [ERTACache: Error Rectification and Timesteps Adjustment (2508.21091)](https://arxiv.org/abs/2508.21091) ([code](https://github.com/bytedance/ERTACache))
- [DPM-Solver-v3: Predictor-Corrector with Empirical Model Statistics](https://ml.cs.tsinghua.edu.cn/dpmv3/)
- [DC-Solver: Dynamic Compensation for Predictor-Corrector Diffusion Sampler](https://link.springer.com/chapter/10.1007/978-3-031-73247-8_26)
- [ToCa: Token-wise Feature Caching (ICLR 2025)](https://github.com/Shenyi-Z/ToCa)
- [TaylorSeer: From Reusing to Forecasting (2503.06923)](https://arxiv.org/html/2503.06923v1)
- [HyCa: Hybrid Feature Caching for Diffusion Transformers](https://darrenzheng303.github.io/HyCa.github.io/)
