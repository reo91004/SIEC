# S-IEC 비판문의 코드 단위 검증 (2026-04-28)

`docs/siec_critique_20260427.md`, `siec_ecc_framing_20260427.md`, `siec_manifold_geometry_20260427.md`, `syndrome_notes_20260427.md`에서 제기된 ECC framing 비판이 실제 코드와 정합한지 1:1로 대조한 결과.

요약: **대부분 정합한다. 한 군데(NFE quick win)만 비판이 다소 과장되어 있고, 두 군데(deployment-conditional calibration / iterative correction dead code)는 코드 증거가 비판문보다 더 강하다.**

---

## 1. Syndrome 정의 — 비판 정합

**주장**: `ŝ_t = x̂_0(t) - x̂_0(t-1)`, `r_t = ||ŝ||²/d` 는 `H·r` 형태의 algebraic parity check가 아니다.

**코드**:
- `IEC/siec_core/syndrome.py:14-17`
  ```python
  syndrome = x0_current - x0_lookahead   # H 없음
  score = (syndrome ** 2).flatten(1).sum(dim=1) / d
  ```
- `IEC/mainddpm/ddpm/functions/deepcache_denoising.py:435-438`
  ```python
  x0_look = (xt_next_tent - et_look * (1 - at_next).sqrt()) / at_next.sqrt()
  syndrome, score = compute_syndrome(x0_t, x0_look)
  ```

**결론**: parity check matrix `H`는 어느 모듈에도 존재하지 않는다. 연속 Tweedie 차이만 사용한다.

---

## 2. γ 보정 공식 — 비판 정합

**주장**: `γ_t = λ_t/(1+λ_t)`, `λ_t = c·σ²/(α²+σ²)` 는 toy Gaussian에서 유도된 식이며, CIFAR에 그대로 transferred.

**코드**:
- `IEC/siec_core/correction.py:19-21`
  ```python
  lam = c * sigma_t_sq / (alpha_t_sq + sigma_t_sq)
  gamma = lam / (1.0 + lam)
  return x0_current - gamma * syndrome
  ```
- `siec_sim/core/siec.py:40-49` (toy) — 정확히 같은 식.
- `IEC/mainddpm/ddpm/functions/deepcache_denoising.py:456-458` — CIFAR에서도 그대로 사용.

**결론**: MAP decoder를 실제 PTQ/DeepCache channel model에서 다시 유도하는 절차는 코드에 없다.

---

## 3. NFE 회계 — 비판 거의 정합 (수치만 약간 보정)

**주장**: IEC ~110 NFE, S-IEC ~201 NFE.

**코드**:
- `mainddpm/ddpm/functions/deepcache_denoising.py:388-433`
  - L394/397: 메인 forward — `step_nfe += 1`
  - L430: lookahead forward (`prv_f=None`로 cache reuse 강제 차단) — `step_nfe += 1`
- 100 step × 2 = **약 200 NFE** (마지막 step은 `next_t=-1`로 lookahead 생략 가능).
- IEC의 110 NFE는 `adaptive_generalized_steps_3:218`의 `max_iter=2` 분기가 interval_seq에서만 발동되어 +10이 붙는 구조.

**결론**: 정확한 production 수치는 200, "201"은 +1 정도 오차.

### 비판이 다소 과장된 부분

비판문은 "Lookahead reuse 미구현"을 마감 1주의 quick win으로 제안했다. production 코드 기준으로는 맞다 (`prv_f=None`을 명시적으로 넘기고, 다음 step의 메인 forward는 캐시 슬롯과 무관하게 새로 계산).

그러나 **`IEC/experiments/yongseong/deepcache_denoising.py`에 `reuse_lookahead` 플래그와 `lookahead_memo` 메모이제이션이 이미 구현되어 있다**:
- L288: `reuse_lookahead=False` 파라미터
- L354-364: `lookahead_memo`로 첫 forward skip
- L437-446: lookahead가 cache feature를 reuse하도록
- L506-516: `triggered=False`인 step에서 다음 step용 memo 저장

`[EXP-FRAMING-D]` 태그로 실험적이지만, "1주일 내 가능한 quick win"은 새로 짜는 일이 아니라 **이 플래그로 FID를 다시 돌리는 일**이다.

---

## 4. Normal/Tangent 라벨링 — 비판이 더 강해질 수 있음

**주장**: toy의 "Normal/Tangent" 구분이 ECC 의미와 다르다.

**코드**:
- `siec_sim/core/gaussian_model.py:74-83`
  ```python
  def project_normal(self, v):
      """P^⊥ v = U (U^T v).  Normal space = span(U)."""
  def project_tangent(self, v):
      """P^∥ v = v - P^⊥ v.  Tangent space = span(U)^⊥."""
  ```
- `siec_sim/utils/geometry.py:8` 도 `Normal space = span(U), Tangent space = span(U)^⊥`라고 명시.

`docs/syndrome_notes_20260427.md:67`이 자인하듯 **toy 코드의 라벨링은 표준 미분기하학과 정반대**:
- toy: "normal" = span(U) = 데이터 방향 / "tangent" = span(U)^⊥ = 직교 방향
- 표준: "tangent" = span(U) (manifold 위) / "normal" = span(U)^⊥ (manifold 밖)

따라서:
- toy의 syndrome은 toy가 부르는 "normal" (= 데이터 방향) 오류를 본다.
- 표준 ECC syndrome은 codeword space "밖"의 오류를 봐야 한다.
- **toy의 syndrome은 라벨이 뒤집혀서 "ECC답게 작동하는 것처럼 보이는" 착시**가 있다.

CIFAR 코드(`mainddpm/ddpm/functions/deepcache_denoising.py`)에는 `project_*` 호출이 전혀 없다. SVD/normal bundle 추정도 없다 (`grep -rn "svd\|SVD\|P_perp"` 결과 0건). **CIFAR S-IEC는 어떤 manifold geometry도 활용하지 않는다.**

---

## 5. Calibration: 가장 결정적인 코드 증거 (비판 강화)

**비판문의 "Channel model이 없음"보다 더 심각한 사실이 코드에 있다.**

`IEC/experiments/real_05_robustness.py:130-214` deployment 정의:
```python
"fp16":       dict(enable_ptq=False, enable_cache_reuse=False),  # reference
"w8a8":       dict(enable_ptq=True,  enable_cache_reuse=False),
"dc10":       dict(enable_ptq=False, enable_cache_reuse=True),
"w4a8":       dict(enable_ptq=True,  enable_cache_reuse=False),
"dc20":       dict(enable_ptq=False, enable_cache_reuse=True),
"cachequant": dict(enable_ptq=True,  enable_cache_reuse=True),
```

`pilot_scores_path(label)`은 `pilot_scores_w8a8.pt`, `pilot_scores_dc10.pt` 등 deployment별로 저장된다. 실제 저장된 pilot scores를 확인했다:

| pilot 파일 | t=0 mean | t=25 mean | t=98 mean |
|---|---|---|---|
| `pilot_scores_w8a8.pt` | **0.110** | 1.72e-3 | 2.1e-20 |
| `pilot_scores_dc10.pt` | 3.19e-4 | 2.82e-4 | 4.2e-12 |
| `pilot_scores_cachequant.pt` | **0.129** | 2.17e-3 | 5.4e-20 |

**`fp16` (clean reference)에 대한 pilot score 파일은 존재하지 않는다.**

함의:
- ECC는 `H · clean_codeword = 0`을 calibration의 기준으로 잡는다.
- S-IEC tau는 clean(fp16) 분포가 아니라 **deployed sampler 자체의 분포**를 기준으로 잡는다.
- 즉 tau는 "이 sample의 syndrome이 W8A8 샘플들 중에서도 유난히 큰가"를 묻는 **per-deployment outlier 검출기**이다.

**이것은 ECC syndrome이 아니라 conditional anomaly detector이다.** 비판문에서 "ECC 프레이밍이 post-hoc rationalization"이라 한 것의 가장 강한 코드 단위 증거.

---

## 6. Clean syndrome ≠ 0 — 비판 정합 (수치 추가)

저장된 tau 값으로 직접 검증:

```
tau_schedule_p80.pt:        shape=(99,), mean=5.85e-3, max=1.15e-1, %nonzero=98%
tau_schedule_w8a8_p80.pt:   max=0.114, mean=5.74e-3
tau_schedule_dc10_p80.pt:   max=1.48e-3, mean=1.68e-4
tau_schedule_cachequant_p80.pt: max=0.134, mean=6.82e-3
```

w8a8 mean이 dc10 mean의 **34배**. linearity (`H·(c+e) = H·e`)가 성립한다면 baseline이 비슷해야 하지만, **deployment마다 baseline이 두 자릿수 이상 다르다.** syndrome이 단일 채널 오류를 측정하는 것이 아니라, **각 deployment의 forward dynamics + Tweedie 차이가 합쳐진 양**임을 시사한다.

---

## 7. Iterative correction은 dead code — 추가 발견

`mainddpm/ddpm/functions/deepcache_denoising.py:460-488`:
```python
for _round in range(siec_max_rounds):
    x0_corrected = apply_consensus_correction(...)
    et_corrected = (xt - at.sqrt() * x0_corrected) / (1 - at).sqrt()
    xt_next_hat = at_next.sqrt() * x0_corrected + c1 * noise + c2 * et_corrected

    if _round < siec_max_rounds - 1:
        # ... re-evaluate lookahead, recompute syndrome ...
```

`ddim_cifar_siec.py:117`의 default가 `siec_max_rounds=1`이라 **inner re-evaluation 분기는 production에서 한 번도 실행되지 않는다**. paper의 "Algorithm 1 inner loop" 인상과 실제 동작이 다르다.

---

## 종합 매핑 표

| 비판 항목 | 코드 증거 | 평가 |
|---|---|---|
| (1) syndrome은 `H·r`이 아닌 연속 Tweedie 차이 | `siec_core/syndrome.py:14` | 정합 |
| (2) γ는 toy Gaussian 식 그대로 transferred | `siec_core/correction.py:19`, `deepcache_denoising.py:458` | 정합 |
| (3) S-IEC ~200 NFE / IEC ~110 NFE | `deepcache_denoising.py:400, 433` | 정합 (수치 200, 201 아님) |
| (4) Lookahead reuse 미구현 | production: 미구현(L430) / yongseong: 이미 구현됨(L288, 354) | production만 정합 — yongseong에 fix 있음 |
| (5) toy=span(U) projection / CIFAR=no projection | `gaussian_model.py:60` vs CIFAR 어디에도 P_⊥ 없음 | 정합 |
| (6) Normal/Tangent 라벨링 비표준 | `gaussian_model.py:74-83`, `syndrome_notes:67` | 정합 (강화 가능) |
| (7) clean syndrome ≠ 0 | `tau_schedule_p80.pt mean=5.85e-3` | 정합 |
| (8) channel model 부재 | `calibration/pilot_scores_{label}.pt` 패턴 | 더 강함: pilot이 deployment 자체로 돌아감, fp16 pilot 없음 |
| (9) iterative correction은 dead code | `siec_max_rounds=1` default | 추가 발견 |

---

## 결론

비판문의 큰 줄기는 코드에서 정확히 확인된다. 가장 강한 단일 증거는 **calibration이 deployment-conditional anomaly detection이지 ECC syndrome이 아니라는 점** (`pilot_scores_{label}.pt` 패턴 + fp16 pilot 부재).

NFE overhead 해결에 대해서는 비판문이 다소 과장되어 있다 — **`IEC/experiments/yongseong/deepcache_denoising.py`에 `reuse_lookahead` fix가 이미 존재**하므로, "마감 1주의 quick win"은 새로 짜는 일이 아니라 그 플래그로 FID를 다시 돌리는 일이다.
