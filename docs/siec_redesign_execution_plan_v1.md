# S-IEC Redesign Execution Plan v1

작성일: 2026-04-29 KST

이 문서는 `S-IEC/docs/siec_redesign_report_v0.md`의 정합성 검토 결과를 실제 구현과 실험으로 옮기기 위한 실행 계획이다. 목표는 다른 연구자나 LLM 에이전트가 이 문서만 보고도 같은 방향으로 작업을 이어갈 수 있게 만드는 것이다.

## 0. 최종 목표

최종적으로 주장해야 할 형태는 다음이다.

> Diffusion sampling produces a temporally redundant trajectory of clean estimates and U-Net features. Efficient deployment methods such as DeepCache, PTQ, and CacheQuant corrupt this trajectory. S-IEC treats this trajectory as an analog codeword and uses calibrated trajectory syndromes to allocate IEC correction budget to unreliable timesteps.

한국어로는 다음처럼 둔다.

**Diffusion의 `x0_hat`/feature trajectory를 중복성을 가진 analog codeword로 보고, DeepCache/PTQ/CacheQuant를 noisy channel로 해석한다. S-IEC는 calibrated trajectory syndrome으로 timestep reliability를 추정하고 IEC correction budget을 더 필요한 곳에 배분한다.**

## 1. 반드시 지킬 원칙

1. `IEC/` 원본은 수정하지 않는다. 모든 구현은 `S-IEC/` 안에서만 수행한다.
2. `data manifold point = codeword` 주장은 버린다.
3. `trajectory = analog codeword`를 기본 프레임으로 둔다.
4. `linear learned parity is sufficient`라고 쓰지 않는다. linear는 ablation 후보일 뿐이다.
5. `rollback`이라는 단어는 조심해서 쓴다. DDIM sampling은 autoregressive이므로 실제 구현은 대부분 `rollback to x_t`가 아니라 `discard speculative lookahead and recompute/correct x_{t-1}`이다.
6. 모든 비교는 NFE-matched여야 한다. NFE는 공식 추정이 아니라 trace의 `nfe_per_step` 합으로 계산한다.
7. calibration seed와 evaluation seed를 분리한다. 같은 샘플로 threshold/statistics를 만들고 성능을 보고하면 안 된다.
8. 실험 결과가 없으면 논문 claim은 이론적 정합성 수준으로만 쓴다.

## 2. 현재 코드 상태

### 이미 있는 것

- S-IEC CLI 옵션:
  - `S-IEC/mainddpm/ddim_cifar_siec.py`
  - `--siec_score_mode {raw,mean,calibrated}`
  - `--siec_stats_path`
  - `--reuse_lookahead`
  - `--trigger_mode {syndrome,random,uniform}`
- calibrated score 계산:
  - `S-IEC/siec_core/syndrome.py`
  - `compute_syndrome(..., score_mode="raw|mean|calibrated", stats=...)`
  - `mu` 보정과 `std/var/q_inv_sqrt` 기반 whitening 지원
- speculative lookahead memo의 일부:
  - `S-IEC/mainddpm/ddpm/functions/deepcache_denoising.py`
  - `lookahead_memo`로 non-triggered lookahead의 `et`, `x0` 재사용
- random/uniform trigger baseline:
  - `trigger_mode=random|uniform`
- trace output:
  - `adaptive_generalized_steps_trace`
  - `syndrome_per_step`, `triggered_per_step`, `nfe_per_step`

### 아직 없는 것

- clean/full-compute trajectory stats를 만드는 공식 파이프라인
- learned parity `A_t, b_t` 또는 diagonal/low-rank parity 구현
- `Delta_t` teacher syndrome 기록
- stepwise IEC gain `g_t` 측정
- true multi-level reliability controller
  - 현재는 threshold gate + consensus correction에 가깝다.
- feature-level syndrome
- cache feature state까지 포함하는 완전한 speculative commit/rollback
  - 현재 memo는 `et`, `x0`만 저장한다.
  - DeepCache quick path는 `prv_f`가 핵심이므로, lookahead의 feature state도 일관되게 다뤄야 한다.

## 3. 구현 Milestone

작업은 M0부터 M6까지 순서대로 진행한다. M0-M2가 없으면 이후 실험은 신뢰할 수 없다.

## M0. 용어와 trace/NFE 정리

### 목표

기존 코드와 실험 wrapper가 같은 의미로 NFE, trigger, correction을 기록하게 만든다.

### 작업

1. `S-IEC/mainddpm/ddpm/functions/deepcache_denoising.py`의 trace에 다음 필드를 추가한다.
   - `step_idx`
   - `t_int`
   - `next_t_int`
   - `refresh_step`
   - `memo_hit`
   - `checked`
   - `triggered`
   - `nfe_per_step`
   - `score_mean`
   - `score_values`
   - `correction_mode`
   - `syndrome_score_mode`
2. 모든 실험 wrapper가 `per_sample_nfe = sum(trace["nfe_per_step"])`로만 NFE를 계산하게 한다.
3. `No correction`은 S-IEC sampler에서 tau를 무한대로 둔 가짜 no-correction이 아니라, correction과 lookahead check가 모두 꺼진 deployed sampler여야 한다.
4. `mode="none"`은 진짜 no-correction path로 정의한다.
   - main denoising forward만 수행
   - syndrome lookahead 없음
   - correction 없음
   - trace만 기록

### 수정 후보 파일

- `S-IEC/mainddpm/ddpm/functions/deepcache_denoising.py`
- `S-IEC/mainddpm/ddpm/runners/deepcache.py`
- `S-IEC/experiments/yongseong/real_04_tradeoff.py`
- `S-IEC/experiments/yongseong/real_05_robustness.py`

### 완료 기준

- `No correction`의 per-sample NFE가 100-step sampling이면 약 100이다.
- `S-IEC`의 NFE가 `100 + checked_steps + extra_recompute_steps`로 trace에서 직접 확인된다.
- 같은 row를 두 번 실행했을 때 trace의 `nfe_per_step` 합과 CSV의 `per_sample_nfe`가 일치한다.

## M1. Clean trajectory stats calibration

### 목표

raw syndrome에서 clean natural drift를 제거한다.

현재 raw score는 다음을 섞어 본다.

```text
natural clean drift
+ model bias
+ discretization gap
+ deployment error
```

S-IEC가 필요한 것은 deployment error 쪽 신호이므로, clean/full-compute trajectory의 drift distribution을 추정해야 한다.

### 정의

calibration residual은 다음으로 통일한다.

```text
drift_t = x0_lookahead(t-1) - x0_current(t)
```

stats는 timestep별로 저장한다.

```text
mu[t]  = E_clean[drift_t]
var[t] = Var_clean[drift_t]       # diagonal variance
std[t] = sqrt(var[t] + eps)
```

full covariance는 CIFAR에서도 비용이 크므로 v1에서는 diagonal whitening을 기본으로 한다. 문서와 논문에는 `diagonal Mahalanobis` 또는 `diagonal-whitened residual`이라고 쓴다. full Mahalanobis라고 쓰면 안 된다.

### 작업

1. 새 스크립트를 만든다.
   - 권장 경로: `S-IEC/mainddpm/calibrate_syndrome_stats_cifar.py`
2. 이 스크립트는 clean/reference sampler로 calibration samples를 돌린다.
   - PTQ off
   - correction off
   - 가능하면 full-compute path
   - DeepCache deployment를 대상으로 할 때도 clean stats는 reference channel에서 먼저 만든다.
3. 각 step에서 `x0_current`, `x0_lookahead`를 얻어 `drift_t`를 누적한다.
4. 저장 포맷은 `torch.save(dict(...))`로 한다.

권장 저장 포맷:

```python
{
    "version": 1,
    "kind": "clean_trajectory_drift_stats",
    "score_space": "x0",
    "residual_definition": "x0_lookahead_minus_x0_current",
    "num_steps": 99,
    "num_samples": N,
    "mu": Tensor[T, C, H, W],
    "var": Tensor[T, C, H, W],
    "std": Tensor[T, C, H, W],
    "eps": 1e-6,
    "config": {...}
}
```

5. `S-IEC/siec_core/syndrome.py`는 이미 `mu`, `var`, `std`, `q_inv_sqrt`를 받을 수 있다. stats key 이름은 이 파일과 호환되게 유지한다.
6. calibration/evaluation split을 파일명에 명시한다.
   - 예: `calibration/syndrome_stats_clean_seed0_999_n2048.pt`
   - 예: `results/.../eval_seed1000_2999/`

### 주의점

- deployment별 pilot score로 threshold를 잡는 것은 anomaly detector baseline이다.
- S-IEC claim에는 clean/reference stats가 필요하다.
- deployment-conditional stats는 ablation으로만 둔다.

### 완료 기준

- `--siec_score_mode mean --siec_stats_path ...`가 end-to-end로 실행된다.
- `--siec_score_mode calibrated --siec_stats_path ...`가 end-to-end로 실행된다.
- clean calibration split에서 calibrated score의 timestep별 mean이 raw score보다 낮아진다.
- evaluation split에서도 clean calibrated score가 raw보다 안정적이다.

## M2. Speculative lookahead commit/rollback 정리

### 목표

S-IEC의 NFE overhead가 매 step unconditional +1 forward로 고정되지 않게 한다. non-triggered lookahead는 다음 step의 실제 prediction으로 재사용한다.

### 현재 문제

현재 `lookahead_memo`는 `et`, `x0`만 저장한다. 그러나 DeepCache quick path는 `prv_f` feature state에 의존한다.

DeepCache model은 full path에서 `features`를 반환하고, sampler는 보통 `prv_f = cur_f[0]`로 다음 cached step에 넘긴다. 따라서 speculative lookahead를 commit하려면 다음 상태도 같이 정해야 한다.

### 작업

1. `_call_model` 결과의 `cur_f`를 lookahead에서도 받는다.
2. lookahead가 full path로 수행된 경우:
   - `memo["cur_f"] = cur_f`
   - 다음 step memo hit 시 필요한 경우 `prv_f = cur_f[0]`로 갱신한다.
3. lookahead가 quick path로 수행된 경우:
   - `cur_f`가 `None`일 수 있다.
   - 이 경우 기존 anchor `prv_f`를 유지해야 한다.
4. memo에는 최소 다음 필드를 둔다.

```python
lookahead_memo = {
    "t_int": int(next_t),
    "et": et_look.detach(),
    "x0": x0_look.detach(),
    "cur_f": detach_feature_list_or_none(cur_f),
    "prv_f_after_commit": ...,
    "source_xt_hash_optional": ...,
}
```

5. trigger되지 않은 경우:
   - `xt_next_hat = xt_next_tent`
   - memo를 commit 가능 상태로 저장한다.
6. trigger된 경우:
   - speculative lookahead는 폐기한다.
   - corrected `xt_next_hat`에 대해 필요한 경우 next step prediction을 다시 계산한다.
   - 이 동작은 `rollback`이 아니라 `discard speculative prediction after correction`이라고 기록한다.
7. trace에 `memo_hit`, `memo_committed`, `memo_discarded`, `lookahead_cache_mode`를 기록한다.

### 완료 기준

- `--reuse_lookahead` on/off 비교에서 on의 NFE가 감소한다.
- `--reuse_lookahead` on에서 FID가 off 대비 심하게 악화되지 않는다.
- memo hit step에서 main forward가 skip되는 것이 trace로 확인된다.
- trigger step 뒤에는 이전 speculative lookahead가 재사용되지 않는다.

## M3. Calibrated trigger와 NFE-matched baselines

### 목표

raw syndrome, mean-calibrated syndrome, diagonal-whitened syndrome이 random/periodic보다 나은 trigger인지 검증한다.

### 작업

1. `real_04_tradeoff.py`에 다음 row를 확실히 만든다.
   - DeepCache only / no correction
   - IEC author baseline
   - S-IEC raw
   - S-IEC mean
   - S-IEC calibrated
   - random same-NFE
   - uniform/periodic same-NFE
   - always-correct
2. random/uniform은 S-IEC와 같은 `checked_steps`, 같은 `nfe_per_step` 범위가 되도록 맞춘다.
3. 결과 CSV에는 다음 필드를 반드시 저장한다.
   - `method`
   - `score_mode`
   - `reuse_lookahead`
   - `fid`
   - `per_sample_nfe`
   - `trigger_rate`
   - `checked_rate`
   - `memo_hit_rate`
   - `syndrome_mean`
   - `tau_path`
   - `stats_path`
   - `seed_start`, `seed_end`

### 완료 기준

- calibrated S-IEC가 raw보다 `z_t -> g_t` 또는 final FID/NFE에서 낫거나, 낫지 않다는 negative result가 명확히 나온다.
- S-IEC가 random/uniform보다 못하면 S-IEC claim은 약화하고, feature-level 또는 learned parity로 넘어간다.

## M4. Teacher syndrome and correction gain measurement

### 목표

syndrome이 실제 IEC benefit을 예측하는지 직접 검증한다.

가장 중요한 검증은 다음이다.

```text
z_t -> g_t
```

여기서 `z_t`는 calibrated/learned syndrome score이고, `g_t`는 그 step에 correction을 적용했을 때 reference error가 줄어든 양이다.

### 정의

IEC teacher residual:

```text
Delta_t = || f_theta(x_{t-1}^{tent}, t) - f_theta(x_t, t) ||
```

stepwise correction gain:

```text
g_t = || x_{t-1}^{dep,no_corr} - x_{t-1}^{ref} ||
    - || x_{t-1}^{dep,corrected_at_t} - x_{t-1}^{ref} ||
```

가능하면 `x_t`와 `x0_hat(t)` 둘 다에서 reference error를 기록한다.

### 작업

1. trace mode에 optional teacher computation을 추가한다.
   - CLI 예: `--trace_teacher_delta`
   - 비용이 크므로 default off
2. reference trajectory를 같은 seed로 저장/로드한다.
   - `trace_include_xs=True`
   - `x_t`, `x0_hat(t)` 둘 다 저장 가능해야 한다.
3. stepwise intervention script를 만든다.
   - 권장 경로: `S-IEC/experiments/yongseong/analyze_syndrome_gain.py`
4. 각 timestep마다 다음을 기록한다.
   - `r_t`
   - `z_t_mean`
   - `z_t_calibrated`
   - `Delta_t`
   - `g_t_xt`
   - `g_t_x0`
   - `deployment_label`
   - `timestep_region` = early/mid/late
5. 분석은 최소 Spearman correlation으로 한다.
   - family별
   - timestep region별
   - 전체 pooled

### 판정 기준

| 결과 | 해석 |
|---|---|
| `corr(z_t, g_t)` 높음 | selective correction 가능 |
| `corr(z_t, Delta_t)`만 높고 `g_t`는 낮음 | IEC residual proxy일 뿐 budget signal은 약함 |
| clean/deploy AUROC만 높고 FID 개선 없음 | anomaly detector이지 decoder는 아님 |
| random과 비슷 | current syndrome 폐기 또는 feature-level로 전환 |

### 완료 기준

- `z_t -> g_t` Spearman 표가 생성된다.
- scatter plot이 family별로 저장된다.
- low correlation이면 그 결과를 숨기지 않고 문서화한다.

## M5. Learned parity ablation

### 목표

`linear parity is sufficient`라는 미검증 주장을 제거하고, 실제로 필요한 최소 복잡도 parity를 찾는다.

### ablation ladder

| 단계 | parity model | 구현 우선순위 |
|---|---|---|
| 0 | raw difference `y_{t-1} - y_t` | 이미 있음 |
| 1 | mean drift `y_{t-1} - y_t - mu_t` | M1 |
| 2 | diagonal affine `y_{t-1} - a_t * y_t - b_t` | 우선 구현 |
| 3 | low-rank linear in PCA subspace | 그다음 |
| 4 | piecewise linear by early/mid/late | 필요 시 |
| 5 | small nonlinear predictor | linear 실패 시 |
| 6 | feature-level predictor | image-space 실패 시 |

### 구현 계획

1. 새 모듈을 만든다.
   - 권장 경로: `S-IEC/siec_core/parity.py`
2. `compute_syndrome` 또는 새 `compute_parity_syndrome`에서 parity model을 선택하게 한다.
3. CLI를 추가한다.
   - `--siec_parity_mode {identity,mean,diag_affine,lowrank}`
   - 기존 `--siec_score_mode`와 충돌하지 않게 한다.
4. diagonal affine은 timestep별/channel별 또는 pixel별 중 하나를 선택한다.
   - v1 기본은 channel별 또는 pixel별 diagonal.
   - 너무 큰 stats 파일이 문제가 되면 channel별로 축소한다.
5. low-rank parity는 PCA basis를 stats 파일에 저장한다.

### 저장 포맷 예시

```python
{
    "version": 1,
    "kind": "parity_model",
    "mode": "diag_affine",
    "a": Tensor[T, C, H, W] or Tensor[T, C, 1, 1],
    "b": Tensor[T, C, H, W] or Tensor[T, C, 1, 1],
    "residual_mu": Tensor[T, C, H, W],
    "residual_var": Tensor[T, C, H, W],
    "train_seed_range": [0, 2047],
    "config": {...}
}
```

### 완료 기준

- raw/mean/calibrated/diag-affine이 같은 script에서 비교된다.
- clean residual 감소율을 표로 낸다.
- `corr(z_t, g_t)`가 raw보다 좋아지는지 확인한다.
- linear가 실패하면 그 결과를 근거로 nonlinear 또는 feature-level로 넘어간다.

## M6. Controller로 확장

### 목표

S-IEC가 단순 anomaly gate가 아니라 ECC-inspired controller임을 보인다.

### 현재 상태

현재 코드는 대부분 다음 구조다.

```text
score > tau -> correction
score <= tau -> skip
```

이것만으로는 anomaly detection과 구분이 약하다.

### v1 controller policy

처음부터 복잡한 learned policy를 만들지 않는다. deterministic rule로 시작한다.

```text
rho_t < tau1
    -> skip
tau1 <= rho_t < tau2
    -> keep memo, defer to next refresh
tau2 <= rho_t < tau3
    -> light S-IEC correction
rho_t >= tau3
    -> discard memo, force full recompute or IEC
```

### 필요한 trace pattern

1. single spike
   - local cached step error로 해석
2. interval drift
   - cache stale / accumulated drift로 해석
3. refresh-adjacent spike
   - anchor instability로 해석

### 작업

1. `rho_t` 계산 함수를 만든다.
   - 권장 경로: `S-IEC/siec_core/reliability.py`
2. 입력 feature:
   - `z_t`
   - `cache_age_t`
   - `refresh_step`
   - recent scores window: `[z_{t-2}, z_{t-1}, z_t]`
   - timestep region
3. correction action enum을 만든다.
   - `skip`
   - `defer`
   - `light_siec`
   - `iec`
   - `early_refresh`
4. trace에 action을 저장한다.

### 완료 기준

- 같은 score를 쓰는 `gate-only`와 `controller`를 비교한다.
- controller가 같은 NFE에서 gate-only보다 좋거나, 아니면 controller claim을 약화한다.
- 논문에는 최소 `gate-only vs controller` table을 넣는다.

## 4. 필수 실험표

### Table A. Baseline 재현

| Method | Correction policy | Expected role |
|---|---|---|
| Full fp16 DDIM | none | reference |
| DeepCache only | none | deployed lower bound |
| DeepCache + IEC author | fixed non-cached steps | author baseline |
| DeepCache + random same-NFE | random | trigger control |
| DeepCache + periodic same-NFE | uniform | schedule control |
| S-IEC raw | raw syndrome | old method |
| S-IEC calibrated | clean calibrated syndrome | proposed signal |
| S-IEC learned parity | learned residual | stronger proposed signal |
| oracle trigger | reference gain | upper bound |

### Table B. Signal quality

| Score | clean residual | clean/deploy AUROC | Spearman with Delta | Spearman with gain |
|---|---:|---:|---:|---:|
| raw | | | | |
| mean | | | | |
| calibrated | | | | |
| diag affine | | | | |
| low-rank | | | | |

### Table C. NFE-matched quality

| Method | FID | NFE | trigger rate | memo hit rate |
|---|---:|---:|---:|---:|
| random | | | | |
| periodic | | | | |
| raw S-IEC | | | | |
| calibrated S-IEC | | | | |
| learned parity S-IEC | | | | |

### Table D. Gate vs controller

| Policy | Score source | FID | NFE | Notes |
|---|---|---:|---:|---|
| gate-only | calibrated | | | |
| controller | calibrated | | | |
| controller | learned parity | | | |

## 5. 문서와 논문에서 써야 할 안전한 표현

### 써도 되는 표현

- `analog trajectory code`
- `statistical parity residual`
- `diagonal-whitened trajectory syndrome`
- `ECC-inspired reliability controller`
- `correction budget allocation`
- `speculative lookahead reuse`
- `discard speculative lookahead after trigger`

### 피해야 할 표현

- `off-manifold ECC decoder`
- `data manifold parity check`
- `Hc=0 exactly holds`
- `linear parity is sufficient`
- `rollback to previous timestep` unless code really restores a previous sampler state
- `Mahalanobis` without saying diagonal/low-rank/full

## 6. 최소 성공 기준

S-IEC를 positive claim으로 밀려면 다음 중 최소 2개 이상이 필요하다.

1. calibrated or learned score가 raw보다 `corr(z_t, g_t)`를 개선한다.
2. same-NFE에서 calibrated/learned S-IEC가 random과 periodic을 이긴다.
3. `--reuse_lookahead`로 NFE가 IEC author baseline 근처까지 내려가면서 FID가 유지된다.
4. controller가 gate-only보다 같은 NFE에서 낫다.
5. oracle과 S-IEC 사이 gap이 해석 가능할 정도로 좁다.

위 조건이 충족되지 않으면 논문 claim은 다음처럼 낮춘다.

> We identify why raw trajectory syndromes fail as ECC-style decoders in CIFAR DDPM, and show that clean-drift calibration and speculative reuse are necessary conditions for an ECC-inspired IEC controller.

즉 negative/diagnostic contribution으로 전환한다.

## 7. 작업 순서 체크리스트

1. M0 trace/NFE 정리
2. true no-correction path 검증
3. `--reuse_lookahead` trace 검증
4. clean stats calibration script 작성
5. raw/mean/calibrated score sanity check
6. NFE-matched random/periodic baseline 재생성
7. `z_t -> Delta_t`, `z_t -> g_t` 분석
8. diagonal affine learned parity 구현
9. learned parity ablation
10. gate-only vs controller 비교
11. paper claim 업데이트

## 8. 다음 작업자가 바로 시작할 때 볼 파일

- redesign report:
  - `S-IEC/docs/siec_redesign_report_v0.md`
- 이 실행 계획:
  - `S-IEC/docs/siec_redesign_execution_plan_v1.md`
- syndrome implementation:
  - `S-IEC/siec_core/syndrome.py`
- correction strength:
  - `S-IEC/siec_core/correction.py`
- main sampler:
  - `S-IEC/mainddpm/ddpm/functions/deepcache_denoising.py`
- CLI entry:
  - `S-IEC/mainddpm/ddim_cifar_siec.py`
- runner dispatch:
  - `S-IEC/mainddpm/ddpm/runners/deepcache.py`
- experiment wrappers:
  - `S-IEC/experiments/yongseong/real_04_tradeoff.py`
  - `S-IEC/experiments/yongseong/real_05_robustness.py`
- previous code verification:
  - `S-IEC/docs/critique/siec_code_verification_20260428.md`

## 9. 최종 판정 로직

실험이 끝나면 아래 순서로 결론을 낸다.

1. `No correction`과 `IEC author` baseline이 정상인가?
   - 아니면 모든 S-IEC 비교는 무효다.
2. `reuse_lookahead`가 NFE를 줄였는가?
   - 아니면 S-IEC는 Pareto에서 불리하다.
3. calibrated score가 raw보다 clean drift를 잘 제거했는가?
   - 아니면 calibration 설계가 틀렸거나 stats split이 잘못됐다.
4. `z_t`가 `g_t`를 예측하는가?
   - 아니면 trigger signal이 correction budget allocation에 부적합하다.
5. same-NFE random/periodic보다 좋은가?
   - 아니면 syndrome selection claim은 성립하지 않는다.
6. controller가 gate-only보다 좋은가?
   - 아니면 ECC-inspired controller claim은 약화한다.

이 순서를 통과해야만 최종 claim을 positive result로 쓴다.
