# Experiment 5 — Setting Inventory

오늘 기준으로 실험 5의 6개 setting이 실행 가능한지 검사한 표.
`runnable`만 wrapper가 바로 돌릴 수 있고, `blocked`는 unblock_via에
적힌 절차를 수행해야 runnable이 된다. 어떤 경우에도 1저자 core 코드를
수정하지 않는다 — core 수정이 필요한 setting은 Candidate 플래그로 표시.

| # | Setting | Status | W | A | Interval | Missing assets | Unblock via |
|---|---|---|---|---|---|---|---|
| 1 | `fp16` | **runnable** | — | — | — | — | Candidate C4: ddim_cifar_siec.py + deepcache.py 실험 복사본에서 --no-ptq/--no-cache 플래그 제공. |
| 2 | `W8A8_DC10` | **runnable** | 8 | 8 | 10 | — | — |
| 3 | `W8A8_DC20` | **blocked** | 8 | 8 | 20 | error_dec/cifar/pre_quanterr_abCov_weight8_interval20_list_timesteps100.pth, error_dec/cifar/pre_cacheerr_abCov_interval20_list_timesteps100.pth | Re-run mainddpm/ddim_cifar_cali.py with --replicate_interval 20 to generate DEC params (no core edit). |
| 4 | `W4A8_DC10` | **blocked** | 4 | 8 | 10 | error_dec/cifar/pre_quanterr_abCov_weight4_interval10_list_timesteps100.pth, error_dec/cifar/weight_params_W4_cache10_timesteps100.pth | Re-run mainddpm/ddim_cifar_params.py with --weight_bit 4, then regenerate DEC params (heavy GPU work). |
| 5 | `W8A8_DC50` | **blocked** | 8 | 8 | 50 | error_dec/cifar/pre_quanterr_abCov_weight8_interval50_list_timesteps100.pth, error_dec/cifar/pre_cacheerr_abCov_interval50_list_timesteps100.pth | Re-run mainddpm/ddim_cifar_cali.py with --replicate_interval 50 (no core edit). |
| 6 | `CacheQuant` | **blocked** | 4 | 8 | 10 | error_dec/cifar/pre_quanterr_abCov_weight4_interval10_list_timesteps100.pth, error_dec/cifar/weight_params_W4_cache10_timesteps100.pth | After Setting 4 (W4A8_DC10) is unblocked; combined regime. |

## Method rows per setting

양자화/캐시 regime (W8A8_DC10 등) 은 세 줄: `no-correction`, `IEC (author)`, `S-IEC`.
**fp16** 은 과학적으로 IEC/S-IEC 가 모두 no-op (양자화 에러가 없어 수정할 것이 없음) 이므로 단일 `fp16 reference` row 로 축소. pilot/calibrate 단계도 생략.

- `fp16 reference`: `--no-use-siec` 로 plain DDIM 호출 (C4 실험 복사본의 `--no-ptq/--no-cache`).
- `no-correction`: `--tau_path calibration/tau_schedule_never.pt --use_siec` (모든 step trigger off; S-IEC sampler 의 unconditional lookahead 는 남아 NFE=110).
- `IEC (author)`: `W8A8_DC10` 은 기존 `iec_samples.npz` (50K seed) 재사용. 다른 setting 에서는 Candidate C2 (`--no-use-siec`) 실험 복사본에서 fresh run 가능. NFE = 100 + n_checks (max_iter=2 at interval_seq).
- `S-IEC`: `--tau_path tau_schedule_{label}_p{P}.pt` (W8A8_DC10 은 canonical `tau_schedule_p{P}.pt` 직접 참조). fp16 에는 적용 불가.

## Error-strength axis

`error_strength(label) = mean_over_t(mean(pilot_scores_{label}[t]))` 를 우선 사용.
Fallback: `fid_no_corr[label] − fid_no_corr[least_errored_runnable]`.