# 용성 실험 4/5 실행 명령

마지막 검증: 2026-04-24 KST

이 문서는 현재 코드 기준으로 다시 썼다. 예전 문서에 있던 `--use-experiment-copy`, 옛 setting 이름(`W8A8_DC10` 등), 고정 결과 디렉토리 가정은 더 이상 맞지 않는다.

현재 wrapper 기본값은 `CUDA_VISIBLE_DEVICES=2`다. 즉, 별도 옵션을 주지 않으면 생성되는 `commands.sh`와 wrapper 내부 subprocess가 모두 2번 GPU를 보게 된다.

중요:

- 실험 4/5 모두 S-IEC semantics가 2026-04-24에 수정됐다.
- 따라서 이전에 생성한 `pilot_scores*.pt`, `tau_schedule*.pt`는 stale일 수 있다.
- 특히 실험 5 wrapper는 기존 pilot/tau 파일이 있으면 재사용하므로, "정합한 재생성"을 원하면 기존 calibration artifact를 백업/삭제한 뒤 다시 만들어야 한다.

## 0. 현재 상태

지금 코드는 아래까지는 확인됐다.

- `real_04_tradeoff.py`: dry-run 정상, `13`개 row 생성, `10`개 sampling/fid command 생성
- `real_05_robustness.py`: inventory/dry-run 정상, `24`개 row 생성, 현재 runnable setting은 `fp16`, `w8a8`, `dc10`
- `TAC`는 wrapper/CSV/plot skeleton만 있고 실제 알고리즘은 미구현이다. 따라서 최종 CSV/plot에는 `TAC not implemented`가 남는다.

즉, wrapper 요구사항은 반영됐지만, 문자 그대로의 "모든 방법 결과"는 아직 아니다. 현재 가능한 최종 산출물은 다음 둘이다.

- 실험 4: `TAC`를 blocked row로 둔 상태의 full tradeoff CSV/plot
- 실험 5: missing asset을 먼저 만든 뒤, `TAC`를 blocked row로 둔 상태의 full robustness CSV/plot

## 1. 공통 준비

```bash
cd /home/user/jowithu/Semantic/IEC
export CUDA_VISIBLE_DEVICES=2
tmux new -s yongseong
watch -d -n 1 nvidia-smi
```

이 문서의 모든 Python 명령은 `conda run --no-capture-output -n iec` 기준으로 적는다. 다른 GPU를 써야 하면 각 wrapper에 `--cuda-visible-devices X`를 넘기면 된다.

## 2. 실험 4 최종 실행

실험 4는 현재 자산 기준으로 바로 돌릴 수 있다. `tau_schedule_p30/p50/p60/p70/p80/p90/p95.pt`가 없더라도 wrapper가 자동으로 calibration command를 `commands.sh`에 넣는다.

가장 안전한 실행은 아래 one-shot 스크립트다. pilot 재생성, tau 7개 재생성, exp4 full run을 순서대로 수행한다.

```bash
cd /home/user/jowithu/Semantic/IEC/experiments/yongseong
./run_exp4_refresh.sh
```

### 권장: wrapper로 한 번에 끝내기

```bash
conda run --no-capture-output -n iec \
  python experiments/real_04_tradeoff.py --no-dry-run
```

완료 후 최신 run 디렉토리 확인:

```bash
RUN04=$(ls -td experiments/yongseong/results/real_04_tradeoff/* | head -n1)
echo "$RUN04"
ls "$RUN04"
```

재플롯만 할 때:

```bash
conda run --no-capture-output -n iec \
  python experiments/real_04_tradeoff.py --plot-only --results-dir "$RUN04"
```

### 선택: dry-run으로 command를 먼저 검토하기

```bash
conda run --no-capture-output -n iec \
  python experiments/real_04_tradeoff.py --dry-run

RUN04=$(ls -td experiments/yongseong/results/real_04_tradeoff/* | head -n1)
sed -n '1,260p' "$RUN04/commands.sh"
bash "$RUN04/commands.sh"

conda run --no-capture-output -n iec \
  python experiments/real_04_tradeoff.py --no-dry-run --results-dir "$RUN04"
```

중요:

- `bash "$RUN04/commands.sh"`를 쓴 뒤에는 반드시 같은 `RUN04`를 `--results-dir`로 다시 넘긴다.
- 새 run 디렉토리에서 `--no-dry-run`을 또 실행하면 같은 실험을 다시 시작하게 된다.

주요 산출물:

- `"$RUN04/results.csv"`
- `"$RUN04/results.json"`
- `"$RUN04/compute_matched.md"`
- `"$RUN04/tradeoff_2panel.png"`
- `"$RUN04/tradeoff_2panel.pdf"`
- `"$RUN04/artifacts/"` 아래 syndrome histogram/json

## 3. 실험 5 현재 inventory 확인

현재 taxonomy는 아래 여섯 개다.

- `fp16`
- `w8a8`
- `dc10`
- `w4a8`
- `dc20`
- `cachequant`

현재 workspace에서 확인된 runnable setting은 `fp16`, `w8a8`, `dc10`이다.

```bash
conda run --no-capture-output -n iec \
  python experiments/real_05_robustness.py --phase inventory

RUN05_INV=$(ls -td experiments/yongseong/results/real_05_robustness/* | head -n1)
sed -n '1,220p' "$RUN05_INV/inventory.md"
```

## 4. 실험 5 blocked asset 해제

실험 5를 "한 번의 최종 run"으로 끝내려면 먼저 `w4a8`, `dc20`, `cachequant`를 풀어야 한다.

### 4.1 `dc20` asset 생성

필요 파일:

- `calibration/cifar100_cache20_uni.pth`
- `calibration/cifar_feature_maps_interval20_timesteps100.pt`
- `error_dec/cifar/pre_cacheerr_abCov_interval20_list_timesteps100.pth`

생성 명령:

```bash
conda run --no-capture-output -n iec \
  python mainddpm/ddim_cifar_cali.py --replicate_interval 20 --timesteps 100

conda run --no-capture-output -n iec \
  python mainddpm/ddim_cifar_predadd.py --replicate_interval 20 --timesteps 100

conda run --no-capture-output -n iec \
  python error_dec/cifar/cifar_dec.py --error cache --replicate_interval 20 --timesteps 100
```

### 4.2 `w4a8` asset 생성

현재 inventory 기준으로 cache10 관련 cache asset은 이미 있다. 부족한 것은 `W4A8` quant DEC asset이다.

필요 파일:

- `error_dec/cifar/weight_quantizer_params_aftercacheadd_W4_cache10_timesteps100.pth`
- `error_dec/cifar/act_quantizer_params_aftercacheadd_W4_cache10_timesteps100.pth`
- `error_dec/cifar/weight_params_W4_cache10_timesteps100.pth`
- `error_dec/cifar/pre_quanterr_abCov_weight4_interval10_list_timesteps100.pth`

생성 명령:

```bash
conda run --no-capture-output -n iec \
  python mainddpm/ddim_cifar_params.py --weight_bit 4 --act_bit 8 --replicate_interval 10 --timesteps 100

conda run --no-capture-output -n iec \
  python error_dec/cifar/cifar_dec.py --error quant --weight_bit 4 --replicate_interval 10 --timesteps 100
```

### 4.3 `cachequant` 해제

`cachequant`는 `W4A8 + DeepCache interval=10` 조합이다. 현재 workspace에서는 cache10 asset이 이미 있으므로, 위 `w4a8` 생성이 끝나면 같이 풀린다.

### 4.4 inventory 재확인

```bash
conda run --no-capture-output -n iec \
  python experiments/real_05_robustness.py --phase inventory

RUN05_INV=$(ls -td experiments/yongseong/results/real_05_robustness/* | head -n1)
sed -n '1,220p' "$RUN05_INV/inventory.md"
```

최종 full run 전에 `inventory.md`에서 여섯 setting이 모두 `runnable`인지 확인한다.

## 5. 실험 5 최종 실행

실험 5는 wrapper만 바로 실행하면 기존 `pilot_scores_{setting}.pt`, `tau_schedule_{setting}_p80.pt`를 재사용할 수 있다. 즉, 문서의 일반 명령만으로는 "새 semantics 기준 재생성"이 보장되지 않는다.

가장 안전한 실행은 아래 one-shot 스크립트다. 이 스크립트는:

- `dc20`, `w4a8`, `cachequant` blocked asset 생성
- 기존 `pilot_scores_{setting}.pt`, `tau_schedule_{setting}_p80.pt`를 timestamped backup으로 이동
- inventory 재확인
- setting별 pilot/tau를 새로 만들면서 full robustness run 수행

```bash
cd /home/user/jowithu/Semantic/IEC/experiments/yongseong
./run_exp5_refresh.sh
```

### 권장: blocked asset을 먼저 다 만든 뒤 full run 한 번에 끝내기

```bash
conda run --no-capture-output -n iec \
  python experiments/real_05_robustness.py --phase all --no-dry-run
```

완료 후 최신 run 디렉토리 확인:

```bash
RUN05=$(ls -td experiments/yongseong/results/real_05_robustness/* | head -n1)
echo "$RUN05"
ls "$RUN05"
```

재플롯만 할 때:

```bash
conda run --no-capture-output -n iec \
  python experiments/real_05_robustness.py --phase plot --results-dir "$RUN05"
```

### 선택: dry-run으로 command를 먼저 검토하기

```bash
conda run --no-capture-output -n iec \
  python experiments/real_05_robustness.py --phase all --dry-run

RUN05=$(ls -td experiments/yongseong/results/real_05_robustness/* | head -n1)
sed -n '1,260p' "$RUN05/commands.sh"
bash "$RUN05/commands.sh"

conda run --no-capture-output -n iec \
  python experiments/real_05_robustness.py --phase all --no-dry-run --results-dir "$RUN05"
```

중요:

- `bash "$RUN05/commands.sh"` 뒤에는 반드시 같은 `RUN05`를 `--results-dir`로 다시 넘긴다.
- 새 run 디렉토리에서 `--phase all --no-dry-run`을 다시 시작하면 pilot/calibrate/main을 처음부터 또 돌리게 된다.

주요 산출물:

- `"$RUN05/inventory.md"`
- `"$RUN05/results.csv"`
- `"$RUN05/results.json"`
- `"$RUN05/robustness_2panel.png"`
- `"$RUN05/robustness_2panel.pdf"`

## 6. 지금 당장 빠르게 sanity run만 할 때

asset 생성 전에 현재 runnable setting만 빨리 확인하고 싶다면:

```bash
conda run --no-capture-output -n iec \
  python experiments/real_05_robustness.py --phase all --settings fp16 w8a8 dc10 --no-dry-run
```

이 명령은 partial robustness 결과를 만든다. 최종 제출용 full figure를 원하면, asset을 다 만든 뒤 `--settings` 없이 fresh full run을 다시 돌리는 편이 맞다.

## 7. 주의사항

- `--use-experiment-copy`는 현재 wrapper에 없다. 쓰면 안 된다.
- `TAC`는 현재 실제 계산이 없으므로, CSV/plot에 numeric TAC 결과가 채워지지 않는 것이 정상이다.
- 실험 5의 `w4a8`, `dc20`, `cachequant` blocked는 wrapper 버그가 아니라 asset 부재 때문이다.
- `commands.sh`를 수동으로 실행했다면, 후처리 wrapper는 반드시 같은 timestamped 결과 디렉토리로 이어서 실행해야 한다.
