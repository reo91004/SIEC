# S-IEC 재설계 보고서  
## IEC에 ECC 이론을 결합해 더 나은 quality/NFE Pareto를 만들기 위한 정리

## 0. 오류 검증 및 추가 보강

제공된 보고서의 큰 방향은 정합하다. DDPM, DDIM, DeepCache, IEC, S-IEC의 역할 구분은 맞고, S-IEC가 ECC의 syndrome 개념을 가져와 IEC의 correction 호출을 더 선택적으로 만들려 했다는 해석도 맞다. 또한 현재 구현에서 syndrome이 실제 deployment error를 제대로 분리하지 못하고, NFE overhead가 커져 IEC에 dominated된다는 결론도 현재 자료와 일치한다.

다만 두 가지 표현은 더 정밀하게 고쳐야 한다.

첫째, **“toy의 결론을 정직하게 따라가도 syndrome은 trigger 신호로 부적합”** 이라는 표현은 방향은 맞지만, 이유는 “부적합”보다 **“분리 불가”** 가 더 정확하다. Toy에서는 syndrome이 $\mathrm{span}(U)$ 방향, 즉 코드 내부 명명 기준의 normal 방향에 반응한다. 그런데 이 $\mathrm{span}(U)$는 표준 미분기하 기준으로는 data manifold의 tangent 방향, 즉 데이터가 실제로 사는 방향이다. 문제는 deployment error도 이 방향에서 나타날 수 있고, clean diffusion의 자연 refinement도 같은 방향에서 일어난다는 점이다. 따라서 syndrome은 “오류가 있는가”를 독립적으로 보는 것이 아니라, **자연 refinement와 deployment error가 섞인 총 변화량**을 본다. 업로드된 `syndrome_notes`도 toy와 CIFAR에서 syndrome 공식은 같지만, 일반 DDPM에서는 clean syndrome이 0으로 보장되지 않고 calibration이 필요하다고 정리한다.

둘째, **“learned parity는 small linear regression 정도면 충분하다”** 는 문장은 claim으로 쓰면 안 된다. 이 의견은 맞다. Diffusion trajectory에서 $\hat{x}_0(t)$와 $\hat{x}_0(t-1)$의 관계는 timestep, noise schedule, sampler, U-Net approximation, DeepCache state에 따라 달라진다. 특히 큰 $t$에서는 semantic formation이 강하고, 작은 $t$에서는 detail refinement가 강하다. 이 두 구간의 dynamics가 같은 선형 map으로 충분히 설명된다는 보장은 없다. 따라서 per-timestep linear predictor는 **가장 먼저 검증할 baseline**이지, 충분하다고 주장할 수 있는 완성된 모델이 아니다.

이 보강점 때문에 재설계된 S-IEC에서는 learned parity를 다음처럼 둔다.

$$
\text{raw difference}
\rightarrow
\text{mean-drift calibration}
\rightarrow
\text{diagonal / scalar affine parity}
\rightarrow
\text{low-rank linear parity}
\rightarrow
\text{small nonlinear parity}
$$

즉 linear predictor는 “충분하다”가 아니라 **ablation ladder의 중간 단계**다. 선형 모델이 clean trajectory residual을 충분히 줄이고, deployment error 또는 IEC benefit을 잘 예측할 때만 최종 방식으로 채택한다. 그렇지 않으면 false positive trigger가 많아져 모든 step에서 correction이 켜질 수 있으므로, nonlinear predictor 또는 feature-level syndrome으로 넘어가야 한다.

---

## 1. 각 기법이 무엇인가

### 1.1 DDPM

DDPM은 노이즈가 가득한 상태에서 시작해 점진적으로 노이즈를 제거하며 이미지를 생성하는 기본 diffusion 모델이다.

Forward 과정은 원본 이미지 $x_0$에 시간 $t$에 따라 노이즈를 섞는다.

$$
x_t = \alpha_t x_0 + \sigma_t \epsilon
$$

여기서 $\alpha_t$는 signal 강도이고, $\sigma_t$는 noise 강도다. $t$가 클수록 $\alpha_t$는 작아지고 $\sigma_t$는 커진다. Reverse 과정에서는 신경망 $f_\theta$, 보통 U-Net이 현재 noisy image $x_t$를 보고 최종 clean image 또는 noise를 예측한다.

$$
\hat{x}_0(t)=f_\theta(x_t,t)
$$

이 $\hat{x}_0(t)$는 이후 DDIM 또는 DDPM sampler가 $x_{t-1}$로 이동하는 데 사용하는 핵심 추정량이다.

### 1.2 DDIM

DDIM은 현재 $x_t$와 모델이 예측한 $\hat{x}_0$를 사용해 다음 상태 $x_{t-1}$로 이동하는 deterministic sampler로 볼 수 있다.

$$
x_{t-1}
=
\alpha_{t-1}\hat{x}_0
+
\sigma_{t-1}
\frac{x_t-\alpha_t\hat{x}_0}{\sigma_t}
$$

직관적으로는 현재 상태 $x_t$를 “예측된 clean image 방향”으로 끌어당기면서, 남아 있는 noise direction도 $\sigma_{t-1}/\sigma_t$ 비율에 맞게 줄이는 방식이다. DDIM step이 반복되면 $x_T$에서 $x_0$로 이동한다.

### 1.3 DeepCache

DeepCache는 U-Net 내부 feature가 인접 diffusion step 사이에서 크게 변하지 않는다는 점을 이용해 계산을 줄이는 training-free 가속 기법이다. DeepCache 논문은 reverse diffusion 과정의 temporal redundancy를 이용해 high-level feature를 캐시하고, 이후 step에서 이를 재사용하며 low-level feature만 저렴하게 업데이트한다고 설명한다.

DeepCache의 기본 구조는 다음과 같다.

$$
x_t
\xrightarrow{f_\theta \text{ full-compute}}
\hat{x}_0(t)
\xrightarrow{\text{DDIM}}
x_{t-1}
\xrightarrow{f_\theta \text{ cached/partial}}
\hat{x}_0(t-1)+\epsilon_{\text{cache}}
$$

이때 $\epsilon_{\text{cache}}$는 feature reuse로 생기는 근사 오차다. DeepCache는 빠르지만, cache interval이 길거나 feature 재사용이 맞지 않는 step에서는 품질이 떨어질 수 있다.

DeepCache supplementary material은 $N$ step 동안 1회의 complete model inference와 $N-1$회의 partial model inference가 수행된다고 설명한다.

### 1.4 IEC

IEC는 efficient diffusion model에서 생기는 approximation error가 timestep을 따라 누적되는 문제를 test-time에서 줄이는 방법이다. IEC 논문은 efficient diffusion model의 approximation error가 diffusion step을 거치며 누적될 수 있고, IEC가 이를 exponential growth에서 linear growth로 완화한다고 주장한다.

IEC의 핵심은 같은 timestep $t$ 기준에서 local self-consistency를 보는 것이다. 현재 $x_t$에서 모델이 $\hat{x}_0$를 예측한다.

$$
\hat{x}_0 = f_\theta(x_t,t)
$$

이를 이용해 tentative step을 만든다.

$$
x_{t-1}^{\text{tent}}
=
\alpha_{t-1}\hat{x}_0
+
\sigma_{t-1}
\frac{x_t-\alpha_t\hat{x}_0}{\sigma_t}
$$

그다음 $x_{t-1}^{\text{tent}}$를 다시 timestep $t$ 조건으로 평가한다.

$$
\hat{x}_0^{\text{new}}
=
f_\theta(x_{t-1}^{\text{tent}},t)
$$

그리고 다음 차이를 본다.

$$
\Delta_t
=
\|\hat{x}_0^{\text{new}}-\hat{x}_0\|
$$

$\Delta_t$가 크다는 것은 현재 모델 출력이 local self-consistency를 만족하지 않는다는 뜻이다. IEC는 이 정보를 이용해 output을 iterative하게 refine한다.

DeepCache와 결합할 때 중요한 점은, IEC가 모든 cached step에 무작정 적용되는 것이 아니라 **non-cached timestep에 적용된다**는 점이다. IEC 논문은 quantization 기반 방법에는 IEC를 every timestep에 적용하고, DeepCache와 CacheQuant에는 non-cached timestep에만 적용한다고 명시한다.

---

## 2. S-IEC가 풀려고 한 문제

IEC의 문제의식은 명확하다. DeepCache나 quantization으로 인해 생긴 approximation error를 test-time에서 줄인다. 하지만 IEC는 correction을 수행할 때 추가 연산을 요구한다. 모든 가능한 step에서 correction을 수행하면 품질은 좋아질 수 있지만 비용이 늘어난다.

IEC 원문도 이 trade-off를 인정한다. DeepCache와 CacheQuant에서는 IEC를 non-cached timestep에만 적용하고, MS-COCO에서는 first timestep에만 적용한다고 설명한다.

S-IEC의 출발점은 여기다.

$$
\text{IEC를 언제 실행해야 하는가?}
$$

기존 IEC는 DeepCache에서는 non-cached timestep, 즉 full-compute refresh step 중심으로 correction을 적용한다. 그러나 그중에서도 어떤 step은 오류가 작고, 어떤 step은 오류가 클 수 있다. 따라서 S-IEC는 ECC의 syndrome 개념을 이용해 **오류가 커 보이는 step에서만 correction을 실행**하려 했다.

현재 S-IEC의 syndrome은 다음이다.

$$
\hat{s}_t
=
\hat{x}_0(t)-\hat{x}_0(t-1)
$$

$$
r_t
=
\frac{\|\hat{s}_t\|^2}{d}
$$

그리고 trigger rule은 다음이다.

$$
r_t > \tau_t
\Rightarrow
\text{correction 실행}
$$

$$
r_t \le \tau_t
\Rightarrow
\text{그냥 진행}
$$

즉 S-IEC의 핵심 목표는 “IEC를 더 많이 하자”가 아니라, **IEC correction budget을 syndrome으로 더 잘 배분하자**다.

---

## 3. S-IEC는 IEC에 ECC를 어떻게 결합하려 했는가

### 3.1 ECC에서 syndrome의 의미

ECC에서 codeword $c$는 parity check matrix $H$를 만족한다.

$$
Hc=0
$$

전송 중 오류 $e$가 생기면 수신값은

$$
r=c+e
$$

가 되고, syndrome은 다음이 된다.

$$
s=Hr=H(c+e)=Hc+He=He
$$

따라서 ECC에서 syndrome은 올바른 codeword 성분을 지우고, 오류 성분만 남기는 역할을 한다.

이 구조의 핵심은 세 가지다.

$$
\text{codeword space가 명시적이다}
$$

$$
Hc=0 \text{이 모든 valid codeword에 대해 성립한다}
$$

$$
s=Hr \text{이 오류 위치와 크기에 대한 정보를 준다}
$$

### 3.2 S-IEC의 초기 ECC mapping

S-IEC는 이 구조를 diffusion에 다음처럼 대응시키려 했다.

| ECC 개념 | S-IEC 초기 대응 |
|---|---|
| codeword $c$ | 오류 없는 diffusion trajectory 또는 data manifold |
| channel | PTQ, DeepCache, CacheQuant |
| received word $r$ | efficient sampler가 만든 오염된 trajectory |
| parity check | 인접 $\hat{x}_0$ 일관성 |
| syndrome | $\hat{x}_0(t)-\hat{x}_0(t-1)$ |
| decoder | syndrome이 크면 correction |

업로드된 `siec_ecc_framing`도 S-IEC가 단순한 selective IEC가 아니라, ECC 개념을 diffusion sampler 위에 올려 syndrome decoder처럼 쓰려는 구조였다고 정리한다.

핵심 직관은 다음이었다.

$$
\text{clean trajectory}
\Rightarrow
\hat{x}_0(t)\approx \hat{x}_0(t-1)
\Rightarrow
\hat{s}_t\approx 0
$$

$$
\text{deployment-corrupted trajectory}
\Rightarrow
\hat{x}_0(t)\not\approx \hat{x}_0(t-1)
\Rightarrow
\|\hat{s}_t\|\gg 0
$$

따라서 $\hat{s}_t$를 ECC의 syndrome처럼 쓰고, $r_t$가 큰 step에만 IEC 또는 S-IEC correction을 실행한다는 아이디어였다.

---

## 4. Toy Gaussian에서는 왜 그럴듯했는가

Toy Gaussian에서는 S-IEC가 상당히 그럴듯하게 보인다. 이유는 posterior mean이 closed form projection 구조를 갖기 때문이다.

$$
X_0=UZ,\quad Z\sim \mathcal{N}(0,\Lambda)
$$

$$
\hat{x}_0(t)=UD_tU^\top x_t
$$

여기서 $U$는 데이터가 사는 저차원 subspace를 span하는 행렬이다. 따라서 $\hat{x}_0(t)$는 항상 $\mathrm{span}(U)$ 안에 있다.

$$
\hat{x}_0(t)\in \mathrm{span}(U)
$$

이 구조 때문에 toy에서는 특정 방향 오류가 posterior mean에 반영되고, 다른 방향 오류는 $U^\top v=0$ 때문에 posterior mean에서 사라진다. 업로드된 `syndrome_notes`도 toy에서 $\mathrm{span}(U)$ 방향 오류는 syndrome에 비례해 나타나고, $\mathrm{span}(U)^\perp$ 방향 오류는 $U^\top v=0$ 때문에 syndrome에 거의 반영되지 않는다고 정리한다.

그러나 여기서 매우 중요한 용어 문제가 있다.

Toy 코드의 명명은 표준 미분기하와 반대다.

$$
\text{코드의 normal space}=\mathrm{span}(U)
$$

$$
\text{코드의 tangent space}=\mathrm{span}(U)^\perp
$$

하지만 표준 미분기하에서는 데이터 manifold가 $M=\mathrm{span}(U)$이면,

$$
T_xM=\mathrm{span}(U)
$$

$$
N_xM=\mathrm{span}(U)^\perp
$$

이다. 업로드된 manifold note도 이 라벨이 표준 정의와 정반대라고 명시한다.

따라서 toy의 결과를 표준 용어로 다시 쓰면 다음이다.

**Toy syndrome은 data manifold tangent 방향, 즉 데이터가 사는 방향의 오류에 반응한다. 반대로 off-manifold normal 방향의 오류는 posterior mean projection과 DDIM dynamics에 의해 자연히 제거되므로 syndrome에 잘 보이지 않는다.**

이 결론은 매우 중요하다. 왜냐하면 이것이 초기 “data manifold ECC” 프레이밍과 충돌하기 때문이다. data manifold를 codeword로 보면 ECC parity check는 off-manifold normal error를 잡아야 한다. 그런데 toy S-IEC가 실제로 잘 잡는 것은 $\mathrm{span}(U)$, 즉 데이터 방향 오류다.

따라서 toy의 정직한 해석은 다음이다.

**Toy는 S-IEC가 data manifold 밖으로 튄 오류를 잡는다는 증거가 아니다. Toy는 posterior trajectory에서 증폭되는 data-subspace error를 $\hat{x}_0$ consistency syndrome으로 감지할 수 있다는 증거다.**

---

## 5. 왜 기존 S-IEC가 실패했는가

### 5.1 원인 1: syndrome이 deployment error만 측정하지 않는다

현재 syndrome은 다음이다.

$$
\hat{s}_t
=
\hat{x}_0(t)-\hat{x}_0(t-1)
$$

이를 더 정직하게 분해하면 다음과 같다.

$$
\hat{s}_t
=
\underbrace{\Delta_{\text{natural}}}_{\text{clean diffusion의 자연 변화}}
+
\underbrace{\Delta_{\text{model bias}}}_{\text{UNet 근사 오차}}
+
\underbrace{\Delta_{\text{discretization}}}_{\text{DDIM 이산화 오차}}
+
\underbrace{\Delta_{\text{deployment}}}_{\text{PTQ/DeepCache 오류}}
$$

S-IEC가 원하는 것은 $\Delta_{\text{deployment}}$다. 하지만 실제 syndrome은 네 성분의 합만 본다. 업로드된 `syndrome_notes`도 일반 DDPM에서는 clean sampler에서도 syndrome이 0이 아니며, 그 안에 model bias, numerical discretization, deployment error가 섞인다고 정리한다.

Toy에서는 posterior mean이 닫힌 형태이고 $\hat{x}_0(t)$의 image가 $\mathrm{span}(U)$에 갇히므로 clean consistency가 이상적으로 성립한다. 그러나 CIFAR DDPM에서는 $f_\theta(x_t,t)$가 learned U-Net이므로, Tweedie martingale 성질이 정확히 성립하지 않는다. 업로드된 `siec_ecc_framing`도 이 martingale gap 때문에 $s_t$의 non-zero mean이 모델 자체에서 나올 수 있다고 정리한다.

따라서 문제는 “syndrome이 전혀 의미 없다”가 아니다. 문제는 **syndrome이 deployment error와 clean refinement를 분리하지 못한다**는 것이다.

### 5.2 원인 2: codeword를 data manifold로 두면 실제 오류를 놓칠 수 있다

초기 ECC framing은 다음과 같았다.

$$
\text{codeword space}=\mathcal{M}_{data}
$$

$$
H\approx P^\perp_{\mathcal{M}}
$$

이 경우 syndrome은 “이미지가 data manifold 밖으로 나갔는가”를 본다. 그러나 PTQ나 DeepCache의 품질 저하는 꼭 off-manifold artifact로 나타나는 것이 아니다. 색감 shift, texture 단순화, detail loss, class/mode drift처럼 여전히 자연 이미지처럼 보이지만 reference trajectory와 달라지는 형태일 수 있다.

ECC 비유로 보면, 한 valid codeword가 다른 valid codeword로 바뀐 상황이다.

$$
c_1\in\mathcal{C},\quad c_2\in\mathcal{C}
$$

$$
Hc_1=0,\quad Hc_2=0
$$

만약 $c_1$이 $c_2$로 바뀌면 parity check만으로는 오류를 감지하지 못한다.

Diffusion에서도 마찬가지다. “자연 이미지인가”와 “원래 clean/full-compute trajectory가 가야 할 이미지인가”는 다르다. 업로드된 `siec_ecc_framing`도 PTQ/DeepCache 오류가 manifold 위에서 class/mode를 옮기는 형태일 수 있고, 이 경우 syndrome-quality 상관성이 깨진다고 정리한다.

따라서 codeword를 data manifold의 한 점으로 두는 프레임은 S-IEC의 실제 목적에 비해 너무 좁다.

### 5.3 원인 3: Toy의 tangent/normal 논리가 CIFAR로 옮겨가지 않는다

Toy에서는 $\mathrm{span}(U)$가 명시적이고 posterior mean이 $UD_tU^\top x_t$로 주어진다. 따라서 어떤 성분이 syndrome에 보이고 어떤 성분이 사라지는지 해석적으로 증명할 수 있다.

그러나 CIFAR에서는 data manifold가 명시적이지 않다.

$$
\hat{x}_0(t)=f_\theta(x_t,t)
$$

이고, $f_\theta$의 image가 특정 subspace에 갇힌다는 보장이 없다. 업로드된 manifold note도 CIFAR에서는 learned U-Net output이 어떤 subspace에 갇히지 않기 때문에 on-manifold drift와 off-manifold error가 syndrome에서 분리되지 않는다고 정리한다.

따라서 toy에서 보인 tangent/normal 분리와 syndrome observability는 CIFAR에서 자동으로 유지되지 않는다. 이를 쓰려면 PCA, score Jacobian, VAE Jacobian 등으로 manifold tangent/normal을 추정하고 실험적으로 검증해야 한다. 그러나 이 방향은 main claim보다는 side ablation에 가깝다.

### 5.4 원인 4: 현재 구현이 DeepCache의 장점을 깨뜨렸다

S-IEC가 $r_t$를 계산하려면 $\hat{x}_0(t-1)$ lookahead가 필요하다.

$$
x_{t-1}^{tent}
=
\text{DDIM}(x_t,\hat{x}_0(t))
$$

$$
\hat{x}_0(t-1)
=
f_\theta(x_{t-1}^{tent},t-1)
$$

naive하게는 추가 U-Net call이 필요하다. 하지만 이 lookahead output은 다음 step에서 재사용할 수 있다. 업로드된 `syndrome_notes`도 correction이 trigger되지 않으면 $x_0(t-1)$을 다음 step에 재사용할 수 있고, 이 경우 실질 NFE overhead를 줄일 수 있다고 정리한다.

문제는 CIFAR 구현에서 lookahead가 재사용되지 않았고, DeepCache cache reuse도 차단되었다는 점이다. 업로드된 `siec_ecc_framing`은 CIFAR 구현에서 `prv_f=None`, `allow_cache_reuse=False`로 lookahead를 매 step 재계산하며 cache reuse를 차단해 NFE가 IEC 110에서 S-IEC p80 약 201로 증가했다고 정리한다.

즉 현재 S-IEC의 NFE 폭증은 “ECC-style syndrome이 본질적으로 비싸다”가 아니라, **speculative lookahead reuse와 DeepCache state handling이 구현되지 않은 문제**다.

### 5.5 원인 5: syndrome trigger가 random과 구별되지 않았다

사용자 보고서 기준으로, S-IEC p80 FID와 random p80 FID가 거의 같았다.

$$
\text{S-IEC p80 FID}=46.95
$$

$$
\text{Random p80 FID}=46.91
$$

이 값은 사용자 제공 실험 결과 기준이며, 독립적으로 원 로그를 확인한 값은 아니다. 하지만 이 결과가 맞다면 해석은 분명하다. 현재 $r_t$는 “어느 step에서 correction을 해야 하는가”를 random보다 잘 알려주지 못했다. 이는 앞의 원인 1과 연결된다. syndrome이 deployment error만 분리하지 못하면, threshold-based trigger가 random budget allocation과 비슷해질 수 있다.

---

## 6. 최종 실패 원인

현재 실패는 이론 하나만의 문제가 아니고, 구현 하나만의 문제도 아니다.

$$
\underbrace{\text{이론 문제}}_{\text{syndrome이 deployment error와 natural refinement를 분리하지 못함}}
\times
\underbrace{\text{구현 문제}}_{\text{lookahead reuse 실패, cache reuse 차단}}
=
\underbrace{\text{결과 문제}}_{\text{IEC에 dominated}}
$$

S-IEC가 처음 의도한 바는 옳았다. “오류가 큰 step에서만 correction을 실행하자”는 아이디어는 IEC 논문의 selective ablation과도 잘 맞는다. 그러나 현재 정의된 syndrome은 clean refinement, model bias, discretization, deployment error를 하나로 섞어 보았고, 현재 구현은 그 signal을 얻기 위해 DeepCache의 cache reuse를 깨뜨렸다.

따라서 현재 S-IEC는 다음 두 가지를 동시에 만족하지 못했다.

$$
\text{좋은 trigger signal}
$$

$$
\text{낮은 NFE overhead}
$$

이 둘 중 하나만 부족해도 어렵지만, 현재는 둘 다 부족한 상태다.

---

## 7. 다시 세워야 할 주장

### 7.1 버려야 할 주장

다음 주장은 버리는 것이 좋다.

> S-IEC는 data manifold를 codeword space로 보고, syndrome으로 off-manifold error를 검출하는 ECC decoder다.

이 주장은 세 이유로 위험하다.

첫째, 현재 구현의 syndrome은 $P^\perp_{\mathcal{M}}$가 아니다.

$$
\hat{s}_t=\hat{x}_0(t)-\hat{x}_0(t-1)
$$

이는 data manifold의 normal projection이 아니라 trajectory residual이다.

둘째, toy에서 syndrome이 감지하는 방향은 표준 off-manifold normal이 아니라 $\mathrm{span}(U)$, 즉 data direction이다. toy 코드의 normal/tangent naming이 표준과 반대라는 점 때문에 이 부분을 잘못 쓰면 논문 전체 논리가 뒤집힌다.

셋째, 실제 PTQ/DeepCache 오류는 off-manifold artifact만이 아니라 on-manifold drift일 수 있다. data manifold parity check는 이런 오류를 놓칠 수 있다.

따라서 “data manifold ECC”는 main claim으로 두기 어렵다.

### 7.2 새로 세울 주장

가장 정합한 주장은 다음이다.

> Diffusion sampling produces a temporally redundant trajectory of clean estimates and U-Net features. Efficient deployment methods such as DeepCache, PTQ, and CacheQuant corrupt this trajectory. We use ECC-inspired syndrome and reliability principles to allocate IEC correction budget to the most unreliable timesteps, improving the quality/NFE trade-off.

한국어로 쓰면 다음이다.

**Diffusion sampling은 $\hat{x}_0$와 U-Net feature의 시간축 중복성을 만든다. DeepCache와 PTQ는 이 중복 trajectory를 오염시키는 noisy channel이다. 우리는 ECC의 syndrome, reliability, erasure/noisy-symbol decoding 관점을 이용해 IEC correction budget을 더 잘 배분한다.**

이 claim에서 codeword는 data manifold 위의 이미지 한 장이 아니다.

$$
\text{codeword}
=
\mathbf{c}
=
(\hat{x}_0(T),\hat{x}_0(T-1),\ldots,\hat{x}_0(0))
$$

또는 DeepCache에서는 feature까지 포함한다.

$$
\text{codeword}
=
(h_T,h_{T-1},\ldots,h_0)
$$

channel은 efficient deployment method다.

$$
\text{channel}
=
\text{PTQ / DeepCache / CacheQuant}
$$

decoder는 S-IEC 단독이 아니라 IEC다.

$$
\text{syndrome}
\rightarrow
\text{where/how much IEC}
$$

이렇게 재정의하면 S-IEC는 “ECC 그 자체”가 아니라 **ECC-guided IEC controller**가 된다. 이쪽이 연구 목표와 더 잘 맞는다. 목표는 완전한 algebraic ECC 이식이 아니라, ECC 이론의 핵심 원리로 IEC의 성능/비용 trade-off를 개선하는 것이기 때문이다.

---

## 8. 재설계된 S-IEC

### 8.1 이름과 핵심 개념

명명은 그대로 **S-IEC**로 두는 것이 좋다. 새로운 이름을 붙이면 기존 문서와 실험 맥락이 불필요하게 갈라진다. 다만 의미를 바꾼다.

기존 의미:

$$
\text{S-IEC}
=
\text{standalone syndrome correction algorithm}
$$

재설계된 의미:

$$
\text{S-IEC}
=
\text{S-IEC controller}
$$

여기서 controller는 S-IEC의 동작 역할을 설명할 뿐 새 이름이나 약어가 아니다. 명칭은 끝까지 **S-IEC**로 통일한다.

핵심은 간단하다.

**기존 IEC는 DeepCache의 non-cached/full-compute step에 고정적으로 correction을 넣는다. 재설계된 S-IEC는 ECC식 syndrome으로 각 step 또는 interval의 reliability를 평가하고, correction budget을 더 필요한 곳에 배분한다.**

이제 S-IEC의 S는 “syndrome으로 직접 이미지를 고친다”가 아니라, **syndrome으로 IEC decoder를 제어한다**는 뜻이 된다.

### 8.2 DeepCache를 ECC 관점으로 다시 보기

DeepCache는 $N$ step 중 1회 full-compute, $N-1$회 partial-compute 구조다.

이를 ECC 관점으로 보면 다음과 같다.

$$
\text{full-compute step}
=
\text{high-reliability anchor symbol}
$$

$$
\text{cached step}
=
\text{low-cost but noisy symbol}
$$

DeepCache interval 하나를 일종의 code block으로 본다.

$$
\mathbf{y}_{k}
=
(y_{t_k},y_{t_k-1},\ldots,y_{t_{k+1}})
$$

여기서 $y_t$는 $\hat{x}_0(t)$ 또는 U-Net feature $h_t$다.

full-compute step은 신뢰도가 높은 anchor이고, cached step은 noisy symbol이다. S-IEC는 이 block 안에서 syndrome을 계산해 다음을 결정한다.

$$
\text{그냥 진행}
$$

$$
\text{다음 full-compute step에서 IEC}
$$

$$
\text{early refresh}
$$

$$
\text{force full-compute + IEC}
$$

이것이 ECC의 erasure/noisy-symbol decoding 원리와 맞는다. 모든 symbol을 동일하게 보지 않고, 신뢰도에 따라 correction resource를 배분한다.

---

## 9. Syndrome을 어떻게 다시 정의해야 하는가

### 9.1 현재 cheap syndrome

현재 구현의 cheap syndrome은 다음이다.

$$
r_t
=
\frac{\|\hat{x}_0(t)-\hat{x}_0(t-1)\|^2}{d}
$$

이 값은 data manifold parity check가 아니라, 가장 단순한 trajectory parity residual이다.

$$
H_t=
\begin{bmatrix}
I & -I
\end{bmatrix}
$$

$$
H_t
\begin{bmatrix}
\hat{x}_0(t)\\
\hat{x}_0(t-1)
\end{bmatrix}
=
\hat{x}_0(t)-\hat{x}_0(t-1)
$$

이 해석은 현재 코드를 살린다. 다만 raw $r_t$는 clean refinement와 deployment error를 섞어 보므로, 그대로 최종 syndrome으로 쓰기에는 약하다.

### 9.2 Calibrated trajectory syndrome

더 나은 방식은 clean/full-compute trajectory에서 timestep별 평균 drift를 보정하는 것이다.

$$
s_t^{\text{cal}}
=
Q_t^{-1/2}
\left(
\hat{x}_0(t-1)
-
\hat{x}_0(t)
-
\mu_t
\right)
$$

여기서

$$
\mu_t
=
\mathbb{E}_{clean}
[
\hat{x}_0(t-1)-\hat{x}_0(t)
]
$$

$$
Q_t
=
\mathrm{Cov}_{clean}
[
\hat{x}_0(t-1)-\hat{x}_0(t)
]
$$

이다.

이렇게 하면 clean trajectory에서 원래 생기는 자연 drift를 제거하고, deployment가 clean distribution에서 얼마나 벗어났는지를 본다.

$$
z_t=\|s_t^{\text{cal}}\|^2
$$

이 값은 raw $r_t$보다 더 ECC syndrome에 가깝다. 완전한 $Hc=0$은 아니지만, clean trajectory residual을 whitening한 **statistical parity check**가 된다.

### 9.3 Learned parity: 좋은 방향이지만 “linear로 충분”은 미검증

더 강하게는 $A_t,b_t$를 calibration한다.

$$
\hat{x}_0(t-1)
\approx
A_t\hat{x}_0(t)+b_t
$$

그러면 syndrome은 다음이다.

$$
s_t^{A}
=
Q_t^{-1/2}
\left(
\hat{x}_0(t-1)-A_t\hat{x}_0(t)-b_t
\right)
$$

이때 parity check는 명시적으로 쓸 수 있다.

$$
H_t
=
Q_t^{-1/2}
\begin{bmatrix}
-A_t & I
\end{bmatrix}
$$

$$
s_t^{A}
=
H_t
\begin{bmatrix}
\hat{x}_0(t)\\
\hat{x}_0(t-1)
\end{bmatrix}
-
Q_t^{-1/2}b_t
$$

이것은 classical ECC의 algebraic parity check는 아니다. 하지만 “diffusion trajectory의 statistical analog parity check”라고 부르기에는 충분히 정합하다.

다만 여기서 중요한 보강이 필요하다. **$A_t,b_t$를 per-timestep linear regression으로 학습하는 것은 자연스럽지만, linear predictor가 충분하다는 주장은 아직 근거가 없다.** Diffusion trajectory는 전 구간에서 같은 성격의 dynamics를 갖지 않는다. 초기 large $t$ 구간에서는 semantic layout과 coarse structure가 형성되고, 후반 small $t$ 구간에서는 texture와 detail refinement가 강하다. 같은 $\hat{x}_0$-space에서도 timestep별 변화가 단순 선형으로 충분히 설명된다고 가정하면 false positive가 커질 수 있다.

따라서 learned parity는 다음 순서로 검증해야 한다.

| 단계 | parity model | 목적 |
|---|---|---|
| 0 | raw difference $y_{t-1}-y_t$ | 현재 S-IEC baseline |
| 1 | mean drift $y_{t-1}-y_t-\mu_t$ | clean natural drift 제거 |
| 2 | scalar/diagonal affine $a_t\odot y_t+b_t$ | timestep별 크기 조정 |
| 3 | low-rank linear $A_t y_t+b_t$ | 주요 PCA subspace에서 선형 dynamics 추정 |
| 4 | piecewise linear by timestep region | early/mid/late dynamics 분리 |
| 5 | small nonlinear predictor $\phi_t(y_t)$ | linear residual이 큰 구간 보완 |
| 6 | feature-level predictor $\phi_t(h_t)$ | DeepCache error에 더 직접 대응 |

이 ablation의 기준은 세 가지다.

첫째, clean trajectory residual이 줄어야 한다.

$$
\mathbb{E}_{clean}\|s_t\|^2
\quad \text{감소}
$$

둘째, deployment와 clean의 residual 분리가 좋아져야 한다.

$$
\mathrm{AUROC}(z_t^{dep},z_t^{clean})
\quad \text{증가}
$$

셋째, 실제 IEC benefit 예측력이 좋아져야 한다.

$$
\mathrm{corr}(z_t,g_t)
\quad \text{증가}
$$

여기서 $g_t$는 해당 step에 IEC를 적용했을 때 reference error가 얼마나 줄었는지를 나타내는 correction gain이다.

따라서 논문에 쓸 수 있는 안전한 문장은 다음이다.

> We start from a calibrated first-order parity residual and progressively evaluate stronger learned parity predictors. Linear parity is treated as the simplest deployable approximation, not as an assumed sufficient model.

한국어로 쓰면 다음이다.

**우리는 linear parity가 충분하다고 가정하지 않는다. Raw difference, mean-drift calibration, diagonal/low-rank linear predictor, nonlinear predictor를 순차적으로 비교하고, 실제 IEC benefit을 가장 잘 예측하는 최소 복잡도 parity를 선택한다.**

이렇게 써야 한다. “small linear regression 정도면 충분하다”는 표현은 제거하는 것이 좋다.

### 9.4 IEC의 $\Delta_t$는 teacher syndrome으로 사용

IEC의 $\Delta_t$는 현재 $r_t$보다 correction 필요성과 더 직접적으로 연결될 가능성이 있다.

$$
\Delta_t
=
\left\|
f_\theta(x_{t-1}^{tent},t)-f_\theta(x_t,t)
\right\|
$$

하지만 $\Delta_t$는 계산 비용이 크다. 또한 $f_\theta(x_{t-1}^{tent},t)$는 다음 step $t-1$의 실제 denoising output으로 재사용하기 어렵다. 반면 $r_t$ 계산에 필요한 $\hat{x}_0(t-1)$는 speculative lookahead로 다음 step에 재사용 가능하다.

따라서 $\Delta_t$를 매번 inference signal로 쓰는 대신, offline calibration에서 teacher syndrome으로 사용한다.

$$
r_t \rightarrow \Delta_t
$$

$$
z_t \rightarrow \Delta_t
$$

$$
z_t \rightarrow \text{IEC benefit}
$$

즉 재설계된 S-IEC는 cheap syndrome $r_t$ 또는 $s_t^{cal}$로 expensive $\Delta_t$와 실제 IEC benefit을 예측한다.

이 구조가 핵심이다.

$$
\text{cheap reusable syndrome}
\rightarrow
\text{expensive IEC-aligned syndrome의 proxy}
\rightarrow
\text{correction budget allocation}
$$

---

## 10. 재설계된 S-IEC의 실제 동작

### 10.1 Calibration 단계

먼저 clean/full-compute reference trajectory를 수집한다.

$$
\{x_t^{ref},\hat{x}_0^{ref}(t),h_t^{ref}\}_{t=1}^{T}
$$

그리고 efficient deployment trajectory도 같은 $x_T$ seed로 수집한다.

$$
\{x_t^{dep},\hat{x}_0^{dep}(t),h_t^{dep}\}_{t=1}^{T}
$$

각 step마다 다음 값을 기록한다.

$$
r_t
=
\frac{\|\hat{x}_0(t)-\hat{x}_0(t-1)\|^2}{d}
$$

$$
s_t^{cal}
=
Q_t^{-1/2}
(
\hat{x}_0(t-1)-\hat{x}_0(t)-\mu_t
)
$$

$$
s_t^{A}
=
Q_t^{-1/2}
(
\hat{x}_0(t-1)-A_t\hat{x}_0(t)-b_t
)
$$

$$
\Delta_t
=
\|f_\theta(x_{t-1}^{tent},t)-f_\theta(x_t,t)\|
$$

$$
g_t
=
\text{IEC 적용 전후 error 감소량}
$$

여기서 $g_t$는 실제 correction gain이다.

$$
g_t
=
\|x_t^{dep}-x_t^{ref}\|
-
\|x_t^{IEC}-x_t^{ref}\|
$$

가능하면 FID까지 step별로 직접 볼 수는 없으므로, per-step reference error reduction 또는 final FID contribution proxy를 사용한다.

### 10.2 Syndrome reliability score

S-IEC는 각 timestep의 reliability를 계산한다.

$$
\rho_t
=
a\cdot z_t
+
b\cdot \widehat{\Delta}_t
+
c\cdot \text{cache\_age}_t
+
d\cdot \text{timestep\_weight}_t
$$

여기서

$$
z_t=\|s_t^{cal}\|^2
$$

또는

$$
z_t=\|s_t^{A}\|^2
$$

이고,

$$
\widehat{\Delta}_t
=
\phi(z_t,\text{cache\_age}_t,t)
$$

이다. $\phi$는 처음부터 복잡한 학습 모델일 필요가 없다. calibration table, isotonic regression, small linear regression 정도에서 시작할 수 있다. 다만 이 역시 “충분하다”가 아니라 **ablation으로 검증할 후보**다.

### 10.3 Correction policy

Reliability score에 따라 correction 정책을 나눈다.

$$
\rho_t < \tau_1
\Rightarrow
\text{do nothing}
$$

$$
\tau_1 \le \rho_t < \tau_2
\Rightarrow
\text{defer correction to next full-compute step}
$$

$$
\tau_2 \le \rho_t < \tau_3
\Rightarrow
\text{IEC at scheduled full-compute step}
$$

$$
\rho_t \ge \tau_3
\Rightarrow
\text{early refresh + IEC}
$$

이 방식이 기존 IEC와 다른 점은 correction을 fixed schedule로만 하지 않는다는 것이다. ECC의 syndrome처럼, 관측된 trajectory inconsistency가 correction 위치와 강도를 정한다.

---

## 11. Lookahead reuse와 speculative commit

S-IEC가 성공하려면 NFE overhead를 반드시 줄여야 한다. 현재 구현처럼 매 step lookahead를 새로 계산하고, cache reuse를 끄면 실패한다.

올바른 구현은 speculative execution이다.

1. step $t$에서 현재 $\hat{x}_0(t)$를 계산한다.
2. $x_{t-1}^{tent}$를 만든다.
3. lookahead로 $\hat{x}_0(t-1)$를 계산한다.
4. syndrome이 작으면 이 lookahead output과 cache state를 다음 step에 그대로 commit한다.
5. syndrome이 크면 correction을 적용하고, 필요한 경우 해당 step만 recompute하거나 rollback한다.

이때 trigger되지 않은 step에서는 lookahead가 낭비되지 않는다.

$$
\text{extra NFE}
\approx
\#\{\text{triggered recompute steps}\}
$$

현재 CIFAR 구현이 실패한 핵심 이유는 이 구조가 없었기 때문이다. 업로드된 `siec_ecc_framing`은 현재 CIFAR 구현이 매 step lookahead를 재계산하고 cache reuse를 차단해 S-IEC NFE가 크게 증가했다고 정리한다.

---

## 12. S-IEC가 “그냥 anomaly detection”이 되지 않으려면

단순히

$$
z_t>\tau
\Rightarrow
\text{IEC 호출}
$$

만 하면 anomaly detection에 가깝다. ECC의 의미를 살리려면 syndrome이 세 가지 역할을 해야 한다.

첫째, **error detection**이다.

$$
z_t \text{가 크면 trajectory corruption 가능성이 크다}
$$

둘째, **error location**이다. 단일 spike인지, interval drift인지, refresh 직후 불안정인지 구분해야 한다.

$$
s_t \text{ spike}
\Rightarrow
\text{local cached step error}
$$

$$
s_t,s_{t-1},s_{t-2} \text{ drift}
\Rightarrow
\text{cache stale / accumulated drift}
$$

셋째, **correction strength allocation**이다.

$$
\text{small syndrome}
\Rightarrow
\text{skip}
$$

$$
\text{medium syndrome}
\Rightarrow
\text{defer or light correction}
$$

$$
\text{large syndrome}
\Rightarrow
\text{IEC / early full refresh}
$$

이 세 가지가 들어가면 S-IEC는 단순 anomaly gate가 아니라 ECC-inspired decoding controller가 된다.

---

## 13. 실험 설계

### 13.1 반드시 재현해야 할 IEC baseline

먼저 IEC 원문 baseline을 정확히 맞춰야 한다.

| 방법 | 의미 |
|---|---|
| DeepCache only | correction 없는 efficient baseline |
| DeepCache + IEC all non-cached steps | IEC 원문 기본 DeepCache 결합 |
| DeepCache + IEC 1/10, 1/20 selective | 원문 selective baseline |
| DeepCache + random same-budget IEC | syndrome 없는 budget control |
| DeepCache + redesigned S-IEC | 제안법 |
| DeepCache + oracle IEC | 상한선 |

IEC 원문은 quantization 기반 방법에서는 IEC를 every timestep에 적용하지만, DeepCache와 CacheQuant에서는 non-cached timestep에만 적용한다고 설명한다. 이 baseline과 비교해야 S-IEC가 실제로 IEC의 correction budget allocation을 개선했는지 판단할 수 있다.

### 13.2 Syndrome이 IEC benefit을 예측하는지 검증

각 timestep에서 다음 값을 기록한다.

$$
r_t,\quad
z_t^{cal}=\|s_t^{cal}\|^2,\quad
z_t^{A}=\|s_t^{A}\|^2,\quad
\Delta_t,\quad
g_t
$$

그리고 다음 상관을 본다.

$$
r_t \leftrightarrow \Delta_t
$$

$$
z_t^{cal} \leftrightarrow \Delta_t
$$

$$
z_t^{A} \leftrightarrow \Delta_t
$$

$$
r_t \leftrightarrow g_t
$$

$$
z_t^{cal} \leftrightarrow g_t
$$

$$
z_t^{A} \leftrightarrow g_t
$$

가장 중요한 것은 $z_t \rightarrow g_t$다. 즉 calibrated 또는 learned syndrome이 실제 IEC correction gain을 예측해야 한다.

판정 기준은 다음처럼 잡을 수 있다.

| 결과 | 해석 |
|---|---|
| $z_t$가 $g_t$와 높은 Spearman 상관 | S-IEC의 선택적 correction 가능 |
| $z_t$가 $\Delta_t$는 예측하지만 $g_t$는 못 예측 | IEC residual proxy는 되지만 correction budget signal은 약함 |
| $z_t$가 clean/deploy를 잘 분리하지만 final FID 개선 없음 | anomaly detection은 되지만 decoding signal은 약함 |
| $z_t$가 random과 비슷 | 현재 S-IEC 방향 폐기 또는 feature-level syndrome 필요 |

### 13.3 Learned parity ablation

특히 9.3의 우려를 반영해, learned parity는 반드시 별도 ablation으로 분리한다.

| 모델 | 검증 질문 |
|---|---|
| raw difference | 현재 구현이 왜 실패했는가 |
| mean-drift calibration | natural refinement 제거만으로 충분한가 |
| diagonal affine | timestep별 signal scale 차이를 보정하면 좋아지는가 |
| low-rank linear | 주요 trajectory subspace에서 선형 parity가 유효한가 |
| piecewise linear | early/mid/late dynamics 분리가 필요한가 |
| nonlinear predictor | 선형 residual이 false positive를 만들 때 개선되는가 |
| feature-level predictor | image-space보다 DeepCache feature-space가 더 좋은가 |

여기서 중요한 것은 최종 complexity가 아니라 Pareto다.

$$
\text{syndrome quality}
\quad vs \quad
\text{calibration cost}
\quad vs \quad
\text{inference overhead}
$$

linear가 가장 좋으면 좋다. 그러나 linear가 충분하다는 주장은 실험 결과가 나온 뒤에만 가능하다.

### 13.4 NFE-matched trigger 비교

같은 NFE budget에서 다음을 비교한다.

| 방법 | 비교 목적 |
|---|---|
| periodic IEC | schedule baseline |
| random IEC | trigger 의미 검증 |
| raw $r_t$-trigger | 현재 S-IEC |
| calibrated $z_t$-trigger | improved syndrome |
| learned parity trigger | 재설계된 S-IEC |
| $\Delta_t$-trigger | expensive teacher / upper trigger |
| two-stage trigger | cheap syndrome + selective expensive confirmation |
| oracle trigger | reference error 기반 상한 |

S-IEC가 random과 periodic보다 좋아야 한다. 그리고 $\Delta_t$-trigger 또는 oracle trigger에 가까워질수록 claim이 강해진다.

### 13.5 Decoder vs gate 비교

같은 syndrome을 쓰되 두 가지를 비교한다.

| 방식 | 의미 |
|---|---|
| syndrome gate + IEC 호출 | anomaly detection에 가까움 |
| syndrome pattern 기반 correction strength/early refresh 결정 | ECC-inspired controller |
| syndrome pattern 기반 correction vector 추정 | ECC decoder에 가장 가까움 |

최소한 두 번째까지는 보여야 한다. 단순 gate만으로는 “time-series anomaly detection과 무엇이 다른가”라는 비판을 피하기 어렵다.

### 13.6 Feature-level syndrome

DeepCache 오류는 image-space $\hat{x}_0$보다 feature-space에서 먼저 나타날 수 있다. 따라서 후속 확장으로 다음을 고려한다.

$$
s_t^{feat}
=
\|h_t^{cached}-\hat{h}_t^{anchor}\|
$$

여기서 $h_t^{cached}$는 cached step의 feature이고, $\hat{h}_t^{anchor}$는 full-compute anchor에서 예측한 feature다.

DeepCache가 high-level feature reuse로 계산을 줄이는 방법이므로, feature-level syndrome은 image-space syndrome보다 DeepCache error에 더 직접적일 수 있다.

---

## 14. 최종 논문 claim

최종 claim은 다음처럼 써야 한다.

> Existing IEC improves efficient diffusion sampling by correcting approximation errors, but its correction budget is allocated by a fixed schedule. We redesign S-IEC as an ECC-inspired controller that treats the clean-estimate trajectory as a redundant analog code and efficient deployment methods as noisy channels. S-IEC uses calibrated trajectory syndromes to estimate timestep reliability and allocate IEC correction budget, improving the quality/NFE Pareto over periodic, random, and fixed selective IEC baselines.

한국어로 정리하면 다음이다.

**기존 IEC는 efficient diffusion sampler의 approximation error를 줄이지만, correction을 언제 쓸지는 고정 schedule에 의존한다. 재설계된 S-IEC는 diffusion의 $\hat{x}_0$ trajectory를 중복성을 가진 analog code로 보고, DeepCache/PTQ를 noisy channel로 해석한다. Calibrated trajectory syndrome으로 각 timestep의 reliability를 추정하고, IEC correction budget을 오류 가능성이 높은 step에 배분해 같은 NFE에서 더 좋은 품질 또는 같은 품질에서 더 낮은 NFE를 달성한다.**

이 claim은 “완벽한 ECC를 diffusion에 이식했다”가 아니다. 하지만 연구 목표에는 더 정확히 맞는다.

$$
\text{ECC 이론}
\Rightarrow
\text{syndrome, reliability, error location, correction budget allocation}
$$

$$
\text{IEC}
\Rightarrow
\text{actual decoder / correction operator}
$$

$$
\text{재설계된 S-IEC}
\Rightarrow
\text{ECC-guided IEC}
$$

---

## 15. 최종 요약

지금까지의 S-IEC는 다음 문제가 있었다.

$$
\text{codeword를 data manifold로 둠}
\Rightarrow
\text{실제 syndrome과 불일치}
$$

$$
\text{toy의 tangent/normal 용어 혼동}
\Rightarrow
\text{무엇을 감지하는지 잘못 해석}
$$

$$
\hat{x}_0(t)-\hat{x}_0(t-1)
\Rightarrow
\text{natural refinement와 deployment error 분리 실패}
$$

$$
\text{lookahead reuse 미구현}
\Rightarrow
\text{NFE 폭증}
$$

$$
\text{linear parity 충분성 가정}
\Rightarrow
\text{미검증 claim, false positive 위험}
$$

따라서 고쳐야 할 방향은 다음이다.

$$
\text{codeword}
=
\text{data manifold}
\quad \text{폐기}
$$

$$
\text{codeword}
=
\text{clean/full-compute diffusion trajectory}
\quad \text{채택}
$$

$$
\text{S-IEC}
=
\text{standalone ECC decoder}
\quad \text{폐기}
$$

$$
\text{S-IEC}
=
\text{ECC-guided IEC controller}
\quad \text{채택}
$$

$$
\text{raw syndrome}
=
\|\hat{x}_0(t)-\hat{x}_0(t-1)\|^2
\quad \text{보완}
$$

$$
\text{calibrated trajectory syndrome}
=
\|Q_t^{-1/2}(\hat{x}_0(t-1)-\hat{x}_0(t)-\mu_t)\|^2
\quad \text{1차 채택}
$$

$$
\text{learned parity syndrome}
=
\|Q_t^{-1/2}(\hat{x}_0(t-1)-A_t\hat{x}_0(t)-b_t)\|^2
\quad \text{ablation 후 채택 여부 결정}
$$

$$
\Delta_t
=
\text{expensive IEC-aligned teacher syndrome}
$$

$$
r_t,z_t
=
\text{cheap inference-time syndrome}
$$

최종 결론은 다음 한 문장으로 압축된다.

**기존 S-IEC의 실패는 ECC 아이디어 자체의 실패라기보다, ECC를 data manifold parity check로 잘못 해석하고, syndrome을 deployment error와 natural refinement를 분리하지 못하는 raw trajectory residual로 사용했으며, 구현상 lookahead reuse를 하지 않아 DeepCache의 비용 이점을 깨뜨린 데서 왔다. 해결책은 S-IEC를 “data-manifold ECC decoder”가 아니라 “clean diffusion trajectory code를 이용해 IEC correction budget을 배분하는 S-IEC controller”로 재정의하고, calibrated trajectory syndrome과 검증된 learned parity를 통해 IEC correction budget을 배분하는 것이다.**

여기서 learned parity는 좋은 방향이지만, **linear predictor가 충분하다는 것은 아직 claim이 아니라 실험 질문**이다. 따라서 논문에는 “linear로 충분하다”가 아니라, **“최소 복잡도의 parity predictor를 ablation으로 찾고, 그 predictor가 IEC benefit을 예측함을 보인다”** 로 써야 한다.

---

## 참고 출처 및 내부 문서

- `syndrome_notes_20260427.md`
- `siec_ecc_framing_20260427.md`
- `siec_manifold_geometry_20260427.md`
- DeepCache: Accelerating Diffusion Models for Free, CVPR 2024
- IEC: Iterative Error Correction for Efficient Diffusion Models
