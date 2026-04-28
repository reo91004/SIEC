## 최종 판정

핵심 문장인 **“S-IEC가 수정하는 것은 B뿐이다. Syndrome은 trajectory 내부의 inconsistency를 측정하지, 어떤 이미지로 수렴하는지를 판단하지 않는다”** 는 거의 맞다. 더 정확히 쓰면 다음 문장이 된다.

**S-IEC는 ‘동일한 생성 문제에서 효율화된 deployment sampler가 clean/reference sampler와 다르게 만든 국소적 trajectory inconsistency’를 잡으려는 방법이지, 최종 이미지가 의미적으로 맞는지, 더 좋은 이미지인지, data manifold 위의 올바른 지점으로 가는지를 직접 판단하는 방법은 아니다.**

따라서 S-IEC를 강하게 주장하려면 “ECC × Diffusion으로 manifold 위 codeword에 사영한다”보다, 우선은 **“deployment-error-induced local inconsistency를 syndrome으로 검출하고, lookahead consensus로 완화한다”** 쪽이 더 안전하다. 업로드된 문서도 S-IEC의 syndrome을 (\hat{x}_0(t)-\hat{x}_0(t-1))로 정의하고, 이것을 “현재 스텝과 다음 스텝에서 바라본 clean image 추정의 불일치”로 설명한다. 즉 syndrome은 본질적으로 **local consistency check**다. 

## 1. Syndrome이 실제로 보는 것

S-IEC의 syndrome은 다음 값이다.

[
\hat{s}_t = \hat{x}_0(t)-\hat{x}_0(t-1)
]

여기서 (\hat{x}_0(t))는 현재 (x_t)에서 얻은 clean estimate이고, (\hat{x}_0(t-1))은 한 스텝 tentative update 후 lookahead로 얻은 clean estimate다. 업로드된 문서에서도 동일하게, ECC의 (s=Hr)에 대응되는 것이 아니라 정확히는 “consecutive Tweedie consistency check”라고 해석된다. 

이걸 deployment error 관점으로 쓰면 더 명확하다.

[
\tilde{\hat{x}}_0(t)=\hat{x}^{\star}_0(t)+e_t
]

[
\tilde{\hat{x}}_0(t-1)=\hat{x}^{\star}*0(t-1)+e*{t-1}^{\text{look}}
]

그러면 deployed syndrome은 대략 다음처럼 분해된다.

[
\tilde{s}_t
===========

\underbrace{\hat{x}^{\star}*0(t)-\hat{x}^{\star}*0(t-1)}*{\text{clean baseline inconsistency}}
+
\underbrace{(e_t-e*{t-1}^{\text{look}})}_{\text{deployment-induced local inconsistency}}
]

이 식이 중요하다. S-IEC가 직접 보는 것은 (e_t) 자체가 아니라 **인접한 두 추정 사이에서 deployment error가 얼마나 일관되지 않은가**다. 따라서 PTQ나 DeepCache가 (\hat{x}_0(t))를 오염시키더라도, 그 오염이 timestep을 따라 거의 같은 방향으로 일관되게 들어가면 syndrome은 작을 수 있다. 반대로 최종 이미지는 괜찮아도 (\hat{x}_0(t))와 lookahead (\hat{x}_0(t-1))가 크게 흔들리면 syndrome은 커질 수 있다.

이 점 때문에 “S-IEC는 B만 수정한다”는 말은 맞지만, 더 좁게는 **B 중에서도 local inconsistency로 드러나는 B만 수정한다**가 정확하다. 같은 (x_T)에서 PTQ/DeepCache가 (\hat{x}_0(t))를 오염시키는 상황은 S-IEC의 주요 타깃이 맞다. 하지만 그 오염이 **trajectory-consistent bias**라면 syndrome decoder로는 잘 안 보인다.

## 2. normal/tangent를 꼭 고려해야 하나

두 가지 답이 있다. **S-IEC를 어떤 주장으로 포장할지에 따라 다르다.**

첫째, 논문 주장을 “deployment error correction”으로 낮추면 normal/tangent 분해는 필수가 아니다. 이 경우 필요한 것은 tangent/normal 증명이 아니라, 다음 세 가지 실험이다.

[
r_t=|\hat{x}_0(t)-\hat{x}_0(t-1)|^2/d
]

이 값이 clean보다 deployed에서 커지는가, 실제 deployment error (|x_t^{\text{deploy}}-x_t^{\text{ref}}|)와 상관되는가, 그리고 correction 이후 FID/CLIP/NFE Pareto가 좋아지는가. 업로드된 `siec_ecc_framing` 문서도 실험 A를 syndrome-error correlation test로 두고, Spearman (\rho)가 낮으면 framing이 위협받는다고 정리한다. 

둘째, 논문 주장을 “ECC × Diffusion, 즉 data manifold codeword로의 syndrome decoding”으로 강하게 유지하려면 normal/tangent를 고려해야 한다. 이 경우 syndrome이 단순히 temporal inconsistency를 보는 것이 아니라, **codeword space에서 벗어난 성분을 검출한다**는 주장이 들어가기 때문이다. 그런데 CIFAR/DDPM에서는 data manifold가 명시적이지 않고, UNet posterior mean도 exact projection이 아니므로 toy Gaussian의 tangent/normal 이론이 그대로 적용되지 않는다. 업로드된 문서도 CIFAR에서는 posterior mean을 학습된 UNet으로 근사하고, manifold가 implicit하며, syndrome이 UNet bias·timestep discretization·stochasticity를 함께 담는다고 지적한다. 

따라서 현재 상황에서 가장 안전한 결론은 이것이다.

**B만 수정한다고 주장할 거면 normal/tangent를 억지로 증명하지 않아도 된다. 하지만 “ECC 이론을 diffusion manifold에 결합했다”고 주장할 거면 normal/tangent 또는 그 대체물인 manifold-aware decomposition 실험이 필요하다.**

## 3. “tangent noise = syndrome inconsistency”라고 부르면 위험하다

현재 문서들에서 가장 위험한 부분은 용어다. 업로드된 `syndrome_notes`에 따르면 toy 코드에서는 `normal space = span(U)`를 데이터가 놓인 방향으로, `tangent space = span(U)^\perp`를 noise-only direction으로 부른다. 그런데 표준 미분기하에서는 보통 반대다. 데이터 manifold가 (M=\mathrm{span}(U))라면 (T_xM=\mathrm{span}(U))가 tangent이고, (N_xM=\mathrm{span}(U)^\perp)가 normal이다. 문서도 이 naming이 표준 관례와 반대라고 명시한다. 

그래서 “tangent noise가 많고 syndrome이 tangent noise를 잡는다”라고 쓰면 reviewer가 바로 물고 늘어질 수 있다. 지금 S-IEC의 syndrome은 **data-manifold tangent error**를 직접 측정하는 것도 아니고, **off-manifold normal error**를 직접 측정하는 것도 아니다. 정확한 이름은 다음에 가깝다.

[
\text{temporal clean-estimate inconsistency}
]

또는

[
\text{Tweedie-consistency violation}
]

또는

[
\text{deployment-induced trajectory inconsistency}
]

toy Gaussian에서는 특정 subspace 성분에 대해 syndrome 민감도가 이론적으로 보일 수 있다. 하지만 CIFAR/DDPM에서는 그 대응이 보장되지 않는다. 업로드된 문서도 일반 DDPM에서는 clean sampler syndrome이 0이라는 보장이 없고, normal/tangent 분해도 분석적으로 불가능하다고 정리한다. 

## 4. normal noise를 무시해도 되는가

“B만 본다”는 관점에서는 **normal noise를 별도 분해하지 않아도 된다**. 실제로 필요한 것은 PTQ/DeepCache가 만든 deployment error가 syndrome을 키우는지다. 이 경우 normal/tangent 대신 다음 두 항만 보면 된다.

[
\Delta r_t = r_t^{\text{deploy}} - r_t^{\text{clean}}
]

[
\Delta x_t = x_t^{\text{deploy}} - x_t^{\text{ref}}
]

즉, “오류가 normal인가 tangent인가”보다 “오류가 clean/reference trajectory와의 local inconsistency로 드러나는가”가 더 직접적이다.

하지만 “normal noise를 아예 고려하지 않아도 된다”는 말은 과하다. 이유는 세 가지다.

첫째, off-manifold error가 자연 denoising으로 사라지는 경우라면 S-IEC가 굳이 고쳐도 이득이 작다. 업로드된 toy 설명에서도 어떤 성분은 DDIM update에서 (\sigma_{t-1}/\sigma_t) 비율로 자연 감쇠될 수 있다고 정리한다. 

둘째, on-manifold semantic drift는 syndrome이 작을 수 있지만 품질에는 치명적일 수 있다. 예를 들어 DeepCache가 feature reuse 때문에 texture나 shape mode를 조금 바꾸는 경우, 두 인접 (\hat{x}_0)는 서로 일관될 수 있다. 이 경우 syndrome은 작고 최종 FID/CLIP은 나빠질 수 있다. 업로드된 framing 문서도 PTQ/DeepCache error가 manifold 위에서 클래스나 모드를 옮기는 형태로 나타날 수 있고, 이 경우 syndrome-quality 상관성이 깨진다고 지적한다. 

셋째, “ECC × Diffusion”이라고 부르려면 codeword space 바깥으로 튄 오류를 syndrome이 검출한다는 구조가 필요하다. 그 구조를 보이려면 normal/tangent 또는 score-Jacobian/PCA 기반 manifold decomposition이 필요하다. 최근 diffusion manifold 연구도 score Jacobian의 spectral structure가 data manifold의 tangent/normal 방향을 반영할 수 있다는 관점으로 진행되고 있다. ([arXiv][1])

그래서 정리하면 다음이다.

**실용적 S-IEC 실험에서는 normal/tangent를 생략해도 된다. 이론적 ECC-manifold 주장에서는 생략하면 안 된다.**

## 5. S-IEC의 NFE 오버헤드는 본질인가

절반은 본질이고, 절반은 구현 문제다.

본질적인 부분은 syndrome 계산 자체다. (\hat{x}*0(t))만 있으면 일반 diffusion step은 진행된다. 하지만 syndrome을 계산하려면 tentative (x*{t-1})에서 다시 (\hat{x}_0(t-1))를 얻어야 한다. 업로드된 문서도 이 과정을 `x0_t = unet(xt,t)`, `x0_t1 = unet(xt_tent,t-1)`로 쓰며, naive하게는 추가 UNet forward가 필요하다고 정리한다. 

구현 문제인 부분은 **lookahead reuse**다. toy 쪽 의도는 trigger되지 않으면 lookahead로 계산한 (\hat{x}_0(t-1))를 다음 step의 current prediction으로 재사용해서 net overhead를 거의 없애는 구조다. 반면 CIFAR 구현에서는 매 step lookahead를 재계산하고, `allow_cache_reuse=False`로 cache reuse도 차단되어 NFE가 IEC 110에서 S-IEC p80 약 201로 폭증했다고 문서가 지적한다. 

따라서 현재 결과 기준으로는 다음 판단이 맞다.

**현재 CIFAR S-IEC의 NFE 폭증은 S-IEC 이론의 필연적 비용이라고 단정할 수 없다. 하지만 lookahead reuse가 구현되지 않은 상태에서는 S-IEC가 IEC보다 효율적이라고 주장할 수 없다.**

특히 DeepCache 환경에서는 lookahead reuse가 더 까다롭다. DeepCache는 인접 denoising step 사이의 feature redundancy를 이용해 high-level feature를 캐시하고 low-level feature만 싸게 업데이트하는 training-free 가속 방식이다. ([CVF Open Access][2]) 그런데 S-IEC lookahead가 실제 sampler의 cache state와 다른 방식으로 계산되면, syndrome은 같은 채널의 두 관측이 아니라 **cache 상태가 다른 두 관측의 차이**를 보게 된다. 업로드된 문서도 이를 lookahead domain shift라고 부르고, `allow_cache_reuse=False`일 때 두 추정이 다른 채널의 관측이 될 수 있다고 지적한다. 

즉 DeepCache + S-IEC에서 진짜 중요한 구현 조건은 이것이다.

**lookahead를 계산하되, 그 결과를 다음 step에 재사용하고, DeepCache의 feature/cache state도 일관되게 commit 또는 rollback해야 한다.**

이게 안 되면 S-IEC는 syndrome 계산을 위해 DeepCache의 장점을 깨뜨리는 구조가 된다.

## 6. “IEC를 DeepCache에서 10 step마다 수행”하는 ablation은 정합한가

정합하긴 하다. 다만 **비교 목적을 정확히 써야 한다.**

IEC 논문 자체는 efficient diffusion model에서 test-time iterative refinement로 오류 전파를 완화하는 방법이고, 오류가 denoising 과정에서 누적될 수 있다는 문제를 다룬다. ([arXiv][3]) DeepCache는 diffusion의 순차 denoising에서 high-level feature를 재사용해 계산을 줄이는 방법이므로, DeepCache로 생긴 approximation error에 IEC를 periodic하게 붙이는 ablation은 자연스럽다. ([CVF Open Access][2])

하지만 “IEC every 10 steps”를 S-IEC와 비교할 때는 반드시 **NFE-matched**로 해야 한다. S-IEC가 syndrome check 때문에 매 step 추가 forward를 쓰고, IEC는 10 step마다만 correction을 한다면, 두 방법은 같은 비용의 알고리즘이 아니다. 그 상태에서 S-IEC가 조금 좋은 FID를 얻어도 “그냥 더 많은 NFE를 쓴 것”이라는 비판을 피하기 어렵다.

따라서 ablation은 다음 세 가지로 나눠야 정합하다.

| 비교                                    | 의미                                          | 판정            |
| ------------------------------------- | ------------------------------------------- | ------------- |
| DeepCache only                        | 가속 baseline                                 | 필수            |
| DeepCache + periodic IEC              | 단순 schedule 기반 correction baseline          | 필수            |
| DeepCache + S-IEC, same NFE           | syndrome trigger가 periodic trigger보다 좋은지 검증 | 핵심            |
| DeepCache + S-IEC, same trigger rate  | trigger 위치 선택 효과 검증                         | 보조            |
| DeepCache + S-IEC, raw implementation | 실제 구현 비용 보고                                 | 보조, 단독 주장은 약함 |

따라서 현재 질문의 답은 다음이다.

**DeepCache에서 10 step마다 IEC를 수행하는 ablation 자체는 정합하다. 다만 S-IEC와의 비교가 NFE-matched가 아니면, S-IEC의 효율성을 주장하는 ablation으로는 정합하지 않다.**

## 7. S-IEC 주장을 어떻게 바꾸는 게 좋은가

현재 가장 안전한 논문 framing은 이것이다.

**S-IEC는 ECC의 syndrome 개념에서 영감을 받은 test-time consistency decoder다. Efficient deployment sampler가 만든 local Tweedie-consistency violation을 감지하고, lookahead consensus correction으로 그 위반을 줄인다. 이 방법은 final image semantics를 직접 판별하지 않으며, 따라서 목표는 “좋은 이미지인지 판단”이 아니라 “clean/reference sampler 대비 local trajectory inconsistency를 줄이는 것”이다.**

이 framing이면 normal/tangent 증명에 실패해도 논문이 산다. 반대로 “ECC × Diffusion = manifold codeword decoder”라고 강하게 쓰면, CIFAR에서 manifold decomposition을 실제로 보여야 한다. 업로드된 manifold 문서도 PCA 또는 score Jacobian으로 tangent space를 추정하고, 실제 deployment error를 (P^\perp/P^\parallel)로 분해하는 실험 G/H/I를 제안한다. 

## 8. 지금 당장 해야 할 검증 순서

가장 먼저 할 것은 tangent/normal 논쟁이 아니라 **S-IEC가 정말 B를 보는지** 확인하는 것이다.

1단계는 clean/reference trajectory와 deployed trajectory를 같은 (x_T), 같은 sampler schedule로 저장하는 것이다. 각 step에서 (r_t^{clean}), (r_t^{deploy}), (|x_t^{deploy}-x_t^{ref}|), (|\hat{x}_0^{deploy}-\hat{x}_0^{ref}|)를 기록한다. 여기서 (r_t^{deploy}-r_t^{clean})이 PTQ/DeepCache에서 유의하게 커지지 않으면 S-IEC의 핵심 가정이 무너진다.

2단계는 Spearman correlation이다. (r_t)와 실제 deployment error 사이의 (\rho)를 family별로 본다. PTQ에서는 높고 DeepCache에서는 낮게 나올 가능성이 있다. 이 경우 S-IEC는 universal decoder가 아니라 **family-dependent deployment-error decoder**가 된다.

3단계는 lookahead reuse fix다. 현재처럼 lookahead가 매 step 추가 NFE를 먹으면 S-IEC는 periodic IEC와의 Pareto 경쟁에서 불리하다. 업로드된 문서도 실험 D를 최우선으로 두고, NFE가 IEC 수준 ±10% 안으로 들어와야 framing 비교가 의미 있다고 정리한다. 

4단계는 NFE-matched periodic IEC 비교다. S-IEC가 200 NFE를 쓴다면 periodic IEC도 200 NFE 근처로 맞춰야 한다. 반대로 S-IEC lookahead reuse 후 110~120 NFE라면 periodic IEC도 같은 NFE로 맞춘다. 여기서 S-IEC가 이겨야 “syndrome이 단순 schedule보다 낫다”는 말이 가능하다.

5단계는 선택적으로 manifold-aware 실험이다. ECC-manifold claim을 끝까지 밀 거면 PCA 또는 score-Jacobian으로 (P^\perp(\hat{s}_t))와 (P^\parallel(\hat{s}_t))를 나눠야 한다. TAC-Diffusion 같은 quantized diffusion correction 계열도 timestep별 quantization error를 직접 보정하는 강한 비교군이므로, PTQ setting에서는 반드시 비교 대상으로 남겨야 한다. ([ECVA][4])

## 최종 문장

현재 상태에서 S-IEC의 가장 정합한 주장은 **“selective IEC”도 아니고, 곧바로 “manifold ECC decoder”도 아니다.**

가장 방어 가능한 주장은 다음이다.

**S-IEC는 efficient deployment sampler가 만든 (\hat{x}_0)-trajectory의 국소 불일치를 syndrome으로 검출하고, lookahead consensus로 줄이는 test-time correction이다. 이 방법은 final image semantics를 직접 평가하지 않으며, normal/tangent decomposition은 핵심 알고리즘의 필수 조건이 아니라 ECC-manifold framing을 강하게 주장하기 위한 추가 검증 조건이다. 현재 CIFAR 구현의 NFE 폭증은 S-IEC의 본질이라기보다 lookahead reuse와 DeepCache state handling이 제대로 구현되지 않은 데서 생긴 비용일 가능성이 크다. 따라서 S-IEC를 주장하려면 먼저 lookahead reuse를 고치고, NFE-matched periodic IEC와 비교해야 한다.**

[1]: https://arxiv.org/html/2510.05509v2?utm_source=chatgpt.com "Discovering Riemannian Metric for Diffusion Models"
[2]: https://openaccess.thecvf.com/content/CVPR2024/papers/Ma_DeepCache_Accelerating_Diffusion_Models_for_Free_CVPR_2024_paper.pdf?utm_source=chatgpt.com "DeepCache: Accelerating Diffusion Models for Free"
[3]: https://arxiv.org/html/2511.06250v1?utm_source=chatgpt.com "Test-Time Iterative Error Correction for Efficient Diffusion ..."
[4]: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08312.pdf?utm_source=chatgpt.com "Timestep-Aware Correction for Quantized Diffusion Models"
