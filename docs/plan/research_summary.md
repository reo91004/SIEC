# S-IEC (Selective Iterative Error Correction) 연구 및 실험 가이드

이 문서는 S-IEC 논문 프로젝트(5월 초 마감)의 핵심 아이디어와 함께, **성공적인 논문 작성을 위해 당장 실행해야 할 구체적인 실험 세팅과 액션 플랜**을 정리한 핸드북입니다.

---

## 1. 연구의 핵심: 왜 S-IEC인가?
기존의 Diffusion 모델은 속도 향상을 위해 딥캐시나 양자화를 쓰는데, 이로 인한 화질 저하를 막기 위해 **IEC(Iterative Error Correction)**를 도입했습니다. 하지만 IEC는 매 스텝마다 보정을 수행해 **연산량(NFE)이 낭비**됩니다.

**S-IEC의 제안:**
- **Syndrome Score**라는 에러 심각도 지표를 도입.
- 점수가 특정 기준치(`tau`)를 넘을 때, 즉 **진짜 보정이 필요한 순간에만 선택적으로 고쳐서 효율을 극대화**합니다.

> **[핵심 인사이트] 단순 화질 경쟁이 아닙니다.**
> 현재 테스트 결과, 단순 FID(화질 지표)만 보면 S-IEC가 기존 IEC를 완벽히 이기지 못합니다. 따라서 논문의 셀링 포인트는 **"비슷한 화질을 훨씬 적은 계산량으로 달성(가성비)"**하거나, **"열악한 환경에서 모델이 무너지는 것을 더 잘 방어(강건성)"**한다는 점을 증명하는 데 있습니다.

### 📚 핵심 용어 정리
- **Diffusion(디퓨전) 모델**: 노이즈를 제거하며 이미지를 생성하는 AI 모델. (화질은 높으나 연산량이 많음)
- **배포 오차 (Distribution Error)**: 생성 속도를 높이기 위한 가속/압축(양자화, DeepCache 등) 기법으로 인해 발생하는 이미지 품질 저하 현상.
- **IEC (Iterative Error Correction)**: 배포 오차를 매 스텝마다 보정하는 기존 기술. 연산량(NFE)이 과도하게 낭비됨.
- **S-IEC (Selective IEC)**: 본 연구의 제안. 에러가 심각할 때(Syndrome Score가 특정 `tau`를 넘을 때)만 선택적으로 보정하여 효율을 극대화.
- **NFE (Number of Function Evaluations)**: 연산량 및 계산 비용 지표. (낮을수록 속도가 빠름)
- **FID (Fréchet Inception Distance)**: 이미지 생성 품질 지표. (낮을수록 고화질)
- **Pareto Frontier (파레토 프론티어)**: 화질(FID)과 연산량(NFE) 사이의 트레이드오프에서 달성할 수 있는 최적의 효율 곡선.

---

## 2. 주요 실험 및 구체적 세팅 (용성님 담당)

### 📊 실험 4: 가성비 증명 (Compute-Quality Tradeoff / Pareto Frontier)
무조건 보정하는 것보다 S-IEC가 '비용 대비 화질' 측면에서 우월함을 증명합니다. **즉, S-IEC가 기존 IEC 대비 동일한 화질(FID)을 더 적은 연산량(NFE)으로 달성하여 최적의 효율(Pareto frontier) 위에 있음을 보이고자 합니다.**
- **실험 환경**: CIFAR-10, 100 steps, W8A8 + DeepCache
- **비교 대상**: 
  1. No correction (보정 안함)
  2. IEC (원저자 방식, 전부 보정)
  3. Naive always-on (무조건 보정)
  4. Random / Uniform trigger (동일 계산량 하에서 단순 비교군)
  5. **S-IEC (tau percentile: 30, 50, 60, 70, 80, 90, 95 스윕)**
- **목표 증명**: `tau` 조절에 따라 연산량(NFE)과 화질(FID)이 부드럽게 조절(Control Knob)되며, 동일 연산량일 때 Random/Uniform보다 화질이 좋고, 궁극적으로 NFE-FID 그래프에서 Pareto frontier를 형성함을 2-panel Plot으로 시각화합니다.

### 🛡️ 실험 5: 강건성 증명 (Robustness & Distribution Error)
실제 배포 환경에서 겪을 수 있는 다양한 에러 상황에서 S-IEC가 얼마나 잘 버티는지 증명합니다. **즉, 배포 오차(Distribution Error)의 강도가 커질수록 S-IEC가 기존 IEC 대비 얻는 이득이 단조 증가(monotonically increasing)한다는 가정을 실제 모델에서 보이고자 합니다.**
- **에러 강도 세팅 순서 (약함 ➡️ 강함)**:
  1. `fp16` (가장 약한 에러)
  2. `W8A8` 양자화
  3. `DeepCache` (replicate_interval=10)
  4. `W4A8` 양자화
  5. `DeepCache` (공격적 세팅: interval=20 or 50)
  6. `CacheQuant` (W4A8 + DeepCache, 가장 강한 에러)
- **목표 증명**: 에러 환경(배포 오차)이 가혹해질수록 (위 순서대로 갈수록) 기존 베이스라인(IEC) 대비 S-IEC의 성능 방어력이 돋보이며, 그 이득의 폭이 단조 증가함을 시각화합니다.

---

## 3. 개발 및 실행 안전 수칙
> [!WARNING]
> 원저자의 핵심 코어 로직(`mainddpm/`, `siec_core/` 등)은 함부로 직접 수정하지 마세요.

- 필요한 모든 데이터 추출, NFE 계산, FID 파싱 등은 **`IEC/experiments/` 폴더 아래에 별도의 Wrapper 스크립트**를 만들어서 해결합니다.
- 새로운 실험을 돌리기 전에는 항상 `--dry-run`이나 작은 샘플로 파일이 잘 로드되는지 확인한 뒤 50K full run을 돌립니다.

---

## 4. 당장 실행해야 할 Action Item (우선순위)

1. **결과 정리**: 기존 로그와 `iec_vs_siec_fid_results.txt`를 기반으로 현재까지의 데이터를 표로 재정리합니다.
2. **Missing Tau Schedule 생성**: 기존 `pilot_scores_nb.pt`를 활용하여 부족한 tau 캘리브레이션 파일(p30, p60, p95)을 우선 생성합니다.
3. **실험 4 Wrapper 작성**: 코어 루프 수정 없이, 결과를 모아 CSV/JSON으로 저장하고 Plot을 그리는 `real_04_tradeoff.py` 래퍼를 작성합니다.
4. **실험 5 Inventory 체크**: 6가지 에러 세팅에 대해 당장 실행 가능한 파일이 준비되어 있는지 확인하고, 누락된 스크립트부터 챙깁니다.
