# Model Card

> **이 문서가 답하는 질문**: fm-det 의 *어떤 변종*이 *어떤 학습 환경*에서 *어떤 성능*을 냈는가? fm-det 은 단일 모델이 아니라 단계적 FM 전환 + ablation 의 **변종 집합** 이다.

## 모델 개요
- **모델군 이름**: fm-det (Flow-Matching DETection, DiffusionDet 의 detectron2-free 재구현 → flow matching 전환 변종군)
- **레퍼런스**: DiffusionDet (Chen et al., arXiv:2211.09788) — `DiffusionDet/` 폴더 (읽기 전용).
- **학습 데이터**: COCO 2017 **또는** PASCAL VOC 07+12 (각 변종은 한 데이터셋에서만 학습·평가) — [DATA_CARD.md](./DATA_CARD.md) 참고.
- **평가 프로토콜**: [EVAL_PROTOCOL.md](./EVAL_PROTOCOL.md) 참고. **변경 금지 (CRITICAL)**.
- **학습 환경**: PyTorch 2.5 + CUDA 12.1 + Python 3.11. GPU = 단일 또는 다중 (DDP 직접 구현, 정확한 GPU 모델은 각 run 의 `runs/{id}/config.yaml` 에 기록).
- **공통 학습 설정** (DiffusionDet 본 repo 와 동일 — 변경 시 ADR 신설 의무):
  - optimizer = AdamW, lr = 2.5e-5, weight_decay = 1e-4
  - total iter = 450K, step lr decay at (47 epoch, 57 epoch), total = 61 epoch
  - batch size = 16, signal scale = 2.0
  - inference: iter step = 4, num eval boxes = 500

## 변종 카탈로그

각 변종은 [EXPERIMENTS.md](./EXPERIMENTS.md) 의 ablation 표 한 행과 대응. 본 카드는 결론 후보(또는 reference) 변종만 정리.

### M0. `repro-baseline` (재구현 baseline)
- **차이**: DiffusionDet 본 repo 의 detectron2 의존성 제거 + 루트 평탄 구조 재구현. 알고리즘 동치.
- **목표**: AP 46.2 ± 0.5 (본 repo 재현치 매칭). 미달 시 후속 phase 전부 무효.
- **상태**: 학습 전 (P0 phase 에서 산출 예정).

### M1. `fm1-sampler-cfm` (FM 1차 전환)
- **차이**: M0 대비 sampler + loss 를 conditional flow matching (CFM) 으로 교체. 다른 모듈(encoder/decoder/head/matcher) 동일.
- **변경 변수**: 1개 (sampler+loss 쌍).
- **상태**: P1 phase 산출 예정.

### M2. `fm2-*` (FM 2차 전환 변종)
- **차이**: M1 대비 matcher 또는 head 를 FM 속성(연속 시간 벡터장)에 맞게 재설계.
- **상태**: P2 phase 산출 예정.

### M3. `abl-*` (모듈 ablation 변종)
- **차이**: encoder backbone 교체 (e.g. ResNet50 → Swin-T), decoder 단순화, head 깊이 조정 등.
- **상태**: P3 phase 산출 예정.

### M_final. `final-*` (3-시드 평균 검증된 최종 변종)
- 결론 후보 변종을 시드 3개(42, 17, 2024) 평균 ± std 로 재실행.
- 본 카드의 "평가 결과 요약" 표는 M_final 변종으로 채운다.

## 의도된 용도
- **1차 사용**: **연구자(본인) 학습/실험용**. 운영 서비스가 아님.
- **입력**: COCO/VOC 도메인의 RGB 이미지 (resolution 800~1333 짧은 변).
- **출력**: 박스 좌표(xyxy) + 클래스 + confidence 리스트.

## 권장하지 않는 용도
- 운영 서비스 배포 — 본 모델군은 학습/연구용. 응답시간/가용성 보장 없음.
- 도메인 외 입력 (의료/위성/저조도/X-ray) — 학습 분포 밖. PRD 의 "범위 외" 와 일치.
- 실시간 시스템 — FPS 측정은 보조 지표일 뿐 응답시간 보장 없음.
- 오픈셋(open-vocabulary) detection — 학습 클래스 80/20 종에 한정.

## 메커니즘 진단 결과 (P0a — DiffusionDet 의 실증적 이해)
> 본 표는 **P0a 진단 실험 종료 후 채운다.** fm-det 의 "내공 형성" 목표의 실증 지표 — 채워지지 않으면 P1 FM 전환 보류. 카탈로그·진단 가설은 [EXPERIMENTS.md#P0a-메커니즘-진단-실험-카탈로그](./EXPERIMENTS.md).

| 진단 tag | 변경 변수 | 측정값 (AP 또는 시각화) | base 대비 | 한 줄 결론 (DiffusionDet 의 동작에 대해 알게 된 것) |
|---------|----------|----------------------|----------|----------------------------------------|
| `coco-diag-signal-scale` | scale ∈ {1.0, 2.0, 4.0} | _(미산출)_ | | |
| `coco-diag-box-renewal` | on / off | _(미산출)_ | | |
| `coco-diag-iter-step` | ∈ {1,2,4,8} | _(미산출)_ | | |
| `coco-diag-num-boxes` | ∈ {100,300,500,1000} | _(미산출)_ | | |
| `coco-diag-nms-iou` | ∈ {0.3,0.5,0.7} | _(미산출)_ | | |

## 평가 결과 요약
> 본 표는 **학습/평가 산출 후 채운다.** 채우기 전에는 비워두며, 임의 추정치 절대 입력 금지 (LLM_GUIDE Simplicity First).

| 변종 | COCO AP | AP50 | AP75 | VOC mAP@0.5 | FPS | 시드 평균 (n=3) |
|------|---------|------|------|-------------|-----|----------------|
| M0 `repro-baseline` | _(미산출)_ | | | | | |
| M1 `fm1-sampler-cfm` | _(미산출)_ | | | | | |
| M2 (선정 후 기입) | _(미산출)_ | | | | | |

자세한 지표·슬라이스 분석은 [EVAL_PROTOCOL.md](./EVAL_PROTOCOL.md), 학습 곡선은 TensorBoard / W&B.

## 알려진 한계 / 실패 모드 (사전 가설 — 학습 후 실증·수정)
- DiffusionDet 의 가설: 박스 좌표 노이즈 분포가 학습/추론 일치할 때 안정 — FM 전환 시 vector field 의 분포 어울림 정도가 성능을 결정.
- 모듈 ablation 시 encoder backbone (ResNet → Swin) 교체로 인한 학습률·scheduler 최적값 변경 가능 — CRITICAL "단일 변수" 위반 위험. 별도 hyperparameter 재탐색 phase 필요.

## 모니터링 / 운영
- **운영 모니터링 없음** — 연구 프로젝트.
- **재학습 트리거 = 새 ablation 가설** — EXPERIMENTS 의 표가 갱신될 때마다 새 phase.

## 관련 문서
- 결정 근거: [ADR.md](./ADR.md)
- 실험 운영·ablation 표: [EXPERIMENTS.md](./EXPERIMENTS.md)
- 데이터셋: [DATA_CARD.md](./DATA_CARD.md)
- 평가 프로토콜: [EVAL_PROTOCOL.md](./EVAL_PROTOCOL.md)
