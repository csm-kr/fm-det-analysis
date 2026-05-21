# PRD: fm-det

> **이 문서가 답하는 질문**: 이 모델은 *무엇을 입력 받아 무엇을 출력*하며, *어떤 지표로 성공*을 판정하는가? 데이터셋 상세는 [DATA_CARD.md](./DATA_CARD.md), 평가는 [EVAL_PROTOCOL.md](./EVAL_PROTOCOL.md), 시스템 구조는 [ARCHITECTURE.md](./ARCHITECTURE.md).

## 목표
DiffusionDet 을 detectron2 없이 재구현하고, diffusion → flow matching 으로 단계적으로 전환하며 encoder/decoder/head/sampler/loss 모듈을 ablation 하여 detection 에서 실제 유효한 구성 요소를 정리한다 — 동시에 diffusion·flow matching 의 내공을 형성한다.

## 사용자 / 호출 시나리오
- 호출 주체: **연구자(본인) 수동** — 사람이 직접 학습·평가·ablation 을 트리거하는 학습/연구용 코드베이스.
- 호출 빈도: 학습은 GPU 가용 시 수일 단위(450K iter 기준), 평가·시각화는 학습 종료 후 수회.
- 다운스트림 운영 서비스 / 외부 호출자 없음.

## 입력 / 출력
- **학습 입력**: COCO 2017 **또는** PASCAL VOC 07+12 (이미지 + 박스 어노테이션). 두 데이터셋은 **각각 독립적으로 학습·평가**한다 — 합본 학습하지 않는다. 자세한 분포는 [DATA_CARD.md](./DATA_CARD.md).
- **추론 입력**: RGB 이미지 한 장 (resolution 800~1333 짧은 변 기준 — DiffusionDet 과 동일 조건, 변경 금지).
- **추론 출력**: 박스 좌표(`xyxy`) + 클래스 + confidence 리스트 (COCO 80개 / VOC 20개 클래스).
- 예외 케이스: 학습 분포 외 도메인(의료/위성/저조도)은 보장하지 않음 — `Out of scope` 참고.

## 범위 외 (Out of scope)
지원하지 않을 입력·태스크 — 학습 분포 밖이라 성능 미보장.
- 실시간 추론의 응답시간 보장 (FPS 측정은 보조 지표일 뿐, 운영 환경에서의 응답시간은 보장하지 않는다)
- 다중 카메라 / 비디오 시계열 detection
- segmentation / keypoint / 3D box (박스만 다룬다)
- 오픈셋(open-vocabulary) detection — 학습 클래스 80/20 종에 한정
- 모델 경량화 (quantization / pruning / distillation) — ablation 의 단순 비교 가능성을 해침

## 평가 지표
- **주 지표**: COCO `val2017` 의 `mAP@[0.5:0.95]` (AP). 목표 — DiffusionDet 본 repo 재현치(AP 46.2) ± 0.5 이내로 baseline 매칭 후, FM 전환 ablation 각 step 의 AP 변동(`ΔAP`) 을 기록.
- 보조 지표: AP50 / AP75 / APs / APm / APl, FPS (단일 GPU), 학습 수렴 step 수.
- **COCO·VOC 동등 위상** (보조 아님): VOC 학습 변종도 `mAP@0.5` (PASCAL VOC 컨벤션) 로 동일 절차의 ablation 표를 가진다. 단, baseline 출처가 다르다 — COCO 만 DiffusionDet 비교가 가능하다.
- 비교 baseline (변경 금지):
  - **COCO**: DiffusionDet 원 논문 (iter step 4 / num eval boxes 500 / AP 46.8) + 본 레퍼런스 repo 재현치 (`DiffusionDet/` — AP 46.2)
  - **VOC**: DiffusionDet 결과 없음 → 본 프로젝트 내 자체 baseline (P0 VOC 재구현치) 만 비교 기준
- 상세는 [EVAL_PROTOCOL.md](./EVAL_PROTOCOL.md).

## 데이터
- 학습 데이터: **각 데이터셋 단독** — COCO 2017 `train` (≈118K 장) **또는** PASCAL VOC 07+12 `trainval` (≈16K 장).
- 평가 데이터: 학습한 데이터셋에 대응 — COCO 학습 → `val2017` (5K 장), VOC 학습 → VOC2007 `test` (≈5K 장).
- 데이터셋 cross-evaluation (COCO 학습 → VOC 평가 등) 은 범위 외 (도메인·클래스 mismatch).
- PII / 민감정보: 모두 공개 학술 데이터셋 — 별도 마스킹 정책 불필요.
- 데이터 위치는 `data/` (git ignore), 다운로드·전처리 절차는 [DATA_CARD.md](./DATA_CARD.md).

## 재현성 요건
- 시드: **`--seed` 인자 의무** (random / numpy / torch / cuda 모두 고정) — CLAUDE.md CRITICAL.
- 데이터 버전 관리: 공개 데이터셋 + 다운로드 스크립트 + SHA256 체크섬으로 고정 (DVC 미사용).
- 코드 버전: 모든 `runs/{ts}-{tag}/` 디렉터리에 `git rev` 자동 저장 (ARCHITECTURE.md 학습 흐름 §3).

## 성공 기준
1. **재현 단계 (P0)** — `DiffusionDet/` 레퍼런스를 detectron2 없이 재구현한 코드베이스(루트 평탄 모듈)가 COCO val AP 46.2 ± 0.5 달성. (baseline 매칭)
1-a. **메커니즘 진단 단계 (P0a — 재구현과 병행)** — DiffusionDet 핵심 메커니즘(signal scale / box renewal / iter step / num eval boxes / NMS 임계)을 변경하며 **실험적으로** 영향을 측정. 결과를 [EXPERIMENTS.md](./EXPERIMENTS.md) 의 진단 표에 기록 — 이 단계의 산출물이 후속 FM 전환의 비교 기준선이 된다. "이해했다" 가 아니라 "표에 N행이 채워졌다" 가 완료 정의.
2. **FM 전환 단계** — sampler+loss 를 flow matching 으로 교체했을 때 COCO val AP 가 baseline 대비 -2.0 AP 이내(또는 향상). 학습 step 수·FPS 변화 기록.
3. **모듈 ablation 단계** — encoder / decoder / head 각 모듈에 대한 ablation 표가 [EXPERIMENTS.md](./EXPERIMENTS.md) 에 채워지고, 각 행의 `ΔAP` 와 한 줄 결론이 명시됨.
4. **문서 산출물** — 위 3 단계가 ADR(전환 결정) + EXPERIMENTS(ablation 표) + MODEL_CARD(최종 fm-det 변종 카드) 에 모두 정합적으로 기록됨.

> **검수**: PRD 본문이 핵심 13 View 를 빠짐없이 다루는지 [PRD_VIEW.md](./PRD_VIEW.md) 체크리스트로 직접 확인하세요.
