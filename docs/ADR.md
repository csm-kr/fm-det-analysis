# Architecture Decision Records

> **이 문서가 답하는 질문**: 왜 *detectron2 제거 / flow matching 전환 / 추적 도구 / 시드 정책*을 골랐는가? *무엇을·왜* 는 [PRD.md](./PRD.md), 평가 프로토콜은 [EVAL_PROTOCOL.md](./EVAL_PROTOCOL.md).

## 철학
- **재현성 최우선** — baseline (DiffusionDet 본 repo 재현치 AP 46.2) 을 못 맞추면 FM 전환·ablation 결과가 의미 없다.
- **단순 baseline 부터** — 무리한 모델 재설계 전, "재구현 → 동치 결과 → 한 변수만 바꾼 ablation" 사이클로 진행.
- **평가 지표 변경 금지** — COCO val 해상도 800~1333, iter step 4 / num eval boxes 500 (DiffusionDet 본 repo 재현 조건) 고정. (CLAUDE.md CRITICAL)

> **맥락 인용 규칙**: 각 ADR 의 "맥락" 줄은 [PRD.md](./PRD.md) / [DATA_CARD.md](./DATA_CARD.md) / [EVAL_PROTOCOL.md](./EVAL_PROTOCOL.md) 등의 섹션을 명시적으로 가리킨다.

---

## ADR-001: detectron2 의존성 제거 — PyTorch 2.5 단독 재구현
- **상태**: accepted
- **날짜**: 2026-05-21
- **맥락**: [PRD.md#목표](./PRD.md), [PRD.md#범위-외](./PRD.md). 레퍼런스 `DiffusionDet/` 는 detectron2 위에 빌드돼 있고, 그 추상화(매처/Box·Instances 클래스/registry 등) 위에서는 박스 회귀·sampler·loss 의 내부 동작을 한 줄씩 따라가기 어렵다. fm-det 의 1차 목적은 **이해와 내공 형성** 이므로 추상화 레이어를 걷어내는 것이 본질.
- **결정**: detectron2 의존성 0. 재구현 코드베이스는 PyTorch 2.5 + torchvision + pycocotools 만으로 작성. matcher / loss / sampler / inference NMS 까지 직접 구현. (당초 결정 시점에는 `src/` 폴더 가정 — 이후 [ADR-005](#adr-005-hydra-config--루트-평탄-디렉터리-구조) 에서 루트 평탄 배치로 변경.)
- **대안**:
  - (기각) detectron2 그대로 사용 — 코드량 적지만 내부 이해 기회 상실, FM 전환 시 추상화와 충돌.
  - (기각) MMDetection 으로 교체 — 의존성 종류만 바꾸는 셈, 같은 문제 반복.
- **결과**:
  - 이점: 모든 컴포넌트의 동작이 가시화 → ablation 의 자유도 확보, **CRITICAL "detectron2 import 금지"** 의 자연스러운 강제.
  - 제약: 초기 구현 비용 증가, 분산 학습(DDP) 직접 작성. baseline 매칭(AP 46.2 ± 0.5)까지 phase 가 길어짐.

## ADR-002: diffusion → flow matching 단계적 전환 (전체 모델 ablation)
- **상태**: accepted
- **날짜**: 2026-05-21
- **맥락**: [PRD.md#성공-기준](./PRD.md), [EXPERIMENTS.md](./EXPERIMENTS.md). DiffusionDet 의 핵심은 박스 좌표를 노이즈로 보고 점차 denoising 해 가는 sampler 와 그에 맞춘 loss. flow matching 은 score-based diffusion 의 대안 학습 프레임이고, sampler 와 loss 외에 head·decoder·encoder 의 step 의존성에도 영향을 줄 수 있어 **전체 모델을 ablation 대상**으로 둔다.
- **결정**: 단계적 전환 로드맵 (각 단계는 별도 phase + 별도 `runs/{tag}/`):
  1. **재구현 단계 (P0)** — diffusion sampler + loss (DiffusionDet 동치) — baseline 매칭.
  1-a. **메커니즘 진단 단계 (P0a)** — 재구현 코드 위에서 DiffusionDet 의 핵심 노브(signal scale / box renewal / iter step / num eval boxes / NMS 임계) 를 변경하며 영향 정량화. **FM 전환 시작 전 최소 5개 진단 행이 [EXPERIMENTS.md](./EXPERIMENTS.md) 의 진단 카탈로그에 채워져야 한다** — fm-det 의 "내공 형성" 목표의 실증 지표.
  2. **FM 1차 전환 (P1)** — sampler + loss 만 flow matching 으로 (ODE / vector field). 다른 모듈 동일.
  3. **FM 2차 전환 (P2)** — matcher / head 도 FM 속성에 맞게 재설계 ablation.
  4. **모듈 ablation (P3)** — encoder backbone 교체, decoder 단순화, head deep/shallow ablation.
- **대안**:
  - (기각) 일시 전환 — 한꺼번에 모든 모듈 교체. 결과 변동이 어디서 왔는지 분리 불가.
  - (기각) FM 만 도입, 모델 구조 동결 — 사용자의 "내공 형성" 목적과 충돌.
- **결과**:
  - 이점: 각 단계의 `ΔAP` 가 모듈 효과를 직접 측정. 학습 노트로서 가치.
  - 제약: 학습 GPU 시간 다배. EXPERIMENTS.md 의 표 관리가 핵심 산출물이 됨.

## ADR-003: 실험 추적 = TensorBoard 기준 + W&B 병행 (학습 동반)
- **상태**: accepted
- **날짜**: 2026-05-21
- **맥락**: [ARCHITECTURE.md#실험-추적](./ARCHITECTURE.md). DiffusionDet 본 repo 가 TensorBoard 를 쓰므로 재현 단계에서 동일 도구가 일관성 유리. 한편 ablation 이 다수가 되면 TB 의 다중 실험 비교가 불편 → W&B 도입. 사용자가 W&B 첫 도입.
- **결정**: 둘 다. TB 는 모든 `runs/{id}/tb_logs/` 에 기본 기록 (외부 의존 0, 오프라인). W&B 는 병행 — 초기에는 `wandb offline` 으로 로컬 기록 → 익숙해진 뒤 `wandb online` sync. W&B 학습은 [EXPERIMENTS.md](./EXPERIMENTS.md) "W&B 학습 노트" 섹션에 누적.
- **대안**:
  - (기각) TB 단독 — ablation 다수 비교 불편.
  - (기각) W&B 단독 — 학습 곡선 가운데 외부 SaaS 장애 시 백업 없음, 첫 도입자 부담.
  - (기각) MLflow — 자체 호스팅 컨테이너 추가 부담.
- **결과**:
  - 이점: 재현성(TB) + 비교 편의(W&B) 양립. 사용자가 W&B 도구 학습 기회.
  - 제약: 추적 코드 2 곳 동기화 필요 — `src/utils/logging.py` 에서 단일 추상화로 호출 분기 (양쪽 모두 호출).

## ADR-005: Hydra config + 루트 평탄 디렉터리 구조
- **상태**: accepted
- **날짜**: 2026-05-21
- **맥락**: [ARCHITECTURE.md#디렉터리-구조](./ARCHITECTURE.md), [EXPERIMENTS.md](./EXPERIMENTS.md). ablation 표 한 행 = config 1개 = `runs/{tag}/` 1개 의 1:1 매핑이 ablation 중심 프로젝트의 관리 단위. 평탄 yaml + 새 파일 복사는 파일 수 폭증, 단순 YAML + 자체 merger 는 sweep / CLI override 지원 없음. **사용자 결정**: Hydra 채택 + `src/` 폴더 제거 후 루트 평탄 배치.
- **결정**:
  - **Hydra 1.3** 도입. `configs/` 는 그룹별 yaml (`model/`, `data/`, `train/`, `loss/`) + 진입점 yaml (`train.yaml`, `eval.yaml`) + ablation 한 행을 담는 `experiment/{tag}.yaml`.
  - Hydra 의 `hydra.run.dir` 을 `runs/{now:%Y%m%d-%H%M}-{tag}` 로 override → 기존 `runs/{ts}-{tag}/` 컨벤션 (EXPERIMENTS.md) 과 자동 일치.
  - `src/` 폴더 폐지. 모듈을 루트 평탄 배치 — `./datasets/`, `./models/`, `./losses/`, `./evals/`, `./utils/`. 진입점도 루트 — `./train.py`, `./eval.py`, `./infer.py`.
  - `scripts/` 는 하네스 정본 (`execute.py`, `test_execute.py`, ml 헬퍼) **만** 둔다. 사용자 진입점은 `scripts/` 에 두지 않는다.
- **대안**:
  - (기각) 단순 YAML + 자체 30줄 merger — 의존성 0 이지만 sweep / CLI override 없음. ablation 다수 시 불편.
  - (기각) ablation 마다 yaml 전체 복사 — 파일 수 폭증, 변경점 추적 어려움.
  - (기각) `src/` 유지 — 한 단계 더 깊어 사용자가 "단순화" 로 거부.
  - (기각) 진입점을 `scripts/` 에 공존 — 사용자가 "scripts 는 하네스만" 으로 명시 거부.
- **결과**:
  - 이점: ablation 표 1행 ↔ `configs/experiment/{tag}.yaml` 1파일 ↔ `runs/{tag}/` 1개 의 1:1 매핑. CLI 에서 `python train.py +experiment=coco-fm1-sampler-cfm seed=42` 만으로 실행.
  - 제약: Hydra 학습 곡선 (W&B 와 함께 첫 도입 — EXPERIMENTS.md "Hydra 학습 노트" 누적). Python 의존성 추가 (`hydra-core`).
  - 부수: `from datasets...` `from models...` 같은 루트 패키지 import — `pip install -e .` 또는 `PYTHONPATH=.` 필요. `pyproject.toml` 도입 phase 에서 처리.

---

## ADR-004: 시드 고정 + ablation 변수 격리 정책
- **상태**: accepted
- **날짜**: 2026-05-21
- **맥락**: [PRD.md#재현성-요건](./PRD.md), [EVAL_PROTOCOL.md#통계적-유의성](./EVAL_PROTOCOL.md), [EXPERIMENTS.md#시드-정책](./EXPERIMENTS.md). 학습/연구 단계에서 비교 신뢰도가 결과 해석의 전부.
- **결정**:
  - 모든 학습/평가 스크립트 (`scripts/train.py`, `scripts/eval.py`) 는 `--seed` 인자 의무. 없으면 시작 차단 (argparse `required=True`).
  - 시드 고정 범위: `random`, `numpy`, `torch`, `torch.cuda.manual_seed_all`, `torch.backends.cudnn.deterministic=True`, `cudnn.benchmark=False`. (속도 손해 감수)
  - ablation 시 변경 변수는 1개 — config 와 코드 변경 모두 단일 변수로 격리. EXPERIMENTS.md 의 ablation 표 각 행은 "변경한 1 변수" 컬럼 명시.
  - 데이터 버전 변경 시 `data/CHANGELOG.md` 한 줄 의무.
- **대안**:
  - (기각) 베스트 에포트 — 우연한 차이가 모듈 효과로 보일 위험.
  - (기각) deterministic=True 해제 (속도 우선) — 비교 신뢰도 손상.
- **결과**:
  - 이점: ablation `ΔAP` 의 해석이 단일 변수 효과로 한정 가능.
  - 제약: 학습 속도 일부 손해 (cudnn deterministic). 그만한 가치.

## 관련 문서
- 무엇을·왜: [PRD.md](./PRD.md)
- 시스템 구조: [ARCHITECTURE.md](./ARCHITECTURE.md)
- 실험 운영: [EXPERIMENTS.md](./EXPERIMENTS.md)
- 데이터셋: [DATA_CARD.md](./DATA_CARD.md)
- 모델: [MODEL_CARD.md](./MODEL_CARD.md)
- 평가: [EVAL_PROTOCOL.md](./EVAL_PROTOCOL.md)
