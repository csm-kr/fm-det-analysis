# 아키텍처

> **이 문서가 답하는 질문**: *코드(src/) ↔ 데이터(data/) ↔ 실험 결과(runs/)*가 어떻게 연결되며, 학습이 어떻게 흘러가는가? 데이터셋·모델·평가 상세는 별도 문서 ([DATA_CARD.md](./DATA_CARD.md) / [MODEL_CARD.md](./MODEL_CARD.md) / [EVAL_PROTOCOL.md](./EVAL_PROTOCOL.md)).

## 시스템 구조
```
[코드]                                    [저장소]
┌────────────────────────┐                ┌──────────────────────┐
│ src/data    (로더)     │ ◀── 읽음 ───── │ data/  (데이터셋)    │
│   ↓                    │                └──────────────────────┘
│ src/models  ({모델})   │                ┌──────────────────────┐
│   ↓                    │ ◀── 읽음 ───── │ configs/{name}.yaml  │
│ src/train   (학습 루프)│                └──────────────────────┘
│   ↓                    │                ┌──────────────────────┐
│ src/eval    (메트릭)   │ ──── 기록 ───▶ │ runs/{ts}-{tag}/     │
│   ↓                    │                │  config.yaml         │
│ src/infer   (추론)     │                │  metrics.csv         │
└────────────────────────┘                │  checkpoints/        │
        │                                 │  logs/               │
        └─── 선택 외부 ──▶ {W&B/TB/MLflow}└──────────────────────┘
```
> 데이터셋·모델 상세는 별도 문서 (DATA_CARD.md / MODEL_CARD.md). 평가는 EVAL_PROTOCOL.md.

## 디렉터리 구조
```
src/
├── data/              # 데이터 로더, 전처리, 증강
├── models/            # 모델 정의
├── train/             # 학습 루프, 옵티마이저, 스케줄러
├── eval/              # 평가 메트릭, 리포트
├── infer/             # 추론 인터페이스
└── utils/             # 시드, 로깅, 체크포인트

configs/               # YAML 하이퍼파라미터
data/                  # 데이터셋 (gitignore) — 상세는 DATA_CARD.md
runs/                  # 실험 결과 — 상세는 EXPERIMENTS.md
```

## 학습 흐름
1. `python -m src.train --config configs/{name}.yaml --seed 42`
2. 시드 고정 (random / numpy / torch / cuda 모두) — 정책: [EXPERIMENTS.md#시드-정책](./EXPERIMENTS.md)
3. `runs/{YYYYMMDD-HHmm}-{tag}/` 디렉터리 생성, `config.yaml` + `git rev` 자동 저장
4. 데이터 로더 → 모델 → epoch 루프
   - 매 epoch: forward → loss → backward → step
   - 메트릭 → `runs/{id}/metrics.csv` + 추적 도구
   - 성능 갱신 시 `checkpoints/best.pt` 저장
5. 학습 종료 → `src.eval` 호출, [EVAL_PROTOCOL.md](./EVAL_PROTOCOL.md) 의 메트릭 산출

## 실험 추적
- 도구: {예: TensorBoard / W&B / MLflow / 자체 CSV}
- 자동 기록: {hyperparams, git rev, 시드, 데이터 버전}
- 상세 컨벤션: [EXPERIMENTS.md](./EXPERIMENTS.md)

## 외부 의존
| 종류 | 대상 | 용도 |
|------|------|------|
| 프레임워크 | {예: PyTorch 2.x} | {용도} |
| GPU | {예: CUDA 12.1, A100} | {용도} |
| 데이터 저장 | {예: S3 / 로컬 NAS} | {용도} |

## Hook 정책
- 자동 포맷터: {예: ruff format / black / 없음}
- 추가 차단 패턴: {예: 모델 가중치 (.pt, .pth) 커밋 차단 / 없음}
- Stop hook 추가 검증: {예: pytest tests/data + tests/models 빠른 단위 / 없음}

## 관련 문서
- 무엇을·왜: [PRD.md](./PRD.md)
- 결정 근거: [ADR.md](./ADR.md)
- 실험 운영: [EXPERIMENTS.md](./EXPERIMENTS.md)
- 데이터셋: [DATA_CARD.md](./DATA_CARD.md)
- 모델 카드: [MODEL_CARD.md](./MODEL_CARD.md)
- 평가 프로토콜: [EVAL_PROTOCOL.md](./EVAL_PROTOCOL.md)
