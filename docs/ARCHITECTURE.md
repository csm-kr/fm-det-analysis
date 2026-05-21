# 아키텍처

> **이 문서가 답하는 질문**: *코드(루트 평탄 모듈) ↔ 데이터(data/) ↔ 실험 결과(runs/)*가 어떻게 연결되며, 학습이 어떻게 흘러가는가? 데이터셋·모델·평가 상세는 별도 문서 ([DATA_CARD.md](./DATA_CARD.md) / [MODEL_CARD.md](./MODEL_CARD.md) / [EVAL_PROTOCOL.md](./EVAL_PROTOCOL.md)). config 도구·튜토리얼은 [HYDRA_GUIDE.md](./HYDRA_GUIDE.md), 실험 추적 도구·튜토리얼은 [WANDB_GUIDE.md](./WANDB_GUIDE.md).

## 설계 원칙
- **단순·평탄** — 모든 모듈을 **루트 바로 아래** 평탄 배치 (`datasets/`, `models/`, `losses/`, `evals/`, `utils/`). `src/` 폴더는 두지 않는다. 진입점(`train.py`, `eval.py`, `infer.py`) 도 루트.
- **`scripts/` 는 하네스 정본 전용** — 사용자 진입점은 두지 않는다. `execute.py`, `test_execute.py`, ml 헬퍼만.
- **detectron2 의존성 0** — 박스 회귀·matcher·loss 까지 모두 직접 구현. (CLAUDE.md CRITICAL)
- **모듈 교환성** — encoder / decoder / head / sampler / loss 각각을 Hydra config group 으로 분리, ablation 으로 갈아끼움.
- **레퍼런스 보존** — `DiffusionDet/` 폴더는 *읽기 전용 레퍼런스*로 유지. 새 구현은 의존하지 않는다.

## 시스템 구조
```
[코드 (루트 평탄)]                              [저장소]
┌─────────────────────────────┐                 ┌──────────────────────┐
│ datasets/   (COCO/VOC)      │ ◀── 읽음 ────── │ data/                │
│   ↓                         │                 │  coco/, voc/         │
│ models/     (enc/dec/       │                 └──────────────────────┘
│              head/sampler)  │ ◀── 합성 ────── ┌──────────────────────┐
│   ↓                         │ Hydra defaults  │ configs/             │
│ losses/     (diff/FM/       │                 │  train.yaml,         │
│              matcher)       │                 │  eval.yaml           │
│   ↓                         │                 │  model/  data/       │
│ evals/      (COCO AP /      │                 │  train/  loss/       │
│              VOC mAP)       │                 │  experiment/         │
│   ↓                         │                 └──────────────────────┘
│ train.py / eval.py /        │                 ┌──────────────────────┐
│ infer.py    (진입점)        │ ──── 기록 ────▶ │ runs/{ts}-{tag}/     │
└─────────────────────────────┘                 │  config.yaml         │
                                                │  metrics.csv         │
        │                                       │  checkpoints/        │
        └── 외부 추적 ──▶ TensorBoard / W&B     │  tb_logs/  wandb/    │
                                                └──────────────────────┘

scripts/      = 하네스 정본 (execute.py, test_execute.py, ml 헬퍼) — 사용자 진입점 아님
DiffusionDet/ = 레퍼런스 (읽기 전용, import 금지)
```

## 디렉터리 구조
```
.  (프로젝트 루트 — src/ 폴더 없음, 모듈을 루트 평탄 배치)
├── datasets/         # 각 데이터셋별 sub 모듈 (coco/, voc/)
│   ├── coco/         # 로더 + download/sanity/visualize/report CLI (python -m datasets.coco.*)
│   └── voc/          # 동일 패턴 (도입 예정)
├── models/           # encoder, decoder, head, sampler (diffusion / FM)
├── losses/           # diffusion_loss, flow_matching_loss, hungarian matcher
├── evals/            # COCO AP (pycocotools), VOC mAP@0.5, FPS 측정
├── utils/            # bbox(cxcywh↔xyxy), seed, vis, label_info, logging
├── train.py          # 진입점 (@hydra.main) — 학습
├── eval.py           # 진입점 (@hydra.main) — 평가
├── infer.py          # 진입점 (@hydra.main) — 단일 이미지 추론
│
├── configs/          # Hydra config — 상세는 HYDRA_GUIDE.md
│   ├── train.yaml          # 학습 진입점 base — defaults 로 group 합성
│   ├── eval.yaml           # 평가 진입점 base
│   ├── model/              # encoder/decoder/head/sampler 변종 group
│   ├── data/               # coco.yaml, voc.yaml
│   ├── train/              # optimizer/scheduler/batch 변종 group
│   ├── loss/               # diffusion.yaml, flow_matching.yaml
│   └── experiment/         # ablation 한 행 = 1 yaml (예: coco-fm1-sampler-cfm.yaml)
│
├── data/             # 데이터셋 (gitignore) — 상세는 DATA_CARD.md
├── runs/             # 실험 결과 (gitignore) — 상세는 EXPERIMENTS.md
├── DiffusionDet/     # 레퍼런스 (읽기 전용)
├── phases/           # /harness 가 생성한 step 파일
├── docs/             # 본 문서들 + HYDRA_GUIDE / WANDB_GUIDE
│
├── scripts/          # 하네스 정본만 (사용자 진입점 X)
│   ├── execute.py / test_execute.py
│   ├── crash_classifier.py / monitor.py / heartbeat.py / budget.py / fairness.py
│   └── README.md
│
├── .claude/          # hooks + skills + 박물관 정본
├── env_docker/       # Dockerfile, docker-compose.yml, docker-entrypoint.sh, .dockerignore (생성 완료)
└── pyproject.toml    # 루트 패키지 설정 (도입 예정) — pip install -e . 로 import 가능
```

## 학습 흐름
1. `python train.py +experiment=coco-repro-baseline seed=42` (Hydra 진입점 — `+experiment=` 로 ablation 한 행 선택).
2. Hydra 가 `configs/train.yaml` 의 `defaults` + `configs/experiment/coco-repro-baseline.yaml` 의 override 를 합성해 최종 config 생성.
3. 시드 고정 (random / numpy / torch / cuda 모두) — 정책: [EXPERIMENTS.md#시드-정책](./EXPERIMENTS.md).
4. Hydra 가 `runs/{YYYYMMDD-HHmm}-{tag}/` 디렉터리 생성 (`hydra.run.dir` override), `config.yaml` + `git_rev.txt` + `seed.txt` + `data_manifest.json` 자동 저장.
5. 데이터 로더 → 모델 → epoch 루프
   - 매 step: forward → loss(diffusion / flow_matching) → backward → optimizer step
   - 메트릭 → `runs/{id}/metrics.csv` + TB 스칼라 + W&B 로그
   - 성능 갱신 시 `checkpoints/best.pt` 저장
6. 학습 종료 → `python eval.py +experiment=coco-repro-baseline run_dir=runs/{id}` 호출, [EVAL_PROTOCOL.md](./EVAL_PROTOCOL.md) 의 메트릭 산출.

## 실험 추적
- **TensorBoard** (기준): 모든 `runs/{id}/tb_logs/` 에 스칼라/이미지. 외부 의존 0, 오프라인.
- **Weights & Biases** (병행, 학습 동반): `runs/{id}/wandb/`. 첫 도입이므로 사용법은 [WANDB_GUIDE.md](./WANDB_GUIDE.md) 참고.
- 자동 기록: hyperparams, git rev, seed, 데이터 버전, GPU 모델.
- 상세 명명·태깅 컨벤션: [EXPERIMENTS.md](./EXPERIMENTS.md).

## 외부 의존
| 종류 | 대상 | 용도 |
|------|------|------|
| 언어/런타임 | Python 3.11 | 베이스 |
| 프레임워크 | PyTorch 2.5 (+ torchvision) | 학습·추론 |
| GPU | CUDA 12.1, 단일 / 다중 GPU (DDP) | 학습 가속 |
| **Config 관리** | **Hydra 1.3 (+ OmegaConf)** | **그룹·상속·CLI override — [HYDRA_GUIDE.md](./HYDRA_GUIDE.md)** |
| 평가 | pycocotools, VOC eval util | mAP 산출 |
| 추적 | TensorBoard, Weights & Biases (wandb) | 실험 로깅 |
| 데이터 저장 | 로컬 디스크 (`data/`) | 데이터셋 |
| 컨테이너 | Docker + NVIDIA Container Toolkit (호스트 가정). `env_docker/` 생성 완료 (PyTorch 2.5.1 + CUDA 12.1 + cudnn9 devel, GPU runtime, shm 8gb) | 격리 환경 |

**금지**: detectron2 (CLAUDE.md CRITICAL).

## Hook 정책
- **PreToolUse 차단** (이미 적용 — `.claude/settings.json`): `rm -rf` / `git push --force` / `git reset --hard` / `DROP TABLE` / `rm -rf runs/` / 가중치 (`.pt/.pth/.ckpt/.safetensors`) `git add` / `git add -f data/`.
- **Stop hook 자동 검증** (이미 적용): `python3 scripts/test_execute.py` — 매 응답 종료 시 step 정합성 + ml 통합본 회귀 (121 passed / 2 skipped 기준).
- **추가 차단 패턴**: `import detectron2` / `from detectron2` 검출 시 PR/CI 실패 — 구현은 향후 phase 에서 lint rule 로 추가.
- **자동 포맷터**: `ruff format` (별도 phase 에서 도입).

## 관련 문서
- 무엇을·왜: [PRD.md](./PRD.md)
- 결정 근거: [ADR.md](./ADR.md)
- 실험 운영(시드/명명/ablation 표): [EXPERIMENTS.md](./EXPERIMENTS.md)
- 데이터셋: [DATA_CARD.md](./DATA_CARD.md)
- 모델 카드: [MODEL_CARD.md](./MODEL_CARD.md)
- 평가 프로토콜: [EVAL_PROTOCOL.md](./EVAL_PROTOCOL.md)
- **Hydra 학습 자료** (첫 도입): [HYDRA_GUIDE.md](./HYDRA_GUIDE.md)
- **W&B 학습 자료** (첫 도입): [WANDB_GUIDE.md](./WANDB_GUIDE.md)
