# 프로젝트: fm-det

## 한 줄 목적
DiffusionDet 을 detectron2 없이 재구현하고, diffusion → flow matching 으로 단계적으로 전환하며 encoder/decoder/head/sampler/loss 모듈을 ablation 하여 detection 에서 실제 유효한 구성 요소를 정리한다 — 동시에 diffusion·flow matching 의 내공을 형성한다.

## 종류 / 기술 스택
- 종류: ai-ml (학습·연구. 운영 서비스 없음)
- 주 언어 + 버전: Python 3.11
- 프레임워크 + 버전: PyTorch 2.5 + torchvision (분산 학습은 직접 DDP)
- Config: **Hydra 1.3 + OmegaConf** — 첫 도입, 사용법은 [docs/HYDRA_GUIDE.md](./docs/HYDRA_GUIDE.md)
- 추가 의존: CUDA 12.1, pycocotools, TensorBoard, Weights & Biases (병행, 첫 도입 — [docs/WANDB_GUIDE.md](./docs/WANDB_GUIDE.md))
- 레퍼런스 (읽기 전용): `DiffusionDet/` — detectron2 기반. 새 코드는 의존하지 않는다.

## 아키텍처 규칙
- **CRITICAL: detectron2 import 절대 금지.** 이유: fm-det 의 존재 이유는 내부 동작을 직접 이해·재구현하는 것. 루트 평탄 모듈(`datasets/`, `models/`, `losses/`, `evals/`, `utils/`) 과 진입점(`train.py`, `eval.py`, `infer.py`) 내 `import detectron2` / `from detectron2` 검출 시 PR/CI 실패 처리.
- **CRITICAL: ablation 시 한 번에 한 변수만 변경.** 이유: 모듈 교체(diff→FM, encoder/decoder 등) 시 다른 변수(시드/lr/scheduler)를 함께 바꾸면 ΔAP 의 원인이 분리 불가 → 결론 무효. EXPERIMENTS.md 의 ablation 표 각 행은 "변경한 1 변수" 컬럼 명시.
- **CRITICAL: 평가 프로토콜 변경 금지.** 이유: DiffusionDet 본 repo 재현 조건(COCO val 해상도 800~1333 / iter step 4 / num eval boxes 500)과 동일 조건이어야 baseline 비교 신뢰도 확보. 변경 시 모든 이전 결과 무효.
- **CRITICAL: 시드 고정 의무.** 이유: 모든 학습/평가 스크립트는 `--seed` 인자 필수 (없으면 시작 차단). random / numpy / torch / cuda 모두 고정 + `cudnn.deterministic=True` / `cudnn.benchmark=False`. ablation 결과 재현 가능성.
- (일반) **`src/` 폴더 없음** — 모든 모듈을 루트 평탄 배치 (`datasets/`, `models/`, `losses/`, `evals/`, `utils/`). 진입점도 루트 (`train.py`, `eval.py`, `infer.py`).
- (일반) **`scripts/` 는 하네스 정본 전용** — 사용자 진입점은 두지 않는다. `execute.py` / `test_execute.py` / ml 헬퍼만.
- (일반) 모든 실험은 `runs/{YYYYMMDD-HHmm}-{tag}/` — Hydra 가 `hydra.run.dir` override 로 자동 생성. `config.yaml` + `git rev` + `seed` 자동 저장.
- (일반) ablation 한 행 = `configs/experiment/{tag}.yaml` 한 파일 = `runs/{ts}-{tag}/` 한 디렉터리 (1:1 매핑).
- (일반) 가중치 (`.pt/.pth/.ckpt/.safetensors`) 와 `data/` 는 gitignore. PreToolUse hook 으로 강제 add 차단.

## 개발 프로세스
- CRITICAL: 새 기능 / 버그 수정은 **테스트를 먼저 작성**하고 통과시키는 구현으로 진행 (TDD).
- **CRITICAL: 실험 1회 = 학습 + 분석 + 비교 + 문서화 한 묶음.** 학습만 돌리고 ablation 표/MODEL_CARD/ADR 미갱신은 "실험 안 한 것" 으로 본다. 이유: fm-det 의 산출물은 가중치가 아니라 **표에 채워진 결론** — DiffusionDet/FM 의 내공 형성이 목표 ([PRD.md#성공-기준](./docs/PRD.md)). 한 실험의 완료 정의는 [docs/EXPERIMENTS.md#실험-1회의-완료-정의](./docs/EXPERIMENTS.md).
- 커밋 메시지: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`, `exp:` — 실험 결과 추가용).
- 브랜치 / PR 정책: 단독 작업 — `feat-{task}` 브랜치 → main 머지 (Self-review). `/harness` 가 생성한 step 별 자동 커밋이 기본 단위.

## 이슈 + 현재 상태 관리
세션 시작 시 컨텍스트 회복과 같은 이슈 재발 방지를 위해 두 문서를 함께 운영한다 — [docs/ISSUE.md](./docs/ISSUE.md) (누적, append) + [docs/PENSIEVE.md](./docs/PENSIEVE.md) (현재 상태, 덮어쓰기).

- **CRITICAL: 모든 작업(코드/문서/실험) 시작 전 [docs/ISSUE.md](./docs/ISSUE.md) 와 [docs/PENSIEVE.md](./docs/PENSIEVE.md) 를 먼저 읽는다.** 이유: (a) 이미 알려진 이슈를 같은 자리에서 재발시키지 않기 위함, (b) 오랜만에 돌아왔을 때 "지금 어디까지 왔는가" 컨텍스트 회복. 둘 다 비어 있거나 변동 없어 보여도 한 번 확인하는 것이 규칙.
- **CRITICAL: 작업 종료 시 [docs/PENSIEVE.md](./docs/PENSIEVE.md) 를 갱신한다.** 최소 갱신 항목: "마지막 업데이트" / "지금 어디" / "다음 한 가지" 세 곳. "최근 변경" 도 한 줄 append (오래된 항목은 5개 유지를 위해 한 줄 drop). 나머지는 변경 시에만.
- 새 이슈 발견 시: [docs/ISSUE.md](./docs/ISSUE.md) 의 "진행 중" 섹션에 양식대로 append. 작업을 멈출 만한 이슈면 `blocked` 상태로. PENSIEVE.md 의 "미해결 결정 / 블로커" 에도 한 줄 미러링.
- 이슈 해결 시: ISSUE.md 의 "해결됨 (resolved)" 섹션 맨 위로 **이동** (삭제 금지) — 미래에 같은 증상을 만났을 때의 컨텍스트. PENSIEVE.md 의 블로커 줄에서도 제거.
- docs/ISSUE.md 에 들어가는 항목: *작업을 막거나 / 재발 가능성이 있거나 / 미래 작업자가 같은 함정을 만날 수 있는* 이슈. 한 줄 메모성 학습 노트는 [docs/EXPERIMENTS.md](./docs/EXPERIMENTS.md) 의 "도구 학습 노트" 로.

## 마일스톤
누적 — 큰 단계 종료 시 한 줄 추가. 시점 의존 상태는 [docs/PENSIEVE.md](./docs/PENSIEVE.md). 누적성 보존(삭제 금지).

- **M0 — 부트스트래핑 + docs + 컨테이너 환경** (2026-05-21): docs 트라이앵글(PRD/ARCH/ADR) + EXPERIMENTS 진단 카탈로그 + env_docker/{Dockerfile, compose, entrypoint} + Makefile + 정본 hook 2개.
- **M1 — Pre-P0 데이터 sanity + 인프라 정리** (2026-05-21): `data-sanity-coco` (val) phase 완료 (CP-1 approved) — 다운(`wget` + zipfile) / 분석(pycocotools) / 시각화(matplotlib + Pillow, 4종) / report 묶음. 인프라 보강 — `pyproject.toml` (R-05) + `.gitignore` ai-ml (R-06) + Dockerfile `jq unzip` 추가 (I-05 rebuild 대기). 모듈화 — `datasets/coco/{download,sanity,visualize,report}` sub 모듈 패턴 확립 (VOC 동일 패턴 예정). [phases/data-sanity-coco/](./phases/data-sanity-coco/) · [docs/ISSUE.md](./docs/ISSUE.md).

## LLM 협업 원칙
**모든 코딩 작업은 [LLM_GUIDE.md](./LLM_GUIDE.md) 의 4원칙을 따른다**:
1. **Think Before Coding** — 가정을 먼저 명시하고, 불확실하면 묻는다.
2. **Simplicity First** — 요청한 것보다 더 만들지 않는다. 추측성 추상화·과한 에러 핸들링 금지.
3. **Surgical Changes** — 요청한 부분만 고친다. 인접 코드를 임의로 "개선"하지 않는다.
4. **Goal-Driven Execution** — 검증 가능한 성공 기준을 먼저 정한다 ("동작한다" 같은 모호한 기준 금지).

## 문서 트라이앵글
어떤 결정을 어디서 내리는지 한 곳에 못박는다 — Claude 와 사람 모두에게 같은 컨텍스트.
- **무엇을 / 왜**: [docs/PRD.md](./docs/PRD.md)
- **어떻게**: [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md)
- **왜 이 결정**: [docs/ADR.md](./docs/ADR.md)
- **종류별 추가 docs**: [docs/EXPERIMENTS.md](./docs/EXPERIMENTS.md) (ablation 표·시드·명명) / [docs/DATA_CARD.md](./docs/DATA_CARD.md) / [docs/MODEL_CARD.md](./docs/MODEL_CARD.md) / [docs/EVAL_PROTOCOL.md](./docs/EVAL_PROTOCOL.md)
- **도구 학습 자료** (첫 도입): [docs/HYDRA_GUIDE.md](./docs/HYDRA_GUIDE.md) / [docs/WANDB_GUIDE.md](./docs/WANDB_GUIDE.md)
- **PRD 검수 렌즈**: [docs/PRD_VIEW.md](./docs/PRD_VIEW.md) (13 View 체크리스트, 공용 정본)

규칙: 새 결정은 ADR 항목으로. 새 사용자 시나리오는 PRD 갱신. 새 컴포넌트/모듈은 ARCHITECTURE 갱신. 새 실험은 EXPERIMENTS 의 ablation 표에 한 행 추가.

## 주요 명령어
**호스트에는 Docker + NVIDIA Container Toolkit 만 있다고 가정.** 아래 명령은 모두 dev 컨테이너 안에서 실행한다 (`make shell` 또는 `docker compose -f env_docker/docker-compose.yml exec dev bash` 후). `env_docker/` 는 생성 완료 — `make up` 으로 컨테이너 띄움.

```bash
# 의존성 설치 (컨테이너 안)
pip install -e .                              # pyproject.toml 도입 후 — 루트 패키지 import 활성화

# 학습 — Hydra 진입점, seed 필수 (CRITICAL)
python train.py +experiment=coco-repro-baseline seed=42

# 평가 — runs/{id}/ 안의 best.pt 대상
python eval.py +experiment=coco-repro-baseline run_dir=runs/{id}

# 추론 (단일 이미지)
python infer.py +experiment=coco-repro-baseline run_dir=runs/{id} image=path/to/img.jpg

# 린트 / 타입 체크 (도입 예정 — 별도 phase)
ruff check .
mypy datasets/ models/ losses/ evals/ utils/

# 테스트 — 정본 step 검증 + 사용자 단위 테스트
python scripts/test_execute.py               # 정본 (Stop hook 이 자동 실행)
pytest tests/                                # 사용자 단위 테스트 (도입 예정)

# 실험 진행 — /harness 가 phase/step 생성 후
python scripts/execute.py {task}             # step 순차 실행 + 자동 커밋
```

## Hooks
이 레포의 정본 hook 2개 (위험 명령 차단 + Stop 자동 검증) — [docs/HOOKS.md](./docs/HOOKS.md) 참고.
ai-ml 추가 차단 패턴 (적용됨): `rm -rf runs/`, 가중치 (`.pt/.pth/.ckpt/.safetensors`) `git add`, `git add -f data/`.
프로젝트별 추가 hook 정책은 [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) 의 "Hook 정책" 섹션.
