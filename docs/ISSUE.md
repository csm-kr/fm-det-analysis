# Issues — fm-det

> **이 문서가 답하는 질문**: 지금 *알려진 이슈는 무엇이며*, *과거에 해결된 이슈는 어떻게 발생하고 어떻게 풀렸는가*? Claude(또는 사람) 가 **모든 작업을 시작하기 전에 먼저 읽는** 문서 — 같은 이슈를 두 번 만나지 않기 위함.

## 사용 규칙
1. **작업 시작 전 본 문서를 먼저 읽는다.** ([../CLAUDE.md#이슈--현재-상태-관리](../CLAUDE.md) CRITICAL)
2. 새 이슈 발견 시 "진행 중" 섹션에 한 항목 append (아래 양식).
3. 이슈 해결 시 "해결됨" 섹션 맨 위로 이동, 해결책·재발 방지 한 줄 추가. **삭제 금지** — 미래의 같은 이슈에 컨텍스트 제공.
4. 한 줄 메모로 충분한 것은 [EXPERIMENTS.md](./EXPERIMENTS.md) "도구 학습 노트" 로. 본 문서는 *작업을 막거나·재발 가능성 있는* 이슈만.

## 양식
```markdown
### I-{NN}: {짧은 제목}
- **상태**: open / blocked / resolved
- **발견일**: YYYY-MM-DD
- **카테고리**: setup / data / model / training / eval / tooling / docs
- **증상**: {무엇이 잘못 동작하는가}
- **원인**: {추정 또는 확정 — 모르면 "조사 중"}
- **해결책 / 우회**: {적용된 방법 — open 이면 "검토 중"}
- **재발 방지**: {다음에 같은 상황을 만나면 무엇을 먼저 확인할지}
```

---

## 진행 중 (open / blocked)

### I-05: `jq` 미설치 — execute.py 의 `success_metric` 자동 검증 skip
- **상태**: blocked (Dockerfile 갱신 완료, 컨테이너 rebuild 대기)
- **발견일**: 2026-05-21
- **카테고리**: tooling
- **증상**: dev 컨테이너에 `jq` 가 없어 (`which jq` → not found) execute.py 의 `_verify_success_metric` 이 모든 jq 표현 검증을 skip. data-sanity-coco phase 의 step0/1 success_metric 도 jq 가 없어 직접 python 으로 검증함.
- **원인**: `env_docker/Dockerfile` 의 apt 패키지 목록에 `jq` 빠짐. harness 문서(`/.claude/skills/harness.md` §5-3)는 "jq 가 설치돼 있어야 한다" 명시인데 첫 빌드 시 누락.
- **해결책 / 우회**:
  - 영구화: `env_docker/Dockerfile` 의 apt-get install 라인에 `jq unzip` 추가 (2026-05-21 패치 완료) → `make build` 로 이미지 rebuild + `make up` 재기동 필요.
  - 즉시 적용 (호스트에서): `docker exec -u root <container> apt-get update && apt-get install -y jq unzip`.
  - 컨테이너 안 docker_user 권한으로는 apt/conda 모두 차단 (확인됨) — 위 두 방법 중 하나 필수.
- **재발 방지**: dev 컨테이너 새 빌드 시 `jq --version` 동작 확인 (entrypoint 헬스체크 한 줄). harness 정본 §5-3 의 "jq 설치 의무" 를 Dockerfile 주석에 미러 완료 (2026-05-21).

### I-04: DiffusionDet 본 repo 재현치(AP 46.2) 미검증 + 메커니즘 진단 미수행
- **상태**: open
- **발견일**: 2026-05-21
- **카테고리**: training
- **증상**: 본 프로젝트의 모든 ablation 결과가 "AP 46.2 ± 0.5" baseline 위에 서야 의미를 가진다 ([PRD.md#성공-기준](./PRD.md)). 또한 fm-det 의 "내공 형성" 목표는 P0a 진단 실험(최소 5행)이 채워져야 시작 — 아직 P0/P0a 학습 미실행.
- **원인**: 학습 시작 전 (현재는 docs/구조 설계 단계).
- **해결책 / 우회**: `/harness` 로 P0 (`{ds}-repro-*`) phase 설계 → 학습 + AP 측정 → ±0.5 이내면 P0a (`{ds}-diag-*`) 진단 5행 실행 → P1 FM 전환. 미달이면 후속 phase 보류.
- **재발 방지**: P0 baseline 매칭 + P0a 진단 5행 채우기 전에 P1/P2 학습 시작 금지. [EXPERIMENTS.md#단계적-FM-전환-로드맵](./EXPERIMENTS.md) 의 P0 → P0a → P1 순서 준수.

---

## 해결됨 (resolved — 미래의 같은 이슈에 참고)

### R-06: `.gitignore` 가 ai-ml 정책을 반영하지 않음
- **해결일**: 2026-05-21
- **카테고리**: setup
- **발생 경위**: 초기 `.gitignore` 는 node_modules / phases 산출물 / .env 만 포함. `data/`, `runs/`, `wandb/`, `outputs/`, `.hydra/`, `*.pt/.pth/.ckpt/.safetensors`, `__pycache__/`, `*.egg-info` 등 ai-ml 산출물이 추적 대상이라 데이터 다운(`data/coco/`)·실험 산출물(`runs/data-sanity-*`) 직후 git status 가 어수선.
- **해결책**: `.gitignore` 에 ai-ml 섹션 (`data/`, `runs/`, `wandb/`, `outputs/`, `.hydra/`, `*.pt/.pth/.ckpt/.safetensors`, `__pycache__/`, `*.py[cod]`, `*.egg-info/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `build/`, `dist/`, `.DS_Store`, `.vscode/`, `.idea/`) 추가. `git check-ignore -v` 로 `data/coco/val2017`, `runs/data-sanity-download-*/manifest.json` 매치 확인.
- **재발 방지**: 새 도구 추가 시 (예: `mlflow/`, `dvc/`) `.gitignore` 도 함께 갱신. PreToolUse hook (가중치·data/ 강제 추가 차단) 과 이중 안전망.

### R-05: `pyproject.toml` 미도입 — 루트 패키지 import 가 PYTHONPATH 의존
- **해결일**: 2026-05-21
- **카테고리**: setup
- **발생 경위**: `src/` 폐지 + 루트 평탄 배치 (ADR-005) 후 패키지 메타데이터 미작성. `from datasets import ...` 가 cwd 의존 → 다른 디렉터리에서 호출 시 `ModuleNotFoundError`.
- **해결책**: `pyproject.toml` 신설 — `[build-system]` setuptools 68+, `[project]` name=fm-det version=0.0.1 requires-python>=3.11, `[tool.setuptools.packages.find]` 의 include 에 `datasets*, models*, losses*, evals*, utils*` / exclude 에 `DiffusionDet*, data*, runs*, wandb*, outputs*, env_docker*, scripts*, phases*, docs*, tests*, .claude*`. `pip install -e .` 로 editable install 가능. (현재 모듈 폴더는 비어 있어 install 자체는 trivial — P0 baseline phase 시 datasets/coco.py 등 추가 시 실제 동작 확인 필요.)
- **재발 방지**: 새 루트 패키지 추가 시 (예: `transforms/`) `pyproject.toml` 의 packages.find.include 에 함께 추가. 폴더 신설 ≠ 자동 패키지화 — `__init__.py` + pyproject 의 include 양쪽 필요.

### R-04: Claude Code statusLine 미표시 — `settings.json` 의 호스트 경로 하드코딩
- **해결일**: 2026-05-21
- **카테고리**: tooling
- **발생 경위**: `~/.claude/settings.json` 의 `statusLine.command` 가 `bash /home/sungmin-cho/.claude/statusline-command.sh` 로 호스트 사용자 경로 하드코딩. 컨테이너 내부 사용자(`docker_user`) 에는 그 경로가 없어 statusLine 실행 자체가 실패 → 화면에 아무것도 안 보임. 또한 statusline 스크립트의 `${USER}` 가 컨테이너 실행 컨텍스트에서 비어 나와 `@hostname` 으로 표시되는 부수 이슈.
- **해결책**: (1) `settings.json` 의 command 를 `bash $HOME/.claude/statusline-command.sh` 로 변경 — 호스트/컨테이너 양쪽에서 동작. (2) statusline-command.sh 의 `"${USER}"` → `"${USER:-$(whoami)}"` 로 fallback 추가.
- **재발 방지**: `~/.claude/settings.json` 의 command 경로에 절대 사용자 홈 경로 하드코딩 금지 — 항상 `$HOME` 또는 `~` 사용. 호스트/컨테이너 동기화 설정 파일은 사용자별 차이를 추상화해야 함.

### R-03: `env_docker/` 미생성 — `/docker-init` 호출로 해소
- **해결일**: 2026-05-21
- **카테고리**: setup
- **발생 경위**: docs 채우기 단계까지 컨테이너 환경 없이 호스트 Python 으로 회귀를 돌렸음. 학습은 GPU 의존이라 호스트에서 실행 불가.
- **해결책**: `/docker-init` 호출 → `env_docker/{Dockerfile, docker-compose.yml, docker-entrypoint.sh, .dockerignore}` + `Makefile` + `.env.example` + `requirements.txt` 생성. 베이스 이미지 = `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel`, GPU runtime(`runtime: nvidia` + `deploy.resources.reservations.devices`), `shm_size: 8gb`, TB 포트 `6007:6006`, `~/.claude` 마운트로 인계.
- **재발 방지**: 컨테이너 의존성(PyTorch / CUDA / Hydra / W&B / pycocotools) 변경 시 `env_docker/Dockerfile` + `requirements.txt` 도 함께 갱신. 호스트에서 직접 `python train.py` 실행 금지 — 항상 `make shell` 후 컨테이너 안에서. 호스트 UID/GID 미스매치 권한 에러는 `.env` 의 `HOST_UID`/`HOST_GID` (`id -u`/`id -g` 결과) 로 보정.

### R-01: bootstrap 시점 `.git` 디렉터리 부재
- **해결일**: 2026-05-21
- **카테고리**: setup
- **발생 경위**: bootstrap §1 인프라 점검 후 git status 가 `fatal: 깃 저장소가 아닙니다` — README Quick Start 0단계의 `git init` 누락. 시스템 컨텍스트의 "Is a git repository: true" 와 어긋났음.
- **해결책**: `git init` 실행 → "기존 깃 저장소를 다시 초기화" 메시지 → 정상화.
- **재발 방지**: 새 프로젝트 클론 직후 `git init` 을 확인. bootstrap 의 §3-B-1 git 기반 dirty 점검은 git 저장소가 있어야 의미 — 없으면 사용자 콘텐츠 의심 분기로 보수적으로 진행.

### R-02: bootstrap.md 기댓값 (`91 passed, 2 skipped`) vs 실측 (`121 passed`)
- **해결일**: 2026-05-21
- **카테고리**: tooling
- **발생 경위**: ai-ml 박물관 §3-B-4 회귀 검증에서 bootstrap.md 의 기댓값보다 실측 테스트 수가 30개 많음.
- **해결책**: 박물관 정본(`templates/ai-ml/scripts/test_execute.py`) 이 그 사이 테스트가 추가됐을 뿐 — 실패가 없으므로 정상. bootstrap.md 의 기댓값 줄이 outdated 일 뿐 동작은 OK.
- **재발 방지**: 회귀 검증 판단 기준은 "실패 0 / skip 0~3" 가 본질. 기댓값 정확 매칭이 아님.
