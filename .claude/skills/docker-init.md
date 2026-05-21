# docker-init

이 프로젝트가 종류·스택에 맞는 **격리 환경** 위에서 돌게 한다.

**호스트 가정**: 호스트에는 Docker 만 있다. Python·Node 등 다른 런타임은 없다고 가정한다. 모든 도구·런타임·테스트는 컨테이너 안에서 실행한다.

**산출물 구조**:

```
프로젝트루트/
├── env_docker/                  # 환경 도커 — 모든 환경 관련 파일이 이 한 폴더에
│   ├── docker-compose.yml       # 서비스 토폴로지 (dev + 사이드 서비스)
│   ├── Dockerfile               # 멀티스테이지 — base(OS+도구) → project(의존성) → dev(런타임)
│   ├── .dockerignore
│   └── docker-entrypoint.sh     # 호스트 git config 주입 등 부속 스크립트
├── .env.example                 # 루트 (compose 는 ../.env 로 참조)
├── Makefile                     # 옵션 — `make up`, `make shell` 로 명령 단축
└── (사용자 프로젝트 코드, 사용자가 만들 자체 Dockerfile / docker-compose.yml 등)
```

compose 안에는 dev 컨테이너(작업용) 와 프로젝트가 필요로 하는 사이드 서비스(DB·Redis·MQ 등)가 **모두 형제(sibling)** 로 정의된다.

**왜 환경 도커 일체를 `env_docker/` 로 분리하나**:
- **루트는 100% 사용자 프로젝트 영역**으로 비워둠 — 사용자가 자기 프로덕션용 `Dockerfile` 이나 자체 `docker-compose.yml` 을 루트에 자유롭게 만들 수 있음. 환경 도커와 충돌 없음.
- 환경 관련 모든 파일(compose, Dockerfile, entrypoint, .dockerignore)이 한 폴더에 모임 — 멘탈 모델 깔끔.
- `.claude/skills/docker_examples/` (참고 예시) 와 동일한 폴더 모듈화 — 일관성.

**대가**: 모든 docker compose 명령에 `-f env_docker/docker-compose.yml` 을 명시해야 함. 루트의 `Makefile` 한 개로 단축한다 (아래 절차 참고).

> **비대상**: 프로덕션/배포용 이미지 설계, 레지스트리 푸시 전략은 이 스킬의 책임이 아니다. 이건 "개발 중 이 프로젝트가 잘 돌아가는 환경" 만 만든다.

---

## 참고 예시 (정독 필수)

먼저 아래 두 파일을 정독하라:

- `.claude/skills/docker_examples/Dockerfile`
- `.claude/skills/docker_examples/docker-compose.yml`

이 두 파일은 **하네스 정본 참고 예시**다. 베끼는 정본이 아니라 다음 **패턴**을 보여주는 레퍼런스:

- 베이스 이미지 핀닝 + digest 핀 주석
- ARG 로 외부 의존성 버전을 한 곳에 모으기
- Claude Code CLI / GitHub CLI / 플러그인 marketplace 사전 등록 (commit 핀)
- non-root user, entrypoint 로 호스트 GIT_USER_NAME/EMAIL 주입
- 호스트 코드 + **호스트 `~/.claude` 디렉터리 전체** 를 컨테이너 사용자 홈으로 마운트 (auth · memory · 글로벌 CLAUDE.md · 프로젝트 히스토리 공유 → 호스트↔컨테이너 claude 세션 인계 자연스러움)

**그러나 환경은 프로젝트마다 완전히 달라진다.** 베이스 OS·언어·프레임워크·추가 서비스·포트·GPU 사용 여부 — 이 중 어느 것도 예시와 같다고 가정하지 마라. 패턴만 가져오고 내용은 처음부터 결정하라.

---

## 핵심 원칙 — 단일 compose, 형제 서비스 (DinD 회피)

**dev 컨테이너 안에서 또 docker 를 쓰지 마라.** 이유: (a) 호스트/외부/내부 어디 컨테이너인지 헷갈리고 (b) 볼륨 경로가 두 번 매핑되며 (c) 포트 노출이 이중이 된다.

대신:

- `docker-compose.yml` 안에 **dev 서비스 + 사이드 서비스(DB·Redis·MQ 등)를 모두 형제로** 정의한다.
- dev 안에서 사이드 서비스는 **서비스명 DNS** 로 접속 — `db:5432`, `redis:6379`. 호스트 포트 매핑은 최소화.
- 호스트는 dev 의 앱 포트 1~2개만 노출.

```
       ┌───────── docker-compose.yml ─────────┐
호스트 │  dev (Claude Code, gh, 언어 런타임)   │
:3000 ─┤  ↕ 서비스명 DNS                       │
       │  db (postgres)    redis    mq         │
       └───────────────────────────────────────┘
```

### 예외 — DooD (Docker-out-of-Docker)

진짜로 dev 안에서 docker SDK 를 호출해야 할 때만 (예: 사용자 코드가 컨테이너를 띄우는 라이브러리/CI 도구). 호스트 docker 소켓을 마운트:

```yaml
volumes:
  - /var/run/docker.sock:/var/run/docker.sock
```

⚠️ 경고: 호스트 docker 권한이 dev 컨테이너에 통째로 노출된다. 형제 서비스 패턴으로 풀 수 있는지부터 다시 검토하라. 진짜 DinD(`docker:dind` 사이드카) 는 마지막 수단.

---

## 절차

### 0. 사전 읽기 + 결정 요약 확인

#### 0-1. 정독
다음을 먼저 읽고 이미 결정된 항목은 **다시 묻지 마라**:

- `CLAUDE.md` — 종류, 기술 스택, 추가 의존
- `docs/PRD.md` — 사용자/시나리오 (포트·외부 서비스 단서)
- `docs/ARCHITECTURE.md` — 디렉터리 구조, 외부 의존, 데이터 흐름
- `docs/ADR.md` — 기술 선택 (DB, 캐시, 메시지큐 등)
- `docs/PRD_VIEW.md` — 검수 렌즈 (직접 결정은 아니지만 누락된 관점 점검용)
- 종류별 추가 docs — `DATA_CARD.md`(ai-ml), `API_SPEC.md`(backend), `RUNBOOK.md`(data-pipeline) 등

#### 0-2. 결정 요약 + 확정 (필수)
정독 후 사용자에게 **추출한 핵심 결정을 한 번에 요약해서 보여주고 확정받은 뒤** 다음 단계로 넘어간다. docker-init 은 이 요약 위에서 컨테이너를 만든다.

요약 형식 (해당 없는 항목은 "(없음)"):
```
[docs 에서 읽은 결정 — 확인하고 한 줄로 답해주세요]
- 종류 / 베이스 스택: {예: backend / Python 3.12 + FastAPI}
- 외부 의존 서비스: {예: PostgreSQL 16, Redis 7}
- 노출 포트: {예: 앱 8000}
- 데이터 디렉터리 / 볼륨: {예: data/ bind, db named}
- 특수 요구: {예: GPU 없음 / shm_size 불필요}
- 누락된 결정: {있으면 — 예: "캐시 TTL 정책 미정 — docker 와 무관하므로 진행 가능"}

이대로 진행할까요? (a) 그대로 / (b) 수정할 항목이 있음 → 어떤 항목 / (c) 취소
```

**비어 있거나 모순되는 항목이 있으면 docker-init 을 진행하지 마라.** 사용자에게 다음을 안내:
- PRD/ARCH/ADR 의 해당 섹션을 먼저 채우고 다시 호출
- 또는 "현재 결정만으로 진행하고 미정 항목은 컨테이너 띄운 뒤 결정" 으로 동의받기

(b) 선택 시 사용자가 답한 수정 내용을 반영해서 요약을 다시 보여주고 다시 확정받는다.

### 1. 정보 수집 (부족분만, 한 항목씩)

[bootstrap.md 의 "대안 제시 (Always Offer Options)" 패턴](./bootstrap.md) 을 따른다. 한꺼번에 묻지 마라.

1. **프로젝트명** — 컨테이너명·이미지명·`working_dir` 에 사용. CLAUDE.md 헤더에 있으면 재사용.
2. **종류 확인** — bootstrap 단계에서 결정됨. 누락이면 `(a)~(d)` 대안 제시.
3. **앱 포트** — 호스트로 노출할 포트 (스택별 기본값 제안: web 3000, backend 8000, ai-ml 6006).
4. **추가 서비스** — ADR/ARCHITECTURE 에 결정돼 있으면 그대로. 없으면 종류별 매트릭스 보고 후보 제시 + (d) 없음.
5. **GPU / 특수 요구** — ai-ml 에서 자주. 베이스 이미지 자체가 달라진다.
6. **data 디렉터리** — 데이터셋·로그·체크포인트 경로 (ai-ml/data-pipeline 에서 자주). 호스트 디렉터리를 그대로 물릴지 named volume 으로 격리할지.

### 2. 종류별 베이스 매트릭스 적용

| 종류 | dev 베이스 | 추가 시스템 패키지 | 사이드 서비스 (compose) | volume / data |
|------|-----------|------------------|----------------------|---------------|
| `ai-ml` | `pytorch/pytorch:<ver>-cuda<ver>-cudnn<ver>-devel` 또는 `nvidia/cuda:<ver>-<base>` | jupyter, ffmpeg(필요시) | tensorboard 컨테이너(선택) | `data/` bind, `runs/` bind, `~/.cache/huggingface` named, **`shm_size: 8gb`** (DataLoader SHM) |
| `web` | `node:<lts>-bookworm` (필요시 + python) | git, gh | postgres, redis (ADR 의존) | DB named, `node_modules` named (성능) |
| `backend` | `python:<ver>-slim` 또는 `node:<lts>-bookworm` | gh, db client | postgres, redis, kafka(선택) | DB named, 마이그레이션 디렉터리 bind |
| `data-pipeline` | `python:<ver>-slim` | gh, java(spark/kafka 클라이언트 필요시) | postgres, kafka, minio(s3 호환), airflow(선택) | source/sink 디렉터리 bind, broker named |
| `mobile` | `node:<lts>-bookworm` | watchman, openjdk(안드로이드용) | (보통 없음) | `node_modules` named, 빌드 캐시 named |
| `cli-lib` | 언어별 공식 이미지 | gh | (없음) | (없음) |
| `custom` | 가장 가까운 base 추천 후 사용자 확인 | — | — | — |

**모든 dev 컨테이너 공통** (예시 Dockerfile 그대로 차용):

- Claude Code CLI (버전 핀)
- GitHub CLI (`gh`)
- 플러그인 marketplace 사전 등록 (`anthropics/claude-plugins-official`, commit 핀)
- non-root user (`docker_user` 등)
- entrypoint 로 호스트 GIT_USER_NAME/EMAIL 주입

베이스 이미지의 Ubuntu 22.04 + Python 3.10 조합을 무비판적으로 따르지 마라. 스택에 따라 `node:22-slim`, `python:3.12-slim`, `nvidia/cuda:12.4-...` 등이 더 적합할 수 있다.

### 3. volume / port / IPC / data 정합 규칙

**소스 코드**:
- `.:/workspace/<프로젝트명>` 한 줄 bind. 호스트에서 편집 → 컨테이너에서 즉시 반영.
- dev 의 `working_dir` = bind 마운트 대상 = CLAUDE.md "주요 명령어" 의 기준.

**`~/.claude` 디렉터리 전체를 마운트하라.** 호스트와 컨테이너가 동일한 신뢰 영역(본인 개발 환경)이라는 전제 하에:

- **마운트 대상**: 호스트 `~/.claude` 와 (있다면) `~/.claude.json` 을 컨테이너 사용자 홈의 같은 경로로.
- **컨테이너 root 사용자**: `~/.claude:/root/.claude`, `~/.claude.json:/root/.claude.json`
- **컨테이너 non-root 사용자** (`docker_user` 등): `~/.claude:/home/docker_user/.claude`, `~/.claude.json:/home/docker_user/.claude.json` — UID/GID 미스매치로 권한 에러가 나면 entrypoint 에서 `chown` 으로 맞추거나, 컨테이너 유저를 호스트 UID 로 생성.

**왜**:
- 호스트에서 이미 로그인된 OAuth 세션 그대로 사용 → 컨테이너 안에서 재로그인 불필요.
- 글로벌 `~/.claude/CLAUDE.md` 의 사용자 룰이 컨테이너 안 claude 에도 그대로 적용됨.
- `~/.claude/projects/{현재 프로젝트}/` 의 대화 히스토리·memory 가 공유되어, 호스트에서 docker-init 까지 진행한 세션을 **컨테이너 안에서 그대로 이어 받을 수 있음**.
- `~/.claude/settings.json` 의 글로벌 권한·허용 명령 목록도 그대로 적용.

**주의**: 자격증명까지 컨테이너에 노출되므로, 컨테이너가 본인 신뢰 영역이 아닌 경우(공유 머신, 이미지를 외부에 푸시, 의존성을 신뢰 못 함)에는 `~/.claude/CLAUDE.md` 만 `:ro` 로 단일 파일 마운트하는 보수적 옵션을 사용자에게 제시한다.

**참고 — 외부 클라우드 자격증명 (필요할 때만 사용자가 직접 추가)**:

`~/.claude` 마운트 패턴은 다른 클라우드 SDK 에도 그대로 적용된다. docker-init 은 이 부분을 자동 생성하지 않는다 — 사용자가 해당 서비스를 실제로 쓸 때 아래를 참고해서 compose 에 직접 추가한다.

| 사용 시 | 마운트 / 환경변수 | 비고 |
|--------|------------------|------|
| GCP / Firebase (개인 dev) | `~/.config/gcloud:/home/docker_user/.config/gcloud` (ADC) | 호스트에서 `gcloud auth application-default login` 선행 |
| GCP / Firebase (팀·prod 일관) | 서비스 계정 키 파일 마운트 + `GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/gcp-sa.json` | 키 파일은 `.gitignore` 필수, `:ro` 권장, dev/prod 계정 분리, 최소 권한 |
| GitHub CLI | `~/.config/gh:/home/docker_user/.config/gh` | `gh auth login` 결과 공유 |
| AWS | `~/.aws:/home/docker_user/.aws` | profile/SSO 캐시 공유 |

**왜 OAuth 브라우저 흐름이 컨테이너 안에서 자주 깨지나** (참고용 — 위 마운트 방식이 우회책):
- 컨테이너에 브라우저가 없어 인증 URL 을 열 수 없음
- 콜백 `http://localhost:PORT/...` 가 호스트 브라우저 → 컨테이너 리스너로 도달하지 못함 (네임스페이스 분리)
- → 호스트에서 먼저 로그인하고 자격증명 디렉터리/키 파일을 마운트하는 게 가장 단순

**포트**:
- dev 의 앱 포트만 호스트로 노출 (web=3000, backend=8000, ai-ml=6006 TB).
- DB/Redis 는 기본적으로 노출하지 않음. 외부 GUI 도구로 디버깅이 필요한지 사용자에게 옵션으로 묻는다.
- 같은 호스트에서 다른 프로젝트와 충돌하면 호스트 쪽 포트만 바꾼다 (`6007:6006`).

**IPC**:
- 기본 격리.
- ai-ml 은 PyTorch DataLoader SHM 부족을 막기 위해 `shm_size: 8gb` (또는 `ipc: host`) 옵션을 dev 서비스에 추가.

**data**:
- 코드/문서: bind 마운트 (호스트 공유, git 추적).
- DB/캐시: **named volume** (호스트 더럽히지 않음, 컨테이너 재생성에도 유지).
- 대용량 데이터셋(ai-ml/data-pipeline): bind 마운트로 호스트 디렉터리를 그대로 물림. `.gitignore` 에 추가하라고 안내.
- HuggingFace/pip/npm 캐시: named volume — 빌드 시간 절감, 호스트와 격리.

### 4. 파일 생성

#### `env_docker/Dockerfile` (멀티스테이지)

환경 도커파일은 **반드시 `env_docker/` 폴더 안에** 둔다. 루트에 두지 마라. 멀티스테이지로 두 단계를 명확히 분리:

```dockerfile
# ───────────────────────────────────────────────
# Stage 1: base — 종류별 OS + 시스템 도구 + 런타임
# ───────────────────────────────────────────────
FROM <종류별 베이스 이미지> AS base

ARG NODE_MAJOR=20
ARG CLAUDE_CODE_VERSION=2.1.123
ARG CLAUDE_PLUGINS_COMMIT=<commit-sha>

# 시스템 패키지 (apt/apk), 언어 런타임
# Claude Code CLI + gh + 플러그인 marketplace 사전 등록 (예시 그대로)
# non-root user 준비

# ───────────────────────────────────────────────
# Stage 2: project — 프로젝트 의존성 (얇은 레이어)
# ───────────────────────────────────────────────
FROM base AS project

# 의존성 매니페스트만 먼저 COPY 해서 캐시 최적화
COPY requirements.txt /tmp/         # 또는 package.json + package-lock.json 등
RUN pip install -r /tmp/requirements.txt   # 또는 npm ci

# ───────────────────────────────────────────────
# Stage 3: dev — 작업용 최종 이미지
# ───────────────────────────────────────────────
FROM project AS dev

WORKDIR /workspace/<프로젝트명>

USER docker_user
ENV HOME=/home/docker_user

COPY --chown=docker_user:docker_user env_docker/docker-entrypoint.sh /usr/local/bin/
USER root
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
USER docker_user

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]
```

핵심:
- **stage 분리** — `base` (재빌드 빈도 낮음, 캐시 잘 탐) → `project` (의존성 추가/변경 시) → `dev` (마지막 조립).
- **외부 의존성 버전은 `ARG`** 로 상단에 모으기.
- **소스 코드는 COPY 하지 마라** — bind 마운트로 들어온다. COPY 는 의존성 매니페스트(`requirements.txt`, `package.json` 등) 만.
- 프로젝트가 의존성 매니페스트가 없는 경우(예: 초기 ai-ml) `project` 스테이지를 비워둬도 무방.

#### `env_docker/.dockerignore`

빌드 컨텍스트는 **프로젝트 루트** (compose 의 `context: ..`). `.dockerignore` 는 `env_docker/` 안에 두면 환경 도커 파일 일체가 한 폴더에 모이는 멘탈 모델 유지 — 단, 일부 docker 버전은 빌드 컨텍스트 루트의 `.dockerignore` 만 인식할 수 있으므로 **양쪽에 두거나 루트에 두는 것도 허용**한다.

```
.git
.venv
__pycache__
.pytest_cache
node_modules
phases/
data/
runs/
*.log
```

#### `env_docker/docker-entrypoint.sh`

호스트 git 정보 주입 등 부속 스크립트. `.claude/skills/docker_examples/` 의 entrypoint 패턴 차용.

#### `env_docker/docker-compose.yml`

`env_docker/` 폴더 안에 둔다. 빌드 컨텍스트·볼륨·env_file 은 모두 **`..` (프로젝트 루트) 기준**으로 잡는다.

```yaml
name: <프로젝트명>                  # compose project name 명시 — 폴더명(env_docker)이 잡히지 않게

services:
  dev:                              # 또는 <프로젝트명>
    build:
      context: ..                   # 빌드 컨텍스트 = 프로젝트 루트 (의존성 파일 COPY 가능)
      dockerfile: env_docker/Dockerfile
      target: dev                   # 멀티스테이지 최종 단계
      args:
        CLAUDE_CODE_VERSION: 2.1.123
    image: <프로젝트명>
    container_name: <프로젝트명>_dev
    volumes:
      - ..:/workspace/<프로젝트명>  # 프로젝트 루트를 bind
      - ~/.claude:/home/docker_user/.claude         # auth/memory/global CLAUDE.md 공유 (non-root user 기준)
      - ~/.claude.json:/home/docker_user/.claude.json   # 있으면
      # ai-ml: ../data, ../runs bind + cache named
      # web/backend: 필요시 node_modules named
    ports:
      - "<host>:<container>"        # 앱 포트만
    # ai-ml: shm_size: 8gb
    stdin_open: true
    tty: true
    working_dir: /workspace/<프로젝트명>
    command: bash
    env_file: ../.env               # 루트의 .env 참조
    depends_on:                     # 사이드 서비스 있을 때만
      db:
        condition: service_healthy

  # 사이드 서비스 — 형제로 정의
  db:
    image: postgres:16
    environment:
      POSTGRES_DB: ...
      POSTGRES_USER: ...
      POSTGRES_PASSWORD: ...
    volumes:
      - db-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U $$POSTGRES_USER"]
      interval: 5s
      retries: 5
    # ports: 호스트 노출은 옵션 (기본은 노출 안함)

volumes:
  db-data:
```

핵심 경로 규칙:
- `build.context: ..` — 빌드 컨텍스트는 프로젝트 루트.
- `build.dockerfile: env_docker/Dockerfile` — context 기준 상대 경로.
- `volumes: - ..:/workspace/<프로젝트명>` — 컨테이너에는 루트가 통째로 들어감.
- `env_file: ../.env` — 루트의 `.env` 참조.
- `name: <프로젝트명>` — compose v2 의 project name 필드. 명시하지 않으면 폴더명(`env_docker`) 이 잡혀 다른 프로젝트와 충돌·헷갈림 발생.

#### `Makefile` (루트, 옵션 — 명령어 단축)

`-f env_docker/docker-compose.yml` 을 매번 치기 번거로우면 루트에 작은 Makefile 을 둔다:

```makefile
COMPOSE := docker compose -f env_docker/docker-compose.yml

.PHONY: up down shell build logs

up:    ; $(COMPOSE) up -d --build
down:  ; $(COMPOSE) down
shell: ; $(COMPOSE) exec dev bash
build: ; $(COMPOSE) build
logs:  ; $(COMPOSE) logs -f
```

`make up`, `make shell` 로 짧아진다. Makefile 을 안 쓰는 프로젝트면 alias 안내(`alias dc='docker compose -f env_docker/docker-compose.yml'`) 또는 매번 명시.

#### `.env.example` (루트)

DB URL, 포트, 시크릿 키, API 키 등 필요한 환경변수 목록. 실제 값은 비워두고 설명 주석. **`.env` 는 절대 생성 금지.**

### 5. 기존 파일 갱신

#### `CLAUDE.md`

`{}` 플레이스홀더 제거하고 채운다:

- `종류 / 기술 스택` — 결정된 스택과 버전.
- `주요 명령어` — dev 컨테이너 안에서 실행 전제. compose 가 `env_docker/` 안에 있으므로 호스트에서 띄우는 명령은 `-f` 명시 (또는 Makefile 단축):

```
# 호스트 (컨테이너 띄우고 들어가기)
docker compose -f env_docker/docker-compose.yml up -d --build
docker compose -f env_docker/docker-compose.yml exec dev bash
# Makefile 을 만들었으면: make up && make shell

# 셸 안에서 (이후는 모두 컨테이너 안 — claude 포함)
claude                                    # 인증/메모리/룰 공유됨 (~/.claude 마운트)
<빌드 / 컴파일>                           # 예: npm run build / pytest
<개발 서버>                               # 예: npm run dev
<린트>
<테스트>
```

#### `.claude/settings.json` Stop hook

`~/.claude` 디렉터리 전체를 마운트하므로 **이후 모든 claude 세션은 컨테이너 안에서 실행하는 것이 기본**이다 (옵션 A). hook 명령은 그대로 `python3 scripts/test_execute.py` — 예시 Dockerfile 이 이미 Claude Code CLI + pytest 를 컨테이너에 설치한다.

옵션 B (호스트에서 `claude` 를 띄우고 hook 만 컨테이너에 위임) 는 비권장 — 인계 모델과 충돌하고 매 응답마다 컨테이너 spinup 비용이 든다. 그래도 필요하면:
```json
"command": "docker compose -f env_docker/docker-compose.yml run --rm dev python3 scripts/test_execute.py 2>&1"
```

#### `phases/index.json`

없으면 `{"phases": []}` 로 생성. 있으면 손대지 마라.

### 6. 멱등성 / 기존 파일 처리

이미 `env_docker/` 가 있으면 사용자에게 대안 제시:

```
이미 env_docker/ 가 있습니다. 어떻게 할까요?
  (a) _archive/{YYYYMMDD-HHMMSS}/env_docker/ 로 통째 백업 후 재생성 (추천)
  (b) 통째 덮어쓰기 — 기존 내용 사라짐
  (c) 취소 — docker-init 종료
```

(a) 선택 시 `_archive/{YYYYMMDD-HHMMSS}/env_docker/` 로 옮겨 보관. 이후 사용자가 비교·복원 가능.

> 직전 버전(루트의 `Dockerfile` / `docker-compose.yml`) 이 남아 있는 프로젝트라면, 마이그레이션 시 **그 두 파일도 같은 archive 로 이동**시키고 새 구조로 만든다. 사용자에게 한 줄 안내: "기존 루트 Dockerfile / docker-compose.yml 을 `_archive/.../` 로 보관했고, 이제 환경 도커는 모두 `env_docker/` 안에 있습니다."

> 루트의 `docker-compose.yml` 또는 `Dockerfile` 은 이제 **사용자 프로젝트의 자체 도커**(예: 프로덕션 배포용) 영역으로 비워두므로, 사용자가 명시적으로 만든 게 아니면 archive 로 옮긴다.

### 7. 완료 안내

생성된 파일 목록 출력 + 시작 방법:

```bash
cp .env.example .env                                                # 시크릿 채우기

# 호스트에서 컨테이너 띄우기 — 둘 중 하나
docker compose -f env_docker/docker-compose.yml up -d --build       # 직접
make up                                                             # 또는 Makefile

# 셸 진입
docker compose -f env_docker/docker-compose.yml exec dev bash       # 직접
make shell                                                          # 또는

# 셸 안에서:
claude                                                              # 이후는 컨테이너 안에서
```

---

### 8. 호스트 → 컨테이너 인계 (Handoff)

**docker-init 자체는 호스트에서 실행되지만, 이후 모든 작업은 컨테이너 안에서 한다.** 이 마지막 단계는 사용자에게 인계 방법을 명확히 알려주는 데 목적이 있다.

**왜 인계가 매끄러운가** — `~/.claude` 디렉터리 전체를 마운트했기 때문에 다음이 호스트와 컨테이너 사이에서 공유된다:
- OAuth 인증 세션 — 컨테이너에서 `claude` 첫 실행 시 재로그인 불필요.
- 글로벌 `~/.claude/CLAUDE.md` 사용자 룰.
- `~/.claude/projects/{현재 프로젝트}/` — 호스트에서 진행한 **이 대화의 히스토리** 와 memory.
- `~/.claude/settings.json` 글로벌 허용 명령·권한.

**인계 멘트** (완료 안내 직후 사용자에게 그대로 출력):

```
[인계 안내]
1. 이 호스트 claude 세션은 종료하세요 (Ctrl+D 또는 exit).
2. 위 명령으로 컨테이너에 진입하세요 (make shell 또는 docker compose ... exec dev bash).
3. 컨테이너 안에서 `claude` 를 실행하세요. 같은 OAuth 세션과 같은 글로벌 룰이 그대로 적용됩니다.
4. 컨테이너 안 claude 에 다음과 같이 입력하면 이 대화를 이어 받습니다:
     "방금 호스트에서 docker-init 까지 마쳤어. /harness 로 첫 phase 설계하자."
   (또는 `claude --continue` / `claude --resume` 으로 직전 세션을 직접 재개)
5. 이후 PRD 수정·코드 작성·테스트는 모두 컨테이너 안 claude 에서 진행합니다.
```

**경계 — 호스트에서 계속해도 되는 작업**:
- `docker compose` 자체 명령 (up/down/logs) 은 호스트에서 실행.
- 그 외 모든 작업(코드·테스트·문서·git 커밋·`/harness` `/review` 등 스킬 호출) 은 컨테이너 안.

**자주 헷갈리는 것**:
- `~/.claude` 가 마운트돼 있으므로 컨테이너 안에서 `/login` 을 다시 할 필요가 없다. 만약 "로그인 안 됨" 이 뜨면 UID/GID 미스매치로 파일 권한이 깨졌을 가능성 — entrypoint 의 `chown` 로직을 확인.
- 컨테이너 안에서 호스트 경로(`/home/사용자명/...`) 는 보이지 않는다. 사용자 코드/문서 참조는 `working_dir` (`/workspace/<프로젝트명>`) 기준.

---

## 주의사항

- **`.claude/skills/docker_examples/` 는 손대지 마라.** 정본 참고 자료. 프로젝트 환경 결정에 영향받지 않음.
- **환경 도커 파일 일체(Dockerfile + docker-compose.yml + entrypoint + .dockerignore) 는 반드시 `env_docker/` 안에 둔다.** 루트에는 환경 도커 파일을 만들지 마라. 이유: 루트는 사용자 프로젝트(자체 Dockerfile / 자체 compose 등) 영역으로 비워둔다. 충돌 방지 + 멘탈 모델 분리.
- **`.env` 는 만들지 마라.** `.env.example` 만 생성. 이유: 시크릿이 git 에 커밋되는 사고 방지. `.env` 는 루트에 둠 (compose 가 `../.env` 로 참조).
- **`CLAUDE.md` 의 `{}` 플레이스홀더는 반드시 실제 값으로 교체하라.** 이유: 플레이스홀더가 남으면 이후 Claude 세션이 잘못된 컨텍스트를 읽는다.
- **기존 `env_docker/` 는 덮어쓰기 전에 사용자 확인.** 직전 버전(루트의 `Dockerfile` / `docker-compose.yml`) 도 같은 archive 로 보낸다.
- **베이스 이미지·서비스를 무비판적으로 예시와 동일하게 두지 마라.** 스택에 따라 처음부터 결정.
- **호스트에 런타임이 있다고 가정하지 마라.** 모든 도구는 dev 컨테이너 안에 설치한다.
- **dev 컨테이너 안에서 또 docker 를 쓰지 마라.** 사이드 서비스는 형제로 추가. DooD 는 마지막 수단.
- **사이드 서비스의 호스트 포트 노출은 기본적으로 하지 마라.** 외부 디버깅이 필요할 때만 사용자에게 묻고 추가.
- **소스 코드는 Dockerfile 에서 COPY 하지 마라.** bind 마운트로 들어온다. COPY 는 `requirements.txt` / `package.json` 같은 의존성 매니페스트만.
- **compose 의 `name:` 필드를 빼먹지 마라.** 안 쓰면 폴더명(`env_docker`)이 project name 으로 잡혀 다른 프로젝트와 충돌·헷갈림 발생.
- **`~/.claude` 마운트 경로의 사용자(non-root vs root)를 정확히 맞춰라.** 컨테이너가 `docker_user` 로 돌면 `/home/docker_user/.claude` 로 마운트. UID/GID 미스매치는 entrypoint `chown` 으로 보정.
- **docs 결정 요약을 사용자에게 확정받기 전에 컨테이너를 만들지 마라.** PRD/ARCH/ADR 의 비어 있는 결정 위에서 만든 컨테이너는 곧 다시 만들게 된다.
- **docker-init 완료 후에는 반드시 인계 안내를 출력하라.** 사용자가 호스트 claude 를 끄고 컨테이너 안 claude 로 옮겨갈 수 있도록.
