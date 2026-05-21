# harness_framework

이 레포지토리는 **하네스(Harness)를 만들기 위한 뼈대**입니다.
실제 하네스 로직이 아니라, 하네스를 구성·실행·검증하는 데 필요한 공통 구조와 도구를 제공합니다.

> "하네스" = 모델·스크립트·실험 코드를 **재현 가능한 환경에서 일관된 방식으로 실행·검증**하기 위한 외피(Wrapper).

---

## Quick Start — 텅 빈 디렉터리에서 시작

새 프로젝트 디렉터리에서 다음 6단계를 따릅니다.

| 단계 | 무엇 | 어디서 |
|------|------|------|
| 1. 클론 | 빈 디렉터리에 이 뼈대 정본을 깐다 | 호스트 셸 (아래 코드블록) |
| 2. `/bootstrap` | 종류별 docs 뼈대를 `docs/` 에 깐다 | Claude Code |
| 3. docs 채우기 | `PRD.md → ARCHITECTURE.md → ADR.md` 순서로 본문 채움 | Claude 와 함께 |
| 4. `/docker-init` | 종류·스택에 맞는 격리 환경을 `env_docker/` 안에 생성 — `Dockerfile` (멀티스테이지) + `docker-compose.yml` + 사이드 서비스 형제 | Claude Code |
| 5. 컨테이너 진입 | **VS Code/Cursor Dev Containers (권장)** 또는 호스트 셸로 직접 | 호스트 IDE / 셸 |
| 6. `/harness` → `execute.py` | 첫 phase 설계 → step 순차 실행 | 컨테이너 안의 Claude Code |

전체 흐름 (호스트 셸 + Claude Code 슬래시 명령):

```bash
# ─── 1. 클론 (호스트 셸) ─────────────────────────────────────────
git clone https://github.com/csm-kr/harness_framework .
rm -rf .git
git init


# ─── 2~4. 호스트에서 Claude Code 띄워 슬래시 명령 ────────────────
claude            # 호스트 Claude Code 세션 시작 (IDE 의 Claude 확장도 OK)

#   Claude 안에서 차례대로:
#     /bootstrap            → 2. 종류·docs/ 뼈대 결정 (대안 제시 형태)
#                             3. docs/PRD.md → ARCHITECTURE.md → ADR.md 본문을
#                                Claude 와 함께 채움 (이 단계가 사실상 가장 오래 걸림)
#     /docker-init          → 4. env_docker/{Dockerfile,docker-compose.yml,...} 생성
#
#   끝나면 호스트 Claude 세션은 종료해도 됨 (대화 히스토리는 컨테이너로 이어지지 않음).


# ─── 5. 컨테이너 진입 — 두 가지 중 하나 ──────────────────────────

# (A) Dev Containers 확장 (권장 — IDE 자체를 컨테이너에 붙임)
#   VS Code / Cursor 에서:
#   ① "Dev Containers" 확장 설치 (없으면)
#   ② 명령 팔레트 → "Dev Containers: Reopen in Container"
#   ③ env_docker/docker-compose.yml 의 dev 서비스를 선택
#   → 터미널·언어 서버·디버거가 모두 컨테이너 환경에서 동작.
#     아래 (B) 의 docker compose up/exec 도 자동으로 처리됨.

# (B) 호스트 셸로 직접
docker compose -f env_docker/docker-compose.yml up -d --build       # 백그라운드 띄움
docker compose -f env_docker/docker-compose.yml exec dev bash       # 셸로 접속
# (또는 /docker-init 이 함께 만든 Makefile 로: make up && make shell)


# ─── 6. 컨테이너 안에서 Claude 새 세션 시작 → /harness → execute.py ─
claude                       # 컨테이너 Claude Code (호스트 세션과 별개)

#   Claude 안에서:
#     /harness               → 첫 phase 설계 (phases/{task}/step{N}.md 생성)
#     python3 scripts/execute.py {task}   → step 순차 실행 + 자동 커밋
```

요점:
- `/bootstrap` 은 프로젝트명·한 줄 목적·종류(`web` / `mobile` / `backend` / `ai-ml` / `data-pipeline` / `cli-lib` / `custom`)를 **대안 제시 형태**로 묻고, 고른 종류에 맞춰 `docs/` 에 PRD/ARCHITECTURE/ADR + 추가 docs 를 깐다.
- 각 docs 맨 위 "이 문서가 답하는 질문" 헤더가 무엇을 적을지 안내. 본문은 `{}` 플레이스홀더로 남아 있고 사용자가 Claude 와 함께 채운다.
- 종류별 추가 docs (예: `web` → `UI_GUIDE.md`, `ai-ml` → `DATA_CARD.md`) 도 같은 방식.
- **단계 1~4 는 호스트의 Claude Code 세션** (단순 파일 생성이라 호스트엔 Docker 만 있어도 OK). **단계 5 부터는 컨테이너 안의 새 Claude Code 세션** — 두 세션은 별개이므로 컨테이너 첫 메시지에서 `docs/` 와 `CLAUDE.md` 를 다시 읽도록 지시 (`/harness` 가 자동 처리).
- IDE 관점: 파일은 bind mount 라 호스트 IDE 에서도 그대로 보이지만 빌드·테스트·`claude` 실행은 반드시 컨테이너 안. **Dev Containers 확장으로 IDE 자체를 컨테이너에 붙이는 게 가장 깔끔** — 터미널·확장·디버거가 모두 컨테이너 환경에서 일관되게 동작.
- `exec` 로 들어간 셸은 `exit` 해도 컨테이너는 살아 있다. 작업을 완전히 마쳤을 때만 `docker compose -f env_docker/docker-compose.yml down` (또는 `make down`).
- **호스트는 Docker 만 있다고 가정**. Python·Node 등 모든 런타임·도구는 dev 컨테이너 안에 살고, `claude` 도 그 안에서 실행한다. 호스트에서 `claude` 를 띄울 거라면 `.claude/skills/docker-init.md` 의 "Stop hook 옵션 B" 를 참고.

---

## 멘탈 모델 — 이 레포는 "사람"처럼 구성돼 있습니다

각 파일·폴더의 역할을 신체 비유로 보면 한눈에 잡힙니다.

| 비유 | 자산 | 역할 |
|------|------|------|
| <sub>📜&nbsp;**헌법**</sub> | <sub>`CLAUDE.md`</sub> | <sub>프로젝트가 반드시 따라야 할 원칙 (종류·스택·CRITICAL). 모든 세션에 자동 주입 → 우선순위 1번.</sub> |
| <sub>🤝&nbsp;**협업&nbsp;원칙**</sub> | <sub>`LLM_GUIDE.md`</sub> | <sub>Claude/LLM 4원칙: Think Before Coding / Simplicity First / Surgical Changes / Goal-Driven Execution. `CLAUDE.md` 가 참고.</sub> |
| <sub>🧠&nbsp;**뇌&nbsp;(의도·지식)**</sub> | <sub>`docs/PRD.md`, `ARCHITECTURE.md`, `ADR.md` (+ 종류별 추가 docs)</sub> | <sub>"무엇을 / 누구에게 / 왜 / 어떻게 / 왜 이 결정". 다이어그램·cross-reference 포함.</sub> |
| <sub>📋&nbsp;**계획서**</sub> | <sub>`phases/index.json`, `phases/{task}/step{N}.md`</sub> | <sub>"무엇을, 어떤 순서로". `/harness` 가 생성, `execute.py` 가 step 씩 실행.</sub> |
| <sub>🦾&nbsp;**손&nbsp;(실행)**</sub> | <sub>`scripts/execute.py`, `scripts/test_execute.py`</sub> | <sub>step 순차 실행 + 자가 교정 + 자동 커밋 + 자동 테스트.</sub> |
| <sub>🧬&nbsp;**신경계&nbsp;(자동&nbsp;반응)**</sub> | <sub>`.claude/settings.json`</sub> | <sub>`PreToolUse` 위험 명령 차단 / `Stop` 시 자동 테스트. ([HOOKS.md](docs/HOOKS.md))</sub> |
| <sub>🎓&nbsp;**습관&nbsp;(반복&nbsp;능력)**</sub> | <sub>`.claude/skills/{bootstrap,docker-init,harness,review}.md` + `templates/{type}/`</sub> | <sub>슬래시 스킬 4종 + 종류별 템플릿 정본. ai-ml 종류면 bootstrap 이 `templates/ai-ml/{scripts,skills}/*` 를 사용자 프로젝트로 덮어써 `/harness` 가 ml 통합본으로 동작.</sub> |
| <sub>🏠&nbsp;**환경&nbsp;(몸이&nbsp;사는&nbsp;곳)**</sub> | <sub>`env_docker/{Dockerfile,docker-compose.yml,...}` (생성물) ← `.claude/skills/docker_examples/` (참고 예시)</sub> | <sub>격리된 컨테이너. `/docker-init` 이 예시 패턴을 참고해 종류·스택에 맞게 dev 컨테이너 + 사이드 서비스를 형제로 생성. 환경 도커 파일 일체가 `env_docker/` 한 폴더에 — 루트는 사용자 프로젝트 영역으로 비워둠. **호스트는 Docker 만 있다고 가정** — 모든 런타임·도구는 컨테이너 안.</sub> |

> 충돌이 생기면 항상 헌법(`CLAUDE.md`)이 우선합니다.

---

## 🔄 표준 작업 흐름

```
0. 텅 빈 디렉터리 클론 (Quick Start 0️⃣)
        ↓
1. /bootstrap          → docs/ 에 종류별 뼈대 깔림
        ↓ 사용자가 docs/PRD → ARCH → ADR 본문 채움 (Claude 와 함께)
        ↓
2. /docker-init        → env_docker/{Dockerfile, docker-compose.yml, ...} 생성
        ↓
3. /harness            → phases/{task}/step{N}.md 생성
        ↓
4. execute.py {task}   → step 순차 실행 + 자동 테스트 + 자동 커밋
        ↓
완료
```

| 단계 | 누가 | 결과물 |
|------|------|--------|
| 0. 클론 | 사람 | 정본이 깔린 빈 프로젝트 |
| 1. `/bootstrap` | Claude (대안 제시 + 사용자 선택) | 종류별 `docs/` 뼈대 |
| 1.5 docs 본문 | 사람 + Claude | 채워진 PRD/ARCH/ADR + 추가 docs |
| 2. `/docker-init` | Claude | 종류·스택에 맞는 컨테이너 환경 |
| 3. `/harness` | Claude (사람 검토·승인) | `phases/{task}/step{N}.md` |
| 4. `execute.py` | 자동화 스크립트 | `feat-{task}` 브랜치의 step별 커밋 |

---

## 📁 구성

| 경로 | 역할 |
|------|------|
| `CLAUDE.md` | 프로젝트 헌법 (플레이스홀더) — 종류·기술 스택·CRITICAL 규칙 |
| `LLM_GUIDE.md` | Claude/LLM 코딩 4원칙 — `CLAUDE.md` 가 참고 |
| `docs/PRD.md` | "무엇을·왜" — bootstrap 이 종류별로 깐 후 사용자가 채움 |
| `docs/ARCHITECTURE.md` | "어떻게" — 시스템 구조·흐름·다이어그램·외부 의존 |
| `docs/ADR.md` | "왜 이 결정" — 트레이드오프 기록 |
| `docs/HOOKS.md` | 이 레포의 정본 hook 2개 설명 + 공식 문서 링크 |
| `docs/{종류별 추가}` | 예: web → `UI_GUIDE.md`, `ACCESSIBILITY.md` / ai-ml → `DATA_CARD.md`, `MODEL_CARD.md`, `EVAL_PROTOCOL.md`, `EXPERIMENTS.md` |
| `env_docker/{Dockerfile,docker-compose.yml,.dockerignore,docker-entrypoint.sh}` | 환경 도커 파일 일체 (멀티스테이지 + 서비스 토폴로지). `/docker-init` 이 생성. 기본 클론에는 없음. 루트는 사용자 프로젝트 자체 도커 영역으로 비워둠 |
| `Makefile` (루트, 옵션) | `make up` / `make shell` 로 `docker compose -f env_docker/docker-compose.yml ...` 단축 |
| `scripts/execute.py` | step 순차 실행 + 자가 교정 + 자동 커밋 |
| `scripts/test_execute.py` | pytest 기반 자동 검증 (Stop hook 이 매 응답 종료 시 실행) |
| `phases/{task}/...` | `/harness` 가 생성한 step 파일과 진행 상태 |
| `.claude/settings.json` | Claude Code hooks (PreToolUse 차단, Stop 자동 검증) |
| `.claude/skills/{bootstrap,docker-init,harness,review}.md` | 슬래시 스킬 정본 (SWE default — ai-ml 종류면 bootstrap 이 `templates/ai-ml/skills/harness.md` 로 덮어쓰기) |
| `.claude/skills/templates/ai-ml/scripts/{execute,test_execute,crash_classifier}.py` | ml 통합본 정본 박물관. bootstrap 이 ai-ml 선택 시 `scripts/` 로 덮어쓰기 |
| `.claude/skills/templates/ai-ml/skills/harness.md` | ml 라이프사이클 9 단계 정본. bootstrap 이 ai-ml 선택 시 `.claude/skills/harness.md` 로 덮어쓰기 |
| `.claude/skills/templates/{type}/` | bootstrap 이 종류별로 `docs/` 에 복사하는 정본 (PRD/ARCH/ADR + 추가 docs) |
| `.claude/skills/docker_examples/{Dockerfile,docker-compose.yml}` | `/docker-init` 이 참고하는 컨테이너 예시 (정본 — 손대지 않음). 환경은 프로젝트마다 달라지므로 패턴만 참고 |

---

## 📐 개발 원칙

- **TDD** — 새 기능은 테스트 먼저, 그다음 구현. (`CLAUDE.md` 의 CRITICAL 규칙)
- **LLM 협업 4원칙** — Think Before Coding / Simplicity First / Surgical Changes / Goal-Driven Execution. ([LLM_GUIDE.md](./LLM_GUIDE.md) 참고)
- **커밋 메시지** — Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
- **언어** — 모든 코드·주석·문서·커밋 메시지는 한국어.

---

## 🎯 목적

이 뼈대를 기반으로 각 프로젝트에 맞는 하네스를 추가·확장해 나가는 것을 전제로 합니다.
세부 설계와 결정 사항은 `docs/` 디렉터리의 문서를 참고하세요.
