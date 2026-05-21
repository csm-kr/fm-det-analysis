# bootstrap 템플릿 정본

이 디렉터리는 `/bootstrap` 스킬이 사용하는 **PRD / ARCHITECTURE / ADR 템플릿 정본**이다.
프로젝트 종류별 디렉터리 안에 3개 파일이 한 세트로 있고, bootstrap 이 사용자가 고른
종류의 세트를 `docs/` 로 복사한다.

## 디렉터리

| 종류 | 코드 | 대표 예시 |
|------|------|----------|
| 웹 / 풀스택 | `web/` | Next.js, React, SvelteKit, 정적 사이트 |
| 모바일 앱 | `mobile/` | React Native, Flutter, 네이티브 |
| 백엔드 API 서버 | `backend/` | REST/gRPC/GraphQL, FastAPI, Express |
| AI / ML 개발 | `ai-ml/` | 학습/추론/평가, PyTorch, TensorFlow |
| 데이터 파이프라인 | `data-pipeline/` | ETL/스트리밍, Airflow, dbt |
| CLI / 라이브러리 | `cli-lib/` | npm, pip, cargo 배포물 |

> 위 6종에 안 맞으면 `custom` 경로 — bootstrap 이 가장 가까운 base 종류를 추천하고
> ARCHITECTURE.md 의 일부 섹션을 사용자가 직접 다시 작성한다.

## 공용 파일 (종류 무관)

종류별 디렉터리 외에 **모든 프로젝트에 공통**으로 깔리는 파일들이 이 디렉터리 최상위에 있다.

| 파일 | 용도 |
|------|------|
| `PRD_VIEW.md` | PRD 작성·검수용 13개 View 렌즈 (제품 도메인 무관 일반 템플릿). bootstrap 이 종류별 PRD.md 와 함께 `docs/PRD_VIEW.md` 로 복사한다. |

사용 방식: 사용자는 종류별 `PRD.md` 의 본문을 채울 때 `PRD_VIEW.md` 의 13개 View 질문을
체크리스트로 사용한다. 각 View 의 "도메인 적용 가이드" 항목을 자기 제품에 맞춰 구체화한다.

## 편집 규칙

- 각 템플릿은 짧게 유지하라 (~30줄). 너무 길면 "가벼운 스킬" 의도가 깨진다.
- 모든 변수는 `{한국어 이름}` 형태의 중괄호 플레이스홀더로 둔다.
- 본문은 채우지 마라. 섹션 제목과 빈 자리만.
- 각 종류의 `ARCHITECTURE.md` 끝에는 공통 `## Hook 정책` 섹션을 둔다 — 종류별 hook 추천을
  여기서 받는다 (별도 파일을 만들지 않는다).

## 자기 표준 박기

자기 회사·팀 표준 PRD/ARCHITECTURE/ADR 섹션이 있다면 이 디렉터리의 파일을 직접 편집해
정본화하면 된다. bootstrap 은 단순 복사만 하므로 변경 사항이 다음 호출부터 바로 반영된다.
