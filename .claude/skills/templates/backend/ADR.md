# Architecture Decision Records

> **이 문서가 답하는 질문**: 왜 *이 프레임워크 / 이 DB / 이 인증 모델 / 이 마이그레이션 정책*을 골랐는가? *무엇을·왜* 는 [PRD.md](./PRD.md), 시스템 구조는 [ARCHITECTURE.md](./ARCHITECTURE.md).

## 철학
{예: stateless 우선 / 도메인 격리 우선 / 운영 단순성 우선}

> **맥락 인용 규칙**: 각 ADR 의 "맥락" 줄은 [PRD.md](./PRD.md) 의 섹션 또는 [ARCHITECTURE.md](./ARCHITECTURE.md) 의 결정을 명시적으로 가리켜야 한다.

---

## ADR-001: {프레임워크 / 언어 (예: FastAPI + Python 3.12)}
- **상태**: accepted
- **날짜**: {YYYY-MM-DD}
- **맥락**: {팀 역량, 생태계, 성능 요건}
- **결정**: {프레임워크 + 버전}
- **대안**: {대안 + 왜 안 골랐는가}
- **결과**: {제약/이점}

## ADR-002: {DB / 데이터 저장소 (예: PostgreSQL 16 + Redis 7)}
- **상태**: accepted
- **날짜**: {YYYY-MM-DD}
- **맥락**: {일관성/확장성/지연 요건}
- **결정**: {DB + 버전, 캐시 사용 여부}
- **대안**: {NoSQL, 다른 RDB 등}
- **결과**: {제약/이점}

## ADR-003: {인증 / 인가 (예: JWT + 역할 기반)}
- **상태**: accepted
- **날짜**: {YYYY-MM-DD}
- **맥락**: {사용자 규모, 권한 복잡도} — [PRD.md](./PRD.md#인증--인가), [API_SPEC.md](./API_SPEC.md) 참고
- **결정**: {JWT/세션/OAuth + 권한 모델}
- **대안**: {대안 + 이유}
- **결과**: {제약/이점}

## 관련 문서
- 무엇을·왜: [PRD.md](./PRD.md)
- 시스템 구조: [ARCHITECTURE.md](./ARCHITECTURE.md)
- API 명세: [API_SPEC.md](./API_SPEC.md)
- DB 마이그레이션: [MIGRATIONS.md](./MIGRATIONS.md)
