# Architecture Decision Records

> **이 문서가 답하는 질문**: 왜 *이 언어·런타임 / 이 배포 채널 / 이 호환 정책*을 골랐는가? *무엇을·왜* 는 [PRD.md](./PRD.md), 공개 API 표면은 [API_REFERENCE.md](./API_REFERENCE.md).

## 철학
{예: 호환성 깨지지 않게 / 외부 네트워크 없이 동작 / 진입 장벽 최소}

> **맥락 인용 규칙**: 각 ADR 의 "맥락" 줄은 [PRD.md](./PRD.md) / [API_REFERENCE.md](./API_REFERENCE.md) 등의 섹션을 명시적으로 가리켜야 한다.

---

## ADR-001: {언어 / 런타임 (예: Python 3.10+ + Click)}
- **상태**: accepted
- **날짜**: {YYYY-MM-DD}
- **맥락**: {대상 사용자 환경, 생태계, 성능}
- **결정**: {언어 + CLI 라이브러리 + 호환 버전}
- **대안**: {대안 + 왜 안 골랐는가}
- **결과**: {제약/이점}

## ADR-002: {배포 채널 (예: PyPI + GitHub Release)}
- **상태**: accepted
- **날짜**: {YYYY-MM-DD}
- **맥락**: {사용자 설치 패턴, 빌드 자동화}
- **결정**: {PyPI / npm / cargo / Homebrew / 직접 바이너리 — 조합}
- **대안**: {대안 + 이유}
- **결과**: {제약/이점}

## ADR-003: {공개 API 안정성 정책 (예: SemVer + 1 minor deprecation)}
- **상태**: accepted
- **날짜**: {YYYY-MM-DD}
- **맥락**: {기존 사용자 충돌 비용, 변화 속도} — [API_REFERENCE.md#호환성-정책](./API_REFERENCE.md), [MIGRATION.md](./MIGRATION.md) 참고
- **결정**: {SemVer 적용. major 전 deprecation 1 minor 의무. 또는 자체 정책}
- **대안**: {CalVer / 무정책}
- **결과**: {제약/이점}

## 관련 문서
- 무엇을·왜: [PRD.md](./PRD.md)
- 시스템 구조: [ARCHITECTURE.md](./ARCHITECTURE.md)
- 공개 API: [API_REFERENCE.md](./API_REFERENCE.md)
- 사용자 마이그레이션: [MIGRATION.md](./MIGRATION.md)
