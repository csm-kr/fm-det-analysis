# Architecture Decision Records

> **이 문서가 답하는 질문**: 왜 *이 기술 / 이 구조 / 이 정책*을 골랐는가? 검토했지만 안 고른 대안은 무엇인가? *무엇을·왜* 는 [PRD.md](./PRD.md), 시스템 구조는 [ARCHITECTURE.md](./ARCHITECTURE.md).

## 철학
{예: MVP 속도 최우선 / 외부 의존 최소화 / 접근성 기본}

> **맥락 인용 규칙**: 각 ADR 의 "맥락" 줄은 [PRD.md](./PRD.md) 의 섹션 또는 [ARCHITECTURE.md](./ARCHITECTURE.md) 의 결정을 명시적으로 가리켜야 한다. 인용 없는 결정은 미래에 "왜 그렇게 했는지" 답을 잃는다.

---

## ADR-001: {프레임워크 선택 (예: Next.js 15 App Router)}
- **상태**: accepted
- **날짜**: {YYYY-MM-DD}
- **맥락**: {왜 이 결정이 필요했는가 — PRD/제약 인용}
- **결정**: {무엇을 선택}
- **대안**: {검토했지만 채택하지 않은 것 + 이유}
- **결과**: {제약/이점}

## ADR-002: {렌더링 모드 (예: SSR + 부분 SSG)}
- **상태**: accepted
- **날짜**: {YYYY-MM-DD}
- **맥락**: {SEO·초기 로드·인터랙션 요건 인용}
- **결정**: {SSR/SSG/CSR 또는 혼합}
- **대안**: {다른 모드 + 왜 안 골랐는가}
- **결과**: {제약/이점}

## ADR-003: {UI 라이브러리 / 디자인 시스템}
- **상태**: accepted
- **날짜**: {YYYY-MM-DD}
- **맥락**: {디자인 일관성·개발 속도 트레이드오프} — [PRD.md](./PRD.md#디자인-방향), [UI_GUIDE.md](./UI_GUIDE.md) 참고
- **결정**: {Tailwind+ShadCN / MUI / 자체 등}
- **대안**: {대안 + 이유}
- **결과**: {제약/이점}

## 관련 문서
- 무엇을·왜: [PRD.md](./PRD.md)
- 시스템 구조: [ARCHITECTURE.md](./ARCHITECTURE.md)
- 디자인 시스템: [UI_GUIDE.md](./UI_GUIDE.md)
- 접근성: [ACCESSIBILITY.md](./ACCESSIBILITY.md)
