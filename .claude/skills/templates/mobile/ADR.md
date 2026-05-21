# Architecture Decision Records

> **이 문서가 답하는 질문**: 왜 *이 프레임워크 / 이 상태관리 / 이 푸시 통합*을 골랐는가? *무엇을·왜* 는 [PRD.md](./PRD.md), 시스템 구조는 [ARCHITECTURE.md](./ARCHITECTURE.md).

## 철학
{예: 크로스플랫폼 일관성 우선 / 네이티브 성능 우선 / 출시 속도 우선}

> **맥락 인용 규칙**: 각 ADR 의 "맥락" 줄은 [PRD.md](./PRD.md) 의 섹션 또는 [ARCHITECTURE.md](./ARCHITECTURE.md) 의 결정을 명시적으로 가리켜야 한다.

---

## ADR-001: {프레임워크 선택 (예: React Native 0.74)}
- **상태**: accepted
- **날짜**: {YYYY-MM-DD}
- **맥락**: {크로스플랫폼 vs 네이티브 트레이드오프, 팀 역량}
- **결정**: {React Native / Flutter / Swift+Kotlin 네이티브}
- **대안**: {대안 + 왜 안 골랐는가}
- **결과**: {제약/이점}

## ADR-002: {상태 관리 (예: Zustand + TanStack Query)}
- **상태**: accepted
- **날짜**: {YYYY-MM-DD}
- **맥락**: {로컬/서버 상태 분리 필요성}
- **결정**: {선택한 라이브러리 조합}
- **대안**: {Redux / MobX / Context only 등}
- **결과**: {제약/이점}

## ADR-003: {푸시 알림 / 백엔드 연동}
- **상태**: accepted
- **날짜**: {YYYY-MM-DD}
- **맥락**: {APNs/FCM 통합 필요성} — [PRD.md](./PRD.md#비기능-요건), [PUSH.md](./PUSH.md) 참고
- **결정**: {Firebase / 자체 / 없음}
- **대안**: {대안 + 이유}
- **결과**: {제약/이점}

## 관련 문서
- 무엇을·왜: [PRD.md](./PRD.md)
- 시스템 구조: [ARCHITECTURE.md](./ARCHITECTURE.md)
- 출시: [RELEASE.md](./RELEASE.md)
- 푸시: [PUSH.md](./PUSH.md)
