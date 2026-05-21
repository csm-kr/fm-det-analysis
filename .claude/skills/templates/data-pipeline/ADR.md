# Architecture Decision Records

> **이 문서가 답하는 질문**: 왜 *이 처리 모드(배치/스트리밍) / 이 오케스트레이터 / 이 저장 포맷*을 골랐는가? *무엇을·왜* 는 [PRD.md](./PRD.md), 시스템 구조는 [ARCHITECTURE.md](./ARCHITECTURE.md).

## 철학
{예: idempotency 최우선 / 배치 우선 / 데이터 품질 게이트 의무}

> **맥락 인용 규칙**: 각 ADR 의 "맥락" 줄은 [PRD.md](./PRD.md) / [DATA_CONTRACTS.md](./DATA_CONTRACTS.md) 등의 섹션을 명시적으로 가리켜야 한다.

---

## ADR-001: {처리 모드 (예: 일 배치 + 시간 단위 증분)}
- **상태**: accepted
- **날짜**: {YYYY-MM-DD}
- **맥락**: {SLA, 데이터 양, 신선도 요건}
- **결정**: {배치 / 스트리밍 / 혼합}
- **대안**: {대안 + 왜 안 골랐는가}
- **결과**: {제약/이점}

## ADR-002: {오케스트레이터 (예: Airflow 2.x)}
- **상태**: accepted
- **날짜**: {YYYY-MM-DD}
- **맥락**: {의존성 복잡도, 운영 비용, 팀 역량}
- **결정**: {Airflow / Prefect / Dagster / cron}
- **대안**: {대안 + 이유}
- **결과**: {제약/이점}

## ADR-003: {저장 포맷 / 파티셔닝 (예: Parquet + 일 파티션)}
- **상태**: accepted
- **날짜**: {YYYY-MM-DD}
- **맥락**: {질의 패턴, 비용, 호환성} — [DATA_CONTRACTS.md#출력--싱크-측-계약](./DATA_CONTRACTS.md), [PRD.md#비기능-요건](./PRD.md) 참고
- **결정**: {포맷 + 파티션 키}
- **대안**: {Avro, ORC, JSON 등}
- **결과**: {제약/이점}

## 관련 문서
- 무엇을·왜: [PRD.md](./PRD.md)
- 시스템 구조: [ARCHITECTURE.md](./ARCHITECTURE.md)
- 데이터 계약: [DATA_CONTRACTS.md](./DATA_CONTRACTS.md)
- 장애 대응: [RUNBOOK.md](./RUNBOOK.md)
