# 데이터 계약 (Data Contracts)

## 목적
파이프라인의 **소스 ↔ 파이프라인 ↔ 싱크** 사이에 어떤 스키마와 규칙이 보장되는지 명시한다.
계약이 깨지면 파이프라인이 의미를 잃는다 — 변경은 양쪽 합의 필수.

## 입력 — 소스 측 계약
### 소스 1: {예: 운영 PostgreSQL.orders}
- 형식: 행(row), 키 `id`(bigint), 갱신 시점 `updated_at`(timestamp)
- 필수 컬럼: `id`, `user_id`, `total_amount`, `status`, `created_at`, `updated_at`
- 선택 컬럼: {예: `coupon_id`, `note`}
- 보장 사항: {예: `id` 는 단조 증가 / `updated_at` 은 변경 시 항상 갱신}
- 변경 통보: {예: 운영팀이 PR 라벨 `data-contract` 로 미리 합의}

### 소스 2: {예: S3 로그 events.jsonl}
- 형식: JSON Lines, 한 줄 = 한 이벤트
- 필수 키: `event_id`(uuid), `event_time`(ISO8601), `type`(enum), `payload`(object)
- 분포 가정: {예: 이벤트 시간 ± 5분 지연 99% / type 종류는 미리 합의된 enum 만}

## 출력 — 싱크 측 계약
### 싱크 1: {예: BigQuery analytics.orders_daily}
- 파티셔닝: `dt` (DATE) — 일 단위
- 키: `(dt, order_id)` 유니크
- 컬럼:
  | 이름 | 타입 | 의미 |
  |------|------|------|
  | dt | DATE | 주문 일자 (KST) |
  | order_id | INT64 | 주문 식별자 |
  | total_amount | NUMERIC | 합계 (원) |
  | status | STRING | 주문 상태 enum |
- idempotency: 같은 `(dt, order_id)` 재적재 시 덮어쓰기 (MERGE)
- SLA: 매일 03:00 KST 까지 전일자 데이터 가용

## 스키마 진화 정책
- **호환 변경** (필드 추가, 옵션 필드 → 더 옵션화): 즉시 적용 가능, 소비자에 알림 정도
- **비호환 변경** (필드 제거, 타입 변경, 의미 변경): 새 버전 컬럼/테이블로 출시 → 양쪽 마이그레이션 → 옛 것 폐기
- 비호환 변경 시 소비자 측 (대시보드·다운스트림 ML) 에 최소 N일 사전 통보 — N: {예: 14일}

## 계약 위반 시 동작
- 입력 계약 위반 (스키마 다름·필수 키 누락·타입 불일치):
  1. 해당 레코드 격리 → DLQ 또는 별도 quarantine 테이블
  2. 알림 → {담당자 / 채널}
  3. 파이프라인 자체는 계속 진행 (한 레코드로 멈추지 않는다)
- 출력 검증 실패 (건수·체크섬 불일치):
  1. 적재 롤백
  2. 알림 → [RUNBOOK.md](./RUNBOOK.md) 의 절차 진입

## 검증 체크리스트
- [ ] 모든 소스의 스키마가 `src/schemas/` 에 코드로 표현됨 (Avro / JSON Schema / pydantic)
- [ ] 새 컬럼 추가 시 PR 에 데이터 계약 변경 라벨
- [ ] 비호환 변경 발견 시 사전 통보 절차 수행됨
- [ ] dt 파티션 단위로 재처리 가능 (idempotency)

## 관련 문서
- 시스템 구조: [ARCHITECTURE.md](./ARCHITECTURE.md)
- 장애 대응: [RUNBOOK.md](./RUNBOOK.md)
- 결정 근거: [ADR.md](./ADR.md)
