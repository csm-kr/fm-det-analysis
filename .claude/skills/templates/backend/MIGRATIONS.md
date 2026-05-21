# DB 마이그레이션

## 도구 / 컨벤션
- 도구: {예: Alembic / Prisma / Flyway / 직접 SQL}
- 위치: `db/migrations/{ts}_{slug}.sql` (또는 도구 표준)
- 네이밍: timestamp + 한 줄 동사 슬러그 (예: `20260501_120000_add_orders_status_index.sql`)
- 한 마이그레이션에 한 가지 변경 — 묶어서 올리지 않는다

## 작성 정책
- **forward 전용 vs forward+rollback**: {예: rollback 의무 / forward only}
- rollback 의무인 경우: 같은 파일에 `-- +goose Up` / `-- +goose Down` 또는 별도 파일
- DDL 은 작은 단위로 — 큰 테이블의 ALTER 는 분할 (PostgreSQL 의 경우 `CONCURRENTLY` 활용)
- DROP / RENAME 은 즉시 적용하지 마라 — N+1 단계로 나눠 호환 유지

## 운영 적용 절차
1. 로컬 DB 에서 forward + rollback 동작 확인
2. PR 에 `migration` 라벨 — 리뷰어 별도 지정
3. 머지 후 staging 에 자동 적용, 헬스체크 통과 확인
4. 운영 적용: {예: 수동 승인 게이트 / 배포 파이프라인 자동}
5. 적용 후 30분 이내 모니터링 (에러율, 쿼리 시간)

## 큰 테이블 변경 패턴
- 컬럼 추가: nullable + 기본값 → 백필 → NOT NULL 전환 (3 마이그레이션)
- 컬럼 삭제: 코드에서 더 이상 안 쓰게 변경 → 한 배포 후 → 컬럼 삭제 (2 마이그레이션)
- 인덱스 추가: PostgreSQL 은 `CREATE INDEX CONCURRENTLY`
- 데이터 타입 변경: 새 컬럼 추가 → 백필 → 코드 전환 → 옛 컬럼 삭제

## 시드 데이터
- 위치: {예: `db/seeds/{env}/`}
- 적용 시점: {예: docker compose up 시 entrypoint / 명시적 명령}
- 운영 환경에는 적용 안 함

## 비상 절차
- 적용 실패 → 자동 롤백? {예: 수동만 / 도구의 자동 트랜잭션 의존}
- 운영 중 잘못 들어간 마이그레이션:
  1. 즉시 모니터링 / 영향 범위 측정
  2. 새 forward 마이그레이션으로 복구 (DB 상태를 직접 SQL 로 만지지 마라)
  3. 사후 [ADR.md](./ADR.md) 에 "사고 #N — 마이그레이션 X" 결정 기록

## 관련 문서
- 시스템 구조: [ARCHITECTURE.md](./ARCHITECTURE.md)
- API 명세 (영향 받는 엔드포인트): [API_SPEC.md](./API_SPEC.md)
- 결정 근거: [ADR.md](./ADR.md)
