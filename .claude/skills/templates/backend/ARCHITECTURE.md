# 아키텍처

> **이 문서가 답하는 질문**: 이 서비스의 *레이어 구조 / 요청 흐름 / 외부 의존*은 어떻게 되는가? 엔드포인트는 [API_SPEC.md](./API_SPEC.md), 마이그레이션은 [MIGRATIONS.md](./MIGRATIONS.md), *무엇을·왜* 는 [PRD.md](./PRD.md).

## 시스템 구조
```
[클라이언트]      [서비스]                                       [외부]
┌──────────┐    ┌──────────────────────────────────────────┐
│ 웹/모바일│ ─▶ │ LB / Gateway                             │
│ 타시스템 │    │   ↓                                      │
└──────────┘    │ Router (/api/v1/*)                       │
                │   ↓                                      │      ┌─────────────┐
                │ Middleware (인증·로깅·CORS)              │ ──▶  │ 주 DB       │
                │   ↓                                      │      │ {PostgreSQL}│
                │ Service Layer ({핵심 도메인})            │ ◀──  └─────────────┘
                │   ↓                                      │
                │ Repository                               │      ┌─────────────┐
                │   ↓                                      │ ──▶  │ 캐시 {Redis}│
                └──────────────────────────────────────────┘      └─────────────┘
                          │
                          ├─▶ 비동기 큐 {RabbitMQ / SQS / 없음}
                          └─▶ 외부 API {결제·알림 등}
```
> `{}` 만 자기 도메인으로 교체. 큐가 없으면 그 줄 삭제.

## 디렉터리 구조
```
src/
├── api/               # 라우터 / 컨트롤러 (HTTP 진입점)
├── services/          # 도메인 로직
├── repositories/      # DB 접근
├── models/            # 도메인 모델 / 스키마
├── middleware/        # 인증, 로깅, CORS
├── jobs/              # 비동기 작업, 큐 핸들러
└── config/            # 설정, 환경변수
```

## 요청 흐름
1. Client → LB → Router 매칭
2. Middleware: 토큰 검증
   - 실패 → 401 즉시 반환
3. Service: 도메인 호출 ({핵심 도메인 메서드})
4. Repository → DB SQL → row 반환
5. Service: DTO 변환, 캐시 갱신(필요 시)
6. Router → 응답 직렬화 → Client (200/2xx)

에러 분기:
- 입력 검증 실패 → 400 + 에러 코드
- 도메인 위반 → 422 + 메시지
- DB/외부 장애 → 5xx + 재시도 가능 여부 헤더

## 데이터 모델
- 주요 엔티티: {예: User, Order, Product}
- 관계: {1:N / N:M 등 핵심 관계}

## 외부 의존
| 종류 | 대상 | 용도 |
|------|------|------|
| DB | {예: PostgreSQL 16} | {용도} |
| 캐시 | {예: Redis 7} | {용도} |
| 큐 | {예: RabbitMQ / SQS / 없음} | {용도} |
| 외부 API | {예: 결제, 알림} | {용도} |

## 마이그레이션
- 도구: {예: Alembic / Prisma / Flyway / 직접 SQL}
- 정책 / 절차 상세: [MIGRATIONS.md](./MIGRATIONS.md) 참고

## Hook 정책
- 자동 포맷터: {예: ruff format / black + isort / 없음}
- 추가 차단 패턴: {예: 운영 DB URL 커밋 차단 / SELECT * 쿼리 / 없음}
- Stop hook 추가 검증: {예: mypy / pytest tests/unit / 없음}

## 관련 문서
- 무엇을·왜: [PRD.md](./PRD.md)
- 결정 근거: [ADR.md](./ADR.md)
- 엔드포인트 명세: [API_SPEC.md](./API_SPEC.md)
- DB 마이그레이션 정책: [MIGRATIONS.md](./MIGRATIONS.md)
