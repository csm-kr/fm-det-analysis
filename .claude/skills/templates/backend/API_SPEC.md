# API 명세

## 버전 관리
- 정책: {예: URL prefix `/v1/` / 헤더 `Accept-Version: v1`}
- breaking change 시: {예: 새 버전 prefix 출시 + 직전 버전 6개월 유지}

## 공통 규약
- 컨텐츠 타입: `application/json`
- 인증: {예: `Authorization: Bearer {jwt}`}
- 페이지네이션: `?page={n}&size={k}` — 응답에 `total`, `next`
- 정렬: `?sort={field},{asc|desc}` (멀티 키 콤마 분리)
- 필터: `?{field}={value}` — 같음 매칭. 범위는 `?{field}_gte`, `?{field}_lte`
- Rate limit: {예: 인증 사용자 60 req/min, 비인증 20 req/min} — 응답 헤더 `X-RateLimit-*`

## 에러 응답 표준
```json
{
  "error": {
    "code": "{slug_code}",
    "message": "{사용자에게 보일 한 줄}",
    "details": { ... }
  }
}
```
| HTTP | 의미 | code 예시 |
|------|------|----------|
| 400 | 입력 검증 실패 | `validation_failed` |
| 401 | 미인증 | `unauthenticated` |
| 403 | 권한 없음 | `forbidden` |
| 404 | 리소스 없음 | `{도메인}_not_found` |
| 409 | 충돌 | `{도메인}_conflict` |
| 422 | 도메인 위반 | `{규칙명}_violated` |
| 429 | rate limit | `rate_limited` |
| 5xx | 서버 장애 | `internal_error` |

## 엔드포인트 — {도메인 1: 예 Users}
| 메서드 | 경로 | 설명 | 권한 |
|--------|------|------|------|
| POST | /v1/users | 가입 | 비인증 |
| GET | /v1/users/me | 내 정보 | 인증 |
| PATCH | /v1/users/me | 내 정보 수정 | 인증 |

요청 / 응답 스키마는 `src/schemas/` 또는 OpenAPI 문서 (`/openapi.json`) 참고.

## 엔드포인트 — {도메인 2: 예 Orders}
| 메서드 | 경로 | 설명 | 권한 |
|--------|------|------|------|
| GET | /v1/orders | 내 주문 목록 | 인증 |
| POST | /v1/orders | 주문 생성 | 인증 |
| GET | /v1/orders/{id} | 주문 상세 | 본인 / 관리자 |

## 멱등성 (Idempotency)
- 결제·주문 생성 등 부작용 있는 POST: `Idempotency-Key` 헤더 의무
- 같은 키로 재요청 시 같은 응답 반환 (재실행 안 함)
- 키 보관 기간: {예: 24시간}

## 관련 문서
- 시스템 구조: [ARCHITECTURE.md](./ARCHITECTURE.md)
- DB 마이그레이션: [MIGRATIONS.md](./MIGRATIONS.md)
- 결정 근거: [ADR.md](./ADR.md)
