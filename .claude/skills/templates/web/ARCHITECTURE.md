# 아키텍처

> **이 문서가 답하는 질문**: 이 웹 프로젝트를 *어떻게* 구성·실행하며, 컴포넌트·요청 흐름·외부 의존이 어떻게 연결되는가? *무엇을·왜* 는 [PRD.md](./PRD.md), 결정 근거는 [ADR.md](./ADR.md).

## 시스템 구조
```
[브라우저]                        [서버]
┌──────────────────┐    HTTP    ┌──────────────────┐    ┌─────────────┐
│ Client Component │ ─────────▶ │ Server Component │ ─▶ │ 외부 DB/API │
│ {인터랙션 화면}  │ ◀───────── │      ↓           │    └─────────────┘
└──────────────────┘   초기 HTML │ /api/* (라우트)  │           ▲
        │                       │      ↓           │ ──────────┘
        └──── fetch ──────────▶ │   응답 JSON      │
                                └──────────────────┘
```
> `{}` 부분만 자기 도메인으로 교체. 외부 의존이 늘면 우측에 박스 추가.

## 디렉터리 구조
```
src/
├── app/               # 페이지 + API 라우트 (또는 routes/)
├── components/        # UI 컴포넌트
├── lib/               # 유틸리티, 헬퍼
├── services/          # 외부 API 래퍼
├── types/             # 타입 정의
└── styles/            # 전역 스타일, 토큰
```

## 렌더링 모드
{예: Server Components 기본 + 인터랙션 필요한 곳만 Client Component / SSG / CSR}

## 요청 흐름
1. 사용자: 입력/클릭
2. Client Component: `fetch('/api/{경로}', ...)`
3. API Route: 인증 검증 → 외부 호출 (`{외부 API/DB}`)
4. 외부 응답 → API Route: JSON 직렬화
5. Client Component: 응답 수신 → UI 갱신

에러 분기: 4xx (입력 오류) → toast / 5xx (외부 장애) → 재시도 또는 fallback UI.

## 상태 관리
- 서버 상태: {예: Server Components / TanStack Query}
- 클라이언트 상태: {예: useState/useReducer / Zustand}

## 외부 의존
| 종류 | 대상 | 용도 |
|------|------|------|
| {예: 인증} | {예: NextAuth / 자체 세션} | {용도} |
| {예: DB} | {예: PostgreSQL 16} | {용도} |

## Hook 정책
- 자동 포맷터: {예: prettier --write / biome / 없음}
- 추가 차단 패턴: {예: process.env 직접 접근 차단 / 없음}
- Stop hook 추가 검증: {예: tsc --noEmit / eslint / 없음}

## 관련 문서
- 무엇을·왜: [PRD.md](./PRD.md)
- 결정 근거: [ADR.md](./ADR.md)
- 디자인 시스템 / AI 슬롭 차단: [UI_GUIDE.md](./UI_GUIDE.md)
- 접근성 체크리스트: [ACCESSIBILITY.md](./ACCESSIBILITY.md)
