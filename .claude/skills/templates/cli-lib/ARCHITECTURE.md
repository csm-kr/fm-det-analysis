# 아키텍처

> **이 문서가 답하는 질문**: *공개(`core/`)와 내부(`internal/`) 경계*는 어떻게 그어지며, CLI/Lib 진입점이 어떻게 흐르는가? 공개 API 시그니처는 [API_REFERENCE.md](./API_REFERENCE.md), 사용자 마이그레이션은 [MIGRATION.md](./MIGRATION.md).

## 시스템 구조
```
[진입점]                       [공개 API — SemVer]              [내부 — 자유 변경]
┌──────────────┐
│ CLI          │              ┌────────────────────┐        ┌──────────────────┐
│ argv 파서    │ ───────────▶ │ src/core           │ ─────▶│ src/internal     │
└──────────────┘              │ {공개 함수·클래스}   │        │ {내부 헬퍼}        │ 
                              └────────────────────┘        └──────────────────┘
┌──────────────┐                       │  ▲                          │
│ Library      │ ───────────▶         │  │  ✕  금지: internal → core │
│ from pkg     │                       │  │  (책임 경계 깸)            │
└──────────────┘                       ▼  │
                                    ┌─────────────┐
                                    │ stdout      │
                                    │ 반환값       │
                                    └─────────────┘
                                       │
                                       └─▶ {외부 네트워크 (선택)}
```
> 핵심 규칙: `internal/` → `core/` import 금지. ([API_REFERENCE.md](./API_REFERENCE.md) 의 공개 표면 정의 참고)

## 디렉터리 구조
```
src/
├── cli/               # CLI 진입점, 인자 파싱, 서브커맨드 분배
├── core/              # 공개 API (라이브러리로도 import 가능)
├── internal/          # 내부 구현 (외부 노출 금지)
└── utils/             # 헬퍼

tests/
├── unit/              # 유닛 테스트
└── integration/       # CLI end-to-end 테스트

examples/              # 자기완결 예제
└── {번호}-{slug}/
    ├── README.md
    └── {스크립트}
```

## 호출 흐름
1. 사용자 / 다른 코드: 진입
   - CLI: `tool {subcmd} --opt val ...`
   - Lib: `from pkg import core`
2. CLI 만: argv 파싱 + 서브커맨드 분배
3. 입력 검증
   - 실패 → stderr + non-zero exit (CLI) / Exception (Lib)
4. core 공개 API 호출 ([API_REFERENCE.md](./API_REFERENCE.md) 의 시그니처 그대로)
5. core → internal 구현 사용
6. 결과 반환 → stdout (CLI) / 반환값 (Lib)

호환성 깨는 변경은 [MIGRATION.md](./MIGRATION.md) 의 절차에 따라 deprecation → 제거.

## 공개 vs 내부 경계
- 공개: `src/core/` — SemVer 적용 대상. 시그니처 변경 = breaking
- 내부: `src/internal/` — 자유롭게 변경. 외부에서 import 금지
- 공개 표면 정의: [API_REFERENCE.md](./API_REFERENCE.md)

## 호환성 정책
- 호환 대상: {예: Python 3.10, 3.11, 3.12 / Node 20, 22}
- 테스트 매트릭스: {CI에서 어떤 조합으로 테스트}
- deprecation 절차: {예: 1 minor 동안 경고 → 다음 minor 에서 제거} — 상세: [MIGRATION.md](./MIGRATION.md)

## 외부 의존
| 종류 | 대상 | 용도 |
|------|------|------|
| 런타임 의존 | {예: requests / chalk} | {용도} |
| 빌드 의존 | {예: setuptools / tsup} | {용도} |

## Hook 정책
- 자동 포맷터: {예: ruff format / prettier / rustfmt / 없음}
- 추가 차단 패턴: {예: src/internal/ 에서 src/core/ 로 import 차단 / 없음}
- Stop hook 추가 검증: {예: pytest tests/unit / npm test / cargo test --lib / 없음}

## 관련 문서
- 무엇을·왜: [PRD.md](./PRD.md)
- 결정 근거: [ADR.md](./ADR.md)
- 공개 API 시그니처: [API_REFERENCE.md](./API_REFERENCE.md)
- 사용자 마이그레이션 (버전 간 이주): [MIGRATION.md](./MIGRATION.md)
