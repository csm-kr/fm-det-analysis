# Migration Guide

사용자가 **이전 버전에서 새 버전으로 코드를 옮길 때 보는 문서**. (내부 deprecation 절차도 포함.)

## Deprecation 절차 (라이브러리 측)
1. 다음 minor 에서 deprecated 라벨 추가 + 대체재 명시
2. 호출 시 `DeprecationWarning` 발생 (라이브러리) 또는 stderr 경고 출력 (CLI)
3. 1 minor 동안 유지 (또는 정책상 N minor)
4. 다음 major 에서 제거 — `MIGRATION.md` 에 "제거" 항목 기록

```
{예: 0.5.0 — deprecated 라벨
0.6.0 — 유지
1.0.0 — 제거}
```

---

## 0.x → 1.0 (예시)

### 변경 1: `process(data, strict)` → `process(data, *, strict)`
- 변경: `strict` 가 키워드 전용 인자로 바뀜
- 영향: `process(d, True)` 같이 위치 인자로 호출하면 깨짐

Before:
```python
result = process(data, True)
```

After:
```python
result = process(data, strict=True)
```

자동 변환: {예: `pip install pkg-codemod && pkg-codemod migrate-1.0 src/` / 없음 — 수동}

### 변경 2: `tool init` 의 `--overwrite` → `--force`
- CLI 옵션 이름 변경
- Before: `tool init . --overwrite`
- After: `tool init . --force`
- 0.x 에서는 두 이름 모두 동작 (deprecation 경고)
- 1.0 에서 `--overwrite` 제거

### 변경 3: 제거된 API
| 제거 | 대체 | 변경 사유 |
|------|------|----------|
| `pkg.legacy_helper()` | `pkg.process(...)` | 통합 |
| `pkg.Old{X}` 클래스 | `pkg.{X}` | 이름 정리 |

---

## 다음 주요 변경 (예고)
{1.x 에서 주의할 deprecation 들 / 비어 있으면 "현재 예고된 변경 없음"}

## 관련 문서
- 공개 API 시그니처: [API_REFERENCE.md](./API_REFERENCE.md)
- 시스템 구조: [ARCHITECTURE.md](./ARCHITECTURE.md)
