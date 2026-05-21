# API Reference

## 공개 표면 정의
**공개**: `src/core/` 의 export. SemVer 적용.
**비공개**: `src/internal/`, 언더스코어 prefix (`_helper`), `__all__` 미포함 — 자유 변경.

이 문서는 공개 표면만 다룬다.

## CLI 서브커맨드
### `tool init [path]`
프로젝트를 현재 (또는 지정) 디렉터리에 초기화.

| 인자 | 타입 | 필수 | 기본 | 설명 |
|------|------|------|------|------|
| path | string | × | `.` | 초기화할 디렉터리 |
| `--force` | flag | × | false | 기존 파일 덮어쓰기 |

종료 코드:
- 0: 성공
- 1: 기존 파일 충돌 (--force 없음)
- 2: 입력 오류

### `tool run --config <path>`
{설명 한 줄}

| 인자 | 타입 | 필수 | 기본 | 설명 |
|------|------|------|------|------|
| `--config` | path | ✓ | - | 설정 파일 |
| `--verbose` | flag | × | false | 상세 로그 |

## 라이브러리 API
### `from pkg import process`
```
def process(data: dict, *, strict: bool = False) -> Result:
    """
    {한 줄 요약}

    Args:
        data: {입력 형식}
        strict: True 면 검증 실패 시 ValueError. False 면 best-effort.

    Returns:
        Result 객체 ({필드 1}, {필드 2}, ...).

    Raises:
        ValueError: strict=True 이고 검증 실패 시.
    """
```

### `class Result`
```
@dataclass
class Result:
    {field_1}: str
    {field_2}: int
    errors: list[str]
```

## 호환성 정책
- SemVer 적용: `MAJOR.MINOR.PATCH`
  - MAJOR: 시그니처 제거 / 필드 삭제 / 동작 의미 변경
  - MINOR: 새 함수 / 새 옵션 키 (호환 유지)
  - PATCH: 버그 수정
- 호환 대상 런타임: {예: Python 3.10, 3.11, 3.12}
- deprecation 절차: [MIGRATION.md](./MIGRATION.md) 참고

## 안정성 표시
공개 함수/클래스에 안정성 라벨을 docstring 첫 줄에 표시:
- `[stable]`: 호환 보장
- `[experimental]`: 다음 minor 에 변경 가능 — 사용 시 알아서
- `[deprecated since X.Y]`: 다음 major 에 제거 예정 — 대체재 명시 의무

## 관련 문서
- 시스템 구조: [ARCHITECTURE.md](./ARCHITECTURE.md)
- 사용자 마이그레이션: [MIGRATION.md](./MIGRATION.md)
- 결정 근거: [ADR.md](./ADR.md)
