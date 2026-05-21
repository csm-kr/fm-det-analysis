# Claude Code Hooks (이 레포)

이 레포의 `.claude/settings.json` 에 박힌 hook 2개의 의도와 동작만 정리한다.
hook 시스템 자체의 매뉴얼(이벤트 종류·페이로드·exit code·matcher·베스트 프랙티스)은
공식 문서에 있고 이 레포와 함께 stale 되지 않는 곳이라 여기 박지 않는다.

## 정본 hook 2개

### 1. PreToolUse (Bash) — 위험 명령 차단

- 매처: `Bash`
- 패턴: `rm -rf`, `git push --force`, `git reset --hard`, `DROP TABLE`
- 동작: 패턴 매칭 시 stderr 메시지 + `exit 2` 로 도구 실행 차단
- 의도: 손이 닿기 전에 막는 안전 차단봉. Claude 가 우발적으로 위험 명령을 실행하는 사고 방지.

### 2. Stop — 자동 검증

- 실행: `python3 scripts/test_execute.py`
- 동작: 매 응답이 끝날 때마다 테스트 실행. 실패 시 exit code 그대로 전달해 Claude 가 알아챔.
- 의도: "테스트 통과 전엔 멈추지 마"를 강제하는 품질 게이트.

---

## 추가하려면

- **팀 공유** — `.claude/settings.json` 에 hook 추가. git 추적됨.
- **개인 환경** — `.claude/settings.local.json` 에. gitignore 권장 (예: 데스크톱 알림, Slack DM).
- **종류별 권장 hook** — `docs/ARCHITECTURE.md` 의 `## Hook 정책` 섹션이 정합 자리. 자동 포맷터·추가 차단 패턴·Stop 추가 검증을 거기서 결정한다.

---

## 참고

- 공식 문서: https://docs.anthropic.com/claude-code/hooks
- 관련 파일: `.claude/settings.json`, `scripts/test_execute.py`
