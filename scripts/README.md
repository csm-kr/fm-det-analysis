# ai-ml scripts — 헬퍼 모듈 가이드

이 디렉터리의 7 파일은 ai-ml 종류 프로젝트의 **ml 통합본 스크립트** 다.

- **정본 위치**: `box-compiler` 의 `.claude/skills/templates/ai-ml/scripts/` (박물관)
- **사용자 프로젝트에 깔리는 위치**: 프로젝트 루트의 `scripts/` — bootstrap §3-B-2 가 ai-ml 종류 선택 시 박물관에서 cp
- 본 README 도 함께 복사되므로 **박물관에서 보든 사용자 프로젝트에서 보든** 같은 정보가 같은 자리에 있다.

SWE default (ai-ml 이 아닌 종류의 `scripts/`) 와 비교하면 **execute.py 가 훨씬 두껍고** + **5 개 헬퍼 모듈 추가** + **test_execute.py 의 ml 전용 테스트** 가 차이.

## 파일 목록

| 파일 | 역할 | SWE default 대비 |
|------|------|----------------|
| `execute.py` | step 순차 실행 + 가드레일 주입 + 자동 커밋 — **ml 통합본** (헬퍼 5 개 활용) | 1차 SWE 정본 + 2차-A/B 통합 |
| `crash_classifier.py` | step 실패 시 stderr/returncode → 10 카테고리 분류 | 🆕 ml 전용 |
| `monitor.py` | `step.monitors` 가 있으면 tensorboard / nvidia-smi 백그라운드 Popen | 🆕 ml 전용 |
| `heartbeat.py` | `phases/{task}/.heartbeat` mtime watchdog — 미갱신 시 SIGTERM | 🆕 ml 전용 |
| `budget.py` | `experiment_budget` 의 총 GPU-시간/디스크 계산 + 임계 검사 | 🆕 ml 전용 |
| `fairness.py` | compare/ablate phase 종료 후 run_dir 의 5 차원 동일 검증 | 🆕 ml 전용 |
| `test_execute.py` | 123 테스트 (121 통과 + 2 jq skip) — 위 6 파일의 단위·통합 | SWE 55 + ml 신규 68 |

## execute.py 가 헬퍼를 부르는 자리

```
execute.py
├── run()
│   ├── _check_blockers()          → 4 + 3 신규 상태 분기
│   ├── _check_budget()            → budget.load_top_budget / compute_budget / format_report
│   └── _execute_all_steps()
│       └── _execute_single_step()
│           ├── _load_guardrails(step)         → step.kind 별 docs 부분집합
│           ├── _build_preamble(...)           → oom/resume/launcher directive 주입
│           ├── _invoke_claude(step, preamble)
│           │   ├── start_monitors(...)        → monitor.py
│           │   ├── HeartbeatWatchdog(...)     → heartbeat.py
│           │   ├── Popen + stdout tee (별도 스레드 → step{N}.log)
│           │   ├── stop_monitors(...)         → monitor.py
│           │   └── watchdog.stop()            → heartbeat.py
│           ├── _index_runs(...)               → runs/ diff → run_dirs/run_metrics
│           ├── _verify_success_metric(...)    → jq 평가 (run_dirs 치환)
│           ├── _handle_checkpoint(...)        → awaiting-review 인터랙티브 prompt
│           └── classify_crash(stderr, ...)    → crash_classifier.py
└── _finalize()
    └── verify_fairness(...)                   → fairness.py (compare/ablate phase 만)
```

## step.json 의 ml 전용 필드 (요약)

execute.py 의 헬퍼들이 의존하는 필드. 자세한 컨벤션은 `templates/ai-ml/skills/harness.md` §5 참고.

| 필드 | 활성화 헬퍼 | 의미 |
|------|----------|------|
| `timeout_sec` | execute.py | 1차 — Popen.wait 의 timeout |
| `max_retries` | execute.py | 1차 — 재시도 횟수 (0 이면 1회) |
| `kind` | _load_guardrails | 2차-A — 가드레일 부분집합 분기 |
| `monitors: [...]` | monitor.py | 2차-B — tensorboard / nvidia-smi 백그라운드 |
| `heartbeat_timeout_sec` | heartbeat.py | 2차-B — watchdog 띄움 |
| `success_metric` | _verify_success_metric | 2차-A — `{run_dir}` 치환 후 jq 평가 |
| `checkpoint: "CP-1\|2\|3"` | _handle_checkpoint | 2차-A — 인터랙티브 prompt |
| `auto_retry_on_oom: true` | execute.py | 2차-A — OOM 한정 1회 추가 시도 |
| `resumable + resume_from` | _build_preamble | 2차-B 잔여 — 재시도 시 ckpt 재개 지시 |
| `launcher: {kind, nproc_per_node, env}` | _build_preamble | 2차-B 잔여 — torchrun wrap 지시 |

task 레벨 (phases/{task}/index.json):
- `eval_spec: {task, metric_primary, metric_lib, ...}` — task 별 메트릭 정의
- `hypothesis`, `baseline_run_id` — 그룹 C 의 가설 추적
- `comparison_targets: [...]`, `ablation_dimensions: [...]` — compare/ablate phase 정의
- `fairness_constraints: {...}` — fairness.py 가 검증할 차원

top-level (phases/index.json):
- `experiment_budget: {datasets, algorithms, ablation_dims, seeds, per_run_hours, ..., gpu_hours_threshold, disk_gb_threshold}` — budget.py 가 검사

## 호출 방법

사용자 프로젝트에서:

```bash
python3 scripts/execute.py {task-name}         # 순차 실행
python3 scripts/execute.py {task-name} --push  # 실행 후 push
```

스크립트는 모두 `scripts/` 안에 함께 있어 import 경로 보정 없이 동작. step.json 의 ml 전용 필드는 위 표를 참고.

## 손대지 않을 것

- `crash_classifier.py` 의 10 카테고리 패턴 순서 — SHM 이 DataLoader 보다 위에 와야 `bus error.*DataLoader` 가 SHM 으로 잡힘.
- `monitor.py` 의 grace_sec=5 — Popen.terminate → wait → kill 순서. 줄이면 종료 미완료 상태로 진행될 위험.
- `heartbeat.py` 의 "파일 없으면 hang 으로 안 봄" 규칙 — 학습 초기화 시간을 hang 으로 잡아버리는 false positive 방지.
- `fairness.py` 의 seed 제외 — 시드 정책은 별도 검증 자리 (시드 세트 평균/표준편차 보고).
