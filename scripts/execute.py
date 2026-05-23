#!/usr/bin/env python3
"""
Harness Step Executor — phase 내 step을 순차 실행하고 자가 교정한다.

Usage:
    python3 scripts/execute.py <phase-dir> [--push]
"""

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import types
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from crash_classifier import classify as classify_crash  # noqa: E402
from monitor import start_monitors, stop_monitors  # noqa: E402
from heartbeat import HeartbeatWatchdog  # noqa: E402
from budget import load_top_budget, compute_budget, format_report  # noqa: E402
from fairness import verify as verify_fairness, format_report as format_fairness  # noqa: E402


@contextlib.contextmanager
def progress_indicator(label: str):
    """터미널 진행 표시기. with 문으로 사용하며 .elapsed 로 경과 시간을 읽는다."""
    frames = "◐◓◑◒"
    stop = threading.Event()
    t0 = time.monotonic()

    def _animate():
        idx = 0
        while not stop.wait(0.12):
            sec = int(time.monotonic() - t0)
            sys.stderr.write(f"\r{frames[idx % len(frames)]} {label} [{sec}s]")
            sys.stderr.flush()
            idx += 1
        sys.stderr.write("\r" + " " * (len(label) + 20) + "\r")
        sys.stderr.flush()

    th = threading.Thread(target=_animate, daemon=True)
    th.start()
    info = types.SimpleNamespace(elapsed=0.0)
    try:
        yield info
    finally:
        stop.set()
        th.join()
        info.elapsed = time.monotonic() - t0


class StepExecutor:
    """Phase 디렉토리 안의 step들을 순차 실행하는 하네스."""

    MAX_RETRIES = 3
    DEFAULT_TIMEOUT_SEC = 1800
    FEAT_MSG = "feat({phase}): step {num} — {name}"
    CHORE_MSG = "chore({phase}): step {num} output"
    TZ = timezone(timedelta(hours=9))

    def __init__(self, phase_dir_name: str, *, auto_push: bool = False):
        self._root = str(ROOT)
        self._phases_dir = ROOT / "phases"
        self._phase_dir = self._phases_dir / phase_dir_name
        self._phase_dir_name = phase_dir_name
        self._top_index_file = self._phases_dir / "index.json"
        self._auto_push = auto_push

        if not self._phase_dir.is_dir():
            print(f"ERROR: {self._phase_dir} not found")
            sys.exit(1)

        self._index_file = self._phase_dir / "index.json"
        if not self._index_file.exists():
            print(f"ERROR: {self._index_file} not found")
            sys.exit(1)

        idx = self._read_json(self._index_file)
        self._project = idx.get("project", "project")
        self._phase_name = idx.get("phase", phase_dir_name)
        self._total = len(idx["steps"])

    def run(self):
        self._print_header()
        self._check_blockers()
        self._check_budget()
        self._checkout_branch()
        guardrails = self._load_guardrails()
        self._ensure_created_at()
        self._execute_all_steps(guardrails)
        self._finalize()

    def _check_budget(self):
        """top-level experiment_budget 이 있으면 임계 검사. 초과 시 confirm prompt."""
        budget = load_top_budget(self._top_index_file)
        if not budget:
            return
        report = compute_budget(budget)
        print("\n  [Budget]")
        print(format_report(report))
        if not report.any_over:
            return
        if not sys.stdin.isatty():
            print(f"  ⚠ budget 임계 초과 — TTY 미감지로 자동 진행 불가. 임계 조정 후 재실행.")
            sys.exit(3)
        try:
            resp = input("    → 임계 초과. 그래도 진행? [y/N]: ").strip().lower()
        except EOFError:
            resp = ""
        if resp not in ("y", "yes"):
            print("    중단.")
            sys.exit(1)
        print("    ✓ 사용자 승인 — 진행.")

    # --- timestamps ---

    def _stamp(self) -> str:
        return datetime.now(self.TZ).strftime("%Y-%m-%dT%H:%M:%S%z")

    # --- JSON I/O ---

    @staticmethod
    def _read_json(p: Path) -> dict:
        return json.loads(p.read_text(encoding="utf-8"))

    @staticmethod
    def _write_json(p: Path, data: dict):
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- git ---

    def _run_git(self, *args) -> subprocess.CompletedProcess:
        cmd = ["git"] + list(args)
        return subprocess.run(cmd, cwd=self._root, capture_output=True, text=True)

    def _checkout_branch(self):
        branch = f"feat-{self._phase_name}"

        r = self._run_git("rev-parse", "--abbrev-ref", "HEAD")
        if r.returncode != 0:
            print(f"  ERROR: git을 사용할 수 없거나 git repo가 아닙니다.")
            print(f"  {r.stderr.strip()}")
            sys.exit(1)

        if r.stdout.strip() == branch:
            return

        r = self._run_git("rev-parse", "--verify", branch)
        r = self._run_git("checkout", branch) if r.returncode == 0 else self._run_git("checkout", "-b", branch)

        if r.returncode != 0:
            print(f"  ERROR: 브랜치 '{branch}' checkout 실패.")
            print(f"  {r.stderr.strip()}")
            print(f"  Hint: 변경사항을 stash하거나 commit한 후 다시 시도하세요.")
            sys.exit(1)

        print(f"  Branch: {branch}")

    def _commit_step(self, step_num: int, step_name: str):
        output_rel = f"phases/{self._phase_dir_name}/step{step_num}-output.json"
        index_rel = f"phases/{self._phase_dir_name}/index.json"

        self._run_git("add", "-A")
        self._run_git("reset", "HEAD", "--", output_rel)
        self._run_git("reset", "HEAD", "--", index_rel)

        if self._run_git("diff", "--cached", "--quiet").returncode != 0:
            msg = self.FEAT_MSG.format(phase=self._phase_name, num=step_num, name=step_name)
            r = self._run_git("commit", "-m", msg)
            if r.returncode == 0:
                print(f"  Commit: {msg}")
            else:
                print(f"  WARN: 코드 커밋 실패: {r.stderr.strip()}")

        self._run_git("add", "-A")
        if self._run_git("diff", "--cached", "--quiet").returncode != 0:
            msg = self.CHORE_MSG.format(phase=self._phase_name, num=step_num)
            r = self._run_git("commit", "-m", msg)
            if r.returncode != 0:
                print(f"  WARN: housekeeping 커밋 실패: {r.stderr.strip()}")

    # --- top-level index ---

    def _update_top_index(self, status: str):
        if not self._top_index_file.exists():
            return
        top = self._read_json(self._top_index_file)
        ts = self._stamp()
        for phase in top.get("phases", []):
            if phase.get("dir") == self._phase_dir_name:
                phase["status"] = status
                ts_key = {"completed": "completed_at", "error": "failed_at", "blocked": "blocked_at"}.get(status)
                if ts_key:
                    phase[ts_key] = ts
                break
        self._write_json(self._top_index_file, top)

    # --- guardrails & context ---

    # step.kind 별로 docs/ 의 부분집합만 주입 (None 이면 전체).
    # 모든 kind 에 CLAUDE.md 는 항상 포함.
    KIND_DOC_FILTERS = {
        "data-sanity": {"PRD", "DATA_CARD"},
        "model-sanity": {"PRD", "ARCHITECTURE", "MODEL_CARD"},
        "loss-sanity": {"PRD", "ARCHITECTURE"},
        "inference-bench": {"MODEL_CARD", "EVAL_PROTOCOL"},
        "quant-bench": {"MODEL_CARD", "EVAL_PROTOCOL"},
        # "code" / "experiment" / "compare" / "ablate" 는 전체 (None)
    }

    def _load_guardrails(self, step: Optional[dict] = None) -> str:
        sections = []
        claude_md = ROOT / "CLAUDE.md"
        if claude_md.exists():
            sections.append(f"## 프로젝트 규칙 (CLAUDE.md)\n\n{claude_md.read_text()}")

        kind = (step or {}).get("kind") if step else None
        allow = self.KIND_DOC_FILTERS.get(kind) if kind else None

        docs_dir = ROOT / "docs"
        if docs_dir.is_dir():
            for doc in sorted(docs_dir.glob("*.md")):
                if allow is not None and doc.stem not in allow:
                    continue
                sections.append(f"## {doc.stem}\n\n{doc.read_text()}")
        return "\n\n---\n\n".join(sections) if sections else ""

    # --- runs/ 자동 인덱싱 ---

    def _snapshot_runs(self) -> set:
        runs_dir = ROOT / "runs"
        if not runs_dir.is_dir():
            return set()
        return {p.name for p in runs_dir.iterdir() if p.is_dir()}

    def _index_runs(self, index: dict, step_num: int, before: set) -> list:
        """step 종료 후 새로 생긴 runs/ 디렉터리를 step 항목에 기록."""
        after = self._snapshot_runs()
        new_dirs = sorted(after - before)
        if not new_dirs:
            return []
        for s in index["steps"]:
            if s["step"] == step_num:
                s["run_dirs"] = new_dirs
                metrics = {}
                for rd in new_dirs:
                    m = self._extract_run_metric(ROOT / "runs" / rd)
                    if m:
                        metrics[rd] = m
                if metrics:
                    s["run_metrics"] = metrics
                break
        return new_dirs

    @staticmethod
    def _extract_run_metric(run_dir: Path) -> Optional[dict]:
        """run_dir 의 eval.json 또는 bench.json 에서 핵심 지표 한 줄 snapshot."""
        for name in ("eval.json", "bench.json"):
            f = run_dir / name
            if not f.exists():
                continue
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            keys = ("metric_primary", "latency_p95_ms", "vram_peak_mb", "fps", "metric_drop")
            return {k: data[k] for k in keys if k in data}
        return None

    # --- success_metric 자동 검증 ---

    def _verify_success_metric(self, index: dict, step_num: int) -> tuple:
        """step.success_metric (jq 표현) 가 있으면 {run_dir} 치환 후 평가.
        반환: (passed, message). success_metric 없으면 (True, '').
        """
        step_record = next((s for s in index["steps"] if s["step"] == step_num), {})
        expr = step_record.get("success_metric")
        if not expr:
            return True, ""

        run_dirs = step_record.get("run_dirs") or []
        needs_run_dir = "{run_dir}" in expr
        if needs_run_dir and not run_dirs:
            return False, f"success_metric 정의됐으나 runs/ 산출물 없음: '{expr}'"

        if "jq " in expr and shutil.which("jq") is None:
            return True, "jq not installed — success_metric 검증 skip"

        # {run_dir} 변수를 첫 번째 신규 run_dir 로 치환 (있을 때만)
        cmd = expr.replace("{run_dir}", f"runs/{run_dirs[0]}") if run_dirs else expr
        result = subprocess.run(cmd, cwd=self._root, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            return False, f"success_metric 미달: {cmd} → exit {result.returncode} | {result.stdout.strip()} {result.stderr.strip()}".strip()
        return True, ""

    # --- checkpoint (awaiting-review) ---

    def _handle_checkpoint(self, index: dict, step_num: int, cp_label: str, step_name: str) -> str:
        """checkpoint step 완료 후 사용자 검토 prompt. 반환: 'approved' / 'rejected' / 'awaiting-review'."""
        step_record = next((s for s in index["steps"] if s["step"] == step_num), {})
        ts = self._stamp()

        if not sys.stdin.isatty():
            for s in index["steps"]:
                if s["step"] == step_num:
                    s["status"] = "awaiting-review"
                    s["awaiting_review_at"] = ts
                    s["checkpoint"] = cp_label
            self._write_json(self._index_file, index)
            print(f"\n  ⏸ [CHECKPOINT {cp_label}] Step {step_num} ({step_name}) 검토 대기 (TTY 미감지).")
            print(f"  phases/{self._phase_dir_name}/index.json 의 status 를 'approved' 또는 'rejected' 로 변경 후 재실행하세요.")
            return "awaiting-review"

        print(f"\n  ⏸ [CHECKPOINT {cp_label}] Step {step_num} ({step_name}) 완료. 검토:")
        if step_record.get("run_dirs"):
            print(f"    산출물: {', '.join(step_record['run_dirs'])}")
        if step_record.get("run_metrics"):
            print(f"    메트릭: {json.dumps(step_record['run_metrics'], ensure_ascii=False)}")
        if step_record.get("summary"):
            print(f"    요약: {step_record['summary']}")

        try:
            resp = input(f"    → 이 step 을 approve 하시겠습니까? [y/N]: ").strip().lower()
        except EOFError:
            resp = ""
        approved = resp in ("y", "yes")

        new_status = "approved" if approved else "rejected"
        for s in index["steps"]:
            if s["step"] == step_num:
                s["status"] = new_status
                s[f"{new_status}_at"] = ts
                s["checkpoint"] = cp_label
        self._write_json(self._index_file, index)
        print(f"    {'✓ approved' if approved else '✗ rejected'} (step {step_num}).")
        return new_status

    @staticmethod
    def _build_step_context(index: dict) -> str:
        lines = [
            f"- Step {s['step']} ({s['name']}): {s['summary']}"
            for s in index["steps"]
            if s["status"] == "completed" and s.get("summary")
        ]
        if not lines:
            return ""
        return "## 이전 Step 산출물\n\n" + "\n".join(lines) + "\n\n"

    def _build_preamble(self, guardrails: str, step_context: str,
                        prev_error: Optional[str] = None,
                        max_attempts: Optional[int] = None,
                        oom_directive: Optional[str] = None,
                        resume_directive: Optional[str] = None,
                        launcher_directive: Optional[str] = None) -> str:
        commit_example = self.FEAT_MSG.format(
            phase=self._phase_name, num="N", name="<step-name>"
        )
        retry_section = ""
        if prev_error:
            retry_section = (
                f"\n## ⚠ 이전 시도 실패 — 아래 에러를 반드시 참고하여 수정하라\n\n"
                f"{prev_error}\n\n---\n\n"
            )
        if oom_directive:
            retry_section += (
                f"\n## ⚠ OOM 자동 재시도 — 다음 지시를 따르라\n\n"
                f"{oom_directive}\n\n---\n\n"
            )
        if resume_directive:
            retry_section += (
                f"\n## ↺ 체크포인트 재개 — 다음 지시를 따르라\n\n"
                f"{resume_directive}\n\n---\n\n"
            )
        if launcher_directive:
            retry_section += (
                f"\n## ⚙ 학습 launcher — 다음 지시를 따르라\n\n"
                f"{launcher_directive}\n\n---\n\n"
            )
        attempts = max_attempts if max_attempts is not None else self.MAX_RETRIES
        return (
            f"당신은 {self._project} 프로젝트의 개발자입니다. 아래 step을 수행하세요.\n\n"
            f"{guardrails}\n\n---\n\n"
            f"{step_context}{retry_section}"
            f"## 작업 규칙\n\n"
            f"1. 이전 step에서 작성된 코드를 확인하고 일관성을 유지하라.\n"
            f"2. 이 step에 명시된 작업만 수행하라. 추가 기능이나 파일을 만들지 마라.\n"
            f"3. 기존 테스트를 깨뜨리지 마라.\n"
            f"4. AC(Acceptance Criteria) 검증을 직접 실행하라.\n"
            f"5. /phases/{self._phase_dir_name}/index.json의 해당 step status를 업데이트하라:\n"
            f"   - AC 통과 → \"completed\" + \"summary\" 필드에 이 step의 산출물을 한 줄로 요약\n"
            f"   - {attempts}회 수정 시도 후에도 실패 → \"error\" + \"error_message\" 기록\n"
            f"   - 사용자 개입이 필요한 경우 (API 키, 인증, 수동 설정 등) → \"blocked\" + \"blocked_reason\" 기록 후 즉시 중단\n"
            f"6. 모든 변경사항을 커밋하라:\n"
            f"   {commit_example}\n\n---\n\n"
        )

    # --- Claude 호출 ---

    def _invoke_claude(self, step: dict, preamble: str) -> dict:
        step_num, step_name = step["step"], step["name"]
        step_file = self._phase_dir / f"step{step_num}.md"

        if not step_file.exists():
            print(f"  ERROR: {step_file} not found")
            sys.exit(1)

        prompt = preamble + step_file.read_text()
        timeout_sec = step.get("timeout_sec", self.DEFAULT_TIMEOUT_SEC)

        # monitor / heartbeat 시작 (옵션)
        monitors = []
        if step.get("monitors"):
            monitors = start_monitors(step["monitors"], ROOT / "runs")

        watchdog: Optional[HeartbeatWatchdog] = None
        proc = subprocess.Popen(
            ["claude", "-p", "--dangerously-skip-permissions", "--output-format", "json", prompt],
            cwd=self._root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
        )

        if step.get("heartbeat_timeout_sec"):
            hb_dir = self._phase_dir
            watchdog = HeartbeatWatchdog(
                hb_dir,
                timeout_sec=step["heartbeat_timeout_sec"],
                on_hang=lambda p=proc: p.terminate(),
            )
            watchdog.start()

        # stdout 라인 tee — step{N}.log 에 풀 기록, 마지막 200 줄만 메모리
        log_path = self._phase_dir / f"step{step_num}.log"
        stdout_keep: list = []
        stderr_buf: list = []

        def _tee_stdout():
            with open(log_path, "w", encoding="utf-8") as lf:
                for line in iter(proc.stdout.readline, ""):
                    if not line:
                        break
                    lf.write(line)
                    lf.flush()
                    stdout_keep.append(line)
                    if len(stdout_keep) > 200:
                        stdout_keep.pop(0)

        def _read_stderr():
            for line in iter(proc.stderr.readline, ""):
                if not line:
                    break
                stderr_buf.append(line)

        t_out = threading.Thread(target=_tee_stdout, daemon=True)
        t_err = threading.Thread(target=_read_stderr, daemon=True)
        t_out.start(); t_err.start()

        timed_out = False
        try:
            proc.wait(timeout=timeout_sec)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            returncode = -9
            timed_out = True
        finally:
            t_out.join(timeout=5)
            t_err.join(timeout=5)
            if watchdog is not None:
                watchdog.stop()
            if monitors:
                stop_monitors(monitors)

        stdout = "".join(stdout_keep)
        stderr = "".join(stderr_buf)
        if timed_out:
            stderr += f"\n[TIMEOUT {timeout_sec}s]"

        # heartbeat watchdog 가 발화했으면 timed_out 으로 표시 (crash_classifier 가 Hung 분류)
        if watchdog is not None and watchdog.fired:
            timed_out = True
            stderr += f"\n[HEARTBEAT TIMEOUT {step.get('heartbeat_timeout_sec')}s]"

        if returncode != 0:
            print(f"\n  WARN: Claude가 비정상 종료됨 (code {returncode}{', timed_out' if timed_out else ''})")
            if stderr:
                print(f"  stderr: {stderr[:500]}")
            print(f"  log: {log_path}")

        try:
            log_rel = str(log_path.relative_to(ROOT))
        except ValueError:
            log_rel = str(log_path)
        output = {
            "step": step_num, "name": step_name,
            "exitCode": returncode,
            "stdout_tail": stdout,    # 마지막 200 줄만 (풀 로그는 step{N}.log)
            "stderr": stderr,
            "log_path": log_rel,
            "timed_out": timed_out,
        }
        out_path = self._phase_dir / f"step{step_num}-output.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        # crash_classifier 호환을 위해 stdout 키도 유지 (deprecated — stdout_tail 사용 권장)
        output["stdout"] = stdout
        return output

    # --- 헤더 & 검증 ---

    def _print_header(self):
        print(f"\n{'='*60}")
        print(f"  Harness Step Executor")
        print(f"  Phase: {self._phase_name} | Steps: {self._total}")
        if self._auto_push:
            print(f"  Auto-push: enabled")
        print(f"{'='*60}")

    def _check_blockers(self):
        index = self._read_json(self._index_file)
        for s in reversed(index["steps"]):
            if s["status"] == "error":
                print(f"\n  ✗ Step {s['step']} ({s['name']}) failed.")
                print(f"  Error: {s.get('error_message', 'unknown')}")
                if s.get("crash_reason"):
                    print(f"  Crash: {s['crash_reason']} — {s.get('recommended_action', '')}")
                print(f"  Fix and reset status to 'pending' to retry.")
                sys.exit(1)
            if s["status"] == "blocked":
                print(f"\n  ⏸ Step {s['step']} ({s['name']}) blocked.")
                print(f"  Reason: {s.get('blocked_reason', 'unknown')}")
                print(f"  Resolve and reset status to 'pending' to retry.")
                sys.exit(2)
            if s["status"] == "awaiting-review":
                print(f"\n  ⏸ Step {s['step']} ({s['name']}) awaiting review (checkpoint={s.get('checkpoint', '?')}).")
                print(f"  status 를 'approved' (진행) 또는 'rejected' (중단) 로 변경 후 재실행하세요.")
                sys.exit(3)
            if s["status"] == "rejected":
                print(f"\n  ✗ Step {s['step']} ({s['name']}) rejected at checkpoint {s.get('checkpoint', '?')}.")
                sys.exit(1)
            if s["status"] != "pending":
                break

    def _ensure_created_at(self):
        index = self._read_json(self._index_file)
        if "created_at" not in index:
            index["created_at"] = self._stamp()
            self._write_json(self._index_file, index)

    # --- 실행 루프 ---

    def _execute_single_step(self, step: dict, guardrails_unused: str) -> bool:
        """단일 step 실행. step.kind 별 가드레일·crash 분류·OOM 재시도·runs 인덱싱·
        success_metric·checkpoint 통합. 완료되면 True.

        step.max_retries (기본 self.MAX_RETRIES-1) 는 *재시도 횟수*. 0 이면 재시도 없음.
        step.auto_retry_on_oom 가 true 면 OOM 한정 1회 추가 시도 슬롯이 부여된다.
        """
        step_num, step_name = step["step"], step["name"]
        done = sum(1 for s in self._read_json(self._index_file)["steps"] if s["status"] == "completed")

        if "max_retries" in step:
            max_attempts = max(1, step["max_retries"] + 1)
        else:
            max_attempts = self.MAX_RETRIES
        oom_slot_available = bool(step.get("auto_retry_on_oom"))

        prev_error: Optional[str] = None
        oom_directive_next: Optional[str] = None
        oom_slot_used = False
        attempt = 0

        while True:
            attempt += 1
            # step 별 가드레일 (kind 기반 부분집합)
            step_guardrails = self._load_guardrails(step)
            index = self._read_json(self._index_file)
            step_context = self._build_step_context(index)

            # launcher 지시 — step.launcher 가 있으면 항상 주입
            launcher_directive = None
            launcher = step.get("launcher")
            if launcher:
                kind = launcher.get("kind", "torchrun")
                nproc = launcher.get("nproc_per_node", 1)
                env_str = " ".join(f"{k}={v}" for k, v in (launcher.get("env") or {}).items())
                env_prefix = (f"env {env_str} " if env_str else "")
                launcher_directive = (
                    f"이 step 의 AC 안의 학습 명령(`python -m src.train ...` 등) 을 다음으로 wrap 해 실행하라:\n"
                    f"  {env_prefix}{kind} --nproc-per-node={nproc} <기존 학습 명령>\n"
                    f"환경변수는 학습 프로세스 트리 전체에 적용된다."
                )

            # resume 지시 — 재시도 중이고 step.resumable 이면 주입
            resume_directive = None
            if attempt > 1 and step.get("resumable"):
                resume_from = step.get("resume_from") or "runs/{run_dir}/checkpoints/last.pt"
                resume_directive = (
                    f"이전 시도가 도중 죽었다. 처음부터 학습을 다시 하지 말고, "
                    f"`{resume_from}` 체크포인트부터 재개하라. 학습 스크립트의 "
                    f"`--resume <ckpt>` 또는 `--resume-from <ckpt>` 인자를 사용하라."
                )

            preamble = self._build_preamble(
                step_guardrails, step_context, prev_error, max_attempts,
                oom_directive_next, resume_directive, launcher_directive,
            )

            tag = f"Step {step_num}/{self._total - 1} ({done} done): {step_name}"
            if attempt > 1:
                tag += f" [retry {attempt}/{max_attempts}{' +oom' if oom_slot_used else ''}]"

            runs_before = self._snapshot_runs()
            with progress_indicator(tag) as pi:
                invoke_result = self._invoke_claude(step, preamble)
                elapsed = int(pi.elapsed)

            index = self._read_json(self._index_file)
            status = next((s.get("status", "pending") for s in index["steps"] if s["step"] == step_num), "pending")
            ts = self._stamp()

            # --- completed 경로 ---
            if status == "completed":
                self._index_runs(index, step_num, runs_before)
                passed, msg = self._verify_success_metric(index, step_num)
                if not passed:
                    # 메트릭 미달 → 강제 error 로 전환 (재시도 가능)
                    for s in index["steps"]:
                        if s["step"] == step_num:
                            s["status"] = "error"
                            s["error_message"] = msg
                    self._write_json(self._index_file, index)
                    print(f"  ✗ Step {step_num}: success_metric 미달 — {msg}")
                    status = "error"  # 아래 실패 경로로 폴스루
                else:
                    for s in index["steps"]:
                        if s["step"] == step_num:
                            s["completed_at"] = ts
                    self._write_json(self._index_file, index)
                    self._commit_step(step_num, step_name)
                    print(f"  ✓ Step {step_num}: {step_name} [{elapsed}s]")
                    # checkpoint 처리
                    cp = step.get("checkpoint")
                    if cp:
                        idx_after = self._read_json(self._index_file)
                        result = self._handle_checkpoint(idx_after, step_num, cp, step_name)
                        if result == "rejected":
                            self._update_top_index("error")
                            sys.exit(1)
                        if result == "awaiting-review":
                            sys.exit(3)
                    return True

            # --- blocked 경로 ---
            if status == "blocked":
                for s in index["steps"]:
                    if s["step"] == step_num:
                        s["blocked_at"] = ts
                self._write_json(self._index_file, index)
                reason = next((s.get("blocked_reason", "") for s in index["steps"] if s["step"] == step_num), "")
                print(f"  ⏸ Step {step_num}: {step_name} blocked [{elapsed}s]")
                print(f"    Reason: {reason}")
                self._update_top_index("blocked")
                sys.exit(2)

            # --- error / 미설정 경로 — crash 분류 ---
            err_msg = next(
                (s.get("error_message", "Step did not update status") for s in index["steps"] if s["step"] == step_num),
                "Step did not update status",
            )
            timed_out = isinstance(invoke_result, dict) and invoke_result.get("timed_out", False)
            crash = classify_crash(
                stderr=(invoke_result.get("stderr", "") if isinstance(invoke_result, dict) else ""),
                returncode=(invoke_result.get("exitCode", 1) if isinstance(invoke_result, dict) else 1),
                stdout_tail=(invoke_result.get("stdout", "") if isinstance(invoke_result, dict) else None),
                timed_out=timed_out,
            )
            for s in index["steps"]:
                if s["step"] == step_num:
                    s["crash_reason"] = crash.reason
                    s["crash_evidence"] = crash.evidence
                    s["recommended_action"] = crash.recommended_action
            self._write_json(self._index_file, index)
            print(f"  ✗ Step {step_num}: crash={crash.reason} — {crash.recommended_action}")

            use_oom_slot = (
                crash.reason == "OOM" and oom_slot_available and not oom_slot_used
            )
            can_retry = (attempt < max_attempts) or use_oom_slot

            if not can_retry:
                for s in index["steps"]:
                    if s["step"] == step_num:
                        s["status"] = "error"
                        s["error_message"] = f"[{attempt}회 시도 후 실패 / crash={crash.reason}] {err_msg}"
                        s["failed_at"] = ts
                self._write_json(self._index_file, index)
                self._commit_step(step_num, step_name)
                print(f"  ✗ Step {step_num}: {step_name} failed after {attempt} attempts [{elapsed}s]")
                self._update_top_index("error")
                sys.exit(1)

            # 재시도 준비
            for s in index["steps"]:
                if s["step"] == step_num:
                    s["status"] = "pending"
                    s.pop("error_message", None)
            self._write_json(self._index_file, index)
            prev_error = f"crash={crash.reason}\n{crash.evidence}"

            if use_oom_slot:
                oom_slot_used = True
                oom_directive_next = (
                    "이전 시도가 CUDA OOM 으로 죽었다. step 의 학습 명령에서 batch_size 를 "
                    "절반으로 줄여 재시도하라. 가능하면 mixed precision (fp16/bf16) 또는 "
                    "gradient accumulation 도 함께 적용해 effective batch size 는 유지하라."
                )
                for s in index["steps"]:
                    if s["step"] == step_num:
                        s["attempted_recovery"] = {"oom_retry": True}
                self._write_json(self._index_file, index)
                print(f"  ↻ Step {step_num}: OOM 자동 재시도 (batch_size 1/2 지시)")
            else:
                oom_directive_next = None
                print(f"  ↻ Step {step_num}: retry {attempt + 1}/{max_attempts} — crash={crash.reason}")

    def _execute_all_steps(self, guardrails: str):
        while True:
            index = self._read_json(self._index_file)
            pending = next((s for s in index["steps"] if s["status"] == "pending"), None)
            if pending is None:
                print("\n  All steps completed!")
                return

            step_num = pending["step"]
            for s in index["steps"]:
                if s["step"] == step_num and "started_at" not in s:
                    s["started_at"] = self._stamp()
                    self._write_json(self._index_file, index)
                    break

            self._execute_single_step(pending, guardrails)

    def _finalize(self):
        index = self._read_json(self._index_file)
        # compare/ablate phase 면 모든 run_dir 의 fairness 검증
        kinds = {s.get("kind") for s in index.get("steps", [])}
        if kinds & {"compare", "ablate"}:
            all_run_dirs = []
            for s in index["steps"]:
                all_run_dirs.extend(s.get("run_dirs", []) or [])
            report = verify_fairness(all_run_dirs, ROOT / "runs")
            print("\n" + format_fairness(report))
            if report.violated:
                index["fairness_violations"] = [
                    {"dimension": d, "values": v} for d, v in report.violations
                ]
                self._write_json(self._index_file, index)
                self._update_top_index("error")
                print(f"  ✗ Fairness 위반으로 phase 실패")
                sys.exit(1)

        index["completed_at"] = self._stamp()
        self._write_json(self._index_file, index)
        self._update_top_index("completed")

        self._run_git("add", "-A")
        if self._run_git("diff", "--cached", "--quiet").returncode != 0:
            msg = f"chore({self._phase_name}): mark phase completed"
            r = self._run_git("commit", "-m", msg)
            if r.returncode == 0:
                print(f"  ✓ {msg}")

        if self._auto_push:
            branch = f"feat-{self._phase_name}"
            r = self._run_git("push", "-u", "origin", branch)
            if r.returncode != 0:
                print(f"\n  ERROR: git push 실패: {r.stderr.strip()}")
                sys.exit(1)
            print(f"  ✓ Pushed to origin/{branch}")

        print(f"\n{'='*60}")
        print(f"  Phase '{self._phase_name}' completed!")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Harness Step Executor")
    parser.add_argument("phase_dir", help="Phase directory name (e.g. 0-mvp)")
    parser.add_argument("--push", action="store_true", help="Push branch after completion")
    args = parser.parse_args()

    StepExecutor(args.phase_dir, auto_push=args.push).run()


if __name__ == "__main__":
    main()