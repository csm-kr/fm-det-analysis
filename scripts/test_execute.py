"""
execute.py 리팩터링 안전망 테스트.
리팩터링 전후 동작이 동일한지 검증한다.
"""

import json
import os
import subprocess
import sys
import textwrap
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import execute as ex


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_project(tmp_path):
    """phases/, CLAUDE.md, docs/ 를 갖춘 임시 프로젝트 구조."""
    phases_dir = tmp_path / "phases"
    phases_dir.mkdir()

    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text("# Rules\n- rule one\n- rule two")

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "arch.md").write_text("# Architecture\nSome content")
    (docs_dir / "guide.md").write_text("# Guide\nAnother doc")

    return tmp_path


@pytest.fixture
def phase_dir(tmp_project):
    """step 3개를 가진 phase 디렉토리."""
    d = tmp_project / "phases" / "0-mvp"
    d.mkdir()

    index = {
        "project": "TestProject",
        "phase": "mvp",
        "steps": [
            {"step": 0, "name": "setup", "status": "completed", "summary": "프로젝트 초기화 완료"},
            {"step": 1, "name": "core", "status": "completed", "summary": "핵심 로직 구현"},
            {"step": 2, "name": "ui", "status": "pending"},
        ],
    }
    (d / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False))
    (d / "step2.md").write_text("# Step 2: UI\n\nUI를 구현하세요.")

    return d


@pytest.fixture
def top_index(tmp_project):
    """phases/index.json (top-level)."""
    top = {
        "phases": [
            {"dir": "0-mvp", "status": "pending"},
            {"dir": "1-polish", "status": "pending"},
        ]
    }
    p = tmp_project / "phases" / "index.json"
    p.write_text(json.dumps(top, indent=2))
    return p


@pytest.fixture
def executor(tmp_project, phase_dir):
    """테스트용 StepExecutor 인스턴스. git 호출은 별도 mock 필요."""
    with patch.object(ex, "ROOT", tmp_project):
        inst = ex.StepExecutor("0-mvp")
    # 내부 경로를 tmp_project 기준으로 재설정
    inst._root = str(tmp_project)
    inst._phases_dir = tmp_project / "phases"
    inst._phase_dir = phase_dir
    inst._phase_dir_name = "0-mvp"
    inst._index_file = phase_dir / "index.json"
    inst._top_index_file = tmp_project / "phases" / "index.json"
    return inst


# ---------------------------------------------------------------------------
# _stamp (= 이전 now_iso)
# ---------------------------------------------------------------------------

class TestStamp:
    def test_returns_kst_timestamp(self, executor):
        result = executor._stamp()
        assert "+0900" in result

    def test_format_is_iso(self, executor):
        result = executor._stamp()
        dt = datetime.strptime(result, "%Y-%m-%dT%H:%M:%S%z")
        assert dt.tzinfo is not None

    def test_is_current_time(self, executor):
        before = datetime.now(ex.StepExecutor.TZ).replace(microsecond=0)
        result = executor._stamp()
        after = datetime.now(ex.StepExecutor.TZ).replace(microsecond=0) + timedelta(seconds=1)
        parsed = datetime.strptime(result, "%Y-%m-%dT%H:%M:%S%z")
        assert before <= parsed <= after


# ---------------------------------------------------------------------------
# _read_json / _write_json
# ---------------------------------------------------------------------------

class TestJsonHelpers:
    def test_roundtrip(self, tmp_path):
        data = {"key": "값", "nested": [1, 2, 3]}
        p = tmp_path / "test.json"
        ex.StepExecutor._write_json(p, data)
        loaded = ex.StepExecutor._read_json(p)
        assert loaded == data

    def test_save_ensures_ascii_false(self, tmp_path):
        p = tmp_path / "test.json"
        ex.StepExecutor._write_json(p, {"한글": "테스트"})
        raw = p.read_text()
        assert "한글" in raw
        assert "\\u" not in raw

    def test_save_indented(self, tmp_path):
        p = tmp_path / "test.json"
        ex.StepExecutor._write_json(p, {"a": 1})
        raw = p.read_text()
        assert "\n" in raw

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ex.StepExecutor._read_json(tmp_path / "nope.json")


# ---------------------------------------------------------------------------
# _load_guardrails
# ---------------------------------------------------------------------------

class TestLoadGuardrails:
    def test_loads_claude_md_and_docs(self, executor, tmp_project):
        with patch.object(ex, "ROOT", tmp_project):
            result = executor._load_guardrails()
        assert "# Rules" in result
        assert "rule one" in result
        assert "# Architecture" in result
        assert "# Guide" in result

    def test_sections_separated_by_divider(self, executor, tmp_project):
        with patch.object(ex, "ROOT", tmp_project):
            result = executor._load_guardrails()
        assert "---" in result

    def test_docs_sorted_alphabetically(self, executor, tmp_project):
        with patch.object(ex, "ROOT", tmp_project):
            result = executor._load_guardrails()
        arch_pos = result.index("arch")
        guide_pos = result.index("guide")
        assert arch_pos < guide_pos

    def test_no_claude_md(self, executor, tmp_project):
        (tmp_project / "CLAUDE.md").unlink()
        with patch.object(ex, "ROOT", tmp_project):
            result = executor._load_guardrails()
        assert "CLAUDE.md" not in result
        assert "Architecture" in result

    def test_no_docs_dir(self, executor, tmp_project):
        import shutil
        shutil.rmtree(tmp_project / "docs")
        with patch.object(ex, "ROOT", tmp_project):
            result = executor._load_guardrails()
        assert "Rules" in result
        assert "Architecture" not in result

    def test_empty_project(self, tmp_path):
        with patch.object(ex, "ROOT", tmp_path):
            # executor가 필요 없는 static-like 동작이므로 임시 인스턴스
            phases_dir = tmp_path / "phases" / "dummy"
            phases_dir.mkdir(parents=True)
            idx = {"project": "T", "phase": "t", "steps": []}
            (phases_dir / "index.json").write_text(json.dumps(idx))
            inst = ex.StepExecutor.__new__(ex.StepExecutor)
            result = inst._load_guardrails()
        assert result == ""


# ---------------------------------------------------------------------------
# _build_step_context
# ---------------------------------------------------------------------------

class TestBuildStepContext:
    def test_includes_completed_with_summary(self, phase_dir):
        index = json.loads((phase_dir / "index.json").read_text())
        result = ex.StepExecutor._build_step_context(index)
        assert "Step 0 (setup): 프로젝트 초기화 완료" in result
        assert "Step 1 (core): 핵심 로직 구현" in result

    def test_excludes_pending(self, phase_dir):
        index = json.loads((phase_dir / "index.json").read_text())
        result = ex.StepExecutor._build_step_context(index)
        assert "ui" not in result

    def test_excludes_completed_without_summary(self, phase_dir):
        index = json.loads((phase_dir / "index.json").read_text())
        del index["steps"][0]["summary"]
        result = ex.StepExecutor._build_step_context(index)
        assert "setup" not in result
        assert "core" in result

    def test_empty_when_no_completed(self):
        index = {"steps": [{"step": 0, "name": "a", "status": "pending"}]}
        result = ex.StepExecutor._build_step_context(index)
        assert result == ""

    def test_has_header(self, phase_dir):
        index = json.loads((phase_dir / "index.json").read_text())
        result = ex.StepExecutor._build_step_context(index)
        assert result.startswith("## 이전 Step 산출물")


# ---------------------------------------------------------------------------
# _build_preamble
# ---------------------------------------------------------------------------

class TestBuildPreamble:
    def test_includes_project_name(self, executor):
        result = executor._build_preamble("", "")
        assert "TestProject" in result

    def test_includes_guardrails(self, executor):
        result = executor._build_preamble("GUARD_CONTENT", "")
        assert "GUARD_CONTENT" in result

    def test_includes_step_context(self, executor):
        ctx = "## 이전 Step 산출물\n\n- Step 0: done"
        result = executor._build_preamble("", ctx)
        assert "이전 Step 산출물" in result

    def test_includes_commit_example(self, executor):
        result = executor._build_preamble("", "")
        assert "feat(mvp):" in result

    def test_includes_rules(self, executor):
        result = executor._build_preamble("", "")
        assert "작업 규칙" in result
        assert "AC" in result

    def test_no_retry_section_by_default(self, executor):
        result = executor._build_preamble("", "")
        assert "이전 시도 실패" not in result

    def test_retry_section_with_prev_error(self, executor):
        result = executor._build_preamble("", "", prev_error="타입 에러 발생")
        assert "이전 시도 실패" in result
        assert "타입 에러 발생" in result

    def test_includes_max_retries(self, executor):
        result = executor._build_preamble("", "")
        assert str(ex.StepExecutor.MAX_RETRIES) in result

    def test_max_attempts_override_appears(self, executor):
        result = executor._build_preamble("", "", max_attempts=7)
        assert "7회 수정 시도" in result

    def test_max_attempts_none_falls_back_to_default(self, executor):
        result = executor._build_preamble("", "", max_attempts=None)
        assert f"{ex.StepExecutor.MAX_RETRIES}회 수정 시도" in result

    def test_includes_index_path(self, executor):
        result = executor._build_preamble("", "")
        assert "/phases/0-mvp/index.json" in result


# ---------------------------------------------------------------------------
# _update_top_index
# ---------------------------------------------------------------------------

class TestUpdateTopIndex:
    def test_completed(self, executor, top_index):
        executor._top_index_file = top_index
        executor._update_top_index("completed")
        data = json.loads(top_index.read_text())
        mvp = next(p for p in data["phases"] if p["dir"] == "0-mvp")
        assert mvp["status"] == "completed"
        assert "completed_at" in mvp

    def test_error(self, executor, top_index):
        executor._top_index_file = top_index
        executor._update_top_index("error")
        data = json.loads(top_index.read_text())
        mvp = next(p for p in data["phases"] if p["dir"] == "0-mvp")
        assert mvp["status"] == "error"
        assert "failed_at" in mvp

    def test_blocked(self, executor, top_index):
        executor._top_index_file = top_index
        executor._update_top_index("blocked")
        data = json.loads(top_index.read_text())
        mvp = next(p for p in data["phases"] if p["dir"] == "0-mvp")
        assert mvp["status"] == "blocked"
        assert "blocked_at" in mvp

    def test_other_phases_unchanged(self, executor, top_index):
        executor._top_index_file = top_index
        executor._update_top_index("completed")
        data = json.loads(top_index.read_text())
        polish = next(p for p in data["phases"] if p["dir"] == "1-polish")
        assert polish["status"] == "pending"

    def test_nonexistent_dir_is_noop(self, executor, top_index):
        executor._top_index_file = top_index
        executor._phase_dir_name = "no-such-dir"
        original = json.loads(top_index.read_text())
        executor._update_top_index("completed")
        after = json.loads(top_index.read_text())
        for p_before, p_after in zip(original["phases"], after["phases"]):
            assert p_before["status"] == p_after["status"]

    def test_no_top_index_file(self, executor, tmp_path):
        executor._top_index_file = tmp_path / "nonexistent.json"
        executor._update_top_index("completed")  # should not raise


# ---------------------------------------------------------------------------
# _checkout_branch (mocked)
# ---------------------------------------------------------------------------

class TestCheckoutBranch:
    def _mock_git(self, executor, responses):
        call_idx = {"i": 0}
        def fake_git(*args):
            idx = call_idx["i"]
            call_idx["i"] += 1
            if idx < len(responses):
                return responses[idx]
            return MagicMock(returncode=0, stdout="", stderr="")
        executor._run_git = fake_git

    def test_already_on_branch(self, executor):
        self._mock_git(executor, [
            MagicMock(returncode=0, stdout="feat-mvp\n", stderr=""),
        ])
        executor._checkout_branch()  # should return without checkout

    def test_branch_exists_checkout(self, executor):
        self._mock_git(executor, [
            MagicMock(returncode=0, stdout="main\n", stderr=""),
            MagicMock(returncode=0, stdout="", stderr=""),
            MagicMock(returncode=0, stdout="", stderr=""),
        ])
        executor._checkout_branch()

    def test_branch_not_exists_create(self, executor):
        self._mock_git(executor, [
            MagicMock(returncode=0, stdout="main\n", stderr=""),
            MagicMock(returncode=1, stdout="", stderr="not found"),
            MagicMock(returncode=0, stdout="", stderr=""),
        ])
        executor._checkout_branch()

    def test_checkout_fails_exits(self, executor):
        self._mock_git(executor, [
            MagicMock(returncode=0, stdout="main\n", stderr=""),
            MagicMock(returncode=1, stdout="", stderr=""),
            MagicMock(returncode=1, stdout="", stderr="dirty tree"),
        ])
        with pytest.raises(SystemExit) as exc_info:
            executor._checkout_branch()
        assert exc_info.value.code == 1

    def test_no_git_exits(self, executor):
        self._mock_git(executor, [
            MagicMock(returncode=1, stdout="", stderr="not a git repo"),
        ])
        with pytest.raises(SystemExit) as exc_info:
            executor._checkout_branch()
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# _commit_step (mocked)
# ---------------------------------------------------------------------------

class TestCommitStep:
    def test_two_phase_commit(self, executor):
        calls = []
        def fake_git(*args):
            calls.append(args)
            if args[:2] == ("diff", "--cached"):
                return MagicMock(returncode=1)
            return MagicMock(returncode=0, stdout="", stderr="")
        executor._run_git = fake_git

        executor._commit_step(2, "ui")

        commit_calls = [c for c in calls if c[0] == "commit"]
        assert len(commit_calls) == 2
        assert "feat(mvp):" in commit_calls[0][2]
        assert "chore(mvp):" in commit_calls[1][2]

    def test_no_code_changes_skips_feat_commit(self, executor):
        call_count = {"diff": 0}
        calls = []
        def fake_git(*args):
            calls.append(args)
            if args[:2] == ("diff", "--cached"):
                call_count["diff"] += 1
                if call_count["diff"] == 1:
                    return MagicMock(returncode=0)
                return MagicMock(returncode=1)
            return MagicMock(returncode=0, stdout="", stderr="")
        executor._run_git = fake_git

        executor._commit_step(2, "ui")

        commit_msgs = [c[2] for c in calls if c[0] == "commit"]
        assert len(commit_msgs) == 1
        assert "chore" in commit_msgs[0]


# ---------------------------------------------------------------------------
# _invoke_claude (mocked)
# ---------------------------------------------------------------------------

class TestInvokeClaude:
    @staticmethod
    def _mock_popen(stdout_lines=('{"result": "ok"}\n',), stderr_lines=(), returncode=0):
        """Popen 객체 mock — proc.stdout.readline iter / proc.stderr.readline iter / wait + returncode."""
        proc = MagicMock()
        # iter(readline, "") 패턴에 맞춰 빈 문자열로 EOF
        stdout_iter = iter(list(stdout_lines) + [""])
        stderr_iter = iter(list(stderr_lines) + [""])
        proc.stdout.readline.side_effect = lambda: next(stdout_iter, "")
        proc.stderr.readline.side_effect = lambda: next(stderr_iter, "")
        proc.wait.return_value = returncode
        proc.returncode = returncode
        return proc

    def test_invokes_claude_with_correct_args(self, executor):
        step = {"step": 2, "name": "ui"}
        preamble = "PREAMBLE\n"

        with patch("subprocess.Popen", return_value=self._mock_popen()) as mock_popen:
            executor._invoke_claude(step, preamble)

        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == "claude"
        assert "-p" in cmd
        assert "--dangerously-skip-permissions" in cmd
        assert "--output-format" in cmd
        assert "PREAMBLE" in cmd[-1]
        assert "UI를 구현하세요" in cmd[-1]

    def test_saves_output_json(self, executor):
        step = {"step": 2, "name": "ui"}

        with patch("subprocess.Popen", return_value=self._mock_popen()):
            executor._invoke_claude(step, "preamble")

        output_file = executor._phase_dir / "step2-output.json"
        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert data["step"] == 2
        assert data["name"] == "ui"
        assert data["exitCode"] == 0
        assert "log_path" in data

    def test_nonexistent_step_file_exits(self, executor):
        step = {"step": 99, "name": "nonexistent"}
        with pytest.raises(SystemExit) as exc_info:
            executor._invoke_claude(step, "preamble")
        assert exc_info.value.code == 1

    def test_timeout_is_1800(self, executor):
        step = {"step": 2, "name": "ui"}
        proc = self._mock_popen()
        with patch("subprocess.Popen", return_value=proc):
            executor._invoke_claude(step, "preamble")
        proc.wait.assert_called_with(timeout=1800)

    def test_timeout_from_step_override(self, executor):
        step = {"step": 2, "name": "ui", "timeout_sec": 7200}
        proc = self._mock_popen()
        with patch("subprocess.Popen", return_value=proc):
            executor._invoke_claude(step, "preamble")
        proc.wait.assert_called_with(timeout=7200)

    def test_timeout_default_when_field_missing(self, executor):
        step = {"step": 2, "name": "ui"}
        proc = self._mock_popen()
        with patch("subprocess.Popen", return_value=proc):
            executor._invoke_claude(step, "preamble")
        proc.wait.assert_called_with(timeout=ex.StepExecutor.DEFAULT_TIMEOUT_SEC)

    def test_stdout_tee_writes_log_file(self, executor):
        step = {"step": 2, "name": "ui"}
        proc = self._mock_popen(stdout_lines=("line1\n", "line2\n"))
        with patch("subprocess.Popen", return_value=proc):
            executor._invoke_claude(step, "preamble")
        log_path = executor._phase_dir / "step2.log"
        assert log_path.exists()
        content = log_path.read_text()
        assert "line1" in content
        assert "line2" in content

    def test_stdout_keeps_last_200_lines(self, executor):
        step = {"step": 2, "name": "ui"}
        # 250 줄 → 메모리에는 마지막 200 만, 로그에는 250 전부
        lines = tuple(f"line {i}\n" for i in range(250))
        proc = self._mock_popen(stdout_lines=lines)
        with patch("subprocess.Popen", return_value=proc):
            executor._invoke_claude(step, "preamble")
        data = json.loads((executor._phase_dir / "step2-output.json").read_text())
        # tail 에는 line 50~249 만
        assert "line 0\n" not in data["stdout_tail"]
        assert "line 49\n" not in data["stdout_tail"]
        assert "line 50\n" in data["stdout_tail"]
        assert "line 249\n" in data["stdout_tail"]
        # log 파일에는 전부
        log = (executor._phase_dir / "step2.log").read_text()
        assert "line 0\n" in log
        assert "line 249\n" in log


# ---------------------------------------------------------------------------
# progress_indicator (= 이전 Spinner)
# ---------------------------------------------------------------------------

class TestProgressIndicator:
    def test_context_manager(self):
        import time
        with ex.progress_indicator("test") as pi:
            time.sleep(0.15)
        assert pi.elapsed >= 0.1

    def test_elapsed_increases(self):
        import time
        with ex.progress_indicator("test") as pi:
            time.sleep(0.2)
        assert pi.elapsed > 0


# ---------------------------------------------------------------------------
# main() CLI 파싱 (mocked)
# ---------------------------------------------------------------------------

class TestMainCli:
    def test_no_args_exits(self):
        with patch("sys.argv", ["execute.py"]):
            with pytest.raises(SystemExit) as exc_info:
                ex.main()
            assert exc_info.value.code == 2  # argparse exits with 2

    def test_invalid_phase_dir_exits(self):
        with patch("sys.argv", ["execute.py", "nonexistent"]):
            with patch.object(ex, "ROOT", Path("/tmp/fake_nonexistent")):
                with pytest.raises(SystemExit) as exc_info:
                    ex.main()
                assert exc_info.value.code == 1

    def test_missing_index_exits(self, tmp_project):
        (tmp_project / "phases" / "empty").mkdir()
        with patch("sys.argv", ["execute.py", "empty"]):
            with patch.object(ex, "ROOT", tmp_project):
                with pytest.raises(SystemExit) as exc_info:
                    ex.main()
                assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# _check_blockers (= 이전 main() error/blocked 체크)
# ---------------------------------------------------------------------------

class TestCheckBlockers:
    def _make_executor_with_steps(self, tmp_project, steps):
        d = tmp_project / "phases" / "test-phase"
        d.mkdir(exist_ok=True)
        index = {"project": "T", "phase": "test", "steps": steps}
        (d / "index.json").write_text(json.dumps(index))

        with patch.object(ex, "ROOT", tmp_project):
            inst = ex.StepExecutor.__new__(ex.StepExecutor)
        inst._root = str(tmp_project)
        inst._phases_dir = tmp_project / "phases"
        inst._phase_dir = d
        inst._phase_dir_name = "test-phase"
        inst._index_file = d / "index.json"
        inst._top_index_file = tmp_project / "phases" / "index.json"
        inst._phase_name = "test"
        inst._total = len(steps)
        return inst

    def test_error_step_exits_1(self, tmp_project):
        steps = [
            {"step": 0, "name": "ok", "status": "completed"},
            {"step": 1, "name": "bad", "status": "error", "error_message": "fail"},
        ]
        inst = self._make_executor_with_steps(tmp_project, steps)
        with pytest.raises(SystemExit) as exc_info:
            inst._check_blockers()
        assert exc_info.value.code == 1

    def test_blocked_step_exits_2(self, tmp_project):
        steps = [
            {"step": 0, "name": "ok", "status": "completed"},
            {"step": 1, "name": "stuck", "status": "blocked", "blocked_reason": "API key"},
        ]
        inst = self._make_executor_with_steps(tmp_project, steps)
        with pytest.raises(SystemExit) as exc_info:
            inst._check_blockers()
        assert exc_info.value.code == 2

    def test_awaiting_review_exits_3(self, tmp_project):
        steps = [
            {"step": 0, "name": "data", "status": "awaiting-review", "checkpoint": "CP-1"},
        ]
        inst = self._make_executor_with_steps(tmp_project, steps)
        with pytest.raises(SystemExit) as exc_info:
            inst._check_blockers()
        assert exc_info.value.code == 3

    def test_rejected_exits_1(self, tmp_project):
        steps = [
            {"step": 0, "name": "data", "status": "rejected", "checkpoint": "CP-1"},
        ]
        inst = self._make_executor_with_steps(tmp_project, steps)
        with pytest.raises(SystemExit) as exc_info:
            inst._check_blockers()
        assert exc_info.value.code == 1

    def test_approved_allows_progress(self, tmp_project):
        steps = [
            {"step": 0, "name": "data", "status": "approved", "checkpoint": "CP-1"},
            {"step": 1, "name": "model", "status": "pending"},
        ]
        inst = self._make_executor_with_steps(tmp_project, steps)
        # 차단 status 가 없으므로 정상 통과 (sys.exit 호출 없음)
        inst._check_blockers()


# ---------------------------------------------------------------------------
# crash_classifier (단위 — 10 카테고리 패턴)
# ---------------------------------------------------------------------------

class TestCrashClassifier:
    def setup_method(self):
        import crash_classifier
        self.classify = crash_classifier.classify

    def test_oom(self):
        r = self.classify("RuntimeError: CUDA out of memory. Tried to allocate ...", 1)
        assert r.reason == "OOM"
        assert "batch_size" in r.recommended_action

    def test_nan(self):
        r = self.classify("AssertionError: loss is not finite", 1)
        assert r.reason == "NaN/Inf"

    def test_dataloader(self):
        r = self.classify("FileNotFoundError: [Errno 2] data/train/img.jpg", 1)
        assert r.reason == "DataLoader"

    def test_gpu_driver(self):
        r = self.classify("CUDA error: unspecified launch failure", 1)
        assert r.reason == "GPU/driver"

    def test_disk_full(self):
        r = self.classify("OSError: [Errno 28] No space left on device", 1)
        assert r.reason == "Disk full"

    def test_shm(self):
        r = self.classify("DataLoader worker (pid 123) is killed by signal: bus error", 1)
        assert r.reason == "SHM"

    def test_nccl(self):
        r = self.classify("ncclSystemError: System call (e.g. socket) failed", 1)
        assert r.reason == "NCCL"

    def test_preemption_sigterm(self):
        r = self.classify("Received SIGTERM, shutting down", 143)
        assert r.reason == "Preemption"

    def test_preemption_by_returncode(self):
        r = self.classify("", 137)  # SIGKILL signal exit code
        assert r.reason == "Preemption"

    def test_hung_when_timed_out(self):
        r = self.classify("normal output, no signal", 0, timed_out=True)
        assert r.reason == "Hung"

    def test_unknown(self):
        r = self.classify("some random error nobody recognizes", 1)
        assert r.reason == "Unknown"

    def test_evidence_is_last_20_lines(self):
        lines = "\n".join(f"line {i}" for i in range(30))
        r = self.classify(lines, 1)
        # 마지막 20줄만
        assert r.evidence.count("\n") == 19
        assert "line 29" in r.evidence
        assert "line 9" not in r.evidence


# ---------------------------------------------------------------------------
# _load_guardrails — kind 기반 부분집합
# ---------------------------------------------------------------------------

class TestGuardrailsKindFilter:
    @pytest.fixture
    def ml_project(self, tmp_path):
        """ml 도메인 docs 가 깔린 임시 프로젝트."""
        (tmp_path / "CLAUDE.md").write_text("# Rules")
        docs = tmp_path / "docs"
        docs.mkdir()
        for name in ["PRD", "ARCHITECTURE", "ADR", "DATA_CARD", "MODEL_CARD",
                     "EVAL_PROTOCOL", "EXPERIMENTS"]:
            (docs / f"{name}.md").write_text(f"# {name}\n{name} body")
        return tmp_path

    def _executor(self, project):
        phases = project / "phases" / "x"
        phases.mkdir(parents=True)
        (phases / "index.json").write_text(json.dumps({"project": "X", "phase": "x", "steps": []}))
        with patch.object(ex, "ROOT", project):
            inst = ex.StepExecutor("x")
        return inst

    def test_no_step_loads_all_docs(self, ml_project):
        with patch.object(ex, "ROOT", ml_project):
            inst = self._executor(ml_project)
            result = inst._load_guardrails()
        for name in ["PRD", "ARCHITECTURE", "DATA_CARD", "MODEL_CARD", "EVAL_PROTOCOL"]:
            assert name in result

    def test_data_sanity_filters_to_prd_and_data_card(self, ml_project):
        with patch.object(ex, "ROOT", ml_project):
            inst = self._executor(ml_project)
            result = inst._load_guardrails({"kind": "data-sanity"})
        assert "PRD" in result and "DATA_CARD" in result
        assert "MODEL_CARD" not in result
        assert "EVAL_PROTOCOL" not in result

    def test_model_sanity_includes_model_card(self, ml_project):
        with patch.object(ex, "ROOT", ml_project):
            inst = self._executor(ml_project)
            result = inst._load_guardrails({"kind": "model-sanity"})
        assert "PRD" in result and "ARCHITECTURE" in result and "MODEL_CARD" in result
        assert "DATA_CARD" not in result

    def test_inference_bench_only_model_card_and_eval(self, ml_project):
        with patch.object(ex, "ROOT", ml_project):
            inst = self._executor(ml_project)
            result = inst._load_guardrails({"kind": "inference-bench"})
        assert "MODEL_CARD" in result and "EVAL_PROTOCOL" in result
        assert "PRD" not in result

    def test_experiment_kind_loads_all(self, ml_project):
        with patch.object(ex, "ROOT", ml_project):
            inst = self._executor(ml_project)
            result = inst._load_guardrails({"kind": "experiment"})
        for name in ["PRD", "ARCHITECTURE", "EXPERIMENTS", "EVAL_PROTOCOL"]:
            assert name in result

    def test_unknown_kind_falls_back_to_all(self, ml_project):
        with patch.object(ex, "ROOT", ml_project):
            inst = self._executor(ml_project)
            result = inst._load_guardrails({"kind": "code"})
        for name in ["PRD", "ARCHITECTURE", "DATA_CARD"]:
            assert name in result

    def test_claude_md_always_included(self, ml_project):
        with patch.object(ex, "ROOT", ml_project):
            inst = self._executor(ml_project)
            result = inst._load_guardrails({"kind": "data-sanity"})
        assert "Rules" in result


# ---------------------------------------------------------------------------
# _snapshot_runs / _index_runs / _extract_run_metric
# ---------------------------------------------------------------------------

class TestRunsIndexing:
    @pytest.fixture
    def project_with_runs(self, tmp_path):
        (tmp_path / "phases" / "x").mkdir(parents=True)
        (tmp_path / "phases" / "x" / "index.json").write_text(
            json.dumps({"project": "X", "phase": "x", "steps": [
                {"step": 0, "name": "a", "status": "pending"}
            ]})
        )
        runs = tmp_path / "runs"
        runs.mkdir()
        return tmp_path, runs

    def _executor(self, project):
        with patch.object(ex, "ROOT", project):
            inst = ex.StepExecutor("x")
        return inst

    def test_snapshot_empty(self, project_with_runs):
        project, _ = project_with_runs
        with patch.object(ex, "ROOT", project):
            inst = self._executor(project)
            assert inst._snapshot_runs() == set()

    def test_snapshot_lists_subdirs(self, project_with_runs):
        project, runs = project_with_runs
        (runs / "20260301-0900-baseline").mkdir()
        (runs / "20260301-1000-aug").mkdir()
        with patch.object(ex, "ROOT", project):
            inst = self._executor(project)
            snap = inst._snapshot_runs()
        assert snap == {"20260301-0900-baseline", "20260301-1000-aug"}

    def test_index_runs_records_new_dirs(self, project_with_runs):
        project, runs = project_with_runs
        before = set()  # 시작 시 비어 있었음
        # step 실행 시뮬레이션 — 두 디렉터리 생성
        (runs / "20260301-1100-trial").mkdir()
        with patch.object(ex, "ROOT", project):
            inst = self._executor(project)
            index = inst._read_json(inst._index_file)
            new = inst._index_runs(index, step_num=0, before=before)
        assert new == ["20260301-1100-trial"]
        assert index["steps"][0]["run_dirs"] == ["20260301-1100-trial"]

    def test_extract_run_metric_from_eval_json(self, project_with_runs):
        project, runs = project_with_runs
        run_dir = runs / "20260301-1200-eval"
        run_dir.mkdir()
        (run_dir / "eval.json").write_text(json.dumps({
            "metric_primary": 0.87, "extra": "ignored"
        }))
        m = ex.StepExecutor._extract_run_metric(run_dir)
        assert m == {"metric_primary": 0.87}

    def test_extract_run_metric_from_bench_json(self, project_with_runs):
        project, runs = project_with_runs
        run_dir = runs / "20260301-1300-bench"
        run_dir.mkdir()
        (run_dir / "bench.json").write_text(json.dumps({
            "latency_p95_ms": 42, "vram_peak_mb": 1024, "fps": 60, "noise": "ignored"
        }))
        m = ex.StepExecutor._extract_run_metric(run_dir)
        assert m == {"latency_p95_ms": 42, "vram_peak_mb": 1024, "fps": 60}

    def test_extract_run_metric_returns_none_when_no_files(self, project_with_runs):
        project, runs = project_with_runs
        run_dir = runs / "20260301-1400-empty"
        run_dir.mkdir()
        assert ex.StepExecutor._extract_run_metric(run_dir) is None


# ---------------------------------------------------------------------------
# _verify_success_metric (jq 평가)
# ---------------------------------------------------------------------------

class TestVerifySuccessMetric:
    @pytest.fixture
    def setup(self, tmp_path):
        (tmp_path / "phases" / "x").mkdir(parents=True)
        (tmp_path / "phases" / "x" / "index.json").write_text(
            json.dumps({"project": "X", "phase": "x", "steps": [
                {"step": 0, "name": "a", "status": "completed"}
            ]})
        )
        (tmp_path / "runs").mkdir()
        with patch.object(ex, "ROOT", tmp_path):
            inst = ex.StepExecutor("x")
        return inst, tmp_path

    def test_no_success_metric_passes(self, setup):
        inst, _ = setup
        index = inst._read_json(inst._index_file)
        passed, msg = inst._verify_success_metric(index, 0)
        assert passed is True

    def test_missing_run_dirs_fails(self, setup):
        inst, _ = setup
        index = inst._read_json(inst._index_file)
        index["steps"][0]["success_metric"] = "jq -e '.f1 >= 0.5' runs/{run_dir}/eval.json"
        passed, msg = inst._verify_success_metric(index, 0)
        assert passed is False
        assert "산출물 없음" in msg

    def test_passes_when_jq_succeeds(self, setup):
        import shutil
        if shutil.which("jq") is None:
            pytest.skip("jq not installed")
        inst, project = setup
        run_dir = project / "runs" / "20260301-1500"
        run_dir.mkdir()
        (run_dir / "eval.json").write_text(json.dumps({"f1": 0.9}))
        index = inst._read_json(inst._index_file)
        index["steps"][0]["success_metric"] = "jq -e '.f1 >= 0.5' runs/{run_dir}/eval.json"
        index["steps"][0]["run_dirs"] = ["20260301-1500"]
        passed, _ = inst._verify_success_metric(index, 0)
        assert passed is True

    def test_fails_when_jq_fails(self, setup):
        import shutil
        if shutil.which("jq") is None:
            pytest.skip("jq not installed")
        inst, project = setup
        run_dir = project / "runs" / "20260301-1600"
        run_dir.mkdir()
        (run_dir / "eval.json").write_text(json.dumps({"f1": 0.3}))
        index = inst._read_json(inst._index_file)
        index["steps"][0]["success_metric"] = "jq -e '.f1 >= 0.5' runs/{run_dir}/eval.json"
        index["steps"][0]["run_dirs"] = ["20260301-1600"]
        passed, msg = inst._verify_success_metric(index, 0)
        assert passed is False
        assert "미달" in msg


# ---------------------------------------------------------------------------
# _handle_checkpoint — 인터랙티브 prompt (stdin mock)
# ---------------------------------------------------------------------------

class TestHandleCheckpoint:
    @pytest.fixture
    def setup(self, tmp_path):
        (tmp_path / "phases" / "x").mkdir(parents=True)
        (tmp_path / "phases" / "x" / "index.json").write_text(
            json.dumps({"project": "X", "phase": "x", "steps": [
                {"step": 0, "name": "data", "status": "completed", "summary": "ok"}
            ]})
        )
        with patch.object(ex, "ROOT", tmp_path):
            inst = ex.StepExecutor("x")
        return inst

    def test_no_tty_returns_awaiting_review(self, setup):
        inst = setup
        index = inst._read_json(inst._index_file)
        with patch("sys.stdin.isatty", return_value=False):
            result = inst._handle_checkpoint(index, 0, "CP-1", "data")
        assert result == "awaiting-review"
        saved = inst._read_json(inst._index_file)
        assert saved["steps"][0]["status"] == "awaiting-review"
        assert saved["steps"][0]["checkpoint"] == "CP-1"

    def test_tty_yes_returns_approved(self, setup):
        inst = setup
        index = inst._read_json(inst._index_file)
        with patch("sys.stdin.isatty", return_value=True), \
             patch("builtins.input", return_value="y"):
            result = inst._handle_checkpoint(index, 0, "CP-1", "data")
        assert result == "approved"
        saved = inst._read_json(inst._index_file)
        assert saved["steps"][0]["status"] == "approved"

    def test_tty_no_returns_rejected(self, setup):
        inst = setup
        index = inst._read_json(inst._index_file)
        with patch("sys.stdin.isatty", return_value=True), \
             patch("builtins.input", return_value="n"):
            result = inst._handle_checkpoint(index, 0, "CP-1", "data")
        assert result == "rejected"
        saved = inst._read_json(inst._index_file)
        assert saved["steps"][0]["status"] == "rejected"

    def test_tty_empty_response_is_rejected(self, setup):
        inst = setup
        index = inst._read_json(inst._index_file)
        with patch("sys.stdin.isatty", return_value=True), \
             patch("builtins.input", return_value=""):
            result = inst._handle_checkpoint(index, 0, "CP-1", "data")
        assert result == "rejected"


# ---------------------------------------------------------------------------
# _build_preamble — oom_directive 인자
# ---------------------------------------------------------------------------

class TestPreambleOomDirective:
    def test_no_oom_directive_default(self, executor):
        result = executor._build_preamble("", "")
        assert "OOM 자동 재시도" not in result

    def test_oom_directive_appears(self, executor):
        result = executor._build_preamble("", "", oom_directive="batch_size 1/2")
        assert "OOM 자동 재시도" in result
        assert "batch_size 1/2" in result


# ---------------------------------------------------------------------------
# monitor (단위)
# ---------------------------------------------------------------------------

class TestMonitor:
    def setup_method(self):
        import monitor
        self.monitor = monitor

    def test_parse_spec_with_port(self):
        assert self.monitor.parse_spec("tensorboard:6006") == ("tensorboard", 6006)

    def test_parse_spec_without_param(self):
        assert self.monitor.parse_spec("tensorboard") == ("tensorboard", None)

    def test_parse_spec_with_interval(self):
        assert self.monitor.parse_spec("nvidia-smi:1000") == ("nvidia-smi", 1000)

    def test_parse_spec_non_numeric_param(self):
        assert self.monitor.parse_spec("custom:hello") == ("custom", None)

    def test_start_monitors_skips_unknown_tool(self, tmp_path):
        # 알 수 없는 종류는 skip
        procs = self.monitor.start_monitors(["unknown-tool:1"], tmp_path)
        assert procs == []

    def test_start_monitors_skips_uninstalled(self, tmp_path):
        # tensorboard 가 호스트에 없으면 skip
        with patch("shutil.which", return_value=None):
            procs = self.monitor.start_monitors(["tensorboard:6006"], tmp_path)
        assert procs == []

    def test_start_monitors_launches_popen_when_installed(self, tmp_path):
        mock_proc = MagicMock(pid=1234, poll=lambda: None)
        with patch("shutil.which", return_value="/usr/bin/tensorboard"), \
             patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
            procs = self.monitor.start_monitors(["tensorboard:6006"], tmp_path)
        assert len(procs) == 1
        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == "tensorboard"
        assert "--port" in cmd
        assert "6006" in cmd

    def test_stop_monitors_terminates_running(self):
        live = MagicMock()
        live.poll.side_effect = [None, 0, 0]  # alive, then dead after wait
        live.wait.return_value = 0
        dead = MagicMock()
        dead.poll.return_value = 0  # already dead
        self.monitor.stop_monitors([live, dead])
        live.terminate.assert_called_once()
        dead.terminate.assert_not_called()

    def test_stop_monitors_kills_after_grace(self):
        stubborn = MagicMock()
        stubborn.poll.return_value = None  # 끝까지 살아있는 것처럼
        stubborn.wait.side_effect = subprocess.TimeoutExpired(cmd="x", timeout=5)
        self.monitor.stop_monitors([stubborn], grace_sec=0.01)
        stubborn.terminate.assert_called_once()
        stubborn.kill.assert_called_once()


# ---------------------------------------------------------------------------
# heartbeat (단위)
# ---------------------------------------------------------------------------

class TestHeartbeatWatchdog:
    def setup_method(self):
        import heartbeat
        self.heartbeat = heartbeat

    def test_no_heartbeat_file_does_not_fire(self, tmp_path):
        called = []
        wd = self.heartbeat.HeartbeatWatchdog(
            tmp_path, timeout_sec=1,
            on_hang=lambda: called.append(True),
            poll_interval=0.05,
        )
        wd.start()
        import time
        time.sleep(0.3)
        wd.stop()
        # 파일 없으니 fire 안 됨
        assert called == []
        assert wd.fired is False

    def test_recent_touch_does_not_fire(self, tmp_path):
        import time
        hb = tmp_path / ".heartbeat"
        hb.touch()
        called = []
        wd = self.heartbeat.HeartbeatWatchdog(
            tmp_path, timeout_sec=10,  # 길게
            on_hang=lambda: called.append(True),
            poll_interval=0.05,
        )
        wd.start()
        # 0.2초 내내 touch 갱신
        for _ in range(4):
            time.sleep(0.05)
            hb.touch()
        wd.stop()
        assert called == []

    def test_stale_heartbeat_fires(self, tmp_path):
        import time
        hb = tmp_path / ".heartbeat"
        hb.touch()
        # mtime 을 과거로 (충분히 오래된)
        past = time.time() - 100
        os.utime(hb, (past, past))
        called = []
        wd = self.heartbeat.HeartbeatWatchdog(
            tmp_path, timeout_sec=1,
            on_hang=lambda: called.append(True),
            poll_interval=0.05,
        )
        wd.start()
        time.sleep(1.5)
        wd.stop()
        assert called == [True]
        assert wd.fired is True

    def test_stop_before_fire_prevents_callback(self, tmp_path):
        import time
        hb = tmp_path / ".heartbeat"
        hb.touch()
        past = time.time() - 100
        os.utime(hb, (past, past))
        called = []
        wd = self.heartbeat.HeartbeatWatchdog(
            tmp_path, timeout_sec=2,
            on_hang=lambda: called.append(True),
            poll_interval=0.1,
        )
        wd.start()
        time.sleep(0.05)
        wd.stop()
        # timeout 도달 전 stop → fire 안 됨
        assert called == []


# ---------------------------------------------------------------------------
# budget (단위)
# ---------------------------------------------------------------------------

class TestBudget:
    def setup_method(self):
        import budget
        self.budget = budget

    def test_total_runs_simple(self):
        b = {"datasets": 2, "algorithms": 3, "seeds": 4}
        assert self.budget.compute_total_runs(b) == 24

    def test_total_runs_with_ablation(self):
        b = {
            "datasets": 1, "algorithms": 1, "seeds": 1,
            "ablation_dims": [{"name": "a", "values": [1, 2, 3]},
                              {"name": "b", "values": ["x", "y"]}],
        }
        assert self.budget.compute_total_runs(b) == 6

    def test_compute_budget_under_threshold(self):
        b = {"datasets": 1, "algorithms": 1, "seeds": 1, "per_run_hours": 10,
             "per_run_disk_gb": 5, "gpu_hours_threshold": 100, "disk_gb_threshold": 50}
        r = self.budget.compute_budget(b)
        assert r.total_gpu_hours == 10
        assert r.over_gpu is False
        assert r.over_disk is False
        assert r.any_over is False

    def test_compute_budget_over_gpu(self):
        b = {"datasets": 10, "algorithms": 10, "seeds": 1, "per_run_hours": 5,
             "gpu_hours_threshold": 100}
        r = self.budget.compute_budget(b)
        assert r.total_gpu_hours == 500
        assert r.over_gpu is True
        assert r.any_over is True

    def test_load_top_budget_returns_none_when_missing(self, tmp_path):
        p = tmp_path / "noexist.json"
        assert self.budget.load_top_budget(p) is None

    def test_load_top_budget_extracts_field(self, tmp_path):
        p = tmp_path / "phases.json"
        p.write_text(json.dumps({"phases": [], "experiment_budget": {"datasets": 5}}))
        b = self.budget.load_top_budget(p)
        assert b == {"datasets": 5}


# ---------------------------------------------------------------------------
# fairness (단위)
# ---------------------------------------------------------------------------

class TestFairness:
    def setup_method(self):
        import fairness
        self.fairness = fairness

    def _make_run(self, root, name, seed="42", git_rev="abc123", hardware="A100"):
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "seed.txt").write_text(seed)
        (d / "git_rev.txt").write_text(git_rev)
        (d / "config.yaml").write_text(f"hardware: {hardware}\ndata:\n  split: standard\n")
        (d / "eval.json").write_text(json.dumps({"metric_primary": 0.85}))
        return d

    def test_single_run_no_violation(self, tmp_path):
        runs_root = tmp_path / "runs"
        runs_root.mkdir()
        self._make_run(runs_root, "20260301-0900-a")
        report = self.fairness.verify(["20260301-0900-a"], runs_root)
        assert report.violated is False

    def test_identical_runs_no_violation(self, tmp_path):
        runs_root = tmp_path / "runs"
        runs_root.mkdir()
        self._make_run(runs_root, "a", seed="42", git_rev="abc", hardware="A100")
        self._make_run(runs_root, "b", seed="42", git_rev="abc", hardware="A100")
        report = self.fairness.verify(["a", "b"], runs_root)
        assert report.violated is False

    def test_different_git_rev_violates(self, tmp_path):
        runs_root = tmp_path / "runs"
        runs_root.mkdir()
        self._make_run(runs_root, "a", git_rev="abc")
        self._make_run(runs_root, "b", git_rev="def")
        report = self.fairness.verify(["a", "b"], runs_root)
        assert report.violated is True
        dims = [d for d, _ in report.violations]
        assert "git_rev" in dims

    def test_different_hardware_violates(self, tmp_path):
        runs_root = tmp_path / "runs"
        runs_root.mkdir()
        self._make_run(runs_root, "a", hardware="A100")
        self._make_run(runs_root, "b", hardware="V100")
        report = self.fairness.verify(["a", "b"], runs_root)
        dims = [d for d, _ in report.violations]
        assert "hardware_tag" in dims

    def test_seed_not_compared(self, tmp_path):
        # seed 는 fairness 검증 대상에서 제외 (시드 정책은 별도)
        runs_root = tmp_path / "runs"
        runs_root.mkdir()
        self._make_run(runs_root, "a", seed="42")
        self._make_run(runs_root, "b", seed="43")
        report = self.fairness.verify(["a", "b"], runs_root)
        dims = [d for d, _ in report.violations]
        assert "seed" not in dims


# ---------------------------------------------------------------------------
# preamble — resume / launcher directive
# ---------------------------------------------------------------------------

class TestPreambleDirectives:
    def test_no_resume_directive_default(self, executor):
        result = executor._build_preamble("", "")
        assert "체크포인트 재개" not in result

    def test_resume_directive_appears(self, executor):
        result = executor._build_preamble("", "", resume_directive="ckpt 에서 재개")
        assert "체크포인트 재개" in result
        assert "ckpt 에서 재개" in result

    def test_no_launcher_directive_default(self, executor):
        result = executor._build_preamble("", "")
        assert "학습 launcher" not in result

    def test_launcher_directive_appears(self, executor):
        result = executor._build_preamble("", "", launcher_directive="torchrun --nproc-per-node=4")
        assert "학습 launcher" in result
        assert "torchrun --nproc-per-node=4" in result