"""
Crash classifier — step 비정상 종료의 원인을 10 카테고리로 분류한다.

plan.md §3-13-B 참고. 분류 결과는 execute.py 가 phases/{task}/index.json 의
step 항목에 crash_reason / crash_evidence / recommended_action 으로 자동 기록한다.
"""

from __future__ import annotations

import re
import signal
from dataclasses import dataclass
from typing import Optional

# (reason, 권장 조치, 패턴들) — 위에서 아래로 매칭, 첫 매치 사용
CRASH_PATTERNS: list[tuple[str, str, list[re.Pattern]]] = [
    (
        "OOM",
        "batch_size 축소 (1/2), mixed precision(fp16/bf16), gradient accumulation",
        [
            re.compile(r"CUDA out of memory", re.IGNORECASE),
            re.compile(r"cuda::OutOfMemoryError", re.IGNORECASE),
            re.compile(r"torch\.cuda\.OutOfMemoryError", re.IGNORECASE),
        ],
    ),
    (
        "NaN/Inf",
        "lr 낮춤, gradient clipping, fp32 sanity 재실행, loss-sanity step 재검증",
        [
            re.compile(r"loss is not finite", re.IGNORECASE),
            re.compile(r"AssertionError.*isfinite", re.IGNORECASE),
            re.compile(r"\bNaN\b.*loss|\bloss\b.*\bNaN\b", re.IGNORECASE),
            re.compile(r"\bInf\b.*loss|\bloss\b.*\bInf\b", re.IGNORECASE),
        ],
    ),
    (
        "SHM",
        "docker-compose.yml 의 dev 서비스에 shm_size: 8gb (또는 ipc: host) 추가",
        [
            re.compile(r"bus error.*DataLoader|DataLoader.*bus error", re.IGNORECASE),
            re.compile(r"shared memory.*(?:full|exhausted|insufficient)", re.IGNORECASE),
        ],
    ),
    (
        "DataLoader",
        "데이터셋 매니페스트 hash 재검증, 마운트/경로 확인, data-sanity step 재실행",
        [
            re.compile(r"FileNotFoundError"),
            re.compile(r"No such file or directory"),
            re.compile(r"DataLoader.*(?:Error|failed)", re.IGNORECASE),
            re.compile(r"Image\.open.*error|cannot identify image file", re.IGNORECASE),
        ],
    ),
    (
        "GPU/driver",
        "nvidia-smi 로 GPU 상태 확인, 컨테이너 재시작, driver 버전 확인",
        [
            re.compile(r"CUDA error: unspecified launch failure", re.IGNORECASE),
            re.compile(r"no CUDA-capable device", re.IGNORECASE),
            re.compile(r"CUDA driver version is insufficient", re.IGNORECASE),
            re.compile(r"NVML.*(?:Error|failed)", re.IGNORECASE),
        ],
    ),
    (
        "Disk full",
        "runs/ cleanup (오래된 체크포인트 삭제), df -h 로 잔량 확인",
        [
            re.compile(r"No space left on device", re.IGNORECASE),
            re.compile(r"Disk quota exceeded", re.IGNORECASE),
        ],
    ),
    (
        "NCCL",
        "네트워크/포트 확인, NCCL_DEBUG=INFO 로 재실행, NCCL_SOCKET_IFNAME 점검",
        [
            re.compile(r"NCCL.*error", re.IGNORECASE),
            re.compile(r"ncclSystemError", re.IGNORECASE),
            re.compile(r"ncclUnhandledCudaError", re.IGNORECASE),
        ],
    ),
    (
        "Preemption",
        "체크포인트 기반 재개 (resume_from), 작업 환경 안정성 점검",
        [
            re.compile(r"Terminated by signal SIGTERM"),
            re.compile(r"Received SIGTERM"),
            re.compile(r"Killed by signal"),
        ],
    ),
]

# Hung 은 별도 — execute.py 의 heartbeat watchdog 가 직접 reason="Hung" 으로 설정


@dataclass(frozen=True)
class CrashReport:
    reason: str
    evidence: str
    recommended_action: str


def classify(
    stderr: str,
    returncode: int,
    *,
    stdout_tail: Optional[str] = None,
    timed_out: bool = False,
) -> CrashReport:
    """stderr (필요 시 stdout 일부) 를 보고 10 카테고리 중 하나로 분류한다.

    timed_out=True 이면 Hung 으로 분류. SIGKILL/SIGTERM 신호 종료도 별도 처리.
    어디에도 안 맞으면 Unknown.
    """
    evidence = _last_lines(stderr, 20)

    if timed_out:
        return CrashReport(
            reason="Hung",
            evidence=evidence,
            recommended_action="heartbeat 끊김 / deadlock 의심. 학습 스크립트의 .heartbeat 갱신과 monitor 로그 확인",
        )

    # 신호 종료 — SIGKILL(-9) / SIGTERM(-15)
    if returncode in (-signal.SIGKILL, -signal.SIGTERM, 137, 143):
        return CrashReport(
            reason="Preemption",
            evidence=evidence,
            recommended_action="체크포인트 기반 재개 (resume_from), 작업 환경 안정성 점검",
        )

    haystack = stderr if stdout_tail is None else f"{stderr}\n{stdout_tail}"

    for reason, recommendation, patterns in CRASH_PATTERNS:
        for pat in patterns:
            if pat.search(haystack):
                return CrashReport(
                    reason=reason,
                    evidence=evidence,
                    recommended_action=recommendation,
                )

    return CrashReport(
        reason="Unknown",
        evidence=evidence,
        recommended_action="stderr 전문을 사용자 검토 후 사인 결정 — 자동 재시도 금지",
    )


def _last_lines(text: str, n: int) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[-n:])
