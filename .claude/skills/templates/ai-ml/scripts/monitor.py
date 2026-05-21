"""
Monitor — step 실행과 함께 백그라운드로 tensorboard / nvidia-smi 를 띄운다.

plan.md §3-11-B 참고. step.monitors 필드의 'tensorboard:6006' / 'nvidia-smi:1000' 같은
스펙을 파싱해 Popen 으로 띄우고, step 종료 시 모두 terminate.

설치 안 된 도구는 경고 + skip (학습 자체를 막지 않는다).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


def parse_spec(spec: str) -> tuple[str, Optional[int]]:
    """'tensorboard:6006' → ('tensorboard', 6006). ':' 없으면 파라미터 None."""
    if ":" not in spec:
        return spec, None
    name, param = spec.split(":", 1)
    try:
        return name.strip(), int(param)
    except ValueError:
        return name.strip(), None


def start_monitors(specs: list[str], run_dir: Path) -> list[subprocess.Popen]:
    """specs (예: ['tensorboard:6006', 'nvidia-smi:1000']) 를 백그라운드 띄움.

    설치 안 됐거나 GPU 없는 경우 해당 monitor 만 skip. 반환된 Popen 리스트는
    stop_monitors() 로 정리.
    """
    procs: list[subprocess.Popen] = []
    run_dir.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        name, param = parse_spec(spec)
        if name == "tensorboard":
            port = param or 6006
            if shutil.which("tensorboard") is None:
                print(f"  ⚠ monitor: tensorboard 미설치 — skip", file=sys.stderr)
                continue
            log_path = run_dir / "tb.log"
            proc = subprocess.Popen(
                ["tensorboard", "--logdir", str(run_dir), "--port", str(port),
                 "--host", "0.0.0.0", "--reload_interval", "5"],
                stdout=log_path.open("ab"), stderr=subprocess.STDOUT,
            )
            procs.append(proc)
            print(f"  ▶ monitor: tensorboard :{port} (pid {proc.pid})")

        elif name == "nvidia-smi":
            interval_ms = param or 1000
            if shutil.which("nvidia-smi") is None:
                print(f"  ⚠ monitor: nvidia-smi 미설치 (GPU 없음?) — skip", file=sys.stderr)
                continue
            gpu_log = run_dir / "gpu.log"
            proc = subprocess.Popen(
                ["nvidia-smi",
                 "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits",
                 f"-lms", str(interval_ms)],
                stdout=gpu_log.open("ab"), stderr=subprocess.DEVNULL,
            )
            procs.append(proc)
            print(f"  ▶ monitor: nvidia-smi {interval_ms}ms (pid {proc.pid})")

        else:
            print(f"  ⚠ monitor: 알 수 없는 종류 '{name}' — skip", file=sys.stderr)

    return procs


def stop_monitors(procs: list[subprocess.Popen], grace_sec: float = 5.0) -> None:
    """모든 monitor process 에 SIGTERM, grace_sec 후에도 살아있으면 SIGKILL."""
    for p in procs:
        if p.poll() is not None:
            continue
        try:
            p.terminate()
        except Exception:
            pass

    for p in procs:
        if p.poll() is not None:
            continue
        try:
            p.wait(timeout=grace_sec)
        except subprocess.TimeoutExpired:
            try:
                p.kill()
            except Exception:
                pass

    if procs:
        alive = sum(1 for p in procs if p.poll() is None)
        print(f"  ◼ monitors stopped ({len(procs) - alive}/{len(procs)} clean)")
