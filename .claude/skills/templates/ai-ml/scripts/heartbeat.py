"""
Heartbeat watchdog — 학습 스크립트가 runs/{id}/.heartbeat 를 매 N초 touch 한다는
컨벤션 위에서, mtime 이 timeout_sec 초 이상 갱신되지 않으면 on_hang 콜백을 호출.

plan.md §3-13-C 참고. execute.py 가 step 실행과 별도 스레드로 띄워 둔다.

학습 스크립트의 의무 (templates/ai-ml/skills/harness.md §5-5 참고):
    Path("runs/{id}/.heartbeat").touch()  # 매 epoch 또는 매 60초
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Optional


class HeartbeatWatchdog:
    """별도 스레드로 .heartbeat 파일의 mtime 을 폴링.

    - heartbeat 파일이 *생성되기 전* 까지는 hang 으로 보지 않음 (학습 초기화 시간 허용).
    - 파일이 한 번이라도 touch 된 뒤로 timeout_sec 초 동안 갱신 없으면 on_hang 호출.
    - stop() 호출 시 즉시 정리.
    """

    DEFAULT_TIMEOUT_SEC = 600
    POLL_INTERVAL_SEC = 5.0

    def __init__(
        self,
        run_dir: Path,
        timeout_sec: int = DEFAULT_TIMEOUT_SEC,
        on_hang: Optional[Callable[[], None]] = None,
        poll_interval: float = POLL_INTERVAL_SEC,
    ):
        self.run_dir = Path(run_dir)
        self.timeout_sec = timeout_sec
        self.on_hang = on_hang or (lambda: None)
        self.poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fired = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self, join_timeout: float = 1.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)

    @property
    def fired(self) -> bool:
        return self._fired

    def _heartbeat_path(self) -> Path:
        return self.run_dir / ".heartbeat"

    def _loop(self) -> None:
        last_seen_mtime: Optional[float] = None
        last_seen_at = time.monotonic()

        while not self._stop.wait(self.poll_interval):
            hb = self._heartbeat_path()
            if not hb.exists():
                # 학습 초기화 중 — hang 으로 보지 않음
                last_seen_at = time.monotonic()
                continue

            try:
                mtime = hb.stat().st_mtime
            except FileNotFoundError:
                continue

            if last_seen_mtime is None or mtime > last_seen_mtime:
                last_seen_mtime = mtime
                last_seen_at = time.monotonic()
                continue

            # mtime 변화 없음 — 얼마나 됐는지
            silent_for = time.monotonic() - last_seen_at
            if silent_for >= self.timeout_sec:
                self._fired = True
                try:
                    self.on_hang()
                except Exception:
                    pass
                return  # 한 번 fire 후 종료
