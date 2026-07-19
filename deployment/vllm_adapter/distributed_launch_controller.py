"""D4B Part G: bounded vLLM server launch lifecycle controller.

Starts a materialized D3B launch spec's argv as a real subprocess (never
via shell=True, never with string interpolation), polls the OpenAI-
compatible health endpoint for readiness under a bounded timeout, tracks
the full descendant process tree, and guarantees termination (graceful
SIGTERM first, escalating to SIGKILL) with a verified zero-descendant
post-condition.

No `force_launch`, `ignore_preflight`, or `allow_unsupported` parameter
exists anywhere on this controller -- it only ever starts an argv that a
passed-in, already-successful preflight result produced.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import psutil
import requests


class LaunchState(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    STARTING = "STARTING"
    WAITING_FOR_READINESS = "WAITING_FOR_READINESS"
    READY = "READY"
    REQUEST_ACTIVE = "REQUEST_ACTIVE"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"


class LaunchControllerError(RuntimeError):
    """Fail-closed: a required lifecycle invariant was violated."""


@dataclass
class LifecycleEvent:
    ts: float
    event: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"ts": self.ts, "event": self.event, "detail": self.detail}


@dataclass
class ServerLaunchController:
    argv: tuple[str, ...]
    env: dict[str, str]
    cwd: str
    log_path: Path
    host: str
    port: int
    state: LaunchState = LaunchState.NOT_STARTED
    process: subprocess.Popen | None = field(default=None, repr=False)
    pid: int | None = None
    pgid: int | None = None
    events: list[LifecycleEvent] = field(default_factory=list)
    start_ts: float | None = None
    ready_ts: float | None = None
    stop_ts: float | None = None
    exit_code: int | None = None
    descendant_pids_at_ready: list[int] = field(default_factory=list)

    def _record(self, event: str, detail: str = "") -> None:
        self.events.append(LifecycleEvent(ts=time.time(), event=event, detail=detail))

    def start(self) -> None:
        if self.state != LaunchState.NOT_STARTED:
            raise LaunchControllerError(f"cannot start from state {self.state}")
        self.state = LaunchState.STARTING
        self._record("starting", " ".join(self.argv))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(self.log_path, "wb", buffering=0)
        self.start_ts = time.time()
        # No shell=True, argv passed as a list; start_new_session=True (setsid)
        # so the whole worker process group can be signaled together.
        self.process = subprocess.Popen(
            list(self.argv), cwd=self.cwd, env=self.env,
            stdout=log_file, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.pid = self.process.pid
        try:
            self.pgid = os.getpgid(self.pid)
        except ProcessLookupError:
            self.pgid = None
        self._record("started", f"pid={self.pid} pgid={self.pgid}")
        self.state = LaunchState.WAITING_FOR_READINESS

    def _premature_exit(self) -> int | None:
        assert self.process is not None
        return self.process.poll()

    def wait_for_readiness(self, *, timeout_s: float, poll_interval_s: float = 2.0) -> bool:
        if self.state != LaunchState.WAITING_FOR_READINESS:
            raise LaunchControllerError(f"cannot wait for readiness from state {self.state}")
        deadline = time.time() + timeout_s
        health_url = f"http://{self.host}:{self.port}/health"
        while time.time() < deadline:
            exit_code = self._premature_exit()
            if exit_code is not None:
                self.exit_code = exit_code
                self.state = LaunchState.FAILED
                self._record("premature_exit", f"exit_code={exit_code}")
                return False
            try:
                resp = requests.get(health_url, timeout=3)
                if resp.status_code == 200:
                    self.ready_ts = time.time()
                    self.state = LaunchState.READY
                    self._record("ready", f"readiness_latency_s={self.ready_ts - self.start_ts:.2f}")
                    self.descendant_pids_at_ready = self.descendant_pids()
                    self._record("descendants_at_ready", str(self.descendant_pids_at_ready))
                    return True
            except requests.RequestException:
                pass
            time.sleep(poll_interval_s)
        self.state = LaunchState.TIMED_OUT
        self._record("startup_timeout", f"timeout_s={timeout_s}")
        return False

    def descendant_pids(self) -> list[int]:
        if self.pid is None:
            return []
        try:
            parent = psutil.Process(self.pid)
        except psutil.NoSuchProcess:
            return []
        return [c.pid for c in parent.children(recursive=True)]

    def all_tracked_pids(self) -> list[int]:
        pids = [] if self.pid is None else [self.pid]
        return pids + self.descendant_pids()

    def mark_request_active(self) -> None:
        if self.state != LaunchState.READY:
            raise LaunchControllerError(f"cannot send a request from state {self.state}")
        self.state = LaunchState.REQUEST_ACTIVE

    def mark_request_done(self) -> None:
        if self.state == LaunchState.REQUEST_ACTIVE:
            self.state = LaunchState.READY

    def stop(self, *, graceful_timeout_s: float = 30.0) -> dict[str, Any]:
        if self.process is None:
            self.state = LaunchState.STOPPED
            return {"already_stopped": True}
        # If the process already exited abnormally (FAILED/TIMED_OUT) before
        # stop() was ever called, preserve that terminal classification --
        # stop() still performs its descendant-sweep/cleanup duty below, but
        # must not silently relabel a real failure as a plain STOPPED.
        already_terminal_failure = self.state in (LaunchState.FAILED, LaunchState.TIMED_OUT)
        if not already_terminal_failure:
            self.state = LaunchState.STOPPING
        self._record("stopping")
        pre_stop_descendants = self.descendant_pids()

        try:
            if self.pgid is not None:
                os.killpg(self.pgid, signal.SIGTERM)
            else:
                self.process.terminate()
        except ProcessLookupError:
            pass

        graceful_ok = False
        deadline = time.time() + graceful_timeout_s
        while time.time() < deadline:
            if self.process.poll() is not None:
                graceful_ok = True
                break
            time.sleep(1.0)

        escalated = False
        if not graceful_ok:
            escalated = True
            self._record("graceful_shutdown_timed_out_escalating")
            try:
                if self.pgid is not None:
                    os.killpg(self.pgid, signal.SIGKILL)
                else:
                    self.process.kill()
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass

        # Sweep any remaining descendants individually (defense in depth --
        # some vLLM worker subprocesses may detach from the process group).
        remaining = self.descendant_pids()
        for rpid in remaining:
            try:
                os.kill(rpid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if remaining:
            time.sleep(2.0)

        self.exit_code = self.process.poll()
        self.stop_ts = time.time()
        final_remaining = self.descendant_pids()
        if not already_terminal_failure:
            self.state = LaunchState.STOPPED
        self._record("stopped", f"exit_code={self.exit_code} escalated={escalated} "
                                f"final_remaining_descendants={final_remaining} "
                                f"preserved_prior_terminal_state={already_terminal_failure}")
        return {
            "graceful": graceful_ok, "escalated": escalated,
            "pre_stop_descendant_pids": pre_stop_descendants,
            "final_remaining_descendant_pids": final_remaining,
            "zero_orphans": len(final_remaining) == 0,
            "shutdown_latency_s": self.stop_ts - (self.ready_ts or self.start_ts),
            "preserved_prior_terminal_state": already_terminal_failure,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv), "host": self.host, "port": self.port,
            "state": self.state.value, "pid": self.pid, "pgid": self.pgid,
            "start_ts": self.start_ts, "ready_ts": self.ready_ts, "stop_ts": self.stop_ts,
            "readiness_latency_s": (self.ready_ts - self.start_ts) if (self.ready_ts and self.start_ts) else None,
            "exit_code": self.exit_code,
            "descendant_pids_at_ready": self.descendant_pids_at_ready,
            "events": [e.to_dict() for e in self.events],
            "log_path": str(self.log_path),
        }
