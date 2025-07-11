"""Core profiling functionality.

Phase 1 implementation: timing-only (memory will be added later).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import tracemalloc
import time

__all__ = [
    "Profiler",
    "default_profiler",
    "start_step",
    "end_step",
    "profile",
]


@dataclass
class _StepRow:
    """Internal representation of a completed profiling step."""

    step: str
    start_time: float
    end_time: float
    duration: float
    memory_bytes: Optional[int] = None  # will be filled once memory tracking is added

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class Profiler:
    """Lightweight, step-based profiler.

    Parameters
    ----------
    enable_memory : bool, default False
        If *True*, memory usage will be captured (not yet implemented).
    metadata : dict[str, str] | None
        Optional user metadata to embed in the final report.
    """

    def __init__(self, *, enable_memory: bool = False, metadata: Optional[Dict[str, str]] = None) -> None:
        self.enable_memory = enable_memory
        self.metadata: Dict[str, str] = metadata or {}
        self.steps: List[Dict[str, object]] = []
        # Map step name -> (start_time, tracemalloc.Snapshot | None)
        self._open_steps: Dict[str, tuple[float, Optional["tracemalloc.Snapshot"]]] = {}
        self._start_time = time.perf_counter()

        # Start tracemalloc if memory tracking requested
        if self.enable_memory and not tracemalloc.is_tracing():
            tracemalloc.start()

    # ---------------------------------------------------------------------
    # Step control API
    # ---------------------------------------------------------------------
    def start_step(self, name: str) -> None:
        """Mark the start of a profiling *step*.

        Raises
        ------
        ValueError
            If a step with the same *name* is already running.
        """
        if name in self._open_steps:
            raise ValueError(f"Step '{name}' already started.")

        start_time = time.perf_counter()
        snapshot_before = tracemalloc.take_snapshot() if self.enable_memory else None
        self._open_steps[name] = (start_time, snapshot_before)

    def end_step(self, name: str) -> Dict[str, object]:
        """Finish a profiling step and record results."""
        try:
            start_time, snap_before = self._open_steps.pop(name)
        except KeyError as exc:
            raise ValueError(f"Step '{name}' was never started.") from exc

        end_time = time.perf_counter()
        duration = end_time - start_time

        memory_bytes: Optional[int] = None
        if self.enable_memory:
            snap_after = tracemalloc.take_snapshot()
            if snap_before is not None:
                stats = snap_after.compare_to(snap_before, "filename")
                memory_bytes = sum(s.size_diff for s in stats)
        row = _StepRow(
            step=name,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            memory_bytes=memory_bytes,
        )
        self.steps.append(row.to_dict())
        return self.steps[-1]

    # ------------------------------------------------------------------
    # Context-manager / decorator helpers
    # ------------------------------------------------------------------
    @contextmanager
    def profile(self, name: str):  # noqa: D401 – short name
        """Context manager that profiles a code block as *name*."""
        self.start_step(name)
        try:
            yield
        finally:
            self.end_step(name)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def total_duration(self) -> float:
        """Total duration (seconds) measured so far."""
        return sum(step["duration"] for step in self.steps)


# -----------------------------------------------------------------------
# Module-level default profiler for convenience use – mirrors R behaviour
# -----------------------------------------------------------------------

default_profiler = Profiler()


def start_step(name: str) -> None:  # pragma: no cover – thin wrapper
    """Start a step using the *default_profiler*."""
    default_profiler.start_step(name)


def end_step(name: str) -> Dict[str, object]:  # pragma: no cover
    """End a step using the *default_profiler*."""
    return default_profiler.end_step(name)


@contextmanager  # pragma: no cover
def profile(name: str):
    """Context manager delegating to :data:`default_profiler`."""
    with default_profiler.profile(name):
        yield
