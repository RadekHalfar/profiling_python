"""profiling_py package

Lightweight step-based profiler with optional memory tracking and HTML reporting.
Ported from the R package `profiling`.
"""

from .core import (
    Profiler,
    default_profiler,
    start_step,
    end_step,
    profile,
)
from .report import generate_profiling_report

__all__ = [
    "Profiler",
    "default_profiler",
    "start_step",
    "end_step",
    "profile",
    "generate_profiling_report",
]
