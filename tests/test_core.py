"""Unit tests for profiling_py.core (phase-1 timing only)."""

import pytest

from profiling_py.core import Profiler, start_step, end_step, profile, default_profiler


def test_profiler_basic():
    prof = Profiler()
    prof.start_step("load_data")
    # simulate work
    prof.end_step("load_data")

    assert len(prof.steps) == 1
    row = prof.steps[0]
    assert row["step"] == "load_data"
    assert row["duration"] >= 0


def test_context_manager():
    prof = Profiler()
    with prof.profile("cm_step"):
        pass
    assert prof.steps[0]["step"] == "cm_step"


def test_memory_tracking():
    prof = Profiler(enable_memory=True)
    with prof.profile("mem_step"):
        x = [0] * 10000  # allocate some memory
    assert prof.steps[0]["memory_bytes"] is not None


def test_module_level_helpers():
    start_step("helper")
    end_step("helper")
    assert default_profiler.steps[-1]["step"] == "helper"
