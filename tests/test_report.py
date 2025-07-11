"""Tests for profiling_py.report generate_profiling_report."""

from pathlib import Path

import pytest

from profiling_py.core import Profiler
from profiling_py.report import generate_profiling_report


@pytest.fixture()
def dummy_profiler() -> Profiler:
    p = Profiler(enable_memory=False, metadata={"script": "unit_test"})
    with p.profile("s1"):
        x = [0] * 1000  # noqa: F841 – allocate
    with p.profile("s2"):
        y = [1] * 2000  # noqa: F841 – allocate
    return p


def test_generate_report_string(dummy_profiler: Profiler):
    html = generate_profiling_report(dummy_profiler, output_dir=None)
    assert "<html" in html.lower()
    assert "Profiling Report" in html
    # Should include the KPI label
    assert "Total Time" in html


def test_generate_report_file(tmp_path: Path, dummy_profiler: Profiler):
    output_dir = tmp_path / "reports"
    html = generate_profiling_report(dummy_profiler, output_dir=str(output_dir), file_name="rep.html")
    out_file = output_dir / "rep.html"
    assert out_file.exists()
    assert html.startswith("<!DOCTYPE html>") or "<html" in html.lower()
