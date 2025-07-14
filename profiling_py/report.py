"""HTML report generation for profiling_py.

Renders a Bootstrap + Plotly report using a Jinja2 template located in
`profiling_py/templates/report.html.jinja2`.
"""
from __future__ import annotations

from importlib import resources
import json
import os
import webbrowser
from datetime import datetime
from typing import Any, Dict, List, Sequence

import jinja2
import plotly.express as px

from .core import Profiler, default_profiler

__all__ = ["generate_profiling_report"]


def _load_template() -> jinja2.Template:
    """Load the HTML Jinja2 template bundled within the package."""

    template_str = resources.files("profiling_py.templates").joinpath("report.html.jinja2").read_text(
        encoding="utf-8"
    )
    env = jinja2.Environment(autoescape=True)
    return env.from_string(template_str)


def _build_plots(rows: List[Dict[str, Any]], has_memory: bool):
    # Prepare data for duration plot
    steps = []
    durations = []
    for r in rows:
        steps.append(str(r["step"]))
        durations.append(float(r["duration"]))
    
    # Create duration plot
    duration_plot = {
        "data": [{
            "type": "bar",
            "x": durations,
            "y": steps,
            "orientation": "h"
        }],
        "layout": {
            "title": "Duration by Step",
            "yaxis": {
                "title": "Step",
                "categoryorder": "total ascending",
                "title_standoff": 30,
                "ticklen": 10,
                "tickfont": {"size": 11}
            },
            "xaxis": {
                "title": "Duration (s)",
                "type": "linear",
                "tickformat": ".3f"
            },
            "height": 400,
            "width": 650,
            "margin": {"l": 120, "r": 30, "t": 50, "b": 50}
        }
    }
    
    memory_plot = None
    if has_memory:
        # Prepare data for plotting
        steps = []
        memory_values = []
        for r in rows:
            mem_kb = r["memory_kb"]
            if mem_kb is None:
                mem_kb = 0.0
            steps.append(str(r["step"]))
            memory_values.append(float(mem_kb))
        
        # Manually create the plot data structure
        memory_plot = {
            "data": [{
                "type": "bar",
                "x": memory_values,
                "y": steps,
                "orientation": "h"
            }],
            "layout": {
                "title": "Memory Δ by Step",
                "yaxis": {
                    "title": "Step",
                    "categoryorder": "total ascending",
                    "title_standoff": 30,
                    "ticklen": 10,
                    "tickfont": {"size": 11},
                    "showticklabels": True,
                    "automargin": True
                },
                "xaxis": {
                    "title": "Memory (KB)",
                    "type": "linear",
                    "tickformat": ".1f"
                },
                "height": 400,
                "width": 650,
                "margin": {"l": 120, "r": 30, "t": 50, "b": 50}
            }
        }
        
    return duration_plot, memory_plot


def generate_profiling_report(
    profiler: Profiler | None = None,
    *,
    output_dir: str | None = "profiling_reports",
    file_name: str | None = None,
    open_browser: bool = False,
) -> str:
    """Generate an interactive HTML profiling report.

    Parameters
    ----------
    profiler
        Profiler instance to summarise. Defaults to the module-level
        :data:`profiling_py.core.default_profiler`.
    output_dir
        Directory where the report will be saved (created if missing).
        If *None*, the HTML string is returned without writing.
    file_name
        Name for the output file (*.html* added if missing). Uses timestamp by
        default.
    open_browser
        If *True*, open the generated report in the default web browser.

    Returns
    -------
    str
        Rendered HTML content.
    """

    profiler = profiler or default_profiler
    if not profiler.steps:
        raise ValueError("No profiling data available.")

    # Prepare tabular rows
    rows: List[Dict[str, Any]] = []
    for s in profiler.steps:
        mem_kb = s["memory_bytes"] / 1024 if s["memory_bytes"] is not None else None
        rows.append({
            "step": s["step"],
            "duration": round(s["duration"], 6),
            "memory_kb": round(mem_kb, 1) if mem_kb is not None else None,
        })

    has_memory = any(r["memory_kb"] is not None for r in rows)

    # KPI metrics
    total_time = sum(r["duration"] for r in rows)
    avg_time = total_time / len(rows)
    longest_step = max(rows, key=lambda r: r["duration"])
    peak_mem_kb = max((r["memory_kb"] or 0) for r in rows) if has_memory else None

    kpis: Sequence[Dict[str, str]] = [
        {"label": "Total Time (s)", "value": f"{total_time:.3f}"},
        {"label": "Average Step (s)", "value": f"{avg_time:.3f}"},
        {"label": "Longest Step (s)", "value": f"{longest_step['duration']:.3f}"},
        {"label": "Steps", "value": str(len(rows))},
    ]
    if has_memory:
        kpis = list(kpis) + [{"label": "Peak Δ (KB)", "value": f"{peak_mem_kb:.1f}"}]

    # Build plots
    duration_plot, memory_plot = _build_plots(rows, has_memory)
    duration_plot_json = json.dumps(duration_plot)
    memory_plot_json = json.dumps(memory_plot) if memory_plot is not None else None

    # Render template
    template = _load_template()
    html = template.render(
        metadata=profiler.metadata,
        kpis=kpis,
        rows=rows,
        has_memory=has_memory,
        duration_plot_json=duration_plot_json,
        memory_plot_json=memory_plot_json,
        generated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    # Persist to disk if requested
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        if file_name is None:
            file_name = f"profiling_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        if not file_name.lower().endswith(".html"):
            file_name += ".html"
        out_path = os.path.join(output_dir, file_name)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(html)
        if open_browser:
            try:
                webbrowser.open(f"file://{os.path.abspath(out_path)}")
            except webbrowser.Error:
                print(f"[profiling_py] Report saved to {out_path}. Unable to launch browser.")
    else:
        # If caller passed output_dir=None they only want the string.
        if open_browser:
            raise ValueError("Cannot open browser when output is not written to disk.")

    return html
