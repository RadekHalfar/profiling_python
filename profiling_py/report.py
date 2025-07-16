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


def _build_plots(rows: List[Dict[str, Any]], has_memory: bool, has_gpu: bool = False):
    """Build plot data for the profiling report.
    
    Args:
        rows: List of step data dictionaries
        has_memory: Whether to include memory plots
        has_gpu: Whether to include GPU metrics plots
        
    Returns:
        Tuple containing plot data for duration, memory, and GPU metrics
    """
    duration_plot = None
    memory_plot = None
    gpu_util_plot = None
    gpu_memory_plot = None
    
    # Check if we have duration data to plot
    if rows and "duration" in rows[0]:
        # Prepare data for duration plot
        steps = []
        durations = []
        for r in rows:
            if "duration" in r and r["duration"] is not None:
                steps.append(str(r["step"]))
                durations.append(float(r["duration"]))
        
        if steps:  # Only create plot if we have data
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
                    "width": 1200,
                    "margin": {"l": 120, "r": 30, "t": 50, "b": 80},
                }
            }
    
    # Check if we have memory data to plot
    if has_memory and rows and "memory_mb" in rows[0]:
        # Prepare data for memory plot
        steps = []
        memory_values = []
        for r in rows:
            if "memory_mb" in r and r["memory_mb"] is not None:
                steps.append(str(r["step"]))
                memory_values.append(float(r["memory_mb"]))
        
        # Manually create the plot data structure
        memory_plot = {
            "data": [{
                "type": "bar",
                "x": memory_values,
                "y": steps,
                "orientation": "h"
            }],
            "layout": {
                "title": "RAM Δ by Step",
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
                    "title": "RAM (MB)",
                    "type": "linear",
                    "tickformat": ".2f"
                },
                "height": 400,
                "width": 1200,
                "margin": {"l": 120, "r": 30, "t": 80, "b": 50}  # Increased top margin for memory plot
            }
        }
        
    # Build GPU utilization plot if GPU data is available
    if has_gpu and rows and any('gpu_utilization' in r for r in rows):
        # GPU Utilization Plot
        gpu_util_steps = []
        gpu_util_values = []
        
        for r in rows:
            if 'gpu_utilization' in r and r['gpu_utilization'] is not None:
                gpu_util_steps.append(str(r['step']))
                gpu_util_values.append(float(r['gpu_utilization']))
        
        if gpu_util_steps:
            gpu_util_plot = {
                "data": [{
                    "type": "bar",
                    "x": gpu_util_values,
                    "y": gpu_util_steps,
                    "orientation": "h",
                    "name": "GPU Utilization"
                }],
                "layout": {
                    "title": "GPU Utilization by Step",
                    "yaxis": {
                        "title": "Step",
                        "categoryorder": "total ascending",
                        "title_standoff": 30,
                        "ticklen": 10,
                        "tickfont": {"size": 11}
                    },
                    "xaxis": {
                        "title": "GPU Utilization (%)",
                        "range": [0, 100],
                        "ticksuffix": "%"
                    },
                    "height": 400,
                    "width": 1200,
                    "margin": {"l": 120, "r": 30, "t": 50, "b": 50}
                }
            }
    
    # Build GPU Memory plot if GPU memory data is available
    if has_gpu and rows and any('gpu_memory_used_mb' in r for r in rows):
        gpu_mem_steps = []
        gpu_mem_used = []
        gpu_mem_total = []
        
        for r in rows:
            if 'gpu_memory_used_mb' in r and r['gpu_memory_used_mb'] is not None:
                gpu_mem_steps.append(str(r['step']))
                gpu_mem_used.append(float(r['gpu_memory_used_mb']))
                gpu_mem_total.append(float(r.get('gpu_memory_total_mb', 0)))
        
        if gpu_mem_steps:
            gpu_memory_plot = {
                "data": [
                    {
                        "type": "bar",
                        "x": gpu_mem_used,
                        "y": gpu_mem_steps,
                        "orientation": "h",
                        "name": "Used Memory"
                    },
                    {
                        "type": "bar",
                        "x": [total - used for used, total in zip(gpu_mem_used, gpu_mem_total)],
                        "y": gpu_mem_steps,
                        "orientation": "h",
                        "name": "Free Memory"
                    }
                ],
                "layout": {
                    "title": "GPU Memory Usage by Step",
                    "yaxis": {
                        "title": "Step",
                        "categoryorder": "total ascending",
                        "title_standoff": 30,
                        "ticklen": 10,
                        "tickfont": {"size": 11}
                    },
                    "xaxis": {
                        "title": "GPU Memory (MB)",
                        "tickformat": ",.1f"
                    },
                    "barmode": "stack",
                    "height": 400,
                    "width": 1200,
                    "margin": {"l": 120, "r": 30, "t": 50, "b": 50},
                    "legend": {"orientation": "h", "y": -0.2}
                }
            }
    
    return duration_plot, memory_plot, gpu_util_plot, gpu_memory_plot


def generate_profiling_report(
    profiler: Profiler | None = None,
    *,
    output_dir: str | None = "profiling_reports",
    file_name: str | None = None,
    open_browser: bool = False,
    measure_time: bool = True,
    measure_ram: bool = True,
    measure_gpu: bool = True,
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
    measure_time
        If *True* (default), include time measurements in the report.
    measure_ram
        If *True* (default), include RAM measurements in the report.

    Returns
    -------
    str
        Rendered HTML content.
    """

    profiler = profiler or default_profiler
    if not profiler.steps:
        raise ValueError("No profiling data available.")

    # Get the raw steps data
    steps = profiler._steps
    rows: List[Dict[str, Any]] = []
    
    # Process each step to create rows with all required fields
    prev_gpu_memory = None
    for step in steps:
        row = {
            "step": step.step,
            "duration": step.duration if measure_time and step.duration is not None else None,
            "memory_mb": (step.memory_bytes / (1024.0 * 1024.0)) if measure_ram and step.memory_bytes is not None else None,
        }
        
        # Add GPU metrics if available
        if measure_gpu and step.gpu_metrics:
            # First, add all GPU metrics with gpu_ prefix
            for key, value in step.gpu_metrics.items():
                if key == 'utilization':
                    # Special handling for utilization to ensure it's always called gpu_utilization
                    row["gpu_utilization"] = value
                elif key != 'name':  # Skip the name as it's in metadata
                    row[f"gpu_{key}"] = value
            
            # Calculate GPU memory delta if not already calculated
            if 'gpu_memory_used_mb' in row and row['gpu_memory_used_mb'] is not None:
                current_memory = float(row['gpu_memory_used_mb'])
                if prev_gpu_memory is not None:
                    # Only calculate delta if not already provided
                    if 'gpu_memory_delta_mb' not in row:
                        row['gpu_memory_delta_mb'] = current_memory - prev_gpu_memory
                prev_gpu_memory = current_memory
        
        rows.append(row)
    
    # Check if we have any memory and GPU data
    has_memory = measure_ram and any(r.get("memory_mb") is not None for r in rows)
    has_gpu = measure_gpu and any(k.startswith('gpu_') for r in rows for k in r)
    
    # Build plots
    duration_plot, memory_plot, gpu_util_plot, gpu_memory_plot = _build_plots(
        rows, has_memory=measure_ram, has_gpu=measure_gpu
    )

    # KPI metrics
    kpis: List[Dict[str, str]] = []
    
    if measure_time:
        total_time = sum(r["duration"] for r in rows if "duration" in r)
        avg_time = total_time / len(rows) if rows else 0
        longest_step = max(rows, key=lambda r: r.get("duration", 0), default={"duration": 0})
        kpis.extend([
            {"label": "Total Time (s)", "value": f"{total_time:.3f}"},
            {"label": "Average Step (s)", "value": f"{avg_time:.3f}"},
            {"label": "Longest Step (s)", "value": f"{longest_step.get('duration', 0):.3f}"},
        ])
    
    kpis.append({"label": "Steps", "value": str(len(rows))})
    
    if profiler.enable_memory and has_memory:
        peak_mem_mb = max((r.get("memory_mb", 0) for r in rows), default=0)
        kpis.append({"label": "Peak RAM Δ (MB)", "value": f"{peak_mem_mb:.2f}"})
    
    # Add GPU KPIs if available
    if profiler.enable_gpu and has_gpu:
        # Get GPU name from metadata if available
        gpu_name = profiler.metadata.get('gpu', 'GPU')
        
        # Add GPU utilization KPI
        gpu_utils = [r.get('gpu_utilization', 0) for r in rows if 'gpu_utilization' in r]
        if gpu_utils:
            avg_gpu_util = sum(gpu_utils) / len(gpu_utils)
            max_gpu_util = max(gpu_utils)
            kpis.extend([
                {"label": f"{gpu_name} Avg Util (%)", "value": f"{avg_gpu_util:.1f}"},
                {"label": f"{gpu_name} Peak Util (%)", "value": f"{max_gpu_util:.1f}"},
            ])
        
        # Add GPU memory KPI
        gpu_mem_deltas = [r.get('gpu_memory_delta_mb', 0) for r in rows if 'gpu_memory_delta_mb' in r]
        if gpu_mem_deltas:
            max_gpu_mem_delta = max(gpu_mem_deltas)
            kpis.append({"label": f"{gpu_name} Peak RAM Δ (MB)", "value": f"{max_gpu_mem_delta:.1f}"})

    # Build plots
    duration_plot_json = None
    memory_plot_json = None
    gpu_util_plot_json = None
    gpu_memory_plot_json = None
    
    # Get rows with the required data for each plot type
    time_rows = [r for r in rows if "duration" in r and r["duration"] is not None]
    memory_rows = [r for r in rows if "memory_mb" in r and r["memory_mb"] is not None]
    gpu_util_rows = [r for r in rows if "gpu_utilization" in r and r["gpu_utilization"] is not None]
    gpu_memory_rows = [r for r in rows if "gpu_memory_delta_mb" in r and r["gpu_memory_delta_mb"] is not None]
    
    # Generate plots
    if profiler.enable_time and time_rows:
        duration_plot, _, _, _ = _build_plots(time_rows, False, False)
        if duration_plot:
            duration_plot_json = json.dumps(duration_plot)
    
    if profiler.enable_memory and memory_rows:
        _, memory_plot, _, _ = _build_plots(memory_rows, True, False)
        if memory_plot:
            memory_plot_json = json.dumps(memory_plot)
    
    if profiler.enable_gpu and gpu_util_rows:
        _, _, gpu_util_plot, _ = _build_plots(gpu_util_rows, False, True)
        if gpu_util_plot:
            gpu_util_plot_json = json.dumps(gpu_util_plot)
    
    if profiler.enable_gpu and gpu_memory_rows:
        _, _, _, gpu_memory_plot = _build_plots(gpu_memory_rows, False, True)
        if gpu_memory_plot:
            gpu_memory_plot_json = json.dumps(gpu_memory_plot)

    # Render template
    template = _load_template()
    html = template.render(
        metadata=profiler.metadata,
        kpis=kpis,
        rows=rows,
        has_time=profiler.enable_time,
        has_memory=has_memory and profiler.enable_memory,
        has_gpu=has_gpu and profiler.enable_gpu,
        duration_plot_json=duration_plot_json or 'null',
        memory_plot_json=memory_plot_json or 'null',
        gpu_util_plot_json=gpu_util_plot_json or 'null',
        gpu_memory_plot_json=gpu_memory_plot_json or 'null',
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
