"""Core profiling functionality with support for time, memory, and GPU measurements."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any, Tuple
import time
import warnings
import psutil
import os

from .gpu import GPUProfiler, get_default_gpu_profiler

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
    memory_bytes: Optional[int] = None
    gpu_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class Profiler:
    """Lightweight, step-based profiler with support for time, memory, and GPU measurements.

    Parameters
    ----------
    enable_time : bool, default True
        If *True*, time measurements will be captured for each step.
    enable_memory : bool, default False
        If *True*, memory usage will be captured for each step.
    enable_gpu : bool, default False
        If *True*, GPU metrics will be captured for each step.
    gpu_device : int, default 0
        Index of the GPU device to monitor when enable_gpu is True.
    metadata : dict[str, str] | None
        Optional user metadata to embed in the final report.
    """

    def __init__(self, *, 
                enable_time: bool = True,
                enable_memory: bool = False,
                enable_gpu: bool = False,
                gpu_device: int = 0,
                metadata: Optional[Dict[str, str]] = None
            ) -> None:
        self.enable_time = enable_time
        self.enable_memory = enable_memory
        self.enable_gpu = enable_gpu
        self.gpu_device = gpu_device
        self.metadata = metadata or {}
        self._active_steps: Dict[str, Tuple[float, Optional[GPUProfiler]]] = {}
        self._steps: List[_StepRow] = []
        self._gpu_profiler: Optional[GPUProfiler] = None
        self._process: Optional[psutil.Process] = None

        if self.enable_memory:
            self._process = psutil.Process(os.getpid())
            
        if self.enable_gpu:
            try:
                self._gpu_profiler = GPUProfiler(device_index=self.gpu_device)
                self._gpu_profiler.initialize()
                # Add GPU info to metadata
                gpu_info = self._gpu_profiler.get_gpu_metrics()
                if gpu_info and 'name' in gpu_info:
                    self.metadata['gpu'] = gpu_info['name']
            except Exception as e:
                warnings.warn(f"Failed to initialize GPU profiler: {e}")
                self.enable_gpu = False

    def __del__(self):
        """Clean up resources when the profiler is garbage collected."""
        if hasattr(self, '_process'):
            del self._process
        if hasattr(self, '_gpu_profiler') and self._gpu_profiler:
            try:
                self._gpu_profiler.shutdown()
            except:
                pass

    # ---------------------------------------------------------------------
    # Step control API
    # ---------------------------------------------------------------------
    def start_step(self, name: str) -> None:
        """Start a new step with the given name.

        Parameters
        ----------
        name : str
            Unique identifier for the step.
        """
        if name in self._active_steps:
            raise ValueError(f"Step '{name}' is already running")

        start_time = time.perf_counter() if self.enable_time else 0.0
        gpu_metrics = {}
        
        if self.enable_gpu and self._gpu_profiler and self._gpu_profiler._initialized:
            try:
                gpu_metrics = self._gpu_profiler.get_gpu_metrics()
            except Exception as e:
                warnings.warn(f"Error getting initial GPU metrics for step '{name}': {e}")
        
        self._active_steps[name] = (start_time, gpu_metrics)

    def end_step(self, name: str) -> Dict[str, object]:
        """Finish a profiling step and record results."""
        try:
            start_time, start_gpu_metrics = self._active_steps.pop(name)
        except KeyError as exc:
            raise ValueError(f"Step '{name}' was never started.") from exc

        end_time = time.perf_counter()
        duration = end_time - start_time

        # Get memory usage
        memory_bytes: Optional[int] = None
        if self.enable_memory:
            memory_bytes = self._process.memory_info().rss
        
        # Get GPU metrics at the end of the step
        gpu_metrics = {}
        if self.enable_gpu and self._gpu_profiler and self._gpu_profiler._initialized:
            try:
                end_gpu_metrics = self._gpu_profiler.get_gpu_metrics()
                
                # Calculate delta metrics
                if start_gpu_metrics and end_gpu_metrics:
                    gpu_metrics = end_gpu_metrics  # Use end metrics as base
                    
                    # Calculate memory delta
                    if 'memory_used_mb' in start_gpu_metrics and 'memory_used_mb' in end_gpu_metrics:
                        gpu_metrics['memory_delta_mb'] = (
                            end_gpu_metrics['memory_used_mb'] - start_gpu_metrics.get('memory_used_mb', 0)
                        )
                    
                    # Use the end metrics for utilization (more representative of the step's work)
                    if 'gpu_utilization' in end_gpu_metrics:
                        gpu_metrics['utilization'] = end_gpu_metrics['gpu_utilization']
                    
                    # Include temperature and other metrics from the end of the step
                    for key in ['temperature_gpu', 'power_usage_w', 'power_limit_w']:
                        if key in end_gpu_metrics:
                            gpu_metrics[key] = end_gpu_metrics[key]
                    
                    # Keep the GPU name if present
                    if 'name' in end_gpu_metrics:
                        gpu_metrics['name'] = end_gpu_metrics['name']
                        
            except Exception as e:
                warnings.warn(f"Error getting final GPU metrics for step '{name}': {e}")
        
        step = _StepRow(
            step=name,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            memory_bytes=memory_bytes,
            gpu_metrics=gpu_metrics if self.enable_gpu else {}
        )
        self._steps.append(step)
        return step.to_dict()

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
    def steps(self) -> List[Dict[str, object]]:
        """Get a list of completed steps as dictionaries."""
        result = []
        for step in self._steps:
            step_dict = step.to_dict()
            # Flatten gpu_metrics into the main dictionary with gpu_ prefix
            gpu_metrics = step_dict.pop('gpu_metrics', {})
            for key, value in gpu_metrics.items():
                step_dict[f'gpu_{key}'] = value
            result.append(step_dict)
        return result

    @property
    def total_duration(self) -> float:
        """Total duration (seconds) measured so far."""
        return sum(step["duration"] for step in self._steps)


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
