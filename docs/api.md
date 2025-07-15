# API Reference

## Core Classes

### `Profiler`

The main profiling class that collects and manages profiling data.

```python
from profiling_py import Profiler

profiler = Profiler(
    enable_time=True,     # Enable time profiling (default: True)
    enable_memory=False,  # Enable memory profiling (default: False)
    enable_gpu=False,     # Enable GPU profiling (default: False)
    gpu_device=0,         # GPU device index (default: 0)
    metadata=None         # Optional metadata dictionary
)
```

#### Methods

- `start_step(name: str) -> None`: Start a new profiling step
- `end_step(name: str) -> Dict[str, Any]`: End the current profiling step
- `profile(name: str) -> ContextManager`: Context manager for profiling a code block
- `get_stats() -> Dict[str, Any]`: Get current profiling statistics

## Decorators

### `@profile`

Decorator to profile a function.

```python
from profiling_py import profile

@profile
def my_function():
    # Your code here
    pass
```

## Context Managers

### `profile(name: str)`

Context manager for profiling a code block.

```python
from profiling_py import profile

with profile("my_operation"):
    # Your code here
    pass
```

## Report Generation

### `generate_profiling_report()`

Generate an HTML profiling report.

```python
from profiling_py import generate_profiling_report

generate_profiling_report(
    profiler=None,           # Profiler instance (default: default_profiler)
    output_dir="./reports",  # Output directory
    file_name=None,          # Report filename (default: auto-generated)
    open_browser=True,       # Open report in browser
    measure_time=True,       # Include time measurements
    measure_ram=True,        # Include memory measurements
    measure_gpu=True         # Include GPU measurements
)
```

## GPU Profiling

### `GPUProfiler`

Low-level GPU profiler using NVML.

```python
from profiling_py import GPUProfiler

with GPUProfiler(device_index=0) as gpu:
    metrics = gpu.get_gpu_metrics()
    # metrics contains GPU utilization, memory usage, etc.
```
