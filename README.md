# profiling_py

A lightweight, step-based profiler for Python with support for time, memory, and GPU metrics.

## Features

- **Time Profiling**: Measure execution time of code blocks
- **Memory Profiling**: Track memory usage and detect leaks
- **GPU Monitoring**: Monitor GPU utilization, memory usage, and temperature (NVIDIA GPUs only)
- **Interactive Reports**: Generate beautiful HTML reports with plots and metrics
- **Simple API**: Easy-to-use decorators and context managers
- **Nested Profiling**: Profile nested function calls with proper hierarchy

## Installation

```bash
# Basic installation
pip install profiling-py

# With GPU support (requires CUDA)
pip install 'profiling-py[gpu]'
```

## Quick Start

```python
from profiling_py import profile, generate_profiling_report

# Profile a function
@profile
def my_function():
    # Your code here
    pass

# Or use as a context manager
def another_function():
    with profile("my_operation"):
        # Your code here
        pass

# Generate a report
generate_profiling_report()
```

## Documentation

- [API Reference](docs/api.md)
- [Examples](examples/)
- [Advanced Usage](docs/advanced.md)
- [Troubleshooting](docs/troubleshooting.md)

## License

MIT
