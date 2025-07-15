# Advanced Usage

## Customizing Reports

### Report Styling
You can customize the appearance of the HTML report by modifying the CSS in the template. The report uses Bootstrap 5 for styling.

### Custom Metrics
Add custom metrics to your profiling data:

```python
from profiling_py import default_profiler

with default_profiler.profile("my_operation") as step:
    # Your code here
    step.metadata["custom_metric"] = 42
```

## GPU Profiling

### Multi-GPU Support
Profile across multiple GPUs by creating multiple profilers:

```python
from profiling_py import Profiler

# Profile on GPU 0
with Profiler(enable_gpu=True, gpu_device=0) as p0:
    # Code for GPU 0
    pass

# Profile on GPU 1
with Profiler(enable_gpu=True, gpu_device=1) as p1:
    # Code for GPU 1
    pass
```

### GPU Metrics
Available GPU metrics include:
- `gpu_utilization`: GPU core utilization (0-100%)
- `memory_used_mb`: Used GPU memory in MB
- `memory_total_mb`: Total GPU memory in MB
- `memory_utilization`: Memory utilization percentage
- `temperature_gpu`: GPU temperature in Celsius
- `power_usage_w`: Current power usage in watts
- `power_limit_w`: Power limit in watts

## Performance Tips

### Reducing Overhead
For minimal overhead:
1. Only enable the metrics you need
2. Use larger step sizes
3. Profile only critical sections

### Memory Profiling
Memory profiling adds overhead. For accurate results:
- Profile for longer durations
- Use larger step sizes
- Avoid frequent small allocations

## Integration

### Jupyter Notebooks
Use the profiler in Jupyter notebooks:

```python
from profiling_py import profile

@profile
def train_model():
    # Training code
    pass

# Run and display report
train_model()
```

### Web Applications
Profile web applications by creating a profiler middleware:

```python
from fastapi import Request
from profiling_py import Profiler

def profiling_middleware(app):
    async def middleware(request: Request, call_next):
        with Profiler() as prof:
            response = await call_next(request)
            prof.generate_report(output_dir="./profiling_reports")
        return response
    return middleware
```
