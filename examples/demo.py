"""Demo script for profiling_py with GPU support.

Run with:
    python examples/demo.py
A browser tab will open showing the generated interactive profiling report.

Note: GPU monitoring requires nvidia-ml-py3 package and an NVIDIA GPU with drivers.
Install with: pip install 'profiling_py[gpu]'
"""
import numpy as np
from time import sleep

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. GPU examples will be skipped.")

from profiling_py import Profiler, generate_profiling_report

# Check for GPU availability
has_gpu = False
if TORCH_AVAILABLE:
    has_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if has_gpu else "cpu")
    print(f"PyTorch using device: {device}")

# Create a profiler with GPU monitoring enabled if available
prof = Profiler(
    enable_time=True,
    enable_memory=True,
    enable_gpu=has_gpu,  # Enable GPU monitoring if available
    gpu_device=0,        # Use first GPU
    metadata={
        "script": "demo.py",
        "description": "Demonstration of profiling_py with GPU support" if has_gpu \
                      else "Demonstration of profiling_py (CPU only - no GPU detected)",
    },
)

def simulate_gpu_work():
    """Simulate GPU workload using PyTorch if available."""
    if not TORCH_AVAILABLE or not has_gpu:
        return
    
    # Create some tensors on GPU
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    
    # Perform matrix multiplication (GPU intensive)
    z = torch.mm(x, y)
    
    # Some more operations
    z = z * 2
    z = torch.relu(z)
    
    # Ensure computation is done
    torch.cuda.synchronize()

# Profile with the context-manager style
with prof.profile("initialise"):
    sleep(0.3)

# Simulate CPU intensive work
with prof.profile("cpu_compute"):
    _ = sum(i * i for i in range(1_000_000))

# Memory allocation example
with prof.profile("memory_allocation"):
    data = [0] * 5_000_00  # noqa: F841
    sleep(0.1)

# GPU workload if available
if has_gpu:
    with prof.profile("gpu_compute"):
        simulate_gpu_work()
    
    # Mixed CPU/GPU workload
    with prof.profile("mixed_workload"):
        # CPU work
        result = sum(i * i for i in range(500_000))
        # GPU work
        if TORCH_AVAILABLE:
            x = torch.randn(500, 500, device=device)
            y = torch.randn(500, 500, device=device)
            z = torch.mm(x, y)
            torch.cuda.synchronize()
        # More CPU work
        result += sum(i for i in range(100_000))

# Decorator style (function-level profiling)
@prof.profile("function_processing")
def process_data():
    """A function that processes data with both CPU and GPU."""
    # CPU work
    data = [i ** 0.5 for i in range(100_000)]
    
    # GPU work if available
    if has_gpu and TORCH_AVAILABLE:
        x = torch.randn(300, 300, device=device)
        y = torch.randn(300, 300, device=device)
        z = torch.mm(x, y)
        torch.cuda.synchronize()
    
    return data

# Run the decorated function
process_data()

if __name__ == "__main__":
    # Generate and open the report
    generate_profiling_report(
        prof,
        open_browser=False,
        measure_time=True,
        measure_ram=True,
        measure_gpu=has_gpu  # Only include GPU metrics if GPU is available
    )
    
    if not has_gpu:
        print("\nNote: No GPU detected or GPU monitoring not available.")
        print("To enable GPU monitoring, install PyTorch and ensure you have an NVIDIA GPU with drivers.")
        print("Install with: pip install 'profiling_py[gpu]' torch")
