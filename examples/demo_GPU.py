#!/usr/bin/env python3
"""
Comprehensive demo of profiling_py capabilities.

This script demonstrates various profiling scenarios:
1. CPU-bound operations
2. Memory-intensive operations
3. GPU-accelerated operations (if available)
4. Nested profiling contexts
5. Manual step profiling
"""
import time
import numpy as np
from typing import List, Dict, Any

# Check for GPU availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from profiling_py import Profiler, start_step, end_step, profile, generate_profiling_report

# Check if GPU is available
has_gpu = TORCH_AVAILABLE and torch.cuda.is_available()
device = 'cuda' if has_gpu else 'cpu'

def create_large_matrix(size: int) -> np.ndarray:
    """Create and process a large matrix (CPU-bound operation)."""
    matrix = np.random.rand(size, size)
    # Perform some CPU-intensive operations
    result = matrix @ matrix.T  # Matrix multiplication
    result = np.linalg.eigvals(result)  # Compute eigenvalues
    return result

def process_large_data(data_size: int) -> List[float]:
    """Process large data with memory allocation (memory-bound operation)."""
    # Create a large list
    data = [float(i) for i in range(data_size)]
    
    # Perform some memory operations
    result = []
    for i in range(0, len(data), 1000):
        chunk = data[i:i+1000]
        result.extend([x * 2 for x in chunk])
        
    # Simulate some processing time
    time.sleep(0.1)
    
    return result

def gpu_matrix_operations(size: int = 1000) -> None:
    """Perform GPU-accelerated matrix operations."""
    if not has_gpu:
        return
        
    # Create large matrices on GPU
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Perform matrix multiplication
    c = torch.mm(a, b)
    
    # Perform element-wise operations
    d = torch.sigmoid(c) * 2.0
    
    # Ensure all operations are completed
    torch.cuda.synchronize()

def process_with_retry(max_retries: int = 3) -> None:
    """Demonstrate nested profiling contexts."""
    for attempt in range(max_retries):
        try:
            with profile(f"attempt_{attempt+1}"):
                # Simulate operation that might fail
                if attempt < max_retries - 1:
                    time.sleep(0.1)
                    raise ValueError("Temporary failure")
                time.sleep(0.2)
                print("Operation succeeded!")
                break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(0.05)

def main():
    """Run the demo with various profiling scenarios."""
    # Create a profiler with all metrics enabled
    prof = Profiler(
        enable_time=True,
        enable_memory=True,
        enable_gpu=has_gpu,
        metadata={
            "script": "comprehensive_demo.py",
            "description": "Comprehensive demonstration of profiling_py features",
            "gpu_available": str(has_gpu),
            "device": device
        }
    )
    
    # Register the profiler as the default
    import profiling_py
    profiling_py.default_profiler = prof
    
    print("Starting comprehensive profiling demo...")
    print(f"GPU available: {has_gpu}")
    
    # 1. CPU-bound operations
    print("\n1. Running CPU-bound operations...")
    with prof.profile("CPU-bound operations"):
        # This will create nested profiling steps
        for size in [500, 1000, 1500]:
            create_large_matrix(size)
    
    # 2. Memory-bound operations
    print("2. Running memory-bound operations...")
    with prof.profile("memory-bound operations"):
        data_sizes = [100000, 500000, 1000000]
        for size in data_sizes:
            process_large_data(size)
    
    # 3. GPU operations (if available)
    if has_gpu:
        print("3. Running GPU-accelerated operations...")
        with prof.profile("GPU operations"):
            for _ in range(3):
                gpu_matrix_operations(size = 10000)
    else:
        print("3. Skipping GPU operations (no GPU available)")
    
    # 4. Nested operations with error handling
    print("4. Running nested operations with error handling...")
    with prof.profile("nested operations with error handling"):
        process_with_retry()
    
    # 5. Manual step profiling
    print("5. Running manual step profiling...")
    
    # Start a step using the module-level function
    prof.start_step("manual step")
    time.sleep(0.1)
    if has_gpu and TORCH_AVAILABLE:
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.mm(x, y)
        torch.cuda.synchronize()
    prof.end_step("manual step")
    
    # Generate and open the report
    print("\nGenerating profiling report...")
    generate_profiling_report(
        prof,
        output_dir="./profiling_reports",
        file_name=f"gpu_profiling_report_{int(time.time())}",
        open_browser=False
    )
    
    print(f"\nProfiling report generated. {'GPU metrics included.' if has_gpu else 'No GPU detected.'}")
    if not has_gpu:
        print("To enable GPU monitoring, install PyTorch and ensure you have an NVIDIA GPU with drivers.")
        print("Install with: pip install 'profiling_py[gpu]' torch")

if __name__ == "__main__":
    main()
