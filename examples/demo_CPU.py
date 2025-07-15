"""
Comprehensive demonstration of profiling_py with CPU and memory profiling.

This demo showcases various profiling scenarios including:
- CPU-bound operations
- Memory-bound operations
- Nested function calls
- Manual step profiling
"""
import time
import random
import numpy as np
from typing import List
from profiling_py import Profiler, generate_profiling_report

# Create a profiler with CPU and memory monitoring
prof = Profiler(
    enable_time=True,
    enable_memory=True,
    enable_gpu=False,
    metadata={
        "script": "comprehensive_cpu_demo.py",
        "description": "Comprehensive CPU and memory profiling demo"
    },
)

def create_large_matrix(size: int) -> np.ndarray:
    """Create and process a large matrix (CPU-bound operation)."""
    with prof.profile(f"create_matrix_{size}x{size}"):
        matrix = np.random.rand(size, size)
        # Perform some CPU-intensive operations
        result = matrix @ matrix.T  # Matrix multiplication
        result = np.linalg.eigvals(result)  # Compute eigenvalues
        return result

def process_large_data(data_size: int) -> List[float]:
    """Process large data with memory allocation (memory-bound operation)."""
    with prof.profile(f"process_data_{data_size}"):
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

def process_with_retry(max_retries: int = 3) -> None:
    """Demonstrate nested profiling contexts with error handling."""
    with prof.profile("process_with_retry"):
        for attempt in range(max_retries):
            try:
                with prof.profile(f"attempt_{attempt+1}"):
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
    print("Starting comprehensive CPU profiling demo...")
    
    # 1. CPU-bound operations
    print("\n1. Running CPU-bound operations...")
    with prof.profile("cpu_operations"):
        for size in [500, 1000, 1500]:
            create_large_matrix(size)
    
    # 2. Memory-bound operations
    print("2. Running memory-bound operations...")
    with prof.profile("memory_operations"):
        data_sizes = [100000, 500000, 1000000]
        for size in data_sizes:
            process_large_data(size)
    
    # 3. Nested operations with error handling
    print("3. Running nested operations with error handling...")
    process_with_retry()
    
    # 4. Manual step profiling
    print("4. Running manual step profiling...")
    
    # Start a step manually
    prof.start_step("manual_step_1")
    time.sleep(0.1)
    # Do some work
    _ = [i**2 for i in range(1000000)]
    prof.end_step("manual_step_1")
    
    # 5. Nested manual steps
    print("5. Running nested manual steps...")
    prof.start_step("outer_step")
    time.sleep(0.05)
    
    prof.start_step("inner_step_1")
    _ = sum(range(1000000))
    prof.end_step("inner_step_1")
    
    prof.start_step("inner_step_2")
    _ = [i**0.5 for i in range(1000000)]
    prof.end_step("inner_step_2")
    
    prof.end_step("outer_step")
    
    # Generate and save the report
    print("\nGenerating profiling report...")
    generate_profiling_report(
        prof,
        output_dir="./profiling_reports",
        file_name=f"cpu_profiling_report_{int(time.time())}",
        open_browser=False
    )
    
    print("\nProfiling complete! Report has been saved to the 'profiling_reports' directory.")

if __name__ == "__main__":
    main()
