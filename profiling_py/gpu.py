"""GPU monitoring utilities for profiling_py.

This module provides functionality to measure GPU metrics like memory usage and utilization
using the NVIDIA Management Library (NVML) via the pynvml package.
"""
from typing import Dict, List, Optional, Tuple, Union
import warnings

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    warnings.warn(
        "nvidia-ml-py3 not installed. GPU monitoring will be disabled. "
        "Install with: pip install 'profiling_py[gpu]'"
    )

class GPUProfiler:
    """Profiler for GPU metrics using NVML."""
    
    def __init__(self, device_index: int = 0):
        """Initialize the GPU profiler.
        
        Args:
            device_index: Index of the GPU device to monitor (default: 0)
        """
        self.device_index = device_index
        self.handle = None
        self._initialized = False
        
    def __enter__(self):
        self.initialize()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        
    def initialize(self) -> None:
        """Initialize the NVML library and get a handle to the GPU device."""
        if not NVML_AVAILABLE:
            return
            
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self._initialized = True
        except pynvml.NVMLError as e:
            warnings.warn(f"Failed to initialize NVML: {e}")
            self._initialized = False
    
    def shutdown(self) -> None:
        """Shutdown the NVML library."""
        if NVML_AVAILABLE and hasattr(pynvml, 'nvmlShutdown'):
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass
        self._initialized = False
    
    def get_gpu_metrics(self) -> Dict[str, Union[float, int, str, None]]:
        """Get current GPU metrics.
        
        Returns:
            Dictionary containing GPU metrics:
            - gpu_utilization: GPU utilization percentage (0-100)
            - memory_used_mb: Used GPU memory in MB
            - memory_total_mb: Total GPU memory in MB
            - memory_utilization: Memory utilization percentage (0-100)
            - temperature_gpu: GPU temperature in Celsius
            - power_usage_w: Power usage in watts
            - power_limit_w: Power limit in watts
            - name: GPU name
        """
        if not self._initialized or not NVML_AVAILABLE:
            return {}
            
        try:
            # Get GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            gpu_util = util.gpu
            
            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            memory_used_mb = mem_info.used / (1024 * 1024)  # Convert to MB
            memory_total_mb = mem_info.total / (1024 * 1024)  # Convert to MB
            memory_util = (mem_info.used / mem_info.total) * 100
            
            # Get temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    self.handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except pynvml.NVMLError:
                temp = None
                
            # Get power usage and limit
            try:
                power_usage_w = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # Convert mW to W
                power_limit_w = pynvml.nvmlDeviceGetEnforcedPowerLimit(self.handle) / 1000.0  # Convert mW to W
            except pynvml.NVMLError:
                power_usage_w = None
                power_limit_w = None
                
            # Get GPU name
            try:
                name = pynvml.nvmlDeviceGetName(self.handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
            except pynvml.NVMLError:
                name = "Unknown GPU"
            
            return {
                "gpu_utilization": gpu_util,
                "memory_used_mb": memory_used_mb,
                "memory_total_mb": memory_total_mb,
                "memory_utilization": memory_util,
                "temperature_gpu": temp,
                "power_usage_w": power_usage_w,
                "power_limit_w": power_limit_w,
                "name": name,
            }
            
        except pynvml.NVMLError as e:
            warnings.warn(f"Error getting GPU metrics: {e}")
            return {}

def get_available_gpus() -> List[Dict[str, Union[int, str]]]:
    """Get a list of available NVIDIA GPUs.
    
    Returns:
        List of dictionaries with GPU index and name for each available GPU.
    """
    if not NVML_AVAILABLE:
        return []
        
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                gpus.append({"index": i, "name": name})
            except pynvml.NVMLError:
                gpus.append({"index": i, "name": f"GPU {i}"})
        pynvml.nvmlShutdown()
        return gpus
    except pynvml.NVMLError as e:
        warnings.warn(f"Error getting GPU list: {e}")
        return []

# Create a default GPU profiler instance
default_gpu_profiler: Optional[GPUProfiler] = None

def get_default_gpu_profiler() -> Optional[GPUProfiler]:
    """Get or create a default GPU profiler instance.
    
    Returns:
        GPUProfiler instance if GPUs are available, None otherwise.
    """
    global default_gpu_profiler
    
    if not NVML_AVAILABLE:
        return None
        
    if default_gpu_profiler is None:
        try:
            default_gpu_profiler = GPUProfiler()
            default_gpu_profiler.initialize()
        except Exception as e:
            warnings.warn(f"Failed to initialize default GPU profiler: {e}")
            return None
            
    return default_gpu_profiler

def get_gpu_metrics(device_index: int = 0) -> Dict[str, Union[float, int, str, None]]:
    """Get GPU metrics for the specified device.
    
    Args:
        device_index: Index of the GPU device to query (default: 0)
        
    Returns:
        Dictionary containing GPU metrics, or empty dict if not available.
    """
    if not NVML_AVAILABLE:
        return {}
        
    with GPUProfiler(device_index) as gpu:
        if gpu._initialized:
            return gpu.get_gpu_metrics()
    return {}

# Clean up on module exit
import atexit
atexit.register(lambda: default_gpu_profiler.shutdown() if default_gpu_profiler else None)
