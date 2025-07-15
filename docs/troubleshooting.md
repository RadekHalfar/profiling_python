# Troubleshooting

## Common Issues

### No Profiling Data Available
**Symptom**: The error "No profiling data available" appears when generating a report.

**Solution**:
1. Ensure you're using the profiler within a `with` block or using the `@profile` decorator
2. Check that you're not stopping the profiler too early
3. Verify that your code is actually being executed

### GPU Not Detected
**Symptom**: GPU metrics are not showing up in the report.

**Solution**:
1. Verify that you have an NVIDIA GPU with appropriate drivers installed
2. Install the GPU version: `pip install 'profiling-py[gpu]'`
3. Check that `nvidia-smi` shows your GPU
4. Ensure you have CUDA installed

### High Overhead
**Symptom**: The profiler is significantly slowing down your application.

**Solution**:
1. Only enable necessary metrics (`enable_memory` and `enable_gpu` add overhead)
2. Increase the size of your profiling steps
3. Profile only critical sections of your code

## Error Messages

### "NVML Library Not Found"
**Cause**: The NVIDIA Management Library (NVML) is not installed or not in your library path.

**Solution**:
1. Install the NVIDIA driver that includes NVML
2. On Linux, install the `nvidia-utils` package
3. Ensure the library is in your `LD_LIBRARY_PATH`

### "Step Already Running"
**Cause**: You're trying to start a step with the same name as an already running step.

**Solution**:
1. Use unique names for concurrent steps
2. Make sure to end steps properly with `end_step()`
3. Use context managers to ensure proper step cleanup

## Performance Tips

### Reducing Memory Usage
1. Disable memory profiling if not needed
2. Increase the time between memory samples
3. Use larger step sizes

### Improving Report Generation
1. Generate reports less frequently
2. Save reports to disk instead of opening in browser
3. Disable interactive plots if not needed

## Getting Help

If you encounter any issues not covered here:
1. Check the [GitHub Issues](https://github.com/yourusername/profiling-py/issues)
2. Create a new issue with:
   - A minimal reproducible example
   - The full error message
   - Your system information (OS, Python version, GPU model, etc.)
   - The version of profiling-py you're using
