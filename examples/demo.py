"""Demo script for profiling_py.

Run with:
    python examples/demo.py
A browser tab will open showing the generated interactive profiling report.
"""
from time import sleep

from profiling_py import Profiler, generate_profiling_report

# Create a dedicated profiler with memory tracking enabled
prof = Profiler(enable_memory=True, enable_time=True, metadata={
    "script": "demo.py",
    "description": "Demonstration of profiling_py (memory on)",
})

# Profile with the context-manager style
with prof.profile("initialise"):
    sleep(0.3)

# Simulate CPU intensive work
with prof.profile("compute"):
    _ = sum(i * i for i in range(1_000_00))

# Memory allocation example
with prof.profile("allocate"):
    data = [0] * 5_000_00  # noqa: F841
    sleep(0.1)

# Decorator style (function-level profiling)
@prof.profile("function_process")
def do_work():
    temp = [i ** 0.5 for i in range(100_000)]  # noqa: F841
    sleep(0.2)


do_work()

# Generate interactive HTML report and open in browser
print("Generating report …")
report_path = generate_profiling_report(prof, open_browser=False)
print("Report saved ✔")
