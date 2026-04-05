#!/usr/bin/env python3
"""
NCU profiling of representative Triton kernels to check hardware utilization.
Profiles kernels from different categories to understand if the GPU is saturated.
"""
import subprocess
import json
import re
import sys
import os

NCU = "/usr/local/cuda/bin/ncu"

# Representative kernels to profile (covering different perf tiers)
PROFILE_KERNELS = {
    # Fast kernels (>5x speedup) - should show good utilization
    "covariance": "../results/polybench/polybench_results/llm_triton/covariance/attempt1.py",
    "heat_3d": "../results/polybench/polybench_results/llm_triton/heat_3d/attempt1.py",
    "doitgen": "../results/polybench/polybench_results/llm_triton/doitgen/attempt1.py",
    # Medium kernels (1-5x) - moderate utilization expected
    "gemm": "../results/polybench/polybench_results/llm_triton/gemm/attempt1.py",
    "bicg": "../results/polybench/polybench_results/llm_triton/bicg/attempt1.py",
    "atax": "../results/polybench/polybench_results/llm_triton/atax/attempt1.py",
    # Slow kernels (<1x) - low utilization expected
    "jacobi_1d": "../results/polybench/polybench_results/llm_triton/jacobi_1d/attempt6.py",
    "nussinov": "../results/polybench/polybench_results/llm_triton/nussinov/attempt5.py",
}

# For each kernel, create a tiny wrapper that imports and runs it
WRAPPER_TEMPLATE = '''
import torch
import sys
sys.path.insert(0, "{script_dir}")

# Import the kernel module
import importlib.util
spec = importlib.util.spec_from_file_location("kernel_mod", "{code_path}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Get kernel params from polybench DB
sys.path.insert(0, "/home/qinxiao/workspace/compiler-guided-triton-gen/analysis")
from extract_polybench_kernels import POLYBENCH_KERNELS
sys.path.insert(0, "utilities")
from polybench_functions_db import POLYBENCH_FUNCTIONS

kernel_name = "{kernel_name}"
orig_name = kernel_name.replace("_", "-")
kernel_info = POLYBENCH_KERNELS.get(orig_name, {{}})
func_info = POLYBENCH_FUNCTIONS.get(kernel_name, {{}})
params = kernel_info.get("params", {{}})
arrays = func_info.get("arrays", {{}})
scalar_params = func_info.get("scalar_params", {{}})
has_2d = func_info.get("has_2d_arrays", False)
has_3d = func_info.get("has_3d_arrays", False)

# Get sizes from params
sizes = {{}}
for k, v in params.items():
    sizes[k] = v

# Determine array dimensions
size_keys = sorted(sizes.keys())
if len(size_keys) == 0:
    N = 120  # default
    sizes = {{"N": N}}

# Create arrays on GPU
torch.manual_seed(42)
tensors = {{}}
for arr_name, mode in arrays.items():
    if has_3d:
        dims = list(sizes.values())[:3]
        if len(dims) < 3:
            dims = [list(sizes.values())[0]] * 3
        t = torch.randn(*dims, device="cuda", dtype=torch.float32)
    elif has_2d:
        dims = list(sizes.values())[:2]
        if len(dims) < 2:
            dims = [list(sizes.values())[0]] * 2
        t = torch.randn(*dims, device="cuda", dtype=torch.float32)
    else:
        dim = list(sizes.values())[0]
        t = torch.randn(dim, device="cuda", dtype=torch.float32)
    tensors[arr_name] = t

# Find the triton wrapper function
func_name = None
for name in dir(mod):
    if name.endswith("_triton"):
        func_name = name
        break

if func_name is None:
    print("ERROR: no _triton function found")
    sys.exit(1)

func = getattr(mod, func_name)

# Build args (arrays + sizes + scalars)
import inspect
sig = inspect.signature(func)
args = []
for pname in sig.parameters:
    if pname in tensors:
        args.append(tensors[pname])
    elif pname in sizes:
        args.append(sizes[pname])
    elif pname in scalar_params:
        args.append(float(scalar_params[pname]))
    elif pname.upper() in sizes:
        args.append(sizes[pname.upper()])
    else:
        # Try to guess
        args.append(list(sizes.values())[0])

# Warmup
try:
    func(*args)
    torch.cuda.synchronize()
except Exception as e:
    print(f"ERROR running kernel: {{e}}")
    sys.exit(1)

# Profile run
func(*args)
torch.cuda.synchronize()
print("PROFILE_DONE")
'''


def profile_kernel(kernel_name, code_path):
    """Profile a single kernel with NCU and extract key metrics."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Verify the code file exists
    full_path = os.path.join(script_dir, code_path)
    if not os.path.exists(full_path):
        # Try to find the latest attempt
        base_dir = os.path.dirname(full_path)
        if os.path.isdir(base_dir):
            attempts = sorted([f for f in os.listdir(base_dir) if f.startswith("attempt")])
            if attempts:
                full_path = os.path.join(base_dir, attempts[-1])
                code_path = os.path.relpath(full_path, script_dir)

    if not os.path.exists(full_path):
        return {"error": f"Code not found: {full_path}"}

    # Write wrapper script
    wrapper_code = WRAPPER_TEMPLATE.format(
        script_dir=script_dir,
        code_path=full_path,
        kernel_name=kernel_name
    )
    wrapper_path = f"/tmp/ncu_wrapper_{kernel_name}.py"
    with open(wrapper_path, "w") as f:
        f.write(wrapper_code)

    # Run NCU with key metrics
    metrics = [
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",  # SM utilization %
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",  # Memory throughput %
        "sm__warps_active.avg.pct_of_peak_sustained_active",  # Occupancy %
        "gpu__time_duration.sum",  # Kernel duration (ns)
        "launch__grid_size",  # Grid size
        "launch__block_size",  # Block size
        "dram__bytes.sum",  # Total DRAM bytes
        "l2__throughput.avg.pct_of_peak_sustained_elapsed",  # L2 throughput %
    ]

    cmd = [
        NCU,
        "--metrics", ",".join(metrics),
        "--target-processes", "all",
        "--set", "full",
        "--csv",
        "python", wrapper_path
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        return {"error": "NCU timeout"}

    # Parse CSV output
    results = []
    lines = output.split("\n")
    header = None
    for line in lines:
        if line.startswith('"ID"') or line.startswith('ID'):
            header = [h.strip('"') for h in line.split('","')]
            continue
        if header and line.strip() and line.startswith('"'):
            values = [v.strip('"') for v in line.split('","')]
            if len(values) >= len(header):
                row = dict(zip(header, values))
                results.append(row)

    return {
        "kernel": kernel_name,
        "ncu_rows": len(results),
        "raw_output_lines": len(lines),
        "output_preview": output[:2000] if len(output) > 0 else "empty"
    }


def main():
    print(f"GPU: RTX 3090 (SM 8.6), 2x GPUs")
    print(f"NCU: {NCU}")
    print()

    all_results = {}
    for kernel_name, code_path in PROFILE_KERNELS.items():
        print(f"Profiling {kernel_name}...")
        result = profile_kernel(kernel_name, code_path)
        all_results[kernel_name] = result
        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Got {result['ncu_rows']} metric rows, {result['raw_output_lines']} output lines")

    # Save results
    with open("../results/polybench/polybench_results/ncu_profile_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to polybench_results/ncu_profile_results.json")


if __name__ == "__main__":
    main()
