#!/usr/bin/env python3
"""
Profile representative Triton kernels with NCU to measure hardware utilization.
Uses the ACTUAL generated Triton code from the latest run.
"""
import subprocess
import json
import csv
import io
import os
import sys
import re

NCU = "/usr/local/cuda-12.4/bin/ncu"
NCU_ENV = {"TMPDIR": "/home/qinxiao/tmp", "PATH": os.environ.get("PATH", "")}

# Metrics to collect
METRICS = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "gpu__time_duration.sum",
    "launch__grid_size",
    "launch__block_size",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
]

# Kernels to profile grouped by performance tier
KERNELS_TO_PROFILE = [
    # High speedup: should show decent utilization
    "covariance", "heat_3d", "doitgen", "fdtd_2d",
    # Medium speedup
    "gemm", "bicg", "atax", "correlation",
    # Low speedup: expected low utilization
    "jacobi_1d", "nussinov", "ludcmp",
]


def find_latest_attempt(kernel_name):
    """Find the latest passing attempt for a kernel."""
    with open("../results/polybench/polybench_results/results.json") as f:
        results = json.load(f)
    kdata = results.get(kernel_name, {})
    if not kdata.get("test_passed"):
        return None
    attempt = kdata.get("final_attempt", kdata.get("attempts", 1))
    path = f"../results/polybench/polybench_results/llm_triton/{kernel_name}/attempt{attempt}.py"
    if os.path.exists(path):
        return path
    # Try other attempts
    for i in range(attempt, 0, -1):
        p = f"../results/polybench/polybench_results/llm_triton/{kernel_name}/attempt{i}.py"
        if os.path.exists(p):
            return p
    return None


def write_profile_script(kernel_name, code_path):
    """Write a standalone script that runs the kernel for NCU profiling."""
    script = f'''
import torch
import sys
import os
sys.path.insert(0, "{os.getcwd()}")
sys.path.insert(0, "{os.getcwd()}/utilities")
sys.path.insert(0, "/home/qinxiao/workspace/compiler-guided-triton-gen/analysis")

import importlib.util
spec = importlib.util.spec_from_file_location("kmod", "{os.path.abspath(code_path)}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

from extract_polybench_kernels import POLYBENCH_KERNELS
from polybench_functions_db import POLYBENCH_FUNCTIONS

kernel_name = "{kernel_name}"
orig = kernel_name.replace("_", "-")
params = POLYBENCH_KERNELS.get(orig, {{}}).get("params", {{}})
func_info = POLYBENCH_FUNCTIONS.get(kernel_name, {{}})
arrays = func_info.get("arrays", {{}})
scalar_params = func_info.get("scalar_params", {{}})
has_2d = func_info.get("has_2d_arrays", False)
has_3d = func_info.get("has_3d_arrays", False)

# Import c_reference for init
sys.path.insert(0, "c_reference")
from polybench_reference import create_torch_tensors
tensors, size_args, scalar_args = create_torch_tensors(kernel_name, "cuda")

# Find the triton function
func = None
for name in dir(mod):
    if name.endswith("_triton"):
        func = getattr(mod, name)
        break

if func is None:
    print("ERROR: no _triton function found")
    sys.exit(1)

# Build args
import inspect
sig = inspect.signature(func)
args = []
for pname in sig.parameters:
    if pname in tensors:
        args.append(tensors[pname])
    elif pname in size_args:
        args.append(size_args[pname])
    elif pname in scalar_args:
        args.append(scalar_args[pname])
    elif pname.upper() in size_args:
        args.append(size_args[pname.upper()])
    else:
        for v in size_args.values():
            args.append(v)
            break

# Warmup (2 runs)
for _ in range(2):
    try:
        func(*args)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"ERROR: {{e}}")
        sys.exit(1)

# Profile run (single invocation)
func(*args)
torch.cuda.synchronize()
'''
    script_path = f"/tmp/ncu_profile_{kernel_name}.py"
    with open(script_path, "w") as f:
        f.write(script)
    return script_path


def run_ncu(kernel_name, script_path):
    """Run NCU and parse results."""
    cmd = [
        NCU,
        "--target-processes", "all",
        "--csv",
        "--metrics", ",".join(METRICS),
        "--kernel-name", "regex:(?!distribution)(?!elementwise).*",  # Skip PyTorch internal kernels
        "python", script_path
    ]

    env = {**os.environ, "TMPDIR": "/home/qinxiao/tmp"}
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
        output = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}

    # Parse CSV
    kernels = {}
    lines = output.strip().split("\n")
    header = None
    for line in lines:
        if not line.strip():
            continue
        if '"ID"' in line:
            header = line
            continue
        if header and line.startswith('"'):
            try:
                reader = csv.reader(io.StringIO(line))
                values = next(reader)
                if len(values) >= 15:
                    kid = values[0]
                    kname = values[4]
                    metric_name = values[12]
                    metric_unit = values[13]
                    metric_value = values[14]
                    if kname not in kernels:
                        kernels[kname] = {"grid": values[8], "block": values[7]}
                    kernels[kname][metric_name] = f"{metric_value} {metric_unit}"
            except:
                pass

    return {"kernels": kernels, "n_kernels": len(kernels)}


def main():
    os.makedirs("/home/qinxiao/tmp", exist_ok=True)

    print("=" * 100)
    print("NCU HARDWARE UTILIZATION PROFILING")
    print("GPU: RTX 3090 (SM 8.6, 82 SMs, 936 GB/s DRAM BW, 35.58 TFLOPS FP32)")
    print("=" * 100)
    print()

    all_results = {}

    for kernel_name in KERNELS_TO_PROFILE:
        code_path = find_latest_attempt(kernel_name)
        if not code_path:
            print(f"SKIP {kernel_name}: no passing attempt found")
            continue

        print(f"\nProfiling: {kernel_name} ({code_path})")
        script_path = write_profile_script(kernel_name, code_path)
        result = run_ncu(kernel_name, script_path)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue

        all_results[kernel_name] = result

        # Print results
        for kname, metrics in result.get("kernels", {}).items():
            # Shorten kernel name for display
            short_name = kname[:60] + "..." if len(kname) > 60 else kname
            sm_pct = metrics.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", "?")
            mem_pct = metrics.get("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", "?")
            occ_pct = metrics.get("sm__warps_active.avg.pct_of_peak_sustained_active", "?")
            duration = metrics.get("gpu__time_duration.sum", "?")
            grid = metrics.get("grid", "?")
            block = metrics.get("block", "?")
            dram_r = metrics.get("dram__bytes_read.sum", "?")
            dram_w = metrics.get("dram__bytes_write.sum", "?")

            print(f"  Kernel: {short_name}")
            print(f"    Grid={grid}, Block={block}")
            print(f"    SM util: {sm_pct}, Mem throughput: {mem_pct}, Occupancy: {occ_pct}")
            print(f"    Duration: {duration}, DRAM R: {dram_r}, DRAM W: {dram_w}")

    # Save results
    with open("../results/polybench/polybench_results/ncu_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(f"{'Kernel':<15} {'Speedup':>8} {'SM%':>8} {'Mem%':>8} {'Occ%':>8} {'Grid':>12} {'Block':>8}")
    print("-" * 75)

    with open("../results/polybench/polybench_results/results.json") as f:
        bench = json.load(f)

    for kernel_name in KERNELS_TO_PROFILE:
        if kernel_name not in all_results:
            continue
        speedup = bench.get(kernel_name, {}).get("benchmark", {}).get("speedup", 0)
        kernels = all_results[kernel_name].get("kernels", {})
        # Get the main Triton kernel (skip PyTorch internals)
        main_kernel = None
        for kname, metrics in kernels.items():
            if "distribution" not in kname and "elementwise" not in kname:
                main_kernel = (kname, metrics)
                break
        if not main_kernel:
            # Take the last kernel
            if kernels:
                main_kernel = list(kernels.items())[-1]
        if main_kernel:
            kname, m = main_kernel
            sm = m.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", "?").split()[0]
            mem = m.get("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", "?").split()[0]
            occ = m.get("sm__warps_active.avg.pct_of_peak_sustained_active", "?").split()[0]
            grid = m.get("grid", "?")
            block = m.get("block", "?")
            print(f"{kernel_name:<15} {speedup:>8.2f} {sm:>8} {mem:>8} {occ:>8} {grid:>12} {block:>8}")


if __name__ == "__main__":
    main()
