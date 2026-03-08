#!/usr/bin/env python3
"""
Re-benchmark existing Triton kernels at larger problem sizes.

NCU profiling showed <10% SM utilization at Polybench SMALL sizes (N=60-120).
This script re-runs the existing (already generated) Triton code at 8-16x larger
problem sizes to measure GPU performance at meaningful utilization levels.

No C reference is used (the .so files are compiled for small sizes).
Instead, we measure Triton kernel time directly and re-profile with NCU.

Usage:
    python benchmark_large_sizes.py [kernel1 kernel2 ...]
    python benchmark_large_sizes.py --all
"""
import sys
import os
import json
import time
import importlib.util
import inspect
import re
from pathlib import Path

import signal
import torch


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Kernel benchmark timed out")

# ── Configuration ──────────────────────────────────────────────────────

# Scale factor: multiply all problem dimensions by this factor
# SMALL_DATASET sizes are ~60-120 → scale 8x gives ~480-960
# This should bring SM utilization to 20-80% range on RTX 3090
SCALE_FACTOR = 8

# Reduced scale for sequential kernels (grid=1) — they iterate O(N^2) or O(N^3)
# so 8x size → 64-512x slower.  Use 2x to keep them under timeout.
SEQUENTIAL_SCALE = 2

# Kernels that are inherently sequential (grid=1) and won't benefit from
# larger sizes — include them but expect similar or worse speedups
SEQUENTIAL_KERNELS = {"nussinov", "seidel_2d", "lu", "ludcmp", "cholesky",
                      "floyd_warshall", "adi", "durbin", "trisolv"}

# Per-kernel timeout in seconds (warmup + benchmark combined)
KERNEL_TIMEOUT = 60

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "polybench_results"
KERNELS_DIR = Path("/home/qinxiao/workspace/pet/isl_analysis/kernels_polybench")

# Import kernel database
sys.path.insert(0, str(Path("/home/qinxiao/workspace/pet/isl_analysis")))
from extract_polybench_kernels import POLYBENCH_KERNELS

sys.path.insert(0, str(BASE_DIR / "utilities"))
from polybench_functions_db import POLYBENCH_FUNCTIONS


# ── Size Scaling ───────────────────────────────────────────────────────

def get_scaled_params(kernel_name: str, scale: int) -> dict:
    """Return scaled params for a kernel. Caps TSTEPS at original value."""
    for orig_name, info in POLYBENCH_KERNELS.items():
        if orig_name.replace("-", "_") == kernel_name:
            params = dict(info["params"])
            scaled = {}
            for k, v in params.items():
                if k in ("TSTEPS", "TMAX"):
                    # Don't scale timesteps — they're loop iterations, not data size
                    scaled[k] = v
                else:
                    scaled[k] = v * scale
            return scaled
    return {}


def get_original_params(kernel_name: str) -> dict:
    for orig_name, info in POLYBENCH_KERNELS.items():
        if orig_name.replace("-", "_") == kernel_name:
            return dict(info["params"])
    return {}


def get_array_shapes(kernel_name: str, params: dict) -> dict:
    """Parse array declarations from C source and resolve with given params."""
    kernel_file = KERNELS_DIR / f"{kernel_name}.c"
    if not kernel_file.exists():
        return {}

    source = kernel_file.read_text()
    shapes = {}

    # Find all array declarations: float/int arr[DIM1][DIM2]...;
    for m in re.finditer(r'(?:float|double|int)\s+(\w+)\s*(\[[^\]]+\](?:\[[^\]]+\])*)\s*;', source):
        arr_name = m.group(1)
        dims_str = m.group(2)
        dims = re.findall(r'\[(\w+)\]', dims_str)

        shape = []
        for d in dims:
            if d in params:
                shape.append(params[d])
            elif d.isdigit():
                shape.append(int(d))
            else:
                define_match = re.search(rf'#define\s+{d}\s+(\d+)', source)
                if define_match:
                    shape.append(int(define_match.group(1)) * SCALE_FACTOR)
                else:
                    shape.append(100 * SCALE_FACTOR)
        shapes[arr_name] = shape

    return shapes


def create_tensors(kernel_name: str, params: dict, arrays_info: dict) -> dict:
    """Create GPU tensors at scaled sizes with domain-appropriate initialization."""
    shapes = get_array_shapes(kernel_name, params)
    tensors = {}

    for arr_name, mode in arrays_info.items():
        shape = shapes.get(arr_name)
        if shape is None:
            # Fallback: 1D with first param value
            first_val = list(params.values())[0]
            shape = [first_val]

        if kernel_name in ("lu", "ludcmp") and arr_name in ("A",):
            # Diagonally dominant for numerical stability
            n = shape[0]
            tensors[arr_name] = torch.randn(*shape, device='cuda', dtype=torch.float32) + n * torch.eye(n, device='cuda', dtype=torch.float32)
        elif kernel_name == "cholesky" and arr_name == "A":
            n = shape[0]
            R = torch.randn(n, n, device='cuda', dtype=torch.float32)
            tensors[arr_name] = R.T @ R + n * torch.eye(n, device='cuda', dtype=torch.float32)
        elif kernel_name == "nussinov" and arr_name == "seq":
            tensors[arr_name] = torch.randint(0, 4, tuple(shape), device='cuda', dtype=torch.float32)
        elif kernel_name == "floyd_warshall" and arr_name == "path":
            tensors[arr_name] = torch.abs(torch.randn(*shape, device='cuda', dtype=torch.float32)) * 10.0 + 1.0
        elif mode in ('w', 'temp'):
            tensors[arr_name] = torch.zeros(*shape, device='cuda', dtype=torch.float32)
        else:
            tensors[arr_name] = torch.randn(*shape, device='cuda', dtype=torch.float32)

    return tensors


# ── Benchmark Runner ───────────────────────────────────────────────────

def load_triton_func(kernel_name: str, use_analysis: bool = True):
    """Load the Triton function from the generated code.

    Args:
        kernel_name: Name of the kernel.
        use_analysis: If True, load from llm_triton/ (WA). If False, load
                      from llm_triton_no_analysis/ (NA).
    """
    suffix = "" if use_analysis else "_no_analysis"
    results_file = RESULTS_DIR / f"results{suffix}.json"

    with open(results_file) as f:
        results = json.load(f)

    kdata = results.get(kernel_name, {})
    attempt = kdata.get("final_attempt", kdata.get("attempts", 1))

    code_dir = RESULTS_DIR / f"llm_triton{suffix}"
    code_path = code_dir / kernel_name / f"attempt{attempt}.py"
    if not code_path.exists():
        return None, None

    spec = importlib.util.spec_from_file_location("kmod", str(code_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Find the *_triton function
    func_id = kernel_name if not kernel_name[0].isdigit() else "k" + kernel_name
    func_name = f"{func_id}_triton"
    func = getattr(mod, func_name, None)
    return func, code_path


def build_args(func, tensors: dict, params: dict, scalar_params: dict):
    """Build argument list matching the function signature."""
    sig = inspect.signature(func)
    args = []
    for pname in sig.parameters:
        if pname in tensors:
            args.append(tensors[pname])
        elif pname in scalar_params:
            val = scalar_params[pname]
            if pname in ('alpha', 'beta'):
                args.append(1.5 if pname == 'alpha' else 1.2)
            elif pname == 'float_n':
                args.append(float(params.get('N', params.get('M', 100))))
            elif pname == 'eps':
                args.append(0.1)
            else:
                args.append(1.0)
        elif pname in params:
            args.append(params[pname])
        elif pname.upper() in params:
            args.append(params[pname.upper()])
        else:
            # Try case-insensitive match
            for k, v in params.items():
                if k.lower() == pname.lower():
                    args.append(v)
                    break
            else:
                args.append(list(params.values())[0])
    return args


def benchmark_kernel(kernel_name: str, scale: int, num_warmup=5, num_iterations=20):
    """Benchmark a kernel at scaled problem size."""
    func, code_path = load_triton_func(kernel_name)
    if func is None:
        return None

    # Get function spec
    func_spec = POLYBENCH_FUNCTIONS.get(kernel_name, {})
    arrays = func_spec.get('arrays', {})
    scalar_params = func_spec.get('scalar_params', {})

    # Get scaled params
    params = get_scaled_params(kernel_name, scale)
    if not params:
        return None

    # Create tensors
    tensors = create_tensors(kernel_name, params, arrays)

    # Build args
    args = build_args(func, tensors, params, scalar_params)

    # Set timeout
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(KERNEL_TIMEOUT)

    # Warmup
    try:
        for _ in range(num_warmup):
            func(*args)
        torch.cuda.synchronize()
    except TimeoutError:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return {"error": f"timeout ({KERNEL_TIMEOUT}s)", "phase": "warmup"}
    except Exception as e:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return {"error": str(e), "phase": "warmup"}

    # Timed runs
    try:
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            func(*args)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / num_iterations
        triton_ms = elapsed * 1000
    except TimeoutError:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return {"error": f"timeout ({KERNEL_TIMEOUT}s)", "phase": "benchmark"}
    except Exception as e:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return {"error": str(e), "phase": "benchmark"}

    signal.alarm(0)
    signal.signal(signal.SIGALRM, old_handler)

    return {
        "triton_time_ms": triton_ms,
        "params": params,
        "scale": scale,
    }


# ── Main ───────────────────────────────────────────────────────────────

def main():
    # Parse args
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        with open(RESULTS_DIR / "results.json") as f:
            results = json.load(f)
        kernels = sorted(k for k, v in results.items() if v.get("test_passed"))
    elif len(sys.argv) > 1:
        kernels = sys.argv[1:]
    else:
        # Default: top kernels by speedup (most interesting for scaling)
        with open(RESULTS_DIR / "results.json") as f:
            results = json.load(f)
        passed = [(k, v.get("benchmark", {}).get("speedup", 0))
                  for k, v in results.items() if v.get("test_passed")]
        passed.sort(key=lambda x: -x[1])
        kernels = [k for k, _ in passed]

    # Load small-size results for comparison
    with open(RESULTS_DIR / "results.json") as f:
        small_results = json.load(f)

    print("=" * 110)
    print(f"LARGE-SIZE BENCHMARK (parallel={SCALE_FACTOR}x, sequential={SEQUENTIAL_SCALE}x)")
    print(f"GPU: RTX 3090 | Comparing Triton kernel time: small vs scaled sizes")
    print(f"Timeout: {KERNEL_TIMEOUT}s per kernel")
    print("=" * 110)
    print()

    header = f"{'Kernel':<15} {'Scale':>5} {'Small_t':>9} {'Large_t':>9} {'Sm_Spd':>7} {'SmSize':>12} {'LgSize':>12} {'Status':>8}"
    print(header)
    print("-" * len(header))

    all_results = {}

    for kernel_name in kernels:
        small = small_results.get(kernel_name, {})
        small_spd = small.get("benchmark", {}).get("speedup", 0)
        small_t = small.get("benchmark", {}).get("triton_time_ms", 0)

        # Use reduced scale for sequential kernels to avoid timeouts
        scale = SEQUENTIAL_SCALE if kernel_name in SEQUENTIAL_KERNELS else SCALE_FACTOR

        orig_params = get_original_params(kernel_name)
        scaled_params = get_scaled_params(kernel_name, scale)

        # Size summary (first non-TSTEPS param)
        sm_size = ""
        lg_size = ""
        for k in orig_params:
            if k not in ("TSTEPS", "TMAX"):
                sm_size = f"{k}={orig_params[k]}"
                lg_size = f"{k}={scaled_params.get(k, '?')}"
                break

        result = benchmark_kernel(kernel_name, scale)

        if result is None:
            print(f"{kernel_name:<15} {scale:>4}x {small_t:>8.3f}ms {'N/A':>9} {small_spd:>6.1f}x {sm_size:>12} {lg_size:>12} {'SKIP':>8}")
            continue

        if "error" in result:
            err = result["error"][:40]
            print(f"{kernel_name:<15} {scale:>4}x {small_t:>8.3f}ms {'ERR':>9} {small_spd:>6.1f}x {sm_size:>12} {lg_size:>12} {err}")
            all_results[kernel_name] = result
            continue

        large_t = result["triton_time_ms"]
        status = "OK"

        print(f"{kernel_name:<15} {scale:>4}x {small_t:>8.3f}ms {large_t:>8.3f}ms {small_spd:>6.1f}x {sm_size:>12} {lg_size:>12} {status:>8}")

        all_results[kernel_name] = {
            "small_triton_ms": small_t,
            "large_triton_ms": large_t,
            "small_speedup": small_spd,
            "scale_factor": scale,
            "small_params": orig_params,
            "large_params": scaled_params,
        }

    print()

    # Summary
    ok_results = {k: v for k, v in all_results.items() if "error" not in v and "large_triton_ms" in v}
    if ok_results:
        print("=" * 100)
        print("SUMMARY")
        print("=" * 100)
        print(f"Kernels benchmarked: {len(ok_results)}/{len(kernels)}")

        small_times = [v["small_triton_ms"] for v in ok_results.values() if v["small_triton_ms"] > 0]
        large_times = [v["large_triton_ms"] for v in ok_results.values()]

        if small_times:
            print(f"Small sizes: median Triton time = {sorted(small_times)[len(small_times)//2]:.3f} ms")
        print(f"Large sizes: median Triton time = {sorted(large_times)[len(large_times)//2]:.3f} ms")
        print()
        print(f"Use these larger sizes for NCU profiling to see real GPU utilization.")

    # Save results (merge with existing)
    output_file = RESULTS_DIR / "results_large_sizes.json"
    existing = {}
    if output_file.exists():
        with open(output_file) as f:
            existing = json.load(f)
    existing.update(all_results)
    with open(output_file, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
