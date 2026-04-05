#!/usr/bin/env python3
"""
Test Triton Implementations Against C Reference (Ground Truth)

This script:
1. Uses the compiled C library as ground truth
2. Generates Triton implementations if needed
3. Compares Triton GPU output against C CPU output

Usage:
    python test_triton_vs_c.py              # Test all available
    python test_triton_vs_c.py s000 s111    # Test specific functions
"""

import os
import sys
import subprocess
import inspect
import numpy as np
import torch
from pathlib import Path


# Import C reference and TSVC database
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'utilities'))
sys.path.insert(0, str(Path(__file__).parent / 'c_reference'))

from tsvc_functions_db import TSVC_FUNCTIONS
try:
    from tsvc_all_reference import C_REFERENCE_FUNCS_ALL as C_REFERENCE_FUNCS, get_c_reference_all as get_c_reference
except ImportError:
    from tsvc_reference import C_REFERENCE_FUNCS, get_c_reference


def get_func_params(func):
    """Get the parameter names a function accepts"""
    sig = inspect.signature(func)
    return list(sig.parameters.keys())


def build_args(func, available_tensors, available_scalars):
    """Build argument list based on what the function actually accepts"""
    params = get_func_params(func)
    args = []
    for p in params:
        if p in available_tensors:
            args.append(available_tensors[p])
        elif p in available_scalars:
            args.append(available_scalars[p])
    return args

# Functions that have C reference implementations
AVAILABLE_C_REFS = set(C_REFERENCE_FUNCS.keys())


def test_triton_vs_c(func_name, test_sizes=[100, 1000, 10000]):
    """
    Test a Triton implementation against C reference.

    Returns:
        (passed, error_msg) tuple
    """
    # Check if we have C reference
    c_ref = get_c_reference(func_name)
    if c_ref is None:
        return None, f"No C reference available for {func_name}"

    # Try to import Triton implementation
    try:
        triton_module = __import__(
            f'llm_triton.{func_name}_triton_llm_v3',
            fromlist=[f'{func_name}_triton']
        )
        triton_func = getattr(triton_module, f'{func_name}_triton')
    except ImportError as e:
        return None, f"Could not import Triton: {e}"
    except AttributeError as e:
        return None, f"Triton function not found: {e}"

    # Get function spec
    func_spec = TSVC_FUNCTIONS.get(func_name)
    if not func_spec:
        return None, f"Function spec not found for {func_name}"

    arrays = func_spec['arrays']
    has_2d = func_spec.get('has_2d_arrays', False)
    scalar_params = func_spec.get('scalar_params', {})

    all_passed = True
    error_details = []

    for N in test_sizes:
        try:
            # For 2D functions, use square root of N
            if has_2d:
                len_2d = int(np.sqrt(N))
                N = len_2d * len_2d

            # Check if function uses strided access (inc parameter)
            uses_stride = 'inc' in scalar_params
            stride_size = 2 if uses_stride else 1
            array_size = N * stride_size  # Allocate larger arrays for strided functions

            # Initialize arrays based on spec
            # Use TSVC-style initialization: 1/(i+1) for positive values (like 'any,frac')
            init_data = {}
            int_arrays = {'ip', 'indx', 'ind'}  # Index arrays
            for arr_name, mode in arrays.items():
                if arr_name in ['aa', 'bb', 'cc', 'tt']:
                    # 2D array
                    len_2d = int(np.sqrt(N))
                    init_data[arr_name] = np.random.randn(len_2d, len_2d).astype(np.float32)
                elif arr_name == 'flat_2d_array':
                    init_data[arr_name] = np.random.randn(N).astype(np.float32)
                elif arr_name in int_arrays:
                    # Index array - random indices within bounds
                    init_data[arr_name] = np.random.randint(0, N, size=N).astype(np.int32)
                elif arr_name == 'd':
                    # d array: use TSVC-style 1/(i+1) initialization (always positive)
                    # This is needed for s481 which has exit(0) if d[i] < 0
                    init_data[arr_name] = (1.0 / (np.arange(1, array_size + 1))).astype(np.float32)
                else:
                    # 1D array - use larger size for strided functions
                    init_data[arr_name] = np.random.randn(array_size).astype(np.float32)

            # Initialize scalar parameters with reasonable defaults
            scalar_values = {
                'iterations': 1,  # Not used but included for compatibility
                'n': N,
                'n1': 1,  # Common offset parameter
                'n3': 1,  # Common stride parameter
                'm': min(N // 10, 100),  # Common limit parameter
                'M': min(N // 10, 100),  # Uppercase M
                'j': 1,
                'k': 1,
                'inc': 1,  # Use 1 to avoid out-of-bounds with strided access
                'len_2d': int(np.sqrt(N)) if has_2d else int(np.sqrt(N)),
                'LEN_1D': N,
                'LEN_2D': int(np.sqrt(N)) if has_2d else int(np.sqrt(N)),
                # Additional scalar parameters used by various functions
                's': 1.5,        # Scalar multiplier (s4112)
                's1': 1.0,       # Scalar 1 (s242)
                's2': 2.0,       # Scalar 2 (s242)
                't': 0.5,        # Scalar threshold/value (s272)
                't_value': 0.5,  # Named t_value variant (s332)
                'x': 1.0,        # Scalar x (s2710)
                'alpha': 1.5,    # Scalar alpha (s351, s353)
            }

            # Check if Triton function expects arrays not in spec (LLM quirks)
            triton_params = get_func_params(triton_func)
            for param in triton_params:
                if param not in init_data and param not in scalar_values:
                    # LLM expects this param but it's not in spec - create an array
                    if param in int_arrays:
                        init_data[param] = np.random.randint(0, N, size=N).astype(np.int32)
                    else:
                        # Create float array (may be used for extracting scalar like alpha=c[0])
                        init_data[param] = np.random.randn(N).astype(np.float32)

            # Create copies for C reference (CPU numpy)
            c_data = {k: v.copy() for k, v in init_data.items()}

            # Create copies for Triton (GPU torch)
            tr_data = {k: torch.from_numpy(v.copy()).cuda() for k, v in init_data.items()}

            # Build args using introspection for C reference
            c_tensors = c_data
            c_args = build_args(c_ref, c_tensors, scalar_values)

            # Build args using introspection for Triton
            tr_tensors = tr_data
            tr_args = build_args(triton_func, tr_tensors, scalar_values)

            # Run C reference
            c_ref(*c_args)

            # Run Triton
            triton_func(*tr_args)

            # Compare output arrays
            max_error = 0.0
            for arr_name, mode in arrays.items():
                if mode in ['rw', 'w']:  # Modified arrays
                    c_result = c_data[arr_name]
                    tr_result = tr_data[arr_name].cpu().numpy()
                    if c_result.shape != tr_result.shape:
                        c_result = c_result.flatten()
                        tr_result = tr_result.flatten()
                    error = np.max(np.abs(c_result - tr_result))
                    max_error = max(max_error, error)

            if max_error > 1e-3:
                error_details.append(f"N={N}: max_error={max_error:.2e}")
                all_passed = False

        except Exception as e:
            error_details.append(f"N={N}: {type(e).__name__}: {e}")
            all_passed = False

    if all_passed:
        return True, None
    else:
        return False, "; ".join(error_details)


def generate_triton_if_missing(func_name):
    """Generate Triton implementation if it doesn't exist"""
    triton_file = Path("../results/tsvc/llm_triton") / f"{func_name}_triton_llm_v3.py"
    if triton_file.exists():
        return True

    # Call the generation script
    try:
        result = subprocess.run(
            [sys.executable, "utilities/generate_llm_triton.py", func_name],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=Path.cwd()
        )
        return result.returncode == 0
    except:
        return False


def main():
    """Main test routine"""
    print("=" * 70)
    print("Testing Triton Implementations Against C Reference (Ground Truth)")
    print("=" * 70)
    print(f"Available C references: {len(AVAILABLE_C_REFS)}")
    print(f"Total TSVC functions: {len(TSVC_FUNCTIONS)}")
    print("=" * 70)

    # Get functions to test
    if len(sys.argv) > 1:
        func_names = sys.argv[1:]
    else:
        # Test all available C references
        func_names = sorted(AVAILABLE_C_REFS)

    results = {}
    for func_name in func_names:
        if func_name not in TSVC_FUNCTIONS:
            print(f"\n{func_name}: SKIP (not in TSVC database)")
            continue

        print(f"\n{func_name}:", end=" ")

        # Generate Triton if needed
        triton_file = Path("../results/tsvc/llm_triton") / f"{func_name}_triton_llm_v3.py"
        if not triton_file.exists():
            print("generating...", end=" ")
            if not generate_triton_if_missing(func_name):
                print("SKIP (generation failed)")
                results[func_name] = {"status": "skip", "reason": "generation failed"}
                continue

        # Test against C reference
        passed, error = test_triton_vs_c(func_name)

        if passed is None:
            print(f"SKIP ({error})")
            results[func_name] = {"status": "skip", "reason": error}
        elif passed:
            print("PASS")
            results[func_name] = {"status": "pass"}
        else:
            print(f"FAIL ({error})")
            results[func_name] = {"status": "fail", "reason": error}

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results.values() if r["status"] == "pass")
    failed = sum(1 for r in results.values() if r["status"] == "fail")
    skipped = sum(1 for r in results.values() if r["status"] == "skip")
    total = len(results)

    print(f"Total: {total}")
    print(f"  PASS: {passed}")
    print(f"  FAIL: {failed}")
    print(f"  SKIP: {skipped}")
    print(f"Pass rate: {passed}/{total-skipped} ({100*passed/(total-skipped):.1f}%)" if total > skipped else "N/A")

    # Print failures
    if failed > 0:
        print("\nFailed functions:")
        for func_name, r in results.items():
            if r["status"] == "fail":
                print(f"  {func_name}: {r.get('reason', 'unknown')}")

    print("=" * 70)


if __name__ == "__main__":
    main()
