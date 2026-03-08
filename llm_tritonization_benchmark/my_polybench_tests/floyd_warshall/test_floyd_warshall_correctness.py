#!/usr/bin/env python3
"""Correctness test for floyd_warshall (Polybench) - attempt 1"""
import sys
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

# Import Triton implementation
try:
    from polybench_results.llm_triton.floyd_warshall.attempt1 import floyd_warshall_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Load C reference
C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs" / "libfloyd_warshall.so"
if not C_LIB_PATH.exists():
    print(f"C reference library not found: {C_LIB_PATH}")
    sys.exit(1)

def run_c_reference(path_c, N):
    """Run C reference kernel via ctypes."""
    lib = ctypes.CDLL(str(C_LIB_PATH))

    # Set global arrays in the .so
    CType_path = ctypes.c_float * (120 * 120)
    c_arr_path = CType_path.in_dll(lib, 'path')
    src_path = np.ascontiguousarray(path_c, dtype=np.float32)
    ctypes.memmove(c_arr_path, src_path.ctypes.data, src_path.nbytes)

    # Set global scalars
    pass

    # Run kernel
    func = getattr(lib, "floyd_warshall_kernel")
    func.argtypes = []
    func.restype = None
    func()

    # Read back output arrays
    CType_path = ctypes.c_float * (120 * 120)
    c_arr_path = CType_path.in_dll(lib, 'path')
    path_c[:] = np.frombuffer(c_arr_path, dtype=np.float32).reshape(120, 120).copy()

def test_correctness():
    """Test Triton vs C reference."""
    num_tests = 3
    all_passed = True

    for test_idx in range(num_tests):
        try:
            # Initialize arrays
            # Non-negative edge weights for shortest-path
            path = torch.abs(torch.randn(120, 120, device='cuda', dtype=torch.float32)) * 10.0 + 1.0
            N = 120

            # Clone for C reference
            path_c = path.cpu().numpy().copy()

            # Clone for Triton
            path_tr = path.clone()

            # Run C reference
            run_c_reference(path_c, N)

            # Run Triton
            floyd_warshall_triton(path_tr, N)

            # Compare output arrays
            max_error = 0.0
            max_rel_error = 0.0
            c_val = torch.from_numpy(path_c).float()
            tr_val = path_tr.cpu().float()
            abs_err = torch.max(torch.abs(c_val - tr_val)).item()
            denom = torch.max(torch.abs(c_val)).item()
            rel_err = abs_err / max(denom, 1e-10)
            max_error = max(max_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)

            # Pass if absolute error < atol OR relative error < rtol
            passed = (max_error < 0.001) or (max_rel_error < 0.0001)
            if passed:
                print(f"  Test {test_idx + 1}: PASS (abs={max_error:.6e} rel={max_rel_error:.6e})")
            else:
                print(f"  Test {test_idx + 1}: FAIL (abs={max_error:.6e} rel={max_rel_error:.6e})")
                all_passed = False

        except Exception as e:
            print(f"  Test {test_idx + 1}: ERROR - {e}")
            all_passed = False

    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    return all_passed

if __name__ == "__main__":
    test_correctness()
