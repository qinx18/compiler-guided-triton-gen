#!/usr/bin/env python3
"""Correctness test for lu (Polybench) - attempt 2"""
import sys
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

# Import Triton implementation
try:
    from polybench_results_scale8x.llm_triton_no_analysis.lu.attempt2 import lu_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Load C reference
C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs_scale8x" / "liblu.so"
if not C_LIB_PATH.exists():
    print(f"C reference library not found: {C_LIB_PATH}")
    sys.exit(1)

def run_c_reference(A_c, N):
    """Run C reference kernel via ctypes."""
    lib = ctypes.CDLL(str(C_LIB_PATH))

    # Set global arrays in the .so
    CType_A = ctypes.c_float * (960 * 960)
    c_arr_A = CType_A.in_dll(lib, 'A')
    src_A = np.ascontiguousarray(A_c, dtype=np.float32)
    ctypes.memmove(c_arr_A, src_A.ctypes.data, src_A.nbytes)

    # Set global scalars
    pass

    # Run kernel
    func = getattr(lib, "lu_kernel")
    func.argtypes = []
    func.restype = None
    func()

    # Read back output arrays
    CType_A = ctypes.c_float * (960 * 960)
    c_arr_A = CType_A.in_dll(lib, 'A')
    A_c[:] = np.frombuffer(c_arr_A, dtype=np.float32).reshape(960, 960).copy()

def test_correctness():
    """Test Triton vs C reference."""
    num_tests = 3
    all_passed = True

    for test_idx in range(num_tests):
        try:
            # Initialize arrays
            # Diagonally dominant for stable pivotless LU
            A = torch.randn(960, 960, device='cuda', dtype=torch.float32) + 960 * torch.eye(960, device='cuda', dtype=torch.float32)
            N = 960

            # Clone for C reference
            A_c = A.cpu().numpy().copy()

            # Clone for Triton
            A_tr = A.clone()

            # Run C reference
            run_c_reference(A_c, N)

            # Run Triton
            lu_triton(A_tr, N)

            # Compare output arrays
            max_error = 0.0
            max_rel_error = 0.0
            c_val = torch.from_numpy(A_c).float()
            tr_val = A_tr.cpu().float()
            abs_err = torch.max(torch.abs(c_val - tr_val)).item()
            denom = torch.max(torch.abs(c_val)).item()
            rel_err = abs_err / max(denom, 1e-10)
            max_error = max(max_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)

            # Pass if absolute error < atol OR relative error < rtol
            passed = (max_error < 5.0) or (max_rel_error < 0.05)
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
