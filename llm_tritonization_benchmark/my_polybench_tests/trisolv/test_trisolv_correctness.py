#!/usr/bin/env python3
"""Correctness test for trisolv (Polybench) - attempt 1"""
import sys
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

# Import Triton implementation
try:
    from polybench_results.llm_triton.trisolv.attempt1 import trisolv_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Load C reference
C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs" / "libtrisolv.so"
if not C_LIB_PATH.exists():
    print(f"C reference library not found: {C_LIB_PATH}")
    sys.exit(1)

def run_c_reference(L_c, b_c, x_c, N):
    """Run C reference kernel via ctypes."""
    lib = ctypes.CDLL(str(C_LIB_PATH))

    # Set global arrays in the .so
    CType_L = ctypes.c_float * (120 * 120)
    c_arr_L = CType_L.in_dll(lib, 'L')
    src_L = np.ascontiguousarray(L_c, dtype=np.float32)
    ctypes.memmove(c_arr_L, src_L.ctypes.data, src_L.nbytes)
    CType_b = ctypes.c_float * (120)
    c_arr_b = CType_b.in_dll(lib, 'b')
    src_b = np.ascontiguousarray(b_c, dtype=np.float32)
    ctypes.memmove(c_arr_b, src_b.ctypes.data, src_b.nbytes)
    CType_x = ctypes.c_float * (120)
    c_arr_x = CType_x.in_dll(lib, 'x')
    src_x = np.ascontiguousarray(x_c, dtype=np.float32)
    ctypes.memmove(c_arr_x, src_x.ctypes.data, src_x.nbytes)

    # Set global scalars
    pass

    # Run kernel
    func = getattr(lib, "trisolv_kernel")
    func.argtypes = []
    func.restype = None
    func()

    # Read back output arrays
    CType_x = ctypes.c_float * (120)
    c_arr_x = CType_x.in_dll(lib, 'x')
    x_c[:] = np.frombuffer(c_arr_x, dtype=np.float32).reshape(120).copy()

def test_correctness():
    """Test Triton vs C reference."""
    num_tests = 3
    all_passed = True

    for test_idx in range(num_tests):
        try:
            # Initialize arrays
            # Lower triangular with |diagonal| >= 1
            L = torch.tril(torch.randn(120, 120, device='cuda', dtype=torch.float32))
            L.diagonal().abs_().clamp_(min=1.0)
            b = torch.randn(120, device='cuda', dtype=torch.float32)
            x = torch.zeros(120, device='cuda', dtype=torch.float32)
            N = 120

            # Clone for C reference
            L_c = L.cpu().numpy().copy()
            b_c = b.cpu().numpy().copy()
            x_c = x.cpu().numpy().copy()

            # Clone for Triton
            L_tr = L.clone()
            b_tr = b.clone()
            x_tr = x.clone()

            # Run C reference
            run_c_reference(L_c, b_c, x_c, N)

            # Run Triton
            trisolv_triton(L_tr, b_tr, x_tr, N)

            # Compare output arrays
            max_error = 0.0
            max_rel_error = 0.0
            c_val = torch.from_numpy(x_c).float()
            tr_val = x_tr.cpu().float()
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
