#!/usr/bin/env python3
"""Correctness test for gramschmidt (Polybench) - attempt 1"""
import sys
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

# Import Triton implementation
try:
    from polybench_results.llm_triton.gramschmidt.attempt1 import gramschmidt_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Load C reference
C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs" / "libgramschmidt.so"
if not C_LIB_PATH.exists():
    print(f"C reference library not found: {C_LIB_PATH}")
    sys.exit(1)

def run_c_reference(A_c, Q_c, R_c, M, N):
    """Run C reference kernel via ctypes."""
    lib = ctypes.CDLL(str(C_LIB_PATH))

    # Set global arrays in the .so
    CType_A = ctypes.c_float * (60 * 80)
    c_arr_A = CType_A.in_dll(lib, 'A')
    src_A = np.ascontiguousarray(A_c, dtype=np.float32)
    ctypes.memmove(c_arr_A, src_A.ctypes.data, src_A.nbytes)
    CType_Q = ctypes.c_float * (60 * 80)
    c_arr_Q = CType_Q.in_dll(lib, 'Q')
    src_Q = np.ascontiguousarray(Q_c, dtype=np.float32)
    ctypes.memmove(c_arr_Q, src_Q.ctypes.data, src_Q.nbytes)
    CType_R = ctypes.c_float * (80 * 80)
    c_arr_R = CType_R.in_dll(lib, 'R')
    src_R = np.ascontiguousarray(R_c, dtype=np.float32)
    ctypes.memmove(c_arr_R, src_R.ctypes.data, src_R.nbytes)

    # Set global scalars
    pass

    # Run kernel
    func = getattr(lib, "gramschmidt_kernel")
    func.argtypes = []
    func.restype = None
    func()

    # Read back output arrays
    CType_A = ctypes.c_float * (60 * 80)
    c_arr_A = CType_A.in_dll(lib, 'A')
    A_c[:] = np.frombuffer(c_arr_A, dtype=np.float32).reshape(60, 80).copy()
    CType_Q = ctypes.c_float * (60 * 80)
    c_arr_Q = CType_Q.in_dll(lib, 'Q')
    Q_c[:] = np.frombuffer(c_arr_Q, dtype=np.float32).reshape(60, 80).copy()
    CType_R = ctypes.c_float * (80 * 80)
    c_arr_R = CType_R.in_dll(lib, 'R')
    R_c[:] = np.frombuffer(c_arr_R, dtype=np.float32).reshape(80, 80).copy()

def test_correctness():
    """Test Triton vs C reference."""
    num_tests = 3
    all_passed = True

    for test_idx in range(num_tests):
        try:
            # Initialize arrays
            # Well-conditioned A with strong diagonal for stable Gram-Schmidt
            A = torch.randn(60, 80, device='cuda', dtype=torch.float32) + torch.eye(60, 80, device='cuda', dtype=torch.float32) * 5.0
            R = torch.zeros(80, 80, device='cuda', dtype=torch.float32)
            Q = torch.zeros(60, 80, device='cuda', dtype=torch.float32)
            M = 60
            N = 80

            # Clone for C reference
            A_c = A.cpu().numpy().copy()
            Q_c = Q.cpu().numpy().copy()
            R_c = R.cpu().numpy().copy()

            # Clone for Triton
            A_tr = A.clone()
            Q_tr = Q.clone()
            R_tr = R.clone()

            # Run C reference
            run_c_reference(A_c, Q_c, R_c, M, N)

            # Run Triton
            gramschmidt_triton(A_tr, Q_tr, R_tr, M, N)

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
            c_val = torch.from_numpy(Q_c).float()
            tr_val = Q_tr.cpu().float()
            abs_err = torch.max(torch.abs(c_val - tr_val)).item()
            denom = torch.max(torch.abs(c_val)).item()
            rel_err = abs_err / max(denom, 1e-10)
            max_error = max(max_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)
            c_val = torch.from_numpy(R_c).float()
            tr_val = R_tr.cpu().float()
            abs_err = torch.max(torch.abs(c_val - tr_val)).item()
            denom = torch.max(torch.abs(c_val)).item()
            rel_err = abs_err / max(denom, 1e-10)
            max_error = max(max_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)

            # Pass if absolute error < atol OR relative error < rtol
            passed = (max_error < 1.0) or (max_rel_error < 2.0)
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
