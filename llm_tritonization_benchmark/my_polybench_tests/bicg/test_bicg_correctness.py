#!/usr/bin/env python3
"""Correctness test for bicg (Polybench) - attempt 1"""
import sys
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

# Import Triton implementation
try:
    from polybench_results.llm_triton.bicg.attempt1 import bicg_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Load C reference
C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs" / "libbicg.so"
if not C_LIB_PATH.exists():
    print(f"C reference library not found: {C_LIB_PATH}")
    sys.exit(1)

def run_c_reference(A_c, p_c, q_c, r_c, s_c, M, N):
    """Run C reference kernel via ctypes."""
    lib = ctypes.CDLL(str(C_LIB_PATH))

    # Set global arrays in the .so
    CType_A = ctypes.c_float * (85 * 75)
    c_arr_A = CType_A.in_dll(lib, 'A')
    src_A = np.ascontiguousarray(A_c, dtype=np.float32)
    ctypes.memmove(c_arr_A, src_A.ctypes.data, src_A.nbytes)
    CType_p = ctypes.c_float * (75)
    c_arr_p = CType_p.in_dll(lib, 'p')
    src_p = np.ascontiguousarray(p_c, dtype=np.float32)
    ctypes.memmove(c_arr_p, src_p.ctypes.data, src_p.nbytes)
    CType_q = ctypes.c_float * (85)
    c_arr_q = CType_q.in_dll(lib, 'q')
    src_q = np.ascontiguousarray(q_c, dtype=np.float32)
    ctypes.memmove(c_arr_q, src_q.ctypes.data, src_q.nbytes)
    CType_r = ctypes.c_float * (85)
    c_arr_r = CType_r.in_dll(lib, 'r')
    src_r = np.ascontiguousarray(r_c, dtype=np.float32)
    ctypes.memmove(c_arr_r, src_r.ctypes.data, src_r.nbytes)
    CType_s = ctypes.c_float * (75)
    c_arr_s = CType_s.in_dll(lib, 's')
    src_s = np.ascontiguousarray(s_c, dtype=np.float32)
    ctypes.memmove(c_arr_s, src_s.ctypes.data, src_s.nbytes)

    # Set global scalars
    pass

    # Run kernel
    func = getattr(lib, "bicg_kernel")
    func.argtypes = []
    func.restype = None
    func()

    # Read back output arrays
    CType_q = ctypes.c_float * (85)
    c_arr_q = CType_q.in_dll(lib, 'q')
    q_c[:] = np.frombuffer(c_arr_q, dtype=np.float32).reshape(85).copy()
    CType_s = ctypes.c_float * (75)
    c_arr_s = CType_s.in_dll(lib, 's')
    s_c[:] = np.frombuffer(c_arr_s, dtype=np.float32).reshape(75).copy()

def test_correctness():
    """Test Triton vs C reference."""
    num_tests = 3
    all_passed = True

    for test_idx in range(num_tests):
        try:
            # Initialize arrays
            A = torch.randn(85, 75, device='cuda', dtype=torch.float32)
            p = torch.randn(75, device='cuda', dtype=torch.float32)
            q = torch.randn(85, device='cuda', dtype=torch.float32)
            r = torch.randn(85, device='cuda', dtype=torch.float32)
            s = torch.randn(75, device='cuda', dtype=torch.float32)
            M = 75
            N = 85

            # Clone for C reference
            A_c = A.cpu().numpy().copy()
            p_c = p.cpu().numpy().copy()
            q_c = q.cpu().numpy().copy()
            r_c = r.cpu().numpy().copy()
            s_c = s.cpu().numpy().copy()

            # Clone for Triton
            A_tr = A.clone()
            p_tr = p.clone()
            q_tr = q.clone()
            r_tr = r.clone()
            s_tr = s.clone()

            # Run C reference
            run_c_reference(A_c, p_c, q_c, r_c, s_c, M, N)

            # Run Triton
            bicg_triton(A_tr, p_tr, q_tr, r_tr, s_tr, M, N)

            # Compare output arrays
            max_error = 0.0
            max_rel_error = 0.0
            c_val = torch.from_numpy(q_c).float()
            tr_val = q_tr.cpu().float()
            abs_err = torch.max(torch.abs(c_val - tr_val)).item()
            denom = torch.max(torch.abs(c_val)).item()
            rel_err = abs_err / max(denom, 1e-10)
            max_error = max(max_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)
            c_val = torch.from_numpy(s_c).float()
            tr_val = s_tr.cpu().float()
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
