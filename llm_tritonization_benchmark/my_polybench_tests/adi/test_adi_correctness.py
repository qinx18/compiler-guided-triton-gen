#!/usr/bin/env python3
"""Correctness test for adi (Polybench) - attempt 9"""
import sys
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

# Import Triton implementation
try:
    from polybench_results.llm_triton.adi.attempt9 import adi_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Load C reference
C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs" / "libadi.so"
if not C_LIB_PATH.exists():
    print(f"C reference library not found: {C_LIB_PATH}")
    sys.exit(1)

def run_c_reference(p_c, q_c, u_c, v_c, N, TSTEPS):
    """Run C reference kernel via ctypes."""
    lib = ctypes.CDLL(str(C_LIB_PATH))

    # Set global arrays in the .so
    CType_p = ctypes.c_float * (60 * 60)
    c_arr_p = CType_p.in_dll(lib, 'p')
    src_p = np.ascontiguousarray(p_c, dtype=np.float32)
    ctypes.memmove(c_arr_p, src_p.ctypes.data, src_p.nbytes)
    CType_q = ctypes.c_float * (60 * 60)
    c_arr_q = CType_q.in_dll(lib, 'q')
    src_q = np.ascontiguousarray(q_c, dtype=np.float32)
    ctypes.memmove(c_arr_q, src_q.ctypes.data, src_q.nbytes)
    CType_u = ctypes.c_float * (60 * 60)
    c_arr_u = CType_u.in_dll(lib, 'u')
    src_u = np.ascontiguousarray(u_c, dtype=np.float32)
    ctypes.memmove(c_arr_u, src_u.ctypes.data, src_u.nbytes)
    CType_v = ctypes.c_float * (60 * 60)
    c_arr_v = CType_v.in_dll(lib, 'v')
    src_v = np.ascontiguousarray(v_c, dtype=np.float32)
    ctypes.memmove(c_arr_v, src_v.ctypes.data, src_v.nbytes)

    # Set global scalars
    pass

    # Run kernel
    func = getattr(lib, "adi_kernel")
    func.argtypes = []
    func.restype = None
    func()

    # Read back output arrays
    CType_p = ctypes.c_float * (60 * 60)
    c_arr_p = CType_p.in_dll(lib, 'p')
    p_c[:] = np.frombuffer(c_arr_p, dtype=np.float32).reshape(60, 60).copy()
    CType_q = ctypes.c_float * (60 * 60)
    c_arr_q = CType_q.in_dll(lib, 'q')
    q_c[:] = np.frombuffer(c_arr_q, dtype=np.float32).reshape(60, 60).copy()
    CType_u = ctypes.c_float * (60 * 60)
    c_arr_u = CType_u.in_dll(lib, 'u')
    u_c[:] = np.frombuffer(c_arr_u, dtype=np.float32).reshape(60, 60).copy()
    CType_v = ctypes.c_float * (60 * 60)
    c_arr_v = CType_v.in_dll(lib, 'v')
    v_c[:] = np.frombuffer(c_arr_v, dtype=np.float32).reshape(60, 60).copy()

def test_correctness():
    """Test Triton vs C reference."""
    num_tests = 3
    all_passed = True

    for test_idx in range(num_tests):
        try:
            # Initialize arrays
            p = torch.randn(60, 60, device='cuda', dtype=torch.float32)
            q = torch.randn(60, 60, device='cuda', dtype=torch.float32)
            u = torch.randn(60, 60, device='cuda', dtype=torch.float32)
            v = torch.randn(60, 60, device='cuda', dtype=torch.float32)
            N = 60
            TSTEPS = 40

            # Clone for C reference
            p_c = p.cpu().numpy().copy()
            q_c = q.cpu().numpy().copy()
            u_c = u.cpu().numpy().copy()
            v_c = v.cpu().numpy().copy()

            # Clone for Triton
            p_tr = p.clone()
            q_tr = q.clone()
            u_tr = u.clone()
            v_tr = v.clone()

            # Run C reference
            run_c_reference(p_c, q_c, u_c, v_c, N, TSTEPS)

            # Run Triton
            adi_triton(p_tr, q_tr, u_tr, v_tr, N, TSTEPS)

            # Compare output arrays
            max_error = 0.0
            max_rel_error = 0.0
            c_val = torch.from_numpy(p_c).float()
            tr_val = p_tr.cpu().float()
            abs_err = torch.max(torch.abs(c_val - tr_val)).item()
            denom = torch.max(torch.abs(c_val)).item()
            rel_err = abs_err / max(denom, 1e-10)
            max_error = max(max_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)
            c_val = torch.from_numpy(q_c).float()
            tr_val = q_tr.cpu().float()
            abs_err = torch.max(torch.abs(c_val - tr_val)).item()
            denom = torch.max(torch.abs(c_val)).item()
            rel_err = abs_err / max(denom, 1e-10)
            max_error = max(max_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)
            c_val = torch.from_numpy(u_c).float()
            tr_val = u_tr.cpu().float()
            abs_err = torch.max(torch.abs(c_val - tr_val)).item()
            denom = torch.max(torch.abs(c_val)).item()
            rel_err = abs_err / max(denom, 1e-10)
            max_error = max(max_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)
            c_val = torch.from_numpy(v_c).float()
            tr_val = v_tr.cpu().float()
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
