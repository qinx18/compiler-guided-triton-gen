#!/usr/bin/env python3
"""Correctness test for gaussian (Real-World) - attempt 1"""
import sys
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

# Import Triton implementation
try:
    from realworld_results.llm_triton.gaussian.attempt1 import gaussian_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Load C reference
C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "realworld_libs" / "libgaussian.so"
if not C_LIB_PATH.exists():
    print(f"C reference library not found: {C_LIB_PATH}")
    sys.exit(1)

def run_c_reference(a_c, b_c, m_c, x_c, N):
    """Run C reference kernel via ctypes."""
    lib = ctypes.CDLL(str(C_LIB_PATH))

    # Set global arrays in the .so
    CType_a = ctypes.c_float * (262144)
    c_arr_a = CType_a.in_dll(lib, 'a')
    src_a = np.ascontiguousarray(a_c, dtype=np.float32)
    ctypes.memmove(c_arr_a, src_a.ctypes.data, src_a.nbytes)
    CType_b = ctypes.c_float * (512)
    c_arr_b = CType_b.in_dll(lib, 'b')
    src_b = np.ascontiguousarray(b_c, dtype=np.float32)
    ctypes.memmove(c_arr_b, src_b.ctypes.data, src_b.nbytes)
    CType_m = ctypes.c_float * (262144)
    c_arr_m = CType_m.in_dll(lib, 'm')
    src_m = np.ascontiguousarray(m_c, dtype=np.float32)
    ctypes.memmove(c_arr_m, src_m.ctypes.data, src_m.nbytes)

    # Set global scalars
    pass

    # Run kernel
    func = getattr(lib, "gaussian_kernel")
    func.argtypes = []
    func.restype = None
    func()

    # Read back output arrays
    CType_a = ctypes.c_float * (262144)
    c_arr_a = CType_a.in_dll(lib, 'a')
    a_c[:] = np.frombuffer(c_arr_a, dtype=np.float32).reshape(262144).copy()
    CType_b = ctypes.c_float * (512)
    c_arr_b = CType_b.in_dll(lib, 'b')
    b_c[:] = np.frombuffer(c_arr_b, dtype=np.float32).reshape(512).copy()
    CType_m = ctypes.c_float * (262144)
    c_arr_m = CType_m.in_dll(lib, 'm')
    m_c[:] = np.frombuffer(c_arr_m, dtype=np.float32).reshape(262144).copy()
    CType_x = ctypes.c_float * (512)
    c_arr_x = CType_x.in_dll(lib, 'x')
    x_c[:] = np.frombuffer(c_arr_x, dtype=np.float32).reshape(512).copy()

def test_correctness():
    """Test Triton vs C reference."""
    num_tests = 3
    all_passed = True

    for test_idx in range(num_tests):
        try:
            # Initialize arrays
            # Diagonally dominant matrix for stability (no pivoting)
            a = torch.randn(262144, device='cuda', dtype=torch.float32)
            # Make diag dominant
            for i_idx in range(512):
                a[i_idx * 512 + i_idx] = abs(a[i_idx * 512 + i_idx].item()) + 512
            m = torch.zeros(262144, device='cuda', dtype=torch.float32)
            b = torch.randn(512, device='cuda', dtype=torch.float32)
            x = torch.zeros(512, device='cuda', dtype=torch.float32)
            N = 512

            # Clone for C reference
            a_c = a.cpu().numpy().copy()
            b_c = b.cpu().numpy().copy()
            m_c = m.cpu().numpy().copy()
            x_c = x.cpu().numpy().copy()

            # Clone for Triton
            a_tr = a.clone()
            b_tr = b.clone()
            m_tr = m.clone()
            x_tr = x.clone()

            # Run C reference
            run_c_reference(a_c, b_c, m_c, x_c, N)

            # Run Triton
            gaussian_triton(a_tr, b_tr, m_tr, x_tr, N)

            # Compare output arrays
            max_error = 0.0
            max_rel_error = 0.0
            c_val = torch.from_numpy(a_c).float()
            tr_val = a_tr.cpu().float()
            abs_err = torch.max(torch.abs(c_val - tr_val)).item()
            denom = torch.max(torch.abs(c_val)).item()
            rel_err = abs_err / max(denom, 1e-10)
            max_error = max(max_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)
            c_val = torch.from_numpy(b_c).float()
            tr_val = b_tr.cpu().float()
            abs_err = torch.max(torch.abs(c_val - tr_val)).item()
            denom = torch.max(torch.abs(c_val)).item()
            rel_err = abs_err / max(denom, 1e-10)
            max_error = max(max_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)
            c_val = torch.from_numpy(m_c).float()
            tr_val = m_tr.cpu().float()
            abs_err = torch.max(torch.abs(c_val - tr_val)).item()
            denom = torch.max(torch.abs(c_val)).item()
            rel_err = abs_err / max(denom, 1e-10)
            max_error = max(max_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)
            c_val = torch.from_numpy(x_c).float()
            tr_val = x_tr.cpu().float()
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
