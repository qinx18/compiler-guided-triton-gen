#!/usr/bin/env python3
"""Correctness test for gemver (Polybench) - attempt 1"""
import sys
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

# Import Triton implementation
try:
    from polybench_results.llm_triton.gemver.attempt1 import gemver_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Load C reference
C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs" / "libgemver.so"
if not C_LIB_PATH.exists():
    print(f"C reference library not found: {C_LIB_PATH}")
    sys.exit(1)

def run_c_reference(A_c, u1_c, u2_c, v1_c, v2_c, w_c, x_c, y_c, z_c, alpha, beta, N):
    """Run C reference kernel via ctypes."""
    lib = ctypes.CDLL(str(C_LIB_PATH))

    # Set global arrays in the .so
    CType_A = ctypes.c_float * (120 * 120)
    c_arr_A = CType_A.in_dll(lib, 'A')
    src_A = np.ascontiguousarray(A_c, dtype=np.float32)
    ctypes.memmove(c_arr_A, src_A.ctypes.data, src_A.nbytes)
    CType_u1 = ctypes.c_float * (120)
    c_arr_u1 = CType_u1.in_dll(lib, 'u1')
    src_u1 = np.ascontiguousarray(u1_c, dtype=np.float32)
    ctypes.memmove(c_arr_u1, src_u1.ctypes.data, src_u1.nbytes)
    CType_u2 = ctypes.c_float * (120)
    c_arr_u2 = CType_u2.in_dll(lib, 'u2')
    src_u2 = np.ascontiguousarray(u2_c, dtype=np.float32)
    ctypes.memmove(c_arr_u2, src_u2.ctypes.data, src_u2.nbytes)
    CType_v1 = ctypes.c_float * (120)
    c_arr_v1 = CType_v1.in_dll(lib, 'v1')
    src_v1 = np.ascontiguousarray(v1_c, dtype=np.float32)
    ctypes.memmove(c_arr_v1, src_v1.ctypes.data, src_v1.nbytes)
    CType_v2 = ctypes.c_float * (120)
    c_arr_v2 = CType_v2.in_dll(lib, 'v2')
    src_v2 = np.ascontiguousarray(v2_c, dtype=np.float32)
    ctypes.memmove(c_arr_v2, src_v2.ctypes.data, src_v2.nbytes)
    CType_w = ctypes.c_float * (120)
    c_arr_w = CType_w.in_dll(lib, 'w')
    src_w = np.ascontiguousarray(w_c, dtype=np.float32)
    ctypes.memmove(c_arr_w, src_w.ctypes.data, src_w.nbytes)
    CType_x = ctypes.c_float * (120)
    c_arr_x = CType_x.in_dll(lib, 'x')
    src_x = np.ascontiguousarray(x_c, dtype=np.float32)
    ctypes.memmove(c_arr_x, src_x.ctypes.data, src_x.nbytes)
    CType_y = ctypes.c_float * (120)
    c_arr_y = CType_y.in_dll(lib, 'y')
    src_y = np.ascontiguousarray(y_c, dtype=np.float32)
    ctypes.memmove(c_arr_y, src_y.ctypes.data, src_y.nbytes)
    CType_z = ctypes.c_float * (120)
    c_arr_z = CType_z.in_dll(lib, 'z')
    src_z = np.ascontiguousarray(z_c, dtype=np.float32)
    ctypes.memmove(c_arr_z, src_z.ctypes.data, src_z.nbytes)

    # Set global scalars
    ctypes.c_float.in_dll(lib, 'alpha').value = float(alpha)
    ctypes.c_float.in_dll(lib, 'beta').value = float(beta)

    # Run kernel
    func = getattr(lib, "gemver_kernel")
    func.argtypes = []
    func.restype = None
    func()

    # Read back output arrays
    CType_A = ctypes.c_float * (120 * 120)
    c_arr_A = CType_A.in_dll(lib, 'A')
    A_c[:] = np.frombuffer(c_arr_A, dtype=np.float32).reshape(120, 120).copy()
    CType_w = ctypes.c_float * (120)
    c_arr_w = CType_w.in_dll(lib, 'w')
    w_c[:] = np.frombuffer(c_arr_w, dtype=np.float32).reshape(120).copy()
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
            A = torch.randn(120, 120, device='cuda', dtype=torch.float32)
            u1 = torch.randn(120, device='cuda', dtype=torch.float32)
            u2 = torch.randn(120, device='cuda', dtype=torch.float32)
            v1 = torch.randn(120, device='cuda', dtype=torch.float32)
            v2 = torch.randn(120, device='cuda', dtype=torch.float32)
            w = torch.randn(120, device='cuda', dtype=torch.float32)
            x = torch.randn(120, device='cuda', dtype=torch.float32)
            y = torch.randn(120, device='cuda', dtype=torch.float32)
            z = torch.randn(120, device='cuda', dtype=torch.float32)
            alpha = 1.5
            beta = 1.5
            N = 120

            # Clone for C reference
            A_c = A.cpu().numpy().copy()
            u1_c = u1.cpu().numpy().copy()
            u2_c = u2.cpu().numpy().copy()
            v1_c = v1.cpu().numpy().copy()
            v2_c = v2.cpu().numpy().copy()
            w_c = w.cpu().numpy().copy()
            x_c = x.cpu().numpy().copy()
            y_c = y.cpu().numpy().copy()
            z_c = z.cpu().numpy().copy()

            # Clone for Triton
            A_tr = A.clone()
            u1_tr = u1.clone()
            u2_tr = u2.clone()
            v1_tr = v1.clone()
            v2_tr = v2.clone()
            w_tr = w.clone()
            x_tr = x.clone()
            y_tr = y.clone()
            z_tr = z.clone()

            # Run C reference
            run_c_reference(A_c, u1_c, u2_c, v1_c, v2_c, w_c, x_c, y_c, z_c, alpha, beta, N)

            # Run Triton
            gemver_triton(A_tr, u1_tr, u2_tr, v1_tr, v2_tr, w_tr, x_tr, y_tr, z_tr, alpha, beta, N)

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
            c_val = torch.from_numpy(w_c).float()
            tr_val = w_tr.cpu().float()
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
