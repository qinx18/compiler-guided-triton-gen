#!/usr/bin/env python3
"""Correctness test for gesummv (Polybench) - attempt 2"""
import sys
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

# Import Triton implementation
try:
    from polybench_results_scale8x.llm_triton_no_analysis.gesummv.attempt2 import gesummv_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Load C reference
C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs_scale8x" / "libgesummv.so"
if not C_LIB_PATH.exists():
    print(f"C reference library not found: {C_LIB_PATH}")
    sys.exit(1)

def run_c_reference(A_c, B_c, tmp_c, x_c, y_c, alpha, beta, N):
    """Run C reference kernel via ctypes."""
    lib = ctypes.CDLL(str(C_LIB_PATH))

    # Set global arrays in the .so
    CType_A = ctypes.c_float * (720 * 720)
    c_arr_A = CType_A.in_dll(lib, 'A')
    src_A = np.ascontiguousarray(A_c, dtype=np.float32)
    ctypes.memmove(c_arr_A, src_A.ctypes.data, src_A.nbytes)
    CType_B = ctypes.c_float * (720 * 720)
    c_arr_B = CType_B.in_dll(lib, 'B')
    src_B = np.ascontiguousarray(B_c, dtype=np.float32)
    ctypes.memmove(c_arr_B, src_B.ctypes.data, src_B.nbytes)
    CType_tmp = ctypes.c_float * (720)
    c_arr_tmp = CType_tmp.in_dll(lib, 'tmp')
    src_tmp = np.ascontiguousarray(tmp_c, dtype=np.float32)
    ctypes.memmove(c_arr_tmp, src_tmp.ctypes.data, src_tmp.nbytes)
    CType_x = ctypes.c_float * (720)
    c_arr_x = CType_x.in_dll(lib, 'x')
    src_x = np.ascontiguousarray(x_c, dtype=np.float32)
    ctypes.memmove(c_arr_x, src_x.ctypes.data, src_x.nbytes)
    CType_y = ctypes.c_float * (720)
    c_arr_y = CType_y.in_dll(lib, 'y')
    src_y = np.ascontiguousarray(y_c, dtype=np.float32)
    ctypes.memmove(c_arr_y, src_y.ctypes.data, src_y.nbytes)

    # Set global scalars
    ctypes.c_float.in_dll(lib, 'alpha').value = float(alpha)
    ctypes.c_float.in_dll(lib, 'beta').value = float(beta)

    # Run kernel
    func = getattr(lib, "gesummv_kernel")
    func.argtypes = []
    func.restype = None
    func()

    # Read back output arrays
    CType_tmp = ctypes.c_float * (720)
    c_arr_tmp = CType_tmp.in_dll(lib, 'tmp')
    tmp_c[:] = np.frombuffer(c_arr_tmp, dtype=np.float32).reshape(720).copy()
    CType_y = ctypes.c_float * (720)
    c_arr_y = CType_y.in_dll(lib, 'y')
    y_c[:] = np.frombuffer(c_arr_y, dtype=np.float32).reshape(720).copy()

def test_correctness():
    """Test Triton vs C reference."""
    num_tests = 3
    all_passed = True

    for test_idx in range(num_tests):
        try:
            # Initialize arrays
            A = torch.randn(720, 720, device='cuda', dtype=torch.float32)
            B = torch.randn(720, 720, device='cuda', dtype=torch.float32)
            tmp = torch.randn(720, device='cuda', dtype=torch.float32)
            x = torch.randn(720, device='cuda', dtype=torch.float32)
            y = torch.randn(720, device='cuda', dtype=torch.float32)
            alpha = 1.5
            beta = 1.5
            N = 720

            # Clone for C reference
            A_c = A.cpu().numpy().copy()
            B_c = B.cpu().numpy().copy()
            tmp_c = tmp.cpu().numpy().copy()
            x_c = x.cpu().numpy().copy()
            y_c = y.cpu().numpy().copy()

            # Clone for Triton
            A_tr = A.clone()
            B_tr = B.clone()
            tmp_tr = tmp.clone()
            x_tr = x.clone()
            y_tr = y.clone()

            # Run C reference
            run_c_reference(A_c, B_c, tmp_c, x_c, y_c, alpha, beta, N)

            # Run Triton
            gesummv_triton(A_tr, B_tr, tmp_tr, x_tr, y_tr, alpha, beta, N)

            # Compare output arrays
            max_error = 0.0
            max_rel_error = 0.0
            c_val = torch.from_numpy(tmp_c).float()
            tr_val = tmp_tr.cpu().float()
            abs_err = torch.max(torch.abs(c_val - tr_val)).item()
            denom = torch.max(torch.abs(c_val)).item()
            rel_err = abs_err / max(denom, 1e-10)
            max_error = max(max_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)
            c_val = torch.from_numpy(y_c).float()
            tr_val = y_tr.cpu().float()
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
