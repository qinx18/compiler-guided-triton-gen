#!/usr/bin/env python3
"""Correctness test for mvt (Polybench) - attempt 1"""
import sys
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

# Import Triton implementation
try:
    from polybench_results_scale8x.llm_triton_no_analysis.mvt.attempt1 import mvt_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Load C reference
C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs_scale8x" / "libmvt.so"
if not C_LIB_PATH.exists():
    print(f"C reference library not found: {C_LIB_PATH}")
    sys.exit(1)

def run_c_reference(A_c, x1_c, x2_c, y_1_c, y_2_c, N):
    """Run C reference kernel via ctypes."""
    lib = ctypes.CDLL(str(C_LIB_PATH))

    # Set global arrays in the .so
    CType_A = ctypes.c_float * (960 * 960)
    c_arr_A = CType_A.in_dll(lib, 'A')
    src_A = np.ascontiguousarray(A_c, dtype=np.float32)
    ctypes.memmove(c_arr_A, src_A.ctypes.data, src_A.nbytes)
    CType_x1 = ctypes.c_float * (960)
    c_arr_x1 = CType_x1.in_dll(lib, 'x1')
    src_x1 = np.ascontiguousarray(x1_c, dtype=np.float32)
    ctypes.memmove(c_arr_x1, src_x1.ctypes.data, src_x1.nbytes)
    CType_x2 = ctypes.c_float * (960)
    c_arr_x2 = CType_x2.in_dll(lib, 'x2')
    src_x2 = np.ascontiguousarray(x2_c, dtype=np.float32)
    ctypes.memmove(c_arr_x2, src_x2.ctypes.data, src_x2.nbytes)
    CType_y_1 = ctypes.c_float * (960)
    c_arr_y_1 = CType_y_1.in_dll(lib, 'y_1')
    src_y_1 = np.ascontiguousarray(y_1_c, dtype=np.float32)
    ctypes.memmove(c_arr_y_1, src_y_1.ctypes.data, src_y_1.nbytes)
    CType_y_2 = ctypes.c_float * (960)
    c_arr_y_2 = CType_y_2.in_dll(lib, 'y_2')
    src_y_2 = np.ascontiguousarray(y_2_c, dtype=np.float32)
    ctypes.memmove(c_arr_y_2, src_y_2.ctypes.data, src_y_2.nbytes)

    # Set global scalars
    pass

    # Run kernel
    func = getattr(lib, "mvt_kernel")
    func.argtypes = []
    func.restype = None
    func()

    # Read back output arrays
    CType_x1 = ctypes.c_float * (960)
    c_arr_x1 = CType_x1.in_dll(lib, 'x1')
    x1_c[:] = np.frombuffer(c_arr_x1, dtype=np.float32).reshape(960).copy()
    CType_x2 = ctypes.c_float * (960)
    c_arr_x2 = CType_x2.in_dll(lib, 'x2')
    x2_c[:] = np.frombuffer(c_arr_x2, dtype=np.float32).reshape(960).copy()

def test_correctness():
    """Test Triton vs C reference."""
    num_tests = 3
    all_passed = True

    for test_idx in range(num_tests):
        try:
            # Initialize arrays
            A = torch.randn(960, 960, device='cuda', dtype=torch.float32)
            x1 = torch.randn(960, device='cuda', dtype=torch.float32)
            x2 = torch.randn(960, device='cuda', dtype=torch.float32)
            y_1 = torch.randn(960, device='cuda', dtype=torch.float32)
            y_2 = torch.randn(960, device='cuda', dtype=torch.float32)
            N = 960

            # Clone for C reference
            A_c = A.cpu().numpy().copy()
            x1_c = x1.cpu().numpy().copy()
            x2_c = x2.cpu().numpy().copy()
            y_1_c = y_1.cpu().numpy().copy()
            y_2_c = y_2.cpu().numpy().copy()

            # Clone for Triton
            A_tr = A.clone()
            x1_tr = x1.clone()
            x2_tr = x2.clone()
            y_1_tr = y_1.clone()
            y_2_tr = y_2.clone()

            # Run C reference
            run_c_reference(A_c, x1_c, x2_c, y_1_c, y_2_c, N)

            # Run Triton
            mvt_triton(A_tr, x1_tr, x2_tr, y_1_tr, y_2_tr, N)

            # Compare output arrays
            max_error = 0.0
            max_rel_error = 0.0
            c_val = torch.from_numpy(x1_c).float()
            tr_val = x1_tr.cpu().float()
            abs_err = torch.max(torch.abs(c_val - tr_val)).item()
            denom = torch.max(torch.abs(c_val)).item()
            rel_err = abs_err / max(denom, 1e-10)
            max_error = max(max_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)
            c_val = torch.from_numpy(x2_c).float()
            tr_val = x2_tr.cpu().float()
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
