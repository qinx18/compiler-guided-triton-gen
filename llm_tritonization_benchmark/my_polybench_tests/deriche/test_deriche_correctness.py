#!/usr/bin/env python3
"""Correctness test for deriche (Polybench) - attempt 4"""
import sys
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

# Import Triton implementation
try:
    from polybench_results_scale8x.llm_triton_no_analysis.deriche.attempt4 import deriche_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Load C reference
C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs_scale8x" / "libderiche.so"
if not C_LIB_PATH.exists():
    print(f"C reference library not found: {C_LIB_PATH}")
    sys.exit(1)

def run_c_reference(imgIn_c, imgOut_c, y2_c, yy1_c, alpha, H, W):
    """Run C reference kernel via ctypes."""
    lib = ctypes.CDLL(str(C_LIB_PATH))

    # Set global arrays in the .so
    CType_imgIn = ctypes.c_float * (1536 * 1024)
    c_arr_imgIn = CType_imgIn.in_dll(lib, 'imgIn')
    src_imgIn = np.ascontiguousarray(imgIn_c, dtype=np.float32)
    ctypes.memmove(c_arr_imgIn, src_imgIn.ctypes.data, src_imgIn.nbytes)
    CType_imgOut = ctypes.c_float * (1536 * 1024)
    c_arr_imgOut = CType_imgOut.in_dll(lib, 'imgOut')
    src_imgOut = np.ascontiguousarray(imgOut_c, dtype=np.float32)
    ctypes.memmove(c_arr_imgOut, src_imgOut.ctypes.data, src_imgOut.nbytes)
    CType_y2 = ctypes.c_float * (1536 * 1024)
    c_arr_y2 = CType_y2.in_dll(lib, 'y2')
    src_y2 = np.ascontiguousarray(y2_c, dtype=np.float32)
    ctypes.memmove(c_arr_y2, src_y2.ctypes.data, src_y2.nbytes)
    CType_yy1 = ctypes.c_float * (1536 * 1024)
    c_arr_yy1 = CType_yy1.in_dll(lib, 'yy1')
    src_yy1 = np.ascontiguousarray(yy1_c, dtype=np.float32)
    ctypes.memmove(c_arr_yy1, src_yy1.ctypes.data, src_yy1.nbytes)

    # Set global scalars
    ctypes.c_float.in_dll(lib, 'alpha').value = float(alpha)

    # Run kernel
    func = getattr(lib, "deriche_kernel")
    func.argtypes = []
    func.restype = None
    func()

    # Read back output arrays
    CType_imgOut = ctypes.c_float * (1536 * 1024)
    c_arr_imgOut = CType_imgOut.in_dll(lib, 'imgOut')
    imgOut_c[:] = np.frombuffer(c_arr_imgOut, dtype=np.float32).reshape(1536, 1024).copy()
    CType_y2 = ctypes.c_float * (1536 * 1024)
    c_arr_y2 = CType_y2.in_dll(lib, 'y2')
    y2_c[:] = np.frombuffer(c_arr_y2, dtype=np.float32).reshape(1536, 1024).copy()
    CType_yy1 = ctypes.c_float * (1536 * 1024)
    c_arr_yy1 = CType_yy1.in_dll(lib, 'yy1')
    yy1_c[:] = np.frombuffer(c_arr_yy1, dtype=np.float32).reshape(1536, 1024).copy()

def test_correctness():
    """Test Triton vs C reference."""
    num_tests = 3
    all_passed = True

    for test_idx in range(num_tests):
        try:
            # Initialize arrays
            imgIn = torch.randn(1536, 1024, device='cuda', dtype=torch.float32)
            imgOut = torch.randn(1536, 1024, device='cuda', dtype=torch.float32)
            y2 = torch.randn(1536, 1024, device='cuda', dtype=torch.float32)
            yy1 = torch.randn(1536, 1024, device='cuda', dtype=torch.float32)
            alpha = 1.5
            H = 1024
            W = 1536

            # Clone for C reference
            imgIn_c = imgIn.cpu().numpy().copy()
            imgOut_c = imgOut.cpu().numpy().copy()
            y2_c = y2.cpu().numpy().copy()
            yy1_c = yy1.cpu().numpy().copy()

            # Clone for Triton
            imgIn_tr = imgIn.clone()
            imgOut_tr = imgOut.clone()
            y2_tr = y2.clone()
            yy1_tr = yy1.clone()

            # Run C reference
            run_c_reference(imgIn_c, imgOut_c, y2_c, yy1_c, alpha, H, W)

            # Run Triton
            deriche_triton(imgIn_tr, imgOut_tr, y2_tr, yy1_tr, alpha, H, W)

            # Compare output arrays
            max_error = 0.0
            max_rel_error = 0.0
            c_val = torch.from_numpy(imgOut_c).float()
            tr_val = imgOut_tr.cpu().float()
            abs_err = torch.max(torch.abs(c_val - tr_val)).item()
            denom = torch.max(torch.abs(c_val)).item()
            rel_err = abs_err / max(denom, 1e-10)
            max_error = max(max_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)
            c_val = torch.from_numpy(y2_c).float()
            tr_val = y2_tr.cpu().float()
            abs_err = torch.max(torch.abs(c_val - tr_val)).item()
            denom = torch.max(torch.abs(c_val)).item()
            rel_err = abs_err / max(denom, 1e-10)
            max_error = max(max_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)
            c_val = torch.from_numpy(yy1_c).float()
            tr_val = yy1_tr.cpu().float()
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
