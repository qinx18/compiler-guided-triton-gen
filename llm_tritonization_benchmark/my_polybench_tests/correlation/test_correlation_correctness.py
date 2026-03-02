#!/usr/bin/env python3
"""Correctness test for correlation (Polybench) - attempt 2"""
import sys
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

# Import Triton implementation
try:
    from polybench_results.llm_triton.correlation.attempt2 import correlation_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Load C reference
C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs" / "libcorrelation.so"
if not C_LIB_PATH.exists():
    print(f"C reference library not found: {C_LIB_PATH}")
    sys.exit(1)

def run_c_reference(corr_c, data_c, mean_c, stddev_c, eps, float_n, M, N):
    """Run C reference kernel via ctypes."""
    lib = ctypes.CDLL(str(C_LIB_PATH))

    # Set global arrays in the .so
    CType_corr = ctypes.c_float * (80 * 80)
    c_arr_corr = CType_corr.in_dll(lib, 'corr')
    src_corr = np.ascontiguousarray(corr_c, dtype=np.float32)
    ctypes.memmove(c_arr_corr, src_corr.ctypes.data, src_corr.nbytes)
    CType_data = ctypes.c_float * (100 * 80)
    c_arr_data = CType_data.in_dll(lib, 'data')
    src_data = np.ascontiguousarray(data_c, dtype=np.float32)
    ctypes.memmove(c_arr_data, src_data.ctypes.data, src_data.nbytes)
    CType_mean = ctypes.c_float * (80)
    c_arr_mean = CType_mean.in_dll(lib, 'mean')
    src_mean = np.ascontiguousarray(mean_c, dtype=np.float32)
    ctypes.memmove(c_arr_mean, src_mean.ctypes.data, src_mean.nbytes)
    CType_stddev = ctypes.c_float * (80)
    c_arr_stddev = CType_stddev.in_dll(lib, 'stddev')
    src_stddev = np.ascontiguousarray(stddev_c, dtype=np.float32)
    ctypes.memmove(c_arr_stddev, src_stddev.ctypes.data, src_stddev.nbytes)

    # Set global scalars
    ctypes.c_float.in_dll(lib, 'eps').value = float(eps)
    ctypes.c_float.in_dll(lib, 'float_n').value = float(float_n)

    # Run kernel
    func = getattr(lib, "correlation_kernel")
    func.argtypes = []
    func.restype = None
    func()

    # Read back output arrays
    CType_corr = ctypes.c_float * (80 * 80)
    c_arr_corr = CType_corr.in_dll(lib, 'corr')
    corr_c[:] = np.frombuffer(c_arr_corr, dtype=np.float32).reshape(80, 80).copy()
    CType_data = ctypes.c_float * (100 * 80)
    c_arr_data = CType_data.in_dll(lib, 'data')
    data_c[:] = np.frombuffer(c_arr_data, dtype=np.float32).reshape(100, 80).copy()
    CType_mean = ctypes.c_float * (80)
    c_arr_mean = CType_mean.in_dll(lib, 'mean')
    mean_c[:] = np.frombuffer(c_arr_mean, dtype=np.float32).reshape(80).copy()
    CType_stddev = ctypes.c_float * (80)
    c_arr_stddev = CType_stddev.in_dll(lib, 'stddev')
    stddev_c[:] = np.frombuffer(c_arr_stddev, dtype=np.float32).reshape(80).copy()

def test_correctness():
    """Test Triton vs C reference."""
    num_tests = 3
    all_passed = True

    for test_idx in range(num_tests):
        try:
            # Initialize arrays
            corr = torch.randn(80, 80, device='cuda', dtype=torch.float32)
            data = torch.randn(100, 80, device='cuda', dtype=torch.float32)
            mean = torch.randn(80, device='cuda', dtype=torch.float32)
            stddev = torch.randn(80, device='cuda', dtype=torch.float32)
            eps = 0.1
            float_n = float(100)
            M = 80
            N = 100

            # Clone for C reference
            corr_c = corr.cpu().numpy().copy()
            data_c = data.cpu().numpy().copy()
            mean_c = mean.cpu().numpy().copy()
            stddev_c = stddev.cpu().numpy().copy()

            # Clone for Triton
            corr_tr = corr.clone()
            data_tr = data.clone()
            mean_tr = mean.clone()
            stddev_tr = stddev.clone()

            # Run C reference
            run_c_reference(corr_c, data_c, mean_c, stddev_c, eps, float_n, M, N)

            # Run Triton
            correlation_triton(corr_tr, data_tr, mean_tr, stddev_tr, eps, float_n, M, N)

            # Compare output arrays
            max_error = 0.0
            max_rel_error = 0.0
            c_val = torch.from_numpy(corr_c).float()
            tr_val = corr_tr.cpu().float()
            abs_err = torch.max(torch.abs(c_val - tr_val)).item()
            denom = torch.max(torch.abs(c_val)).item()
            rel_err = abs_err / max(denom, 1e-10)
            max_error = max(max_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)
            c_val = torch.from_numpy(data_c).float()
            tr_val = data_tr.cpu().float()
            abs_err = torch.max(torch.abs(c_val - tr_val)).item()
            denom = torch.max(torch.abs(c_val)).item()
            rel_err = abs_err / max(denom, 1e-10)
            max_error = max(max_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)
            c_val = torch.from_numpy(mean_c).float()
            tr_val = mean_tr.cpu().float()
            abs_err = torch.max(torch.abs(c_val - tr_val)).item()
            denom = torch.max(torch.abs(c_val)).item()
            rel_err = abs_err / max(denom, 1e-10)
            max_error = max(max_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)
            c_val = torch.from_numpy(stddev_c).float()
            tr_val = stddev_tr.cpu().float()
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
