#!/usr/bin/env python3
"""Correctness test for fdtd_2d (Polybench) - attempt 1"""
import sys
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

# Import Triton implementation
try:
    from polybench_results.llm_triton.fdtd_2d.attempt1 import fdtd_2d_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Load C reference
C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs" / "libfdtd_2d.so"
if not C_LIB_PATH.exists():
    print(f"C reference library not found: {C_LIB_PATH}")
    sys.exit(1)

def run_c_reference(_fict__c, ex_c, ey_c, hz_c, NX, NY, TMAX):
    """Run C reference kernel via ctypes."""
    lib = ctypes.CDLL(str(C_LIB_PATH))

    # Set global arrays in the .so
    CType__fict_ = ctypes.c_float * (20)
    c_arr__fict_ = CType__fict_.in_dll(lib, '_fict_')
    src__fict_ = np.ascontiguousarray(_fict__c, dtype=np.float32)
    ctypes.memmove(c_arr__fict_, src__fict_.ctypes.data, src__fict_.nbytes)
    CType_ex = ctypes.c_float * (60 * 80)
    c_arr_ex = CType_ex.in_dll(lib, 'ex')
    src_ex = np.ascontiguousarray(ex_c, dtype=np.float32)
    ctypes.memmove(c_arr_ex, src_ex.ctypes.data, src_ex.nbytes)
    CType_ey = ctypes.c_float * (60 * 80)
    c_arr_ey = CType_ey.in_dll(lib, 'ey')
    src_ey = np.ascontiguousarray(ey_c, dtype=np.float32)
    ctypes.memmove(c_arr_ey, src_ey.ctypes.data, src_ey.nbytes)
    CType_hz = ctypes.c_float * (60 * 80)
    c_arr_hz = CType_hz.in_dll(lib, 'hz')
    src_hz = np.ascontiguousarray(hz_c, dtype=np.float32)
    ctypes.memmove(c_arr_hz, src_hz.ctypes.data, src_hz.nbytes)

    # Set global scalars
    pass

    # Run kernel
    func = getattr(lib, "fdtd_2d_kernel")
    func.argtypes = []
    func.restype = None
    func()

    # Read back output arrays
    CType_ex = ctypes.c_float * (60 * 80)
    c_arr_ex = CType_ex.in_dll(lib, 'ex')
    ex_c[:] = np.frombuffer(c_arr_ex, dtype=np.float32).reshape(60, 80).copy()
    CType_ey = ctypes.c_float * (60 * 80)
    c_arr_ey = CType_ey.in_dll(lib, 'ey')
    ey_c[:] = np.frombuffer(c_arr_ey, dtype=np.float32).reshape(60, 80).copy()
    CType_hz = ctypes.c_float * (60 * 80)
    c_arr_hz = CType_hz.in_dll(lib, 'hz')
    hz_c[:] = np.frombuffer(c_arr_hz, dtype=np.float32).reshape(60, 80).copy()

def test_correctness():
    """Test Triton vs C reference."""
    num_tests = 3
    all_passed = True

    for test_idx in range(num_tests):
        try:
            # Initialize arrays
            _fict_ = torch.randn(20, device='cuda', dtype=torch.float32)
            ex = torch.randn(60, 80, device='cuda', dtype=torch.float32)
            ey = torch.randn(60, 80, device='cuda', dtype=torch.float32)
            hz = torch.randn(60, 80, device='cuda', dtype=torch.float32)
            NX = 60
            NY = 80
            TMAX = 20

            # Clone for C reference
            _fict__c = _fict_.cpu().numpy().copy()
            ex_c = ex.cpu().numpy().copy()
            ey_c = ey.cpu().numpy().copy()
            hz_c = hz.cpu().numpy().copy()

            # Clone for Triton
            _fict__tr = _fict_.clone()
            ex_tr = ex.clone()
            ey_tr = ey.clone()
            hz_tr = hz.clone()

            # Run C reference
            run_c_reference(_fict__c, ex_c, ey_c, hz_c, NX, NY, TMAX)

            # Run Triton
            fdtd_2d_triton(_fict__tr, ex_tr, ey_tr, hz_tr, NX, NY, TMAX)

            # Compare output arrays
            max_error = 0.0
            max_rel_error = 0.0
            c_val = torch.from_numpy(ex_c).float()
            tr_val = ex_tr.cpu().float()
            abs_err = torch.max(torch.abs(c_val - tr_val)).item()
            denom = torch.max(torch.abs(c_val)).item()
            rel_err = abs_err / max(denom, 1e-10)
            max_error = max(max_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)
            c_val = torch.from_numpy(ey_c).float()
            tr_val = ey_tr.cpu().float()
            abs_err = torch.max(torch.abs(c_val - tr_val)).item()
            denom = torch.max(torch.abs(c_val)).item()
            rel_err = abs_err / max(denom, 1e-10)
            max_error = max(max_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)
            c_val = torch.from_numpy(hz_c).float()
            tr_val = hz_tr.cpu().float()
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
