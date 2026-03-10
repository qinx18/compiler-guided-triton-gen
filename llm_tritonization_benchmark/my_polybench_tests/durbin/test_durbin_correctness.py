#!/usr/bin/env python3
"""Correctness test for durbin (Polybench) - attempt 10"""
import sys
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

# Import Triton implementation
try:
    from polybench_results_scale8x.llm_triton_no_analysis.durbin.attempt10 import durbin_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Load C reference
C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs_scale8x" / "libdurbin.so"
if not C_LIB_PATH.exists():
    print(f"C reference library not found: {C_LIB_PATH}")
    sys.exit(1)

def run_c_reference(r_c, y_c, z_c, N):
    """Run C reference kernel via ctypes."""
    lib = ctypes.CDLL(str(C_LIB_PATH))

    # Set global arrays in the .so
    CType_r = ctypes.c_float * (960)
    c_arr_r = CType_r.in_dll(lib, 'r')
    src_r = np.ascontiguousarray(r_c, dtype=np.float32)
    ctypes.memmove(c_arr_r, src_r.ctypes.data, src_r.nbytes)
    CType_y = ctypes.c_float * (960)
    c_arr_y = CType_y.in_dll(lib, 'y')
    src_y = np.ascontiguousarray(y_c, dtype=np.float32)
    ctypes.memmove(c_arr_y, src_y.ctypes.data, src_y.nbytes)
    CType_z = ctypes.c_float * (960)
    c_arr_z = CType_z.in_dll(lib, 'z')
    src_z = np.ascontiguousarray(z_c, dtype=np.float32)
    ctypes.memmove(c_arr_z, src_z.ctypes.data, src_z.nbytes)

    # Set global scalars
    pass

    # Run kernel
    func = getattr(lib, "durbin_kernel")
    func.argtypes = []
    func.restype = None
    func()

    # Read back output arrays
    CType_y = ctypes.c_float * (960)
    c_arr_y = CType_y.in_dll(lib, 'y')
    y_c[:] = np.frombuffer(c_arr_y, dtype=np.float32).reshape(960).copy()

def test_correctness():
    """Test Triton vs C reference."""
    num_tests = 3
    all_passed = True

    for test_idx in range(num_tests):
        try:
            # Initialize arrays
            r = torch.randn(960, device='cuda', dtype=torch.float32)
            y = torch.randn(960, device='cuda', dtype=torch.float32)
            z = torch.randn(960, device='cuda', dtype=torch.float32)
            N = 960

            # Clone for C reference
            r_c = r.cpu().numpy().copy()
            y_c = y.cpu().numpy().copy()
            z_c = z.cpu().numpy().copy()

            # Clone for Triton
            r_tr = r.clone()
            y_tr = y.clone()
            z_tr = z.clone()

            # Run C reference
            run_c_reference(r_c, y_c, z_c, N)

            # Run Triton
            durbin_triton(r_tr, y_tr, z_tr, N)

            # Compare output arrays
            max_error = 0.0
            max_rel_error = 0.0
            c_val = torch.from_numpy(y_c).float()
            tr_val = y_tr.cpu().float()
            abs_err = torch.max(torch.abs(c_val - tr_val)).item()
            denom = torch.max(torch.abs(c_val)).item()
            rel_err = abs_err / max(denom, 1e-10)
            max_error = max(max_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)

            # Pass if absolute error < atol OR relative error < rtol
            passed = (max_error < 0.05) or (max_rel_error < 0.01)
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
