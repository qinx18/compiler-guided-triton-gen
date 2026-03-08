#!/usr/bin/env python3
"""Correctness test for hotspot (Rodinia) - attempt 1"""
import sys
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

# Import Triton implementation
try:
    from rodinia_results.llm_triton.hotspot.attempt1 import hotspot_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Load C reference
C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "rodinia_libs" / "libhotspot.so"
if not C_LIB_PATH.exists():
    print(f"C reference library not found: {C_LIB_PATH}")
    sys.exit(1)

def run_c_reference(power_c, result_c, temp_c, Cap_1, Rx_1, Ry_1, Rz_1, amb_temp, COLS, ROWS, TSTEPS):
    """Run C reference kernel via ctypes."""
    lib = ctypes.CDLL(str(C_LIB_PATH))

    # Set global arrays in the .so
    CType_power = ctypes.c_float * (256 * 256)
    c_arr_power = CType_power.in_dll(lib, 'power')
    src_power = np.ascontiguousarray(power_c, dtype=np.float32)
    ctypes.memmove(c_arr_power, src_power.ctypes.data, src_power.nbytes)
    CType_result = ctypes.c_float * (256 * 256)
    c_arr_result = CType_result.in_dll(lib, 'result')
    src_result = np.ascontiguousarray(result_c, dtype=np.float32)
    ctypes.memmove(c_arr_result, src_result.ctypes.data, src_result.nbytes)
    CType_temp = ctypes.c_float * (256 * 256)
    c_arr_temp = CType_temp.in_dll(lib, 'temp')
    src_temp = np.ascontiguousarray(temp_c, dtype=np.float32)
    ctypes.memmove(c_arr_temp, src_temp.ctypes.data, src_temp.nbytes)

    # Set global scalars
    ctypes.c_float.in_dll(lib, 'Cap_1').value = float(Cap_1)
    ctypes.c_float.in_dll(lib, 'Rx_1').value = float(Rx_1)
    ctypes.c_float.in_dll(lib, 'Ry_1').value = float(Ry_1)
    ctypes.c_float.in_dll(lib, 'Rz_1').value = float(Rz_1)
    ctypes.c_float.in_dll(lib, 'amb_temp').value = float(amb_temp)

    # Run kernel
    func = getattr(lib, "hotspot_kernel")
    func.argtypes = []
    func.restype = None
    func()

    # Read back output arrays
    CType_result = ctypes.c_float * (256 * 256)
    c_arr_result = CType_result.in_dll(lib, 'result')
    result_c[:] = np.frombuffer(c_arr_result, dtype=np.float32).reshape(256, 256).copy()
    CType_temp = ctypes.c_float * (256 * 256)
    c_arr_temp = CType_temp.in_dll(lib, 'temp')
    temp_c[:] = np.frombuffer(c_arr_temp, dtype=np.float32).reshape(256, 256).copy()

def test_correctness():
    """Test Triton vs C reference."""
    num_tests = 3
    all_passed = True

    for test_idx in range(num_tests):
        try:
            # Initialize arrays
            # Realistic temperatures ~300K with variation
            temp = torch.randn(256, 256, device='cuda', dtype=torch.float32) * 50.0 + 300.0
            power = torch.abs(torch.randn(256, 256, device='cuda', dtype=torch.float32)) * 0.5
            result = torch.zeros(256, 256, device='cuda', dtype=torch.float32)
            Cap_1 = 0.0002
            Rx_1 = 51200.0
            Ry_1 = 51200.0
            Rz_1 = 320000.0
            amb_temp = 80.0
            COLS = 256
            ROWS = 256
            TSTEPS = 10

            # Clone for C reference
            power_c = power.cpu().numpy().copy()
            result_c = result.cpu().numpy().copy()
            temp_c = temp.cpu().numpy().copy()

            # Clone for Triton
            power_tr = power.clone()
            result_tr = result.clone()
            temp_tr = temp.clone()

            # Run C reference
            run_c_reference(power_c, result_c, temp_c, Cap_1, Rx_1, Ry_1, Rz_1, amb_temp, COLS, ROWS, TSTEPS)

            # Run Triton
            hotspot_triton(power_tr, result_tr, temp_tr, Cap_1, Rx_1, Ry_1, Rz_1, amb_temp, COLS, ROWS, TSTEPS)

            # Compare output arrays
            max_error = 0.0
            max_rel_error = 0.0
            c_val = torch.from_numpy(result_c).float()
            tr_val = result_tr.cpu().float()
            abs_err = torch.max(torch.abs(c_val - tr_val)).item()
            denom = torch.max(torch.abs(c_val)).item()
            rel_err = abs_err / max(denom, 1e-10)
            max_error = max(max_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)
            c_val = torch.from_numpy(temp_c).float()
            tr_val = temp_tr.cpu().float()
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
