#!/usr/bin/env python3
"""Correctness test for 2mm (Polybench) - attempt 10"""
import sys
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

# Import Triton implementation
try:
    import importlib
    _mod = importlib.import_module("polybench_results.llm_triton_no_analysis.2mm.attempt10")
    k2mm_triton = _mod.k2mm_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Load C reference
C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs" / "lib2mm.so"
if not C_LIB_PATH.exists():
    print(f"C reference library not found: {C_LIB_PATH}")
    sys.exit(1)

def run_c_reference(A_c, B_c, C_c, D_c, tmp_c, alpha, beta, NI, NJ, NK, NL):
    """Run C reference kernel via ctypes."""
    lib = ctypes.CDLL(str(C_LIB_PATH))

    # Set global arrays in the .so
    CType_A = ctypes.c_float * (40 * 70)
    c_arr_A = CType_A.in_dll(lib, 'A')
    src_A = np.ascontiguousarray(A_c, dtype=np.float32)
    ctypes.memmove(c_arr_A, src_A.ctypes.data, src_A.nbytes)
    CType_B = ctypes.c_float * (70 * 50)
    c_arr_B = CType_B.in_dll(lib, 'B')
    src_B = np.ascontiguousarray(B_c, dtype=np.float32)
    ctypes.memmove(c_arr_B, src_B.ctypes.data, src_B.nbytes)
    CType_C = ctypes.c_float * (50 * 80)
    c_arr_C = CType_C.in_dll(lib, 'C')
    src_C = np.ascontiguousarray(C_c, dtype=np.float32)
    ctypes.memmove(c_arr_C, src_C.ctypes.data, src_C.nbytes)
    CType_D = ctypes.c_float * (40 * 80)
    c_arr_D = CType_D.in_dll(lib, 'D')
    src_D = np.ascontiguousarray(D_c, dtype=np.float32)
    ctypes.memmove(c_arr_D, src_D.ctypes.data, src_D.nbytes)
    CType_tmp = ctypes.c_float * (40 * 50)
    c_arr_tmp = CType_tmp.in_dll(lib, 'tmp')
    src_tmp = np.ascontiguousarray(tmp_c, dtype=np.float32)
    ctypes.memmove(c_arr_tmp, src_tmp.ctypes.data, src_tmp.nbytes)

    # Set global scalars
    ctypes.c_float.in_dll(lib, 'alpha').value = float(alpha)
    ctypes.c_float.in_dll(lib, 'beta').value = float(beta)

    # Run kernel
    func = getattr(lib, "k2mm_kernel")
    func.argtypes = []
    func.restype = None
    func()

    # Read back output arrays
    CType_D = ctypes.c_float * (40 * 80)
    c_arr_D = CType_D.in_dll(lib, 'D')
    D_c[:] = np.frombuffer(c_arr_D, dtype=np.float32).reshape(40, 80).copy()
    CType_tmp = ctypes.c_float * (40 * 50)
    c_arr_tmp = CType_tmp.in_dll(lib, 'tmp')
    tmp_c[:] = np.frombuffer(c_arr_tmp, dtype=np.float32).reshape(40, 50).copy()

def test_correctness():
    """Test Triton vs C reference."""
    num_tests = 3
    all_passed = True

    for test_idx in range(num_tests):
        try:
            # Initialize arrays
            A = torch.randn(40, 70, device='cuda', dtype=torch.float32)
            B = torch.randn(70, 50, device='cuda', dtype=torch.float32)
            C = torch.randn(50, 80, device='cuda', dtype=torch.float32)
            D = torch.randn(40, 80, device='cuda', dtype=torch.float32)
            tmp = torch.randn(40, 50, device='cuda', dtype=torch.float32)
            alpha = 1.5
            beta = 1.5
            NI = 40
            NJ = 50
            NK = 70
            NL = 80

            # Clone for C reference
            A_c = A.cpu().numpy().copy()
            B_c = B.cpu().numpy().copy()
            C_c = C.cpu().numpy().copy()
            D_c = D.cpu().numpy().copy()
            tmp_c = tmp.cpu().numpy().copy()

            # Clone for Triton
            A_tr = A.clone()
            B_tr = B.clone()
            C_tr = C.clone()
            D_tr = D.clone()
            tmp_tr = tmp.clone()

            # Run C reference
            run_c_reference(A_c, B_c, C_c, D_c, tmp_c, alpha, beta, NI, NJ, NK, NL)

            # Run Triton
            k2mm_triton(A_tr, B_tr, C_tr, D_tr, tmp_tr, alpha, beta, NI, NJ, NK, NL)

            # Compare output arrays
            max_error = 0.0
            max_rel_error = 0.0
            c_val = torch.from_numpy(D_c).float()
            tr_val = D_tr.cpu().float()
            abs_err = torch.max(torch.abs(c_val - tr_val)).item()
            denom = torch.max(torch.abs(c_val)).item()
            rel_err = abs_err / max(denom, 1e-10)
            max_error = max(max_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)
            c_val = torch.from_numpy(tmp_c).float()
            tr_val = tmp_tr.cpu().float()
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
