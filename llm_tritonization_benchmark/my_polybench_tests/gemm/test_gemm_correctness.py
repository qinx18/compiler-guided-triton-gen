#!/usr/bin/env python3
"""Correctness test for gemm (Polybench) - attempt 1"""
import sys
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

# Import Triton implementation
try:
    from polybench_results_scale8x.llm_triton_no_analysis.gemm.attempt1 import gemm_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Load C reference
C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs_scale8x" / "libgemm.so"
if not C_LIB_PATH.exists():
    print(f"C reference library not found: {C_LIB_PATH}")
    sys.exit(1)

def run_c_reference(A_c, B_c, C_c, alpha, beta, NI, NJ, NK):
    """Run C reference kernel via ctypes."""
    lib = ctypes.CDLL(str(C_LIB_PATH))

    # Set global arrays in the .so
    CType_A = ctypes.c_float * (480 * 640)
    c_arr_A = CType_A.in_dll(lib, 'A')
    src_A = np.ascontiguousarray(A_c, dtype=np.float32)
    ctypes.memmove(c_arr_A, src_A.ctypes.data, src_A.nbytes)
    CType_B = ctypes.c_float * (640 * 560)
    c_arr_B = CType_B.in_dll(lib, 'B')
    src_B = np.ascontiguousarray(B_c, dtype=np.float32)
    ctypes.memmove(c_arr_B, src_B.ctypes.data, src_B.nbytes)
    CType_C = ctypes.c_float * (480 * 560)
    c_arr_C = CType_C.in_dll(lib, 'C')
    src_C = np.ascontiguousarray(C_c, dtype=np.float32)
    ctypes.memmove(c_arr_C, src_C.ctypes.data, src_C.nbytes)

    # Set global scalars
    ctypes.c_float.in_dll(lib, 'alpha').value = float(alpha)
    ctypes.c_float.in_dll(lib, 'beta').value = float(beta)

    # Run kernel
    func = getattr(lib, "gemm_kernel")
    func.argtypes = []
    func.restype = None
    func()

    # Read back output arrays
    CType_C = ctypes.c_float * (480 * 560)
    c_arr_C = CType_C.in_dll(lib, 'C')
    C_c[:] = np.frombuffer(c_arr_C, dtype=np.float32).reshape(480, 560).copy()

def test_correctness():
    """Test Triton vs C reference."""
    num_tests = 3
    all_passed = True

    for test_idx in range(num_tests):
        try:
            # Initialize arrays
            A = torch.randn(480, 640, device='cuda', dtype=torch.float32)
            B = torch.randn(640, 560, device='cuda', dtype=torch.float32)
            C = torch.randn(480, 560, device='cuda', dtype=torch.float32)
            alpha = 1.5
            beta = 1.5
            NI = 480
            NJ = 560
            NK = 640

            # Clone for C reference
            A_c = A.cpu().numpy().copy()
            B_c = B.cpu().numpy().copy()
            C_c = C.cpu().numpy().copy()

            # Clone for Triton
            A_tr = A.clone()
            B_tr = B.clone()
            C_tr = C.clone()

            # Run C reference
            run_c_reference(A_c, B_c, C_c, alpha, beta, NI, NJ, NK)

            # Run Triton
            gemm_triton(A_tr, B_tr, C_tr, alpha, beta, NI, NJ, NK)

            # Compare output arrays
            max_error = 0.0
            max_rel_error = 0.0
            c_val = torch.from_numpy(C_c).float()
            tr_val = C_tr.cpu().float()
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
