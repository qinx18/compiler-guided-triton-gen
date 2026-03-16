#!/usr/bin/env python3
"""Correctness test for srad (Real-World) - attempt 1"""
import sys
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

# Import Triton implementation
try:
    from realworld_results.llm_triton.srad.attempt1 import srad_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Load C reference
C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "realworld_libs" / "libsrad.so"
if not C_LIB_PATH.exists():
    print(f"C reference library not found: {C_LIB_PATH}")
    sys.exit(1)

def run_c_reference(J_c, c_c, dE_c, dN_c, dS_c, dW_c, iN_c, iS_c, jE_c, jW_c, lambda_val, C1, C2, COLS, NITER, R1, R2, ROWS):
    """Run C reference kernel via ctypes."""
    lib = ctypes.CDLL(str(C_LIB_PATH))

    # Set global arrays in the .so
    CType_J = ctypes.c_float * (262144)
    c_arr_J = CType_J.in_dll(lib, 'J')
    src_J = np.ascontiguousarray(J_c, dtype=np.float32)
    ctypes.memmove(c_arr_J, src_J.ctypes.data, src_J.nbytes)
    CType_c = ctypes.c_float * (262144)
    c_arr_c = CType_c.in_dll(lib, 'c')
    src_c = np.ascontiguousarray(c_c, dtype=np.float32)
    ctypes.memmove(c_arr_c, src_c.ctypes.data, src_c.nbytes)
    CType_dE = ctypes.c_float * (262144)
    c_arr_dE = CType_dE.in_dll(lib, 'dE')
    src_dE = np.ascontiguousarray(dE_c, dtype=np.float32)
    ctypes.memmove(c_arr_dE, src_dE.ctypes.data, src_dE.nbytes)
    CType_dN = ctypes.c_float * (262144)
    c_arr_dN = CType_dN.in_dll(lib, 'dN')
    src_dN = np.ascontiguousarray(dN_c, dtype=np.float32)
    ctypes.memmove(c_arr_dN, src_dN.ctypes.data, src_dN.nbytes)
    CType_dS = ctypes.c_float * (262144)
    c_arr_dS = CType_dS.in_dll(lib, 'dS')
    src_dS = np.ascontiguousarray(dS_c, dtype=np.float32)
    ctypes.memmove(c_arr_dS, src_dS.ctypes.data, src_dS.nbytes)
    CType_dW = ctypes.c_float * (262144)
    c_arr_dW = CType_dW.in_dll(lib, 'dW')
    src_dW = np.ascontiguousarray(dW_c, dtype=np.float32)
    ctypes.memmove(c_arr_dW, src_dW.ctypes.data, src_dW.nbytes)
    CType_iN = ctypes.c_int * (512)
    c_arr_iN = CType_iN.in_dll(lib, 'iN')
    src_iN = np.ascontiguousarray(iN_c.astype(np.int32), dtype=np.int32)
    ctypes.memmove(c_arr_iN, src_iN.ctypes.data, src_iN.nbytes)
    CType_iS = ctypes.c_int * (512)
    c_arr_iS = CType_iS.in_dll(lib, 'iS')
    src_iS = np.ascontiguousarray(iS_c.astype(np.int32), dtype=np.int32)
    ctypes.memmove(c_arr_iS, src_iS.ctypes.data, src_iS.nbytes)
    CType_jE = ctypes.c_int * (512)
    c_arr_jE = CType_jE.in_dll(lib, 'jE')
    src_jE = np.ascontiguousarray(jE_c.astype(np.int32), dtype=np.int32)
    ctypes.memmove(c_arr_jE, src_jE.ctypes.data, src_jE.nbytes)
    CType_jW = ctypes.c_int * (512)
    c_arr_jW = CType_jW.in_dll(lib, 'jW')
    src_jW = np.ascontiguousarray(jW_c.astype(np.int32), dtype=np.int32)
    ctypes.memmove(c_arr_jW, src_jW.ctypes.data, src_jW.nbytes)

    # Set global scalars
    ctypes.c_float.in_dll(lib, 'lambda_val').value = float(lambda_val)

    # Run kernel
    func = getattr(lib, "srad_kernel")
    func.argtypes = []
    func.restype = None
    func()

    # Read back output arrays
    CType_J = ctypes.c_float * (262144)
    c_arr_J = CType_J.in_dll(lib, 'J')
    J_c[:] = np.frombuffer(c_arr_J, dtype=np.float32).reshape(262144).copy()
    CType_c = ctypes.c_float * (262144)
    c_arr_c = CType_c.in_dll(lib, 'c')
    c_c[:] = np.frombuffer(c_arr_c, dtype=np.float32).reshape(262144).copy()
    CType_dE = ctypes.c_float * (262144)
    c_arr_dE = CType_dE.in_dll(lib, 'dE')
    dE_c[:] = np.frombuffer(c_arr_dE, dtype=np.float32).reshape(262144).copy()
    CType_dN = ctypes.c_float * (262144)
    c_arr_dN = CType_dN.in_dll(lib, 'dN')
    dN_c[:] = np.frombuffer(c_arr_dN, dtype=np.float32).reshape(262144).copy()
    CType_dS = ctypes.c_float * (262144)
    c_arr_dS = CType_dS.in_dll(lib, 'dS')
    dS_c[:] = np.frombuffer(c_arr_dS, dtype=np.float32).reshape(262144).copy()
    CType_dW = ctypes.c_float * (262144)
    c_arr_dW = CType_dW.in_dll(lib, 'dW')
    dW_c[:] = np.frombuffer(c_arr_dW, dtype=np.float32).reshape(262144).copy()

def test_correctness():
    """Test Triton vs C reference."""
    num_tests = 3
    all_passed = True

    for test_idx in range(num_tests):
        try:
            # Initialize arrays
            # J: positive image values (exp of random)
            J = torch.exp(torch.randn(262144, device='cuda', dtype=torch.float32).clamp(-3, 3))
            # Boundary index arrays (clamp-to-edge)
            iN = torch.arange(512, device='cuda', dtype=torch.int32)
            iN[0] = 0
            iN[1:] = torch.arange(0, 511, device='cuda', dtype=torch.int32)
            iS = torch.arange(512, device='cuda', dtype=torch.int32)
            iS[:-1] = torch.arange(1, 512, device='cuda', dtype=torch.int32)
            iS[-1] = 511
            jW = torch.arange(512, device='cuda', dtype=torch.int32)
            jW[0] = 0
            jW[1:] = torch.arange(0, 511, device='cuda', dtype=torch.int32)
            jE = torch.arange(512, device='cuda', dtype=torch.int32)
            jE[:-1] = torch.arange(1, 512, device='cuda', dtype=torch.int32)
            jE[-1] = 511
            # Work arrays initialized to zero
            c = torch.zeros(262144, device='cuda', dtype=torch.float32)
            dN = torch.zeros(262144, device='cuda', dtype=torch.float32)
            dS = torch.zeros(262144, device='cuda', dtype=torch.float32)
            dW = torch.zeros(262144, device='cuda', dtype=torch.float32)
            dE = torch.zeros(262144, device='cuda', dtype=torch.float32)
            lambda_val = 0.5
            lambda_val = 0.5
            C1 = 0
            C2 = 127
            COLS = 512
            NITER = 10
            R1 = 0
            R2 = 127
            ROWS = 512

            # Clone for C reference
            J_c = J.cpu().numpy().copy()
            c_c = c.cpu().numpy().copy()
            dE_c = dE.cpu().numpy().copy()
            dN_c = dN.cpu().numpy().copy()
            dS_c = dS.cpu().numpy().copy()
            dW_c = dW.cpu().numpy().copy()
            iN_c = iN.cpu().numpy().copy()
            iS_c = iS.cpu().numpy().copy()
            jE_c = jE.cpu().numpy().copy()
            jW_c = jW.cpu().numpy().copy()

            # Clone for Triton
            J_tr = J.clone()
            c_tr = c.clone()
            dE_tr = dE.clone()
            dN_tr = dN.clone()
            dS_tr = dS.clone()
            dW_tr = dW.clone()
            iN_tr = iN.clone()
            iS_tr = iS.clone()
            jE_tr = jE.clone()
            jW_tr = jW.clone()

            # Run C reference
            run_c_reference(J_c, c_c, dE_c, dN_c, dS_c, dW_c, iN_c, iS_c, jE_c, jW_c, lambda_val, C1, C2, COLS, NITER, R1, R2, ROWS)

            # Run Triton
            srad_triton(J_tr, c_tr, dE_tr, dN_tr, dS_tr, dW_tr, iN_tr, iS_tr, jE_tr, jW_tr, lambda_val, C1, C2, COLS, NITER, R1, R2, ROWS)

            # Compare output arrays
            max_error = 0.0
            max_rel_error = 0.0
            c_val = torch.from_numpy(J_c).float()
            tr_val = J_tr.cpu().float()
            abs_err = torch.max(torch.abs(c_val - tr_val)).item()
            denom = torch.max(torch.abs(c_val)).item()
            rel_err = abs_err / max(denom, 1e-10)
            max_error = max(max_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)
            c_val = torch.from_numpy(c_c).float()
            tr_val = c_tr.cpu().float()
            abs_err = torch.max(torch.abs(c_val - tr_val)).item()
            denom = torch.max(torch.abs(c_val)).item()
            rel_err = abs_err / max(denom, 1e-10)
            max_error = max(max_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)
            c_val = torch.from_numpy(dE_c).float()
            tr_val = dE_tr.cpu().float()
            abs_err = torch.max(torch.abs(c_val - tr_val)).item()
            denom = torch.max(torch.abs(c_val)).item()
            rel_err = abs_err / max(denom, 1e-10)
            max_error = max(max_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)
            c_val = torch.from_numpy(dN_c).float()
            tr_val = dN_tr.cpu().float()
            abs_err = torch.max(torch.abs(c_val - tr_val)).item()
            denom = torch.max(torch.abs(c_val)).item()
            rel_err = abs_err / max(denom, 1e-10)
            max_error = max(max_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)
            c_val = torch.from_numpy(dS_c).float()
            tr_val = dS_tr.cpu().float()
            abs_err = torch.max(torch.abs(c_val - tr_val)).item()
            denom = torch.max(torch.abs(c_val)).item()
            rel_err = abs_err / max(denom, 1e-10)
            max_error = max(max_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)
            c_val = torch.from_numpy(dW_c).float()
            tr_val = dW_tr.cpu().float()
            abs_err = torch.max(torch.abs(c_val - tr_val)).item()
            denom = torch.max(torch.abs(c_val)).item()
            rel_err = abs_err / max(denom, 1e-10)
            max_error = max(max_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)

            # Pass if absolute error < atol OR relative error < rtol
            passed = (max_error < 0.01) or (max_rel_error < 0.001)
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
