#!/usr/bin/env python3
"""Correctness test for 3mm (Polybench) - attempt 6"""
import sys
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

# Import Triton implementation
try:
    import importlib
    _mod = importlib.import_module("polybench_results.llm_triton.3mm.attempt6")
    k3mm_triton = _mod.k3mm_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Load C reference
C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs" / "lib3mm.so"
if not C_LIB_PATH.exists():
    print(f"C reference library not found: {C_LIB_PATH}")
    sys.exit(1)

def run_c_reference(A_c, B_c, C_c, D_c, E_c, F_c, G_c, NI, NJ, NK, NL, NM):
    """Run C reference kernel via ctypes."""
    lib = ctypes.CDLL(str(C_LIB_PATH))

    # Set global arrays in the .so
    CType_A = ctypes.c_float * (40 * 60)
    c_arr_A = CType_A.in_dll(lib, 'A')
    src_A = np.ascontiguousarray(A_c, dtype=np.float32)
    ctypes.memmove(c_arr_A, src_A.ctypes.data, src_A.nbytes)
    CType_B = ctypes.c_float * (60 * 50)
    c_arr_B = CType_B.in_dll(lib, 'B')
    src_B = np.ascontiguousarray(B_c, dtype=np.float32)
    ctypes.memmove(c_arr_B, src_B.ctypes.data, src_B.nbytes)
    CType_C = ctypes.c_float * (50 * 80)
    c_arr_C = CType_C.in_dll(lib, 'C')
    src_C = np.ascontiguousarray(C_c, dtype=np.float32)
    ctypes.memmove(c_arr_C, src_C.ctypes.data, src_C.nbytes)
    CType_D = ctypes.c_float * (80 * 70)
    c_arr_D = CType_D.in_dll(lib, 'D')
    src_D = np.ascontiguousarray(D_c, dtype=np.float32)
    ctypes.memmove(c_arr_D, src_D.ctypes.data, src_D.nbytes)
    CType_E = ctypes.c_float * (40 * 50)
    c_arr_E = CType_E.in_dll(lib, 'E')
    src_E = np.ascontiguousarray(E_c, dtype=np.float32)
    ctypes.memmove(c_arr_E, src_E.ctypes.data, src_E.nbytes)
    CType_F = ctypes.c_float * (50 * 70)
    c_arr_F = CType_F.in_dll(lib, 'F')
    src_F = np.ascontiguousarray(F_c, dtype=np.float32)
    ctypes.memmove(c_arr_F, src_F.ctypes.data, src_F.nbytes)
    CType_G = ctypes.c_float * (40 * 70)
    c_arr_G = CType_G.in_dll(lib, 'G')
    src_G = np.ascontiguousarray(G_c, dtype=np.float32)
    ctypes.memmove(c_arr_G, src_G.ctypes.data, src_G.nbytes)

    # Set global scalars
    pass

    # Run kernel
    func = getattr(lib, "k3mm_kernel")
    func.argtypes = []
    func.restype = None
    func()

    # Read back output arrays
    CType_E = ctypes.c_float * (40 * 50)
    c_arr_E = CType_E.in_dll(lib, 'E')
    E_c[:] = np.frombuffer(c_arr_E, dtype=np.float32).reshape(40, 50).copy()
    CType_F = ctypes.c_float * (50 * 70)
    c_arr_F = CType_F.in_dll(lib, 'F')
    F_c[:] = np.frombuffer(c_arr_F, dtype=np.float32).reshape(50, 70).copy()
    CType_G = ctypes.c_float * (40 * 70)
    c_arr_G = CType_G.in_dll(lib, 'G')
    G_c[:] = np.frombuffer(c_arr_G, dtype=np.float32).reshape(40, 70).copy()

def test_correctness():
    """Test Triton vs C reference."""
    num_tests = 3
    all_passed = True

    for test_idx in range(num_tests):
        try:
            # Initialize arrays
            A = torch.randn(40, 60, device='cuda', dtype=torch.float32)
            B = torch.randn(60, 50, device='cuda', dtype=torch.float32)
            C = torch.randn(50, 80, device='cuda', dtype=torch.float32)
            D = torch.randn(80, 70, device='cuda', dtype=torch.float32)
            E = torch.randn(40, 50, device='cuda', dtype=torch.float32)
            F = torch.randn(50, 70, device='cuda', dtype=torch.float32)
            G = torch.randn(40, 70, device='cuda', dtype=torch.float32)
            NI = 40
            NJ = 50
            NK = 60
            NL = 70
            NM = 80

            # Clone for C reference
            A_c = A.cpu().numpy().copy()
            B_c = B.cpu().numpy().copy()
            C_c = C.cpu().numpy().copy()
            D_c = D.cpu().numpy().copy()
            E_c = E.cpu().numpy().copy()
            F_c = F.cpu().numpy().copy()
            G_c = G.cpu().numpy().copy()

            # Clone for Triton
            A_tr = A.clone()
            B_tr = B.clone()
            C_tr = C.clone()
            D_tr = D.clone()
            E_tr = E.clone()
            F_tr = F.clone()
            G_tr = G.clone()

            # Run C reference
            run_c_reference(A_c, B_c, C_c, D_c, E_c, F_c, G_c, NI, NJ, NK, NL, NM)

            # Run Triton
            k3mm_triton(A_tr, B_tr, C_tr, D_tr, E_tr, F_tr, G_tr, NI, NJ, NK, NL, NM)

            # Compare output arrays
            max_error = 0.0
            max_rel_error = 0.0
            c_val = torch.from_numpy(E_c).float()
            tr_val = E_tr.cpu().float()
            abs_err = torch.max(torch.abs(c_val - tr_val)).item()
            denom = torch.max(torch.abs(c_val)).item()
            rel_err = abs_err / max(denom, 1e-10)
            max_error = max(max_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)
            c_val = torch.from_numpy(F_c).float()
            tr_val = F_tr.cpu().float()
            abs_err = torch.max(torch.abs(c_val - tr_val)).item()
            denom = torch.max(torch.abs(c_val)).item()
            rel_err = abs_err / max(denom, 1e-10)
            max_error = max(max_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)
            c_val = torch.from_numpy(G_c).float()
            tr_val = G_tr.cpu().float()
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
