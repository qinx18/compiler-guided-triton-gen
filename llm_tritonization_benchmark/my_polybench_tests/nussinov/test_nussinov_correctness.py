#!/usr/bin/env python3
"""Correctness test for nussinov (Polybench) - attempt 5"""
import sys
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

# Import Triton implementation
try:
    from polybench_results.llm_triton.nussinov.attempt5 import nussinov_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Load C reference
C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs" / "libnussinov.so"
if not C_LIB_PATH.exists():
    print(f"C reference library not found: {C_LIB_PATH}")
    sys.exit(1)

def run_c_reference(seq_c, table_c, N):
    """Run C reference kernel via ctypes."""
    lib = ctypes.CDLL(str(C_LIB_PATH))

    # Set global arrays in the .so
    CType_seq = ctypes.c_int * (180)
    c_arr_seq = CType_seq.in_dll(lib, 'seq')
    src_seq = np.ascontiguousarray(seq_c.astype(np.int32), dtype=np.int32)
    ctypes.memmove(c_arr_seq, src_seq.ctypes.data, src_seq.nbytes)
    CType_table = ctypes.c_int * (180 * 180)
    c_arr_table = CType_table.in_dll(lib, 'table')
    src_table = np.ascontiguousarray(table_c.astype(np.int32), dtype=np.int32)
    ctypes.memmove(c_arr_table, src_table.ctypes.data, src_table.nbytes)

    # Set global scalars
    pass

    # Run kernel
    func = getattr(lib, "nussinov_kernel")
    func.argtypes = []
    func.restype = None
    func()

    # Read back output arrays
    CType_table = ctypes.c_int * (180 * 180)
    c_arr_table = CType_table.in_dll(lib, 'table')
    table_c[:] = np.frombuffer(c_arr_table, dtype=np.int32).reshape(180, 180).astype(np.float32).copy()

def test_correctness():
    """Test Triton vs C reference."""
    num_tests = 3
    all_passed = True

    for test_idx in range(num_tests):
        try:
            # Initialize arrays
            # Integer base sequence {0..3} and zero-initialized score table
            seq = torch.randint(0, 4, (180,), device='cuda').float()
            table = torch.zeros(180, 180, device='cuda', dtype=torch.float32)
            N = 180

            # Clone for C reference
            seq_c = seq.cpu().numpy().copy()
            table_c = table.cpu().numpy().copy()

            # Clone for Triton
            seq_tr = seq.clone()
            table_tr = table.clone()

            # Run C reference
            run_c_reference(seq_c, table_c, N)

            # Run Triton
            nussinov_triton(seq_tr, table_tr, N)

            # Compare output arrays
            max_error = 0.0
            max_rel_error = 0.0
            c_val = torch.from_numpy(table_c).float()
            tr_val = table_tr.cpu().float()
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
