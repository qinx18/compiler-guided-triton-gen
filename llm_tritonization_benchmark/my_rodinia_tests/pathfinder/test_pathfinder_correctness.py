#!/usr/bin/env python3
"""Correctness test for pathfinder (Rodinia) - attempt 1"""
import sys
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

# Import Triton implementation
try:
    from rodinia_results.llm_triton.pathfinder.attempt1 import pathfinder_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Load C reference
C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "rodinia_libs" / "libpathfinder.so"
if not C_LIB_PATH.exists():
    print(f"C reference library not found: {C_LIB_PATH}")
    sys.exit(1)

def run_c_reference(dst_c, src_c, wall_c, COLS, ROWS):
    """Run C reference kernel via ctypes."""
    lib = ctypes.CDLL(str(C_LIB_PATH))

    # Set global arrays in the .so
    CType_dst = ctypes.c_float * (256)
    c_arr_dst = CType_dst.in_dll(lib, 'dst')
    src_dst = np.ascontiguousarray(dst_c, dtype=np.float32)
    ctypes.memmove(c_arr_dst, src_dst.ctypes.data, src_dst.nbytes)
    CType_src = ctypes.c_float * (256)
    c_arr_src = CType_src.in_dll(lib, 'src')
    src_src = np.ascontiguousarray(src_c, dtype=np.float32)
    ctypes.memmove(c_arr_src, src_src.ctypes.data, src_src.nbytes)
    CType_wall = ctypes.c_float * (100 * 256)
    c_arr_wall = CType_wall.in_dll(lib, 'wall')
    src_wall = np.ascontiguousarray(wall_c, dtype=np.float32)
    ctypes.memmove(c_arr_wall, src_wall.ctypes.data, src_wall.nbytes)

    # Set global scalars
    pass

    # Run kernel
    func = getattr(lib, "pathfinder_kernel")
    func.argtypes = []
    func.restype = None
    func()

    # Read back output arrays
    CType_dst = ctypes.c_float * (256)
    c_arr_dst = CType_dst.in_dll(lib, 'dst')
    dst_c[:] = np.frombuffer(c_arr_dst, dtype=np.float32).reshape(256).copy()
    CType_src = ctypes.c_float * (256)
    c_arr_src = CType_src.in_dll(lib, 'src')
    src_c[:] = np.frombuffer(c_arr_src, dtype=np.float32).reshape(256).copy()

def test_correctness():
    """Test Triton vs C reference."""
    num_tests = 3
    all_passed = True

    for test_idx in range(num_tests):
        try:
            # Initialize arrays
            # Positive weights for pathfinder
            wall = torch.abs(torch.randn(100, 256, device='cuda', dtype=torch.float32)) * 10.0 + 1.0
            src = torch.zeros(256, device='cuda', dtype=torch.float32)
            dst = torch.zeros(256, device='cuda', dtype=torch.float32)
            COLS = 256
            ROWS = 100

            # Clone for C reference
            dst_c = dst.cpu().numpy().copy()
            src_c = src.cpu().numpy().copy()
            wall_c = wall.cpu().numpy().copy()

            # Clone for Triton
            dst_tr = dst.clone()
            src_tr = src.clone()
            wall_tr = wall.clone()

            # Run C reference
            run_c_reference(dst_c, src_c, wall_c, COLS, ROWS)

            # Run Triton
            pathfinder_triton(dst_tr, src_tr, wall_tr, COLS, ROWS)

            # Compare output arrays
            max_error = 0.0
            max_rel_error = 0.0
            c_val = torch.from_numpy(dst_c).float()
            tr_val = dst_tr.cpu().float()
            abs_err = torch.max(torch.abs(c_val - tr_val)).item()
            denom = torch.max(torch.abs(c_val)).item()
            rel_err = abs_err / max(denom, 1e-10)
            max_error = max(max_error, abs_err)
            max_rel_error = max(max_rel_error, rel_err)
            c_val = torch.from_numpy(src_c).float()
            tr_val = src_tr.cpu().float()
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
