#!/usr/bin/env python3
"""Performance Benchmark for mvt (Polybench)"""
import sys
import time
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from polybench_results.llm_triton.mvt.attempt1 import mvt_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs" / "libmvt.so"

def run_c_reference(A_c, x1_c, x2_c, y_1_c, y_2_c, N):
    lib = ctypes.CDLL(str(C_LIB_PATH))
    CType_A = ctypes.c_float * (120 * 120)
    c_arr_A = CType_A.in_dll(lib, 'A')
    src_A = np.ascontiguousarray(A_c, dtype=np.float32)
    ctypes.memmove(c_arr_A, src_A.ctypes.data, src_A.nbytes)
    CType_x1 = ctypes.c_float * (120)
    c_arr_x1 = CType_x1.in_dll(lib, 'x1')
    src_x1 = np.ascontiguousarray(x1_c, dtype=np.float32)
    ctypes.memmove(c_arr_x1, src_x1.ctypes.data, src_x1.nbytes)
    CType_x2 = ctypes.c_float * (120)
    c_arr_x2 = CType_x2.in_dll(lib, 'x2')
    src_x2 = np.ascontiguousarray(x2_c, dtype=np.float32)
    ctypes.memmove(c_arr_x2, src_x2.ctypes.data, src_x2.nbytes)
    CType_y_1 = ctypes.c_float * (120)
    c_arr_y_1 = CType_y_1.in_dll(lib, 'y_1')
    src_y_1 = np.ascontiguousarray(y_1_c, dtype=np.float32)
    ctypes.memmove(c_arr_y_1, src_y_1.ctypes.data, src_y_1.nbytes)
    CType_y_2 = ctypes.c_float * (120)
    c_arr_y_2 = CType_y_2.in_dll(lib, 'y_2')
    src_y_2 = np.ascontiguousarray(y_2_c, dtype=np.float32)
    ctypes.memmove(c_arr_y_2, src_y_2.ctypes.data, src_y_2.nbytes)
    pass
    func = getattr(lib, "mvt_kernel")
    func.argtypes = []
    func.restype = None
    func()
    CType_x1 = ctypes.c_float * (120)
    c_arr_x1 = CType_x1.in_dll(lib, 'x1')
    x1_c[:] = np.frombuffer(c_arr_x1, dtype=np.float32).reshape(120).copy()
    CType_x2 = ctypes.c_float * (120)
    c_arr_x2 = CType_x2.in_dll(lib, 'x2')
    x2_c[:] = np.frombuffer(c_arr_x2, dtype=np.float32).reshape(120).copy()

def benchmark():
    num_warmup = 5
    num_iterations = 50

    A = torch.randn(120, 120, device='cuda', dtype=torch.float32)
    x1 = torch.randn(120, device='cuda', dtype=torch.float32)
    x2 = torch.randn(120, device='cuda', dtype=torch.float32)
    y_1 = torch.randn(120, device='cuda', dtype=torch.float32)
    y_2 = torch.randn(120, device='cuda', dtype=torch.float32)
    N = 120

    # C reference benchmark
    c_time = None
    try:
        for _ in range(num_warmup):
            A_c = A.cpu().numpy().copy()
            x1_c = x1.cpu().numpy().copy()
            x2_c = x2.cpu().numpy().copy()
            y_1_c = y_1.cpu().numpy().copy()
            y_2_c = y_2.cpu().numpy().copy()
            run_c_reference(A_c, x1_c, x2_c, y_1_c, y_2_c, N)
        start = time.perf_counter()
        for _ in range(num_iterations):
            A_c = A.cpu().numpy().copy()
            x1_c = x1.cpu().numpy().copy()
            x2_c = x2.cpu().numpy().copy()
            y_1_c = y_1.cpu().numpy().copy()
            y_2_c = y_2.cpu().numpy().copy()
            run_c_reference(A_c, x1_c, x2_c, y_1_c, y_2_c, N)
        c_time = (time.perf_counter() - start) / num_iterations
    except Exception as e:
        print(f"C ref error: {e}")

    # Triton benchmark
    tr_time = None
    try:
        for _ in range(num_warmup):
            A_tr = A.clone()
            x1_tr = x1.clone()
            x2_tr = x2.clone()
            y_1_tr = y_1.clone()
            y_2_tr = y_2.clone()
            mvt_triton(A_tr, x1_tr, x2_tr, y_1_tr, y_2_tr, N)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            A_tr = A.clone()
            x1_tr = x1.clone()
            x2_tr = x2.clone()
            y_1_tr = y_1.clone()
            y_2_tr = y_2.clone()
            mvt_triton(A_tr, x1_tr, x2_tr, y_1_tr, y_2_tr, N)
        torch.cuda.synchronize()
        tr_time = (time.perf_counter() - start) / num_iterations
    except Exception as e:
        print(f"Triton error: {e}")

    # Report
    speedup = c_time / tr_time if c_time and tr_time and tr_time > 0 else None
    c_ms = c_time * 1000 if c_time else -1
    tr_ms = tr_time * 1000 if tr_time else -1
    sp = speedup if speedup else -1

    print(f"C ref:   {c_ms:8.3f} ms")
    print(f"Triton:  {tr_ms:8.3f} ms")
    if speedup:
        print(f"Speedup: {speedup:8.2f}x")
    else:
        print(f"Speedup: N/A")
    print(f"BENCHMARK_RESULT:{c_ms:.6f},{tr_ms:.6f},{sp:.6f}")

if __name__ == "__main__":
    benchmark()
