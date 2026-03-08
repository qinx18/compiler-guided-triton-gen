#!/usr/bin/env python3
"""Performance Benchmark for ludcmp (Polybench)"""
import sys
import time
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from polybench_results.llm_triton.ludcmp.attempt1 import ludcmp_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs" / "libludcmp.so"

def run_c_reference(A_c, b_c, x_c, y_c, N):
    lib = ctypes.CDLL(str(C_LIB_PATH))
    CType_A = ctypes.c_float * (120 * 120)
    c_arr_A = CType_A.in_dll(lib, 'A')
    src_A = np.ascontiguousarray(A_c, dtype=np.float32)
    ctypes.memmove(c_arr_A, src_A.ctypes.data, src_A.nbytes)
    CType_b = ctypes.c_float * (120)
    c_arr_b = CType_b.in_dll(lib, 'b')
    src_b = np.ascontiguousarray(b_c, dtype=np.float32)
    ctypes.memmove(c_arr_b, src_b.ctypes.data, src_b.nbytes)
    CType_x = ctypes.c_float * (120)
    c_arr_x = CType_x.in_dll(lib, 'x')
    src_x = np.ascontiguousarray(x_c, dtype=np.float32)
    ctypes.memmove(c_arr_x, src_x.ctypes.data, src_x.nbytes)
    CType_y = ctypes.c_float * (120)
    c_arr_y = CType_y.in_dll(lib, 'y')
    src_y = np.ascontiguousarray(y_c, dtype=np.float32)
    ctypes.memmove(c_arr_y, src_y.ctypes.data, src_y.nbytes)
    pass
    func = getattr(lib, "ludcmp_kernel")
    func.argtypes = []
    func.restype = None
    func()
    CType_x = ctypes.c_float * (120)
    c_arr_x = CType_x.in_dll(lib, 'x')
    x_c[:] = np.frombuffer(c_arr_x, dtype=np.float32).reshape(120).copy()

def benchmark():
    num_warmup = 5
    num_iterations = 50

    # Diagonally dominant for stable pivotless LU; x,y are outputs
    A = torch.randn(120, 120, device='cuda', dtype=torch.float32) + 120 * torch.eye(120, device='cuda', dtype=torch.float32)
    b = torch.randn(120, device='cuda', dtype=torch.float32)
    x = torch.zeros(120, device='cuda', dtype=torch.float32)
    y = torch.zeros(120, device='cuda', dtype=torch.float32)
    N = 120

    # C reference benchmark
    c_time = None
    try:
        for _ in range(num_warmup):
            A_c = A.cpu().numpy().copy()
            b_c = b.cpu().numpy().copy()
            x_c = x.cpu().numpy().copy()
            y_c = y.cpu().numpy().copy()
            run_c_reference(A_c, b_c, x_c, y_c, N)
        start = time.perf_counter()
        for _ in range(num_iterations):
            A_c = A.cpu().numpy().copy()
            b_c = b.cpu().numpy().copy()
            x_c = x.cpu().numpy().copy()
            y_c = y.cpu().numpy().copy()
            run_c_reference(A_c, b_c, x_c, y_c, N)
        c_time = (time.perf_counter() - start) / num_iterations
    except Exception as e:
        print(f"C ref error: {e}")

    # Triton benchmark
    tr_time = None
    try:
        for _ in range(num_warmup):
            A_tr = A.clone()
            b_tr = b.clone()
            x_tr = x.clone()
            y_tr = y.clone()
            ludcmp_triton(A_tr, b_tr, x_tr, y_tr, N)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            A_tr = A.clone()
            b_tr = b.clone()
            x_tr = x.clone()
            y_tr = y.clone()
            ludcmp_triton(A_tr, b_tr, x_tr, y_tr, N)
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
