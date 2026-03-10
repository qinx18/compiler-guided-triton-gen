#!/usr/bin/env python3
"""Performance Benchmark for jacobi_2d (Polybench)"""
import sys
import time
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from polybench_results_scale8x.llm_triton.jacobi_2d.attempt1 import jacobi_2d_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs_scale8x" / "libjacobi_2d.so"

def run_c_reference(A_c, B_c, N, TSTEPS):
    lib = ctypes.CDLL(str(C_LIB_PATH))
    CType_A = ctypes.c_float * (720 * 720)
    c_arr_A = CType_A.in_dll(lib, 'A')
    src_A = np.ascontiguousarray(A_c, dtype=np.float32)
    ctypes.memmove(c_arr_A, src_A.ctypes.data, src_A.nbytes)
    CType_B = ctypes.c_float * (720 * 720)
    c_arr_B = CType_B.in_dll(lib, 'B')
    src_B = np.ascontiguousarray(B_c, dtype=np.float32)
    ctypes.memmove(c_arr_B, src_B.ctypes.data, src_B.nbytes)
    pass
    func = getattr(lib, "jacobi_2d_kernel")
    func.argtypes = []
    func.restype = None
    func()
    CType_A = ctypes.c_float * (720 * 720)
    c_arr_A = CType_A.in_dll(lib, 'A')
    A_c[:] = np.frombuffer(c_arr_A, dtype=np.float32).reshape(720, 720).copy()
    CType_B = ctypes.c_float * (720 * 720)
    c_arr_B = CType_B.in_dll(lib, 'B')
    B_c[:] = np.frombuffer(c_arr_B, dtype=np.float32).reshape(720, 720).copy()

def benchmark():
    num_warmup = 5
    num_iterations = 50

    A = torch.randn(720, 720, device='cuda', dtype=torch.float32)
    B = torch.randn(720, 720, device='cuda', dtype=torch.float32)
    N = 720
    TSTEPS = 40

    # C reference benchmark
    c_time = None
    try:
        for _ in range(num_warmup):
            A_c = A.cpu().numpy().copy()
            B_c = B.cpu().numpy().copy()
            run_c_reference(A_c, B_c, N, TSTEPS)
        start = time.perf_counter()
        for _ in range(num_iterations):
            A_c = A.cpu().numpy().copy()
            B_c = B.cpu().numpy().copy()
            run_c_reference(A_c, B_c, N, TSTEPS)
        c_time = (time.perf_counter() - start) / num_iterations
    except Exception as e:
        print(f"C ref error: {e}")

    # Triton benchmark
    tr_time = None
    try:
        for _ in range(num_warmup):
            A_tr = A.clone()
            B_tr = B.clone()
            jacobi_2d_triton(A_tr, B_tr, N, TSTEPS)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            A_tr = A.clone()
            B_tr = B.clone()
            jacobi_2d_triton(A_tr, B_tr, N, TSTEPS)
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
