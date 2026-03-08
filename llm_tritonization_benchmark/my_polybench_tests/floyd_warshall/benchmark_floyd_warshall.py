#!/usr/bin/env python3
"""Performance Benchmark for floyd_warshall (Polybench)"""
import sys
import time
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from polybench_results.llm_triton.floyd_warshall.attempt1 import floyd_warshall_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs" / "libfloyd_warshall.so"

def run_c_reference(path_c, N):
    lib = ctypes.CDLL(str(C_LIB_PATH))
    CType_path = ctypes.c_float * (120 * 120)
    c_arr_path = CType_path.in_dll(lib, 'path')
    src_path = np.ascontiguousarray(path_c, dtype=np.float32)
    ctypes.memmove(c_arr_path, src_path.ctypes.data, src_path.nbytes)
    pass
    func = getattr(lib, "floyd_warshall_kernel")
    func.argtypes = []
    func.restype = None
    func()
    CType_path = ctypes.c_float * (120 * 120)
    c_arr_path = CType_path.in_dll(lib, 'path')
    path_c[:] = np.frombuffer(c_arr_path, dtype=np.float32).reshape(120, 120).copy()

def benchmark():
    num_warmup = 5
    num_iterations = 50

    # Non-negative edge weights for shortest-path
    path = torch.abs(torch.randn(120, 120, device='cuda', dtype=torch.float32)) * 10.0 + 1.0
    N = 120

    # C reference benchmark
    c_time = None
    try:
        for _ in range(num_warmup):
            path_c = path.cpu().numpy().copy()
            run_c_reference(path_c, N)
        start = time.perf_counter()
        for _ in range(num_iterations):
            path_c = path.cpu().numpy().copy()
            run_c_reference(path_c, N)
        c_time = (time.perf_counter() - start) / num_iterations
    except Exception as e:
        print(f"C ref error: {e}")

    # Triton benchmark
    tr_time = None
    try:
        for _ in range(num_warmup):
            path_tr = path.clone()
            floyd_warshall_triton(path_tr, N)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            path_tr = path.clone()
            floyd_warshall_triton(path_tr, N)
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
