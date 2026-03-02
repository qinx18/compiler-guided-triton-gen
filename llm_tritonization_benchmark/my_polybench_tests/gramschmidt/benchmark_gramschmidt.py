#!/usr/bin/env python3
"""Performance Benchmark for gramschmidt (Polybench)"""
import sys
import time
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from polybench_results.llm_triton.gramschmidt.attempt1 import gramschmidt_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs" / "libgramschmidt.so"

def run_c_reference(A_c, Q_c, R_c, M, N):
    lib = ctypes.CDLL(str(C_LIB_PATH))
    CType_A = ctypes.c_float * (60 * 80)
    c_arr_A = CType_A.in_dll(lib, 'A')
    src_A = np.ascontiguousarray(A_c, dtype=np.float32)
    ctypes.memmove(c_arr_A, src_A.ctypes.data, src_A.nbytes)
    CType_Q = ctypes.c_float * (60 * 80)
    c_arr_Q = CType_Q.in_dll(lib, 'Q')
    src_Q = np.ascontiguousarray(Q_c, dtype=np.float32)
    ctypes.memmove(c_arr_Q, src_Q.ctypes.data, src_Q.nbytes)
    CType_R = ctypes.c_float * (80 * 80)
    c_arr_R = CType_R.in_dll(lib, 'R')
    src_R = np.ascontiguousarray(R_c, dtype=np.float32)
    ctypes.memmove(c_arr_R, src_R.ctypes.data, src_R.nbytes)
    pass
    func = getattr(lib, "gramschmidt_kernel")
    func.argtypes = []
    func.restype = None
    func()
    CType_A = ctypes.c_float * (60 * 80)
    c_arr_A = CType_A.in_dll(lib, 'A')
    A_c[:] = np.frombuffer(c_arr_A, dtype=np.float32).reshape(60, 80).copy()
    CType_Q = ctypes.c_float * (60 * 80)
    c_arr_Q = CType_Q.in_dll(lib, 'Q')
    Q_c[:] = np.frombuffer(c_arr_Q, dtype=np.float32).reshape(60, 80).copy()
    CType_R = ctypes.c_float * (80 * 80)
    c_arr_R = CType_R.in_dll(lib, 'R')
    R_c[:] = np.frombuffer(c_arr_R, dtype=np.float32).reshape(80, 80).copy()

def benchmark():
    num_warmup = 5
    num_iterations = 50

    # Well-conditioned A with strong diagonal for stable Gram-Schmidt
    A = torch.randn(60, 80, device='cuda', dtype=torch.float32) + torch.eye(60, 80, device='cuda', dtype=torch.float32) * 5.0
    R = torch.zeros(80, 80, device='cuda', dtype=torch.float32)
    Q = torch.zeros(60, 80, device='cuda', dtype=torch.float32)
    M = 60
    N = 80

    # C reference benchmark
    c_time = None
    try:
        for _ in range(num_warmup):
            A_c = A.cpu().numpy().copy()
            Q_c = Q.cpu().numpy().copy()
            R_c = R.cpu().numpy().copy()
            run_c_reference(A_c, Q_c, R_c, M, N)
        start = time.perf_counter()
        for _ in range(num_iterations):
            A_c = A.cpu().numpy().copy()
            Q_c = Q.cpu().numpy().copy()
            R_c = R.cpu().numpy().copy()
            run_c_reference(A_c, Q_c, R_c, M, N)
        c_time = (time.perf_counter() - start) / num_iterations
    except Exception as e:
        print(f"C ref error: {e}")

    # Triton benchmark
    tr_time = None
    try:
        for _ in range(num_warmup):
            A_tr = A.clone()
            Q_tr = Q.clone()
            R_tr = R.clone()
            gramschmidt_triton(A_tr, Q_tr, R_tr, M, N)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            A_tr = A.clone()
            Q_tr = Q.clone()
            R_tr = R.clone()
            gramschmidt_triton(A_tr, Q_tr, R_tr, M, N)
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
