#!/usr/bin/env python3
"""Performance Benchmark for doitgen (Polybench)"""
import sys
import time
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from polybench_results.llm_triton.doitgen.attempt1 import doitgen_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs" / "libdoitgen.so"

def run_c_reference(A_c, C4_c, sum_c, NP, NQ, NR):
    lib = ctypes.CDLL(str(C_LIB_PATH))
    CType_A = ctypes.c_float * (25 * 20 * 30)
    c_arr_A = CType_A.in_dll(lib, 'A')
    src_A = np.ascontiguousarray(A_c, dtype=np.float32)
    ctypes.memmove(c_arr_A, src_A.ctypes.data, src_A.nbytes)
    CType_C4 = ctypes.c_float * (30 * 30)
    c_arr_C4 = CType_C4.in_dll(lib, 'C4')
    src_C4 = np.ascontiguousarray(C4_c, dtype=np.float32)
    ctypes.memmove(c_arr_C4, src_C4.ctypes.data, src_C4.nbytes)
    CType_sum = ctypes.c_float * (30)
    c_arr_sum = CType_sum.in_dll(lib, 'sum')
    src_sum = np.ascontiguousarray(sum_c, dtype=np.float32)
    ctypes.memmove(c_arr_sum, src_sum.ctypes.data, src_sum.nbytes)
    pass
    func = getattr(lib, "doitgen_kernel")
    func.argtypes = []
    func.restype = None
    func()
    CType_A = ctypes.c_float * (25 * 20 * 30)
    c_arr_A = CType_A.in_dll(lib, 'A')
    A_c[:] = np.frombuffer(c_arr_A, dtype=np.float32).reshape(25, 20, 30).copy()

def benchmark():
    num_warmup = 5
    num_iterations = 50

    A = torch.randn(25, 20, 30, device='cuda', dtype=torch.float32)
    C4 = torch.randn(30, 30, device='cuda', dtype=torch.float32)
    sum = torch.randn(30, device='cuda', dtype=torch.float32)
    NP = 30
    NQ = 20
    NR = 25

    # C reference benchmark
    c_time = None
    try:
        for _ in range(num_warmup):
            A_c = A.cpu().numpy().copy()
            C4_c = C4.cpu().numpy().copy()
            sum_c = sum.cpu().numpy().copy()
            run_c_reference(A_c, C4_c, sum_c, NP, NQ, NR)
        start = time.perf_counter()
        for _ in range(num_iterations):
            A_c = A.cpu().numpy().copy()
            C4_c = C4.cpu().numpy().copy()
            sum_c = sum.cpu().numpy().copy()
            run_c_reference(A_c, C4_c, sum_c, NP, NQ, NR)
        c_time = (time.perf_counter() - start) / num_iterations
    except Exception as e:
        print(f"C ref error: {e}")

    # Triton benchmark
    tr_time = None
    try:
        for _ in range(num_warmup):
            A_tr = A.clone()
            C4_tr = C4.clone()
            sum_tr = sum.clone()
            doitgen_triton(A_tr, C4_tr, sum_tr, NP, NQ, NR)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            A_tr = A.clone()
            C4_tr = C4.clone()
            sum_tr = sum.clone()
            doitgen_triton(A_tr, C4_tr, sum_tr, NP, NQ, NR)
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
