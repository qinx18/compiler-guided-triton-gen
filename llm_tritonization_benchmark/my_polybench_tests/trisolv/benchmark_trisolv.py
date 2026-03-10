#!/usr/bin/env python3
"""Performance Benchmark for trisolv (Polybench)"""
import sys
import time
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from polybench_results_scale8x.llm_triton_no_analysis.trisolv.attempt1 import trisolv_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs_scale8x" / "libtrisolv.so"

def run_c_reference(L_c, b_c, x_c, N):
    lib = ctypes.CDLL(str(C_LIB_PATH))
    CType_L = ctypes.c_float * (960 * 960)
    c_arr_L = CType_L.in_dll(lib, 'L')
    src_L = np.ascontiguousarray(L_c, dtype=np.float32)
    ctypes.memmove(c_arr_L, src_L.ctypes.data, src_L.nbytes)
    CType_b = ctypes.c_float * (960)
    c_arr_b = CType_b.in_dll(lib, 'b')
    src_b = np.ascontiguousarray(b_c, dtype=np.float32)
    ctypes.memmove(c_arr_b, src_b.ctypes.data, src_b.nbytes)
    CType_x = ctypes.c_float * (960)
    c_arr_x = CType_x.in_dll(lib, 'x')
    src_x = np.ascontiguousarray(x_c, dtype=np.float32)
    ctypes.memmove(c_arr_x, src_x.ctypes.data, src_x.nbytes)
    pass
    func = getattr(lib, "trisolv_kernel")
    func.argtypes = []
    func.restype = None
    func()
    CType_x = ctypes.c_float * (960)
    c_arr_x = CType_x.in_dll(lib, 'x')
    x_c[:] = np.frombuffer(c_arr_x, dtype=np.float32).reshape(960).copy()

def benchmark():
    num_warmup = 5
    num_iterations = 50

    # Lower triangular with |diagonal| >= 1
    L = torch.tril(torch.randn(960, 960, device='cuda', dtype=torch.float32))
    L.diagonal().abs_().clamp_(min=1.0)
    b = torch.randn(960, device='cuda', dtype=torch.float32)
    x = torch.zeros(960, device='cuda', dtype=torch.float32)
    N = 960

    # C reference benchmark
    c_time = None
    try:
        for _ in range(num_warmup):
            L_c = L.cpu().numpy().copy()
            b_c = b.cpu().numpy().copy()
            x_c = x.cpu().numpy().copy()
            run_c_reference(L_c, b_c, x_c, N)
        start = time.perf_counter()
        for _ in range(num_iterations):
            L_c = L.cpu().numpy().copy()
            b_c = b.cpu().numpy().copy()
            x_c = x.cpu().numpy().copy()
            run_c_reference(L_c, b_c, x_c, N)
        c_time = (time.perf_counter() - start) / num_iterations
    except Exception as e:
        print(f"C ref error: {e}")

    # Triton benchmark
    tr_time = None
    try:
        for _ in range(num_warmup):
            L_tr = L.clone()
            b_tr = b.clone()
            x_tr = x.clone()
            trisolv_triton(L_tr, b_tr, x_tr, N)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            L_tr = L.clone()
            b_tr = b.clone()
            x_tr = x.clone()
            trisolv_triton(L_tr, b_tr, x_tr, N)
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
