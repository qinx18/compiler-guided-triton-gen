#!/usr/bin/env python3
"""Performance Benchmark for gaussian (Real-World)"""
import sys
import time
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from realworld_results.llm_triton.gaussian.attempt1 import gaussian_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "realworld_libs" / "libgaussian.so"

def run_c_reference(a_c, b_c, m_c, x_c, N):
    lib = ctypes.CDLL(str(C_LIB_PATH))
    CType_a = ctypes.c_float * (262144)
    c_arr_a = CType_a.in_dll(lib, 'a')
    src_a = np.ascontiguousarray(a_c, dtype=np.float32)
    ctypes.memmove(c_arr_a, src_a.ctypes.data, src_a.nbytes)
    CType_b = ctypes.c_float * (512)
    c_arr_b = CType_b.in_dll(lib, 'b')
    src_b = np.ascontiguousarray(b_c, dtype=np.float32)
    ctypes.memmove(c_arr_b, src_b.ctypes.data, src_b.nbytes)
    CType_m = ctypes.c_float * (262144)
    c_arr_m = CType_m.in_dll(lib, 'm')
    src_m = np.ascontiguousarray(m_c, dtype=np.float32)
    ctypes.memmove(c_arr_m, src_m.ctypes.data, src_m.nbytes)
    pass
    func = getattr(lib, "gaussian_kernel")
    func.argtypes = []
    func.restype = None
    func()
    CType_a = ctypes.c_float * (262144)
    c_arr_a = CType_a.in_dll(lib, 'a')
    a_c[:] = np.frombuffer(c_arr_a, dtype=np.float32).reshape(262144).copy()
    CType_b = ctypes.c_float * (512)
    c_arr_b = CType_b.in_dll(lib, 'b')
    b_c[:] = np.frombuffer(c_arr_b, dtype=np.float32).reshape(512).copy()
    CType_m = ctypes.c_float * (262144)
    c_arr_m = CType_m.in_dll(lib, 'm')
    m_c[:] = np.frombuffer(c_arr_m, dtype=np.float32).reshape(262144).copy()
    CType_x = ctypes.c_float * (512)
    c_arr_x = CType_x.in_dll(lib, 'x')
    x_c[:] = np.frombuffer(c_arr_x, dtype=np.float32).reshape(512).copy()

def benchmark():
    num_warmup = 5
    num_iterations = 50

    # Diagonally dominant matrix for stability (no pivoting)
    a = torch.randn(262144, device='cuda', dtype=torch.float32)
    # Make diag dominant
    for i_idx in range(512):
        a[i_idx * 512 + i_idx] = abs(a[i_idx * 512 + i_idx].item()) + 512
    m = torch.zeros(262144, device='cuda', dtype=torch.float32)
    b = torch.randn(512, device='cuda', dtype=torch.float32)
    x = torch.zeros(512, device='cuda', dtype=torch.float32)
    N = 512

    # C reference benchmark
    c_time = None
    try:
        for _ in range(num_warmup):
            a_c = a.cpu().numpy().copy()
            b_c = b.cpu().numpy().copy()
            m_c = m.cpu().numpy().copy()
            x_c = x.cpu().numpy().copy()
            run_c_reference(a_c, b_c, m_c, x_c, N)
        start = time.perf_counter()
        for _ in range(num_iterations):
            a_c = a.cpu().numpy().copy()
            b_c = b.cpu().numpy().copy()
            m_c = m.cpu().numpy().copy()
            x_c = x.cpu().numpy().copy()
            run_c_reference(a_c, b_c, m_c, x_c, N)
        c_time = (time.perf_counter() - start) / num_iterations
    except Exception as e:
        print(f"C ref error: {e}")

    # Triton benchmark
    tr_time = None
    try:
        for _ in range(num_warmup):
            a_tr = a.clone()
            b_tr = b.clone()
            m_tr = m.clone()
            x_tr = x.clone()
            gaussian_triton(a_tr, b_tr, m_tr, x_tr, N)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            a_tr = a.clone()
            b_tr = b.clone()
            m_tr = m.clone()
            x_tr = x.clone()
            gaussian_triton(a_tr, b_tr, m_tr, x_tr, N)
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
