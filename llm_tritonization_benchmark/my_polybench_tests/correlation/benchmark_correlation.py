#!/usr/bin/env python3
"""Performance Benchmark for correlation (Polybench)"""
import sys
import time
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from polybench_results.llm_triton.correlation.attempt2 import correlation_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs" / "libcorrelation.so"

def run_c_reference(corr_c, data_c, mean_c, stddev_c, eps, float_n, M, N):
    lib = ctypes.CDLL(str(C_LIB_PATH))
    CType_corr = ctypes.c_float * (80 * 80)
    c_arr_corr = CType_corr.in_dll(lib, 'corr')
    src_corr = np.ascontiguousarray(corr_c, dtype=np.float32)
    ctypes.memmove(c_arr_corr, src_corr.ctypes.data, src_corr.nbytes)
    CType_data = ctypes.c_float * (100 * 80)
    c_arr_data = CType_data.in_dll(lib, 'data')
    src_data = np.ascontiguousarray(data_c, dtype=np.float32)
    ctypes.memmove(c_arr_data, src_data.ctypes.data, src_data.nbytes)
    CType_mean = ctypes.c_float * (80)
    c_arr_mean = CType_mean.in_dll(lib, 'mean')
    src_mean = np.ascontiguousarray(mean_c, dtype=np.float32)
    ctypes.memmove(c_arr_mean, src_mean.ctypes.data, src_mean.nbytes)
    CType_stddev = ctypes.c_float * (80)
    c_arr_stddev = CType_stddev.in_dll(lib, 'stddev')
    src_stddev = np.ascontiguousarray(stddev_c, dtype=np.float32)
    ctypes.memmove(c_arr_stddev, src_stddev.ctypes.data, src_stddev.nbytes)
    ctypes.c_float.in_dll(lib, 'eps').value = float(eps)
    ctypes.c_float.in_dll(lib, 'float_n').value = float(float_n)
    func = getattr(lib, "correlation_kernel")
    func.argtypes = []
    func.restype = None
    func()
    CType_corr = ctypes.c_float * (80 * 80)
    c_arr_corr = CType_corr.in_dll(lib, 'corr')
    corr_c[:] = np.frombuffer(c_arr_corr, dtype=np.float32).reshape(80, 80).copy()
    CType_data = ctypes.c_float * (100 * 80)
    c_arr_data = CType_data.in_dll(lib, 'data')
    data_c[:] = np.frombuffer(c_arr_data, dtype=np.float32).reshape(100, 80).copy()
    CType_mean = ctypes.c_float * (80)
    c_arr_mean = CType_mean.in_dll(lib, 'mean')
    mean_c[:] = np.frombuffer(c_arr_mean, dtype=np.float32).reshape(80).copy()
    CType_stddev = ctypes.c_float * (80)
    c_arr_stddev = CType_stddev.in_dll(lib, 'stddev')
    stddev_c[:] = np.frombuffer(c_arr_stddev, dtype=np.float32).reshape(80).copy()

def benchmark():
    num_warmup = 5
    num_iterations = 50

    corr = torch.randn(80, 80, device='cuda', dtype=torch.float32)
    data = torch.randn(100, 80, device='cuda', dtype=torch.float32)
    mean = torch.randn(80, device='cuda', dtype=torch.float32)
    stddev = torch.randn(80, device='cuda', dtype=torch.float32)
    eps = 0.1
    float_n = float(100)
    M = 80
    N = 100

    # C reference benchmark
    c_time = None
    try:
        for _ in range(num_warmup):
            corr_c = corr.cpu().numpy().copy()
            data_c = data.cpu().numpy().copy()
            mean_c = mean.cpu().numpy().copy()
            stddev_c = stddev.cpu().numpy().copy()
            run_c_reference(corr_c, data_c, mean_c, stddev_c, eps, float_n, M, N)
        start = time.perf_counter()
        for _ in range(num_iterations):
            corr_c = corr.cpu().numpy().copy()
            data_c = data.cpu().numpy().copy()
            mean_c = mean.cpu().numpy().copy()
            stddev_c = stddev.cpu().numpy().copy()
            run_c_reference(corr_c, data_c, mean_c, stddev_c, eps, float_n, M, N)
        c_time = (time.perf_counter() - start) / num_iterations
    except Exception as e:
        print(f"C ref error: {e}")

    # Triton benchmark
    tr_time = None
    try:
        for _ in range(num_warmup):
            corr_tr = corr.clone()
            data_tr = data.clone()
            mean_tr = mean.clone()
            stddev_tr = stddev.clone()
            correlation_triton(corr_tr, data_tr, mean_tr, stddev_tr, eps, float_n, M, N)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            corr_tr = corr.clone()
            data_tr = data.clone()
            mean_tr = mean.clone()
            stddev_tr = stddev.clone()
            correlation_triton(corr_tr, data_tr, mean_tr, stddev_tr, eps, float_n, M, N)
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
