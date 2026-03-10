#!/usr/bin/env python3
"""Performance Benchmark for durbin (Polybench)"""
import sys
import time
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from polybench_results_scale8x.llm_triton.durbin.attempt9 import durbin_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs_scale8x" / "libdurbin.so"

def run_c_reference(r_c, y_c, z_c, N):
    lib = ctypes.CDLL(str(C_LIB_PATH))
    CType_r = ctypes.c_float * (960)
    c_arr_r = CType_r.in_dll(lib, 'r')
    src_r = np.ascontiguousarray(r_c, dtype=np.float32)
    ctypes.memmove(c_arr_r, src_r.ctypes.data, src_r.nbytes)
    CType_y = ctypes.c_float * (960)
    c_arr_y = CType_y.in_dll(lib, 'y')
    src_y = np.ascontiguousarray(y_c, dtype=np.float32)
    ctypes.memmove(c_arr_y, src_y.ctypes.data, src_y.nbytes)
    CType_z = ctypes.c_float * (960)
    c_arr_z = CType_z.in_dll(lib, 'z')
    src_z = np.ascontiguousarray(z_c, dtype=np.float32)
    ctypes.memmove(c_arr_z, src_z.ctypes.data, src_z.nbytes)
    pass
    func = getattr(lib, "durbin_kernel")
    func.argtypes = []
    func.restype = None
    func()
    CType_y = ctypes.c_float * (960)
    c_arr_y = CType_y.in_dll(lib, 'y')
    y_c[:] = np.frombuffer(c_arr_y, dtype=np.float32).reshape(960).copy()

def benchmark():
    num_warmup = 5
    num_iterations = 50

    r = torch.randn(960, device='cuda', dtype=torch.float32)
    y = torch.randn(960, device='cuda', dtype=torch.float32)
    z = torch.randn(960, device='cuda', dtype=torch.float32)
    N = 960

    # C reference benchmark
    c_time = None
    try:
        for _ in range(num_warmup):
            r_c = r.cpu().numpy().copy()
            y_c = y.cpu().numpy().copy()
            z_c = z.cpu().numpy().copy()
            run_c_reference(r_c, y_c, z_c, N)
        start = time.perf_counter()
        for _ in range(num_iterations):
            r_c = r.cpu().numpy().copy()
            y_c = y.cpu().numpy().copy()
            z_c = z.cpu().numpy().copy()
            run_c_reference(r_c, y_c, z_c, N)
        c_time = (time.perf_counter() - start) / num_iterations
    except Exception as e:
        print(f"C ref error: {e}")

    # Triton benchmark
    tr_time = None
    try:
        for _ in range(num_warmup):
            r_tr = r.clone()
            y_tr = y.clone()
            z_tr = z.clone()
            durbin_triton(r_tr, y_tr, z_tr, N)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            r_tr = r.clone()
            y_tr = y.clone()
            z_tr = z.clone()
            durbin_triton(r_tr, y_tr, z_tr, N)
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
