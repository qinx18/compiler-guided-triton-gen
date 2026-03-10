#!/usr/bin/env python3
"""Performance Benchmark for gemver (Polybench)"""
import sys
import time
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from polybench_results_scale8x.llm_triton_no_analysis.gemver.attempt3 import gemver_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs_scale8x" / "libgemver.so"

def run_c_reference(A_c, u1_c, u2_c, v1_c, v2_c, w_c, x_c, y_c, z_c, alpha, beta, N):
    lib = ctypes.CDLL(str(C_LIB_PATH))
    CType_A = ctypes.c_float * (960 * 960)
    c_arr_A = CType_A.in_dll(lib, 'A')
    src_A = np.ascontiguousarray(A_c, dtype=np.float32)
    ctypes.memmove(c_arr_A, src_A.ctypes.data, src_A.nbytes)
    CType_u1 = ctypes.c_float * (960)
    c_arr_u1 = CType_u1.in_dll(lib, 'u1')
    src_u1 = np.ascontiguousarray(u1_c, dtype=np.float32)
    ctypes.memmove(c_arr_u1, src_u1.ctypes.data, src_u1.nbytes)
    CType_u2 = ctypes.c_float * (960)
    c_arr_u2 = CType_u2.in_dll(lib, 'u2')
    src_u2 = np.ascontiguousarray(u2_c, dtype=np.float32)
    ctypes.memmove(c_arr_u2, src_u2.ctypes.data, src_u2.nbytes)
    CType_v1 = ctypes.c_float * (960)
    c_arr_v1 = CType_v1.in_dll(lib, 'v1')
    src_v1 = np.ascontiguousarray(v1_c, dtype=np.float32)
    ctypes.memmove(c_arr_v1, src_v1.ctypes.data, src_v1.nbytes)
    CType_v2 = ctypes.c_float * (960)
    c_arr_v2 = CType_v2.in_dll(lib, 'v2')
    src_v2 = np.ascontiguousarray(v2_c, dtype=np.float32)
    ctypes.memmove(c_arr_v2, src_v2.ctypes.data, src_v2.nbytes)
    CType_w = ctypes.c_float * (960)
    c_arr_w = CType_w.in_dll(lib, 'w')
    src_w = np.ascontiguousarray(w_c, dtype=np.float32)
    ctypes.memmove(c_arr_w, src_w.ctypes.data, src_w.nbytes)
    CType_x = ctypes.c_float * (960)
    c_arr_x = CType_x.in_dll(lib, 'x')
    src_x = np.ascontiguousarray(x_c, dtype=np.float32)
    ctypes.memmove(c_arr_x, src_x.ctypes.data, src_x.nbytes)
    CType_y = ctypes.c_float * (960)
    c_arr_y = CType_y.in_dll(lib, 'y')
    src_y = np.ascontiguousarray(y_c, dtype=np.float32)
    ctypes.memmove(c_arr_y, src_y.ctypes.data, src_y.nbytes)
    CType_z = ctypes.c_float * (960)
    c_arr_z = CType_z.in_dll(lib, 'z')
    src_z = np.ascontiguousarray(z_c, dtype=np.float32)
    ctypes.memmove(c_arr_z, src_z.ctypes.data, src_z.nbytes)
    ctypes.c_float.in_dll(lib, 'alpha').value = float(alpha)
    ctypes.c_float.in_dll(lib, 'beta').value = float(beta)
    func = getattr(lib, "gemver_kernel")
    func.argtypes = []
    func.restype = None
    func()
    CType_A = ctypes.c_float * (960 * 960)
    c_arr_A = CType_A.in_dll(lib, 'A')
    A_c[:] = np.frombuffer(c_arr_A, dtype=np.float32).reshape(960, 960).copy()
    CType_w = ctypes.c_float * (960)
    c_arr_w = CType_w.in_dll(lib, 'w')
    w_c[:] = np.frombuffer(c_arr_w, dtype=np.float32).reshape(960).copy()
    CType_x = ctypes.c_float * (960)
    c_arr_x = CType_x.in_dll(lib, 'x')
    x_c[:] = np.frombuffer(c_arr_x, dtype=np.float32).reshape(960).copy()

def benchmark():
    num_warmup = 5
    num_iterations = 50

    A = torch.randn(960, 960, device='cuda', dtype=torch.float32)
    u1 = torch.randn(960, device='cuda', dtype=torch.float32)
    u2 = torch.randn(960, device='cuda', dtype=torch.float32)
    v1 = torch.randn(960, device='cuda', dtype=torch.float32)
    v2 = torch.randn(960, device='cuda', dtype=torch.float32)
    w = torch.randn(960, device='cuda', dtype=torch.float32)
    x = torch.randn(960, device='cuda', dtype=torch.float32)
    y = torch.randn(960, device='cuda', dtype=torch.float32)
    z = torch.randn(960, device='cuda', dtype=torch.float32)
    alpha = 1.5
    beta = 1.5
    N = 960

    # C reference benchmark
    c_time = None
    try:
        for _ in range(num_warmup):
            A_c = A.cpu().numpy().copy()
            u1_c = u1.cpu().numpy().copy()
            u2_c = u2.cpu().numpy().copy()
            v1_c = v1.cpu().numpy().copy()
            v2_c = v2.cpu().numpy().copy()
            w_c = w.cpu().numpy().copy()
            x_c = x.cpu().numpy().copy()
            y_c = y.cpu().numpy().copy()
            z_c = z.cpu().numpy().copy()
            run_c_reference(A_c, u1_c, u2_c, v1_c, v2_c, w_c, x_c, y_c, z_c, alpha, beta, N)
        start = time.perf_counter()
        for _ in range(num_iterations):
            A_c = A.cpu().numpy().copy()
            u1_c = u1.cpu().numpy().copy()
            u2_c = u2.cpu().numpy().copy()
            v1_c = v1.cpu().numpy().copy()
            v2_c = v2.cpu().numpy().copy()
            w_c = w.cpu().numpy().copy()
            x_c = x.cpu().numpy().copy()
            y_c = y.cpu().numpy().copy()
            z_c = z.cpu().numpy().copy()
            run_c_reference(A_c, u1_c, u2_c, v1_c, v2_c, w_c, x_c, y_c, z_c, alpha, beta, N)
        c_time = (time.perf_counter() - start) / num_iterations
    except Exception as e:
        print(f"C ref error: {e}")

    # Triton benchmark
    tr_time = None
    try:
        for _ in range(num_warmup):
            A_tr = A.clone()
            u1_tr = u1.clone()
            u2_tr = u2.clone()
            v1_tr = v1.clone()
            v2_tr = v2.clone()
            w_tr = w.clone()
            x_tr = x.clone()
            y_tr = y.clone()
            z_tr = z.clone()
            gemver_triton(A_tr, u1_tr, u2_tr, v1_tr, v2_tr, w_tr, x_tr, y_tr, z_tr, alpha, beta, N)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            A_tr = A.clone()
            u1_tr = u1.clone()
            u2_tr = u2.clone()
            v1_tr = v1.clone()
            v2_tr = v2.clone()
            w_tr = w.clone()
            x_tr = x.clone()
            y_tr = y.clone()
            z_tr = z.clone()
            gemver_triton(A_tr, u1_tr, u2_tr, v1_tr, v2_tr, w_tr, x_tr, y_tr, z_tr, alpha, beta, N)
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
