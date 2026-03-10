#!/usr/bin/env python3
"""Performance Benchmark for bicg (Polybench)"""
import sys
import time
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from polybench_results_scale8x.llm_triton_no_analysis.bicg.attempt1 import bicg_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs_scale8x" / "libbicg.so"

def run_c_reference(A_c, p_c, q_c, r_c, s_c, M, N):
    lib = ctypes.CDLL(str(C_LIB_PATH))
    CType_A = ctypes.c_float * (680 * 600)
    c_arr_A = CType_A.in_dll(lib, 'A')
    src_A = np.ascontiguousarray(A_c, dtype=np.float32)
    ctypes.memmove(c_arr_A, src_A.ctypes.data, src_A.nbytes)
    CType_p = ctypes.c_float * (600)
    c_arr_p = CType_p.in_dll(lib, 'p')
    src_p = np.ascontiguousarray(p_c, dtype=np.float32)
    ctypes.memmove(c_arr_p, src_p.ctypes.data, src_p.nbytes)
    CType_q = ctypes.c_float * (680)
    c_arr_q = CType_q.in_dll(lib, 'q')
    src_q = np.ascontiguousarray(q_c, dtype=np.float32)
    ctypes.memmove(c_arr_q, src_q.ctypes.data, src_q.nbytes)
    CType_r = ctypes.c_float * (680)
    c_arr_r = CType_r.in_dll(lib, 'r')
    src_r = np.ascontiguousarray(r_c, dtype=np.float32)
    ctypes.memmove(c_arr_r, src_r.ctypes.data, src_r.nbytes)
    CType_s = ctypes.c_float * (600)
    c_arr_s = CType_s.in_dll(lib, 's')
    src_s = np.ascontiguousarray(s_c, dtype=np.float32)
    ctypes.memmove(c_arr_s, src_s.ctypes.data, src_s.nbytes)
    pass
    func = getattr(lib, "bicg_kernel")
    func.argtypes = []
    func.restype = None
    func()
    CType_q = ctypes.c_float * (680)
    c_arr_q = CType_q.in_dll(lib, 'q')
    q_c[:] = np.frombuffer(c_arr_q, dtype=np.float32).reshape(680).copy()
    CType_s = ctypes.c_float * (600)
    c_arr_s = CType_s.in_dll(lib, 's')
    s_c[:] = np.frombuffer(c_arr_s, dtype=np.float32).reshape(600).copy()

def benchmark():
    num_warmup = 5
    num_iterations = 50

    A = torch.randn(680, 600, device='cuda', dtype=torch.float32)
    p = torch.randn(600, device='cuda', dtype=torch.float32)
    q = torch.randn(680, device='cuda', dtype=torch.float32)
    r = torch.randn(680, device='cuda', dtype=torch.float32)
    s = torch.randn(600, device='cuda', dtype=torch.float32)
    M = 600
    N = 680

    # C reference benchmark
    c_time = None
    try:
        for _ in range(num_warmup):
            A_c = A.cpu().numpy().copy()
            p_c = p.cpu().numpy().copy()
            q_c = q.cpu().numpy().copy()
            r_c = r.cpu().numpy().copy()
            s_c = s.cpu().numpy().copy()
            run_c_reference(A_c, p_c, q_c, r_c, s_c, M, N)
        start = time.perf_counter()
        for _ in range(num_iterations):
            A_c = A.cpu().numpy().copy()
            p_c = p.cpu().numpy().copy()
            q_c = q.cpu().numpy().copy()
            r_c = r.cpu().numpy().copy()
            s_c = s.cpu().numpy().copy()
            run_c_reference(A_c, p_c, q_c, r_c, s_c, M, N)
        c_time = (time.perf_counter() - start) / num_iterations
    except Exception as e:
        print(f"C ref error: {e}")

    # Triton benchmark
    tr_time = None
    try:
        for _ in range(num_warmup):
            A_tr = A.clone()
            p_tr = p.clone()
            q_tr = q.clone()
            r_tr = r.clone()
            s_tr = s.clone()
            bicg_triton(A_tr, p_tr, q_tr, r_tr, s_tr, M, N)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            A_tr = A.clone()
            p_tr = p.clone()
            q_tr = q.clone()
            r_tr = r.clone()
            s_tr = s.clone()
            bicg_triton(A_tr, p_tr, q_tr, r_tr, s_tr, M, N)
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
