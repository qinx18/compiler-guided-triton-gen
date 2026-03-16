#!/usr/bin/env python3
"""Performance Benchmark for adi (Polybench)"""
import sys
import time
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from polybench_results_scale8x.llm_triton.adi.attempt3 import adi_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs_scale8x" / "libadi.so"

def run_c_reference(p_c, q_c, u_c, v_c, N, TSTEPS):
    lib = ctypes.CDLL(str(C_LIB_PATH))
    CType_p = ctypes.c_float * (480 * 480)
    c_arr_p = CType_p.in_dll(lib, 'p')
    src_p = np.ascontiguousarray(p_c, dtype=np.float32)
    ctypes.memmove(c_arr_p, src_p.ctypes.data, src_p.nbytes)
    CType_q = ctypes.c_float * (480 * 480)
    c_arr_q = CType_q.in_dll(lib, 'q')
    src_q = np.ascontiguousarray(q_c, dtype=np.float32)
    ctypes.memmove(c_arr_q, src_q.ctypes.data, src_q.nbytes)
    CType_u = ctypes.c_float * (480 * 480)
    c_arr_u = CType_u.in_dll(lib, 'u')
    src_u = np.ascontiguousarray(u_c, dtype=np.float32)
    ctypes.memmove(c_arr_u, src_u.ctypes.data, src_u.nbytes)
    CType_v = ctypes.c_float * (480 * 480)
    c_arr_v = CType_v.in_dll(lib, 'v')
    src_v = np.ascontiguousarray(v_c, dtype=np.float32)
    ctypes.memmove(c_arr_v, src_v.ctypes.data, src_v.nbytes)
    pass
    func = getattr(lib, "adi_kernel")
    func.argtypes = []
    func.restype = None
    func()
    CType_p = ctypes.c_float * (480 * 480)
    c_arr_p = CType_p.in_dll(lib, 'p')
    p_c[:] = np.frombuffer(c_arr_p, dtype=np.float32).reshape(480, 480).copy()
    CType_q = ctypes.c_float * (480 * 480)
    c_arr_q = CType_q.in_dll(lib, 'q')
    q_c[:] = np.frombuffer(c_arr_q, dtype=np.float32).reshape(480, 480).copy()
    CType_u = ctypes.c_float * (480 * 480)
    c_arr_u = CType_u.in_dll(lib, 'u')
    u_c[:] = np.frombuffer(c_arr_u, dtype=np.float32).reshape(480, 480).copy()
    CType_v = ctypes.c_float * (480 * 480)
    c_arr_v = CType_v.in_dll(lib, 'v')
    v_c[:] = np.frombuffer(c_arr_v, dtype=np.float32).reshape(480, 480).copy()

def benchmark():
    num_warmup = 5
    num_iterations = 50

    p = torch.randn(480, 480, device='cuda', dtype=torch.float32)
    q = torch.randn(480, 480, device='cuda', dtype=torch.float32)
    u = torch.randn(480, 480, device='cuda', dtype=torch.float32)
    v = torch.randn(480, 480, device='cuda', dtype=torch.float32)
    N = 480
    TSTEPS = 40

    # C reference benchmark
    c_time = None
    try:
        for _ in range(num_warmup):
            p_c = p.cpu().numpy().copy()
            q_c = q.cpu().numpy().copy()
            u_c = u.cpu().numpy().copy()
            v_c = v.cpu().numpy().copy()
            run_c_reference(p_c, q_c, u_c, v_c, N, TSTEPS)
        start = time.perf_counter()
        for _ in range(num_iterations):
            p_c = p.cpu().numpy().copy()
            q_c = q.cpu().numpy().copy()
            u_c = u.cpu().numpy().copy()
            v_c = v.cpu().numpy().copy()
            run_c_reference(p_c, q_c, u_c, v_c, N, TSTEPS)
        c_time = (time.perf_counter() - start) / num_iterations
    except Exception as e:
        print(f"C ref error: {e}")

    # Triton benchmark
    tr_time = None
    try:
        for _ in range(num_warmup):
            p_tr = p.clone()
            q_tr = q.clone()
            u_tr = u.clone()
            v_tr = v.clone()
            adi_triton(p_tr, q_tr, u_tr, v_tr, N, TSTEPS)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            p_tr = p.clone()
            q_tr = q.clone()
            u_tr = u.clone()
            v_tr = v.clone()
            adi_triton(p_tr, q_tr, u_tr, v_tr, N, TSTEPS)
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
