#!/usr/bin/env python3
"""Performance Benchmark for 2mm (Polybench)"""
import sys
import time
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    import importlib
    _mod = importlib.import_module("polybench_results.llm_triton.2mm.attempt1")
    k2mm_triton = _mod.k2mm_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs" / "lib2mm.so"

def run_c_reference(A_c, B_c, C_c, D_c, tmp_c, alpha, beta, NI, NJ, NK, NL):
    lib = ctypes.CDLL(str(C_LIB_PATH))
    CType_A = ctypes.c_float * (40 * 70)
    c_arr_A = CType_A.in_dll(lib, 'A')
    src_A = np.ascontiguousarray(A_c, dtype=np.float32)
    ctypes.memmove(c_arr_A, src_A.ctypes.data, src_A.nbytes)
    CType_B = ctypes.c_float * (70 * 50)
    c_arr_B = CType_B.in_dll(lib, 'B')
    src_B = np.ascontiguousarray(B_c, dtype=np.float32)
    ctypes.memmove(c_arr_B, src_B.ctypes.data, src_B.nbytes)
    CType_C = ctypes.c_float * (50 * 80)
    c_arr_C = CType_C.in_dll(lib, 'C')
    src_C = np.ascontiguousarray(C_c, dtype=np.float32)
    ctypes.memmove(c_arr_C, src_C.ctypes.data, src_C.nbytes)
    CType_D = ctypes.c_float * (40 * 80)
    c_arr_D = CType_D.in_dll(lib, 'D')
    src_D = np.ascontiguousarray(D_c, dtype=np.float32)
    ctypes.memmove(c_arr_D, src_D.ctypes.data, src_D.nbytes)
    CType_tmp = ctypes.c_float * (40 * 50)
    c_arr_tmp = CType_tmp.in_dll(lib, 'tmp')
    src_tmp = np.ascontiguousarray(tmp_c, dtype=np.float32)
    ctypes.memmove(c_arr_tmp, src_tmp.ctypes.data, src_tmp.nbytes)
    ctypes.c_float.in_dll(lib, 'alpha').value = float(alpha)
    ctypes.c_float.in_dll(lib, 'beta').value = float(beta)
    func = getattr(lib, "k2mm_kernel")
    func.argtypes = []
    func.restype = None
    func()
    CType_D = ctypes.c_float * (40 * 80)
    c_arr_D = CType_D.in_dll(lib, 'D')
    D_c[:] = np.frombuffer(c_arr_D, dtype=np.float32).reshape(40, 80).copy()
    CType_tmp = ctypes.c_float * (40 * 50)
    c_arr_tmp = CType_tmp.in_dll(lib, 'tmp')
    tmp_c[:] = np.frombuffer(c_arr_tmp, dtype=np.float32).reshape(40, 50).copy()

def benchmark():
    num_warmup = 5
    num_iterations = 50

    A = torch.randn(40, 70, device='cuda', dtype=torch.float32)
    B = torch.randn(70, 50, device='cuda', dtype=torch.float32)
    C = torch.randn(50, 80, device='cuda', dtype=torch.float32)
    D = torch.randn(40, 80, device='cuda', dtype=torch.float32)
    tmp = torch.randn(40, 50, device='cuda', dtype=torch.float32)
    alpha = 1.5
    beta = 1.5
    NI = 40
    NJ = 50
    NK = 70
    NL = 80

    # C reference benchmark
    c_time = None
    try:
        for _ in range(num_warmup):
            A_c = A.cpu().numpy().copy()
            B_c = B.cpu().numpy().copy()
            C_c = C.cpu().numpy().copy()
            D_c = D.cpu().numpy().copy()
            tmp_c = tmp.cpu().numpy().copy()
            run_c_reference(A_c, B_c, C_c, D_c, tmp_c, alpha, beta, NI, NJ, NK, NL)
        start = time.perf_counter()
        for _ in range(num_iterations):
            A_c = A.cpu().numpy().copy()
            B_c = B.cpu().numpy().copy()
            C_c = C.cpu().numpy().copy()
            D_c = D.cpu().numpy().copy()
            tmp_c = tmp.cpu().numpy().copy()
            run_c_reference(A_c, B_c, C_c, D_c, tmp_c, alpha, beta, NI, NJ, NK, NL)
        c_time = (time.perf_counter() - start) / num_iterations
    except Exception as e:
        print(f"C ref error: {e}")

    # Triton benchmark
    tr_time = None
    try:
        for _ in range(num_warmup):
            A_tr = A.clone()
            B_tr = B.clone()
            C_tr = C.clone()
            D_tr = D.clone()
            tmp_tr = tmp.clone()
            k2mm_triton(A_tr, B_tr, C_tr, D_tr, tmp_tr, alpha, beta, NI, NJ, NK, NL)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            A_tr = A.clone()
            B_tr = B.clone()
            C_tr = C.clone()
            D_tr = D.clone()
            tmp_tr = tmp.clone()
            k2mm_triton(A_tr, B_tr, C_tr, D_tr, tmp_tr, alpha, beta, NI, NJ, NK, NL)
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
