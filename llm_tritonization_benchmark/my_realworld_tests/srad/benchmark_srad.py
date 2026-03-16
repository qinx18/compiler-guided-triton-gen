#!/usr/bin/env python3
"""Performance Benchmark for srad (Real-World)"""
import sys
import time
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from realworld_results.llm_triton.srad.attempt1 import srad_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "realworld_libs" / "libsrad.so"

def run_c_reference(J_c, c_c, dE_c, dN_c, dS_c, dW_c, iN_c, iS_c, jE_c, jW_c, lambda_val, C1, C2, COLS, NITER, R1, R2, ROWS):
    lib = ctypes.CDLL(str(C_LIB_PATH))
    CType_J = ctypes.c_float * (262144)
    c_arr_J = CType_J.in_dll(lib, 'J')
    src_J = np.ascontiguousarray(J_c, dtype=np.float32)
    ctypes.memmove(c_arr_J, src_J.ctypes.data, src_J.nbytes)
    CType_c = ctypes.c_float * (262144)
    c_arr_c = CType_c.in_dll(lib, 'c')
    src_c = np.ascontiguousarray(c_c, dtype=np.float32)
    ctypes.memmove(c_arr_c, src_c.ctypes.data, src_c.nbytes)
    CType_dE = ctypes.c_float * (262144)
    c_arr_dE = CType_dE.in_dll(lib, 'dE')
    src_dE = np.ascontiguousarray(dE_c, dtype=np.float32)
    ctypes.memmove(c_arr_dE, src_dE.ctypes.data, src_dE.nbytes)
    CType_dN = ctypes.c_float * (262144)
    c_arr_dN = CType_dN.in_dll(lib, 'dN')
    src_dN = np.ascontiguousarray(dN_c, dtype=np.float32)
    ctypes.memmove(c_arr_dN, src_dN.ctypes.data, src_dN.nbytes)
    CType_dS = ctypes.c_float * (262144)
    c_arr_dS = CType_dS.in_dll(lib, 'dS')
    src_dS = np.ascontiguousarray(dS_c, dtype=np.float32)
    ctypes.memmove(c_arr_dS, src_dS.ctypes.data, src_dS.nbytes)
    CType_dW = ctypes.c_float * (262144)
    c_arr_dW = CType_dW.in_dll(lib, 'dW')
    src_dW = np.ascontiguousarray(dW_c, dtype=np.float32)
    ctypes.memmove(c_arr_dW, src_dW.ctypes.data, src_dW.nbytes)
    CType_iN = ctypes.c_int * (512)
    c_arr_iN = CType_iN.in_dll(lib, 'iN')
    src_iN = np.ascontiguousarray(iN_c.astype(np.int32), dtype=np.int32)
    ctypes.memmove(c_arr_iN, src_iN.ctypes.data, src_iN.nbytes)
    CType_iS = ctypes.c_int * (512)
    c_arr_iS = CType_iS.in_dll(lib, 'iS')
    src_iS = np.ascontiguousarray(iS_c.astype(np.int32), dtype=np.int32)
    ctypes.memmove(c_arr_iS, src_iS.ctypes.data, src_iS.nbytes)
    CType_jE = ctypes.c_int * (512)
    c_arr_jE = CType_jE.in_dll(lib, 'jE')
    src_jE = np.ascontiguousarray(jE_c.astype(np.int32), dtype=np.int32)
    ctypes.memmove(c_arr_jE, src_jE.ctypes.data, src_jE.nbytes)
    CType_jW = ctypes.c_int * (512)
    c_arr_jW = CType_jW.in_dll(lib, 'jW')
    src_jW = np.ascontiguousarray(jW_c.astype(np.int32), dtype=np.int32)
    ctypes.memmove(c_arr_jW, src_jW.ctypes.data, src_jW.nbytes)
    ctypes.c_float.in_dll(lib, 'lambda_val').value = float(lambda_val)
    func = getattr(lib, "srad_kernel")
    func.argtypes = []
    func.restype = None
    func()
    CType_J = ctypes.c_float * (262144)
    c_arr_J = CType_J.in_dll(lib, 'J')
    J_c[:] = np.frombuffer(c_arr_J, dtype=np.float32).reshape(262144).copy()
    CType_c = ctypes.c_float * (262144)
    c_arr_c = CType_c.in_dll(lib, 'c')
    c_c[:] = np.frombuffer(c_arr_c, dtype=np.float32).reshape(262144).copy()
    CType_dE = ctypes.c_float * (262144)
    c_arr_dE = CType_dE.in_dll(lib, 'dE')
    dE_c[:] = np.frombuffer(c_arr_dE, dtype=np.float32).reshape(262144).copy()
    CType_dN = ctypes.c_float * (262144)
    c_arr_dN = CType_dN.in_dll(lib, 'dN')
    dN_c[:] = np.frombuffer(c_arr_dN, dtype=np.float32).reshape(262144).copy()
    CType_dS = ctypes.c_float * (262144)
    c_arr_dS = CType_dS.in_dll(lib, 'dS')
    dS_c[:] = np.frombuffer(c_arr_dS, dtype=np.float32).reshape(262144).copy()
    CType_dW = ctypes.c_float * (262144)
    c_arr_dW = CType_dW.in_dll(lib, 'dW')
    dW_c[:] = np.frombuffer(c_arr_dW, dtype=np.float32).reshape(262144).copy()

def benchmark():
    num_warmup = 5
    num_iterations = 50

    # J: positive image values (exp of random)
    J = torch.exp(torch.randn(262144, device='cuda', dtype=torch.float32).clamp(-3, 3))
    # Boundary index arrays (clamp-to-edge)
    iN = torch.arange(512, device='cuda', dtype=torch.int32)
    iN[0] = 0
    iN[1:] = torch.arange(0, 511, device='cuda', dtype=torch.int32)
    iS = torch.arange(512, device='cuda', dtype=torch.int32)
    iS[:-1] = torch.arange(1, 512, device='cuda', dtype=torch.int32)
    iS[-1] = 511
    jW = torch.arange(512, device='cuda', dtype=torch.int32)
    jW[0] = 0
    jW[1:] = torch.arange(0, 511, device='cuda', dtype=torch.int32)
    jE = torch.arange(512, device='cuda', dtype=torch.int32)
    jE[:-1] = torch.arange(1, 512, device='cuda', dtype=torch.int32)
    jE[-1] = 511
    # Work arrays initialized to zero
    c = torch.zeros(262144, device='cuda', dtype=torch.float32)
    dN = torch.zeros(262144, device='cuda', dtype=torch.float32)
    dS = torch.zeros(262144, device='cuda', dtype=torch.float32)
    dW = torch.zeros(262144, device='cuda', dtype=torch.float32)
    dE = torch.zeros(262144, device='cuda', dtype=torch.float32)
    lambda_val = 0.5
    lambda_val = 0.5
    C1 = 0
    C2 = 127
    COLS = 512
    NITER = 10
    R1 = 0
    R2 = 127
    ROWS = 512

    # C reference benchmark
    c_time = None
    try:
        for _ in range(num_warmup):
            J_c = J.cpu().numpy().copy()
            c_c = c.cpu().numpy().copy()
            dE_c = dE.cpu().numpy().copy()
            dN_c = dN.cpu().numpy().copy()
            dS_c = dS.cpu().numpy().copy()
            dW_c = dW.cpu().numpy().copy()
            iN_c = iN.cpu().numpy().copy()
            iS_c = iS.cpu().numpy().copy()
            jE_c = jE.cpu().numpy().copy()
            jW_c = jW.cpu().numpy().copy()
            run_c_reference(J_c, c_c, dE_c, dN_c, dS_c, dW_c, iN_c, iS_c, jE_c, jW_c, lambda_val, C1, C2, COLS, NITER, R1, R2, ROWS)
        start = time.perf_counter()
        for _ in range(num_iterations):
            J_c = J.cpu().numpy().copy()
            c_c = c.cpu().numpy().copy()
            dE_c = dE.cpu().numpy().copy()
            dN_c = dN.cpu().numpy().copy()
            dS_c = dS.cpu().numpy().copy()
            dW_c = dW.cpu().numpy().copy()
            iN_c = iN.cpu().numpy().copy()
            iS_c = iS.cpu().numpy().copy()
            jE_c = jE.cpu().numpy().copy()
            jW_c = jW.cpu().numpy().copy()
            run_c_reference(J_c, c_c, dE_c, dN_c, dS_c, dW_c, iN_c, iS_c, jE_c, jW_c, lambda_val, C1, C2, COLS, NITER, R1, R2, ROWS)
        c_time = (time.perf_counter() - start) / num_iterations
    except Exception as e:
        print(f"C ref error: {e}")

    # Triton benchmark
    tr_time = None
    try:
        for _ in range(num_warmup):
            J_tr = J.clone()
            c_tr = c.clone()
            dE_tr = dE.clone()
            dN_tr = dN.clone()
            dS_tr = dS.clone()
            dW_tr = dW.clone()
            iN_tr = iN.clone()
            iS_tr = iS.clone()
            jE_tr = jE.clone()
            jW_tr = jW.clone()
            srad_triton(J_tr, c_tr, dE_tr, dN_tr, dS_tr, dW_tr, iN_tr, iS_tr, jE_tr, jW_tr, lambda_val, C1, C2, COLS, NITER, R1, R2, ROWS)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            J_tr = J.clone()
            c_tr = c.clone()
            dE_tr = dE.clone()
            dN_tr = dN.clone()
            dS_tr = dS.clone()
            dW_tr = dW.clone()
            iN_tr = iN.clone()
            iS_tr = iS.clone()
            jE_tr = jE.clone()
            jW_tr = jW.clone()
            srad_triton(J_tr, c_tr, dE_tr, dN_tr, dS_tr, dW_tr, iN_tr, iS_tr, jE_tr, jW_tr, lambda_val, C1, C2, COLS, NITER, R1, R2, ROWS)
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
