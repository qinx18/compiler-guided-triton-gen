#!/usr/bin/env python3
"""Performance Benchmark for fdtd_2d (Polybench)"""
import sys
import time
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from polybench_results_scale8x.llm_triton_no_analysis.fdtd_2d.attempt1 import fdtd_2d_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs_scale8x" / "libfdtd_2d.so"

def run_c_reference(_fict__c, ex_c, ey_c, hz_c, NX, NY, TMAX):
    lib = ctypes.CDLL(str(C_LIB_PATH))
    CType__fict_ = ctypes.c_float * (20)
    c_arr__fict_ = CType__fict_.in_dll(lib, '_fict_')
    src__fict_ = np.ascontiguousarray(_fict__c, dtype=np.float32)
    ctypes.memmove(c_arr__fict_, src__fict_.ctypes.data, src__fict_.nbytes)
    CType_ex = ctypes.c_float * (480 * 640)
    c_arr_ex = CType_ex.in_dll(lib, 'ex')
    src_ex = np.ascontiguousarray(ex_c, dtype=np.float32)
    ctypes.memmove(c_arr_ex, src_ex.ctypes.data, src_ex.nbytes)
    CType_ey = ctypes.c_float * (480 * 640)
    c_arr_ey = CType_ey.in_dll(lib, 'ey')
    src_ey = np.ascontiguousarray(ey_c, dtype=np.float32)
    ctypes.memmove(c_arr_ey, src_ey.ctypes.data, src_ey.nbytes)
    CType_hz = ctypes.c_float * (480 * 640)
    c_arr_hz = CType_hz.in_dll(lib, 'hz')
    src_hz = np.ascontiguousarray(hz_c, dtype=np.float32)
    ctypes.memmove(c_arr_hz, src_hz.ctypes.data, src_hz.nbytes)
    pass
    func = getattr(lib, "fdtd_2d_kernel")
    func.argtypes = []
    func.restype = None
    func()
    CType_ex = ctypes.c_float * (480 * 640)
    c_arr_ex = CType_ex.in_dll(lib, 'ex')
    ex_c[:] = np.frombuffer(c_arr_ex, dtype=np.float32).reshape(480, 640).copy()
    CType_ey = ctypes.c_float * (480 * 640)
    c_arr_ey = CType_ey.in_dll(lib, 'ey')
    ey_c[:] = np.frombuffer(c_arr_ey, dtype=np.float32).reshape(480, 640).copy()
    CType_hz = ctypes.c_float * (480 * 640)
    c_arr_hz = CType_hz.in_dll(lib, 'hz')
    hz_c[:] = np.frombuffer(c_arr_hz, dtype=np.float32).reshape(480, 640).copy()

def benchmark():
    num_warmup = 5
    num_iterations = 50

    _fict_ = torch.randn(20, device='cuda', dtype=torch.float32)
    ex = torch.randn(480, 640, device='cuda', dtype=torch.float32)
    ey = torch.randn(480, 640, device='cuda', dtype=torch.float32)
    hz = torch.randn(480, 640, device='cuda', dtype=torch.float32)
    NX = 480
    NY = 640
    TMAX = 20

    # C reference benchmark
    c_time = None
    try:
        for _ in range(num_warmup):
            _fict__c = _fict_.cpu().numpy().copy()
            ex_c = ex.cpu().numpy().copy()
            ey_c = ey.cpu().numpy().copy()
            hz_c = hz.cpu().numpy().copy()
            run_c_reference(_fict__c, ex_c, ey_c, hz_c, NX, NY, TMAX)
        start = time.perf_counter()
        for _ in range(num_iterations):
            _fict__c = _fict_.cpu().numpy().copy()
            ex_c = ex.cpu().numpy().copy()
            ey_c = ey.cpu().numpy().copy()
            hz_c = hz.cpu().numpy().copy()
            run_c_reference(_fict__c, ex_c, ey_c, hz_c, NX, NY, TMAX)
        c_time = (time.perf_counter() - start) / num_iterations
    except Exception as e:
        print(f"C ref error: {e}")

    # Triton benchmark
    tr_time = None
    try:
        for _ in range(num_warmup):
            _fict__tr = _fict_.clone()
            ex_tr = ex.clone()
            ey_tr = ey.clone()
            hz_tr = hz.clone()
            fdtd_2d_triton(_fict__tr, ex_tr, ey_tr, hz_tr, NX, NY, TMAX)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            _fict__tr = _fict_.clone()
            ex_tr = ex.clone()
            ey_tr = ey.clone()
            hz_tr = hz.clone()
            fdtd_2d_triton(_fict__tr, ex_tr, ey_tr, hz_tr, NX, NY, TMAX)
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
