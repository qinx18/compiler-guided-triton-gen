#!/usr/bin/env python3
"""Performance Benchmark for deriche (Polybench)"""
import sys
import time
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from polybench_results.llm_triton.deriche.attempt3 import deriche_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs" / "libderiche.so"

def run_c_reference(imgIn_c, imgOut_c, y2_c, yy1_c, alpha, H, W):
    lib = ctypes.CDLL(str(C_LIB_PATH))
    CType_imgIn = ctypes.c_float * (192 * 128)
    c_arr_imgIn = CType_imgIn.in_dll(lib, 'imgIn')
    src_imgIn = np.ascontiguousarray(imgIn_c, dtype=np.float32)
    ctypes.memmove(c_arr_imgIn, src_imgIn.ctypes.data, src_imgIn.nbytes)
    CType_imgOut = ctypes.c_float * (192 * 128)
    c_arr_imgOut = CType_imgOut.in_dll(lib, 'imgOut')
    src_imgOut = np.ascontiguousarray(imgOut_c, dtype=np.float32)
    ctypes.memmove(c_arr_imgOut, src_imgOut.ctypes.data, src_imgOut.nbytes)
    CType_y2 = ctypes.c_float * (192 * 128)
    c_arr_y2 = CType_y2.in_dll(lib, 'y2')
    src_y2 = np.ascontiguousarray(y2_c, dtype=np.float32)
    ctypes.memmove(c_arr_y2, src_y2.ctypes.data, src_y2.nbytes)
    CType_yy1 = ctypes.c_float * (192 * 128)
    c_arr_yy1 = CType_yy1.in_dll(lib, 'yy1')
    src_yy1 = np.ascontiguousarray(yy1_c, dtype=np.float32)
    ctypes.memmove(c_arr_yy1, src_yy1.ctypes.data, src_yy1.nbytes)
    ctypes.c_float.in_dll(lib, 'alpha').value = float(alpha)
    func = getattr(lib, "deriche_kernel")
    func.argtypes = []
    func.restype = None
    func()
    CType_imgOut = ctypes.c_float * (192 * 128)
    c_arr_imgOut = CType_imgOut.in_dll(lib, 'imgOut')
    imgOut_c[:] = np.frombuffer(c_arr_imgOut, dtype=np.float32).reshape(192, 128).copy()
    CType_y2 = ctypes.c_float * (192 * 128)
    c_arr_y2 = CType_y2.in_dll(lib, 'y2')
    y2_c[:] = np.frombuffer(c_arr_y2, dtype=np.float32).reshape(192, 128).copy()
    CType_yy1 = ctypes.c_float * (192 * 128)
    c_arr_yy1 = CType_yy1.in_dll(lib, 'yy1')
    yy1_c[:] = np.frombuffer(c_arr_yy1, dtype=np.float32).reshape(192, 128).copy()

def benchmark():
    num_warmup = 5
    num_iterations = 50

    imgIn = torch.randn(192, 128, device='cuda', dtype=torch.float32)
    imgOut = torch.randn(192, 128, device='cuda', dtype=torch.float32)
    y2 = torch.randn(192, 128, device='cuda', dtype=torch.float32)
    yy1 = torch.randn(192, 128, device='cuda', dtype=torch.float32)
    alpha = 1.5
    H = 128
    W = 192

    # C reference benchmark
    c_time = None
    try:
        for _ in range(num_warmup):
            imgIn_c = imgIn.cpu().numpy().copy()
            imgOut_c = imgOut.cpu().numpy().copy()
            y2_c = y2.cpu().numpy().copy()
            yy1_c = yy1.cpu().numpy().copy()
            run_c_reference(imgIn_c, imgOut_c, y2_c, yy1_c, alpha, H, W)
        start = time.perf_counter()
        for _ in range(num_iterations):
            imgIn_c = imgIn.cpu().numpy().copy()
            imgOut_c = imgOut.cpu().numpy().copy()
            y2_c = y2.cpu().numpy().copy()
            yy1_c = yy1.cpu().numpy().copy()
            run_c_reference(imgIn_c, imgOut_c, y2_c, yy1_c, alpha, H, W)
        c_time = (time.perf_counter() - start) / num_iterations
    except Exception as e:
        print(f"C ref error: {e}")

    # Triton benchmark
    tr_time = None
    try:
        for _ in range(num_warmup):
            imgIn_tr = imgIn.clone()
            imgOut_tr = imgOut.clone()
            y2_tr = y2.clone()
            yy1_tr = yy1.clone()
            deriche_triton(imgIn_tr, imgOut_tr, y2_tr, yy1_tr, alpha, H, W)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            imgIn_tr = imgIn.clone()
            imgOut_tr = imgOut.clone()
            y2_tr = y2.clone()
            yy1_tr = yy1.clone()
            deriche_triton(imgIn_tr, imgOut_tr, y2_tr, yy1_tr, alpha, H, W)
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
