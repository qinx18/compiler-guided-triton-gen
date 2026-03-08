#!/usr/bin/env python3
"""Performance Benchmark for pathfinder (Rodinia)"""
import sys
import time
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from rodinia_results.llm_triton.pathfinder.attempt1 import pathfinder_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "rodinia_libs" / "libpathfinder.so"

def run_c_reference(dst_c, src_c, wall_c, COLS, ROWS):
    lib = ctypes.CDLL(str(C_LIB_PATH))
    CType_dst = ctypes.c_float * (256)
    c_arr_dst = CType_dst.in_dll(lib, 'dst')
    src_dst = np.ascontiguousarray(dst_c, dtype=np.float32)
    ctypes.memmove(c_arr_dst, src_dst.ctypes.data, src_dst.nbytes)
    CType_src = ctypes.c_float * (256)
    c_arr_src = CType_src.in_dll(lib, 'src')
    src_src = np.ascontiguousarray(src_c, dtype=np.float32)
    ctypes.memmove(c_arr_src, src_src.ctypes.data, src_src.nbytes)
    CType_wall = ctypes.c_float * (100 * 256)
    c_arr_wall = CType_wall.in_dll(lib, 'wall')
    src_wall = np.ascontiguousarray(wall_c, dtype=np.float32)
    ctypes.memmove(c_arr_wall, src_wall.ctypes.data, src_wall.nbytes)
    pass
    func = getattr(lib, "pathfinder_kernel")
    func.argtypes = []
    func.restype = None
    func()
    CType_dst = ctypes.c_float * (256)
    c_arr_dst = CType_dst.in_dll(lib, 'dst')
    dst_c[:] = np.frombuffer(c_arr_dst, dtype=np.float32).reshape(256).copy()
    CType_src = ctypes.c_float * (256)
    c_arr_src = CType_src.in_dll(lib, 'src')
    src_c[:] = np.frombuffer(c_arr_src, dtype=np.float32).reshape(256).copy()

def benchmark():
    num_warmup = 5
    num_iterations = 50

    # Positive weights for pathfinder
    wall = torch.abs(torch.randn(100, 256, device='cuda', dtype=torch.float32)) * 10.0 + 1.0
    src = torch.zeros(256, device='cuda', dtype=torch.float32)
    dst = torch.zeros(256, device='cuda', dtype=torch.float32)
    COLS = 256
    ROWS = 100

    # C reference benchmark
    c_time = None
    try:
        for _ in range(num_warmup):
            dst_c = dst.cpu().numpy().copy()
            src_c = src.cpu().numpy().copy()
            wall_c = wall.cpu().numpy().copy()
            run_c_reference(dst_c, src_c, wall_c, COLS, ROWS)
        start = time.perf_counter()
        for _ in range(num_iterations):
            dst_c = dst.cpu().numpy().copy()
            src_c = src.cpu().numpy().copy()
            wall_c = wall.cpu().numpy().copy()
            run_c_reference(dst_c, src_c, wall_c, COLS, ROWS)
        c_time = (time.perf_counter() - start) / num_iterations
    except Exception as e:
        print(f"C ref error: {e}")

    # Triton benchmark
    tr_time = None
    try:
        for _ in range(num_warmup):
            dst_tr = dst.clone()
            src_tr = src.clone()
            wall_tr = wall.clone()
            pathfinder_triton(dst_tr, src_tr, wall_tr, COLS, ROWS)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            dst_tr = dst.clone()
            src_tr = src.clone()
            wall_tr = wall.clone()
            pathfinder_triton(dst_tr, src_tr, wall_tr, COLS, ROWS)
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
