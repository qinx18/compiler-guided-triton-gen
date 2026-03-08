#!/usr/bin/env python3
"""Performance Benchmark for hotspot (Rodinia)"""
import sys
import time
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from rodinia_results.llm_triton.hotspot.attempt1 import hotspot_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "rodinia_libs" / "libhotspot.so"

def run_c_reference(power_c, result_c, temp_c, Cap_1, Rx_1, Ry_1, Rz_1, amb_temp, COLS, ROWS, TSTEPS):
    lib = ctypes.CDLL(str(C_LIB_PATH))
    CType_power = ctypes.c_float * (256 * 256)
    c_arr_power = CType_power.in_dll(lib, 'power')
    src_power = np.ascontiguousarray(power_c, dtype=np.float32)
    ctypes.memmove(c_arr_power, src_power.ctypes.data, src_power.nbytes)
    CType_result = ctypes.c_float * (256 * 256)
    c_arr_result = CType_result.in_dll(lib, 'result')
    src_result = np.ascontiguousarray(result_c, dtype=np.float32)
    ctypes.memmove(c_arr_result, src_result.ctypes.data, src_result.nbytes)
    CType_temp = ctypes.c_float * (256 * 256)
    c_arr_temp = CType_temp.in_dll(lib, 'temp')
    src_temp = np.ascontiguousarray(temp_c, dtype=np.float32)
    ctypes.memmove(c_arr_temp, src_temp.ctypes.data, src_temp.nbytes)
    ctypes.c_float.in_dll(lib, 'Cap_1').value = float(Cap_1)
    ctypes.c_float.in_dll(lib, 'Rx_1').value = float(Rx_1)
    ctypes.c_float.in_dll(lib, 'Ry_1').value = float(Ry_1)
    ctypes.c_float.in_dll(lib, 'Rz_1').value = float(Rz_1)
    ctypes.c_float.in_dll(lib, 'amb_temp').value = float(amb_temp)
    func = getattr(lib, "hotspot_kernel")
    func.argtypes = []
    func.restype = None
    func()
    CType_result = ctypes.c_float * (256 * 256)
    c_arr_result = CType_result.in_dll(lib, 'result')
    result_c[:] = np.frombuffer(c_arr_result, dtype=np.float32).reshape(256, 256).copy()
    CType_temp = ctypes.c_float * (256 * 256)
    c_arr_temp = CType_temp.in_dll(lib, 'temp')
    temp_c[:] = np.frombuffer(c_arr_temp, dtype=np.float32).reshape(256, 256).copy()

def benchmark():
    num_warmup = 5
    num_iterations = 50

    # Realistic temperatures ~300K with variation
    temp = torch.randn(256, 256, device='cuda', dtype=torch.float32) * 50.0 + 300.0
    power = torch.abs(torch.randn(256, 256, device='cuda', dtype=torch.float32)) * 0.5
    result = torch.zeros(256, 256, device='cuda', dtype=torch.float32)
    Cap_1 = 0.0002
    Rx_1 = 51200.0
    Ry_1 = 51200.0
    Rz_1 = 320000.0
    amb_temp = 80.0
    COLS = 256
    ROWS = 256
    TSTEPS = 10

    # C reference benchmark
    c_time = None
    try:
        for _ in range(num_warmup):
            power_c = power.cpu().numpy().copy()
            result_c = result.cpu().numpy().copy()
            temp_c = temp.cpu().numpy().copy()
            run_c_reference(power_c, result_c, temp_c, Cap_1, Rx_1, Ry_1, Rz_1, amb_temp, COLS, ROWS, TSTEPS)
        start = time.perf_counter()
        for _ in range(num_iterations):
            power_c = power.cpu().numpy().copy()
            result_c = result.cpu().numpy().copy()
            temp_c = temp.cpu().numpy().copy()
            run_c_reference(power_c, result_c, temp_c, Cap_1, Rx_1, Ry_1, Rz_1, amb_temp, COLS, ROWS, TSTEPS)
        c_time = (time.perf_counter() - start) / num_iterations
    except Exception as e:
        print(f"C ref error: {e}")

    # Triton benchmark
    tr_time = None
    try:
        for _ in range(num_warmup):
            power_tr = power.clone()
            result_tr = result.clone()
            temp_tr = temp.clone()
            hotspot_triton(power_tr, result_tr, temp_tr, Cap_1, Rx_1, Ry_1, Rz_1, amb_temp, COLS, ROWS, TSTEPS)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            power_tr = power.clone()
            result_tr = result.clone()
            temp_tr = temp.clone()
            hotspot_triton(power_tr, result_tr, temp_tr, Cap_1, Rx_1, Ry_1, Rz_1, amb_temp, COLS, ROWS, TSTEPS)
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
