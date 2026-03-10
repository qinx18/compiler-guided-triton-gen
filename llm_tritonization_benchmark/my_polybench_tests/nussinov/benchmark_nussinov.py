#!/usr/bin/env python3
"""Performance Benchmark for nussinov (Polybench)"""
import sys
import time
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    from polybench_results_scale8x.llm_triton_no_analysis.nussinov.attempt1 import nussinov_triton
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs_scale8x" / "libnussinov.so"

def run_c_reference(seq_c, table_c, N):
    lib = ctypes.CDLL(str(C_LIB_PATH))
    CType_seq = ctypes.c_int * (1440)
    c_arr_seq = CType_seq.in_dll(lib, 'seq')
    src_seq = np.ascontiguousarray(seq_c.astype(np.int32), dtype=np.int32)
    ctypes.memmove(c_arr_seq, src_seq.ctypes.data, src_seq.nbytes)
    CType_table = ctypes.c_int * (1440 * 1440)
    c_arr_table = CType_table.in_dll(lib, 'table')
    src_table = np.ascontiguousarray(table_c.astype(np.int32), dtype=np.int32)
    ctypes.memmove(c_arr_table, src_table.ctypes.data, src_table.nbytes)
    pass
    func = getattr(lib, "nussinov_kernel")
    func.argtypes = []
    func.restype = None
    func()
    CType_table = ctypes.c_int * (1440 * 1440)
    c_arr_table = CType_table.in_dll(lib, 'table')
    table_c[:] = np.frombuffer(c_arr_table, dtype=np.int32).reshape(1440, 1440).astype(np.float32).copy()

def benchmark():
    num_warmup = 5
    num_iterations = 50

    # Integer base sequence {0..3} and zero-initialized score table
    seq = torch.randint(0, 4, (1440,), device='cuda').float()
    table = torch.zeros(1440, 1440, device='cuda', dtype=torch.float32)
    N = 1440

    # C reference benchmark
    c_time = None
    try:
        for _ in range(num_warmup):
            seq_c = seq.cpu().numpy().copy()
            table_c = table.cpu().numpy().copy()
            run_c_reference(seq_c, table_c, N)
        start = time.perf_counter()
        for _ in range(num_iterations):
            seq_c = seq.cpu().numpy().copy()
            table_c = table.cpu().numpy().copy()
            run_c_reference(seq_c, table_c, N)
        c_time = (time.perf_counter() - start) / num_iterations
    except Exception as e:
        print(f"C ref error: {e}")

    # Triton benchmark
    tr_time = None
    try:
        for _ in range(num_warmup):
            seq_tr = seq.clone()
            table_tr = table.clone()
            nussinov_triton(seq_tr, table_tr, N)
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            seq_tr = seq.clone()
            table_tr = table.clone()
            nussinov_triton(seq_tr, table_tr, N)
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
