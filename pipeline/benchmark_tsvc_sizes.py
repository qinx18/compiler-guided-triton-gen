#!/usr/bin/env python3
"""
Benchmark TSVC kernels at multiple data sizes with single-threaded and OpenMP C reference.
Uses subprocess with timeout to prevent hangs on serial kernels.

Usage:
    python benchmark_tsvc_sizes.py                          # 32KB + 32MB
    python benchmark_tsvc_sizes.py --all-sizes              # 32KB + 32MB + 3200MB
    python benchmark_tsvc_sizes.py --size 32MB              # 32MB only
    python benchmark_tsvc_sizes.py --size 32MB s000 s1161   # Specific kernels
"""
import sys
import os
import json
import subprocess
import time
import statistics
from pathlib import Path

ROOT = Path(__file__).parent

SIZE_CONFIGS = {
    '32KB':   {'1d': 32_000,       '2d': 256},
    '32MB':   {'1d': 8_000_000,    '2d': 2_828},
    '3200MB': {'1d': 800_000_000,  '2d': 28_284},
}

# Kernels known to have loop-carried deps (skip at large sizes — both CPU/GPU very slow)
SERIAL_KERNELS = {
    's111', 's1111', 's112', 's1112', 's113', 's1113',
    's114', 's115', 's116', 's118', 's119',
    's121', 's122', 's123', 's124', 's125', 's126', 's127', 's128',
    's131', 's132', 's141',
    's151', 's152', 's161', 's162',
    's171', 's172', 's173', 's174', 's175', 's176',
    's211', 's212',
    's221', 's222', 's231', 's232', 's233', 's234', 's235',
    's241', 's242', 's243', 's244',
    's251', 's252', 's253', 's254', 's255', 's256', 's257', 's258',
    's261', 's271', 's272', 's273', 's274', 's275', 's276', 's277', 's278', 's279',
    's281', 's291', 's292', 's293',
    's311', 's312', 's313', 's314', 's315', 's316', 's317', 's318', 's319',
    's321', 's322', 's323',
    's331', 's332', 's341', 's342', 's343',
    's351', 's352', 's353',
    's411', 's412', 's413', 's414', 's415', 's421', 's422', 's423', 's424',
    's431', 's441', 's442', 's443',
    's451', 's452', 's453',
    's471', 's481', 's482', 's491',
}

WORKER_SCRIPT = '''
import sys, os, time, json, ctypes, importlib, numpy as np
sys.path.insert(0, "{root}")
import torch
os.environ['OMP_NUM_THREADS'] = '{omp_threads}'

func_name = sys.argv[1]
N = int(sys.argv[2])
has_2d = sys.argv[3] == '1'
use_omp = sys.argv[4] == '1'

sys.path.append("{root}/utilities")
from tsvc_functions_db import TSVC_FUNCTIONS

# Load libraries
lib_st = ctypes.CDLL("{root}/c_reference/libtsvc_all.so")
lib_omp = None
if use_omp and os.path.exists("{root}/c_reference/libtsvc_all_omp.so"):
    lib_omp = ctypes.CDLL("{root}/c_reference/libtsvc_all_omp.so")
# OMP GPU loaded in separate process (conflicts with torch CUDA)
lib_omp_gpu = None

# Setup function signatures
from c_reference.tsvc_all_reference import _setup_functions
from c_reference import tsvc_all_reference as ref
for name in dir(ref._lib):
    if name.endswith('_kernel'):
        try:
            src = getattr(ref._lib, name)
            for lib in [lib_st] + ([lib_omp] if lib_omp else []):
                dst = getattr(lib, name)
                dst.argtypes = src.argtypes
                dst.restype = src.restype
        except: pass

func_spec = TSVC_FUNCTIONS.get(func_name, {{}})
arrays = func_spec.get('arrays', {{}})
scalars = func_spec.get('scalar_params', {{}})

# Create arrays
np_arrays = {{}}
for arr_name, mode in arrays.items():
    if arr_name in ('aa', 'bb', 'cc', 'tt'):
        np_arrays[arr_name] = np.random.randn(N, N).astype(np.float32)
    elif arr_name == 'flat_2d_array':
        np_arrays[arr_name] = np.random.randn(N * N).astype(np.float32)
    elif arr_name == 'indx':
        np_arrays[arr_name] = np.random.randint(0, max(1, N//2), size=N).astype(np.int32)
    else:
        np_arrays[arr_name] = np.random.randn(N).astype(np.float32)

def bench_c(lib):
    try:
        c_func = getattr(lib, f'{{func_name}}_kernel')
    except: return None
    args = []
    for arr_name in np_arrays:
        arr = np_arrays[arr_name]
        if arr.dtype == np.int32:
            args.append(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
        elif arr.ndim == 2:
            flat = np.ascontiguousarray(arr.flatten())
            args.append(flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        else:
            args.append(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    args.append(ctypes.c_int(N if not has_2d else N*N))
    if has_2d: args.append(ctypes.c_int(N))
    # Single test
    t0 = time.perf_counter()
    c_func(*args)
    t1 = time.perf_counter() - t0
    if t1 > 5: return t1 * 1000
    # Warmup + bench
    for _ in range(3): c_func(*args)
    iters = min(20, max(3, int(5.0 / max(t1, 1e-6))))
    s = time.perf_counter()
    for _ in range(iters): c_func(*args)
    return (time.perf_counter() - s) / iters * 1000

def bench_triton():
    results_file = "{root}/test29/results.json"
    if not os.path.exists(results_file): return None
    results = json.load(open(results_file))
    fr = results.get(func_name, {{}})
    if not fr.get('test_passed'): return None
    attempt = fr.get('final_attempt', fr.get('attempts', 1))
    try:
        mod = importlib.import_module(f'test29.llm_triton.{{func_name}}.attempt{{attempt}}')
        triton_func = getattr(mod, f'{{func_name}}_triton')
    except: return None
    gpu = {{k: torch.from_numpy(v.copy()).cuda() for k, v in np_arrays.items()}}
    import inspect
    sig = inspect.signature(triton_func)
    kwargs = {{}}
    for p in sig.parameters:
        if p in gpu: kwargs[p] = gpu[p]
        elif p in scalars: kwargs[p] = scalars[p]
        elif p in ('N', 'n'): kwargs[p] = N
        elif p == 'abs': kwargs[p] = scalars.get('abs', 0)
    for _ in range(3):
        try: triton_func(**kwargs)
        except: return None
    torch.cuda.synchronize()
    torch.cuda.synchronize()
    s = time.perf_counter()
    iters = 10
    for _ in range(iters): triton_func(**kwargs)
    torch.cuda.synchronize()
    return (time.perf_counter() - s) / iters * 1000

c_st = bench_c(lib_st)
c_omp = bench_c(lib_omp) if lib_omp else None
tr = bench_triton()

result = {{"c_st_ms": c_st, "c_omp_ms": c_omp, "triton_ms": tr}}
print("RESULT:" + json.dumps(result))
'''


OMP_GPU_SCRIPT = '''
import ctypes, numpy as np, time, os, sys, json
os.environ['LD_LIBRARY_PATH'] = '/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/compilers/lib:/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
sys.path.insert(0, "{root}")
sys.path.append("{root}/utilities")
from tsvc_functions_db import TSVC_FUNCTIONS
from c_reference import tsvc_all_reference as ref

func_name = sys.argv[1]
N = int(sys.argv[2])
has_2d = sys.argv[3] == '1'

lib_gpu = ctypes.CDLL("{root}/c_reference/libtsvc_all_omp_gpu.so")
for name in dir(ref._lib):
    if name.endswith('_kernel'):
        try:
            src = getattr(ref._lib, name)
            dst = getattr(lib_gpu, name)
            dst.argtypes = src.argtypes
            dst.restype = src.restype
        except: pass

func_spec = TSVC_FUNCTIONS.get(func_name, {{}})
arrays = func_spec.get('arrays', {{}})
np_arrays = {{}}
for arr_name in arrays:
    if arr_name in ('aa','bb','cc','tt'): np_arrays[arr_name] = np.random.randn(N,N).astype(np.float32)
    elif arr_name == 'flat_2d_array': np_arrays[arr_name] = np.random.randn(N*N).astype(np.float32)
    elif arr_name == 'indx': np_arrays[arr_name] = np.random.randint(0,max(1,N//2),size=N).astype(np.int32)
    else: np_arrays[arr_name] = np.random.randn(N).astype(np.float32)

try:
    c_func = getattr(lib_gpu, f'{{func_name}}_kernel')
except: print("RESULT_GPU:null"); sys.exit(0)

args = []
for arr_name in np_arrays:
    arr = np_arrays[arr_name]
    if arr.dtype == np.int32: args.append(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
    elif arr.ndim == 2: args.append(np.ascontiguousarray(arr.flatten()).ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    else: args.append(arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
args.append(ctypes.c_int(N if not has_2d else N*N))
if has_2d: args.append(ctypes.c_int(N))

try:
    c_func(*args)  # warmup (includes device init)
    c_func(*args)  # second warmup
    t0 = time.perf_counter()
    c_func(*args)
    t1 = time.perf_counter() - t0
    if t1 > 5: print(f"RESULT_GPU:{{t1*1000}}"); sys.exit(0)
    iters = min(10, max(3, int(5.0/max(t1,1e-6))))
    s = time.perf_counter()
    for _ in range(iters): c_func(*args)
    ms = (time.perf_counter()-s)/iters*1000
    print(f"RESULT_GPU:{{ms}}")
except Exception as e:
    print(f"RESULT_GPU:null")
'''


def run_kernel_benchmark(func_name, N, has_2d, use_omp, omp_threads, timeout=60):
    """Run a single kernel benchmark in a subprocess with timeout."""
    script = WORKER_SCRIPT.format(root=str(ROOT), omp_threads=str(omp_threads))
    try:
        result = subprocess.run(
            [sys.executable, '-c', script, func_name, str(N),
             '1' if has_2d else '0', '1' if use_omp else '0'],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(ROOT)
        )
        for line in result.stdout.strip().split('\n'):
            if line.startswith('RESULT:'):
                return json.loads(line[7:])
    except subprocess.TimeoutExpired:
        return {'c_st_ms': None, 'c_omp_ms': None, 'triton_ms': None, 'timeout': True}
    except Exception as e:
        return {'c_st_ms': None, 'c_omp_ms': None, 'triton_ms': None, 'error': str(e)}
    return {'c_st_ms': None, 'c_omp_ms': None, 'triton_ms': None}


def run_omp_gpu_benchmark(func_name, N, has_2d, timeout=90):
    """Run OMP GPU benchmark as standalone executable (avoids CUDA conflicts)."""
    exe = ROOT / 'c_reference' / 'omp_gpu_bench' / f'bench_{func_name}'
    if not exe.exists():
        return None
    env = {**os.environ,
           'LD_LIBRARY_PATH': '/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/compilers/lib:'
                               '/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/lib64:'
                               + os.environ.get('LD_LIBRARY_PATH', '')}
    args = [str(exe), str(N)]
    if has_2d:
        args.append(str(N))  # len_2d
    try:
        result = subprocess.run(args, capture_output=True, text=True,
                                timeout=timeout, env=env)
        for line in result.stdout.strip().split('\n'):
            if line.startswith('TIME_MS:'):
                val = line[8:]
                return float(val) if val != 'null' else None
    except (subprocess.TimeoutExpired, Exception):
        return None
    return None


def main():
    sizes_to_test = ['32KB', '32MB']
    args = sys.argv[1:]

    if '--all-sizes' in args:
        args.remove('--all-sizes')
        sizes_to_test = ['32KB', '32MB', '3200MB']
    if '--size' in args:
        idx = args.index('--size')
        sizes_to_test = [args[idx + 1]]
        args.pop(idx); args.pop(idx)

    kernel_names = args if args else None

    import multiprocessing
    n_threads = multiprocessing.cpu_count()
    os.environ['OMP_NUM_THREADS'] = str(n_threads)
    # NVIDIA HPC SDK libraries for OMP GPU offloading
    nvhpc_lib = '/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/compilers/lib'
    nvhpc_cuda = '/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/lib64'
    os.environ['LD_LIBRARY_PATH'] = f"{nvhpc_lib}:{nvhpc_cuda}:" + os.environ.get('LD_LIBRARY_PATH', '')
    omp_avail = (ROOT / 'c_reference' / 'libtsvc_all_omp.so').exists()
    omp_gpu_avail = (ROOT / 'c_reference' / 'libtsvc_all_omp_gpu.so').exists()

    print(f"CPU: {n_threads} threads | OMP CPU: {'yes' if omp_avail else 'no'} | OMP GPU: {'yes' if omp_gpu_avail else 'no'}")

    # Get passed kernels
    results = json.load(open(ROOT / 'test29' / 'results.json'))
    passed = [k for k, v in results.items() if v.get('test_passed')]
    if kernel_names:
        passed = [k for k in kernel_names if k in passed]

    # Import TSVC DB for array info
    sys.path.append(str(ROOT / 'utilities'))
    from tsvc_functions_db import TSVC_FUNCTIONS

    all_results = {}

    for size_label in sizes_to_test:
        cfg = SIZE_CONFIGS[size_label]
        print(f"\n{'='*90}")
        print(f"Data size: {size_label} (1D: N={cfg['1d']:,}, 2D: N={cfg['2d']:,})")
        print(f"{'='*90}")

        size_results = {}
        benchmarked = 0

        for i, fn in enumerate(passed):
            spec = TSVC_FUNCTIONS.get(fn, {})
            has_2d = any(a in spec.get('arrays', {}) for a in ('aa', 'bb', 'cc', 'tt'))
            N = cfg['2d'] if has_2d else cfg['1d']

            # Skip at large sizes
            if size_label in ('32MB', '3200MB') and fn in SERIAL_KERNELS:
                continue
            if size_label in ('32MB', '3200MB') and has_2d:
                continue
            if size_label == '3200MB':
                arr_count = len(spec.get('arrays', {}))
                mem_gb = arr_count * N * 4 / (1024**3)
                if mem_gb > 20:
                    print(f"  [{i+1}/{len(passed)}] {fn}: SKIP ({mem_gb:.0f}GB)")
                    continue

            print(f"  [{i+1}/{len(passed)}] {fn} (N={N:,})...", end='', flush=True)

            timeout = 45 if size_label == '32KB' else 90
            r = run_kernel_benchmark(fn, N, has_2d, omp_avail, n_threads, timeout)

            c_st = r.get('c_st_ms')
            c_omp = r.get('c_omp_ms')
            tr = r.get('triton_ms')

            # OMP GPU in separate process (CUDA conflict with torch)
            c_omp_gpu = None
            if omp_gpu_avail and fn not in SERIAL_KERNELS:
                c_omp_gpu = run_omp_gpu_benchmark(fn, N, has_2d, timeout)

            parts = []
            if r.get('timeout'):
                parts.append('TIMEOUT')
            else:
                if c_st is not None: parts.append(f"C_ST={c_st:.3f}ms")
                if c_omp is not None: parts.append(f"C_OMP={c_omp:.3f}ms")
                if c_omp_gpu is not None: parts.append(f"OMP_GPU={c_omp_gpu:.3f}ms")
                if tr is not None: parts.append(f"Triton={tr:.3f}ms")
                if c_st and tr and tr > 0: parts.append(f"Tri/ST={c_st/tr:.2f}x")
                if c_omp_gpu and tr and tr > 0: parts.append(f"Tri/OMP_GPU={c_omp_gpu/tr:.2f}x")

            print(f"  {' | '.join(parts)}")

            size_results[fn] = {
                'c_st_ms': c_st, 'c_omp_ms': c_omp, 'c_omp_gpu_ms': c_omp_gpu,
                'triton_ms': tr, 'N': N,
                'speedup_vs_st': c_st / tr if c_st and tr and tr > 0 else None,
                'speedup_vs_omp': c_omp / tr if c_omp and tr and tr > 0 else None,
                'speedup_vs_omp_gpu': c_omp_gpu / tr if c_omp_gpu and tr and tr > 0 else None,
            }
            benchmarked += 1

        all_results[size_label] = size_results

        # Summary
        valid_st = [v['speedup_vs_st'] for v in size_results.values()
                    if v.get('speedup_vs_st') and v['speedup_vs_st'] > 0.001]
        valid_omp = [v['speedup_vs_omp'] for v in size_results.values()
                     if v.get('speedup_vs_omp') and v['speedup_vs_omp'] > 0.001]
        valid_omp_gpu = [v['speedup_vs_omp_gpu'] for v in size_results.values()
                         if v.get('speedup_vs_omp_gpu') and v['speedup_vs_omp_gpu'] > 0.001]

        print(f"\n  --- {size_label} Summary ---")
        print(f"  Benchmarked: {benchmarked} kernels (of {len(passed)} passed)")
        if valid_st:
            print(f"  Triton vs C single-thread: "
                  f"median={statistics.median(valid_st):.2f}x, "
                  f"mean={statistics.mean(valid_st):.2f}x, "
                  f">1x: {sum(1 for s in valid_st if s > 1)}/{len(valid_st)}")
        if valid_omp:
            print(f"  Triton vs C OpenMP CPU ({n_threads}t): "
                  f"median={statistics.median(valid_omp):.2f}x, "
                  f"mean={statistics.mean(valid_omp):.2f}x, "
                  f">1x: {sum(1 for s in valid_omp if s > 1)}/{len(valid_omp)}")
        if valid_omp_gpu:
            print(f"  Triton vs OMP GPU offload: "
                  f"median={statistics.median(valid_omp_gpu):.2f}x, "
                  f"mean={statistics.mean(valid_omp_gpu):.2f}x, "
                  f">1x: {sum(1 for s in valid_omp_gpu if s > 1)}/{len(valid_omp_gpu)}")

    # Save
    out = ROOT / 'test29' / 'benchmark_sizes_results.json'
    json.dump(all_results, open(out, 'w'), indent=2)
    print(f"\nResults saved to {out}")


if __name__ == '__main__':
    main()
