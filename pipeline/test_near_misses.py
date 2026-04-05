#!/usr/bin/env python3
"""Test near-miss kernels with detailed abs/rel error reporting."""
import sys, importlib, ctypes, numpy as np, torch, time

sys.path.insert(0, '.')
sys.path.insert(0, '/home/qinxiao/workspace/compiler-guided-triton-gen/analysis')
sys.path.append('utilities')

import polybench_functions_db
importlib.reload(polybench_functions_db)
from polybench_functions_db import POLYBENCH_FUNCTIONS

import generate_and_test_polybench as gtp
importlib.reload(gtp)

# Near misses and their best attempts
NEAR_MISSES = {
    'gemver': [5, 4, 3],      # 5e-4 on attempt 5
    'durbin': [3, 4, 5],      # 2e-3 on attempt 3
    'jacobi_1d': [2, 5, 1],   # 5e-7 on attempt 2
    'jacobi_2d': [1, 2, 3],   # ~1.1 on attempt 1
    'seidel_2d': [2, 3, 1],   # ~0.07 on attempt 2
    'covariance': [4, 2, 1],  # ~0.3 on attempt 4
}

def load_triton_fn(kernel_name, attempt):
    """Load the triton function from a specific attempt."""
    mod = importlib.import_module(f'polybench_results.llm_triton.{kernel_name}.attempt{attempt}')
    func_name = f'{kernel_name}_triton'
    if kernel_name[0].isdigit():
        func_name = f'k{kernel_name}_triton'
    return getattr(mod, func_name)

def run_c_reference(kernel_name, func_spec, array_data, params):
    """Run C reference and return output arrays as numpy."""
    arrays = func_spec['arrays']
    scalar_params = func_spec.get('scalar_params', {})
    c_data = {k: v.cpu().numpy().copy() for k, v in array_data.items()}

    lib = ctypes.CDLL(f'c_reference/polybench_libs/lib{kernel_name}.so')

    for arr_name, mode in sorted(arrays.items()):
        if mode in ['r', 'rw']:
            shape = gtp._get_array_shape(kernel_name, arr_name, params)
            if shape:
                total = 1
                for s in shape: total *= s
                CType = ctypes.c_float * total
                c_arr = CType.in_dll(lib, arr_name)
                src = np.ascontiguousarray(c_data[arr_name], dtype=np.float32)
                ctypes.memmove(c_arr, src.ctypes.data, src.nbytes)

    for sp_name in sorted(scalar_params.keys()):
        try:
            ctypes.c_float.in_dll(lib, sp_name).value = 1.5
        except: pass

    func_id = kernel_name
    if func_id[0].isdigit(): func_id = 'k' + func_id
    func = getattr(lib, f'{func_id}_kernel')
    func.argtypes = []
    func.restype = None
    func()

    for arr_name, mode in sorted(arrays.items()):
        if mode in ['rw', 'w']:
            shape = gtp._get_array_shape(kernel_name, arr_name, params)
            if shape:
                total = 1
                for s in shape: total *= s
                CType = ctypes.c_float * total
                c_arr = CType.in_dll(lib, arr_name)
                c_data[arr_name][:] = np.frombuffer(c_arr, dtype=np.float32).reshape(shape).copy()

    return c_data

def test_kernel(kernel_name, attempt, num_tests=5, threshold=1e-3):
    """Test a kernel attempt with detailed error reporting."""
    func_spec = POLYBENCH_FUNCTIONS[kernel_name]
    params = gtp.get_kernel_params(kernel_name)
    arrays = func_spec['arrays']
    scalar_params = func_spec.get('scalar_params', {})

    try:
        triton_fn = load_triton_fn(kernel_name, attempt)
    except Exception as e:
        return None, f"Import error: {e}"

    results = []
    for test_idx in range(num_tests):
        torch.manual_seed(test_idx * 42 + 7)
        np.random.seed(test_idx * 42 + 7)

        # Init arrays
        array_data = {}
        for arr_name, mode in sorted(arrays.items()):
            shape = gtp._get_array_shape(kernel_name, arr_name, params)
            if shape:
                array_data[arr_name] = torch.randn(*shape, device='cuda', dtype=torch.float32)

        # C reference
        c_data = run_c_reference(kernel_name, func_spec, array_data, params)

        # Triton
        tr_data = {k: v.clone() for k, v in array_data.items()}
        tr_args = []
        for arr_name in sorted(arrays.keys()):
            tr_args.append(tr_data[arr_name])
        for sp_name in sorted(scalar_params.keys()):
            tr_args.append(1.5)
        for p_name in sorted(p for p in params.keys() if p not in scalar_params):
            tr_args.append(params[p_name])

        try:
            triton_fn(*tr_args)
        except Exception as e:
            results.append({'abs': float('inf'), 'rel': float('inf'), 'error': str(e)})
            continue

        # Compare
        max_abs = 0.0
        max_rel = 0.0
        for arr_name, mode in sorted(arrays.items()):
            if mode in ['rw', 'w']:
                c_val = torch.from_numpy(c_data[arr_name]).float()
                tr_val = tr_data[arr_name].cpu().float()
                abs_err = torch.max(torch.abs(c_val - tr_val)).item()
                denom = torch.max(torch.abs(c_val)).item()
                rel_err = abs_err / max(denom, 1e-10)
                max_abs = max(max_abs, abs_err)
                max_rel = max(max_rel, rel_err)
        results.append({'abs': max_abs, 'rel': max_rel})

    return results, None


if __name__ == '__main__':
    print("=" * 70)
    print("Near-Miss Kernel Analysis")
    print("=" * 70)

    for kernel_name, attempts_to_try in NEAR_MISSES.items():
        print(f"\n{'='*50}")
        print(f"  {kernel_name}")
        print(f"{'='*50}")

        best_abs = float('inf')
        best_attempt = None

        for attempt in attempts_to_try:
            print(f"\n  Attempt {attempt}:")
            results, error = test_kernel(kernel_name, attempt, num_tests=5)
            if error:
                print(f"    ERROR: {error}")
                continue

            abs_errors = [r['abs'] for r in results if 'error' not in r]
            rel_errors = [r['rel'] for r in results if 'error' not in r]

            if not abs_errors:
                print(f"    All tests errored")
                continue

            max_abs = max(abs_errors)
            avg_abs = sum(abs_errors) / len(abs_errors)
            max_rel = max(rel_errors)
            avg_rel = sum(rel_errors) / len(rel_errors)

            pass_1e3 = sum(1 for e in abs_errors if e < 1e-3)
            pass_1e2 = sum(1 for e in abs_errors if e < 1e-2)
            pass_rel = sum(1 for e in rel_errors if e < 1e-2)

            for i, r in enumerate(results):
                if 'error' in r:
                    print(f"    Test {i+1}: ERROR - {r['error'][:60]}")
                else:
                    status = "PASS" if r['abs'] < 1e-3 else ("near" if r['abs'] < 1e-2 else "FAIL")
                    print(f"    Test {i+1}: abs={r['abs']:.6e}  rel={r['rel']:.6e}  [{status}]")

            print(f"    Summary: max_abs={max_abs:.6e} avg_abs={avg_abs:.6e}")
            print(f"             max_rel={max_rel:.6e} avg_rel={avg_rel:.6e}")
            print(f"    Pass@1e-3: {pass_1e3}/5  Pass@1e-2: {pass_1e2}/5  Pass@rel<1%: {pass_rel}/5")

            if max_abs < best_abs:
                best_abs = max_abs
                best_attempt = attempt

        if best_attempt:
            would_pass_1e2 = best_abs < 1e-2
            would_pass_1e1 = best_abs < 1e-1
            print(f"\n  BEST: attempt {best_attempt}, max_abs={best_abs:.6e}")
            print(f"    Would pass @ 1e-3: {'YES' if best_abs < 1e-3 else 'NO'}")
            print(f"    Would pass @ 1e-2: {'YES' if would_pass_1e2 else 'NO'}")
            print(f"    Would pass @ 1e-1: {'YES' if would_pass_1e1 else 'NO'}")
