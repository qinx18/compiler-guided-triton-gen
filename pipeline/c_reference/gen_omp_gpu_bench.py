#!/usr/bin/env python3
"""
Generate and compile per-kernel OMP GPU benchmark executables.
Each executable: ./bench_<name> <N> → prints TIME_MS:<ms>
"""
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'utilities'))
from tsvc_functions_db import TSVC_FUNCTIONS

NVC = "/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/compilers/bin/nvc"
C_DIR = Path(__file__).parent
BENCH_DIR = C_DIR / 'omp_gpu_bench'
BENCH_DIR.mkdir(exist_ok=True)

# Same serial list
from compile_omp import TSVC_SERIAL_KERNELS

def gen_bench_c(func_name, func_spec):
    """Generate a standalone C benchmark for one kernel with OMP GPU."""
    arrays = func_spec.get('arrays', {})
    arr_names = list(arrays.keys())
    n_arrs = len(arr_names)

    # Build kernel source from the main file
    # Read the OMP GPU source and extract just this kernel
    omp_src = (C_DIR / 'tsvc_all_kernels_omp_gpu.c').read_text()

    # Find the kernel function
    import re
    pattern = rf'/\* {func_name} \*/\n(void {func_name}_kernel\(.*?\n\}})'
    m = re.search(pattern, omp_src, re.DOTALL)
    if not m:
        return None
    kernel_code = m.group(0)

    # Determine parameter list
    sig_match = re.search(rf'void {func_name}_kernel\((.*?)\)', kernel_code, re.DOTALL)
    if not sig_match:
        return None
    params_str = sig_match.group(1)
    params = [p.strip() for p in params_str.split(',')]

    # Build call args
    call_args = []
    has_int_arr = False
    for p in params:
        p = p.strip()
        if 'int*' in p or 'int *' in p:
            call_args.append('indx')
            has_int_arr = True
        elif 'real_t*' in p or 'real_t *' in p:
            # Map to a, b, c, d, e in order
            idx = len([x for x in call_args if x in ('a','b','c','d','e')])
            call_args.append(['a','b','c','d','e'][min(idx, 4)])
        elif 'int' in p:
            if 'len_2d' in p:
                call_args.append('len_2d')
            else:
                call_args.append('N')

    bench_c = f'''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

typedef float real_t;

{kernel_code}

int main(int argc, char **argv) {{
    int N = atoi(argv[1]);
    int len_2d = (argc > 2) ? atoi(argv[2]) : 0;

    real_t *a = (real_t*)calloc(N, sizeof(real_t));
    real_t *b = (real_t*)calloc(N, sizeof(real_t));
    real_t *c = (real_t*)calloc(N, sizeof(real_t));
    real_t *d = (real_t*)calloc(N, sizeof(real_t));
    real_t *e = (real_t*)calloc(N, sizeof(real_t));
    int *indx = (int*)calloc(N, sizeof(int));

    srand(42);
    for (int i = 0; i < N; i++) {{
        a[i] = (real_t)rand()/RAND_MAX;
        b[i] = (real_t)rand()/RAND_MAX;
        c[i] = (real_t)rand()/RAND_MAX;
        d[i] = (real_t)rand()/RAND_MAX;
        e[i] = (real_t)rand()/RAND_MAX;
        indx[i] = rand() % (N > 1 ? N/2 : 1);
    }}

    /* Warmup */
    {func_name}_kernel({', '.join(call_args)});
    {func_name}_kernel({', '.join(call_args)});

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int r = 0; r < 10; r++)
        {func_name}_kernel({', '.join(call_args)});
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double ms = ((t1.tv_sec - t0.tv_sec)*1e9 + (t1.tv_nsec - t0.tv_nsec))/10.0/1e6;
    printf("TIME_MS:%.6f\\n", ms);

    free(a); free(b); free(c); free(d); free(e); free(indx);
    return 0;
}}
'''
    return bench_c


def main():
    compiled = 0
    failed = 0

    for func_name, spec in sorted(TSVC_FUNCTIONS.items()):
        if func_name in TSVC_SERIAL_KERNELS:
            continue

        c_code = gen_bench_c(func_name, spec)
        if c_code is None:
            continue

        c_file = BENCH_DIR / f'bench_{func_name}.c'
        exe_file = BENCH_DIR / f'bench_{func_name}'
        c_file.write_text(c_code)

        result = subprocess.run(
            [NVC, '-mp=gpu', '-gpu=cc86,mem:managed', '-O2', '-o', str(exe_file), str(c_file), '-lm'],
            capture_output=True, text=True, timeout=30
        )

        if result.returncode == 0:
            compiled += 1
        else:
            failed += 1
            print(f"  FAIL: {func_name}: {result.stderr[:100]}")

    print(f"Compiled: {compiled}, Failed: {failed}")
    print(f"Executables in: {BENCH_DIR}/")


if __name__ == '__main__':
    main()
