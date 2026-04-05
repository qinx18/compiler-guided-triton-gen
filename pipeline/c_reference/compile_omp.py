#!/usr/bin/env python3
"""
Compile TSVC and Polybench C references with OpenMP support.

Usage:
    python compile_omp.py              # Compile TSVC with OpenMP
    python compile_omp.py --threads 40 # Use 40 threads
"""
import subprocess
import sys
import re
from pathlib import Path

# Kernels with known loop-carried dependencies (do NOT add OMP pragma)
# These have recurrences like a[i] = a[i-1] + ..., cross-iteration deps, etc.
TSVC_SERIAL_KERNELS = {
    's111', 's1111', 's112', 's1112', 's113', 's1113',
    's114', 's115', 's116', 's118', 's119',
    's121', 's122', 's123', 's124', 's125', 's126', 's127', 's128',
    's131', 's132',
    's141',
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
    # Helper functions
    's151s', 's152s', 's471s',
}


def add_omp_pragmas(source: str) -> str:
    """Add #pragma omp parallel for to parallelizable kernel functions."""
    lines = source.split('\n')
    result = ['#include <omp.h>']
    i = 0
    while i < len(lines):
        line = lines[i]
        # Detect kernel function: void <name>_kernel(...)
        m = re.match(r'^void\s+(\w+)_kernel\s*\(', line)
        if m:
            kernel_name = m.group(1)
            is_parallel = kernel_name not in TSVC_SERIAL_KERNELS
            result.append(line)
            i += 1
            # Find the first for-loop inside the function body
            brace_depth = 0
            pragma_added = False
            while i < len(lines):
                line = lines[i]
                stripped = line.strip()
                brace_depth += stripped.count('{') - stripped.count('}')
                if is_parallel and not pragma_added and stripped.startswith('for'):
                    indent = line[:len(line) - len(line.lstrip())]
                    result.append(f'{indent}#pragma omp parallel for')
                    pragma_added = True
                result.append(line)
                i += 1
                if brace_depth <= 0:
                    break
        else:
            result.append(line)
            i += 1
    return '\n'.join(result)


def main():
    threads = 80
    if '--threads' in sys.argv:
        idx = sys.argv.index('--threads')
        threads = int(sys.argv[idx + 1])

    c_dir = Path(__file__).parent
    src = c_dir / 'tsvc_all_kernels.c'
    omp_src = c_dir / 'tsvc_all_kernels_omp.c'
    omp_so = c_dir / 'libtsvc_all_omp.so'

    print(f"Adding OpenMP pragmas to parallelizable kernels...")
    source = src.read_text()
    omp_source = add_omp_pragmas(source)
    omp_src.write_text(omp_source)

    # Count pragmas added
    n_pragmas = omp_source.count('#pragma omp parallel for')
    print(f"  Added {n_pragmas} OpenMP parallel for pragmas")

    print(f"Compiling with OpenMP ({threads} threads)...")
    cmd = ['gcc', '-O2', '-fopenmp', '-march=native', '-fPIC', '-shared',
           '-o', str(omp_so), str(omp_src), '-lm']
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Compilation failed:\n{result.stderr}")
        sys.exit(1)

    print(f"  Compiled: {omp_so}")
    print(f"\nTo use: set OMP_NUM_THREADS={threads} and load libtsvc_all_omp.so")
    print(f"  Or run benchmarks with: python generate_and_test.py --omp")


if __name__ == '__main__':
    main()
