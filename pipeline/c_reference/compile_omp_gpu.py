#!/usr/bin/env python3
"""
Compile TSVC C reference with OpenMP GPU offloading using NVIDIA nvc compiler.
Generates libtsvc_all_omp_gpu.so with #pragma omp target on parallelizable kernels.

Usage:
    python compile_omp_gpu.py
"""
import subprocess
import sys
import re
from pathlib import Path

NVC = "/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/compilers/bin/nvc"

# Same serial kernels as compile_omp.py
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


def add_omp_gpu_pragmas(source: str) -> str:
    """Add #pragma omp target teams distribute parallel for to parallelizable kernels."""
    lines = source.split('\n')
    result = ['#include <omp.h>']
    i = 0
    n_pragmas = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'^void\s+(\w+)_kernel\s*\(', line)
        if m:
            kernel_name = m.group(1)
            is_parallel = kernel_name not in SERIAL_KERNELS
            result.append(line)
            i += 1
            brace_depth = 0
            pragma_added = False
            # Collect all parameter names from the function signature
            while i < len(lines):
                line = lines[i]
                stripped = line.strip()
                brace_depth += stripped.count('{') - stripped.count('}')

                if is_parallel and not pragma_added and stripped.startswith('for'):
                    indent = line[:len(line) - len(line.lstrip())]
                    # For GPU offloading, we need map clauses
                    # Since these are pointer args, we use is_device_ptr or map
                    result.append(f'{indent}#pragma omp target teams distribute parallel for')
                    pragma_added = True
                    n_pragmas += 1

                result.append(line)
                i += 1
                if brace_depth <= 0:
                    break
        else:
            result.append(line)
            i += 1
    print(f"  Added {n_pragmas} OMP target GPU pragmas")
    return '\n'.join(result)


def main():
    c_dir = Path(__file__).parent
    src = c_dir / 'tsvc_all_kernels.c'
    gpu_src = c_dir / 'tsvc_all_kernels_omp_gpu.c'
    gpu_so = c_dir / 'libtsvc_all_omp_gpu.so'

    if not Path(NVC).exists():
        print(f"ERROR: nvc not found at {NVC}")
        print("Install NVIDIA HPC SDK: sudo apt-get install nvhpc-25-5")
        sys.exit(1)

    print("Adding OpenMP GPU target pragmas...")
    source = src.read_text()
    gpu_source = add_omp_gpu_pragmas(source)
    gpu_src.write_text(gpu_source)

    print("Compiling with nvc -mp=gpu...")
    cmd = [NVC, '-mp=gpu', '-gpu=cc86', '-O2', '-fPIC', '-shared',
           '-o', str(gpu_so), str(gpu_src), '-lm']
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Compilation failed:\n{result.stderr}")
        sys.exit(1)

    print(f"  Compiled: {gpu_so}")
    print(f"\nTo use in benchmarks: python benchmark_tsvc_sizes.py --omp-gpu")


if __name__ == '__main__':
    main()
