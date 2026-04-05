#!/usr/bin/env python3
"""
Python wrappers for Polybench/C reference kernels via ctypes.

Analogous to tsvc_all_reference.py, this compiles Polybench kernels
into a shared library and provides Python callables for correctness testing.

The kernels are compiled from the extracted standalone .c files in
kernels_polybench/ with SMALL_DATASET sizes.
"""

import ctypes
import numpy as np
import os
import subprocess
import tempfile
from pathlib import Path

KERNELS_DIR = "/home/qinxiao/workspace/compiler-guided-triton-gen/analysis/kernels_polybench"
CLANG = "/usr/local/bin/clang"


def _compile_kernel(kernel_name: str, lib_dir: str = None) -> str:
    """Compile a single Polybench kernel .c file to a shared library."""
    c_name = kernel_name.replace("-", "_")
    c_file = os.path.join(KERNELS_DIR, f"{c_name}.c")
    if not os.path.exists(c_file):
        raise FileNotFoundError(f"Kernel file not found: {c_file}")

    if lib_dir is None:
        lib_dir = os.path.join(os.path.dirname(__file__), "polybench_libs")
    os.makedirs(lib_dir, exist_ok=True)

    so_file = os.path.join(lib_dir, f"lib{c_name}.so")

    # Only recompile if source is newer
    if os.path.exists(so_file) and os.path.getmtime(so_file) >= os.path.getmtime(c_file):
        return so_file

    result = subprocess.run(
        [CLANG, "-shared", "-fPIC", "-O2", "-o", so_file, c_file, "-lm"],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        raise RuntimeError(f"Compilation failed for {c_name}: {result.stderr}")

    return so_file


def _load_kernel(kernel_name: str) -> ctypes.CDLL:
    """Load a compiled Polybench kernel."""
    so_file = _compile_kernel(kernel_name)
    return ctypes.CDLL(so_file)


# Cache for loaded libraries
_lib_cache = {}


def get_kernel_lib(kernel_name: str) -> ctypes.CDLL:
    """Get or load the shared library for a kernel."""
    c_name = kernel_name.replace("-", "_")
    if c_name not in _lib_cache:
        _lib_cache[c_name] = _load_kernel(kernel_name)
    return _lib_cache[c_name]


def run_kernel(kernel_name: str) -> bool:
    """
    Run a Polybench kernel's function.
    The kernel modifies global arrays in the .so directly.
    Returns True if the kernel ran without error.
    """
    c_name = kernel_name.replace("-", "_")
    lib = get_kernel_lib(kernel_name)
    # C function names can't start with digits, prefix with 'k'
    func_id = c_name
    if func_id[0].isdigit():
        func_id = "k" + func_id
    func_name = f"{func_id}_kernel"
    if not hasattr(lib, func_name):
        raise AttributeError(f"Function {func_name} not found in library")

    func = getattr(lib, func_name)
    func.argtypes = []
    func.restype = None
    func()
    return True


def compile_all_kernels():
    """Compile all Polybench kernel .c files to shared libraries."""
    import sys
    sys.path.insert(0, "/home/qinxiao/workspace/compiler-guided-triton-gen/analysis")
    from extract_polybench_kernels import POLYBENCH_KERNELS

    results = {}
    for kernel_name in sorted(POLYBENCH_KERNELS.keys()):
        try:
            so_file = _compile_kernel(kernel_name)
            results[kernel_name] = {"status": "ok", "file": so_file}
            print(f"  {kernel_name}: OK -> {so_file}")
        except Exception as e:
            results[kernel_name] = {"status": "error", "error": str(e)}
            print(f"  {kernel_name}: ERROR - {e}")

    return results


if __name__ == "__main__":
    print("Compiling all Polybench kernels...")
    results = compile_all_kernels()
    ok = sum(1 for r in results.values() if r["status"] == "ok")
    err = sum(1 for r in results.values() if r["status"] == "error")
    print(f"\nResults: {ok} compiled, {err} errors, {len(results)} total")
