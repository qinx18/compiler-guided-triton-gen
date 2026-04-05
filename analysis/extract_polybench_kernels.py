#!/usr/bin/env python3
"""
Extract Polybench/C kernels and prepare them for PET analysis.
Handles Polybench macro expansion (POLYBENCH_2D, DATA_TYPE, _PB_*, SCALAR_VAL, etc.)
and generates standalone .c files in kernels_polybench/ with #pragma scop.
"""

import re
import os
import glob

POLYBENCH_ROOT = "/home/qinxiao/workspace/compiler-guided-triton-gen/benchmarks_src/polybench-c-4.2.1"
OUTPUT_DIR = "/home/qinxiao/workspace/compiler-guided-triton-gen/analysis/kernels_polybench"

# All 30 Polybench/C 4.2.1 kernels with their source paths, parameters, and array info.
# We use SMALL_DATASET sizes to keep PET analysis fast.
POLYBENCH_KERNELS = {
    # --- datamining ---
    "correlation": {
        "path": "datamining/correlation/correlation.c",
        "params": {"M": 80, "N": 100},
        "pb_macros": {"_PB_M": "M", "_PB_N": "N"},
        "data_type": "float",
        "extra_defines": [],
        "extra_globals": ["float float_n = 100.0f;", "float eps = 0.1f;"],
        "math_replacements": {"SQRT_FUN": "sqrtf"},
    },
    "covariance": {
        "path": "datamining/covariance/covariance.c",
        "params": {"M": 80, "N": 100},
        "pb_macros": {"_PB_M": "M", "_PB_N": "N"},
        "data_type": "float",
        "extra_defines": [],
        "extra_globals": ["float float_n = 100.0f;"],
        "math_replacements": {},
    },
    # --- linear-algebra/blas ---
    "gemm": {
        "path": "linear-algebra/blas/gemm/gemm.c",
        "params": {"NI": 60, "NJ": 70, "NK": 80},
        "pb_macros": {"_PB_NI": "NI", "_PB_NJ": "NJ", "_PB_NK": "NK"},
        "data_type": "float",
        "extra_defines": [],
        "extra_globals": ["float alpha = 1.5f;", "float beta = 1.2f;"],
        "math_replacements": {},
    },
    "gemver": {
        "path": "linear-algebra/blas/gemver/gemver.c",
        "params": {"N": 120},
        "pb_macros": {"_PB_N": "N"},
        "data_type": "float",
        "extra_defines": [],
        "extra_globals": ["float alpha = 1.5f;", "float beta = 1.2f;"],
        "math_replacements": {},
    },
    "gesummv": {
        "path": "linear-algebra/blas/gesummv/gesummv.c",
        "params": {"N": 90},
        "pb_macros": {"_PB_N": "N"},
        "data_type": "float",
        "extra_defines": [],
        "extra_globals": ["float alpha = 1.5f;", "float beta = 1.2f;"],
        "math_replacements": {},
    },
    "symm": {
        "path": "linear-algebra/blas/symm/symm.c",
        "params": {"M": 60, "N": 80},
        "pb_macros": {"_PB_M": "M", "_PB_N": "N"},
        "data_type": "float",
        "extra_defines": [],
        "extra_globals": ["float alpha = 1.5f;", "float beta = 1.2f;"],
        "math_replacements": {},
    },
    "syr2k": {
        "path": "linear-algebra/blas/syr2k/syr2k.c",
        "params": {"M": 60, "N": 80},
        "pb_macros": {"_PB_M": "M", "_PB_N": "N"},
        "data_type": "float",
        "extra_defines": [],
        "extra_globals": ["float alpha = 1.5f;", "float beta = 1.2f;"],
        "math_replacements": {},
    },
    "syrk": {
        "path": "linear-algebra/blas/syrk/syrk.c",
        "params": {"M": 60, "N": 80},
        "pb_macros": {"_PB_M": "M", "_PB_N": "N"},
        "data_type": "float",
        "extra_defines": [],
        "extra_globals": ["float alpha = 1.5f;", "float beta = 1.2f;"],
        "math_replacements": {},
    },
    "trmm": {
        "path": "linear-algebra/blas/trmm/trmm.c",
        "params": {"M": 60, "N": 80},
        "pb_macros": {"_PB_M": "M", "_PB_N": "N"},
        "data_type": "float",
        "extra_defines": [],
        "extra_globals": ["float alpha = 1.5f;"],
        "math_replacements": {},
    },
    # --- linear-algebra/kernels ---
    "2mm": {
        "path": "linear-algebra/kernels/2mm/2mm.c",
        "params": {"NI": 40, "NJ": 50, "NK": 70, "NL": 80},
        "pb_macros": {"_PB_NI": "NI", "_PB_NJ": "NJ", "_PB_NK": "NK", "_PB_NL": "NL"},
        "data_type": "float",
        "extra_defines": [],
        "extra_globals": ["float alpha = 1.5f;", "float beta = 1.2f;"],
        "math_replacements": {},
    },
    "3mm": {
        "path": "linear-algebra/kernels/3mm/3mm.c",
        "params": {"NI": 40, "NJ": 50, "NK": 60, "NL": 70, "NM": 80},
        "pb_macros": {"_PB_NI": "NI", "_PB_NJ": "NJ", "_PB_NK": "NK", "_PB_NL": "NL", "_PB_NM": "NM"},
        "data_type": "float",
        "extra_defines": [],
        "extra_globals": [],
        "math_replacements": {},
    },
    "atax": {
        "path": "linear-algebra/kernels/atax/atax.c",
        "params": {"M": 65, "N": 85},
        "pb_macros": {"_PB_M": "M", "_PB_N": "N"},
        "data_type": "float",
        "extra_defines": [],
        "extra_globals": [],
        "math_replacements": {},
    },
    "bicg": {
        "path": "linear-algebra/kernels/bicg/bicg.c",
        "params": {"M": 75, "N": 85},
        "pb_macros": {"_PB_M": "M", "_PB_N": "N"},
        "data_type": "float",
        "extra_defines": [],
        "extra_globals": [],
        "math_replacements": {},
    },
    "doitgen": {
        "path": "linear-algebra/kernels/doitgen/doitgen.c",
        "params": {"NR": 25, "NQ": 20, "NP": 30},
        "pb_macros": {"_PB_NR": "NR", "_PB_NQ": "NQ", "_PB_NP": "NP"},
        "data_type": "float",
        "extra_defines": [],
        "extra_globals": [],
        "math_replacements": {},
    },
    "mvt": {
        "path": "linear-algebra/kernels/mvt/mvt.c",
        "params": {"N": 120},
        "pb_macros": {"_PB_N": "N"},
        "data_type": "float",
        "extra_defines": [],
        "extra_globals": [],
        "math_replacements": {},
    },
    # --- linear-algebra/solvers ---
    "cholesky": {
        "path": "linear-algebra/solvers/cholesky/cholesky.c",
        "params": {"N": 120},
        "pb_macros": {"_PB_N": "N"},
        "data_type": "float",
        "extra_defines": [],
        "extra_globals": [],
        "math_replacements": {"SQRT_FUN": "sqrtf"},
    },
    "durbin": {
        "path": "linear-algebra/solvers/durbin/durbin.c",
        "params": {"N": 120},
        "pb_macros": {"_PB_N": "N"},
        "data_type": "float",
        "extra_defines": [],
        "extra_globals": [],
        "math_replacements": {},
    },
    "gramschmidt": {
        "path": "linear-algebra/solvers/gramschmidt/gramschmidt.c",
        "params": {"M": 60, "N": 80},
        "pb_macros": {"_PB_M": "M", "_PB_N": "N"},
        "data_type": "float",
        "extra_defines": [],
        "extra_globals": [],
        "math_replacements": {"SQRT_FUN": "sqrtf"},
    },
    "lu": {
        "path": "linear-algebra/solvers/lu/lu.c",
        "params": {"N": 120},
        "pb_macros": {"_PB_N": "N"},
        "data_type": "float",
        "extra_defines": [],
        "extra_globals": [],
        "math_replacements": {},
    },
    "ludcmp": {
        "path": "linear-algebra/solvers/ludcmp/ludcmp.c",
        "params": {"N": 120},
        "pb_macros": {"_PB_N": "N"},
        "data_type": "float",
        "extra_defines": [],
        "extra_globals": [],
        "math_replacements": {},
    },
    "trisolv": {
        "path": "linear-algebra/solvers/trisolv/trisolv.c",
        "params": {"N": 120},
        "pb_macros": {"_PB_N": "N"},
        "data_type": "float",
        "extra_defines": [],
        "extra_globals": [],
        "math_replacements": {},
    },
    # --- medley ---
    "deriche": {
        "path": "medley/deriche/deriche.c",
        "params": {"W": 192, "H": 128},
        "pb_macros": {"_PB_W": "W", "_PB_H": "H"},
        "data_type": "float",
        "extra_defines": [],
        "extra_globals": ["float alpha = 0.25f;"],
        "math_replacements": {"EXP_FUN": "expf", "POW_FUN": "powf"},
        # y1 conflicts with POSIX Bessel function y1() in math.h, rename to yy1
        "name_remap": {"y1": "yy1"},
    },
    "floyd-warshall": {
        "path": "medley/floyd-warshall/floyd-warshall.c",
        "params": {"N": 120},
        "pb_macros": {"_PB_N": "N"},
        "data_type": "float",
        "extra_defines": [],
        "extra_globals": [],
        "math_replacements": {},
    },
    "nussinov": {
        "path": "medley/nussinov/nussinov.c",
        "params": {"N": 180},
        "pb_macros": {"_PB_N": "N"},
        "data_type": "int",
        "extra_defines": [
            "typedef char base;",
            "#define match(b1, b2) (((b1)+(b2)) == 3 ? 1 : 0)",
            "#define max_score(s1, s2) ((s1 >= s2) ? s1 : s2)",
        ],
        "extra_globals": ["base seq[N];"],
        "math_replacements": {},
    },
    # --- stencils ---
    "adi": {
        "path": "stencils/adi/adi.c",
        "params": {"TSTEPS": 40, "N": 60},
        "pb_macros": {"_PB_TSTEPS": "TSTEPS", "_PB_N": "N"},
        "data_type": "float",
        "extra_defines": [],
        "extra_globals": [],
        "math_replacements": {},
    },
    "fdtd-2d": {
        "path": "stencils/fdtd-2d/fdtd-2d.c",
        "params": {"TMAX": 20, "NX": 60, "NY": 80},
        "pb_macros": {"_PB_TMAX": "TMAX", "_PB_NX": "NX", "_PB_NY": "NY"},
        "data_type": "float",
        "extra_defines": [],
        "extra_globals": [],
        "math_replacements": {},
    },
    "heat-3d": {
        "path": "stencils/heat-3d/heat-3d.c",
        "params": {"TSTEPS": 20, "N": 40},
        "pb_macros": {"_PB_TSTEPS": "TSTEPS", "_PB_N": "N"},
        "data_type": "float",
        "extra_defines": [],
        "extra_globals": [],
        "math_replacements": {},
    },
    "jacobi-1d": {
        "path": "stencils/jacobi-1d/jacobi-1d.c",
        "params": {"TSTEPS": 40, "N": 120},
        "pb_macros": {"_PB_TSTEPS": "TSTEPS", "_PB_N": "N"},
        "data_type": "float",
        "extra_defines": [],
        "extra_globals": [],
        "math_replacements": {},
    },
    "jacobi-2d": {
        "path": "stencils/jacobi-2d/jacobi-2d.c",
        "params": {"TSTEPS": 40, "N": 90},
        "pb_macros": {"_PB_TSTEPS": "TSTEPS", "_PB_N": "N"},
        "data_type": "float",
        "extra_defines": [],
        "extra_globals": [],
        "math_replacements": {},
    },
    "seidel-2d": {
        "path": "stencils/seidel-2d/seidel-2d.c",
        "params": {"TSTEPS": 40, "N": 120},
        "pb_macros": {"_PB_TSTEPS": "TSTEPS", "_PB_N": "N"},
        "data_type": "float",
        "extra_defines": [],
        "extra_globals": [],
        "math_replacements": {},
    },
}


def extract_scop_region(source_file):
    """Extract code between #pragma scop and #pragma endscop."""
    with open(source_file, 'r') as f:
        content = f.read()

    # Find the kernel function's scop region
    match = re.search(r'#pragma\s+scop\s*\n(.*?)#pragma\s+endscop', content, re.DOTALL)
    if not match:
        return None
    return match.group(1)


def extract_kernel_local_vars(source_file):
    """Extract local variable declarations inside the kernel_* function but before #pragma scop."""
    with open(source_file, 'r') as f:
        content = f.read()

    # Find the opening brace of kernel function (handle nested parens in signature)
    sig_match = re.search(r'void\s+kernel_\w+\s*\(', content)
    if not sig_match:
        return [], []

    # Skip past balanced parentheses of signature
    pos = sig_match.end()
    depth = 1
    while depth > 0 and pos < len(content):
        if content[pos] == '(':
            depth += 1
        elif content[pos] == ')':
            depth -= 1
        pos += 1

    # Find the opening brace after the signature
    brace_match = re.search(r'\{', content[pos:])
    if not brace_match:
        return [], []

    body_start = pos + brace_match.end()

    # Find #pragma scop
    scop_match = re.search(r'#pragma\s+scop', content[body_start:])
    if not scop_match:
        return [], []

    pre_scop = content[body_start:body_start + scop_match.start()]
    local_vars = []
    iter_vars = []

    for line in pre_scop.split('\n'):
        stripped = line.strip()
        if not stripped or stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
            continue
        # Capture iteration variable declarations like "int i, j, k;" or "int t, i, j;"
        iter_match = re.match(r'^int\s+([a-z](?:\s*,\s*[a-z])*)\s*;', stripped)
        if iter_match:
            vars_str = iter_match.group(1)
            iter_vars.extend([v.strip() for v in vars_str.split(',')])
            continue
        # Match variable declarations with initializers
        if re.match(r'^(DATA_TYPE|float|double|int)\s+\w+', stripped):
            local_vars.append(stripped)

    return local_vars, iter_vars


def extract_kernel_arrays(source_file):
    """Extract array parameters from the kernel_* function signature."""
    with open(source_file, 'r') as f:
        content = f.read()

    # Find kernel function signature - need to handle nested parens from POLYBENCH macros
    match = re.search(r'void\s+kernel_\w+\s*\(', content)
    if not match:
        return []

    # Extract balanced parentheses
    start = match.end() - 1  # position of opening (
    depth = 1
    pos = start + 1
    while depth > 0 and pos < len(content):
        if content[pos] == '(':
            depth += 1
        elif content[pos] == ')':
            depth -= 1
        pos += 1
    sig = content[start + 1:pos - 1]
    arrays = []

    # Match POLYBENCH_1D(var, dim1, ddim1) patterns
    for m in re.finditer(r'POLYBENCH_1D\(\s*(\w+)\s*,\s*(\w+)\s*,\s*\w+\s*\)', sig):
        arrays.append({"name": m.group(1), "dims": [m.group(2)], "ndim": 1})

    # Match POLYBENCH_2D(var, dim1, dim2, ddim1, ddim2) patterns
    for m in re.finditer(r'POLYBENCH_2D\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*,\s*\w+\s*,\s*\w+\s*\)', sig):
        arrays.append({"name": m.group(1), "dims": [m.group(2), m.group(3)], "ndim": 2})

    # Match POLYBENCH_3D(var, dim1, dim2, dim3, ddim1, ddim2, ddim3) patterns
    for m in re.finditer(r'POLYBENCH_3D\(\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*,\s*(\w+)\s*,\s*\w+\s*,\s*\w+\s*,\s*\w+\s*\)', sig):
        arrays.append({"name": m.group(1), "dims": [m.group(2), m.group(3), m.group(4)], "ndim": 3})

    # Match scalar parameters: DATA_TYPE varname (not array)
    # These are params like alpha, beta, float_n
    for m in re.finditer(r'(?:DATA_TYPE|float|double|int)\s+(\w+)(?:\s*,|\s*\))', sig):
        name = m.group(1)
        # Skip if this name was already captured as an array
        if not any(a["name"] == name for a in arrays):
            # Check it's not a size parameter like ni, nj, n, m etc.
            if not re.match(r'^(n[ijklmrqp]?|m|w|h|tsteps|nr|nq|np|tmax|nx|ny)$', name):
                arrays.append({"name": name, "dims": [], "ndim": 0})

    return arrays


def process_scop_code(scop_code, kernel_info):
    """Process scop code: expand _PB_* macros, SCALAR_VAL, DATA_TYPE casts, math fns."""
    code = scop_code

    # Replace _PB_* macros with their constant equivalents
    for pb_macro, param_name in kernel_info["pb_macros"].items():
        code = re.sub(r'\b' + re.escape(pb_macro) + r'\b', param_name, code)

    # Replace SCALAR_VAL(x) -> x (for float, we could add 'f' suffix but
    # PET doesn't care about suffixes; plain literals work fine)
    code = re.sub(r'SCALAR_VAL\(([^)]+)\)', r'\1', code)

    # Replace DATA_TYPE casts: (DATA_TYPE) -> (float) or (double)
    dtype = kernel_info["data_type"]
    code = re.sub(r'\bDATA_TYPE\b', dtype, code)

    # Replace math functions
    for macro, replacement in kernel_info.get("math_replacements", {}).items():
        code = re.sub(r'\b' + re.escape(macro) + r'\b', replacement, code)

    return code


def process_local_vars(local_vars, kernel_info):
    """Process local variable declarations: expand DATA_TYPE, SCALAR_VAL."""
    processed = []
    dtype = kernel_info["data_type"]
    for var in local_vars:
        v = var
        v = re.sub(r'\bDATA_TYPE\b', dtype, v)
        v = re.sub(r'SCALAR_VAL\(([^)]+)\)', r'\1', v)
        for macro, replacement in kernel_info.get("math_replacements", {}).items():
            v = re.sub(r'\b' + re.escape(macro) + r'\b', replacement, v)
        processed.append(v)
    return processed


def create_kernel_file(kernel_name, kernel_info, scop_code, arrays, local_vars, iter_vars):
    """Create a standalone .c file with the kernel for PET analysis."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dtype = kernel_info["data_type"]

    # Apply name remapping (e.g., y1 -> yy1 for deriche to avoid math.h conflicts)
    name_remap = kernel_info.get("name_remap", {})
    if name_remap:
        for old_name, new_name in name_remap.items():
            # Remap array names
            for arr in arrays:
                if arr["name"] == old_name:
                    arr["name"] = new_name
            # Remap in scop code
            scop_code = re.sub(r'\b' + re.escape(old_name) + r'\b', new_name, scop_code)
            # Remap in local vars
            local_vars = [re.sub(r'\b' + re.escape(old_name) + r'\b', new_name, v) for v in local_vars]

    lines = []
    lines.append("#include <math.h>")

    lines.append("")

    # Define size constants
    for param_name, param_value in kernel_info["params"].items():
        lines.append(f"#define {param_name} {param_value}")
    lines.append("")

    # Extra defines (e.g., nussinov's match/max_score macros)
    for d in kernel_info.get("extra_defines", []):
        lines.append(d)
    if kernel_info.get("extra_defines"):
        lines.append("")

    # Track declared global names to avoid duplicates
    declared_names = set()

    # Declare arrays as globals
    for arr in arrays:
        if arr["ndim"] == 0:
            continue
        declared_names.add(arr["name"])
        if arr["ndim"] == 1:
            dim_str = f"[{arr['dims'][0]}]"
            lines.append(f"{dtype} {arr['name']}{dim_str};")
        elif arr["ndim"] == 2:
            dim_str = f"[{arr['dims'][0]}][{arr['dims'][1]}]"
            lines.append(f"{dtype} {arr['name']}{dim_str};")
        elif arr["ndim"] == 3:
            dim_str = f"[{arr['dims'][0]}][{arr['dims'][1]}][{arr['dims'][2]}]"
            lines.append(f"{dtype} {arr['name']}{dim_str};")
    lines.append("")

    # Extra globals (scalar params like alpha, beta, float_n)
    # Skip any that duplicate an already-declared array name
    for g in kernel_info.get("extra_globals", []):
        # Extract the variable name from the declaration
        gm = re.match(r'(?:float|double|int|base)\s+(\w+)', g)
        if gm and gm.group(1) in declared_names:
            continue
        if gm:
            declared_names.add(gm.group(1))
        lines.append(g)
    if kernel_info.get("extra_globals"):
        lines.append("")

    # Processed local variables (promoted to globals for PET)
    # Skip any that duplicate already-declared names (from arrays or extra_globals)
    processed_locals = process_local_vars(local_vars, kernel_info)
    if processed_locals:
        deduped_locals = []
        for v in processed_locals:
            vm = re.match(r'(?:float|double|int|base)\s+(\w+)', v)
            if vm and vm.group(1) in declared_names:
                continue
            if vm:
                declared_names.add(vm.group(1))
            deduped_locals.append(v)
        if deduped_locals:
            lines.append("// Local variables from kernel function")
            for v in deduped_locals:
                lines.append(v)
            lines.append("")

    # Sanitize kernel name for C identifier (replace - with _)
    c_name = kernel_name.replace("-", "_")
    # C identifiers can't start with a digit; prefix with 'k' if needed
    func_name = c_name
    if func_name[0].isdigit():
        func_name = "k" + func_name  # e.g., 2mm -> k2mm (valid C identifier)

    # Declare the kernel function with iterator variables as parameters
    # so PET can infer loop bounds properly, or declare them inside
    if iter_vars:
        lines.append(f"void {func_name}_kernel() {{")
        # Declare loop iterators as local variables inside the function
        lines.append(f"  int {', '.join(iter_vars)};")
    else:
        lines.append(f"void {func_name}_kernel() {{")

    lines.append("#pragma scop")
    lines.append(scop_code.rstrip())
    lines.append("#pragma endscop")
    lines.append("}")
    lines.append("")

    content = "\n".join(lines)
    # Keep filename based on original c_name (not func_name) for consistency
    filepath = os.path.join(OUTPUT_DIR, f"{c_name}.c")
    with open(filepath, 'w') as f:
        f.write(content)

    return filepath


def extract_all_kernels():
    """Extract all 30 Polybench kernels."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {}
    for kernel_name, kernel_info in sorted(POLYBENCH_KERNELS.items()):
        source_file = os.path.join(POLYBENCH_ROOT, kernel_info["path"])

        if not os.path.exists(source_file):
            print(f"  ERROR: Source file not found: {source_file}")
            results[kernel_name] = {"error": "source not found"}
            continue

        print(f"Processing {kernel_name}...")

        # Extract scop region
        scop_code = extract_scop_region(source_file)
        if not scop_code:
            print(f"  ERROR: Could not extract #pragma scop region")
            results[kernel_name] = {"error": "no scop region"}
            continue

        # Extract array info from function signature
        arrays = extract_kernel_arrays(source_file)

        # Extract local variables and iterator variables
        local_vars, iter_vars = extract_kernel_local_vars(source_file)

        # Process the scop code (expand macros)
        processed_scop = process_scop_code(scop_code, kernel_info)

        # Create the standalone kernel file
        filepath = create_kernel_file(kernel_name, kernel_info, processed_scop, arrays, local_vars, iter_vars)

        if filepath:
            c_name = kernel_name.replace("-", "_")
            results[kernel_name] = {
                "file": filepath,
                "arrays": [a["name"] for a in arrays],
                "ndims": {a["name"]: a["ndim"] for a in arrays},
                "local_vars": local_vars,
            }
            print(f"  -> {filepath}")
            print(f"     Arrays: {[a['name'] for a in arrays if a['ndim'] > 0]}")
            if local_vars:
                print(f"     Local vars: {local_vars}")
        else:
            results[kernel_name] = {"error": "failed to create kernel file"}

    return results


def main():
    print("Extracting Polybench/C kernels for PET analysis...")
    print(f"Source: {POLYBENCH_ROOT}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Kernels: {len(POLYBENCH_KERNELS)}")
    print("=" * 60)

    results = extract_all_kernels()

    # Summary
    success = sum(1 for r in results.values() if "file" in r)
    errors = sum(1 for r in results.values() if "error" in r)

    print("\n" + "=" * 60)
    print(f"SUMMARY: {success} extracted, {errors} errors, {len(POLYBENCH_KERNELS)} total")

    if errors > 0:
        print("\nFailed kernels:")
        for name, r in sorted(results.items()):
            if "error" in r:
                print(f"  {name}: {r['error']}")

    print(f"\nKernel files in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
