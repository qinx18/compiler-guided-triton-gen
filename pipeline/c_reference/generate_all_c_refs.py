#!/usr/bin/env python3
"""
Automatically generate C reference implementations from TSVC database.

This script:
1. Reads loop_code from tsvc_functions_db.py
2. Extracts the inner computation (removes timing loop)
3. Generates C kernel functions with proper signatures
4. Generates Python ctypes wrappers
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'utilities'))
from tsvc_functions_db import TSVC_FUNCTIONS


def extract_inner_loop(loop_code):
    """
    Extract inner computation from TSVC loop code.
    Removes the outer timing loop (for nl = ...).
    """
    lines = loop_code.strip().split('\n')
    result_lines = []
    brace_depth = 0
    in_timing_loop = False
    skip_next_brace = False

    for line in lines:
        stripped = line.strip()

        # Detect timing loop start
        if 'for' in stripped and ('nl' in stripped or 'iterations' in stripped):
            in_timing_loop = True
            brace_depth = stripped.count('{') - stripped.count('}')
            continue

        # Track braces
        open_braces = stripped.count('{')
        close_braces = stripped.count('}')

        if in_timing_loop:
            brace_depth += open_braces - close_braces
            if brace_depth <= 0:
                in_timing_loop = False
                # Skip the closing brace of timing loop
                if stripped == '}':
                    continue

        # Skip dummy function calls
        if 'dummy(' in stripped:
            continue

        result_lines.append(line)

    return '\n'.join(result_lines)


def translate_to_c_kernel(func_name, func_spec):
    """
    Generate C kernel function from TSVC spec.
    """
    loop_code = func_spec['loop_code']
    arrays = func_spec['arrays']
    scalar_params = func_spec.get('scalar_params', {})
    has_2d = func_spec.get('has_2d_arrays', False)

    # Extract inner loop
    inner_code = extract_inner_loop(loop_code)

    # Skip functions that call external functions we don't have helpers for
    skip_patterns = [
        (r'\bs1213s\(', 's1213s'),
        (r'\bcalc_result\(', 'calc_result'),
    ]
    for pattern, name in skip_patterns:
        if re.search(pattern, inner_code):
            raise ValueError(f"Function calls external function: {name}()")

    # Rename helper function calls to match our helper implementations
    inner_code = re.sub(r'\btest\(', 'test_helper(', inner_code)
    inner_code = re.sub(r'\bf\(', 'f_helper(', inner_code)

    # Add n parameter to s151s calls: s151s(a, b, m) -> s151s(a, b, n, m)
    inner_code = re.sub(r'\bs151s\(([^,]+),\s*([^,]+),\s*([^)]+)\)',
                        r's151s(\1, \2, n, \3)', inner_code)

    # Handle x variable conflict: if 'x' is an array and code uses x as local variable
    if 'x' in arrays:
        # Replace local variable x with x_local
        inner_code = re.sub(r'\bx\s*=\s*(?![^;]*\[)', 'x_local = ', inner_code)
        inner_code = re.sub(r'(?<!\w)x(?!\s*\[)(?!\w)', 'x_local', inner_code)
        # But keep x[...] as is for array access
        inner_code = re.sub(r'\bx_local\s*\[', 'x[', inner_code)

    # Build parameter list
    params = []
    # Integer arrays (used for indexing)
    int_arrays = {'ip', 'indx', 'ind'}
    for arr_name in sorted(arrays.keys()):
        if arr_name in int_arrays:
            params.append(f'int* {arr_name}')
        else:
            params.append(f'real_t* {arr_name}')

    # Add size parameter
    params.append('int n')

    # Add LEN_2D if needed
    if has_2d:
        params.append('int len_2d')

    # Add other scalar params (excluding iterations and special keywords)
    skip_scalars = {'iterations', '__restrict__', 'restrict', 'sinf', 'cosf', 'sqrtf', 'expf', 'fabsf'}
    float_scalars = {'s', 's1', 's2', 't', 't_value', 'x', 'alpha'}  # Float scalar parameters
    for scalar_name in sorted(scalar_params.keys()):
        if scalar_name not in skip_scalars:
            if scalar_name in float_scalars:
                params.append(f'real_t {scalar_name}')
            else:
                params.append(f'int {scalar_name}')

    param_str = ', '.join(params)

    # Check if M is used in code but not already in params
    if ' M;' in inner_code or ' M)' in inner_code or ' M ' in inner_code or '<M' in inner_code or 'i < M' in inner_code:
        if 'int M' not in param_str and 'int m' not in param_str:
            params.append('int M')
            param_str = ', '.join(params)

    # Replace LEN_1D with n, LEN_2D with len_2d
    inner_code = inner_code.replace('LEN_1D', 'n')
    inner_code = inner_code.replace('LEN_2D', 'len_2d')

    # Replace M with m for consistency (some TSVC uses M as macro for m param)
    inner_code = re.sub(r'\bM\b', 'm', inner_code)

    # Convert 2D array access: arr[i][j] -> arr[(i) * len_2d + (j)]
    # Use iterative approach to handle nested brackets
    def convert_2d_array(code, arr_name):
        result = []
        i = 0
        while i < len(code):
            # Look for arr_name[
            if code[i:i+len(arr_name)+1] == arr_name + '[':
                # Find first index with balanced brackets
                start1 = i + len(arr_name) + 1
                depth = 1
                end1 = start1
                while end1 < len(code) and depth > 0:
                    if code[end1] == '[':
                        depth += 1
                    elif code[end1] == ']':
                        depth -= 1
                    end1 += 1
                end1 -= 1  # Point to closing ]

                # Check if there's a second index
                if end1 + 1 < len(code) and code[end1 + 1] == '[':
                    start2 = end1 + 2
                    depth = 1
                    end2 = start2
                    while end2 < len(code) and depth > 0:
                        if code[end2] == '[':
                            depth += 1
                        elif code[end2] == ']':
                            depth -= 1
                        end2 += 1
                    end2 -= 1  # Point to closing ]

                    idx1 = code[start1:end1]
                    idx2 = code[start2:end2]
                    result.append(f'{arr_name}[({idx1}) * len_2d + ({idx2})]')
                    i = end2 + 1
                else:
                    result.append(code[i:end1+1])
                    i = end1 + 1
            else:
                result.append(code[i])
                i += 1
        return ''.join(result)

    for arr_2d in ['aa', 'bb', 'cc', 'tt']:
        inner_code = convert_2d_array(inner_code, arr_2d)

    # Find and declare local variables (j, k, s, etc.)
    local_vars = set()
    # Look for assignments to undeclared variables
    for match in re.finditer(r'^\s*(\w+)\s*=', inner_code, re.MULTILINE):
        var = match.group(1)
        # Skip if it's an array element assignment or known param
        param_names = [p.split()[-1] for p in params]
        if var not in arrays and var not in ['i', 'nl'] and var not in param_names:
            # Skip 'x' if there's an array x, skip variables with (real_t) prefix
            if var != 'x' or 'x' not in arrays:
                local_vars.add(var)

    # Remove variables that are redeclared with type in the code
    for var in list(local_vars):
        if re.search(rf'(int|real_t)\s+{var}\s*=', inner_code):
            local_vars.discard(var)

    # Add local variable declarations
    local_decl = ''
    # Add x_local if we renamed x variable
    if 'x' in arrays and 'x_local' in inner_code:
        local_vars.add('x_local')
    if local_vars:
        int_vars = [v for v in local_vars if v in ['j', 'k', 'ip', 'index', 'im1', 'im2', 'off']]
        float_vars = [v for v in local_vars if v not in int_vars]
        if int_vars:
            local_decl += '    int ' + ', '.join(sorted(int_vars)) + ';\n'
        if float_vars:
            local_decl += '    real_t ' + ', '.join(sorted(float_vars)) + ';\n'

    c_code = f'''/* {func_name} */
void {func_name}_kernel({param_str}) {{
{local_decl}{inner_code}
}}
'''
    return c_code


def generate_python_wrapper(func_name, func_spec):
    """
    Generate Python ctypes wrapper for a function.
    """
    arrays = func_spec['arrays']
    scalar_params = func_spec.get('scalar_params', {})
    has_2d = func_spec.get('has_2d_arrays', False)

    # Build argtypes
    argtypes = []
    int_arrays = {'ip', 'indx', 'ind'}
    for arr_name in sorted(arrays.keys()):
        if arr_name in int_arrays:
            argtypes.append('ctypes.POINTER(ctypes.c_int)')
        else:
            argtypes.append('ctypes.POINTER(ctypes.c_float)')
    argtypes.append('ctypes.c_int')  # n

    if has_2d:
        argtypes.append('ctypes.c_int')  # len_2d

    skip_scalars = {'iterations', '__restrict__', 'restrict', 'sinf', 'cosf', 'sqrtf', 'expf', 'fabsf'}
    float_scalars = {'s', 's1', 's2', 't', 't_value', 'x', 'alpha'}  # Float scalar parameters
    for scalar_name in sorted(scalar_params.keys()):
        if scalar_name not in skip_scalars:
            if scalar_name in float_scalars:
                argtypes.append('ctypes.c_float')
            else:
                argtypes.append('ctypes.c_int')

    argtypes_str = ',\n        '.join(argtypes)

    # Build function signature
    func_params = []
    for arr_name in sorted(arrays.keys()):
        func_params.append(arr_name)

    # Add n parameter if no arrays (scalar-only function)
    if not arrays:
        func_params.append('n=100')

    if has_2d:
        func_params.append('len_2d=None')

    for scalar_name in sorted(scalar_params.keys()):
        if scalar_name not in skip_scalars:
            func_params.append(f'{scalar_name}=1')

    func_params_str = ', '.join(func_params)

    # Build function body
    body_lines = []
    for arr_name in sorted(arrays.keys()):
        if arr_name in ['aa', 'bb', 'cc', 'tt']:
            body_lines.append(f'    {arr_name} = np.ascontiguousarray({arr_name}.flatten(), dtype=np.float32)')
        elif arr_name in int_arrays:
            body_lines.append(f'    {arr_name} = np.ascontiguousarray({arr_name}, dtype=np.int32)')
        else:
            body_lines.append(f'    {arr_name} = np.ascontiguousarray({arr_name}, dtype=np.float32)')

    # Determine n
    if arrays:
        first_arr = sorted(arrays.keys())[0]
        body_lines.append(f'    n = len({first_arr})')
    else:
        # No arrays - use a default n parameter
        body_lines.append('    n = n if n else 100')

    # Build call args
    call_args = []
    for arr_name in sorted(arrays.keys()):
        if arr_name in int_arrays:
            call_args.append(f'_to_ptr_int({arr_name})')
        else:
            call_args.append(f'_to_ptr({arr_name})')
    call_args.append('n')

    if has_2d:
        call_args.append('len_2d if len_2d else int(np.sqrt(n))')

    for scalar_name in sorted(scalar_params.keys()):
        if scalar_name not in skip_scalars:
            call_args.append(scalar_name)

    call_args_str = ', '.join(call_args)

    # Return value
    output_arrays = [arr for arr, mode in arrays.items() if mode in ['rw', 'w']]
    if output_arrays:
        return_arr = sorted(output_arrays)[0]
        if return_arr in ['aa', 'bb', 'cc', 'tt']:
            body_lines.append(f'    _lib.{func_name}_kernel({call_args_str})')
            body_lines.append(f'    return {return_arr}.reshape(len_2d if len_2d else int(np.sqrt(n)), -1)')
        else:
            body_lines.append(f'    _lib.{func_name}_kernel({call_args_str})')
            body_lines.append(f'    return {return_arr}')
    else:
        body_lines.append(f'    _lib.{func_name}_kernel({call_args_str})')

    body_str = '\n'.join(body_lines)

    setup_code = f'''    # {func_name}
    _lib.{func_name}_kernel.argtypes = [
        {argtypes_str}
    ]
    _lib.{func_name}_kernel.restype = None
'''

    wrapper_code = f'''
def {func_name}_c({func_params_str}):
    """C reference for {func_name}"""
{body_str}
'''

    return setup_code, wrapper_code, func_name


def main():
    """Generate all C references"""

    c_code_parts = ['''/*
 * Auto-generated TSVC C Reference Implementations
 * Generated from tsvc_functions_db.py
 */

#include <math.h>
#include <stdlib.h>

typedef float real_t;

#define ABS(x) (((x) < 0) ? -(x) : (x))

/* Helper functions used by some TSVC kernels */
void s151s(real_t* a, real_t* b, int n, int m) {
    for (int i = 0; i < n-1; i++) {
        a[i] = a[i + m] + b[i];
    }
}

void s152s(real_t* a, real_t* b, real_t* c, int i) {
    a[i] += b[i] * c[i];
}

real_t test_helper(real_t* A) {
    real_t s = (real_t)0.0;
    for (int i = 0; i < 4; i++)
        s += A[i];
    return s;
}

real_t f_helper(real_t a, real_t b) {
    return a * b;
}

int s471s(void) {
    return 0;
}

''']

    setup_parts = []
    wrapper_parts = []
    registry_entries = []

    success_count = 0
    error_count = 0

    for func_name, func_spec in sorted(TSVC_FUNCTIONS.items()):
        try:
            # Generate C code
            c_code = translate_to_c_kernel(func_name, func_spec)
            c_code_parts.append(c_code)

            # Generate Python wrapper
            setup, wrapper, name = generate_python_wrapper(func_name, func_spec)
            setup_parts.append(setup)
            wrapper_parts.append(wrapper)
            registry_entries.append(f"    '{name}': {name}_c,")

            success_count += 1
            print(f"✓ {func_name}")

        except Exception as e:
            error_count += 1
            print(f"✗ {func_name}: {e}")

    # Write C file
    c_file = Path(__file__).parent / 'tsvc_all_kernels.c'
    with open(c_file, 'w') as f:
        f.write('\n'.join(c_code_parts))
    print(f"\nWritten {c_file}")

    # Write Python file
    py_code = f'''#!/usr/bin/env python3
"""
Auto-generated Python wrappers for all TSVC C references.
"""

import ctypes
import numpy as np
from pathlib import Path

_lib_path = Path(__file__).parent / 'libtsvc_all.so'
_lib = ctypes.CDLL(str(_lib_path))

def _to_ptr(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

def _to_ptr_int(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

def _setup_functions():
{''.join(setup_parts)}

_setup_functions()

{''.join(wrapper_parts)}

# Registry
C_REFERENCE_FUNCS_ALL = {{
{chr(10).join(registry_entries)}
}}

def get_c_reference_all(func_name):
    return C_REFERENCE_FUNCS_ALL.get(func_name)
'''

    py_file = Path(__file__).parent / 'tsvc_all_reference.py'
    with open(py_file, 'w') as f:
        f.write(py_code)
    print(f"Written {py_file}")

    print(f"\nSummary: {success_count} success, {error_count} errors")
    print(f"\nTo compile: gcc -O2 -fPIC -shared -o libtsvc_all.so tsvc_all_kernels.c -lm")


if __name__ == '__main__':
    main()
