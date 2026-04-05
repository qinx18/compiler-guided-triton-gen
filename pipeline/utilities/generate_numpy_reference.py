#!/usr/bin/env python3
"""
Generate NumPy Reference Implementations for TSVC Functions

These are deterministic, explicit loop implementations that match C semantics exactly.
No LLM involved - just programmatic translation of C to Python loops.
"""

import re
import numpy as np
from pathlib import Path
from tsvc_functions_db import TSVC_FUNCTIONS


def parse_c_loop(loop_code):
    """
    Parse C loop code and extract:
    - Loop structure (nested loops, bounds)
    - Array accesses
    - Arithmetic operations
    """
    # Remove the outer timing loop (for nl = ...)
    lines = loop_code.strip().split('\n')
    inner_code = []
    brace_depth = 0
    skip_outer = True

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Skip the outer timing loop
        if skip_outer and 'for' in line and ('nl' in line or 'iterations' in line):
            brace_depth += line.count('{') - line.count('}')
            continue

        # Track brace depth
        brace_depth += line.count('{') - line.count('}')

        # Skip closing braces of timing loop
        if brace_depth <= 0 and line == '}':
            continue

        skip_outer = False
        inner_code.append(line)

    return '\n'.join(inner_code)


def translate_c_to_numpy(func_name, func_spec):
    """
    Translate C loop code to NumPy reference implementation.
    Uses explicit Python loops to match C semantics exactly.
    """
    loop_code = func_spec['loop_code']
    arrays = func_spec['arrays']
    has_2d = func_spec.get('has_2d_arrays', False)

    # Parse out the inner computation
    inner_code = parse_c_loop(loop_code)

    # Build parameter list (alphabetically sorted)
    params = sorted(arrays.keys())
    param_str = ', '.join(params)

    # Generate docstring with original C code
    docstring = f'''"""
NumPy Reference Implementation for {func_name}

This is a direct translation of C semantics using explicit loops.
NOT optimized - just correct reference for testing.

Original C code:
{loop_code}
"""'''

    # Start building the function
    code_lines = [
        'import numpy as np',
        '',
        docstring,
        f'def {func_name}_numpy_ref({param_str}):',
    ]

    # Add contiguity check
    for arr in params:
        code_lines.append(f'    {arr} = np.ascontiguousarray({arr})')

    code_lines.append('    n = len(a) if "a" in dir() else len(list(filter(None, [{", ".join(params)}]))[0])')
    code_lines.append('')

    # Translate the inner loop code
    translated = translate_inner_code(inner_code, arrays, has_2d)
    for line in translated:
        code_lines.append('    ' + line)

    return '\n'.join(code_lines)


def translate_inner_code(c_code, arrays, has_2d):
    """
    Translate inner C loop code to Python.
    Handles:
    - for loops
    - array access with indices
    - arithmetic
    - conditionals
    """
    lines = []

    # Split into lines and translate each
    for line in c_code.split('\n'):
        line = line.strip()
        if not line or line == '{':
            continue
        if line == '}':
            lines.append('')  # End of block
            continue

        # Translate for loop
        if line.startswith('for'):
            py_for = translate_for_loop(line)
            lines.append(py_for + ':')
            continue

        # Translate if statement
        if line.startswith('if'):
            py_if = translate_if_statement(line)
            lines.append(py_if + ':')
            continue

        # Translate assignment
        if '=' in line and not line.startswith('//'):
            py_assign = translate_assignment(line)
            if py_assign:
                lines.append(py_assign)
            continue

        # Skip comments
        if line.startswith('//'):
            continue

    return lines


def translate_for_loop(c_for):
    """Translate C for loop to Python for loop"""
    # Match patterns like: for (int i = 0; i < N; i++)
    # or: for (int i = N-1; i >= 0; i--)

    # Extract loop variable, start, end, step
    match = re.search(r'for\s*\(\s*(?:int\s+)?(\w+)\s*=\s*([^;]+);\s*(\w+)\s*([<>=!]+)\s*([^;]+);\s*(\w+)([+\-]{2}|[+\-]=\d+)', c_for)

    if not match:
        # Try simpler pattern
        match = re.search(r'for\s*\(\s*(?:int\s+)?(\w+)\s*=\s*(\d+);\s*\w+\s*<\s*(\w+);\s*\w+\+\+\)', c_for)
        if match:
            var, start, end = match.groups()
            return f'for {var} in range({start}, {end})'
        return '# COULD NOT TRANSLATE: ' + c_for

    var, start, check_var, op, end, inc_var, inc_op = match.groups()

    # Translate C expressions to Python
    start = translate_expr(start)
    end = translate_expr(end)

    # Determine direction and step
    if '++' in inc_op or '+=' in inc_op:
        step_val = 1 if '++' in inc_op else int(re.search(r'\d+', inc_op).group())
        if '<' in op or '<=' in op:
            if '<=' in op:
                return f'for {var} in range({start}, {end}+1, {step_val})' if step_val != 1 else f'for {var} in range({start}, {end}+1)'
            else:
                return f'for {var} in range({start}, {end}, {step_val})' if step_val != 1 else f'for {var} in range({start}, {end})'
    elif '--' in inc_op or '-=' in inc_op:
        step_val = -1 if '--' in inc_op else -int(re.search(r'\d+', inc_op).group())
        if '>' in op or '>=' in op:
            if '>=' in op:
                return f'for {var} in range({start}, {end}-1, {step_val})'
            else:
                return f'for {var} in range({start}, {end}, {step_val})'

    return '# COULD NOT TRANSLATE: ' + c_for


def translate_if_statement(c_if):
    """Translate C if statement to Python"""
    # Extract condition
    match = re.search(r'if\s*\((.+)\)', c_if)
    if match:
        cond = match.group(1)
        cond = translate_expr(cond)
        return f'if {cond}'
    return '# COULD NOT TRANSLATE: ' + c_if


def translate_assignment(c_assign):
    """Translate C assignment to Python"""
    # Remove semicolon and trailing brace
    c_assign = c_assign.rstrip(';').rstrip('}').strip()
    if not c_assign:
        return None

    # Handle compound assignments (+=, -=, etc.)
    for op in ['+=', '-=', '*=', '/=']:
        if op in c_assign:
            parts = c_assign.split(op)
            if len(parts) == 2:
                lhs = translate_expr(parts[0].strip())
                rhs = translate_expr(parts[1].strip())
                return f'{lhs} {op} {rhs}'

    # Handle simple assignment
    if '=' in c_assign:
        parts = c_assign.split('=', 1)
        lhs = translate_expr(parts[0].strip())
        rhs = translate_expr(parts[1].strip())
        return f'{lhs} = {rhs}'

    return None


def translate_expr(c_expr):
    """Translate C expression to Python"""
    expr = c_expr.strip()

    # Replace C constants
    expr = re.sub(r'\bLEN_1D\b', 'n', expr)
    expr = re.sub(r'\bLEN_2D\b', 'int(np.sqrt(n))', expr)
    expr = re.sub(r'\(real_t\)\s*', '', expr)  # Remove casts
    expr = re.sub(r'\(float\)\s*', '', expr)
    expr = re.sub(r'\(double\)\s*', '', expr)

    # Replace 2D array access aa[i][j] -> aa[i, j]
    expr = re.sub(r'(\w+)\[([^\]]+)\]\[([^\]]+)\]', r'\1[\2, \3]', expr)

    # Replace sqrt -> np.sqrt
    expr = re.sub(r'\bsqrt\b', 'np.sqrt', expr)
    expr = re.sub(r'\bfabs\b', 'np.abs', expr)
    expr = re.sub(r'\bexp\b', 'np.exp', expr)
    expr = re.sub(r'\bsin\b', 'np.sin', expr)
    expr = re.sub(r'\bcos\b', 'np.cos', expr)

    return expr


def generate_all_references():
    """Generate NumPy reference for all TSVC functions"""
    output_dir = Path(__file__).parent.parent / 'numpy_references'
    output_dir.mkdir(exist_ok=True)

    for func_name, func_spec in TSVC_FUNCTIONS.items():
        print(f'Generating reference for {func_name}...')
        try:
            code = translate_c_to_numpy(func_name, func_spec)

            output_file = output_dir / f'{func_name}_numpy_ref.py'
            with open(output_file, 'w') as f:
                f.write(code)

            print(f'  ✓ Saved to {output_file}')
        except Exception as e:
            print(f'  ✗ Failed: {e}')


if __name__ == '__main__':
    generate_all_references()
