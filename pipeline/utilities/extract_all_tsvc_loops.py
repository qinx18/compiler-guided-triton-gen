#!/usr/bin/env python3
"""
Extract all TSVC function loops from tsvc_orig.c

Extracts only the main computation loop, removing:
- initialise_arrays(__func__)
- gettimeofday() calls
- return checksum() statements
"""

import re
from pathlib import Path
from typing import Dict

TSVC_SOURCE = Path("/home/qinxiao/workspace/compiler-guided-triton-gen/benchmarks_src/TSVC_2/src/archive/tsvc_orig.c")


def extract_function_loop(content: str, func_name: str) -> str:
    """Extract only the main computation loop for a function"""

    # Find function definition
    pattern = rf"real_t {func_name}\(struct args_t \* func_args\)\s*{{"
    match = re.search(pattern, content)
    if not match:
        return None

    # Get function body (up to 10000 chars to be safe)
    start_pos = match.end()
    func_body = content[start_pos:start_pos + 10000]

    # Find the return statement to know where function ends
    return_match = re.search(r'return\s+calc_checksum', func_body)
    if return_match:
        func_body = func_body[:return_match.start()]

    # Extract just the for loops - look for the outermost for loop
    # Pattern: for (int nl = 0; ...) { ... }
    for_match = re.search(r'for\s*\(\s*int\s+nl\s*=', func_body)
    if not for_match:
        # Some functions might not have nl loop, look for direct for loop
        for_match = re.search(r'for\s*\(\s*int\s+i\s*=', func_body)

    if not for_match:
        return None

    # Find the matching closing brace for this for loop
    loop_start_pos = for_match.start()
    # Find the opening brace
    brace_pos = func_body.find('{', for_match.end())
    if brace_pos == -1:
        return None

    # Count braces to find matching close
    brace_count = 1
    pos = brace_pos + 1
    while brace_count > 0 and pos < len(func_body):
        if func_body[pos] == '{':
            brace_count += 1
        elif func_body[pos] == '}':
            brace_count -= 1
        pos += 1

    if brace_count != 0:
        return None

    # Extract the complete loop
    loop_code = func_body[loop_start_pos:pos].strip()

    # Remove dummy() calls - they're just for preventing optimization
    # Use DOTALL to match across lines
    loop_code = re.sub(r'\s*dummy\([^;]+\);', '', loop_code, flags=re.DOTALL)

    # Clean up multiple empty lines and lines with only whitespace
    lines = [line.rstrip() for line in loop_code.split('\n') if line.strip()]
    loop_code = '\n'.join(lines)

    return loop_code if loop_code else None


def analyze_arrays(loop_code: str) -> Dict[str, str]:
    """Analyze which arrays are read/written"""
    arrays = {}

    # Common TSVC array names
    array_names = ['a', 'b', 'c', 'd', 'e', 'x', 'y', 'z', 'xx', 'yy', 'zz',
                   'flat_2d_array', 'flat_3d_array', 'aa', 'bb', 'cc', 'tt',
                   'indx', 'index', 'ip']

    for arr in array_names:
        # Check for writes: arr[...] = or arr[...] += etc
        write_pattern = rf'\b{arr}\s*\[[^\]]+\]\s*[+\-*/]?='
        has_writes = bool(re.search(write_pattern, loop_code))

        # Check for reads: arr[...] on right side
        read_pattern = rf'\b{arr}\s*\[[^\]]+\]'
        has_reads = bool(re.search(read_pattern, loop_code))

        # Check for array passed to function calls: function(arr, ...)
        # Exclude control structures and type casts
        func_call_pattern = rf'\b(?!if|for|while|real_t|int|float|double)\w+\s*\([^)]*\b{arr}\b[^)]*\)'
        in_func_call = bool(re.search(func_call_pattern, loop_code))

        # Check for pointer assignments: ptr = arr;
        pointer_pattern = rf'\*[^=]*=\s*{arr}\b'
        has_pointer = bool(re.search(pointer_pattern, loop_code))

        if has_writes and has_reads:
            arrays[arr] = 'rw'
        elif has_writes:
            arrays[arr] = 'w'
        elif has_reads or in_func_call or has_pointer:
            arrays[arr] = 'r'

    return arrays


def has_2d_arrays(loop_code: str) -> bool:
    """Check if loop uses 2D array indexing like arr[i][j]"""
    # Pattern: array_name[index1][index2]
    pattern_2d = r'\b[a-z_]+\[[^\]]+\]\[[^\]]+\]'
    return bool(re.search(pattern_2d, loop_code))


def has_offset_access(loop_code: str) -> bool:
    """Check if loop uses offset array access like a[i+1]"""
    offset_pattern = r'\[[^\]]*[+\-]\s*\d+\s*\]'
    return bool(re.search(offset_pattern, loop_code))


def has_conditional(loop_code: str) -> bool:
    """Check if loop has if statements"""
    return bool(re.search(r'\bif\s*\(', loop_code))


def has_reduction(loop_code: str) -> bool:
    """Check if loop has reduction (sum, max, etc.)"""
    reduction_keywords = ['sum', 'max', 'min', 'prod']
    for keyword in reduction_keywords:
        if re.search(rf'\b{keyword}\s*[+\-*/]?=', loop_code):
            return True
    return False


def extract_scalar_params(loop_code: str, arrays: Dict[str, str]) -> Dict[str, any]:
    """Extract scalar parameters used in the loop (not arrays)"""
    scalar_params = {}

    # Get list of known array names
    array_names = set(arrays.keys())

    # Find locally assigned variables (k = 1, j = 1, sum = 0) - these are NOT parameters
    local_vars = set()
    # Pattern: variable = value (where value is a number or simple expression)
    assignment_pattern = r'^\s*([a-z_][a-z0-9_]*)\s*=\s*[0-9.()\-+*/\s]+'
    for line in loop_code.split('\n'):
        match = re.search(assignment_pattern, line.strip())
        if match:
            var = match.group(1)
            local_vars.add(var)

    # C keywords and special identifiers to skip
    c_keywords = {'for', 'int', 'real_t', 'nl', 'if', 'else', 'while', 'return'}
    # LEN_1D, LEN_2D, LEN are derived from array size, not parameters
    len_constants = {'len_1d', 'len_2d', 'len', 'len_3d'}
    # Common loop control variables (when used as loop variable, not parameter)
    loop_vars_in_for = set()
    # Find loop variables: for (int i = ...) or for (int j = ...)
    for_var_pattern = r'for\s*\(\s*int\s+([a-z_][a-z0-9_]*)\s*='
    for match in re.finditer(for_var_pattern, loop_code):
        loop_vars_in_for.add(match.group(1))

    # Pattern 1: Scalars in conditionals
    # Match: if (k > 0), if (i+1 < mid), if (x > (real_t)0.)
    cond_pattern = r'if\s*\([^)]+\)'
    for match in re.finditer(cond_pattern, loop_code):
        cond_expr = match.group(0)
        # Find all variable-like tokens
        var_tokens = re.findall(r'\b([a-z_][a-z0-9_]*)\b', cond_expr.lower())
        for var in var_tokens:
            if (var not in c_keywords and
                var not in len_constants and
                var not in array_names and
                var not in local_vars and
                var not in loop_vars_in_for and
                var != 'if'):
                scalar_params[var] = 'scalar'

    # Pattern 2: Scalars used in array indexing
    # Match: a[i + m], a[i + k], aa[j][i], aa[k][i-1], b[ip[i]] * s
    index_pattern = r'\[([^\]]+)\]'
    for match in re.finditer(index_pattern, loop_code):
        index_expr = match.group(1)
        # Find all variable-like tokens in the index expression
        var_tokens = re.findall(r'\b([a-z_][a-z0-9_]*)\b', index_expr.lower())
        for var in var_tokens:
            if (var not in c_keywords and
                var not in len_constants and
                var not in array_names and
                var not in local_vars and
                var not in loop_vars_in_for):
                scalar_params[var] = 'scalar'

    # Pattern 3: Scalars in arithmetic expressions
    # Match: a[i] = ... + s1 + s2 + ..., a[i] += alpha * b[i], b[i] * s
    # Look for variables in RHS of assignments (after = or += or *= etc)
    arith_pattern = r'[+\-*/]\s*([a-z_][a-z0-9_]*)\b'
    for match in re.finditer(arith_pattern, loop_code):
        var = match.group(1).lower()
        if (var not in c_keywords and
            var not in len_constants and
            var not in array_names and
            var not in local_vars and
            var not in loop_vars_in_for):
            scalar_params[var] = 'scalar'

    # Pattern 4: Scalars in loop control and bounds
    # Match: for (int i = 0; i < m; i++), for (int nl = 0; nl < iterations; ...)
    for_pattern = r'for\s*\([^)]+\)'
    for match in re.finditer(for_pattern, loop_code):
        for_stmt = match.group(0)
        # Extract variables from the for statement
        var_tokens = re.findall(r'\b([a-z_][a-z0-9_]*)\b', for_stmt.lower())
        for var in var_tokens:
            if (var not in c_keywords and
                var not in len_constants and
                var not in array_names and
                var not in local_vars and
                var not in loop_vars_in_for and
                var != 'for'):
                scalar_params[var] = 'scalar'

    # Pattern 5: Multi-character scalars used as multipliers
    # Match: alpha * b[i], mid, iterations
    # Look for variables that appear in expressions but aren't arrays or loop vars
    all_vars_pattern = r'\b([a-z_][a-z0-9_]*)\b'
    for match in re.finditer(all_vars_pattern, loop_code):
        var = match.group(1).lower()
        # Only consider multi-character names (2+ chars) for this pattern
        if len(var) >= 2:
            if (var not in c_keywords and
                var not in len_constants and
                var not in array_names and
                var not in local_vars and
                var not in loop_vars_in_for):
                # Check if it appears in a context that suggests it's a scalar
                # (next to operators, in indexing, in conditionals)
                var_context_pattern = rf'[\[\(+\-*/=<>]\s*{re.escape(var)}\b|\b{re.escape(var)}\s*[\]\)+\-*/=<>]'
                if re.search(var_context_pattern, loop_code, re.IGNORECASE):
                    scalar_params[var] = 'scalar'

    # Remove 'i' if it somehow got added (primary loop variable)
    scalar_params.pop('i', None)

    return scalar_params


def extract_all_functions() -> Dict[str, Dict]:
    """Extract all TSVC functions"""

    if not TSVC_SOURCE.exists():
        print(f"Error: {TSVC_SOURCE} not found")
        return {}

    with open(TSVC_SOURCE, 'r') as f:
        content = f.read()

    # Find all function names from the time_function calls
    # This ensures we get the exact list of functions to test
    time_func_pattern = r'time_function\(&(s\w+|v\w+),'
    func_names = []

    for match in re.finditer(time_func_pattern, content):
        func_name = match.group(1)
        if func_name not in func_names:
            func_names.append(func_name)

    print(f"Found {len(func_names)} functions to extract")

    # Extract each function
    functions = {}
    for func_name in func_names:
        loop_code = extract_function_loop(content, func_name)

        if not loop_code:
            print(f"⚠ Could not extract {func_name}")
            continue

        # Analyze the loop
        arrays = analyze_arrays(loop_code)
        has_offset = has_offset_access(loop_code)
        has_cond = has_conditional(loop_code)
        has_red = has_reduction(loop_code)
        has_2d = has_2d_arrays(loop_code)
        scalar_params = extract_scalar_params(loop_code, arrays)

        functions[func_name] = {
            'name': func_name,
            'loop_code': loop_code,
            'arrays': arrays,
            'has_offset': has_offset,
            'has_conditional': has_cond,
            'has_reduction': has_red,
            'has_2d_arrays': has_2d,
            'scalar_params': scalar_params,
        }

        scalars_str = f", scalars={list(scalar_params.keys())}" if scalar_params else ""
        print(f"✓ {func_name}: arrays={list(arrays.keys())}, offset={has_offset}, cond={has_cond}, red={has_red}, 2d={has_2d}{scalars_str}")

    return functions


def save_to_python_db(functions: Dict, output_file: Path):
    """Save extracted functions to Python file"""

    with open(output_file, 'w') as f:
        f.write('"""\n')
        f.write('TSVC Function Database\n')
        f.write('Auto-extracted from tsvc_orig.c\n')
        f.write('"""\n\n')
        f.write('TSVC_FUNCTIONS = {\n')

        for func_name in sorted(functions.keys()):
            data = functions[func_name]

            # Escape the loop code for Python string
            loop_code_escaped = data['loop_code'].replace('\\', '\\\\').replace('"""', r'\"\"\"')

            f.write(f'    "{func_name}": {{\n')
            f.write(f'        "name": "{func_name}",\n')
            f.write(f'        "loop_code": """\n{loop_code_escaped}\n""",\n')
            f.write(f'        "arrays": {data["arrays"]},\n')
            f.write(f'        "has_offset": {data["has_offset"]},\n')
            f.write(f'        "has_conditional": {data["has_conditional"]},\n')
            f.write(f'        "has_reduction": {data["has_reduction"]},\n')
            f.write(f'        "has_2d_arrays": {data["has_2d_arrays"]},\n')
            f.write(f'        "scalar_params": {data["scalar_params"]},\n')
            f.write(f'    }},\n')

        f.write('}\n')


def main():
    """Main extraction routine"""
    print("="*70)
    print("Extracting TSVC Functions from tsvc_orig.c")
    print("="*70)

    functions = extract_all_functions()

    print(f"\n{'='*70}")
    print(f"Total functions extracted: {len(functions)}")
    print(f"{'='*70}")

    # Save to Python database
    output_file = Path(__file__).parent / "tsvc_functions_db.py"
    save_to_python_db(functions, output_file)

    print(f"\n✅ Saved to: {output_file}")

    # Print statistics
    print(f"\nStatistics:")
    print(f"  With offset access: {sum(1 for f in functions.values() if f['has_offset'])}")
    print(f"  With conditionals: {sum(1 for f in functions.values() if f['has_conditional'])}")
    print(f"  With reductions: {sum(1 for f in functions.values() if f['has_reduction'])}")
    print(f"  With 1 array: {sum(1 for f in functions.values() if len(f['arrays']) == 1)}")
    print(f"  With 2 arrays: {sum(1 for f in functions.values() if len(f['arrays']) == 2)}")
    print(f"  With 3+ arrays: {sum(1 for f in functions.values() if len(f['arrays']) >= 3)}")


if __name__ == "__main__":
    main()
