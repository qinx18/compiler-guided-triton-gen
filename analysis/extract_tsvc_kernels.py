#!/usr/bin/env python3
"""
Extract TSVC kernel loops and prepare them for PET analysis.
This script extracts functions s000 through s113 and their variants.
"""

import re
import os
import subprocess
import yaml
import json
from collections import defaultdict

# TSVC source file
TSVC_SRC = "/home/qinxiao/workspace/compiler-guided-triton-gen/benchmarks_src/TSVC_2/src/archive/tsvc_orig.c"
OUTPUT_DIR = "/home/qinxiao/workspace/compiler-guided-triton-gen/analysis/kernels"
PET_PATH = "/home/qinxiao/workspace/pet/pet"
RESULTS_DIR = "/home/qinxiao/workspace/compiler-guided-triton-gen/analysis/results"

# Common header for all kernel files
KERNEL_HEADER = """
#define LEN_1D 32000
#define LEN_2D 256

typedef float real_t;

real_t a[LEN_1D], b[LEN_1D], c[LEN_1D], d[LEN_1D], e[LEN_1D];
real_t aa[LEN_2D][LEN_2D], bb[LEN_2D][LEN_2D], cc[LEN_2D][LEN_2D], tt[LEN_2D][LEN_2D];
real_t flat_2d_array[LEN_2D*LEN_2D];
int indx[LEN_1D];

"""

def extract_functions(source_file):
    """Extract all function bodies from the TSVC source file."""
    with open(source_file, 'r') as f:
        content = f.read()

    # Pattern to match function definitions like "real_t s000(..." to the closing brace
    # We'll use a simpler approach: find function starts and extract between braces
    functions = {}

    # Match functions like: real_t s000(struct args_t * func_args) and real_t va(struct args_t *)
    # Includes both s### functions and v functions (like va, vag, vsumr, etc.)
    func_pattern = r'real_t\s+(s\d+|v\w+)\s*\([^)]*\)\s*\{'

    for match in re.finditer(func_pattern, content):
        func_name = match.group(1)
        start = match.end() - 1  # Start at the opening brace

        # Find the matching closing brace
        brace_count = 1
        pos = start + 1
        while brace_count > 0 and pos < len(content):
            if content[pos] == '{':
                brace_count += 1
            elif content[pos] == '}':
                brace_count -= 1
            pos += 1

        func_body = content[start:pos]
        functions[func_name] = func_body

    return functions


def extract_local_variables(func_body):
    """Extract local variable declarations from function body (e.g., int m = 1;)."""
    variables = []

    # Match variable declarations: type varname = value; or type var1, var2;
    # Common patterns in TSVC: int m = 1; real_t sum = 0.0; int ip, i1, i2;
    var_patterns = [
        r'^\s*(int\s+(?!nl\b)(?!i\b)(?!j\b)(?!k\b)\w+(?:\s*,\s*\w+)*\s*(?:=\s*[^;]+)?)\s*;',
        r'^\s*(real_t\s+\w+(?:\s*,\s*\w+)*\s*(?:=\s*[^;]+)?)\s*;',
        r'^\s*(float\s+\w+(?:\s*,\s*\w+)*\s*(?:=\s*[^;]+)?)\s*;',
        r'^\s*(double\s+\w+(?:\s*,\s*\w+)*\s*(?:=\s*[^;]+)?)\s*;',
    ]

    lines = func_body.split('\n')
    for line in lines:
        # Skip lines inside loops or after gettimeofday
        if 'for' in line and ('nl' in line or 'i =' in line or 'j =' in line):
            continue
        if 'gettimeofday' in line:
            continue

        for pattern in var_patterns:
            match = re.match(pattern, line)
            if match:
                decl = match.group(1).strip()
                variables.append(decl + ';')
                break

    return variables


def extract_inner_loops(func_body, func_name):
    """Extract the innermost loop(s) that do the actual computation."""
    # Remove the outer timing/initialization boilerplate
    # Look for the pattern: for (int nl = ...) { ... for loops ... }

    # Find the main loop content (skip the nl loop wrapper if present)
    lines = func_body.split('\n')

    # Find where actual loops start (skip initialise_arrays, gettimeofday, etc.)
    loop_lines = []
    in_main_loop = False
    brace_depth = 0
    skip_nl_loop = True

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Skip initialization and timing code
        if 'initialise_arrays' in stripped or 'gettimeofday' in stripped:
            continue
        if 'dummy(' in stripped or 'calc_checksum' in stripped:
            continue
        if stripped.startswith('//'):
            continue

        # Check for the nl iteration loop (outer timing loop)
        if 'for' in stripped and 'nl' in stripped and skip_nl_loop:
            # This is the outer timing loop, we want to extract its body
            in_main_loop = True
            # Find the opening brace
            if '{' in stripped:
                brace_depth = 1
                # Extract just what's after the brace
                brace_pos = stripped.find('{')
                remainder = stripped[brace_pos+1:].strip()
                if remainder:
                    loop_lines.append(remainder)
            continue

        if in_main_loop:
            # Count braces to know when we exit
            for char in stripped:
                if char == '{':
                    brace_depth += 1
                elif char == '}':
                    brace_depth -= 1

            if brace_depth > 0:
                # Skip dummy calls inside the loop
                if 'dummy(' not in stripped:
                    loop_lines.append(line)
            else:
                break

    # If we couldn't find the pattern, try simpler extraction
    if not loop_lines:
        # Just look for for loops
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('for') and 'nl' not in stripped:
                # Found a computational loop
                # Extract from here to the matching close
                start_idx = i
                brace_depth = 0
                for j in range(i, len(lines)):
                    for char in lines[j]:
                        if char == '{':
                            brace_depth += 1
                        elif char == '}':
                            brace_depth -= 1
                    loop_lines.append(lines[j])
                    if brace_depth == 0 and '{' in ''.join(lines[i:j+1]):
                        break
                break

    return '\n'.join(loop_lines)


def create_kernel_file(func_name, loop_code, output_dir, local_vars=None):
    """Create a C file with the kernel wrapped in scop pragmas."""
    os.makedirs(output_dir, exist_ok=True)

    # Clean up the loop code
    loop_code = loop_code.strip()
    if not loop_code:
        return None

    # Build the kernel content
    kernel_content = KERNEL_HEADER

    # Add local variables as global declarations (so PET can see them as parameters)
    if local_vars:
        kernel_content += "// Local variables from original function\n"
        for var in local_vars:
            kernel_content += f"{var}\n"
        kernel_content += "\n"

    kernel_content += f"""void {func_name}_kernel() {{
#pragma scop
{loop_code}
#pragma endscop
}}
"""

    filepath = os.path.join(output_dir, f"{func_name}.c")
    with open(filepath, 'w') as f:
        f.write(kernel_content)

    return filepath


def run_pet(kernel_file):
    """Run PET on a kernel file and return the YAML output."""
    try:
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = '/home/qinxiao/workspace/pet/isl/.libs:' + env.get('LD_LIBRARY_PATH', '')

        result = subprocess.run(
            [PET_PATH, kernel_file],
            capture_output=True,
            text=True,
            timeout=30,
            env=env
        )

        if result.returncode != 0:
            return None, result.stderr

        return result.stdout, None
    except subprocess.TimeoutExpired:
        return None, "Timeout"
    except Exception as e:
        return None, str(e)


def parse_pet_output(yaml_output):
    """Parse PET YAML output and extract dependency information."""
    try:
        data = yaml.safe_load(yaml_output)

        if not data:
            return None

        result = {
            'arrays': [],
            'statements': [],
            'schedule': data.get('schedule', ''),
            'accesses': {'reads': [], 'writes': []}
        }

        # Extract array information
        for arr in data.get('arrays', []):
            result['arrays'].append({
                'extent': arr.get('extent', ''),
                'element_type': arr.get('element_type', '')
            })

        # Extract statement information and access patterns
        for stmt in data.get('statements', []):
            domain = stmt.get('domain', '')
            result['statements'].append({
                'domain': domain,
                'line': stmt.get('line', 0)
            })

            # Recursively extract accesses from body
            def extract_accesses(node):
                if not isinstance(node, dict):
                    return
                if node.get('type') == 'access':
                    access_info = {
                        'index': node.get('index', ''),
                        'read': node.get('read', 0),
                        'write': node.get('write', 0)
                    }
                    if access_info['read']:
                        result['accesses']['reads'].append(access_info['index'])
                    if access_info['write']:
                        result['accesses']['writes'].append(access_info['index'])

                # Recurse into arguments
                for arg in node.get('arguments', []):
                    extract_accesses(arg)

                # Recurse into body
                if 'body' in node:
                    extract_accesses(node['body'])

                if 'expr' in node:
                    extract_accesses(node['expr'])

            if 'body' in stmt:
                extract_accesses(stmt['body'])

        return result

    except Exception as e:
        return None


def classify_dependency_pattern(analysis):
    """Classify the dependency pattern of a kernel."""
    if not analysis:
        return "unknown"

    reads = analysis['accesses']['reads']
    writes = analysis['accesses']['writes']

    # Check for various patterns
    patterns = []

    # Check if any array is both read and written
    read_arrays = set()
    write_arrays = set()

    for r in reads:
        # Extract array name from ISL format like "{ S_0[i] -> a[(i)] }"
        match = re.search(r'->\s*(\w+)\[', r)
        if match:
            read_arrays.add(match.group(1))

    for w in writes:
        match = re.search(r'->\s*(\w+)\[', w)
        if match:
            write_arrays.add(match.group(1))

    # Check for self-dependencies (same array read and written)
    self_deps = read_arrays & write_arrays
    if self_deps:
        patterns.append(f"self_dependency:{','.join(sorted(self_deps))}")

    # Check for multiple read arrays
    if len(read_arrays) > 1:
        patterns.append(f"multi_read:{len(read_arrays)}")

    # Check for 2D array access
    for r in reads + writes:
        if re.search(r'\]\[', r) or r.count('[') > 1:
            patterns.append("2d_access")
            break

    # Check for reduction patterns (same index pattern in read and write)
    for w in writes:
        for r in reads:
            # Simple check: same array with offset access
            w_match = re.search(r'(\w+)\[([^\]]+)\]', w)
            r_match = re.search(r'(\w+)\[([^\]]+)\]', r)
            if w_match and r_match:
                if w_match.group(1) == r_match.group(1):
                    if w_match.group(2) != r_match.group(2):
                        patterns.append("offset_dependency")
                        break

    if not patterns:
        if not self_deps:
            patterns.append("independent")
        else:
            patterns.append("simple_dependency")

    return "|".join(sorted(set(patterns)))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Extracting functions from {TSVC_SRC}...")
    functions = extract_functions(TSVC_SRC)

    # Include all TSVC functions: s### functions and v functions
    target_funcs = {}
    for name, body in functions.items():
        # Match any s### pattern or v functions (like va, vag, vsumr)
        if re.match(r's\d{3,5}$', name) or re.match(r'v\w+$', name):
            target_funcs[name] = body

    print(f"Found {len(target_funcs)} target functions")

    results = {}
    for func_name in sorted(target_funcs.keys()):
        print(f"Processing {func_name}...")

        func_body = target_funcs[func_name]
        local_vars = extract_local_variables(func_body)
        loop_code = extract_inner_loops(func_body, func_name)

        if not loop_code.strip():
            print(f"  Warning: Could not extract loop from {func_name}")
            continue

        if local_vars:
            print(f"  Found local vars: {local_vars}")

        kernel_file = create_kernel_file(func_name, loop_code, OUTPUT_DIR, local_vars)
        if not kernel_file:
            continue

        pet_output, error = run_pet(kernel_file)
        if error:
            print(f"  PET error for {func_name}: {error[:100]}")
            results[func_name] = {'error': error}
            continue

        analysis = parse_pet_output(pet_output)
        if analysis:
            pattern = classify_dependency_pattern(analysis)
            results[func_name] = {
                'analysis': analysis,
                'pattern': pattern
            }
            print(f"  Pattern: {pattern}")
        else:
            results[func_name] = {'error': 'Failed to parse PET output'}

    # Group functions by pattern
    pattern_groups = defaultdict(list)
    for func_name, result in results.items():
        if 'pattern' in result:
            pattern_groups[result['pattern']].append(func_name)

    # Print summary
    print("\n" + "="*60)
    print("DEPENDENCY PATTERN GROUPS")
    print("="*60)

    for pattern, funcs in sorted(pattern_groups.items()):
        print(f"\n{pattern}:")
        print(f"  Functions: {', '.join(sorted(funcs))}")

    # Save results
    results_file = os.path.join(RESULTS_DIR, "dependency_analysis.json")
    with open(results_file, 'w') as f:
        json.dump({
            'functions': results,
            'groups': dict(pattern_groups)
        }, f, indent=2)

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
