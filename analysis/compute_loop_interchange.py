#!/usr/bin/env python3
"""
Detect loop interchange requirements for multi-statement kernels.

This analysis identifies patterns where multiple statements in the same loop nest
have dependencies along DIFFERENT dimensions, requiring different parallelization
axes for each statement.

Example s233:
    for (int i = 1; i < LEN_2D; i++) {
        for (int j = 1; j < LEN_2D; j++) {
            aa[j][i] = aa[j-1][i] + cc[j][i];   // dependency on j -> parallelize i
        }
        for (int j = 1; j < LEN_2D; j++) {
            bb[j][i] = bb[j][i-1] + cc[j][i];   // dependency on i -> parallelize j
        }
    }

The aa loop has a dependency along the first subscript (j), so it can parallelize
across the second subscript (i). The bb loop has a dependency along the second
subscript (i), making it a prefix sum along i — it must parallelize across j instead.

A single kernel that parallelizes across one dimension will get one statement wrong.
The solution is to use separate kernels (or separate grid launches) with different
parallelization axes for each statement group.
"""

import subprocess
import yaml
import re
import os

PET_PATH = "/home/qinxiao/workspace/pet/pet"
KERNELS_DIR = "/home/qinxiao/workspace/compiler-guided-triton-gen/analysis/kernels"


def run_pet(kernel_file):
    """Run PET on a kernel file and return the YAML output."""
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
        return None

    output = result.stdout
    for op in ['=', '+', '-', '*', '/', '%', '&&', '||', '<', '>', '<=', '>=', '==', '!=']:
        output = re.sub(rf'operation: {re.escape(op)}\s*$', f'operation: "{op}"', output, flags=re.MULTILINE)

    return output


def parse_access_subscripts(access_str):
    """
    Parse an ISL access string to extract array name and subscript expressions.

    Example: '{ S_0[i, j] -> aa[(j), (i)] }' returns ('aa', ['j', 'i'])
    Example: '{ S_1[i, j] -> bb[(j), (-1 + i)] }' returns ('bb', ['j', '-1 + i'])
    """
    # Match array name and subscripts
    match = re.search(r'->\s*(\w+)\[([^\]]*)\]', access_str)
    if not match:
        return None, []

    array_name = match.group(1)
    subscripts_str = match.group(2)

    # Split subscripts by ), ( pattern
    subscripts = []
    for sub in re.findall(r'\(([^)]*)\)', subscripts_str):
        subscripts.append(sub.strip())

    if not subscripts and subscripts_str.strip():
        subscripts = [s.strip() for s in subscripts_str.split(',')]

    return array_name, subscripts


def get_dependency_dim(subscripts, loop_dims):
    """
    Determine which loop dimension has a dependency based on subscript offsets.

    Returns the index of the loop dimension that has a carried dependency,
    or None if no dependency is detected.

    A dependency exists when a subscript references dim-1 or dim+offset (offset != 0).
    """
    for sub_idx, sub_expr in enumerate(subscripts):
        sub_expr_clean = sub_expr.replace('(', '').replace(')', '').strip()
        for dim_idx, dim_var in enumerate(loop_dims):
            if dim_var not in sub_expr_clean:
                continue
            # Check if there's an offset
            # Remove the variable and see what's left
            remaining = sub_expr_clean.replace(dim_var, '').strip()
            remaining = remaining.replace('+', '').replace('-', ' -').strip()
            if remaining:
                # There's an offset — this dimension has a dependency
                try:
                    offset = int(remaining) if remaining else 0
                    if offset != 0:
                        return dim_idx
                except ValueError:
                    pass
    return None


def analyze_loop_interchange(statements_data):
    """
    Analyze statements for loop interchange requirements.

    Detects when different statements in the same loop nest require
    parallelization along different dimensions.

    Returns dict with:
    - 'applicable': bool
    - 'statements': list of statement analysis info
    - 'interchange_advice': str
    """
    result = {
        'applicable': False,
        'statements': [],
        'interchange_advice': None,
    }

    if len(statements_data) < 2:
        return result

    # Analyze each statement's dependency dimension
    stmt_analyses = []
    for stmt_idx, stmt in enumerate(statements_data):
        domain = stmt.get('domain', '')

        # Extract loop dimensions from domain
        dim_match = re.search(r'S_\d+\[([^\]]+)\]', domain)
        if not dim_match:
            continue
        loop_dims = [d.strip() for d in dim_match.group(1).split(',')]

        if len(loop_dims) < 2:
            # Need at least 2D loop for interchange to matter
            continue

        # Extract all accesses
        reads = []
        writes = []

        def traverse(node):
            if not isinstance(node, dict):
                return
            if node.get('type') == 'access':
                access_str = node.get('index', '')
                arr_name, subscripts = parse_access_subscripts(access_str)
                if arr_name:
                    entry = {
                        'array': arr_name,
                        'subscripts': subscripts,
                        'index': access_str,
                    }
                    if node.get('read', 0):
                        reads.append(entry)
                    if node.get('write', 0):
                        writes.append(entry)
            for key in ['arguments', 'body', 'expr']:
                if key in node:
                    if isinstance(node[key], list):
                        for item in node[key]:
                            traverse(item)
                    else:
                        traverse(node[key])

        traverse(stmt.get('body', {}))

        # Find the dependency dimension by comparing read vs write of the same array
        dep_dim = None
        dep_array = None
        parallel_dim = None
        for w in writes:
            for r in reads:
                if w['array'] == r['array'] and len(w['subscripts']) >= 2:
                    dep = get_dependency_dim(r['subscripts'], loop_dims)
                    if dep is not None:
                        dep_dim = dep
                        dep_array = w['array']
                        # The parallel dimension is the other one
                        parallel_dim = 1 - dep_dim if len(loop_dims) == 2 else None
                        break
            if dep_dim is not None:
                break

        stmt_analyses.append({
            'stmt_idx': stmt_idx,
            'loop_dims': loop_dims,
            'dep_dim': dep_dim,
            'dep_array': dep_array,
            'parallel_dim': parallel_dim,
            'reads': reads,
            'writes': writes,
            'domain': domain,
        })

    result['statements'] = stmt_analyses

    # Check if different statements need different parallelization axes
    stmts_with_deps = [s for s in stmt_analyses if s['dep_dim'] is not None]
    if len(stmts_with_deps) < 2:
        return result

    # Check if dependency dimensions differ
    dep_dims = set(s['dep_dim'] for s in stmts_with_deps)
    if len(dep_dims) < 2:
        return result

    # Different statements have dependencies on different dimensions!
    result['applicable'] = True

    # Group statements by their parallel dimension
    groups = {}
    for s in stmts_with_deps:
        pdim = s['parallel_dim']
        if pdim not in groups:
            groups[pdim] = []
        groups[pdim].append(s)

    # Build advice
    advice_lines = [
        "LOOP INTERCHANGE / SPLIT PARALLELIZATION REQUIRED",
        "",
        "This kernel contains multiple statement groups that require DIFFERENT",
        "parallelization axes. Using a single parallelization axis for all statements",
        "will produce INCORRECT results for at least one group.",
        "",
        "Statement analysis:",
    ]

    for s in stmts_with_deps:
        dims = s['loop_dims']
        dep_dim_name = dims[s['dep_dim']]
        par_dim_name = dims[s['parallel_dim']] if s['parallel_dim'] is not None else '?'
        dep_arr = s['dep_array']

        # Show the write pattern
        write_info = s['writes'][0] if s['writes'] else None
        if write_info:
            subs = ', '.join(write_info['subscripts'])
            advice_lines.append(
                f"  - S{s['stmt_idx']}: {dep_arr}[{subs}] has dependency along '{dep_dim_name}' "
                f"→ must parallelize along '{par_dim_name}'"
            )

    advice_lines.extend([
        "",
        "SOLUTION: Use SEPARATE KERNELS for each statement group, each with",
        "its own parallelization axis:",
        "",
    ])

    for pdim, group_stmts in sorted(groups.items(), key=lambda x: (x[0] is None, x[0] if x[0] is not None else 0)):
        dim_name = group_stmts[0]['loop_dims'][pdim] if pdim is not None else '?'
        arrays = [s['dep_array'] for s in group_stmts]
        other_dim_idx = 1 - pdim if pdim is not None else None
        seq_dim_name = group_stmts[0]['loop_dims'][other_dim_idx] if other_dim_idx is not None else '?'
        advice_lines.append(
            f"  Kernel for {', '.join(arrays)}: parallelize across '{dim_name}', "
            f"loop sequentially over '{seq_dim_name}'"
        )

    advice_lines.extend([
        "",
        "CRITICAL: Do NOT parallelize both statements along the same dimension.",
        "Each statement group MUST have its dependency dimension executed sequentially.",
    ])

    result['interchange_advice'] = '\n'.join(advice_lines)
    return result


def format_interchange_for_prompt(result):
    """Format loop interchange analysis for inclusion in LLM prompt."""
    if not result or not result.get('applicable'):
        return None

    lines = []
    lines.append("=" * 60)
    lines.append(result['interchange_advice'])
    lines.append("=" * 60)
    return '\n'.join(lines)


def analyze_kernel_loop_interchange(kernel_name):
    """
    Analyze a kernel for loop interchange requirements.

    Args:
        kernel_name: Name of the kernel (e.g., 's233')

    Returns:
        dict with analysis results, or None if not applicable
    """
    kernel_file = os.path.join(KERNELS_DIR, f"{kernel_name}.c")
    if not os.path.exists(kernel_file):
        return None

    pet_output = run_pet(kernel_file)
    if not pet_output:
        return None

    try:
        data = yaml.safe_load(pet_output)
    except:
        return None

    statements = data.get('statements', [])
    if len(statements) < 2:
        return None

    result = analyze_loop_interchange(statements)

    if result['applicable']:
        return result
    return None


def main():
    """Test loop interchange detection."""
    test_kernels = ['s233', 's2233']

    print("=" * 80)
    print("LOOP INTERCHANGE ANALYSIS")
    print("=" * 80)

    for kernel in test_kernels:
        kernel_file = os.path.join(KERNELS_DIR, f"{kernel}.c")
        if not os.path.exists(kernel_file):
            print(f"\n{kernel}: kernel file not found")
            continue

        print(f"\n{'=' * 40}")
        print(f"Kernel: {kernel}")
        print(f"{'=' * 40}")

        with open(kernel_file, 'r') as f:
            c_code = f.read()
        scop_match = re.search(r'#pragma scop\s*(.*?)\s*#pragma endscop', c_code, re.DOTALL)
        if scop_match:
            print(f"\nC Code:\n{scop_match.group(1).strip()}")

        result = analyze_kernel_loop_interchange(kernel)
        if result:
            print(f"\nApplicable: {result['applicable']}")
            formatted = format_interchange_for_prompt(result)
            if formatted:
                print(f"\n{formatted}")
        else:
            print("\nNo loop interchange requirement detected")


if __name__ == "__main__":
    main()
