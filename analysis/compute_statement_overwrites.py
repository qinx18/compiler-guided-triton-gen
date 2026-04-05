#!/usr/bin/env python3
"""
Detect statement overwrite patterns in loops.

This analysis identifies patterns where one statement's result gets overwritten
by another statement in subsequent iterations, making most iterations of the
overwritten statement redundant.

Example s2244:
    for (int i = 0; i < N-1; i++) {
        a[i+1] = b[i] + e[i];    // S0: write a[i+1]
        a[i] = b[i] + c[i];      // S1: write a[i]
    }

S0 writes a[i+1], S1 writes a[i].
At iteration i+1, S1 writes a[i+1], overwriting S0's result from iteration i.
Only S0's result at the last iteration (i=N-2) survives.

Optimization: Skip S0 for iterations 0 to N-3, only execute at i=N-2.
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


def extract_accesses(stmt):
    """Recursively extract all accesses from a statement."""
    reads = []
    writes = []

    def traverse(node):
        if not isinstance(node, dict):
            return
        if node.get('type') == 'access':
            access = {
                'index': node.get('index', ''),
                'ref': node.get('reference', '')
            }
            if node.get('read', 0):
                reads.append(access)
            if node.get('write', 0):
                writes.append(access)
        for key in ['arguments', 'body', 'expr']:
            if key in node:
                if isinstance(node[key], list):
                    for item in node[key]:
                        traverse(item)
                else:
                    traverse(node[key])

    traverse(stmt.get('body', {}))
    return reads, writes


def parse_access_array_and_index(access_str):
    """
    Parse an ISL access string to extract array name and index expression.
    Example: '{ S_0[i] -> a[(1 + i)] }' returns ('a', '1 + i')
    """
    match = re.search(r'(\w+)\[([^\]]+)\](?:\s*\})?$', access_str)
    if match:
        # Clean up the index expression
        idx = match.group(2).strip()
        # Remove outer parentheses if present
        if idx.startswith('(') and idx.endswith(')'):
            idx = idx[1:-1].strip()
        return match.group(1), idx
    return None, None


def extract_index_offset(index_expr, dim_var):
    """
    Extract the offset from a dimension variable in an index expression.
    Example: '1 + i' with dim_var='i' returns 1
    Example: 'i' with dim_var='i' returns 0
    Example: 'i - 1' with dim_var='i' returns -1
    """
    if index_expr is None:
        return None

    index_expr = index_expr.strip()

    # Check if it's just the variable
    if index_expr == dim_var:
        return 0

    # Parse expressions like "1 + i", "i + 1", "-1 + i", "i - 1"
    # Normalize: remove spaces around operators
    expr = re.sub(r'\s*([+\-])\s*', r' \1 ', index_expr)
    parts = expr.split()

    offset = 0
    sign = 1
    for part in parts:
        if part == '+':
            sign = 1
        elif part == '-':
            sign = -1
        elif part == dim_var:
            continue
        else:
            try:
                offset += sign * int(part)
                sign = 1  # Reset sign after using it
            except ValueError:
                return None  # Complex expression

    return offset


def extract_loop_stride(domain):
    """
    Extract loop stride from ISL domain constraint.

    Example: '{ S_0[i] : (i) mod 5 = 0 and 0 <= i <= 31994 }' returns 5
    Returns 1 if no modular constraint found (unit stride).
    """
    # Look for pattern like "(i) mod N = 0" or "i mod N = 0"
    mod_match = re.search(r'\(?\w+\)?\s*mod\s*(\d+)\s*=\s*0', domain)
    if mod_match:
        return int(mod_match.group(1))
    return 1  # Default to unit stride


def analyze_statement_overwrites(statements_data):
    """
    Analyze multiple statements in a loop for overwrite patterns.

    This detects the pattern where:
    - Statement S_i writes to arr[i+k] (forward offset, k > 0)
    - Statement S_j writes to arr[i+m] (m < k)
    - S_j's write at iteration i overwrites S_i's write at iteration i-(k-m)
    - Only S_i's result at the last (k-m) iterations survives

    NOTE: This pattern only applies to unit-stride loops (i++).
    For strided loops (i += stride), statements within each iteration
    write to disjoint indices and cannot overwrite each other.

    Returns dict with:
    - 'applicable': bool - whether overwrite pattern detected
    - 'overwrites': list - detected overwrite pairs
    - 'optimization_advice': str - how to optimize
    """
    result = {
        'applicable': False,
        'overwrites': [],
        'optimization_advice': None,
        'statements': []
    }

    if len(statements_data) < 2:
        return result

    # Extract write info from each statement
    statements = []
    loop_stride = 1  # Will be updated from domain

    for stmt_idx, stmt in enumerate(statements_data):
        reads, writes = extract_accesses(stmt)
        domain = stmt.get('domain', '')

        # Parse domain to get loop dimension
        dim_match = re.search(r'S_\d+\[(\w+)\]', domain)
        dim_var = dim_match.group(1) if dim_match else 'i'

        # Extract stride from first statement's domain
        if stmt_idx == 0:
            loop_stride = extract_loop_stride(domain)

        stmt_info = {
            'idx': stmt_idx,
            'domain': domain,
            'dim_var': dim_var,
            'writes': []
        }

        for w in writes:
            arr, idx = parse_access_array_and_index(w['index'])
            if arr:
                offset = extract_index_offset(idx, dim_var)
                stmt_info['writes'].append({
                    'array': arr,
                    'index_expr': idx,
                    'offset': offset
                })

        statements.append(stmt_info)

    result['statements'] = statements
    result['loop_stride'] = loop_stride

    # For strided loops (stride > 1), statements within each iteration
    # write to indices i+0, i+1, ..., i+(stride-1), which are disjoint
    # from the next iteration's writes at i+stride, i+stride+1, etc.
    # No overwrite pattern applies.
    if loop_stride > 1:
        # Check if all write offsets are less than the stride
        # If so, writes within each iteration are disjoint - no overwrites
        all_offsets_within_stride = True
        for stmt in statements:
            for w in stmt['writes']:
                if w['offset'] is not None and w['offset'] >= loop_stride:
                    all_offsets_within_stride = False
                    break

        if all_offsets_within_stride:
            # This is a manually unrolled loop - no overwrites
            return result

    # Detect overwrite patterns: S_i writes arr[i+k], S_j writes arr[i+m] where k > m
    # S_j at iteration t overwrites S_i at iteration t-(k-m)
    overwrites = []

    for i, s_i in enumerate(statements):
        for j, s_j in enumerate(statements):
            if i == j:
                continue

            for w_i in s_i['writes']:
                for w_j in s_j['writes']:
                    if w_i['array'] != w_j['array']:
                        continue

                    if w_i['offset'] is None or w_j['offset'] is None:
                        continue

                    # Check if S_i writes ahead of S_j (higher offset)
                    offset_diff = w_i['offset'] - w_j['offset']

                    if offset_diff > 0:
                        # S_i writes to arr[i+k], S_j writes to arr[i+m], k > m
                        # At iteration t, S_i writes arr[t+k]
                        # At iteration t+offset_diff, S_j writes arr[t+offset_diff+m] = arr[t+k]
                        # So S_j overwrites S_i's result after offset_diff iterations

                        overwrites.append({
                            'overwritten_stmt': i,
                            'overwriting_stmt': j,
                            'array': w_i['array'],
                            'overwritten_offset': w_i['offset'],
                            'overwriting_offset': w_j['offset'],
                            'offset_diff': offset_diff,
                            'description': f"S{i} writes {w_i['array']}[i{w_i['offset']:+d}], "
                                          f"S{j} writes {w_j['array']}[i{w_j['offset']:+d}] - "
                                          f"S{j} overwrites S{i}'s result after {offset_diff} iteration(s)"
                        })

    result['overwrites'] = overwrites

    if overwrites:
        result['applicable'] = True

        # Generate optimization advice
        advice_lines = [
            "STATEMENT OVERWRITE PATTERN DETECTED",
            "",
            "One statement's output is overwritten by another statement in subsequent iterations.",
            "Only the last iteration(s) of the overwritten statement have effect.",
            "",
            "Detected overwrites:"
        ]

        for ow in overwrites:
            advice_lines.append(f"  - {ow['description']}")

        advice_lines.extend([
            "",
            "OPTIMIZATION: Remove redundant writes - only execute overwritten statement at last iteration(s):",
            ""
        ])

        # Group by overwritten statement
        overwritten_stmts = {}
        for ow in overwrites:
            stmt_idx = ow['overwritten_stmt']
            if stmt_idx not in overwritten_stmts:
                overwritten_stmts[stmt_idx] = ow
            elif ow['offset_diff'] > overwritten_stmts[stmt_idx]['offset_diff']:
                overwritten_stmts[stmt_idx] = ow

        for stmt_idx, ow in overwritten_stmts.items():
            offset_diff = ow['offset_diff']
            advice_lines.extend([
                f"For S{stmt_idx} (writes {ow['array']}[i{ow['overwritten_offset']:+d}]):",
                f"  - Skip for iterations i = 0 to N-1-{offset_diff} (result will be overwritten)",
                f"  - Only execute for the LAST {offset_diff} iteration(s): i = N-{offset_diff} to N-1",
                ""
            ])

        advice_lines.extend([
            "IMPLEMENTATION PATTERN:",
            "```python",
            "# Main loop - execute non-overwritten statements for all iterations",
            "for i in range(N - 1):  # or parallel",
        ])

        # List statements that are NOT overwritten
        non_overwritten = [s['idx'] for s in statements if s['idx'] not in overwritten_stmts]
        for idx in non_overwritten:
            advice_lines.append(f"    # S{idx}: execute for all iterations")

        advice_lines.extend([
            "",
            "# Epilogue - execute overwritten statements only for last iteration(s)",
        ])

        for stmt_idx, ow in overwritten_stmts.items():
            offset_diff = ow['offset_diff']
            if offset_diff == 1:
                advice_lines.append(f"# S{stmt_idx}: execute only at i = N-2 (last iteration)")
                advice_lines.append(f"i = N - 2")
                advice_lines.append(f"# Execute S{stmt_idx}")
            else:
                advice_lines.append(f"# S{stmt_idx}: execute for last {offset_diff} iterations")
                advice_lines.append(f"for i in range(N - 1 - {offset_diff}, N - 1):")
                advice_lines.append(f"    # Execute S{stmt_idx}")

        advice_lines.append("```")

        result['optimization_advice'] = '\n'.join(advice_lines)

    return result


def format_overwrite_for_prompt(overwrite_result):
    """Format overwrite analysis for inclusion in LLM prompt."""
    if not overwrite_result['applicable']:
        return None

    lines = []
    lines.append("=" * 60)
    lines.append("STATEMENT OVERWRITE OPTIMIZATION DETECTED")
    lines.append("=" * 60)
    lines.append("")

    if overwrite_result['optimization_advice']:
        lines.append(overwrite_result['optimization_advice'])

    lines.append("")
    lines.append("=" * 60)

    return '\n'.join(lines)


def analyze_kernel_overwrites(kernel_file):
    """Analyze a kernel file for overwrite patterns."""
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

    return analyze_statement_overwrites(statements)


def main():
    """Test overwrite detection on s244 and s2244."""
    test_kernels = ['s244', 's2244']

    print("=" * 80)
    print("STATEMENT OVERWRITE DETECTION ANALYSIS")
    print("=" * 80)

    for kernel in test_kernels:
        kernel_file = os.path.join(KERNELS_DIR, f"{kernel}.c")
        if not os.path.exists(kernel_file):
            print(f"\n{kernel}: kernel file not found")
            continue

        print(f"\n{'=' * 40}")
        print(f"Kernel: {kernel}")
        print(f"{'=' * 40}")

        # Read and display C code
        with open(kernel_file, 'r') as f:
            c_code = f.read()
        scop_match = re.search(r'#pragma scop\s*(.*?)\s*#pragma endscop', c_code, re.DOTALL)
        if scop_match:
            print(f"\nC Code:\n{scop_match.group(1).strip()}")

        result = analyze_kernel_overwrites(kernel_file)
        if result:
            prompt_text = format_overwrite_for_prompt(result)
            if prompt_text:
                print(f"\n{prompt_text}")
            else:
                print("\nNo overwrite pattern detected")
        else:
            print("\nFailed to analyze kernel")


if __name__ == "__main__":
    main()
