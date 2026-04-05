#!/usr/bin/env python3
"""
Indirect Addressing Analysis Module

Detects patterns where array indices come from another array, e.g.:
- Gather: b[ip[i]] - reading from scattered locations
- Scatter: a[ip[i]] = expr - writing to scattered locations
- Gather reduction: sum += a[i] * b[ip[i]] - reduction with gather

These patterns are fully parallelizable using Triton's indirect load/store.
"""

import os
import re
import yaml
from typing import Optional, Dict, List

# Path configuration
KERNELS_DIR = "/home/qinxiao/workspace/compiler-guided-triton-gen/analysis/kernels"
PET_PATH = "/home/qinxiao/workspace/pet/pet"


def run_pet(kernel_file: str) -> Optional[str]:
    """Run PET on a kernel file and return the YAML output."""
    import subprocess
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/home/qinxiao/workspace/pet/isl/.libs:' + env.get('LD_LIBRARY_PATH', '')

    try:
        result = subprocess.run(
            [PET_PATH, kernel_file],
            capture_output=True,
            text=True,
            env=env,
            timeout=30
        )
        if result.returncode == 0:
            return result.stdout
    except Exception:
        pass
    return None


def detect_indirect_addressing_from_code(c_code: str) -> Dict:
    """
    Detect indirect addressing patterns from C code using regex.

    Patterns detected:
    - Gather: array[index_array[i]] on RHS
    - Scatter: array[index_array[i]] = ... on LHS
    - Gather reduction: sum += a[i] * b[index[i]]

    Returns:
        dict with pattern details
    """
    result = {
        'has_indirect_addressing': False,
        'patterns': [],
        'index_arrays': [],
        'data_arrays': [],
        'is_gather': False,
        'is_scatter': False,
        'is_gather_reduction': False,
    }

    # Normalize code
    code = c_code.replace('\n', ' ').replace('\t', ' ')

    # Pattern 1: Gather in reduction context
    # sum += a[i] * b[ip[i]] or sum += b[ip[i]] * a[i]
    gather_reduction_match = re.search(
        r'(\w+)\s*\+=\s*(\w+)\[(\w+)\]\s*\*\s*(\w+)\[(\w+)\[(\w+)\]\]',
        code
    )
    if not gather_reduction_match:
        # Try reversed order: sum += b[ip[i]] * a[i]
        gather_reduction_match = re.search(
            r'(\w+)\s*\+=\s*(\w+)\[(\w+)\[(\w+)\]\]\s*\*\s*(\w+)\[(\w+)\]',
            code
        )
        if gather_reduction_match:
            result['has_indirect_addressing'] = True
            result['is_gather'] = True
            result['is_gather_reduction'] = True
            acc_var = gather_reduction_match.group(1)
            data_arr = gather_reduction_match.group(2)
            idx_arr = gather_reduction_match.group(3)
            loop_var = gather_reduction_match.group(4)
            other_arr = gather_reduction_match.group(5)
            result['patterns'].append({
                'type': 'gather_reduction',
                'accumulator': acc_var,
                'data_array': data_arr,
                'index_array': idx_arr,
                'other_array': other_arr,
                'loop_var': loop_var,
                'expression': f'{acc_var} += {data_arr}[{idx_arr}[{loop_var}]] * {other_arr}[{loop_var}]'
            })
            result['index_arrays'].append(idx_arr)
            result['data_arrays'].extend([data_arr, other_arr])

    if gather_reduction_match and not result['is_gather_reduction']:
        result['has_indirect_addressing'] = True
        result['is_gather'] = True
        result['is_gather_reduction'] = True
        acc_var = gather_reduction_match.group(1)
        other_arr = gather_reduction_match.group(2)
        loop_var = gather_reduction_match.group(3)
        data_arr = gather_reduction_match.group(4)
        idx_arr = gather_reduction_match.group(5)
        result['patterns'].append({
            'type': 'gather_reduction',
            'accumulator': acc_var,
            'data_array': data_arr,
            'index_array': idx_arr,
            'other_array': other_arr,
            'loop_var': loop_var,
            'expression': f'{acc_var} += {other_arr}[{loop_var}] * {data_arr}[{idx_arr}[{loop_var}]]'
        })
        result['index_arrays'].append(idx_arr)
        result['data_arrays'].extend([data_arr, other_arr])

    # Pattern 2: Simple gather (RHS): expr = ... array[index_array[i]] ...
    # But not already captured as gather_reduction
    if not result['is_gather_reduction']:
        gather_matches = re.findall(r'(\w+)\[(\w+)\[(\w+)\]\]', code)
        for match in gather_matches:
            data_arr, idx_arr, loop_var = match
            # Check if this is on the RHS (not being assigned to)
            # Look for pattern where this is NOT the target of assignment
            assign_pattern = rf'{data_arr}\[{idx_arr}\[{loop_var}\]\]\s*='
            if not re.search(assign_pattern, code):
                result['has_indirect_addressing'] = True
                result['is_gather'] = True
                if idx_arr not in result['index_arrays']:
                    result['index_arrays'].append(idx_arr)
                if data_arr not in result['data_arrays']:
                    result['data_arrays'].append(data_arr)
                result['patterns'].append({
                    'type': 'gather',
                    'data_array': data_arr,
                    'index_array': idx_arr,
                    'loop_var': loop_var,
                    'expression': f'{data_arr}[{idx_arr}[{loop_var}]]'
                })

    # Pattern 3: Scatter (LHS): array[index_array[i]] = expr
    scatter_matches = re.findall(r'(\w+)\[(\w+)\[(\w+)\]\]\s*=', code)
    for match in scatter_matches:
        data_arr, idx_arr, loop_var = match
        result['has_indirect_addressing'] = True
        result['is_scatter'] = True
        if idx_arr not in result['index_arrays']:
            result['index_arrays'].append(idx_arr)
        if data_arr not in result['data_arrays']:
            result['data_arrays'].append(data_arr)
        result['patterns'].append({
            'type': 'scatter',
            'data_array': data_arr,
            'index_array': idx_arr,
            'loop_var': loop_var,
            'expression': f'{data_arr}[{idx_arr}[{loop_var}]] = ...'
        })

    # Deduplicate
    result['index_arrays'] = list(set(result['index_arrays']))
    result['data_arrays'] = list(set(result['data_arrays']))

    return result


def detect_indirect_addressing_from_pet(kernel_file: str) -> Dict:
    """
    Detect indirect addressing using PET analysis.

    Looks for accesses where the index expression contains another array access.
    """
    result = {
        'has_indirect_addressing': False,
        'nested_accesses': [],
    }

    pet_output = run_pet(kernel_file)
    if not pet_output:
        return result

    try:
        data = yaml.safe_load(pet_output)
    except:
        return result

    def find_nested_array_access(node, depth=0):
        """Find array accesses that have other array accesses in their index."""
        nested = []
        if isinstance(node, dict):
            if node.get('type') == 'access':
                index = node.get('index', '')
                # Check if the index expression contains another array reference
                # This is tricky with ISL format, but we can look for nested patterns
                if '->' in index:
                    # Count the depth of array references
                    parts = index.split('->')
                    if len(parts) > 2:  # More than one level of indirection
                        nested.append({
                            'index': index,
                            'read': node.get('read', 0),
                            'write': node.get('write', 0)
                        })

            for v in node.values():
                nested.extend(find_nested_array_access(v, depth + 1))
        elif isinstance(node, list):
            for item in node:
                nested.extend(find_nested_array_access(item, depth + 1))
        return nested

    for stmt in data.get('statements', []):
        body = stmt.get('body', {})
        expr = body.get('expr', {})
        nested = find_nested_array_access(expr)
        if nested:
            result['has_indirect_addressing'] = True
            result['nested_accesses'].extend(nested)

    return result


def analyze_indirect_addressing(kernel_name: str) -> Optional[Dict]:
    """
    Analyze a kernel for indirect addressing patterns.

    Args:
        kernel_name: Name of the kernel (e.g., 's4115')

    Returns:
        dict with complete indirect addressing analysis
    """
    kernel_file = os.path.join(KERNELS_DIR, f"{kernel_name}.c")
    if not os.path.exists(kernel_file):
        return None

    # Read C code
    with open(kernel_file, 'r') as f:
        c_code = f.read()

    # Code-based detection
    code_result = detect_indirect_addressing_from_code(c_code)

    # PET-based detection (supplementary)
    pet_result = detect_indirect_addressing_from_pet(kernel_file)

    # Combine results
    result = {
        'kernel': kernel_name,
        'has_indirect_addressing': code_result['has_indirect_addressing'] or pet_result['has_indirect_addressing'],
        'is_gather': code_result['is_gather'],
        'is_scatter': code_result['is_scatter'],
        'is_gather_reduction': code_result['is_gather_reduction'],
        'patterns': code_result['patterns'],
        'index_arrays': code_result['index_arrays'],
        'data_arrays': code_result['data_arrays'],
    }

    return result


def build_indirect_addressing_instructions(result: Optional[Dict]) -> str:
    """
    Build prompt instructions for detected indirect addressing patterns.

    Returns a string to be included in the LLM prompt.
    """
    if not result or not result.get('has_indirect_addressing'):
        return ""

    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("INDIRECT ADDRESSING PATTERN DETECTED")
    lines.append("=" * 60)
    lines.append("")

    if result['is_gather_reduction']:
        lines.append("**Pattern: GATHER REDUCTION** (fully parallelizable)")
        lines.append("")
        for pattern in result['patterns']:
            if pattern['type'] == 'gather_reduction':
                lines.append(f"Expression: `{pattern['expression']}`")
                lines.append(f"- Accumulator: `{pattern['accumulator']}`")
                lines.append(f"- Index array: `{pattern['index_array']}` (contains indices into `{pattern['data_array']}`)")
                lines.append(f"- Data array: `{pattern['data_array']}` (accessed via indirect indexing)")
                lines.append(f"- Other array: `{pattern['other_array']}` (accessed directly)")
                lines.append("")

        lines.append("**CRITICAL: This is NOT sequential!**")
        lines.append("The gather and reduction are fully parallelizable:")
        lines.append("")
        lines.append("1. Use `tl.load(index_ptr + offsets)` to load indices in parallel")
        lines.append("2. Use `tl.load(data_ptr + indices)` for parallel GATHER (indirect load)")
        lines.append("3. Use `tl.sum()` for parallel reduction")
        lines.append("")
        lines.append("**Correct implementation pattern:**")
        lines.append("```python")
        lines.append("@triton.jit")
        lines.append("def kernel(a_ptr, b_ptr, ip_ptr, partial_sums_ptr, n_elements, BLOCK_SIZE: tl.constexpr):")
        lines.append("    pid = tl.program_id(0)")
        lines.append("    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)")
        lines.append("    mask = offsets < n_elements")
        lines.append("    ")
        lines.append("    # Load direct array and indices")
        lines.append("    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)")
        lines.append("    indices = tl.load(ip_ptr + offsets, mask=mask, other=0)")
        lines.append("    ")
        lines.append("    # GATHER: Load from scattered locations using indices")
        lines.append("    b_vals = tl.load(b_ptr + indices, mask=mask, other=0.0)")
        lines.append("    ")
        lines.append("    # Parallel multiply and reduce")
        lines.append("    block_sum = tl.sum(a_vals * b_vals, axis=0)")
        lines.append("    tl.store(partial_sums_ptr + pid, block_sum)")
        lines.append("")
        lines.append("def wrapper(a, b, ip):")
        lines.append("    n = a.shape[0]")
        lines.append("    BLOCK_SIZE = 1024")
        lines.append("    num_blocks = triton.cdiv(n, BLOCK_SIZE)")
        lines.append("    partial_sums = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)")
        lines.append("    kernel[(num_blocks,)](a, b, ip, partial_sums, n, BLOCK_SIZE)")
        lines.append("    return partial_sums.sum().item()  # Final reduction")
        lines.append("```")
        lines.append("")
        lines.append("**DO NOT use scalar loops inside the kernel!**")
        lines.append("Scalar `for` loops defeat GPU parallelization and are 50-100x slower.")
        lines.append("")

    elif result['is_gather']:
        lines.append("**Pattern: GATHER** (parallel indirect load)")
        lines.append("")
        for pattern in result['patterns']:
            if pattern['type'] == 'gather':
                lines.append(f"Expression: `{pattern['expression']}`")
                lines.append(f"- Index array: `{pattern['index_array']}`")
                lines.append(f"- Data array: `{pattern['data_array']}`")
                lines.append("")

        lines.append("**Implementation:**")
        lines.append("```python")
        lines.append("# Load indices")
        lines.append("indices = tl.load(index_ptr + offsets, mask=mask)")
        lines.append("# Gather from data array using indices")
        lines.append("gathered = tl.load(data_ptr + indices, mask=mask)")
        lines.append("```")
        lines.append("")

    if result['is_scatter']:
        lines.append("**Pattern: SCATTER** (parallel indirect store)")
        lines.append("")
        for pattern in result['patterns']:
            if pattern['type'] == 'scatter':
                lines.append(f"Expression: `{pattern['expression']}`")
                lines.append(f"- Index array: `{pattern['index_array']}`")
                lines.append(f"- Target array: `{pattern['data_array']}`")
                lines.append("")

        lines.append("**Implementation:**")
        lines.append("```python")
        lines.append("# Load indices")
        lines.append("indices = tl.load(index_ptr + offsets, mask=mask)")
        lines.append("# Scatter to data array using indices")
        lines.append("tl.store(data_ptr + indices, values, mask=mask)")
        lines.append("```")
        lines.append("")
        lines.append("**Warning:** Scatter may have race conditions if indices are not unique.")
        lines.append("Use `tl.atomic_add()` if duplicates are possible.")
        lines.append("")

    return "\n".join(lines)


def main():
    """Test the indirect addressing analysis."""
    test_kernels = ['s4115', 's4116', 's4117', 's4121']

    for kernel in test_kernels:
        print(f"\n{'=' * 60}")
        print(f"Analyzing: {kernel}")
        print('=' * 60)

        result = analyze_indirect_addressing(kernel)
        if result:
            print(f"  Has indirect addressing: {result['has_indirect_addressing']}")
            print(f"  Is gather: {result['is_gather']}")
            print(f"  Is scatter: {result['is_scatter']}")
            print(f"  Is gather reduction: {result['is_gather_reduction']}")
            print(f"  Index arrays: {result['index_arrays']}")
            print(f"  Data arrays: {result['data_arrays']}")
            for p in result['patterns']:
                print(f"  Pattern: {p['type']} - {p['expression']}")

            # Print the generated instructions
            instructions = build_indirect_addressing_instructions(result)
            if instructions:
                print("\nGenerated Instructions:")
                print(instructions)
        else:
            print("  Analysis failed (kernel file not found)")


if __name__ == "__main__":
    main()
