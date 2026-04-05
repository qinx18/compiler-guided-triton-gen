#!/usr/bin/env python3
"""
Detect stream compaction (conditional copy/pack) patterns in loops.

This analysis identifies patterns where elements are conditionally copied
to a packed output array, with a counter tracking the output position.

Example s341:
    j = -1;
    for (int i = 0; i < LEN_1D; i++) {
        if (b[i] > 0.) {
            j++;
            a[j] = b[i];
        }
    }

Key characteristics:
1. A counter variable (j, k) initialized before the loop
2. Conditional increment of the counter inside the loop
3. Write to output array using the counter as index

This pattern CANNOT be naively parallelized because:
- The output index depends on how many previous elements satisfied the condition
- Requires prefix sum (scan) to compute output indices for parallel execution

IMPORTANT: The output array should NOT be cleared/zeroed - only positions 0..counter
are written, and the rest should remain unchanged.
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
                'ref': node.get('reference', ''),
                'arguments': node.get('arguments', [])  # Include nested arguments
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


def check_conditional_statement(stmt):
    """Check if a statement has a conditional (if statement)."""
    body = stmt.get('body', {})

    def has_conditional(node):
        if not isinstance(node, dict):
            return False
        if node.get('type') == 'if':
            return True
        for key in ['body', 'then_body', 'else_body', 'arguments']:
            if key in node:
                if isinstance(node[key], list):
                    for item in node[key]:
                        if has_conditional(item):
                            return True
                elif has_conditional(node[key]):
                    return True
        return False

    return has_conditional(body)


def parse_write_index(write_access):
    """
    Parse a write access to check if it uses a non-loop variable as index.

    Stream compaction pattern: a[j] where j is not the loop variable.
    Returns the index variable name if it's a simple variable, None otherwise.
    """
    index_str = write_access.get('index', '')

    # Pattern: { S_X[i] -> array[(var)] } where var is not i
    # Look for the array index part
    match = re.search(r'->\s*\w+\[\((\w+)\)\]', index_str)
    if match:
        return match.group(1)

    # Also check for just variable name without parentheses
    match = re.search(r'->\s*\w+\[(\w+)\]', index_str)
    if match:
        return match.group(1)

    return None


def check_uses_scalar_index(access):
    """
    Check if an access uses a scalar variable as an index argument.

    PET represents a[j] where j is a scalar as:
    - index: '{ [S_3[i] -> [i1]] -> a[((i1) : i1 >= 0)] }'
    - arguments: [{index: '{ S_3[i] -> j[] }', ...}]

    The scalar j[] appears in the arguments.
    """
    arguments = access.get('arguments', [])

    for arg in arguments:
        if isinstance(arg, dict):
            index_str = arg.get('index', '')
            # Look for scalar access pattern: var[] (empty brackets)
            match = re.search(r'->\s*(\w+)\[\]', index_str)
            if match:
                return match.group(1)

    return None


def get_loop_variables(domain_str):
    """Extract ALL loop variables from a domain string.

    For 2D loops like S_0[j, i], returns ['j', 'i'].
    For 1D loops like S_0[i], returns ['i'].
    """
    # Match S_N[var1, var2, ...] pattern
    match = re.search(r'S_\d+\[([\w,\s]+)\]', domain_str)
    if match:
        vars_str = match.group(1)
        # Split by comma and strip whitespace
        return [v.strip() for v in vars_str.split(',')]
    return []


def analyze_stream_compaction(statements_data):
    """
    Analyze statements for stream compaction patterns.

    Detection criteria:
    1. Statement has a conditional (if)
    2. Write to array uses a variable that is NOT the loop variable
    3. This indicates a counter-based output index

    Returns dict with:
    - 'applicable': bool - whether stream compaction pattern detected
    - 'details': list - detected patterns
    - 'advice': str - implementation advice
    """
    result = {
        'applicable': False,
        'details': [],
        'advice': None,
        'output_arrays': []
    }

    for stmt_idx, stmt in enumerate(statements_data):
        domain = stmt.get('domain', '')
        loop_vars = get_loop_variables(domain)

        if not loop_vars:
            continue

        # Check if statement has conditional
        has_cond = check_conditional_statement(stmt)

        reads, writes = extract_accesses(stmt)

        # Check writes for counter-based indexing (pack pattern: a[j] = b[i])
        for write in writes:
            # First check the simple case
            write_idx_var = parse_write_index(write)

            # Also check for scalar index in arguments (PET format for a[j] where j is scalar)
            scalar_idx_var = check_uses_scalar_index(write)

            # Check if write index is NOT any of the loop variables
            # For 2D loops like for(j) for(i) { a[i] = ... }, 'i' IS a loop variable, not a counter
            counter_var = None
            if write_idx_var and write_idx_var not in loop_vars:
                counter_var = write_idx_var
            elif scalar_idx_var and scalar_idx_var not in loop_vars:
                counter_var = scalar_idx_var

            # If write index uses a variable different from ALL loop variables,
            # this might be stream compaction
            if counter_var:
                # Extract array name from the index string
                index_str = write.get('index', '')
                # Pattern: -> array[...] at the end
                array_match = re.search(r'->\s*(\w+)\[', index_str)
                if not array_match:
                    # Try alternative pattern for complex index
                    array_match = re.search(r'\]\s*->\s*(\w+)\[', index_str)
                array_name = array_match.group(1) if array_match else 'unknown'

                # Skip internal PET arrays
                if array_name.startswith('__pet'):
                    continue

                detail = {
                    'statement': stmt_idx,
                    'loop_vars': loop_vars,
                    'counter_var': counter_var,
                    'output_array': array_name,
                    'has_conditional': has_cond,
                    'write_pattern': write.get('index', ''),
                    'direction': 'pack'  # a[counter] = source[i]
                }
                result['details'].append(detail)

                if array_name not in result['output_arrays']:
                    result['output_arrays'].append(array_name)

        # Check reads for counter-based indexing (unpack pattern: a[i] = b[j])
        for read in reads:
            scalar_idx_var = check_uses_scalar_index(read)
            if not scalar_idx_var or scalar_idx_var in loop_vars:
                continue

            # Extract array name from the read index
            index_str = read.get('index', '')
            array_match = re.search(r'->\s*(\w+)\[', index_str)
            if not array_match:
                array_match = re.search(r'\]\s*->\s*(\w+)\[', index_str)
            array_name = array_match.group(1) if array_match else 'unknown'

            # Skip internal PET arrays
            if array_name.startswith('__pet'):
                continue

            # Find which array is being written to (the output)
            for write in writes:
                w_index = write.get('index', '')
                w_match = re.search(r'->\s*(\w+)\[', w_index)
                if w_match and not w_match.group(1).startswith('__pet'):
                    write_array = w_match.group(1)
                    break
            else:
                write_array = 'unknown'

            detail = {
                'statement': stmt_idx,
                'loop_vars': loop_vars,
                'counter_var': scalar_idx_var,
                'output_array': write_array,
                'source_array': array_name,
                'has_conditional': has_cond,
                'write_pattern': f'{write_array}[i] = {array_name}[{scalar_idx_var}]',
                'direction': 'unpack'  # a[i] = source[counter]
            }
            result['details'].append(detail)

            if write_array not in result['output_arrays']:
                result['output_arrays'].append(write_array)

    if result['details']:
        result['applicable'] = True
        result['advice'] = generate_stream_compaction_advice(result['details'], result['output_arrays'])

    return result


def generate_stream_compaction_advice(details, output_arrays):
    """Generate advice for implementing stream compaction correctly."""

    # Determine if we have pack, unpack, or both patterns
    has_pack = any(d.get('direction') == 'pack' for d in details)
    has_unpack = any(d.get('direction') == 'unpack' for d in details)

    if has_unpack and not has_pack:
        return generate_unpack_advice(details, output_arrays)
    else:
        return generate_pack_advice(details, output_arrays)


def generate_pack_advice(details, output_arrays):
    """Generate advice for pack/compaction pattern: a[counter] = source[i]."""

    lines = [
        "STREAM COMPACTION (CONDITIONAL COPY/PACK) PATTERN DETECTED",
        "",
        "This kernel filters elements based on a condition and packs them",
        "into the beginning of an output array using a counter variable.",
        "",
        "Pattern detected:"
    ]

    for d in details:
        lines.append(f"  - S{d['statement']}: writes to {d['output_array']}[{d['counter_var']}]")
        loop_vars_str = ', '.join(d['loop_vars']) if d['loop_vars'] else 'unknown'
        lines.append(f"    Loop variables: {loop_vars_str}, Counter variable: {d['counter_var']}")
        if d['has_conditional']:
            lines.append(f"    Has conditional: write only occurs when condition is true")

    lines.extend([
        "",
        "CRITICAL IMPLEMENTATION REQUIREMENTS:",
        "",
        "1. DO NOT clear/zero the output array!",
        "   The original code only writes to positions 0..counter.",
        "   Elements after counter should remain UNCHANGED.",
        "",
        "2. DO NOT use a serial single-thread loop — it will be extremely slow on GPU.",
        "   Use PyTorch's boolean indexing to implement this in the Python wrapper.",
        "",
        "3. IMPLEMENTATION (use this approach in the Python wrapper, with a stub @triton.jit kernel):",
        "   ```python",
    ])

    # Generate example code based on detected arrays
    if output_arrays:
        out_arr = output_arrays[0]
        lines.extend([
            f"   # For stream compaction: {out_arr}[counter] = source[i] when condition",
            f"   mask = source > 0.0  # or whatever the condition is",
            f"   packed_values = source[mask]",
            f"   num_packed = packed_values.numel()",
            f"   {out_arr}[:num_packed] = packed_values",
            f"   # DO NOT touch {out_arr}[num_packed:] - leave it unchanged!",
        ])

    lines.extend([
        "   ```",
        "",
        "WARNING: Do NOT use .zero_() or .fill_() on the output array!",
    ])

    return '\n'.join(lines)


def generate_unpack_advice(details, output_arrays):
    """Generate advice for unpack/scatter pattern: a[i] = source[counter]."""

    lines = [
        "CONDITIONAL SCATTER (UNPACK) PATTERN DETECTED",
        "",
        "This kernel reads from a packed source array using a counter variable,",
        "and writes to positions in the output array where a condition is true.",
        "",
        "Pattern detected:"
    ]

    for d in details:
        source = d.get('source_array', '?')
        lines.append(f"  - S{d['statement']}: {d['output_array']}[i] = {source}[{d['counter_var']}] when condition is true")
        loop_vars_str = ', '.join(d['loop_vars']) if d['loop_vars'] else 'unknown'
        lines.append(f"    Loop variables: {loop_vars_str}, Counter variable: {d['counter_var']}")

    lines.extend([
        "",
        "CRITICAL IMPLEMENTATION REQUIREMENTS:",
        "",
        "1. Only elements where the condition is true should be modified.",
        "   Other elements must remain UNCHANGED.",
        "",
        "2. DO NOT use a serial single-thread loop — it will be extremely slow on GPU.",
        "   Use PyTorch's parallel prefix sum to compute the counter indices.",
        "",
        "3. IMPLEMENTATION (use this approach in the Python wrapper, with a stub @triton.jit kernel):",
        "   ```python",
    ])

    if output_arrays and details:
        out_arr = output_arrays[0]
        source = details[0].get('source_array', 'source')
        lines.extend([
            f"   mask = ({out_arr} > 0.0).to(torch.int32)  # or whatever the condition is",
            f"   indices = torch.cumsum(mask, dim=0) - 1  # prefix sum gives counter value at each position",
            f"   bool_mask = mask.bool()",
            f"   {out_arr}[bool_mask] = {source}[indices[bool_mask]]  # gather from source at counter positions",
            f"   # Only positions where condition is true are modified",
        ])

    lines.extend([
        "   ```",
        "",
        "WARNING: Do NOT modify elements where the condition is false!",
    ])

    return '\n'.join(lines)


def format_stream_compaction_for_prompt(compaction_result):
    """Format stream compaction analysis for inclusion in LLM prompt."""
    if not compaction_result['applicable']:
        return None

    lines = []
    lines.append("=" * 60)
    lines.append("STREAM COMPACTION PATTERN DETECTED")
    lines.append("=" * 60)
    lines.append("")

    if compaction_result['advice']:
        lines.append(compaction_result['advice'])

    lines.append("")
    lines.append("=" * 60)

    return '\n'.join(lines)


def analyze_kernel_stream_compaction(kernel_file):
    """Analyze a kernel file for stream compaction patterns."""
    pet_output = run_pet(kernel_file)
    if not pet_output:
        return None

    try:
        data = yaml.safe_load(pet_output)
    except:
        return None

    statements = data.get('statements', [])
    return analyze_stream_compaction(statements)


def main():
    """Test stream compaction detection on s341 and s343."""
    test_kernels = ['s341', 's343']

    print("=" * 80)
    print("STREAM COMPACTION PATTERN DETECTION")
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

        result = analyze_kernel_stream_compaction(kernel_file)
        if result:
            prompt_text = format_stream_compaction_for_prompt(result)
            if prompt_text:
                print(f"\n{prompt_text}")
            else:
                print("\nNo stream compaction pattern detected")
        else:
            print("\nFailed to analyze kernel")


if __name__ == "__main__":
    main()
