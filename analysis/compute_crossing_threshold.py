#!/usr/bin/env python3
"""
Detect crossing threshold patterns in loops.

This analysis identifies patterns where an array element is read and the read
location gets modified during the loop, creating a "crossing threshold" where
the loop must be split into two phases:
- Phase 1: Reading original values
- Phase 2: Reading updated values

Example s1113:
    for (int i = 0; i < LEN_1D; i++) {
        a[i] = a[LEN_1D/2] + b[i];
    }
    - Reads a[LEN_1D/2] (constant)
    - Writes a[i]
    - At i = LEN_1D/2, the read location is written
    - Before: read original a[LEN_1D/2]
    - After: read updated a[LEN_1D/2]

Example s281:
    for (int i = 0; i < LEN_1D; i++) {
        x = a[LEN_1D-i-1] + b[i] * c[i];
        a[i] = x - 1.0;
    }
    - Reads a[LEN_1D-i-1] (reverse)
    - Writes a[i] (forward)
    - When i > (LEN_1D-1)/2, read index < write indices from earlier iterations
    - Before threshold: read original values
    - After threshold: read updated values
"""

import subprocess
import yaml
import re
import os
from typing import Optional, Dict, List, Tuple

PET_PATH = "/home/qinxiao/workspace/pet/pet"
KERNELS_DIR = "/home/qinxiao/workspace/compiler-guided-triton-gen/analysis/kernels"


def run_pet(kernel_file):
    """Run PET on a kernel file and return the YAML output."""
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/home/qinxiao/workspace/pet/isl/.libs:' + env.get('LD_LIBRARY_PATH', '')

    try:
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
    except:
        return None


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


def parse_index_expression(index_str: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse an ISL index string to extract array name and index expression.
    Example: '{ S_0[i] -> a[(31999 - i)] }' returns ('a', '31999 - i')
    Example: '{ S_0[i] -> a[(i)] }' returns ('a', 'i')
    Example: '{ S_0[i] -> a[(16000)] }' returns ('a', '16000')
    """
    # Match pattern: array_name[(index_expr)]
    match = re.search(r'(\w+)\[\(([^)]+)\)\]', index_str)
    if match:
        return match.group(1), match.group(2)
    # Try without inner parentheses
    match = re.search(r'(\w+)\[([^\]]+)\](?:\s*\})?$', index_str)
    if match:
        return match.group(1), match.group(2)
    return None, None


def get_loop_var_and_bound(domain_str: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """
    Extract loop variable, lower bound, and upper bound from domain string.
    Example: '{ S_0[i] : 0 <= i <= 31999 }' returns ('i', 0, 31999)
    Example: '{ S_0[i] : 1 <= i <= 31999 }' returns ('i', 1, 31999)
    Example: '{ S_0[i] : 0 < i <= 31999 }' returns ('i', 1, 31999) - strict inequality
    """
    # Extract loop variable
    var_match = re.search(r'S_\d+\[(\w+)\]', domain_str)
    if not var_match:
        return None, None, None
    loop_var = var_match.group(1)

    # Extract lower bound
    # Pattern 1: "N <= var" (non-strict, lower bound = N)
    # Pattern 2: "N < var" (strict, lower bound = N+1)
    lower_bound = 0  # Default to 0

    # Try non-strict first: "N <= var"
    lower_match = re.search(rf'(\d+)\s*<=\s*{loop_var}', domain_str)
    if lower_match:
        lower_bound = int(lower_match.group(1))
    else:
        # Try strict: "N < var" (means var > N, so lower bound is N+1)
        lower_match_strict = re.search(rf'(\d+)\s*<\s*{loop_var}', domain_str)
        if lower_match_strict:
            lower_bound = int(lower_match_strict.group(1)) + 1

    # Extract upper bound (pattern: "var <= N")
    upper_bound = None
    upper_match = re.search(rf'{loop_var}\s*<=\s*(\d+)', domain_str)
    if upper_match:
        upper_bound = int(upper_match.group(1))

    return loop_var, lower_bound, upper_bound


def analyze_index_pattern(read_idx: str, write_idx: str, loop_var: str, lower_bound: int, upper_bound: int) -> Optional[Dict]:
    """
    Analyze read and write index patterns to detect crossing threshold.

    Returns dict with:
    - 'pattern_type': 'constant_read', 'reverse_access', 'offset_access', etc.
    - 'threshold': the iteration where crossing occurs
    - 'threshold_expr': expression for the threshold
    """
    read_idx = read_idx.strip()
    write_idx = write_idx.strip()

    # Check if write is simple loop variable
    if write_idx != loop_var:
        return None  # Only handle simple write patterns for now

    # Pattern 1: Constant read index (like s1113)
    # a[i] = a[CONST] + ...
    if loop_var not in read_idx:
        try:
            const_val = int(read_idx)
            # CRITICAL: Check if threshold is within loop bounds
            # If threshold < lower_bound, the read location is NEVER written in this loop
            if const_val < lower_bound:
                return None  # No crossing - read location is never written
            return {
                'pattern_type': 'constant_read',
                'threshold': const_val,
                'threshold_expr': str(const_val),
                'description': f'Read index is constant {const_val}, written at i={const_val}'
            }
        except ValueError:
            # Try to evaluate expression like "32000 / 2" or "LEN_1D/2"
            # For now, check if it's a division by 2
            if '/' in read_idx or '//' in read_idx:
                parts = re.split(r'[/]+', read_idx)
                if len(parts) == 2:
                    try:
                        num = int(parts[0].strip())
                        denom = int(parts[1].strip())
                        const_val = num // denom
                        # CRITICAL: Check if threshold is within loop bounds
                        if const_val < lower_bound:
                            return None  # No crossing - read location is never written
                        return {
                            'pattern_type': 'constant_read',
                            'threshold': const_val,
                            'threshold_expr': f'{num}//{denom}',
                            'description': f'Read index is constant {const_val}, written at i={const_val}'
                        }
                    except ValueError:
                        pass
            return None

    # Pattern 2: Reverse access (like s281)
    # a[i] = a[N - 1 - i] + ... or a[i] = a[N - i - 1] + ...
    # Read: N - 1 - i, Write: i
    # Crossing when: N - 1 - i < i (read from already-written location)
    # i.e., N - 1 < 2i, i.e., i > (N-1)/2

    # Parse reverse pattern: "CONST - i" or "CONST - 1 - i"
    reverse_match = re.match(rf'^(\d+)\s*-\s*{loop_var}$', read_idx)
    if reverse_match:
        n_minus_1 = int(reverse_match.group(1))
        threshold = (n_minus_1 + 1) // 2  # When i >= threshold, we read updated values
        return {
            'pattern_type': 'reverse_access',
            'threshold': threshold,
            'threshold_expr': f'({n_minus_1} + 1) // 2',
            'description': f'Reverse access: read a[{n_minus_1}-i], write a[i]. When i >= {threshold}, reads updated values.'
        }

    # Pattern: "CONST - 1 - i" which simplifies to "(CONST-1) - i"
    reverse_match2 = re.match(rf'^(\d+)\s*-\s*1\s*-\s*{loop_var}$', read_idx)
    if reverse_match2:
        const = int(reverse_match2.group(1))
        n_minus_1 = const - 1
        threshold = (n_minus_1 + 1) // 2
        return {
            'pattern_type': 'reverse_access',
            'threshold': threshold,
            'threshold_expr': f'({const} - 1 + 1) // 2 = {const} // 2',
            'description': f'Reverse access: read a[{const}-1-i], write a[i]. When i >= {threshold}, reads updated values.'
        }

    # Pattern: "- i + CONST" (equivalent to CONST - i)
    reverse_match3 = re.match(rf'^-\s*{loop_var}\s*\+\s*(\d+)$', read_idx)
    if reverse_match3:
        n_minus_1 = int(reverse_match3.group(1))
        threshold = (n_minus_1 + 1) // 2
        return {
            'pattern_type': 'reverse_access',
            'threshold': threshold,
            'threshold_expr': f'({n_minus_1} + 1) // 2',
            'description': f'Reverse access: read a[{n_minus_1}-i], write a[i]. When i >= {threshold}, reads updated values.'
        }

    return None


def analyze_crossing_threshold(statements_data: List[Dict]) -> Dict:
    """
    Analyze statements for crossing threshold patterns.

    Returns dict with:
    - 'applicable': bool
    - 'patterns': list of detected patterns
    - 'advice': str
    """
    result = {
        'applicable': False,
        'patterns': [],
        'advice': None
    }

    # Collect all reads and writes across all statements
    all_reads = []
    all_writes = []
    loop_var = None
    lower_bound = 0
    upper_bound = None

    for stmt in statements_data:
        domain = stmt.get('domain', '')
        lv, lb, ub = get_loop_var_and_bound(domain)
        if lv:
            loop_var = lv
        if lb is not None:
            lower_bound = lb
        if ub is not None:
            upper_bound = ub

        reads, writes = extract_accesses(stmt)
        for r in reads:
            arr, idx = parse_index_expression(r['index'])
            if arr and idx:
                all_reads.append({'array': arr, 'index': idx, 'raw': r['index']})
        for w in writes:
            arr, idx = parse_index_expression(w['index'])
            if arr and idx:
                all_writes.append({'array': arr, 'index': idx, 'raw': w['index']})

    if not loop_var:
        return result

    # Find arrays that are both read and written
    read_arrays = {r['array'] for r in all_reads}
    write_arrays = {w['array'] for w in all_writes}
    shared_arrays = read_arrays & write_arrays

    for arr in shared_arrays:
        arr_reads = [r for r in all_reads if r['array'] == arr]
        arr_writes = [w for w in all_writes if w['array'] == arr]

        for r in arr_reads:
            for w in arr_writes:
                # Skip if read and write have same index
                if r['index'] == w['index']:
                    continue

                pattern = analyze_index_pattern(r['index'], w['index'], loop_var, lower_bound, upper_bound)
                if pattern:
                    pattern['array'] = arr
                    pattern['read_index'] = r['index']
                    pattern['write_index'] = w['index']
                    pattern['loop_var'] = loop_var
                    pattern['lower_bound'] = lower_bound
                    pattern['upper_bound'] = upper_bound
                    result['patterns'].append(pattern)

    if result['patterns']:
        result['applicable'] = True
        result['advice'] = generate_crossing_threshold_advice(result['patterns'])

    return result


def generate_crossing_threshold_advice(patterns: List[Dict]) -> str:
    """Generate advice for implementing crossing threshold patterns."""
    lines = [
        "CROSSING THRESHOLD PATTERN DETECTED",
        "",
        "This loop reads from and writes to the same array with different indices.",
        "At some point in the loop, the read index matches a previously-written index,",
        "creating a 'crossing threshold' where the loop behavior changes.",
        "",
        "Pattern(s) detected:"
    ]

    for p in patterns:
        lines.append(f"")
        lines.append(f"  Array: {p['array']}")
        lines.append(f"  Read index: {p['array']}[{p['read_index']}]")
        lines.append(f"  Write index: {p['array']}[{p['write_index']}]")
        lines.append(f"  Pattern type: {p['pattern_type']}")
        lines.append(f"  Threshold: i = {p['threshold']}")
        lines.append(f"  {p['description']}")

    lines.extend([
        "",
        "CRITICAL IMPLEMENTATION REQUIREMENTS:",
        "",
        "The loop MUST be split into two phases:",
        ""
    ])

    for p in patterns:
        threshold = p['threshold']
        arr = p['array']

        if p['pattern_type'] == 'constant_read':
            lines.extend([
                f"Phase 1 (i = 0 to {threshold}, inclusive):",
                f"  - Uses ORIGINAL {arr}[{p['read_index']}]",
                f"  - Can be parallelized",
                f"  - Clone {arr}[{p['read_index']}] before this phase",
                "",
                f"Phase 2 (i = {threshold + 1} to end):",
                f"  - Uses UPDATED {arr}[{p['read_index']}] (from Phase 1)",
                f"  - Can be parallelized separately",
                f"  - Read {arr}[{p['read_index']}] after Phase 1 completes",
            ])
        elif p['pattern_type'] == 'reverse_access':
            lines.extend([
                f"Phase 1 (i = 0 to {threshold - 1}):",
                f"  - Reads from indices > {threshold} (original values)",
                f"  - Writes to indices < {threshold}",
                f"  - Can be parallelized with cloned array for reads",
                "",
                f"Phase 2 (i = {threshold} to end):",
                f"  - Reads from indices < {threshold} (UPDATED in Phase 1)",
                f"  - MUST use values written in Phase 1",
                f"  - Can be parallelized AFTER Phase 1 completes",
            ])

    lines.extend([
        "",
        "IMPLEMENTATION TEMPLATE:",
        "```python",
        "def kernel_triton(a, b, ...):",
        "    n = a.shape[0]",
    ])

    for p in patterns:
        threshold = p['threshold']
        arr = p['array']

        if p['pattern_type'] == 'constant_read':
            lines.extend([
                f"    threshold = {p['threshold_expr']}",
                f"",
                f"    # Save original value before it gets modified",
                f"    orig_{arr}_at_threshold = {arr}[threshold].clone()",
                f"",
                f"    # Phase 1: i = 0 to threshold (uses original value)",
                f"    {arr}[:threshold+1] = orig_{arr}_at_threshold + ...  # parallel",
                f"",
                f"    # Phase 2: i = threshold+1 to end (uses updated value)",
                f"    {arr}[threshold+1:] = {arr}[threshold] + ...  # parallel",
            ])
        elif p['pattern_type'] == 'reverse_access':
            lines.extend([
                f"    threshold = {p['threshold_expr']}",
                f"    n = {arr}.shape[0]",
                f"",
                f"    # Clone array for Phase 1 reads",
                f"    {arr}_copy = {arr}.clone()",
                f"",
                f"    # Phase 1: i = 0 to threshold-1",
                f"    # - Reads from {arr}_copy[n-1-i] (indices >= threshold, original values)",
                f"    # - Writes to {arr}[i] (indices < threshold)",
                f"    for i in range(threshold):  # or parallel kernel",
                f"        {arr}[i] = {arr}_copy[n-1-i] + ...",
                f"",
                f"    # Phase 2: i = threshold to n-1",
                f"    # - Reads from {arr}[n-1-i] (indices < threshold, UPDATED in Phase 1!)",
                f"    # - Writes to {arr}[i] (indices >= threshold)",
                f"    # CRITICAL: Must read from FULL array, NOT sliced!",
                f"    for i in range(threshold, n):  # or parallel kernel",
                f"        {arr}[i] = {arr}[n-1-i] + ...  # reads updated values from phase 1",
                f"",
                f"CRITICAL BUG TO AVOID:",
                f"  DO NOT pass {arr}[threshold:] to Phase 2 kernel!",
                f"  Phase 2 needs to read {arr}[0..threshold-1] which were UPDATED in Phase 1.",
                f"  Pass the FULL array pointer and compute indices correctly.",
            ])

    lines.extend([
        "```",
        "",
        "WARNING: Simple parallel implementation will give WRONG results!",
        "The two phases MUST be executed separately with synchronization between them.",
    ])

    return '\n'.join(lines)


def format_crossing_threshold_for_prompt(analysis_result: Dict) -> Optional[str]:
    """Format crossing threshold analysis for inclusion in LLM prompt."""
    if not analysis_result or not analysis_result.get('applicable'):
        return None

    lines = []
    lines.append("=" * 60)
    lines.append("CROSSING THRESHOLD DEPENDENCY DETECTED")
    lines.append("=" * 60)
    lines.append("")

    if analysis_result['advice']:
        lines.append(analysis_result['advice'])

    lines.append("")
    lines.append("=" * 60)

    return '\n'.join(lines)


def analyze_kernel_crossing_threshold(kernel_name: str) -> Optional[Dict]:
    """
    Analyze a kernel for crossing threshold patterns.

    Args:
        kernel_name: Name of the kernel (e.g., 's1113', 's281')

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
    if not statements:
        return None

    result = analyze_crossing_threshold(statements)

    if result['applicable']:
        return result
    return None


def main():
    """Test crossing threshold detection."""
    test_kernels = ['s1113', 's281', 's111', 's112']

    print("=" * 80)
    print("CROSSING THRESHOLD PATTERN DETECTION")
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

        result = analyze_kernel_crossing_threshold(kernel)
        if result:
            print(f"\nDetected: {result['applicable']}")
            for p in result['patterns']:
                print(f"  Pattern: {p['pattern_type']}")
                print(f"  Array: {p['array']}")
                print(f"  Threshold: {p['threshold']}")
            print()
            formatted = format_crossing_threshold_for_prompt(result)
            if formatted:
                print(formatted)
        else:
            print("\nNo crossing threshold pattern detected")


if __name__ == "__main__":
    main()
