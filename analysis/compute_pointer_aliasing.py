#!/usr/bin/env python3
"""
Detect pointer aliasing patterns in loops (s421-s424 style).

This analysis identifies patterns where a pointer (like xx) is an alias
to another array with an offset (xx = flat_2d_array + vl).

Example s424:
    int vl = 63;
    xx = flat_2d_array + vl;
    for (int i = 0; i < LEN_1D - 1; i++) {
        xx[i+1] = flat_2d_array[i] + a[i];
    }

    Actual operation: flat_2d_array[64+i] = flat_2d_array[i] + a[i]

    This has a loop-carried RAW dependency because:
    - Write offset: 64 (vl + 1)
    - Read offset: 0
    - At iteration i=64, we read position 64 which was written at i=0

    However, it CAN be vectorized in strips of 64 elements because
    within each strip, there's no dependency.

Key patterns:
1. Write ahead, read behind (s424): RAW dependency, strip-vectorizable
2. Write behind, read ahead (s423): No dependency, fully parallelizable
3. Different arrays (s422): No dependency, fully parallelizable
4. Explicit copy before loop (s421): No dependency with copy

This script analyzes the C code to detect these patterns.
"""

import re
import os
import sys

# Add the parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KERNELS_DIR = "/home/qinxiao/workspace/compiler-guided-triton-gen/analysis/kernels"


def extract_scop_code(c_code):
    """Extract the code between #pragma scop and #pragma endscop."""
    match = re.search(r'#pragma scop\s*(.*?)\s*#pragma endscop', c_code, re.DOTALL)
    if match:
        return match.group(1).strip()
    return c_code


def extract_pointer_assignments(c_code):
    """
    Extract pointer aliasing assignments like:
    xx = flat_2d_array + vl;
    yy = xx;

    Returns dict: {pointer_name: (base_array, offset_expr)}
    """
    aliases = {}

    # Pattern: ptr = array + offset or ptr = array
    # Examples: xx = flat_2d_array + vl; or yy = xx;
    pattern = r'(\w+)\s*=\s*(\w+)(?:\s*\+\s*(\w+|\d+))?\s*;'

    for match in re.finditer(pattern, c_code):
        ptr_name = match.group(1)
        base = match.group(2)
        offset = match.group(3) if match.group(3) else '0'

        # Skip loop variable assignments like i = 0
        if ptr_name in ['i', 'j', 'k', 'nl']:
            continue

        aliases[ptr_name] = (base, offset)

    return aliases


def extract_loop_bounds(c_code):
    """Extract loop variable, bounds, and stride."""
    # Pattern: for (int i = start; i < end; i++ / i+=stride)
    match = re.search(r'for\s*\(\s*(?:int\s+)?(\w+)\s*=\s*(\d+)\s*;\s*\w+\s*<\s*([^;]+)\s*;\s*([^)]+)\)', c_code)
    if match:
        var = match.group(1)
        start = int(match.group(2))
        end_expr = match.group(3).strip()
        increment = match.group(4).strip()

        # Parse stride from increment expression
        stride = 1
        # i++ or ++i
        if increment in (f'{var}++', f'++{var}'):
            stride = 1
        # i += N
        else:
            stride_match = re.match(rf'{var}\s*\+=\s*(\d+)', increment)
            if stride_match:
                stride = int(stride_match.group(1))
            else:
                # i = i + N
                stride_match = re.match(rf'{var}\s*=\s*{var}\s*\+\s*(\d+)', increment)
                if stride_match:
                    stride = int(stride_match.group(1))

        return {
            'var': var,
            'start': start,
            'end_expr': end_expr,
            'stride': stride
        }
    return None


def detect_mutually_exclusive_regions(c_code):
    """
    Detect if code has mutually exclusive regions (if-else with goto pattern).

    Pattern like s161:
        if (condition) { goto L20; }
        statement1;  // branch 1
        goto L10;
        L20:
        statement2;  // branch 2
        L10:

    Returns dict mapping line ranges to branch IDs, or None if no branches detected.
    """
    lines = c_code.split('\n')

    # Look for pattern: if (cond) { goto Lxx; } ... goto Lyy; Lxx: ... Lyy:
    # This indicates mutually exclusive branches

    # Use a more flexible pattern that handles nested parentheses in condition
    # Match: if (...) { goto LABEL; }
    goto_if_pattern = r'if\s*\(.*?\)\s*\{\s*goto\s+(\w+)\s*;\s*\}'
    goto_pattern = r'^\s*goto\s+(\w+)\s*;'
    label_pattern = r'^\s*(\w+)\s*:\s*$'

    # Find the if-goto statement
    if_goto_match = re.search(goto_if_pattern, c_code, re.DOTALL)
    if not if_goto_match:
        return None

    else_label = if_goto_match.group(1)  # Label for else branch (e.g., L20)

    # Find position of if-goto, else label, and end label
    if_goto_pos = if_goto_match.end()

    # Find the "goto Lend;" that skips the else branch
    skip_goto_match = re.search(goto_pattern, c_code[if_goto_pos:], re.MULTILINE)
    if not skip_goto_match:
        return None

    end_label = skip_goto_match.group(1)  # Label at end (e.g., L10)
    skip_goto_end = if_goto_pos + skip_goto_match.end()

    # Find else label position
    else_label_match = re.search(rf'^\s*{else_label}\s*:', c_code[skip_goto_end:], re.MULTILINE)
    if not else_label_match:
        return None

    else_label_pos = skip_goto_end + else_label_match.start()

    # Find end label position
    end_label_match = re.search(rf'^\s*{end_label}\s*:', c_code[else_label_pos:], re.MULTILINE)
    if not end_label_match:
        return None

    end_label_pos = else_label_pos + end_label_match.start()

    return {
        'has_branches': True,
        'if_branch': (if_goto_pos, skip_goto_end - len(skip_goto_match.group(0))),  # Between if-goto and goto-end
        'else_branch': (else_label_pos, end_label_pos),  # Between else label and end label
        'else_label': else_label,
        'end_label': end_label
    }


def get_branch_for_position(pos, branch_info):
    """Return which branch a position belongs to, or None if neither."""
    if branch_info is None:
        return None

    if_start, if_end = branch_info['if_branch']
    else_start, else_end = branch_info['else_branch']

    if if_start <= pos <= if_end:
        return 'if'
    elif else_start <= pos <= else_end:
        return 'else'
    return None


def extract_array_accesses(c_code, loop_var='i'):
    """
    Extract array accesses from loop body.

    Returns list of dicts: {array, index_expr, offset, is_write, branch}
    where branch is 'if', 'else', or None (not in a mutually exclusive region)
    """
    accesses = []

    # Detect mutually exclusive regions
    branch_info = detect_mutually_exclusive_regions(c_code)

    # Find assignment statements
    # Pattern: array[index] = expr  or  expr uses array[index]

    # First, find the assignment target (write)
    write_pattern = r'(\w+)\[([^\]]+)\]\s*='
    for match in re.finditer(write_pattern, c_code):
        array = match.group(1)
        index_expr = match.group(2).strip()
        offset = parse_index_offset(index_expr, loop_var)
        branch = get_branch_for_position(match.start(), branch_info)
        accesses.append({
            'array': array,
            'index_expr': index_expr,
            'offset': offset,
            'is_write': True,
            'branch': branch
        })

    # Find reads (array accesses not at start of assignment)
    # This is a simplified approach - find all array[index] and filter out writes
    read_pattern = r'(\w+)\[([^\]]+)\]'
    write_positions = set()
    for match in re.finditer(write_pattern, c_code):
        write_positions.add(match.start())

    for match in re.finditer(read_pattern, c_code):
        if match.start() in write_positions:
            continue
        array = match.group(1)
        index_expr = match.group(2).strip()
        offset = parse_index_offset(index_expr, loop_var)
        branch = get_branch_for_position(match.start(), branch_info)
        accesses.append({
            'array': array,
            'index_expr': index_expr,
            'offset': offset,
            'is_write': False,
            'branch': branch
        })

    return accesses


def parse_index_offset(index_expr, loop_var='i'):
    """
    Parse index expression to extract offset from loop variable.

    Examples:
    - 'i' -> 0
    - 'i+1' -> 1
    - 'i + 1' -> 1
    - '1 + i' -> 1
    - 'i-1' -> -1
    """
    index_expr = index_expr.strip()

    # Just the variable
    if index_expr == loop_var:
        return 0

    # Try to parse as loop_var + constant or constant + loop_var
    # Remove spaces
    expr = index_expr.replace(' ', '')

    # Pattern: i+N or i-N
    match = re.match(rf'{loop_var}([+\-])(\d+)$', expr)
    if match:
        sign = 1 if match.group(1) == '+' else -1
        return sign * int(match.group(2))

    # Pattern: N+i
    match = re.match(rf'(\d+)\+{loop_var}$', expr)
    if match:
        return int(match.group(1))

    # Pattern: -N+i or N-i (less common)
    match = re.match(rf'(\-?\d+)\+{loop_var}$', expr)
    if match:
        return int(match.group(1))

    return None  # Complex expression


def resolve_aliases(accesses, aliases, local_vars=None):
    """
    Resolve pointer aliases to get actual array accesses.

    For example, if xx = flat_2d_array + 63, then:
    xx[i+1] becomes flat_2d_array[i + 64]

    local_vars: dict mapping variable names to their values (e.g., {'vl': 63})
    """
    if local_vars is None:
        local_vars = {}

    resolved = []

    for acc in accesses:
        array = acc['array']
        offset = acc['offset']
        branch = acc.get('branch')  # Preserve branch info

        if array in aliases:
            base_array, alias_offset = aliases[array]

            # Resolve the alias offset
            if alias_offset.isdigit():
                alias_offset_val = int(alias_offset)
            elif alias_offset in local_vars:
                alias_offset_val = local_vars[alias_offset]
            else:
                alias_offset_val = None

            if alias_offset_val is not None and offset is not None:
                resolved.append({
                    'original_array': array,
                    'resolved_array': base_array,
                    'original_offset': offset,
                    'alias_offset': alias_offset_val,
                    'resolved_offset': offset + alias_offset_val,
                    'is_write': acc['is_write'],
                    'branch': branch
                })
            else:
                resolved.append({
                    'original_array': array,
                    'resolved_array': base_array,
                    'original_offset': offset,
                    'alias_offset': alias_offset,
                    'resolved_offset': None,
                    'is_write': acc['is_write'],
                    'branch': branch
                })
        else:
            resolved.append({
                'original_array': array,
                'resolved_array': array,
                'original_offset': offset,
                'alias_offset': 0,
                'resolved_offset': offset,
                'is_write': acc['is_write'],
                'branch': branch
            })

    return resolved


def analyze_branch_reordering(resolved_accesses):
    """
    Analyze if mutually exclusive branches can be reordered for parallelization.

    Pattern like s161:
    - Branch 1 (if): a[i] = c[i] + d[i] * e[i]  (reads c[i], writes a[i])
    - Branch 2 (else): c[i+1] = a[i] + d[i] * d[i]  (reads a[i], writes c[i+1])

    Key insight:
    - Branch 2 reads a[i] (needs ORIGINAL value before any writes)
    - Branch 1 reads c[i] (needs UPDATED value from branch 2's c[i+1] at i-1)

    So we should process: Branch 2 first, then Branch 1.

    Returns dict with reordering info, or None if not applicable.
    """
    # Separate accesses by branch
    if_accesses = [a for a in resolved_accesses if a.get('branch') == 'if']
    else_accesses = [a for a in resolved_accesses if a.get('branch') == 'else']

    # Need both branches to have accesses
    if not if_accesses or not else_accesses:
        return None

    # Get reads and writes for each branch
    if_reads = {a['resolved_array']: a for a in if_accesses if not a['is_write']}
    if_writes = {a['resolved_array']: a for a in if_accesses if a['is_write']}
    else_reads = {a['resolved_array']: a for a in else_accesses if not a['is_write']}
    else_writes = {a['resolved_array']: a for a in else_accesses if a['is_write']}

    # Check for cross-branch dependencies:
    # - If branch reads X, else branch writes X (or vice versa)
    cross_deps = []

    # Check: else writes array that if reads
    for arr in else_writes:
        if arr in if_reads:
            w = else_writes[arr]
            r = if_reads[arr]
            # else writes arr[i+w_off], if reads arr[i+r_off]
            # If w_off > r_off, then else's write at i-1 affects if's read at i
            if w['resolved_offset'] is not None and r['resolved_offset'] is not None:
                if w['resolved_offset'] > r['resolved_offset']:
                    cross_deps.append({
                        'array': arr,
                        'writer_branch': 'else',
                        'reader_branch': 'if',
                        'write_offset': w['resolved_offset'],
                        'read_offset': r['resolved_offset']
                    })

    # Check: if writes array that else reads
    for arr in if_writes:
        if arr in else_reads:
            w = if_writes[arr]
            r = else_reads[arr]
            # if writes arr[i+w_off], else reads arr[i+r_off]
            # If w_off == r_off, then if's write affects else's read at same i
            # else needs ORIGINAL value, so else should go first
            if w['resolved_offset'] is not None and r['resolved_offset'] is not None:
                if w['resolved_offset'] == r['resolved_offset']:
                    cross_deps.append({
                        'array': arr,
                        'writer_branch': 'if',
                        'reader_branch': 'else',
                        'write_offset': w['resolved_offset'],
                        'read_offset': r['resolved_offset'],
                        'else_needs_original': True
                    })

    if not cross_deps:
        return None

    # Determine correct order
    # If else needs original values, else goes first
    else_first = any(d.get('else_needs_original') for d in cross_deps)
    # If else writes values that if needs (updated), else also goes first
    else_first = else_first or any(d['writer_branch'] == 'else' for d in cross_deps)

    return {
        'applicable': True,
        'cross_deps': cross_deps,
        'order': ['else', 'if'] if else_first else ['if', 'else'],
        'if_accesses': if_accesses,
        'else_accesses': else_accesses
    }


def generate_branch_reorder_advice(branch_reorder):
    """Generate advice for branch reordering pattern."""
    order = branch_reorder['order']
    first_branch = order[0]
    second_branch = order[1]

    lines = [
        "MUTUALLY EXCLUSIVE BRANCH REORDERING PATTERN DETECTED",
        "",
        "This loop has if-else branches that can be parallelized by processing",
        "all iterations of one branch first, then all iterations of the other.",
        "",
        f"CORRECT ORDER: Process '{first_branch}' branch COMPLETELY first, then '{second_branch}' branch",
        "",
        "Cross-branch dependencies:",
    ]

    for dep in branch_reorder['cross_deps']:
        lines.append(f"  - Array '{dep['array']}': {dep['writer_branch']} writes [i+{dep['write_offset']}], "
                    f"{dep['reader_branch']} reads [i+{dep['read_offset']}]")

    lines.extend([
        "",
        "CRITICAL: Use TWO SEPARATE kernel launches or PyTorch operations!",
        "A single kernel with both branches will have race conditions between blocks.",
        "",
        "CORRECT IMPLEMENTATION (using PyTorch for simplicity):",
        "```python",
        "def s161_triton(a, b, c, d, e):",
        "    n = a.shape[0] - 1",
        "    idx = torch.arange(n, device=a.device)",
        "    ",
        f"    # Step 1: Process ALL '{first_branch}' branch iterations FIRST",
    ])

    if first_branch == 'else':
        lines.extend([
            "    else_mask = b[:n] < 0.0",
            "    # c[i+1] = a[i] + d[i] * d[i] for all i where b[i] < 0",
            "    c[1:n+1][else_mask] = (a[:n] + d[:n] * d[:n])[else_mask]",
        ])
    else:
        lines.extend([
            "    if_mask = b[:n] >= 0.0",
            "    # a[i] = c[i] + d[i] * e[i] for all i where b[i] >= 0",
            "    a[:n][if_mask] = (c[:n] + d[:n] * e[:n])[if_mask]",
        ])

    lines.extend([
        "    ",
        f"    # Step 2: Process ALL '{second_branch}' branch iterations AFTER step 1 completes",
    ])

    if second_branch == 'if':
        lines.extend([
            "    if_mask = b[:n] >= 0.0",
            "    # a[i] = c[i] + d[i] * e[i] for all i where b[i] >= 0",
            "    # c[i] may have been updated by step 1 (c[i+1] write at i-1)",
            "    a[:n][if_mask] = (c[:n] + d[:n] * e[:n])[if_mask]",
        ])
    else:
        lines.extend([
            "    else_mask = b[:n] < 0.0",
            "    # c[i+1] = a[i] + d[i] * d[i] for all i where b[i] < 0",
            "    c[1:n+1][else_mask] = (a[:n] + d[:n] * d[:n])[else_mask]",
        ])

    lines.extend([
        "```",
        "",
        "This works because:",
        f"- {first_branch.capitalize()} branch operations use original values (no dependencies)",
        f"- {second_branch.capitalize()} branch reads values that may have been written by {first_branch} branch",
        f"- The two steps are SEPARATE operations, ensuring proper ordering",
    ])

    return '\n'.join(lines)


def extract_local_variables(c_code):
    """
    Extract local variable definitions like:
    int vl = 63;
    """
    local_vars = {}
    pattern = r'(?:int|float|double|real_t)\s+(\w+)\s*=\s*(\d+)\s*;'

    for match in re.finditer(pattern, c_code):
        var_name = match.group(1)
        value = int(match.group(2))
        local_vars[var_name] = value

    return local_vars


def analyze_pointer_aliasing(c_code, eliminated_deps=None, distribution_result=None):
    """
    Analyze C code for pointer aliasing patterns.

    Args:
        c_code: The C source code to analyze
        eliminated_deps: Optional list of array names whose loop-carried dependencies
                        are eliminated by statement reordering. These should be
                        excluded from sequential pattern detection.
        distribution_result: Optional loop distribution analysis result. If strided
                           prefix sum is detected with verified_safe=True, the
                           aliasing analysis should use the per-stream analysis
                           instead of whole-loop analysis.

    Returns dict with:
    - 'has_aliasing': bool
    - 'aliases': dict of pointer aliases
    - 'same_array_accesses': list of read/write pairs to same array
    - 'strip_size': int or None - safe vectorization strip size
    - 'pattern_type': str - 'fully_parallel', 'strip_vectorizable', 'sequential'
    - 'advice': str - implementation advice
    """
    if eliminated_deps is None:
        eliminated_deps = []

    # Check for strided prefix sum with verified safety
    # In this case, the per-stream analysis shows no loop-carried RAW
    strided_ps_arrays = {}
    if distribution_result and distribution_result.get('strided_prefix_sums'):
        for sps in distribution_result['strided_prefix_sums']:
            if sps.get('verified_safe'):
                arr = sps.get('array')
                if arr:
                    strided_ps_arrays[arr] = sps
    result = {
        'has_aliasing': False,
        'aliases': {},
        'same_array_accesses': [],
        'strip_size': None,
        'pattern_type': 'fully_parallel',
        'advice': None,
        'details': {}
    }

    scop_code = extract_scop_code(c_code)

    # Check for nested loops - pointer aliasing analysis is designed for 1D loops only
    # For nested loops, defer to parallel dims analysis
    nested_loop_pattern = r'for\s*\([^)]*\)\s*\{[^}]*for\s*\('
    if re.search(nested_loop_pattern, scop_code, re.DOTALL):
        result['pattern_type'] = 'nested_loops_skip'
        result['advice'] = "Nested loops detected - use parallel dims analysis instead of pointer aliasing analysis"
        return result

    # Extract local variables (for resolving offset expressions)
    local_vars = extract_local_variables(c_code)
    result['details']['local_vars'] = local_vars

    # Extract pointer aliases
    aliases = extract_pointer_assignments(scop_code)
    result['aliases'] = aliases

    # Check if any alias points to an array with offset
    for ptr, (base, offset) in aliases.items():
        if offset != '0':
            result['has_aliasing'] = True
            break

    # Extract loop info
    loop_info = extract_loop_bounds(scop_code)
    if not loop_info:
        return result

    loop_var = loop_info['var']
    stride = loop_info.get('stride', 1)
    result['details']['loop_var'] = loop_var
    result['details']['stride'] = stride

    # Extract array accesses
    accesses = extract_array_accesses(scop_code, loop_var)

    # Resolve aliases
    resolved = resolve_aliases(accesses, aliases, local_vars)
    result['details']['resolved_accesses'] = resolved

    # Find same-array read/write pairs
    writes = [a for a in resolved if a['is_write']]
    reads = [a for a in resolved if not a['is_write']]

    same_array_pairs = []
    for w in writes:
        for r in reads:
            if w['resolved_array'] == r['resolved_array']:
                # Skip pairs in different branches (mutually exclusive regions)
                w_branch = w.get('branch')
                r_branch = r.get('branch')
                if w_branch is not None and r_branch is not None and w_branch != r_branch:
                    # These are in mutually exclusive branches, no actual dependency
                    continue

                same_array_pairs.append({
                    'array': w['resolved_array'],
                    'write_offset': w['resolved_offset'],
                    'read_offset': r['resolved_offset'],
                    'write_original': w['original_array'],
                    'read_original': r['original_array']
                })

    result['same_array_accesses'] = same_array_pairs

    # Check for mutually exclusive branch reordering pattern
    branch_reorder = analyze_branch_reordering(resolved)
    if branch_reorder:
        result['pattern_type'] = 'branch_reorder'
        result['branch_reorder'] = branch_reorder
        result['advice'] = generate_branch_reorder_advice(branch_reorder)
        return result

    # Analyze dependency pattern
    if not same_array_pairs:
        result['pattern_type'] = 'fully_parallel'
        result['advice'] = "No same-array read/write - fully parallelizable."
        return result

    # Track minimum positive offset_diff (most restrictive RAW dependency)
    # For loops with stride > 1 (manually unrolled), only offset_diff >= stride
    # represents a true inter-iteration RAW dependency. Pairs with
    # 0 < offset_diff < stride are intra-iteration dependencies handled
    # by statement ordering within the loop body.
    min_positive_offset_diff = None
    min_offset_pair = None
    has_unknown = False

    for pair in same_array_pairs:
        # Skip arrays whose dependencies are eliminated by statement reordering
        if pair['array'] in eliminated_deps:
            continue

        # Check if this array has strided prefix sum transformation
        # If verified_safe, the per-stream analysis shows no loop-carried RAW
        if pair['array'] in strided_ps_arrays:
            sps = strided_ps_arrays[pair['array']]
            if sps.get('verified_safe'):
                # Per-stream analysis: each stream reads initial value once,
                # writes to non-overlapping indices. No loop-carried RAW.
                # Skip this pair - it's handled by loop distribution.
                continue

        w_off = pair['write_offset']
        r_off = pair['read_offset']

        if w_off is None or r_off is None:
            has_unknown = True
            continue

        offset_diff = w_off - r_off

        if offset_diff >= stride:
            # Write is ahead of read by at least one full stride -
            # true inter-iteration RAW dependency
            if min_positive_offset_diff is None or offset_diff < min_positive_offset_diff:
                min_positive_offset_diff = offset_diff
                min_offset_pair = pair

    # Determine pattern based on minimum positive offset_diff
    if has_unknown and min_positive_offset_diff is None:
        result['pattern_type'] = 'unknown'
        result['advice'] = "Complex index expression - manual analysis needed."
    elif min_positive_offset_diff is not None:
        if min_positive_offset_diff == 1:
            # Strip size 1 means NO parallelization - must be fully sequential
            result['pattern_type'] = 'sequential'
            result['strip_size'] = 1
            result['advice'] = generate_sequential_advice(min_offset_pair)
        else:
            # Can be vectorized in strips
            result['pattern_type'] = 'strip_vectorizable'
            result['strip_size'] = min_positive_offset_diff
            result['advice'] = generate_strip_advice(min_offset_pair, min_positive_offset_diff)
    else:
        # No positive offset_diff found - fully parallelizable
        result['pattern_type'] = 'fully_parallel'
        result['advice'] = "No loop-carried RAW dependency - fully parallelizable."

    return result


def generate_sequential_advice(pair):
    """Generate advice for sequential (non-parallelizable) pattern."""
    lines = [
        "⚠️ STRICTLY SEQUENTIAL - NO PARALLELIZATION POSSIBLE",
        "",
        f"Pattern: {pair['array']}[i+{pair['write_offset']}] = {pair['array']}[i+{pair['read_offset']}] + ...",
        f"Write offset: {pair['write_offset']}, Read offset: {pair['read_offset']}",
        "Offset difference: 1",
        "",
        "This pattern has a DIRECT loop-carried RAW dependency:",
        "- Each iteration reads the value written in the PREVIOUS iteration",
        "- Iterations CANNOT be parallelized or vectorized",
        "",
        "IMPLEMENTATION: Process ALL iterations strictly sequentially:",
        "```python",
        "# SEQUENTIAL ONLY - no parallelization!",
        "for i in range(start, end):",
        f"    # Load {pair['array']}[i-1] (from previous iteration)",
        f"    # Compute {pair['array']}[i] = ...",
        f"    # Store {pair['array']}[i]",
        "```",
        "",
        "For Triton: Use a single thread with a sequential loop inside the kernel.",
    ]
    return "\n".join(lines)


def generate_strip_advice(pair, strip_size):
    """Generate advice for strip-vectorizable pattern."""
    lines = [
        "POINTER ALIASING WITH LOOP-CARRIED RAW DEPENDENCY DETECTED",
        "",
        f"Pattern: {pair['array']}[i+{pair['write_offset']}] = {pair['array']}[i+{pair['read_offset']}] + ...",
        f"Write offset: {pair['write_offset']}, Read offset: {pair['read_offset']}",
        f"Offset difference: {strip_size}",
        "",
        "This pattern has a loop-carried RAW dependency:",
        f"- At iteration i={strip_size}, we read position {strip_size} which was written at i=0",
        "",
        f"HOWEVER, this CAN be vectorized in strips of {strip_size} elements:",
        f"- Within each strip of {strip_size}, there's no dependency",
        "- Process strips sequentially, parallelize within each strip",
        "",
        "IMPLEMENTATION PATTERN:",
        "```python",
        f"STRIP_SIZE = {strip_size}",
        "n_elements = LEN_1D - 1",
        "",
        "# Process in strips - strips must be sequential",
        "for strip_start in range(0, n_elements, STRIP_SIZE):",
        "    strip_end = min(strip_start + STRIP_SIZE, n_elements)",
        "    strip_len = strip_end - strip_start",
        "    ",
        "    # Within strip - can be parallelized",
        "    idx = torch.arange(strip_start, strip_end, device=arr.device)",
        f"    arr[idx + {pair['write_offset']}] = arr[idx + {pair['read_offset']}] + ...",
        "```",
        "",
        "For Triton kernel (single kernel launch with strip loop inside):",
        "```python",
        "@triton.jit",
        f"def kernel(arr_ptr, ..., n_elements, STRIP_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):",
        f"    # Loop over strips sequentially inside the kernel",
        f"    offsets = tl.arange(0, BLOCK_SIZE)  # BLOCK_SIZE <= {strip_size}",
        f"    for strip_start in range(0, n_elements, STRIP_SIZE):",
        "        idx = strip_start + offsets",
        "        mask = idx < n_elements",
        f"        # Load from read position (idx + {pair['read_offset']})",
        f"        vals = tl.load(arr_ptr + idx + {pair['read_offset']}, mask=mask)",
        f"        # Store to write position (idx + {pair['write_offset']})",
        f"        tl.store(arr_ptr + idx + {pair['write_offset']}, result, mask=mask)",
        "",
        "# Launch a SINGLE kernel (grid=1), strip loop runs inside the kernel",
        f"kernel[(1,)](arr, ..., n_elements=N, STRIP_SIZE={strip_size}, BLOCK_SIZE={strip_size})",
        "```",
        "",
        f"CRITICAL: BLOCK_SIZE and STRIP_SIZE must be <= {strip_size} to avoid race conditions!",
        "Strips are processed sequentially inside the kernel to avoid multiple kernel launches.",
    ]

    return '\n'.join(lines)


def format_aliasing_for_prompt(aliasing_result):
    """Format pointer aliasing analysis for inclusion in LLM prompt."""
    if aliasing_result['pattern_type'] in ('fully_parallel', 'unknown', 'nested_loops_skip'):
        return None  # No special instructions needed (or can't provide useful advice)

    lines = []
    lines.append("=" * 60)
    lines.append("POINTER ALIASING PATTERN DETECTED")
    lines.append("=" * 60)
    lines.append("")

    if aliasing_result['aliases']:
        lines.append("Detected pointer aliases:")
        for ptr, (base, offset) in aliasing_result['aliases'].items():
            if offset != '0':
                lines.append(f"  {ptr} = {base} + {offset}")

    lines.append("")

    if aliasing_result['same_array_accesses']:
        lines.append("Same-array read/write pairs (after resolving aliases):")
        for pair in aliasing_result['same_array_accesses']:
            lines.append(f"  Array: {pair['array']}")
            lines.append(f"    Write: [{pair['write_original']}] -> offset {pair['write_offset']}")
            lines.append(f"    Read:  [{pair['read_original']}] -> offset {pair['read_offset']}")

    lines.append("")
    lines.append(f"Pattern type: {aliasing_result['pattern_type']}")

    if aliasing_result['strip_size']:
        lines.append(f"Safe strip size: {aliasing_result['strip_size']}")

    lines.append("")

    if aliasing_result['advice']:
        lines.append(aliasing_result['advice'])

    lines.append("")
    lines.append("=" * 60)

    return '\n'.join(lines)


def analyze_kernel_aliasing(kernel_file, eliminated_deps=None, distribution_result=None):
    """Analyze a kernel file for pointer aliasing patterns.

    Args:
        kernel_file: Path to the kernel C file
        eliminated_deps: Optional list of array names whose loop-carried dependencies
                        are eliminated by statement reordering
        distribution_result: Optional loop distribution analysis result
    """
    if not os.path.exists(kernel_file):
        return None

    with open(kernel_file, 'r') as f:
        c_code = f.read()

    return analyze_pointer_aliasing(c_code, eliminated_deps=eliminated_deps, distribution_result=distribution_result)


def main():
    """Test pointer aliasing detection on s421-s424."""
    test_kernels = ['s421', 's422', 's423', 's424']

    print("=" * 80)
    print("POINTER ALIASING PATTERN DETECTION")
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

        result = analyze_kernel_aliasing(kernel_file)
        if result:
            print(f"\nPattern type: {result['pattern_type']}")
            print(f"Has aliasing: {result['has_aliasing']}")
            if result['aliases']:
                print(f"Aliases: {result['aliases']}")
            if result['strip_size']:
                print(f"Strip size: {result['strip_size']}")

            prompt_text = format_aliasing_for_prompt(result)
            if prompt_text:
                print(f"\n{prompt_text}")
        else:
            print("\nFailed to analyze kernel")


if __name__ == "__main__":
    main()
