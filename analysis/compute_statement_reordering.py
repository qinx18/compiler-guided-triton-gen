#!/usr/bin/env python3
"""
Detect statement reordering opportunities in loops.

This analysis identifies patterns where reordering statements can enable parallelization
by eliminating loop-carried dependencies.

Example s211:
    for (int i = 1; i < LEN_1D-1; i++) {
        a[i] = b[i - 1] + c[i] * d[i];   // S0: reads b[i-1]
        b[i] = b[i + 1] - e[i] * d[i];   // S1: writes b[i]
    }

S0 reads b[i-1] which was written by S1 in the previous iteration.
This creates a loop-carried dependency that prevents parallelization.

Transformation: Reorder to compute b[i] first, then use it immediately:
    - PROLOGUE: a[1] = b[0] + c[1]*d[1]  (b[0] never modified)
    - MAIN LOOP (parallel): b[i] = b_orig[i+1] - e[i]*d[i]; a[i+1] = b[i] + c[i+1]*d[i+1]
    - EPILOGUE: b[n-2] = b_orig[n-1] - e[n-2]*d[n-2]
"""

import subprocess
import yaml
import re
import os

PET_PATH = "/home/qinxiao/workspace/pet/pet"
KERNELS_DIR = "/home/qinxiao/workspace/compiler-guided-triton-gen/analysis/kernels"


def extract_c_statements(kernel_name):
    """Extract original C statements from kernel file."""
    kernel_file = os.path.join(KERNELS_DIR, f"{kernel_name}.c")
    if not os.path.exists(kernel_file):
        return []

    with open(kernel_file, 'r') as f:
        content = f.read()

    # Extract code between #pragma scop and #pragma endscop
    scop_match = re.search(r'#pragma scop\s*(.*?)\s*#pragma endscop', content, re.DOTALL)
    if not scop_match:
        return []

    scop_code = scop_match.group(1)

    # Extract individual statements (lines ending with ;)
    statements = []
    for line in scop_code.split('\n'):
        line = line.strip()
        if line and ';' in line and not line.startswith('for') and not line.startswith('if'):
            # Extract just the assignment part
            stmt = line.rstrip(';').strip()
            if '=' in stmt:
                statements.append(stmt)

    return statements


def inline_scalar_intermediates(statements):
    """
    Inline scalar intermediate variables in statements.

    For patterns like:
        t = a[i] + b[i]
        a[i] = t + c[i-1]
        t = c[i] * d[i]
        c[i] = t

    Returns inlined statements:
        a[i] = (a[i] + b[i]) + c[i-1]
        c[i] = c[i] * d[i]

    This eliminates same-iteration scalar temporaries that can be trivially inlined.
    """
    if not statements:
        return []

    # First pass: identify scalar assignments and track their definitions in order
    # scalar_defs[scalar_name] = list of (stmt_idx, rhs_expression)
    scalar_defs = {}

    for idx, stmt in enumerate(statements):
        parts = stmt.split('=', 1)
        if len(parts) != 2:
            continue
        lhs = parts[0].strip()
        rhs = parts[1].strip()

        # Check if LHS is a scalar (no brackets)
        if '[' not in lhs and re.match(r'^[a-zA-Z_]\w*$', lhs):
            # This is a scalar assignment
            scalar_name = lhs
            if scalar_name not in scalar_defs:
                scalar_defs[scalar_name] = []
            scalar_defs[scalar_name].append((idx, rhs))

    if not scalar_defs:
        return statements  # No scalars to inline

    # Build inlined statements - merge scalar assignments into their uses
    inlined = []

    for idx, stmt in enumerate(statements):
        parts = stmt.split('=', 1)
        if len(parts) != 2:
            inlined.append(stmt)
            continue

        lhs = parts[0].strip()
        rhs = parts[1].strip()

        # Skip scalar definitions - they will be inlined into their uses
        if '[' not in lhs and re.match(r'^[a-zA-Z_]\w*$', lhs):
            continue

        # This is an array assignment - inline any scalar references in RHS
        new_rhs = rhs
        for scalar_name, defs in scalar_defs.items():
            # Find the most recent definition before this statement
            pattern = rf'\b{scalar_name}\b'
            if re.search(pattern, new_rhs):
                # Find the most recent def_idx < idx
                recent_def = None
                for def_idx, scalar_expr in defs:
                    if def_idx < idx:
                        recent_def = (def_idx, scalar_expr)
                    else:
                        break  # defs are in order, no need to check further

                if recent_def:
                    _, scalar_expr = recent_def
                    new_rhs = re.sub(pattern, f'({scalar_expr})', new_rhs)

        inlined.append(f'{lhs} = {new_rhs}')

    return inlined


def extract_c_statements_inlined(kernel_name):
    """
    Extract C statements with scalar intermediates inlined.

    This is the preferred function for statement reordering analysis
    because it produces cleaner statements without intermediate scalars.
    """
    raw_statements = extract_c_statements(kernel_name)
    return inline_scalar_intermediates(raw_statements)


def transform_c_statement(stmt, transform_type, arr_name, use_copy_arrays=None, loop_upper_offset=1):
    """
    Transform a C statement according to reordering rules.

    transform_type:
        'producer': The statement that writes arr[i], reads from arr[i+1] -> use copy for arr
        'consumer_shifted': The statement that reads arr[i-1] -> shift all indices by +1, arr[i-1] becomes arr[i]
        'prologue': Consumer at i=1
        'epilogue': Producer at i=(N-loop_upper_offset), e.g., N-1 for i<N, N-2 for i<N-1

    loop_upper_offset: Offset from N for the last iteration (1 for i<N, 2 for i<N-1, etc.)
    """
    if use_copy_arrays is None:
        use_copy_arrays = set()

    if transform_type == 'producer':
        # Replace arr[i+1] reads with arr_copy[i+1] for arrays needing copy
        # BUT keep the LHS (write target) unchanged
        parts = stmt.split('=', 1)
        if len(parts) == 2:
            lhs, rhs = parts
            # Only replace in RHS (reads)
            for copy_arr in use_copy_arrays:
                rhs = re.sub(rf'\b{copy_arr}\[([^\]]+)\]', rf'{copy_arr}_copy[\1]', rhs)
            return lhs + '=' + rhs
        return stmt

    elif transform_type == 'consumer_shifted':
        # Shift all indices by +1: arr[i-1] -> arr[i], arr[i] -> arr[i+1], etc.
        result = stmt

        # First, handle the LHS (shift by +1)
        lhs_match = re.match(r'(\w+)\[([^\]]+)\]', result)
        if lhs_match:
            lhs_arr = lhs_match.group(1)
            lhs_idx = lhs_match.group(2).strip()
            new_lhs_idx = shift_index(lhs_idx, 1)
            result = re.sub(rf'^{lhs_arr}\[[^\]]+\]', f'{lhs_arr}[{new_lhs_idx}]', result)

        # Then, handle RHS accesses (shift by +1, but arr[i-1] becomes arr[i] using the just-computed value)
        def shift_rhs_access(match):
            arr = match.group(1)
            idx = match.group(2).strip()
            # If this is arr[i-1] and arr is the producer array, use arr[i] (just computed)
            if arr == arr_name:
                # This reads the value just computed in this iteration
                return f'{arr}[i]'
            else:
                new_idx = shift_index(idx, 1)
                return f'{arr}[{new_idx}]'

        # Skip the LHS when processing RHS
        parts = result.split('=', 1)
        if len(parts) == 2:
            lhs, rhs = parts
            rhs = re.sub(r'(\w+)\[([^\]]+)\]', shift_rhs_access, rhs)
            result = lhs + '=' + rhs

        return result

    elif transform_type == 'prologue':
        # Consumer at i=1: substitute i=1 into original consumer statement
        result = stmt
        result = re.sub(r'\bi\b', '1', result)
        # Simplify expressions like 1-1=0, 1+0=1 (with or without spaces)
        result = re.sub(r'\[1\s*-\s*1\]', '[0]', result)
        result = re.sub(r'\[1\s*\+\s*0\]', '[1]', result)
        return result

    elif transform_type == 'epilogue':
        # Producer at i=(N-loop_upper_offset) (last iteration): use copy for reads only, keep LHS unchanged
        parts = stmt.split('=', 1)
        if len(parts) == 2:
            lhs, rhs = parts
            # Only replace in RHS (reads)
            for copy_arr in use_copy_arrays:
                rhs = re.sub(rf'\b{copy_arr}\[([^\]]+)\]', rf'{copy_arr}_copy[\1]', rhs)
            result = lhs + '=' + rhs
        else:
            result = stmt

        # Replace i with (N-offset)
        offset = loop_upper_offset
        if offset == 1:
            result = re.sub(r'\bi\b', '(N-1)', result)
            # Simplify: (N-1)+1 = N, (N-1)-1 = N-2
            result = re.sub(r'\(N-1\)\s*\+\s*1', 'N', result)
            result = re.sub(r'\(N-1\)\s*-\s*1', 'N-2', result)
            result = re.sub(r'\[\(N-1\)\]', '[N-1]', result)
        elif offset == 2:
            result = re.sub(r'\bi\b', '(N-2)', result)
            # Simplify: (N-2)+1 = N-1, (N-2)-1 = N-3
            result = re.sub(r'\(N-2\)\s*\+\s*1', 'N-1', result)
            result = re.sub(r'\(N-2\)\s*-\s*1', 'N-3', result)
            result = re.sub(r'\[\(N-2\)\]', '[N-2]', result)
        else:
            result = re.sub(r'\bi\b', f'(N-{offset})', result)
        return result

    return stmt


def shift_index(idx_expr, shift):
    """Shift an index expression by a constant."""
    idx_expr = idx_expr.strip()

    # Handle simple cases
    if idx_expr == 'i':
        if shift == 0:
            return 'i'
        elif shift > 0:
            return f'i+{shift}'
        else:
            return f'i{shift}'

    # Handle i-1, i+1, etc.
    match = re.match(r'i\s*([+-])\s*(\d+)', idx_expr)
    if match:
        sign = match.group(1)
        val = int(match.group(2))
        if sign == '-':
            val = -val
        new_val = val + shift
        if new_val == 0:
            return 'i'
        elif new_val > 0:
            return f'i+{new_val}'
        else:
            return f'i{new_val}'

    # Fallback: just append the shift
    if shift == 0:
        return idx_expr
    elif shift > 0:
        return f'({idx_expr})+{shift}'
    else:
        return f'({idx_expr}){shift}'


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
    Example: '{ S_0[i] -> b[(-1 + i)] }' returns ('b', '-1 + i')
    """
    match = re.search(r'(\w+)\[([^\]]+)\](?:\s*\})?$', access_str)
    if match:
        return match.group(1), match.group(2)
    return None, None


def extract_index_offset(index_expr, dim_var):
    """
    Extract the offset from a dimension variable in an index expression.
    Example: '-1 + i' with dim_var='i' returns -1
    Example: '1 + i' with dim_var='i' returns 1
    Example: 'i' with dim_var='i' returns 0
    """
    index_expr = index_expr.strip().replace('(', '').replace(')', '')

    # Check if it's just the variable
    if index_expr == dim_var:
        return 0

    # Parse expressions like "-1 + i", "i + 1", "1 + i", etc.
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


def extract_loop_upper_bound(domain):
    """
    Extract the loop upper bound from ISL domain constraint.

    Example: '{ S_0[i] : 0 < i <= 31998 }' returns 31998
    Example: '{ S_0[i] : 1 <= i < 31999 }' returns 31998
    Example: '{ S_0[i] : 0 < i < 32000 }' returns 31999

    Returns the upper bound value, or None if not found.
    Also returns whether the original loop uses < or <= comparison.
    """
    # Pattern: i <= N (inclusive upper bound)
    match_le = re.search(r'\bi\s*<=\s*(\d+)', domain)
    if match_le:
        return int(match_le.group(1)), 'inclusive'

    # Pattern: i < N (exclusive upper bound)
    match_lt = re.search(r'\bi\s*<\s*(\d+)', domain)
    if match_lt:
        return int(match_lt.group(1)) - 1, 'exclusive'

    return None, None


def detect_scalar_intermediates(statements_data):
    """
    Detect intermediate scalar variables that can be inlined.

    Pattern: t = expr1; arr[i] = t + expr2  ->  arr[i] = expr1 + expr2

    Returns a mapping of scalar names to their defining expressions,
    and identifies which statements can be merged.
    """
    scalar_defs = {}  # scalar_name -> (stmt_idx, defining reads)
    scalar_uses = {}  # scalar_name -> list of (stmt_idx, usage context)

    for stmt_idx, stmt in enumerate(statements_data):
        domain = stmt.get('domain', '')
        body = stmt.get('body', {})

        # Check for scalar write (no array index in domain mapping)
        # Scalars in PET appear as writes without array subscripts
        def find_scalar_writes(node, reads_so_far):
            if not isinstance(node, dict):
                return
            if node.get('type') == 'access':
                index_str = node.get('index', '')
                # Scalar access pattern: { S_n[i] -> t[] }
                if '[]' in index_str or re.search(r'->\s*\w+\s*\}', index_str):
                    ref = node.get('reference', '')
                    arr_match = re.search(r'->\s*(\w+)', index_str)
                    if arr_match:
                        scalar_name = arr_match.group(1)
                        if node.get('write', 0):
                            scalar_defs[scalar_name] = (stmt_idx, reads_so_far.copy())
                        if node.get('read', 0):
                            if scalar_name not in scalar_uses:
                                scalar_uses[scalar_name] = []
                            scalar_uses[scalar_name].append(stmt_idx)

            for key in ['arguments', 'body', 'expr']:
                if key in node:
                    if isinstance(node[key], list):
                        for item in node[key]:
                            find_scalar_writes(item, reads_so_far)
                    else:
                        find_scalar_writes(node[key], reads_so_far)

        # First collect reads, then look for writes
        reads, writes = extract_accesses(stmt)
        find_scalar_writes(body, reads)

    return scalar_defs, scalar_uses


def simplify_with_scalar_inlining(statements_data):
    """
    Create a simplified view of statements by conceptually inlining scalar intermediates.

    For patterns like:
        t = a[i] + b[i]
        arr[i] = t + c[i-1]

    Returns simplified statement descriptions for clearer advice.
    """
    simplified = []
    scalar_defs, scalar_uses = detect_scalar_intermediates(statements_data)

    # Group consecutive statements that use intermediate scalars
    # For now, return a simplified description based on array accesses only

    for stmt_idx, stmt in enumerate(statements_data):
        reads, writes = extract_accesses(stmt)
        domain = stmt.get('domain', '')

        # Only include statements that write to arrays (not scalars)
        array_writes = [w for w in writes if '[]' not in w.get('index', '')]
        if array_writes:
            simplified.append({
                'original_indices': [stmt_idx],
                'reads': reads,
                'writes': array_writes,
                'domain': domain
            })

    return simplified


def analyze_statement_reordering(statements_data, kernel_name=None):
    """
    Analyze multiple statements in a loop for potential statement reordering optimization.

    This detects the pattern where:
    - Statement S0 reads array[i-k] (from previous iterations)
    - Statement S1 writes array[i] (producer)
    - Reordering S1 before S0 (with index shift) can eliminate inter-iteration dependency

    For patterns with intermediate scalars (like s261), the analysis conceptually
    inlines them to identify the core dependency pattern.

    Args:
        statements_data: PET analysis output for statements
        kernel_name: Name of kernel (for extracting original C statements)

    Returns dict with:
    - 'applicable': bool - whether reordering can help
    - 'pattern_type': str - type of reordering pattern detected
    - 'reordering_advice': str - detailed advice for implementation
    - 'statements': list - analyzed statement info
    - 'dependencies': list - detected cross-statement dependencies
    - 'simplified_pattern': str - human-readable simplified pattern
    """
    # Extract original C statements with scalar intermediates inlined
    c_statements = extract_c_statements_inlined(kernel_name) if kernel_name else []

    result = {
        'applicable': False,
        'pattern_type': None,
        'reordering_advice': None,
        'statements': [],
        'dependencies': [],
        'transformation': None,
        'simplified_pattern': None
    }

    if len(statements_data) < 2:
        return result

    # Check loop stride - skip manually unrolled loops (stride > 1)
    # These have multiple statements per iteration operating on different indices,
    # not producer-consumer patterns that can be reordered
    first_domain = statements_data[0].get('domain', '')
    loop_stride = extract_loop_stride(first_domain)
    if loop_stride > 1:
        # This is a manually unrolled loop (like s116 with stride 5)
        # Statement reordering doesn't apply
        return result

    # Extract information from each statement
    statements = []
    for stmt in statements_data:
        reads, writes = extract_accesses(stmt)
        domain = stmt.get('domain', '')

        # Parse domain to get loop dimension(s)
        dim_match = re.search(r'S_\d+\[(\w+)(?:,\s*(\w+))?\]', domain)
        if dim_match:
            dims = [d for d in dim_match.groups() if d]
        else:
            dims = []

        stmt_info = {
            'domain': domain,
            'dims': dims,
            'reads': [],
            'writes': []
        }

        # Parse each read/write access
        for r in reads:
            arr, idx = parse_access_array_and_index(r['index'])
            if arr and dims:
                offset = extract_index_offset(idx, dims[0])
                stmt_info['reads'].append({
                    'array': arr,
                    'index_expr': idx,
                    'offset': offset
                })

        for w in writes:
            arr, idx = parse_access_array_and_index(w['index'])
            if arr and dims:
                offset = extract_index_offset(idx, dims[0])
                stmt_info['writes'].append({
                    'array': arr,
                    'index_expr': idx,
                    'offset': offset
                })

        statements.append(stmt_info)

    result['statements'] = statements

    # Detect producer-consumer dependencies across statements
    dependencies = []

    for i, s0 in enumerate(statements):
        for j, s1 in enumerate(statements):
            if i == j:
                continue

            for r in s0['reads']:
                for w in s1['writes']:
                    if r['array'] == w['array']:
                        if r['offset'] is not None and w['offset'] is not None:
                            dep_distance = r['offset'] - w['offset']

                            if dep_distance != 0:
                                dependencies.append({
                                    'consumer_stmt': i,
                                    'producer_stmt': j,
                                    'array': r['array'],
                                    'consumer_offset': r['offset'],
                                    'producer_offset': w['offset'],
                                    'distance': dep_distance,
                                    'type': 'flow' if dep_distance < 0 else 'anti'
                                })

    result['dependencies'] = dependencies

    # Categorize dependencies
    flow_deps = [d for d in dependencies if d['type'] == 'flow']
    anti_deps = [d for d in dependencies if d['type'] == 'anti']

    # Check for TRUE circular: flow dependencies in both directions
    has_true_circular = False
    for fd1 in flow_deps:
        for fd2 in flow_deps:
            if fd1 != fd2:
                if (fd1['consumer_stmt'] == fd2['producer_stmt'] and
                    fd1['producer_stmt'] == fd2['consumer_stmt']):
                    has_true_circular = True
                    break
        if has_true_circular:
            break

    if has_true_circular:
        result['applicable'] = True
        result['pattern_type'] = 'true_circular_dependency'

        advice_lines = [
            f"TRUE CIRCULAR DEPENDENCY DETECTED",
            f"",
            f"This pattern has FLOW dependencies in both directions:",
        ]

        for dep in flow_deps:
            arr = dep['array']
            advice_lines.append(f"  - Flow: S{dep['consumer_stmt']} reads {arr}[i{dep['consumer_offset']:+d}] <- "
                               f"S{dep['producer_stmt']} writes {arr}[i{dep['producer_offset']:+d}]")

        advice_lines.extend([
            f"",
            f"TRANSFORMATION: Requires temporary arrays or wavefront parallelization",
        ])

        result['reordering_advice'] = '\n'.join(advice_lines)
        return result

    # Check for the reorderable pattern
    for dep in flow_deps:
        if dep['distance'] == -1:
            if dep['consumer_stmt'] < dep['producer_stmt']:
                consumer_idx = dep['consumer_stmt']
                producer_idx = dep['producer_stmt']
                arr = dep['array']

                # Check for same-iteration dependency: producer reads from consumer's output
                # If S1 (producer) reads from any array that S0 (consumer) writes,
                # then we can't reorder because S1 depends on S0 within the same iteration
                consumer_stmt_info = statements[consumer_idx]
                producer_stmt_info = statements[producer_idx]

                has_same_iter_dep = False
                for p_read in producer_stmt_info['reads']:
                    for c_write in consumer_stmt_info['writes']:
                        if p_read['array'] == c_write['array']:
                            # Check if offsets indicate same-iteration access
                            p_offset = p_read['offset']
                            c_offset = c_write['offset']
                            if p_offset is not None and c_offset is not None and p_offset == c_offset:
                                has_same_iter_dep = True
                                break
                    if has_same_iter_dep:
                        break

                if has_same_iter_dep:
                    # Can't reorder - producer depends on consumer in same iteration
                    continue

                result['applicable'] = True
                result['pattern_type'] = 'producer_consumer_reorder'

                # Check for safe anti-dependencies
                safe_anti_deps = []
                arrays_needing_copy = set()
                for ad in anti_deps:
                    if ad['distance'] > 0:
                        safe_anti_deps.append(ad)
                        arrays_needing_copy.add(ad['array'])

                # Check for forward reads within same statement
                for stmt_idx, stmt_info in enumerate(statements):
                    for r in stmt_info['reads']:
                        for w in stmt_info['writes']:
                            if r['array'] == w['array'] and r['offset'] is not None and w['offset'] is not None:
                                if r['offset'] > w['offset']:
                                    arrays_needing_copy.add(r['array'])

                result['transformation'] = {
                    'original_order': [f'S{consumer_idx}', f'S{producer_idx}'],
                    'new_order': [f'S{producer_idx}', f'S{consumer_idx}_shifted'],
                    'consumer_stmt': consumer_idx,
                    'producer_stmt': producer_idx,
                    'shared_array': arr,
                    'index_shift': 1,
                    'needs_prologue': True,
                    'needs_epilogue': True,
                    'safe_anti_deps': safe_anti_deps,
                    'arrays_needing_copy': list(arrays_needing_copy),
                }

                # Mark that after reordering, the loop becomes parallelizable
                result['is_parallelizable_after_reordering'] = True
                # The shared array's loop-carried dependency is eliminated
                result['eliminated_dependencies'] = [arr]

                # Collect all array reads/writes to show simplified pattern
                consumer_stmt = statements[consumer_idx]
                producer_stmt = statements[producer_idx]

                # Build simplified pattern description
                consumer_writes = [w for w in consumer_stmt['writes'] if w['array'] != arr]
                consumer_reads = [r for r in consumer_stmt['reads'] if r['array'] != arr]
                producer_reads = [r for r in producer_stmt['reads']]

                # Detect if there are intermediate scalars (more statements than array writes)
                num_array_writes = sum(1 for s in statements for w in s['writes'])
                has_scalar_intermediates = len(statements_data) > num_array_writes

                # Generate detailed advice
                advice_lines = [
                    f"STATEMENT REORDERING TRANSFORMATION AVAILABLE",
                    f"",
                    f"Dependency pattern detected:",
                    f"  - Consumer (S{consumer_idx}): reads {arr}[i-1] (from previous iteration)",
                    f"  - Producer (S{producer_idx}): writes {arr}[i]",
                ]

                if has_scalar_intermediates:
                    advice_lines.extend([
                        f"",
                        f"NOTE: Intermediate scalar variables detected. Simplified semantic pattern:",
                    ])

                    # Show simplified pattern
                    for s_idx, s_info in enumerate(statements):
                        if s_info['writes']:
                            write_arrs = [f"{w['array']}[i{w['offset']:+d}]" if w['offset'] else f"{w['array']}[i]"
                                         for w in s_info['writes']]
                            read_arrs = [f"{r['array']}[i{r['offset']:+d}]" if r['offset'] else f"{r['array']}[i]"
                                        for r in s_info['reads']]
                            if write_arrs:
                                advice_lines.append(f"  {', '.join(write_arrs)} = f({', '.join(read_arrs)})")

                consumer_write_arr = consumer_stmt['writes'][0]['array'] if consumer_stmt['writes'] else 'result'

                # Build expression formatters
                def format_access(arr_name, offset, use_copy=False):
                    """Format array access with optional _copy suffix."""
                    base = f"{arr_name}_copy" if use_copy else arr_name
                    if offset is None:
                        return f"{base}[?]"
                    elif offset == 0:
                        return f"{base}[i]"
                    else:
                        return f"{base}[i{offset:+d}]"

                def format_access_shifted(arr_name, offset, shift, use_copy=False):
                    """Format array access with index shift for transformed consumer."""
                    base = f"{arr_name}_copy" if use_copy else arr_name
                    new_offset = (offset if offset else 0) + shift
                    if new_offset == 0:
                        return f"{base}[i]"
                    else:
                        return f"{base}[i{new_offset:+d}]"

                # Identify which arrays need _copy
                producer_copy_arrays = set()
                for r in producer_stmt['reads']:
                    if r['offset'] is not None and r['offset'] > 0:
                        for s in statements:
                            for w in s['writes']:
                                if w['array'] == r['array']:
                                    producer_copy_arrays.add(r['array'])
                                    break
                all_copy_arrays = set(arrays_needing_copy) | producer_copy_arrays

                # Build the producer expression (e.g., "a_copy[i+1] * d[i]")
                producer_write = producer_stmt['writes'][0] if producer_stmt['writes'] else None
                producer_read_parts = []
                for r in producer_stmt['reads']:
                    use_copy = r['array'] in all_copy_arrays
                    producer_read_parts.append(format_access(r['array'], r['offset'], use_copy))
                producer_expr = " * ".join(producer_read_parts) if len(producer_read_parts) <= 2 else f"f({', '.join(producer_read_parts)})"

                # Build the consumer expression with shifted indices (e.g., "b[i] + c[i+1]")
                consumer_write = consumer_stmt['writes'][0] if consumer_stmt['writes'] else None
                consumer_read_parts = []
                for r in consumer_stmt['reads']:
                    if r['array'] == arr:
                        # This read now uses the value computed in SAME iteration (no shift needed in access)
                        consumer_read_parts.append(f"{arr}[i]")
                    else:
                        # Other reads get shifted by +1
                        use_copy = r['array'] in all_copy_arrays
                        consumer_read_parts.append(format_access_shifted(r['array'], r['offset'], 1, use_copy))
                consumer_expr = " + ".join(consumer_read_parts) if len(consumer_read_parts) <= 2 else f"g({', '.join(consumer_read_parts)})"

                # Build prologue expression (original indices, using original arrays)
                prologue_read_parts = []
                for r in consumer_stmt['reads']:
                    off = r['offset'] if r['offset'] else 0
                    if off == -1:
                        prologue_read_parts.append(f"{r['array']}[0]")
                    elif off == 0:
                        prologue_read_parts.append(f"{r['array']}[1]")
                    else:
                        prologue_read_parts.append(f"{r['array']}[{1 + off}]")
                prologue_expr = " + ".join(prologue_read_parts) if len(prologue_read_parts) <= 2 else f"f({', '.join(prologue_read_parts)})"

                # Build epilogue expression (last producer at i=N-2)
                epilogue_read_parts = []
                for r in producer_stmt['reads']:
                    use_copy = r['array'] in all_copy_arrays
                    base = f"{r['array']}_copy" if use_copy else r['array']
                    off = r['offset'] if r['offset'] else 0
                    # At i=N-2, the access is base[N-2 + off]
                    total_off = -2 + off
                    if total_off == 0:
                        epilogue_read_parts.append(f"{base}[N]")
                    elif total_off == -1:
                        epilogue_read_parts.append(f"{base}[N-1]")
                    else:
                        epilogue_read_parts.append(f"{base}[N{total_off:+d}]")
                epilogue_expr = " * ".join(epilogue_read_parts) if len(epilogue_read_parts) <= 2 else f"f({', '.join(epilogue_read_parts)})"

                advice_lines.extend([
                    f"",
                    f"This creates a loop-carried dependency: iteration i needs {arr}[i-1] from iteration i-1.",
                    f"",
                    f"TRANSFORMATION: Reorder statements and shift consumer index by 1.",
                    f"After transformation, each iteration computes {arr}[i] first, then uses it immediately.",
                ])

                # Extract loop upper bound from domain to determine epilogue index
                # e.g., "i <= 31998" means upper bound is N-2 (for LEN_1D=32000)
                first_domain = statements_data[0].get('domain', '')
                loop_upper, bound_type = extract_loop_upper_bound(first_domain)

                # Calculate loop_upper_offset: offset from N for the last iteration
                # Default to 1 (i < N means last iteration is N-1)
                loop_upper_offset = 1
                if loop_upper is not None:
                    LEN_1D = 32000
                    loop_upper_offset = LEN_1D - loop_upper

                # Try to use actual C statements if available
                # Since scalar intermediates are inlined, statement indices may not match PET indices
                # Find producer and consumer by pattern matching instead
                producer_c = None
                consumer_c = None

                if len(c_statements) >= 2:
                    for stmt in c_statements:
                        # Producer writes arr[i]: look for "arr[i] = " pattern
                        if re.match(rf'^{arr}\[i\]\s*=', stmt):
                            producer_c = stmt
                        # Consumer reads arr[i-1]: look for arr[i-1] on RHS
                        elif re.search(rf'{arr}\[i\s*-\s*1\]', stmt):
                            consumer_c = stmt

                    if producer_c and consumer_c:
                        # Transform to actual expressions
                        producer_transformed = transform_c_statement(producer_c, 'producer', arr, all_copy_arrays)
                        consumer_transformed = transform_c_statement(consumer_c, 'consumer_shifted', arr, all_copy_arrays)
                        prologue_c = transform_c_statement(consumer_c, 'prologue', arr, all_copy_arrays)
                        epilogue_c = transform_c_statement(producer_c, 'epilogue', arr, all_copy_arrays, loop_upper_offset)

                        advice_lines.extend([
                            f"",
                            f"ORIGINAL C STATEMENTS (after scalar inlining):",
                            f"    {consumer_c};  // consumer (reads {arr}[i-1])",
                            f"    {producer_c};  // producer (writes {arr}[i])",
                            f"",
                            f"TRANSFORMED C CODE (with statement reordering):",
                            f"```c",
                        ])
                        for copy_arr in sorted(all_copy_arrays):
                            advice_lines.append(f"real_t {copy_arr}_copy[N];  // clone {copy_arr} before parallel loop")
                            advice_lines.append(f"memcpy({copy_arr}_copy, {copy_arr}, N * sizeof(real_t));")
                        # Calculate the correct loop bounds based on loop_upper_offset
                        # Main loop: i from 1 to (N - loop_upper_offset - 1) inclusive
                        # because a[i+1] must not exceed a[N - loop_upper_offset]
                        main_loop_end = f"N-{loop_upper_offset}" if loop_upper_offset > 0 else "N"
                        main_loop_last = f"N-{loop_upper_offset + 1}" if loop_upper_offset >= 0 else f"N-1"
                        epilogue_idx = f"N-{loop_upper_offset}" if loop_upper_offset > 0 else "N"

                        advice_lines.extend([
                            f"",
                            f"// PROLOGUE (i=1):",
                            f"{prologue_c};",
                            f"",
                            f"// MAIN PARALLEL LOOP (i from 1 to {main_loop_last} inclusive):",
                            f"// Each iteration is independent - can be parallelized",
                            f"for (int i = 1; i < {main_loop_end}; i++) {{  // Triton mask: (i >= 1) & (i < {main_loop_end})",
                            f"    {producer_transformed};  // Step 1: compute {arr}[i]",
                            f"    {consumer_transformed};  // Step 2: compute {consumer_write_arr}[i+1] using {arr}[i]",
                            f"}}",
                            f"",
                            f"// EPILOGUE (i={epilogue_idx}):",
                            f"{epilogue_c};",
                            f"```",
                        ])
                    else:
                        # Fallback to computed expressions
                        advice_lines.extend([
                            f"",
                            f"EXACT TRANSFORMED FUNCTION:",
                            f"```python",
                            f"def func_triton({', '.join(sorted(set(r['array'] for s in statements for r in s['reads']) | set(w['array'] for s in statements for w in s['writes'])))}):",
                            f"    N = {consumer_write_arr}.shape[0]",
                        ])
                        for copy_arr in sorted(all_copy_arrays):
                            advice_lines.append(f"    {copy_arr}_copy = {copy_arr}.clone()")
                        advice_lines.extend([
                            f"",
                            f"    # PROLOGUE: {consumer_write_arr}[1] = {prologue_expr}",
                            f"",
                            f"    # MAIN PARALLEL LOOP (i from 1 to N-3, i.e., range(1, N-2)):",
                            f"    for i in range(1, N-2):  # mask: (i >= 1) & (i < N-2)",
                            f"        {arr}[i] = {producer_expr}",
                            f"        {consumer_write_arr}[i+1] = {consumer_expr}",
                            f"",
                            f"    # EPILOGUE: {arr}[N-2] = {epilogue_expr}",
                            f"```",
                        ])
                else:
                    # No C statements available, use computed expressions
                    advice_lines.extend([
                        f"",
                        f"EXACT TRANSFORMED FUNCTION:",
                        f"```python",
                        f"def func_triton({', '.join(sorted(set(r['array'] for s in statements for r in s['reads']) | set(w['array'] for s in statements for w in s['writes'])))}):",
                        f"    N = {consumer_write_arr}.shape[0]",
                    ])
                    for copy_arr in sorted(all_copy_arrays):
                        advice_lines.append(f"    {copy_arr}_copy = {copy_arr}.clone()")
                    advice_lines.extend([
                        f"",
                        f"    # PROLOGUE: {consumer_write_arr}[1] = {prologue_expr}",
                        f"",
                        f"    # MAIN PARALLEL LOOP (i from 1 to N-3, i.e., range(1, N-2)):",
                        f"    for i in range(1, N-2):  # mask: (i >= 1) & (i < N-2)",
                        f"        {arr}[i] = {producer_expr}",
                        f"        {consumer_write_arr}[i+1] = {consumer_expr}",
                        f"",
                        f"    # EPILOGUE: {arr}[N-2] = {epilogue_expr}",
                        f"```",
                    ])

                # Calculate the correct loop bounds for the CRITICAL message
                main_loop_end_msg = f"N-{loop_upper_offset}" if loop_upper_offset > 0 else "N"
                main_loop_last_msg = f"N-{loop_upper_offset + 1}" if loop_upper_offset >= 0 else "N-1"

                advice_lines.extend([
                    f"",
                    f"For Triton: Convert the main loop to a parallel kernel where each thread",
                    f"handles one value of i, computing both {arr}[i] and {consumer_write_arr}[i+1].",
                    f"",
                    f"CRITICAL: The main loop mask should be (i >= 1) & (i < {main_loop_end_msg}). The loop runs i from 1 to {main_loop_last_msg} inclusive.",
                ])

                if arrays_needing_copy:
                    advice_lines.extend([
                        f"",
                        f"WAR RACE CONDITION WARNING:",
                        f"Arrays needing read-only copy: {list(arrays_needing_copy)}",
                        f"Clone these arrays BEFORE the parallel loop to preserve original values.",
                    ])

                if safe_anti_deps:
                    advice_lines.append(f"")
                    advice_lines.append(f"Additional arrays read with forward offset (need original values):")
                    for ad in safe_anti_deps:
                        advice_lines.append(f"  - {ad['array']}[i{ad['consumer_offset']:+d}] in S{ad['consumer_stmt']}")

                result['reordering_advice'] = '\n'.join(advice_lines)
                break

    return result


def format_reordering_for_prompt(reordering_result):
    """Format statement reordering analysis for inclusion in LLM prompt."""
    if not reordering_result or not reordering_result.get('applicable'):
        return None

    lines = []
    lines.append("=" * 60)
    lines.append("STATEMENT REORDERING OPPORTUNITY DETECTED")
    lines.append("=" * 60)
    lines.append("")

    if reordering_result['reordering_advice']:
        lines.append(reordering_result['reordering_advice'])

    lines.append("")
    lines.append("=" * 60)

    return '\n'.join(lines)


def analyze_kernel_reordering(kernel_name):
    """
    Analyze a kernel for statement reordering opportunities.

    Args:
        kernel_name: Name of the kernel (e.g., 's211')

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

    result = analyze_statement_reordering(statements, kernel_name)

    if result['applicable']:
        return result
    return None


def main():
    """Test statement reordering detection."""
    test_kernels = ['s211', 's1213', 's212']

    print("=" * 80)
    print("STATEMENT REORDERING ANALYSIS")
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

        result = analyze_kernel_reordering(kernel)
        if result:
            print(f"\nApplicable: {result['applicable']}")
            print(f"Pattern type: {result['pattern_type']}")
            if result['transformation']:
                print(f"Arrays needing copy: {result['transformation'].get('arrays_needing_copy', [])}")
            print()
            formatted = format_reordering_for_prompt(result)
            if formatted:
                print(formatted)
        else:
            print("\nNo statement reordering opportunity detected")


if __name__ == "__main__":
    main()
