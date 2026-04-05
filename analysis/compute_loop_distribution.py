#!/usr/bin/env python3
"""
Detect loop distribution opportunities in loops.

Loop distribution splits a single loop with multiple statements into
separate loops, allowing parallelizable statements to run independently
from sequential ones.

Example s222:
    for (int i = 1; i < LEN_1D; i++) {
        a[i] += b[i] * c[i];         // S0: parallelizable
        e[i] = e[i - 1] * e[i - 1];  // S1: sequential (recurrence)
        a[i] -= b[i] * c[i];         // S2: parallelizable (but cancels S0!)
    }

Analysis:
  1. S0 and S2 cancel out (a += x then a -= x) -> dead code
  2. S1 has loop-carried dependency (e[i] depends on e[i-1]) -> sequential
  3. But S1 is a power recurrence: e[i] = e[i-1]^2 = e[0]^(2^i) -> parallelizable!

Transformations:
  - Recognize and eliminate canceling operations
  - Distribute remaining statements into separate loops
  - Apply mathematical transformations for power recurrences
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


def parse_access(index_str):
    """Parse ISL access string to get array name and index."""
    match = re.search(r'(\w+)\[([^\]]*)\](?:\s*\})?$', index_str)
    if match:
        return match.group(1), match.group(2)
    return None, None


def extract_index_offset(index_expr, dim_var):
    """Extract offset from dimension variable in index expression."""
    if not index_expr or not dim_var:
        return None

    index_expr = index_expr.strip().replace('(', '').replace(')', '')

    if index_expr == dim_var:
        return 0

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
                return None

    return offset


def extract_accesses(node, reads=None, writes=None):
    """Recursively extract all accesses from a node."""
    if reads is None:
        reads = []
    if writes is None:
        writes = []

    if not isinstance(node, dict):
        return reads, writes

    if node.get('type') == 'access':
        access = {
            'index': node.get('index', ''),
            'ref': node.get('reference', '')
        }
        arr, idx = parse_access(access['index'])
        access['array'] = arr
        access['index_expr'] = idx

        if node.get('read', 0):
            reads.append(access)
        if node.get('write', 0):
            writes.append(access)

    for key in ['arguments', 'body', 'expr']:
        if key in node:
            if isinstance(node[key], list):
                for item in node[key]:
                    extract_accesses(item, reads, writes)
            else:
                extract_accesses(node[key], reads, writes)

    return reads, writes


def detect_compound_assignment(stmt):
    """
    Detect compound assignments like a[i] += expr or a[i] -= expr.

    PET represents these directly with operation: '+=' or operation: '-='

    Returns dict with array, index, operation, and expression info.
    """
    body = stmt.get('body', {})

    def find_compound(node):
        if not isinstance(node, dict):
            return None

        if node.get('type') == 'op':
            op = node.get('operation', '')
            args = node.get('arguments', [])

            # PET directly represents += and -= operations
            if op in ['+=', '-='] and len(args) == 2:
                lhs = args[0]
                rhs = args[1]

                if lhs.get('type') == 'access':
                    lhs_idx = lhs.get('index', '')
                    lhs_arr, _ = parse_access(lhs_idx)
                    return {
                        'array': lhs_arr,
                        'index': lhs_idx,
                        'operation': op,
                        'expr_node': rhs
                    }

        # Check inside expression wrapper
        if node.get('type') == 'expression' and 'expr' in node:
            return find_compound(node['expr'])

        # Recurse
        for key in ['body', 'expr']:
            if key in node:
                result = find_compound(node[key])
                if result:
                    return result

        return None

    return find_compound(body)


def detect_self_power_recurrence(stmt, dim_var):
    """
    Detect pattern: arr[i] = arr[i-1] * arr[i-1] (self-squaring)
    or more generally: arr[i] = arr[i-1]^k

    Returns dict with array, power, and pattern info.
    """
    reads, writes = extract_accesses(stmt.get('body', {}))

    if len(writes) != 1:
        return None

    write = writes[0]
    write_arr = write['array']
    write_idx = write['index_expr']

    if not write_arr or not write_idx:
        return None

    # Check if write index is just [i] (offset 0)
    write_offset = extract_index_offset(write_idx, dim_var)
    if write_offset != 0:
        return None

    # Check if all reads are from same array with offset -1
    same_array_reads = [r for r in reads if r['array'] == write_arr]

    if len(same_array_reads) >= 2:
        # Check if all reads have offset -1 (reading from previous iteration)
        offsets = [extract_index_offset(r['index_expr'], dim_var) for r in same_array_reads]

        if all(o == -1 for o in offsets):
            # This is arr[i] = arr[i-1] * arr[i-1] * ... (power = count of reads)
            return {
                'array': write_arr,
                'power': len(same_array_reads),
                'pattern': 'self_power'
            }

    return None


def detect_strided_prefix_sum(stmt, dim_var):
    """
    Detect pattern: arr[i] = arr[i-k] + other_arr[i] (strided prefix sum)

    Example s1221: b[i] = b[i-4] + a[i]

    This is k independent prefix sums that can be parallelized:
    - Stream 0: indices 0, k, 2k, 3k, ...
    - Stream 1: indices 1, k+1, 2k+1, ...
    - etc.

    Returns dict with array, stride, pattern info, and verification of safety.
    """
    reads, writes = extract_accesses(stmt.get('body', {}))

    if len(writes) != 1:
        return None

    write = writes[0]
    write_arr = write['array']
    write_idx = write['index_expr']

    if not write_arr or not write_idx:
        return None

    # Check if write index is [i] (offset 0)
    write_offset = extract_index_offset(write_idx, dim_var)
    if write_offset != 0:
        return None

    # Check for exactly one same-array read with negative offset
    same_array_reads = [r for r in reads if r['array'] == write_arr]

    if len(same_array_reads) == 1:
        read = same_array_reads[0]
        read_offset = extract_index_offset(read['index_expr'], dim_var)

        if read_offset is not None and read_offset < 0:
            stride = -read_offset  # Convert to positive stride

            # Check for other array reads (the addend)
            other_reads = [r for r in reads if r['array'] != write_arr]

            if other_reads:
                # Verify safety: analyze the transformed representation
                # After stream decomposition:
                # - Stream s writes to indices: s+stride, s+2*stride, s+3*stride, ...
                # - Stream s reads arr[s] once as initial value (not in loop)
                # - Stream s reads other_arr at same indices (no dependency)
                #
                # Per-stream analysis:
                # - Write indices: {s+k*stride | k >= 1}
                # - Read index for initial: {s}
                # - No overlap: s not in {s+k*stride} for k >= 1 (since stride > 0)
                # Therefore: no same-array loop-carried RAW within each stream

                per_stream_analysis = {
                    'write_offset_in_stream': 0,  # Each stream writes at its own indices
                    'read_offset_in_stream': -stride,  # Reads from stride positions back
                    'initial_value_read': True,  # Reads arr[s] once as initial
                    'no_loop_carried_raw': True,  # Stream indices don't overlap
                    'reason': f"Stream s reads {write_arr}[s] once (initial), writes to {write_arr}[s+{stride}], {write_arr}[s+{2*stride}], ... (no overlap)"
                }

                return {
                    'array': write_arr,
                    'stride': stride,
                    'other_arrays': [r['array'] for r in other_reads],
                    'pattern': 'strided_prefix_sum',
                    'verified_safe': True,
                    'per_stream_analysis': per_stream_analysis
                }

    return None


def analyze_loop_distribution(statements_data):
    """
    Analyze statements for loop distribution opportunities.

    Returns:
        dict with:
        - 'applicable': bool
        - 'statements': list of analyzed statements
        - 'canceling_pairs': list of statement pairs that cancel out
        - 'power_recurrences': list of power recurrence patterns
        - 'parallelizable': list of parallelizable statement indices
        - 'sequential': list of sequential statement indices
        - 'distribution_advice': str
    """
    result = {
        'applicable': False,
        'statements': [],
        'canceling_pairs': [],
        'power_recurrences': [],
        'strided_prefix_sums': [],
        'parallelizable': [],
        'sequential': [],
        'distribution_advice': None
    }

    # For strided prefix sum, we only need 1 statement
    if len(statements_data) < 1:
        return result

    # Get loop dimension variable
    first_domain = statements_data[0].get('domain', '')
    dim_match = re.search(r'S_\d+\[(\w+)', first_domain)
    dim_var = dim_match.group(1) if dim_match else 'i'

    # Analyze each statement
    stmt_info_list = []
    compound_assignments = {}  # array -> list of (stmt_idx, compound_info)

    for idx, stmt in enumerate(statements_data):
        reads, writes = extract_accesses(stmt.get('body', {}))

        # Check for compound assignment
        compound = detect_compound_assignment(stmt)
        if compound:
            arr = compound['array']
            if arr not in compound_assignments:
                compound_assignments[arr] = []
            compound_assignments[arr].append((idx, compound))

        # Check for power recurrence
        power_rec = detect_self_power_recurrence(stmt, dim_var)

        # Check for strided prefix sum
        strided_ps = detect_strided_prefix_sum(stmt, dim_var)

        # Determine if statement has loop-carried dependency
        has_loop_carried = False
        for r in reads:
            for w in writes:
                if r['array'] == w['array']:
                    r_offset = extract_index_offset(r['index_expr'], dim_var)
                    w_offset = extract_index_offset(w['index_expr'], dim_var)
                    if r_offset is not None and w_offset is not None:
                        if r_offset < w_offset:
                            has_loop_carried = True
                            break

        stmt_info = {
            'index': idx,
            'reads': reads,
            'writes': writes,
            'compound': compound,
            'power_recurrence': power_rec,
            'strided_prefix_sum': strided_ps,
            'has_loop_carried': has_loop_carried or (power_rec is not None) or (strided_ps is not None)
        }
        stmt_info_list.append(stmt_info)

        if power_rec:
            power_rec['stmt_idx'] = idx
            result['power_recurrences'].append(power_rec)

        if strided_ps:
            strided_ps['stmt_idx'] = idx
            result['strided_prefix_sums'].append(strided_ps)

    result['statements'] = stmt_info_list

    # Detect canceling pairs (a += x followed by a -= x, or vice versa)
    def normalize_index(index_str):
        """Extract just the array access part: '{ S_0[i] -> a[(i)] }' -> 'a[(i)]'"""
        match = re.search(r'->\s*(\w+\[[^\]]*\])', index_str)
        return match.group(1) if match else index_str

    def exprs_equivalent(expr1, expr2):
        """Check if two expression nodes are structurally equivalent."""
        if type(expr1) != type(expr2):
            return False
        if not isinstance(expr1, dict):
            return expr1 == expr2

        if expr1.get('type') != expr2.get('type'):
            return False

        if expr1.get('type') == 'access':
            # Compare normalized indices
            return normalize_index(expr1.get('index', '')) == normalize_index(expr2.get('index', ''))

        if expr1.get('type') == 'op':
            if expr1.get('operation') != expr2.get('operation'):
                return False
            args1 = expr1.get('arguments', [])
            args2 = expr2.get('arguments', [])
            if len(args1) != len(args2):
                return False
            return all(exprs_equivalent(a1, a2) for a1, a2 in zip(args1, args2))

        return True

    for arr, assignments in compound_assignments.items():
        plus_eqs = [(idx, info) for idx, info in assignments if info['operation'] == '+=']
        minus_eqs = [(idx, info) for idx, info in assignments if info['operation'] == '-=']

        for plus_idx, plus_info in plus_eqs:
            for minus_idx, minus_info in minus_eqs:
                # Check if they operate on same index (normalized)
                plus_norm = normalize_index(plus_info['index'])
                minus_norm = normalize_index(minus_info['index'])

                if plus_norm == minus_norm:
                    # Check if expressions are equivalent
                    if exprs_equivalent(plus_info['expr_node'], minus_info['expr_node']):
                        result['canceling_pairs'].append({
                            'array': arr,
                            'plus_stmt': plus_idx,
                            'minus_stmt': minus_idx,
                            'index': plus_norm
                        })

    # Classify statements as parallelizable or sequential
    canceled_stmts = set()
    for pair in result['canceling_pairs']:
        canceled_stmts.add(pair['plus_stmt'])
        canceled_stmts.add(pair['minus_stmt'])

    for stmt_info in stmt_info_list:
        idx = stmt_info['index']
        if idx in canceled_stmts:
            continue  # Skip canceled statements

        if stmt_info['has_loop_carried']:
            if stmt_info['power_recurrence']:
                # Power recurrence can be parallelized with math transform
                result['parallelizable'].append(idx)
            elif stmt_info.get('strided_prefix_sum'):
                # Strided prefix sum can be parallelized with stream decomposition
                result['parallelizable'].append(idx)
            else:
                result['sequential'].append(idx)
        else:
            result['parallelizable'].append(idx)

    # Check for multi-statement distribution opportunity
    # This is when statements have WAR/RAW dependencies but can be distributed
    # into separate loops that are each parallelizable
    result['distributable_loops'] = detect_distributable_statements(stmt_info_list, dim_var)

    # Determine if loop distribution is applicable
    # Loop distribution provides benefit when:
    # 1. There are canceling pairs (dead code elimination)
    # 2. There are power recurrences (math transform enables parallelization)
    # 3. There are strided prefix sums (stream decomposition)
    # 4. Mix of parallelizable AND sequential statements (run parallel ones in parallel)
    #
    # Loop distribution does NOT help when:
    # - All statements are already individually parallelizable (no benefit from distributing)
    # - Cross-statement dependencies require WAR handling, not distribution
    if result['canceling_pairs'] or result['power_recurrences'] or result['strided_prefix_sums'] or (
        len(result['parallelizable']) > 0 and len(result['sequential']) > 0):
        result['applicable'] = True

    return result


def detect_distributable_statements(stmt_info_list, dim_var):
    """
    Detect if statements can be distributed into separate parallel loops.

    This handles patterns like s3251:
        S0: a[i+1] = b[i] + c[i]  (reads b, writes a)
        S1: b[i] = c[i] * e[i]   (writes b)
        S2: d[i] = a[i] * e[i]   (reads a)

    Where the original loop is sequential due to cross-iteration RAW dependency
    (S0 writes a[i+1], S2 reads a[i]), but distributing into separate loops
    makes each loop parallelizable.

    Returns list of distributable loop info with per-statement dependency analysis, or None.
    """
    if len(stmt_info_list) < 2:
        return None

    # Collect all arrays written and read by each statement
    stmt_arrays = []
    for stmt in stmt_info_list:
        written = set()
        read = set()
        for w in stmt['writes']:
            if w['array']:
                written.add(w['array'])
        for r in stmt['reads']:
            if r['array']:
                read.add(r['array'])
        stmt_arrays.append({
            'index': stmt['index'],
            'writes': written,
            'reads': read,
            'all_writes': stmt['writes'],
            'all_reads': stmt['reads']
        })

    # Check for cross-statement dependencies that require ordering
    # WAR: Statement A reads X, later Statement B writes X
    # RAW across iterations: Statement A writes X[i+k], Statement B reads X[i]

    # Build dependency graph
    dependencies = []  # (from_stmt, to_stmt, type, array)

    for i, si in enumerate(stmt_arrays):
        for j, sj in enumerate(stmt_arrays):
            if i == j:
                continue

            # Check WAR: si reads, sj writes (same array, same iteration)
            for arr in si['reads'] & sj['writes']:
                if i < j:  # si before sj in original order
                    dependencies.append((i, j, 'WAR', arr))

            # Check RAW across iterations: si writes arr[i+k], sj reads arr[i]
            for wi in si['all_writes']:
                for rj in sj['all_reads']:
                    if wi['array'] == rj['array']:
                        w_offset = extract_index_offset(wi['index_expr'], dim_var)
                        r_offset = extract_index_offset(rj['index_expr'], dim_var)
                        if w_offset is not None and r_offset is not None:
                            if w_offset > r_offset:
                                # si writes arr[i+k], sj reads arr[i] (k>0)
                                # This creates cross-iteration RAW dependency
                                dependencies.append((i, j, 'RAW_cross', wi['array']))

    # Check if we can distribute: each statement in its own loop should be parallelizable
    # A distributed loop is parallelizable if:
    # 1. It has no self-loop-carried dependency (already checked in has_loop_carried)
    # 2. When run separately with proper ordering, dependencies are satisfied

    distributable = []

    for stmt in stmt_info_list:
        # For each statement, analyze same-array read/write pairs
        # This is the key analysis: does this statement have a loop-carried RAW on same array?
        same_array_pairs = []
        is_self_parallelizable = True
        blocking_pair = None

        for r in stmt['reads']:
            for w in stmt['writes']:
                if r['array'] == w['array']:
                    r_offset = extract_index_offset(r['index_expr'], dim_var)
                    w_offset = extract_index_offset(w['index_expr'], dim_var)

                    same_array_pairs.append({
                        'array': r['array'],
                        'read_offset': r_offset,
                        'write_offset': w_offset
                    })

                    if r_offset is not None and w_offset is not None:
                        if r_offset < w_offset:
                            # Reading from previous iteration - NOT self-parallelizable
                            is_self_parallelizable = False
                            blocking_pair = {
                                'array': r['array'],
                                'read_offset': r_offset,
                                'write_offset': w_offset,
                                'reason': f"reads {r['array']}[i+{r_offset}], writes {r['array']}[i+{w_offset}]"
                            }

        distributable.append({
            'stmt_index': stmt['index'],
            'writes': list(stmt_arrays[stmt['index']]['writes']),
            'reads': list(stmt_arrays[stmt['index']]['reads']),
            'parallelizable': is_self_parallelizable,
            'same_array_pairs': same_array_pairs,
            'blocking_pair': blocking_pair
        })

    # If all statements are distributable and parallelizable, and there are
    # cross-statement dependencies, loop distribution MAY help
    all_parallelizable = all(d['parallelizable'] for d in distributable)
    if all_parallelizable and len(distributable) >= 2:
        # Check dependency types
        # WAR: Safe for distribution - read before write in same iteration
        # RAW_cross: NOT safe - cross-iteration RAW means true loop-carried dependency
        #            Example s323: S1 writes b[i], S0 reads b[i-1] in next iteration
        #            Distributing breaks this chain!
        has_war = any(d[2] == 'WAR' for d in dependencies)
        has_raw_cross = any(d[2] == 'RAW_cross' for d in dependencies)

        if has_war and not has_raw_cross:
            # Only WAR dependencies - safe to distribute
            for d in distributable:
                d['verified_safe'] = True
            return distributable

    return None


def format_loop_distribution_for_prompt(analysis_result):
    """Format loop distribution analysis for inclusion in LLM prompt."""
    if not analysis_result or not analysis_result.get('applicable'):
        return None

    lines = []
    lines.append("=" * 60)
    lines.append("LOOP DISTRIBUTION OPPORTUNITY DETECTED")
    lines.append("=" * 60)
    lines.append("")

    # Canceling operations
    if analysis_result['canceling_pairs']:
        lines.append("CANCELING OPERATIONS DETECTED (Dead Code):")
        lines.append("")
        for pair in analysis_result['canceling_pairs']:
            arr = pair['array']
            lines.append(f"  Statement S{pair['plus_stmt']}: {arr}[i] += expr")
            lines.append(f"  Statement S{pair['minus_stmt']}: {arr}[i] -= expr")
            lines.append(f"  -> These CANCEL OUT! Array '{arr}' is UNCHANGED.")
            lines.append("")
        lines.append("CRITICAL: Do NOT compute these operations. Skip them entirely!")
        lines.append("")

    # Power recurrences
    if analysis_result['power_recurrences']:
        lines.append("POWER RECURRENCE DETECTED:")
        lines.append("")
        for rec in analysis_result['power_recurrences']:
            arr = rec['array']
            power = rec['power']
            lines.append(f"  Statement S{rec['stmt_idx']}: {arr}[i] = {arr}[i-1]^{power}")
            lines.append("")
            lines.append("  This APPEARS sequential but CAN BE PARALLELIZED!")
            lines.append("")
            lines.append("  Mathematical transformation:")
            lines.append(f"    Original:  {arr}[i] = {arr}[i-1]^{power}  (sequential dependency)")
            lines.append(f"    Transform: {arr}[i] = {arr}[0]^({power}^i)  (parallel - no dependency!)")
            lines.append("")
            lines.append("  Implementation:")
            lines.append("  ```python")
            lines.append(f"  def compute_{arr}_parallel({arr}):")
            lines.append(f"      n = {arr}.shape[0]")
            lines.append(f"      {arr}0 = {arr}[0].clone()  # Save initial value")
            lines.append(f"      # Compute exponents: {power}^0, {power}^1, {power}^2, ..., {power}^(n-1)")
            lines.append(f"      exponents = torch.pow({float(power)}, torch.arange(n, device={arr}.device, dtype={arr}.dtype))")
            lines.append(f"      # Parallel computation: {arr}[i] = {arr}0^(exponent[i])")
            lines.append(f"      {arr}[:] = torch.pow({arr}0, exponents)")
            lines.append("  ```")
            lines.append("")
            lines.append("  This transforms O(n) sequential operations into O(1) parallel operations!")
            lines.append("")

    # Strided prefix sum
    if analysis_result.get('strided_prefix_sums'):
        lines.append("STRIDED PREFIX SUM DETECTED:")
        lines.append("")
        for sps in analysis_result['strided_prefix_sums']:
            arr = sps['array']
            stride = sps['stride']
            other = ', '.join(sps.get('other_arrays', []))
            lines.append(f"  Statement S{sps['stmt_idx']}: {arr}[i] = {arr}[i-{stride}] + ...")
            lines.append("")
            lines.append("  This APPEARS sequential but CAN BE PARALLELIZED!")
            lines.append("")
            lines.append(f"  Key insight: This is {stride} INDEPENDENT prefix sums!")
            lines.append("")
            for s in range(stride):
                lines.append(f"    Stream {s}: indices {s}, {s+stride}, {s+2*stride}, ... (i % {stride} == {s})")
            lines.append("")
            lines.append("  Each stream can be computed as a parallel prefix sum using `torch.cumsum()`.")
            lines.append("")
            lines.append("  Implementation:")
            lines.append("  ```python")
            lines.append(f"  def compute_{arr}_parallel({arr}, {other if other else 'addend'}):")
            lines.append(f"      n = {arr}.shape[0]")
            lines.append(f"      stride = {stride}")
            lines.append(f"      start_idx = stride  # Loop starts at i={stride}")
            lines.append(f"      ")
            lines.append(f"      # Process each stream independently")
            lines.append(f"      for stream in range(stride):")
            lines.append(f"          # Extract indices for this stream: stream, stream+stride, stream+2*stride, ...")
            lines.append(f"          stream_indices = torch.arange(stream + stride, n, stride, device={arr}.device)")
            lines.append(f"          if len(stream_indices) == 0:")
            lines.append(f"              continue")
            lines.append(f"          ")
            lines.append(f"          # Extract the addends for this stream")
            lines.append(f"          addend_vals = {other if other else 'addend'}[stream_indices]")
            lines.append(f"          ")
            lines.append(f"          # Compute prefix sum of addends")
            lines.append(f"          prefix_sums = torch.cumsum(addend_vals, dim=0)")
            lines.append(f"          ")
            lines.append(f"          # Add the initial value {arr}[stream]")
            lines.append(f"          {arr}[stream_indices] = {arr}[stream] + prefix_sums")
            lines.append("  ```")
            lines.append("")
            lines.append(f"  This transforms O(n) sequential into {stride} parallel prefix sums!")
            lines.append("")

    # Distributable loops pattern
    distributable = analysis_result.get('distributable_loops')
    if distributable and len(distributable) > 1:
        lines.append("LOOP DISTRIBUTION FOR PARALLELIZATION:")
        lines.append("")
        lines.append("This loop can be SPLIT into separate parallel loops!")
        lines.append("")
        lines.append("Original loop has cross-statement dependencies, but when distributed:")
        lines.append("")
        for i, d in enumerate(distributable):
            writes = ', '.join(d['writes']) if d['writes'] else 'none'
            reads = ', '.join(d['reads']) if d['reads'] else 'none'
            lines.append(f"  Loop {i+1} (Statement S{d['stmt_index']}): writes {writes}, reads {reads}")
            # Show verification of parallelizability
            if d.get('same_array_pairs'):
                for pair in d['same_array_pairs']:
                    if pair['read_offset'] is not None and pair['write_offset'] is not None:
                        if pair['read_offset'] >= pair['write_offset']:
                            lines.append(f"    ✓ {pair['array']}: read offset {pair['read_offset']} >= write offset {pair['write_offset']} (no RAW)")
            if d.get('verified_safe'):
                lines.append(f"    -> VERIFIED PARALLELIZABLE (no same-array loop-carried RAW)")
        lines.append("")
        lines.append("IMPLEMENTATION STRATEGY:")
        lines.append("Split into separate Triton kernels, each processing in parallel:")
        lines.append("")
        lines.append("```python")
        lines.append("@triton.jit")
        lines.append("def kernel_loop1(...):")
        lines.append("    pid = tl.program_id(0)")
        lines.append("    # Parallel: each thread handles different indices")
        lines.append("    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)")
        lines.append("    # Process statement S0 for all indices in parallel")
        lines.append("    ...")
        lines.append("")
        lines.append("@triton.jit")
        lines.append("def kernel_loop2(...):")
        lines.append("    # Similar parallel structure for statement S1")
        lines.append("    ...")
        lines.append("")
        lines.append("def wrapper(...):")
        lines.append("    # Run loops sequentially, but each loop is internally parallel")
        lines.append("    kernel_loop1[grid](...)")
        lines.append("    kernel_loop2[grid](...)")
        lines.append("    # etc.")
        lines.append("```")
        lines.append("")
        lines.append("KEY: Each distributed loop processes ALL iterations in parallel,")
        lines.append("     then the next loop runs. This satisfies cross-statement dependencies.")
        lines.append("")

    # Summary
    if analysis_result['canceling_pairs'] and analysis_result['power_recurrences']:
        lines.append("=" * 60)
        lines.append("COMPLETE OPTIMIZATION STRATEGY:")
        lines.append("=" * 60)
        lines.append("")
        lines.append("1. SKIP the canceling operations entirely (they produce no effect)")
        lines.append("2. Use parallel power computation for the recurrence")
        lines.append("")
        lines.append("Final implementation should be:")
        lines.append("```python")
        lines.append("def kernel_triton(a, b, c, e):  # example for s222-like pattern")
        lines.append("    # a operations cancel out - DO NOTHING with a")
        lines.append("    # Only compute e using parallel power formula")
        lines.append("    n = e.shape[0]")
        lines.append("    e0 = e[0].clone()")
        lines.append("    exponents = torch.pow(2.0, torch.arange(n, device=e.device, dtype=e.dtype))")
        lines.append("    e[:] = torch.pow(e0, exponents)")
        lines.append("```")
        lines.append("")

    lines.append("=" * 60)

    return '\n'.join(lines)


def analyze_kernel_loop_distribution(kernel_name):
    """
    Analyze a kernel for loop distribution opportunities.

    Args:
        kernel_name: Name of the kernel (e.g., 's222')

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
    if len(statements) < 1:
        return None

    result = analyze_loop_distribution(statements)

    if result['applicable']:
        return result
    return None


def main():
    """Test loop distribution detection."""
    test_kernels = ['s222', 's231', 's232']

    print("=" * 80)
    print("LOOP DISTRIBUTION ANALYSIS")
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

        result = analyze_kernel_loop_distribution(kernel)
        if result:
            print(f"\nApplicable: {result['applicable']}")
            print(f"Canceling pairs: {len(result['canceling_pairs'])}")
            print(f"Power recurrences: {len(result['power_recurrences'])}")
            print(f"Parallelizable stmts: {result['parallelizable']}")
            print(f"Sequential stmts: {result['sequential']}")
            print()
            formatted = format_loop_distribution_for_prompt(result)
            if formatted:
                print(formatted)
        else:
            print("\nNo loop distribution opportunity detected")


if __name__ == "__main__":
    main()
