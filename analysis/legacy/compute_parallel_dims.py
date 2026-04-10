#!/usr/bin/env python3
"""
Analyze loop parallelism dimensions for multi-dimensional loops.

For nested loops, determines:
1. Which loop levels have loop-carried dependencies (must be sequential)
2. Which loop levels are fully parallel
3. Recommended Triton parallelization strategy
4. Whether in-kernel loop is safe or requires multiple kernel launches

KEY INSIGHT: For 2D triangular loops (like s115, s118), EITHER dimension can be
made sequential while the other is parallel. The choice affects parallelism type:
- Option A: dim0-sequential, dim1-parallel (may be independent writes OR reduction)
- Option B: dim1-sequential, dim0-parallel (may be independent writes OR reduction)

CRITICAL DISTINCTION FOR TRITON CODE GENERATION:

Pattern A: Row-Shift in 2D Arrays (s1119, s126) - SAFE for SINGLE_KERNEL_INLOOP
  - Write: arr[seq][par], Read: arr[seq-c][par]
  - The parallel dimension index is IDENTICAL in both read and write
  - Each thread reads from memory that THE SAME THREAD wrote in previous iteration
  - GPU guarantees program order for same thread's reads/writes
  - Can use: Single kernel with in-kernel for-loop over sequential dimension

Pattern B: Diagonal-Shift in 2D Arrays (s119) - REQUIRES MULTI_KERNEL_LAUNCH
  - Write: arr[seq][par], Read: arr[seq-c][par-c]
  - BOTH indices shift by constants (diagonal dependency)
  - Thread par reads from memory that thread (par-1) wrote in previous iteration
  - This is a CROSS-THREAD dependency!
  - GPU does NOT guarantee thread par sees thread (par-1)'s writes without explicit sync
  - Must use: Multiple kernel launches (one per sequential iteration)

Pattern C: Cross-Block Dependency (s115, s118) - REQUIRES MULTI_KERNEL_LAUNCH
  - Read involves values that may be written by OTHER parallel blocks
  - Example: s115 reads a[j] which was written when j was i in earlier iteration
  - Example: s118 reads a[i-j-1] from earlier i iterations across blocks
  - Must use: Multiple kernel launches (one per sequential iteration)

This script analyzes BOTH orderings and determines the correct Triton implementation.
"""

import subprocess
import yaml
import re
import os
import json
from collections import defaultdict

# Try to import islpy, but make it optional for basic analysis
try:
    import islpy as isl
    HAS_ISL = True
except ImportError:
    HAS_ISL = False
    print("Warning: islpy not available, using basic analysis only")

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


def get_loop_dimensions(domain_str):
    """Extract loop dimension names from domain string."""
    # Parse: { S_0[j, i] : constraints }
    match = re.search(r'S_\d+\[([^\]]+)\]', domain_str)
    if match:
        dims = [d.strip() for d in match.group(1).split(',')]
        return dims
    return []


def parse_access_index(access_str):
    """Parse ISL access string to extract array name and index expression."""
    match = re.search(r'->\s*(\w+)\[(.*?)\]', access_str)
    if match:
        return {
            'array': match.group(1),
            'index_expr': match.group(2),
            'full': access_str
        }
    return None


def extract_dims_in_expr(expr, dims):
    """Find which loop dimensions appear in an index expression."""
    found = set()
    for dim in dims:
        if re.search(rf'\b{dim}\b', expr):
            found.add(dim)
    return found


def is_actual_reduction(c_code, write_array):
    """
    Check if writes to an array are actual reductions (+=, *=, etc.) or just overwrites (=).

    Args:
        c_code: C source code
        write_array: Name of the array being written

    Returns:
        bool: True if it's a reduction (uses +=, *=, etc.), False if it's just assignment (=)
    """
    # Normalize code
    code_normalized = c_code.replace('\n', ' ').replace('\t', ' ')

    # Check for reduction operators
    reduction_patterns = [
        rf'{write_array}\s*\[[^\]]+\]\s*\+=',  # sum reduction: arr[i] += ...
        rf'{write_array}\s*\[[^\]]+\]\s*\*=',  # product reduction: arr[i] *= ...
        rf'{write_array}\s*\[[^\]]+\]\s*\|=',  # bitwise or
        rf'{write_array}\s*\[[^\]]+\]\s*&=',   # bitwise and
        rf'{write_array}\s*\[[^\]]+\]\s*\^=',  # bitwise xor
    ]

    for pattern in reduction_patterns:
        if re.search(pattern, code_normalized):
            return True

    # Also check for conditional updates that look like max/min reductions
    # Pattern: if (expr > arr[i]) arr[i] = expr
    max_min_pattern = rf'if\s*\([^)]*[<>][^)]*{write_array}\s*\[[^\]]+\]\s*\)\s*{{?\s*{write_array}\s*\[[^\]]+\]\s*='
    if re.search(max_min_pattern, code_normalized):
        return True

    return False


def get_index_parts(index_expr):
    """
    Split a multi-dimensional index expression into parts.
    E.g., "(i), (j)" -> ["i", "j"] or "(-1 + i), (j)" -> ["-1 + i", "j"]
    """
    parts = []
    depth = 0
    current = []
    for char in index_expr:
        if char == '(':
            depth += 1
            current.append(char)
        elif char == ')':
            depth -= 1
            current.append(char)
        elif char == ',' and depth == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        parts.append(''.join(current).strip())
    # Clean up parentheses
    parts = [p.strip('() ') for p in parts]
    return parts


def is_safe_for_inkernel_loop(write_expr, read_expr, dims, seq_dim, par_dim, array_is_1d, triangular_info=None):
    """
    Determine if this read/write pattern is safe for single-kernel in-kernel loop.

    SAFE patterns (can use SINGLE_KERNEL_INLOOP):
    1. Row-shift in 2D array: write arr[seq][par], read arr[seq-c][par]
       - par_dim index is IDENTICAL -> different blocks access disjoint slices
    2. Diagonal-shift in 2D array: write arr[seq][par], read arr[seq-c][par-c]
       - Both indices shift by constants -> no cross-block conflicts
    3. Same location: read == write (same-iteration dependency)
    4. Broadcast read: read doesn't use par_dim (all blocks read same value)
    5. Transpose access with triangular bounds: write arr[i][j], read arr[j][i] with j < i
       - Write region (lower triangle) and read region (upper triangle) are DISJOINT

    UNSAFE patterns (require MULTI_KERNEL_LAUNCH):
    1. 1D array with cross-iteration dependency where different blocks need
       values written by other blocks in earlier sequential iterations
    2. Read index uses par_dim differently than write (cross-block dependency)

    Returns: (is_safe: bool, reason: str, pattern_type: str)
    """
    write_dims = extract_dims_in_expr(write_expr, dims)
    read_dims = extract_dims_in_expr(read_expr, dims)

    # Same location - safe only if subscript contains loop dims (not a scalar)
    if read_expr == write_expr:
        expr_dims = extract_dims_in_expr(write_expr, dims)
        if not expr_dims:
            return False, "Scalar dependency: loop-carried scalar", "scalar_dep"
        return True, "Same location read/write", "same_location"

    # Check for transpose access pattern with triangular bounds
    # Pattern: write arr[i][j], read arr[j][i] with constraint j < i (or i < j)
    # In this case, write region (lower/upper triangle) is disjoint from read region (upper/lower triangle)
    if len(dims) == 2 and triangular_info and not array_is_1d:
        write_parts = get_index_parts(write_expr)
        read_parts = get_index_parts(read_expr)

        if len(write_parts) == 2 and len(read_parts) == 2:
            w0 = re.sub(r'[()]', '', write_parts[0]).strip()
            w1 = re.sub(r'[()]', '', write_parts[1]).strip()
            r0 = re.sub(r'[()]', '', read_parts[0]).strip()
            r1 = re.sub(r'[()]', '', read_parts[1]).strip()

            # Check if indices are swapped: write[i][j] and read[j][i]
            if w0 == r1 and w1 == r0 and w0 != w1:
                # Indices are swapped - this is a transpose access
                # With triangular bounds (j < i or i < j), the regions are DISJOINT
                # Lower triangle (row > col) and upper triangle (row < col) never overlap
                return True, f"Transpose access with triangular bounds: write [{write_expr}] and read [{read_expr}] are in disjoint triangle regions", "transpose_disjoint"

    # Broadcast read - par_dim not in read index
    if par_dim not in read_dims:
        # Write-conflict: par_dim absent from both read AND write
        if par_dim not in write_dims:
            return False, f"Write conflict: all {par_dim}-threads write same [{write_expr}]", "write_conflict"
        # For 1D arrays, reads indexed by seq_dim access values written in
        # earlier sequential iterations.  With triangular bounds where
        # par_dim > seq_dim (e.g. i > j), the write location a[par_dim] is
        # ALWAYS different from the read location a[seq_dim] within the same
        # sequential iteration.  Therefore, with an in-kernel loop all blocks
        # process seq_dim iterations in order and a[seq_dim] was finalized in
        # a prior iteration — this is SAFE.
        if array_is_1d and seq_dim in read_dims:
            if triangular_info:
                # Triangular bounds guarantee writes don't alias the read
                # location in the same iteration (par > seq or par < seq).
                return True, (f"1D broadcast read [{read_expr}]: triangular bounds ensure "
                              f"a[{seq_dim}] is finalized before it is read"), "broadcast_triangular"
            return False, f"1D array cross-block: read [{read_expr}] accesses values written by other blocks", "cross_block_1d"
        return True, f"Broadcast read: [{read_expr}] doesn't use {par_dim}", "broadcast"

    # par_dim is in read - check if it's identical to write's par_dim index
    write_parts = get_index_parts(write_expr)
    read_parts = get_index_parts(read_expr)

    if len(write_parts) == 2 and len(read_parts) == 2:
        # 2D array - check for row-shift or diagonal-shift

        # Find which part contains par_dim in write
        w_par_idx = -1
        for i, part in enumerate(write_parts):
            if re.search(rf'\b{par_dim}\b', part):
                w_par_idx = i
                break

        if w_par_idx >= 0:
            # Check if par_dim part is IDENTICAL in read and write
            if read_parts[w_par_idx] == write_parts[w_par_idx]:
                # par_dim index is identical - ROW SHIFT pattern - SAFE!
                return True, f"Row-shift: {par_dim} index identical in read/write", "row_shift"

            # Check for diagonal shift (BOTH indices must actually shift by constants)
            def is_actual_shift(r_idx, w_idx):
                """Check if r_idx = w_idx +/- constant (a TRUE shift, not identical)."""
                r_clean = re.sub(r'[()]', '', r_idx).strip()
                w_clean = re.sub(r'[()]', '', w_idx).strip()
                # Identical indices are NOT a shift - return False
                if r_clean == w_clean:
                    return False
                for dim in dims:
                    if dim in r_clean and dim in w_clean:
                        if re.match(rf'^-?\d+\s*\+\s*{dim}$', r_clean):
                            return True
                        if re.match(rf'^{dim}\s*-\s*\d+$', r_clean):
                            return True
                return False

            idx0_shifts = is_actual_shift(read_parts[0], write_parts[0])
            idx1_shifts = is_actual_shift(read_parts[1], write_parts[1])

            # True diagonal shift: BOTH indices actually shift (not one identical, one shifts)
            # Example: aa[i][j] = aa[i-1][j-1] is diagonal shift
            #
            # CRITICAL: Diagonal shift is NOT SAFE for in-kernel loop!
            # Reason: When i is sequential and j is parallel:
            #   - Thread j writes aa[i][j]
            #   - Thread j reads aa[i-1][j-1] which was written by thread (j-1) in iteration i-1
            #   - This is a CROSS-THREAD dependency!
            #   - GPU does NOT guarantee thread j sees thread (j-1)'s writes without explicit sync
            #
            # This is different from row-shift aa[i][j] = aa[i-1][j] where:
            #   - Thread j reads aa[i-1][j] which was written by the SAME thread j
            #   - GPU guarantees a thread sees its own previous writes (program order)
            #
            # Therefore: diagonal shift requires MULTI_KERNEL_LAUNCH
            if idx0_shifts and idx1_shifts:
                return False, f"Diagonal-shift: read [{read_expr}] involves cross-thread dependency (thread {par_dim} reads from thread {par_dim}-1)", "diagonal_shift_cross_thread"

            # If only one index shifts and the other is identical, this is a CHAIN DEPENDENCY
            # along the shifting dimension - NOT safe for in-kernel loop when par_dim is the shifting one
            idx0_identical = (re.sub(r'[()]', '', read_parts[0]).strip() == re.sub(r'[()]', '', write_parts[0]).strip())
            idx1_identical = (re.sub(r'[()]', '', read_parts[1]).strip() == re.sub(r'[()]', '', write_parts[1]).strip())

            # Check if par_dim is in the shifting index - if so, there's a chain dependency
            if (idx0_shifts and idx1_identical and re.search(rf'\b{par_dim}\b', write_parts[0])):
                # par_dim is in the shifting index - chain dependency along par_dim
                return False, f"Chain dependency: {par_dim} index shifts but other index identical - RAW dependency along {par_dim}", "chain_dependency"
            if (idx1_shifts and idx0_identical and re.search(rf'\b{par_dim}\b', write_parts[1])):
                # par_dim is in the shifting index - chain dependency along par_dim
                return False, f"Chain dependency: {par_dim} index shifts but other index identical - RAW dependency along {par_dim}", "chain_dependency"

    # 1D array patterns
    if len(write_parts) == 1 and len(read_parts) == 1:
        # 1D array - need to check for cross-block dependencies carefully
        #
        # Key insight: For 1D arrays, ALL blocks share the same array elements.
        # With in-kernel loop, blocks can be at different sequential iterations.
        # Block 0 at i=k might read a[k-1] while Block 1 at i=k-1 is still writing a[k-1]!
        #
        # This is different from 2D arrays where each block owns a disjoint slice.
        #
        # For s118: a[i] += bb[j][i] * a[i-j-1]
        # - Write a[i] is a reduction (same for all j in same i)
        # - Read a[i-j-1] accesses values from earlier i iterations
        # - BUT: those values are ALSO reductions that need all blocks to complete!
        # - If block 0 is at i=k and block 1 is at i=k-1, block 0 reads incomplete a[k-1]
        #
        # Therefore: 1D arrays with cross-iteration dependencies REQUIRE MULTI_KERNEL_LAUNCH

        if par_dim not in write_dims:
            # Write doesn't use par_dim -> REDUCTION pattern
            # Reduction means multiple parallel blocks contribute to same location
            # We need ALL blocks to complete seq=k before ANY block starts seq=k+1
            # In-kernel loop does NOT guarantee this! -> MULTI_KERNEL_LAUNCH required
            return False, f"1D reduction: a[{write_expr}] needs all blocks to sync per {seq_dim} iteration", "reduction_needs_sync"

        # Write uses par_dim but it's 1D - this shouldn't happen normally
        # But if it does, check if read could access other blocks' writes
        return False, f"1D array cross-block: write [{write_expr}], read [{read_expr}]", "cross_block_1d"

    # Default: UNSAFE - potential cross-block dependency
    return False, f"Cross-block dependency: read [{read_expr}] may conflict with writes from other blocks", "cross_block"


def is_triangular_domain(domain_str, dims):
    """
    Check if domain has triangular bounds (e.g., j < i, i > j).
    Returns (is_triangular, info_dict).
    """
    if len(dims) < 2:
        return False, None

    d0, d1 = dims[0], dims[1]
    # Check for patterns like: d0 < d1, d0 <= d1 - 1, d1 > d0, etc.
    patterns = [
        (rf'\b{d0}\s*<\s*{d1}\b', d0, d1),
        (rf'\b{d0}\s*<=\s*{d1}\s*-', d0, d1),
        (rf'\b{d1}\s*>\s*{d0}\b', d0, d1),
        (rf'\b{d1}\s*>=\s*{d0}\s*\+', d0, d1),
        (rf'\b{d1}\s*<\s*{d0}\b', d1, d0),
        (rf'\b{d1}\s*<=\s*{d0}\s*-', d1, d0),
        (rf'\b{d0}\s*>\s*{d1}\b', d1, d0),
        (rf'\b{d0}\s*>=\s*{d1}\s*\+', d1, d0),
    ]

    for pattern, smaller, larger in patterns:
        if re.search(pattern, domain_str):
            return True, {'smaller_dim': smaller, 'larger_dim': larger}

    return False, None


def analyze_single_ordering(seq_dim, par_dim, dims, domain_str, reads, writes, c_code=""):
    """
    Analyze what happens when seq_dim is sequential and par_dim is parallel.

    Determines:
    - Is this ordering valid (are dependencies satisfied)?
    - What type of parallelism: 'independent' (different locations), 'reduction' (same location with +=/*=), or 'overwrite' (same location with =)?
    - Are all reads from already-computed values when seq_dim is processed in order?

    Returns dict with analysis results.
    """
    result = {
        'sequential_dim': seq_dim,
        'parallel_dim': par_dim,
        'valid': True,
        'parallelism_type': None,  # 'independent', 'reduction', or 'overwrite'
        'write_pattern': None,
        'read_pattern': None,
        'issues': []
    }

    # Analyze write patterns
    writes_use_par_dim = False
    writes_use_seq_dim = False
    write_arrays = {}

    for w in writes:
        parsed = parse_access_index(w['index'])
        if not parsed:
            continue
        dims_used = extract_dims_in_expr(parsed['index_expr'], dims)
        write_arrays[parsed['array']] = {
            'expr': parsed['index_expr'],
            'dims': dims_used
        }
        if par_dim in dims_used:
            writes_use_par_dim = True
        if seq_dim in dims_used:
            writes_use_seq_dim = True

    # Determine parallelism type based on write pattern
    if writes_use_par_dim:
        # Different par_dim iterations write to different locations
        result['parallelism_type'] = 'independent'
        result['write_pattern'] = f"Each {par_dim} iteration writes to different location"
    else:
        # All par_dim iterations write to same location (indexed by seq_dim only or constant)
        # Check if it's an actual reduction (+=, *=) or just overwrite (=)
        is_reduction = False
        for array_name in write_arrays:
            if is_actual_reduction(c_code, array_name):
                is_reduction = True
                break

        if is_reduction:
            result['parallelism_type'] = 'reduction'
            result['write_pattern'] = f"All {par_dim} iterations write to same location (reduction with += or *=)"
        else:
            result['parallelism_type'] = 'overwrite'
            result['write_pattern'] = f"All {par_dim} iterations write to same location (last write wins)"

    # Analyze read patterns - check if reads are from "completed" iterations
    for r in reads:
        parsed = parse_access_index(r['index'])
        if not parsed:
            continue

        # Check if this array is also written (potential dependency)
        if parsed['array'] in write_arrays:
            w_info = write_arrays[parsed['array']]
            r_dims = extract_dims_in_expr(parsed['index_expr'], dims)

            # Key check: when seq_dim is processed sequentially, are read indices
            # guaranteed to be from earlier seq_dim iterations?

            if parsed['index_expr'] == w_info['expr']:
                # Same expression - read and write same location
                # This is OK (same-iteration read-before-write in expression evaluation)
                continue

            # Different expressions - analyze the dependency
            # For the ordering to be valid:
            # - Reads indexed by seq_dim should access seq_dim' < seq_dim (earlier iterations)
            # - OR reads should be indexed by par_dim only (parallel within seq_dim iteration)

            if seq_dim in r_dims:
                # Read uses seq_dim in its index
                # Check if it could access current or future seq_dim iterations
                # This requires more sophisticated analysis
                # For now, assume triangular patterns are OK since they typically
                # read from strictly earlier iterations

                result['read_pattern'] = f"Reads {parsed['array']}[{parsed['index_expr']}] - check bounds"

    return result


def analyze_both_orderings(domain_str, reads, writes):
    """
    Analyze BOTH possible orderings for a 2D loop.

    For triangular loops like s115 and s118, either dimension can be made
    sequential while the other is parallel. This function analyzes both options.

    Returns dict with:
    - dims: [dim0, dim1]
    - is_triangular: bool
    - orderings: list of analysis for each valid ordering
    - self_dependencies: list of same-array read/write patterns
    """
    dims = get_loop_dimensions(domain_str)
    if len(dims) != 2:
        return None

    is_triangular, tri_info = is_triangular_domain(domain_str, dims)

    # Find self-dependencies (same array read and written)
    self_deps = []
    for w in writes:
        w_parsed = parse_access_index(w['index'])
        if not w_parsed:
            continue
        for r in reads:
            r_parsed = parse_access_index(r['index'])
            if not r_parsed:
                continue
            if w_parsed['array'] == r_parsed['array']:
                self_deps.append({
                    'array': w_parsed['array'],
                    'write_expr': w_parsed['index_expr'],
                    'read_expr': r_parsed['index_expr'],
                    'write_dims': extract_dims_in_expr(w_parsed['index_expr'], dims),
                    'read_dims': extract_dims_in_expr(r_parsed['index_expr'], dims)
                })

    result = {
        'dims': dims,
        'domain': domain_str,
        'is_triangular': is_triangular,
        'triangular_info': tri_info,
        'self_dependencies': self_deps,
        'orderings': []
    }

    # Analyze both orderings
    for seq_idx in range(2):
        seq_dim = dims[seq_idx]
        par_dim = dims[1 - seq_idx]

        ordering = analyze_single_ordering(seq_dim, par_dim, dims, domain_str, reads, writes)
        result['orderings'].append(ordering)

    return result


def analyze_dependency_dimensions(domain_str, reads, writes):
    """
    Analyze which loop dimensions carry dependencies.

    Key insight: A dimension carries a dependency if different iterations
    of that dimension access the same memory location (and one writes).

    For s115: for j in ...: for i in ...: a[i] -= aa[j][i] * a[j]
    - Write to a[i] - different i values write to different locations
    - Read from a[j] - this value was written when j was i in earlier j iteration
    - OUTER j carries dependency (must be sequential)
    - INNER i is parallel (different i don't conflict within same j)

    Returns dict with:
    - dims: list of dimension names (outer to inner)
    - dep_dims: set of dimensions that carry dependencies (must be sequential)
    - parallel_dims: set of dimensions that are parallel
    """
    dims = get_loop_dimensions(domain_str)
    if not dims:
        return None

    result = {
        'dims': dims,
        'dep_dims': set(),
        'parallel_dims': set(dims),  # Start assuming all parallel
        'dep_details': [],
        'is_reduction': {}
    }

    try:
        domain = isl.Set(domain_str)
    except:
        return result

    # Analyze each read-write pair for dependencies
    for write in writes:
        w_match = re.search(r'->\s*(\w+)\[(.*?)\]', write['index'])
        if not w_match:
            continue
        w_array = w_match.group(1)
        w_idx_expr = w_match.group(2)

        for read in reads:
            r_match = re.search(r'->\s*(\w+)\[(.*?)\]', read['index'])
            if not r_match:
                continue
            r_array = r_match.group(1)
            r_idx_expr = r_match.group(2)

            if w_array != r_array:
                continue

            # Same array accessed - analyze which dimensions carry the dependency

            detail = {
                'array': w_array,
                'write_idx': w_idx_expr,
                'read_idx': r_idx_expr,
            }
            result['dep_details'].append(detail)

            # Parse which dimensions appear in each index expression
            w_dims_used = set(dim for dim in dims if re.search(rf'\b{dim}\b', w_idx_expr))
            r_dims_used = set(dim for dim in dims if re.search(rf'\b{dim}\b', r_idx_expr))

            # Case 1: Same index expression (e.g., a[i] read and written)
            # This is a reduction - the loop dimension NOT in the index is parallel,
            # but the dimension IN the index may have self-dependency
            if w_idx_expr == r_idx_expr:
                # Check if this is same-iteration (not a loop-carried dependency)
                # e.g., a[i] = a[i] + x is same-iteration, not loop-carried
                # But if multiple iterations write to same location, it's a reduction

                # Find dimensions NOT used in the index - these iterate over same location
                dims_not_in_index = set(dims) - w_dims_used
                for dim in dims_not_in_index:
                    result['is_reduction'][dim] = w_array
                    # Reduction dimension needs atomic or sequential accumulation
                    # But can be parallelized with proper reduction strategy

                # Dimensions IN the index - different iterations access different locations
                # These are parallel for this access pattern
                continue

            # Case 2: Different index expressions
            # e.g., write a[i], read a[j] where i and j are different dims
            # The dependency crosses iterations of the outer dimension

            # Key insight: write uses dimension D, read uses dimension D'
            # If D != D', the dependency is carried by the outer of the two
            if w_dims_used != r_dims_used:
                # Different dimensions used - dependency crosses dimensions
                # The OUTER dimension that differs carries the dependency

                # Find which dimension appears in one but not the other
                only_in_write = w_dims_used - r_dims_used
                only_in_read = r_dims_used - w_dims_used

                if only_in_read:
                    # Read uses a dimension not in write
                    # e.g., write a[i], read a[j] - dependency carried by j (outer)
                    for dim in only_in_read:
                        result['dep_dims'].add(dim)
                        result['parallel_dims'].discard(dim)

                if only_in_write and not only_in_read:
                    # Write uses dim not in read - unusual pattern
                    # e.g., write a[i], read a[const] - outer loop carries it
                    result['dep_dims'].add(dims[0])
                    result['parallel_dims'].discard(dims[0])

            else:
                # Same dimensions used but different expressions
                # e.g., write a[i], read a[i-j-1]
                # Need to check if loop-carried dependency exists

                try:
                    write_map = isl.Map(write['index'])
                    read_map = isl.Map(read['index'])

                    # Compute dependency relation
                    write_inv = write_map.reverse()
                    dep_rel = read_map.apply_range(write_inv)
                    dep_rel = dep_rel.intersect_domain(domain).intersect_range(domain)

                    if not dep_rel.is_empty():
                        # There is a dependency - analyze direction
                        # Check if it's loop-carried (different iteration) or same-iteration

                        identity = isl.Map.identity(domain.get_space().map_from_set())
                        cross_iter = dep_rel.subtract(identity)

                        if not cross_iter.is_empty():
                            # Loop-carried dependency exists
                            # Determine which dimension by checking if outer dim changes

                            # For nested loops, outer dimension carries if source and sink
                            # can have different outer dim values
                            # Use projection to check each dimension

                            dep_str = str(cross_iter)
                            # Heuristic: if the dependency relation shows outer dim varies,
                            # outer is sequential

                            # Check each dimension for whether it can vary in the dependency
                            for dim_idx, dim in enumerate(dims):
                                # If read index has a different expression involving this dim,
                                # and write doesn't, the dim carries the dependency
                                if dim in r_idx_expr and w_idx_expr != r_idx_expr:
                                    # More complex expression in read - likely loop-carried
                                    result['dep_dims'].add(dims[0])  # Conservative: outer
                                    result['parallel_dims'].discard(dims[0])
                                    break

                except Exception as e:
                    # Fallback: mark outer as sequential if indices differ
                    result['dep_dims'].add(dims[0])
                    result['parallel_dims'].discard(dims[0])

    return result


def generate_single_kernel_strategy(ordering, dims, domain_str, self_deps):
    """
    Generate a single-kernel Triton strategy for one ordering.

    Key principle: Launch ONE kernel with in-kernel sequential loop.
    Block parallelism handles the parallel dimension.
    """
    seq_dim = ordering['sequential_dim']
    par_dim = ordering['parallel_dim']
    par_type = ordering['parallelism_type']

    # Find the dependent array info
    dep_array = None
    write_expr = None
    read_expr = None
    for dep in self_deps:
        if dep['write_expr'] != dep['read_expr']:
            dep_array = dep['array']
            write_expr = dep['write_expr']
            read_expr = dep['read_expr']
            break

    if par_type == 'independent':
        strategy = f"""
### Single-Kernel Strategy: {seq_dim}-sequential, {par_dim}-parallel (Independent Writes)

**Pattern**: Write index uses {par_dim}, so each {par_dim} iteration writes to different location.
**Key**: Launch ONE kernel. Use in-kernel for-loop for {seq_dim}. Blocks parallelize {par_dim}.

```python
@triton.jit
def kernel({dep_array}_ptr, matrix_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Each block handles a contiguous chunk of {par_dim} dimension
    block_id = tl.program_id(0)
    {par_dim}_base = block_id * BLOCK_SIZE
    {par_dim}_offsets = {par_dim}_base + tl.arange(0, BLOCK_SIZE)

    # In-kernel sequential loop over {seq_dim}
    for {seq_dim} in range(N):
        # Compute valid mask for triangular bounds (e.g., {par_dim} > {seq_dim})
        # Adjust condition based on actual loop bounds
        mask = ({par_dim}_offsets > {seq_dim}) & ({par_dim}_offsets < N)

        # Load values needed for computation
        {dep_array}_{seq_dim} = tl.load({dep_array}_ptr + {seq_dim})  # Broadcast to all threads
        matrix_vals = tl.load(matrix_ptr + {seq_dim} * N + {par_dim}_offsets, mask=mask, other=0.0)
        {dep_array}_{par_dim} = tl.load({dep_array}_ptr + {par_dim}_offsets, mask=mask, other=0.0)

        # Compute: {dep_array}[{write_expr}] op= matrix[{seq_dim},{par_dim}] * {dep_array}[{read_expr}]
        result = {dep_array}_{par_dim} - matrix_vals * {dep_array}_{seq_dim}  # Adjust operator
        tl.store({dep_array}_ptr + {par_dim}_offsets, result, mask=mask)

# Single kernel launch - blocks cover {par_dim} dimension
grid = (triton.cdiv(N, BLOCK_SIZE),)
kernel[grid]({dep_array}, matrix, N, BLOCK_SIZE=256)
```

**Why this works**:
- All blocks process {seq_dim}=0 together, then {seq_dim}=1, etc.
- Within each {seq_dim} iteration, {par_dim} values are independent (write different locations)
- Reads from {dep_array}[{read_expr}] access earlier {seq_dim} iterations (already complete)
"""
    else:  # reduction
        strategy = f"""
### Single-Kernel Strategy: {seq_dim}-sequential, {par_dim}-parallel (Reduction)

**Pattern**: Write index uses only {seq_dim}, so all {par_dim} iterations write to SAME location.
**Key**: Each program handles one {seq_dim} value. Parallel reduction over {par_dim}.

```python
@triton.jit
def kernel({dep_array}_ptr, matrix_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Each program processes one {seq_dim} value with full {par_dim} reduction
    {seq_dim} = tl.program_id(0) + 1  # Adjust start based on loop bounds

    if {seq_dim} < N:
        # Initialize accumulator
        accumulator = tl.zeros([1], dtype=tl.float32)

        # Parallel reduction over {par_dim} (may need multiple chunks)
        for {par_dim}_base in range(0, {seq_dim}, BLOCK_SIZE):  # {par_dim} < {seq_dim} for triangular
            {par_dim}_offsets = {par_dim}_base + tl.arange(0, BLOCK_SIZE)
            mask = {par_dim}_offsets < {seq_dim}

            # Load matrix[{par_dim}, {seq_dim}] and {dep_array}[read_index]
            matrix_vals = tl.load(matrix_ptr + {par_dim}_offsets * N + {seq_dim}, mask=mask, other=0.0)
            # Read index: adjust based on actual pattern (e.g., {seq_dim} - {par_dim} - 1)
            read_idx = ...  # Compute based on {read_expr}
            {dep_array}_vals = tl.load({dep_array}_ptr + read_idx, mask=mask, other=0.0)

            # Accumulate partial sum
            partial = tl.sum(matrix_vals * {dep_array}_vals)
            accumulator += partial

        # Update {dep_array}[{seq_dim}]
        current = tl.load({dep_array}_ptr + {seq_dim})
        tl.store({dep_array}_ptr + {seq_dim}, current + accumulator)

# Launch: one program per {seq_dim} value
grid = (N - 1,)  # Adjust based on loop bounds
kernel[grid]({dep_array}, matrix, N, BLOCK_SIZE=256)
```

**Why this works**:
- Each {seq_dim} value processed independently (different programs)
- Within each {seq_dim}, reduction over {par_dim} uses tl.sum()
- Dependencies satisfied: {seq_dim}=k reads from {seq_dim}'<k (earlier programs complete first via grid ordering)
"""

    return strategy


def analyze_parallelization_strategy_both_orderings(kernel_name, both_analysis):
    """
    Generate Triton strategies for BOTH valid orderings.

    For triangular 2D loops, provides strategies for:
    - Option 1: dim0-sequential, dim1-parallel
    - Option 2: dim1-sequential, dim0-parallel
    """
    if not both_analysis:
        return None

    dims = both_analysis['dims']
    self_deps = both_analysis['self_dependencies']

    result = {
        'kernel': kernel_name,
        'dims': dims,
        'is_triangular': both_analysis['is_triangular'],
        'self_dependencies': self_deps,
        'strategies': []
    }

    for ordering in both_analysis['orderings']:
        strategy = generate_single_kernel_strategy(
            ordering, dims, both_analysis['domain'], self_deps
        )
        result['strategies'].append({
            'sequential_dim': ordering['sequential_dim'],
            'parallel_dim': ordering['parallel_dim'],
            'parallelism_type': ordering['parallelism_type'],
            'strategy': strategy
        })

    return result


def analyze_parallelization_strategy(kernel_name, analysis_result):
    """
    Generate Triton parallelization strategy based on dependency analysis.
    (Legacy function - kept for backward compatibility)
    """
    if not analysis_result:
        return None

    dims = analysis_result['dims']
    dep_dims = analysis_result['dep_dims']
    parallel_dims = analysis_result['parallel_dims']

    strategy = {
        'kernel': kernel_name,
        'loop_dims': dims,
        'sequential_dims': list(dep_dims),
        'parallel_dims': list(parallel_dims),
        'pattern': None,
        'triton_strategy': None
    }

    n_dims = len(dims)

    if n_dims == 1:
        if dep_dims:
            strategy['pattern'] = '1D-sequential'
            strategy['triton_strategy'] = 'Process sequentially in wrapper, or use single block with in-kernel loop'
        else:
            strategy['pattern'] = '1D-parallel'
            strategy['triton_strategy'] = 'Fully parallel across blocks'

    elif n_dims == 2:
        if len(dep_dims) == 0:
            strategy['pattern'] = '2D-fully-parallel'
            strategy['triton_strategy'] = 'Parallelize both dimensions across blocks'
        elif len(dep_dims) == 1:
            seq_dim = list(dep_dims)[0]
            par_dim = list(parallel_dims)[0] if parallel_dims else None
            seq_idx = dims.index(seq_dim)

            if seq_idx == 0:
                strategy['pattern'] = '2D-outer-sequential-inner-parallel'
            else:
                strategy['pattern'] = '2D-inner-sequential-outer-parallel'

            strategy['triton_strategy'] = f'''
Single-kernel approach for {seq_dim}-sequential, {par_dim}-parallel:
1. Launch ONE kernel (not N kernels!)
2. Use in-kernel for-loop over {seq_dim} (sequential dimension)
3. Block parallelism over {par_dim} (parallel dimension)
4. Each block handles a chunk of {par_dim} values
5. All blocks synchronize implicitly at each {seq_dim} iteration boundary

```python
@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr):
    block_id = tl.program_id(0)
    {par_dim}_offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    for {seq_dim} in range(N):  # Sequential in-kernel loop
        mask = compute_valid_mask({par_dim}_offsets, {seq_dim}, N)
        # Load, compute, store for this {seq_dim} iteration
        ...

grid = (triton.cdiv(N, BLOCK_SIZE),)
kernel[grid](...)
```'''
        else:
            strategy['pattern'] = '2D-fully-sequential'
            strategy['triton_strategy'] = 'Both dimensions have dependencies - limited parallelism'

    # Check for reduction pattern
    if analysis_result.get('is_reduction'):
        for dim, array in analysis_result['is_reduction'].items():
            strategy['reduction_dim'] = dim
            strategy['reduction_array'] = array
            strategy['triton_strategy'] += f'''

Note: {dim} dimension performs reduction on {array}.
Consider using tl.sum() or atomic operations for the reduction.'''

    return strategy


def format_strategy_for_prompt(strategy):
    """Format the parallelization strategy for LLM prompt."""
    if not strategy:
        return ""

    lines = []
    lines.append(f"## Parallelization Strategy Analysis")
    lines.append("")
    lines.append(f"**Loop dimensions**: {strategy['loop_dims']}")
    lines.append(f"**Pattern**: {strategy['pattern']}")
    lines.append(f"**Sequential dimensions** (must process in order): {strategy['sequential_dims']}")
    lines.append(f"**Parallel dimensions** (can distribute across blocks): {strategy['parallel_dims']}")
    lines.append("")
    lines.append("### Recommended Triton Implementation:")
    lines.append(strategy['triton_strategy'])

    return "\n".join(lines)


def analyze_kernel(kernel_file):
    """Full analysis of a kernel."""
    pet_output = run_pet(kernel_file)
    if not pet_output:
        return None

    try:
        data = yaml.safe_load(pet_output)
    except:
        return None

    results = []
    for stmt in data.get('statements', []):
        domain = stmt.get('domain', '')
        reads, writes = extract_accesses(stmt)

        dim_analysis = analyze_dependency_dimensions(
            domain,
            [{'index': r} for r in [r['index'] for r in reads]],
            [{'index': w} for w in [w['index'] for w in writes]]
        )

        if dim_analysis:
            results.append(dim_analysis)

    return results


def _analyze_nd_parallelization(kernel_name, ref_dims, data, stmt_loops, c_code):
    """
    Simple N-dimensional parallelization analysis for loops with >2 nesting levels.

    For each dimension, checks whether it appears in write array indices (parallel
    candidate: each iteration writes to a unique location) or only as an outer
    loop counter (sequential candidate: timestep loop).
    """
    # Collect all write accesses from computation statements (those with max dims)
    loop_groups = defaultdict(lambda: {'reads': [], 'writes': []})
    for stmt in data.get('statements', []):
        domain = stmt.get('domain', '')
        dims = get_loop_dimensions(domain)
        if len(dims) >= len(ref_dims) and dims[:len(ref_dims)] == ref_dims:
            reads_raw, writes_raw = extract_accesses(stmt)
            stmt_match = re.search(r'(S_\d+)', domain)
            stmt_name = stmt_match.group(1) if stmt_match else None
            loop_id = stmt_loops.get(stmt_name, 'default') if stmt_name else 'default'
            if loop_id is None:
                loop_id = 'default'
            loop_groups[loop_id]['reads'].extend([{'index': r['index']} for r in reads_raw])
            loop_groups[loop_id]['writes'].extend([{'index': w['index']} for w in writes_raw])

    if not loop_groups:
        return None

    # For each dimension, check if it indexes write arrays
    # A dim that indexes writes is a parallel candidate (each iteration writes unique location)
    # A dim that doesn't index any write is a sequential/outer loop
    dim_in_writes = {d: False for d in ref_dims}
    dim_reads_same_array = {d: False for d in ref_dims}

    all_write_arrays = set()
    all_read_arrays = set()

    for group in loop_groups.values():
        for w in group['writes']:
            w_match = re.search(r'->\s*(\w+)\[(.*?)\]', w['index'])
            if w_match:
                w_array, w_expr = w_match.group(1), w_match.group(2)
                all_write_arrays.add(w_array)
                for d in ref_dims:
                    if re.search(rf'\b{d}\b', w_expr):
                        dim_in_writes[d] = True
        for r in group['reads']:
            r_match = re.search(r'->\s*(\w+)\[(.*?)\]', r['index'])
            if r_match:
                all_read_arrays.add(r_match.group(1))

    # Check for read-write overlap (same array read and written)
    rw_overlap = all_write_arrays & all_read_arrays

    # For each group, check if reads and writes go to DIFFERENT arrays
    # (double-buffering pattern: read A write B, then read B write A)
    per_group_disjoint = True
    for group in loop_groups.values():
        g_writes = set()
        g_reads = set()
        for w in group['writes']:
            m = re.search(r'->\s*(\w+)\[', w['index'])
            if m:
                g_writes.add(m.group(1))
        for r in group['reads']:
            m = re.search(r'->\s*(\w+)\[', r['index'])
            if m:
                g_reads.add(m.group(1))
        if g_writes & g_reads:
            per_group_disjoint = False

    # Build options for each dimension
    options = []
    parallel_dims = []
    sequential_dims = []
    for d in ref_dims:
        if dim_in_writes[d]:
            # This dim indexes array writes -> parallel candidate
            # Safe if: within each group, reads and writes go to different arrays
            # OR reads don't depend on this dim across iterations
            if per_group_disjoint or not rw_overlap:
                parallel_dims.append(d)
            else:
                # Same array read+written in same group: could have cross-iteration dep
                parallel_dims.append(d)  # Still parallel (indexed in write = unique location)
        else:
            sequential_dims.append(d)

    # Build result in same format as 2D analysis
    result_options = []

    if sequential_dims and parallel_dims:
        # Primary option: parallelize spatial dims, sequential outer
        opt = {
            'sequential_dim': ', '.join(sequential_dims),
            'parallel_dim': ', '.join(parallel_dims),
            'valid': True,
            'parallelism_type': 'independent',
            'triton_strategy': 'SINGLE_KERNEL_INLOOP' if len(sequential_dims) == 1 else 'MULTI_KERNEL_LAUNCH',
            'issues': [],
            'explanations': [],
            'inkernel_safety_details': [],
        }
        if per_group_disjoint:
            opt['explanations'].append(
                f"Within each loop nest, reads and writes go to different arrays (double-buffering)")
        result_options.append(opt)
    elif parallel_dims:
        # All dims are parallel (no sequential outer loop)
        opt = {
            'sequential_dim': 'none',
            'parallel_dim': ', '.join(parallel_dims),
            'valid': True,
            'parallelism_type': 'independent',
            'triton_strategy': 'SINGLE_KERNEL_INLOOP',
            'issues': [],
            'explanations': [],
            'inkernel_safety_details': [],
        }
        result_options.append(opt)

    if not result_options:
        return None

    summary_parts = []
    if sequential_dims:
        summary_parts.append(f"{', '.join(sequential_dims)}: sequential")
    if parallel_dims:
        summary_parts.append(f"{', '.join(parallel_dims)}: parallel")

    return {
        'kernel': kernel_name,
        'c_code': c_code,
        'dims': ref_dims,
        'is_triangular': False,
        'triangular_info': None,
        'self_dependencies': [],
        'options': result_options,
        'summary': '; '.join(summary_parts),
    }


def analyze_kernel_parallelization(kernel_name: str, kernel_file: str = None):
    """
    Analyze parallelization options for a kernel (on-the-fly).

    This is the exported function to be called by generate_and_test.py.
    Returns structured analysis data for 2D loops.

    Args:
        kernel_name: Name of the kernel (e.g., 's119')
        kernel_file: Optional full path to kernel .c file. If not provided,
                     looks in KERNELS_DIR (TSVC default).

    Returns:
        dict with keys:
            - kernel: kernel name
            - c_code: C code of the loop
            - dims: list of dimension names
            - is_triangular: bool
            - triangular_info: dict or None
            - self_dependencies: list of {array, write_expr, read_expr}
            - options: list of parallelization options with triton_strategy
            - summary: string summary
        Returns None if analysis fails or kernel is not 2D.
    """
    if kernel_file is None:
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

    # Get C code from kernel file
    c_code = ""
    try:
        with open(kernel_file, 'r') as f:
            c_code = f.read().strip()
    except:
        pass

    # Parse schedule to group statements by innermost loop
    stmt_loops = {}
    schedule_str = data.get('schedule', '')
    if schedule_str:
        try:
            from compute_war_dependences import parse_isl_schedule, get_stmt_innermost_loops
            tree = parse_isl_schedule(schedule_str)
            stmt_loops = get_stmt_innermost_loops(tree)
        except Exception:
            pass

    # Find the reference dims from the statement with the MOST dimensions.
    # Previously took first 2D+ statement and truncated to 2 dims, which
    # could pick a counter-init statement (e.g., S_2[t,i]) instead of the
    # computation statement (e.g., S_4[t,i,j,k]).
    #
    # Strategy:
    # - For 4D+ loops: use N-D analysis (2D analysis can't handle it)
    # - For 3D loops: take first 2 dims from max-dim statement, use 2D analysis
    #   (preserves per-dim options and WAR cross-reference)
    # - For 2D loops: existing 2D analysis unchanged
    ref_dims = None
    ref_domain = None
    max_ndims = 0
    max_dims_all = None
    for stmt in data.get('statements', []):
        domain = stmt.get('domain', '')
        dims = get_loop_dimensions(domain)
        if len(dims) >= 2 and len(dims) > max_ndims:
            max_ndims = len(dims)
            max_dims_all = dims

    if max_dims_all is None:
        return None

    if max_ndims >= 4:
        # 4D+ loops: use N-D analysis
        return _analyze_nd_parallelization(kernel_name, max_dims_all, data, stmt_loops, c_code)

    # For 2D and 3D: use 2D analysis
    if max_ndims == 3:
        # For 3D loops, pick the 2 dims that index write arrays (spatial dims).
        # PET domain order follows schedule nesting, not spatial relevance.
        # e.g. gemm domain [i,k,j] but writes C[i][j] → spatial = [i,j]
        write_dims = set()
        for stmt in data.get('statements', []):
            domain = stmt.get('domain', '')
            dims = get_loop_dimensions(domain)
            if len(dims) >= 2:
                _, writes_raw = extract_accesses(stmt)
                for w in writes_raw:
                    w_match = re.search(r'->\s*\w+\[(.*?)\]', w['index'])
                    if w_match:
                        for d in max_dims_all:
                            if re.search(rf'\b{d}\b', w_match.group(1)):
                                write_dims.add(d)
        spatial = [d for d in max_dims_all if d in write_dims]
        ref_dims = spatial[:2] if len(spatial) >= 2 else max_dims_all[:2]
    else:
        ref_dims = max_dims_all[:2]
    # Find a 2D domain matching ref_dims for the analysis function
    for stmt in data.get('statements', []):
        domain = stmt.get('domain', '')
        dims = get_loop_dimensions(domain)
        if len(dims) == 2 and dims[0] == ref_dims[0] and dims[1] == ref_dims[1]:
            ref_domain = domain
            break

    # Group statements by innermost loop, collecting reads/writes per group
    loop_groups = defaultdict(lambda: {'reads': [], 'writes': []})
    for stmt in data.get('statements', []):
        domain = stmt.get('domain', '')
        dims = get_loop_dimensions(domain)
        if len(dims) >= 2 and dims[0] == ref_dims[0] and dims[1] == ref_dims[1]:
            reads_raw, writes_raw = extract_accesses(stmt)
            # Determine which loop group this statement belongs to
            stmt_match = re.search(r'(S_\d+)', domain)
            stmt_name = stmt_match.group(1) if stmt_match else None
            loop_id = stmt_loops.get(stmt_name, 'default') if stmt_name else 'default'
            if loop_id is None:
                loop_id = 'default'
            loop_groups[loop_id]['reads'].extend([{'index': r['index']} for r in reads_raw])
            loop_groups[loop_id]['writes'].extend([{'index': w['index']} for w in writes_raw])
            if len(dims) == 2 and ref_domain is None:
                ref_domain = domain

    if not loop_groups:
        return None

    if ref_domain is None:
        return None

    # Analyze each loop group separately, then take the most conservative result
    # A dimension is only safe to parallelize if it's safe in ALL groups
    all_reads = []
    all_writes = []
    for group in loop_groups.values():
        all_reads.extend(group['reads'])
        all_writes.extend(group['writes'])

    if len(loop_groups) <= 1:
        # Single group (or no schedule info): analyze all together
        analysis = analyze_parallelization_options_full(kernel_name, ref_domain, all_reads, all_writes, c_code)
    else:
        # Multiple groups: analyze each group, merge results conservatively
        group_analyses = []
        for loop_id, group in loop_groups.items():
            if group['reads'] and group['writes']:
                ga = analyze_parallelization_options_full(kernel_name, ref_domain, group['reads'], group['writes'], c_code)
                if ga:
                    group_analyses.append(ga)

        if not group_analyses:
            analysis = analyze_parallelization_options_full(kernel_name, ref_domain, all_reads, all_writes, c_code)
        else:
            # Start from the first group's analysis, then merge in others
            analysis = group_analyses[0]
            for ga in group_analyses[1:]:
                for i, opt in enumerate(analysis['options']):
                    if i < len(ga['options']):
                        other_opt = ga['options'][i]
                        # If ANY group says invalid, the option is invalid
                        if not other_opt['valid']:
                            opt['valid'] = False
                            opt['issues'].extend(other_opt['issues'])
                            opt['triton_strategy'] = 'INVALID'
                        # Merge inkernel_safety_details
                        opt.get('inkernel_safety_details', []).extend(
                            other_opt.get('inkernel_safety_details', []))
                # Merge self_dependencies
                analysis['self_dependencies'].extend(ga['self_dependencies'])

        # Cross-group WAR check: if group A writes array X and group B reads X,
        # the outermost dim binding both groups is NOT parallelizable.
        if len(loop_groups) > 1 and analysis and analysis.get('options'):
            group_list = list(loop_groups.values())
            has_cross_group_dep = False
            for gi, g1 in enumerate(group_list):
                g1_writes = set()
                for w in g1['writes']:
                    m = re.search(r'->\s*(\w+)\[', w['index'])
                    if m:
                        g1_writes.add(m.group(1))
                for gj, g2 in enumerate(group_list):
                    if gi == gj:
                        continue
                    for r_acc in g2['reads']:
                        m = re.search(r'->\s*(\w+)\[', r_acc['index'])
                        if m and m.group(1) in g1_writes:
                            has_cross_group_dep = True
                            break
                    if has_cross_group_dep:
                        break
                if has_cross_group_dep:
                    break

            if has_cross_group_dep:
                # Mark the OUTER dim (ref_dims[0]) as invalid — it binds the groups
                for opt in analysis['options']:
                    if opt.get('parallel_dim') == ref_dims[0]:
                        opt['valid'] = False
                        opt['issues'].append(
                            f"Cross-phase dependency: different loop bodies under `{ref_dims[0]}` "
                            f"read/write the same arrays (e.g., phase 1 writes B, phase 2 reads B)"
                        )

    if not analysis:
        return None

    # Build result in the format expected by generate_and_test.py
    result = {
        'kernel': kernel_name,
        'c_code': c_code,
        'dims': analysis['dims'],
        'is_triangular': analysis['is_triangular'],
        'triangular_info': analysis['triangular_info'],
        'self_dependencies': analysis['self_dependencies'],
        'options': [],
        'summary': ''
    }

    # Convert options to the format expected by generate_and_test.py
    valid_opts = []
    for opt in analysis['options']:
        converted_opt = {
            'sequential_dim': opt['sequential_dim'],
            'parallel_dim': opt['parallel_dim'],
            'valid': opt['valid'],
            'parallelism_type': opt['parallelism_type'],
            'triton_strategy': opt['triton_strategy'],
            'issues': opt['issues'],
            'explanations': [],
            'inkernel_safety_details': opt.get('inkernel_safety_details', [])
        }

        # Build explanations from inkernel_safety_details
        for detail in opt.get('inkernel_safety_details', []):
            safe_marker = '✓' if detail['safe'] else '✗'
            converted_opt['explanations'].append(
                f"{safe_marker} {detail['array']}: {detail['reason']} [{detail['pattern_type']}]"
            )

        result['options'].append(converted_opt)

        if opt['valid']:
            valid_opts.append(f"{opt['sequential_dim']}-seq, {opt['parallel_dim']}-par: {opt['triton_strategy']}")

    # Build summary
    if len(valid_opts) == 2:
        result['summary'] = f"Both orderings valid (choose based on performance)\n    -> {valid_opts[0]}\n    -> {valid_opts[1]}"
    elif len(valid_opts) == 1:
        result['summary'] = f"One valid ordering: {valid_opts[0]}"
    else:
        result['summary'] = "No valid parallelization found"

    return result


def analyze_read_dependency_type(read_expr, write_expr, dims, par_dim, seq_dim, parallelism_type, triangular_info=None):
    """
    Analyze read dependency type when par_dim is parallelized.

    Key patterns:
    1. Broadcast read: read index doesn't contain par_dim -> Safe
    2. Same location: read index = write index -> Safe (reduction handles write conflicts)
    3. Chain dependency within par_dim: different par_dim values read what others write -> UNSAFE
    4. Row shift pattern: read from different seq_dim value -> Safe (like s1119: aa[i-1][j])
    5. Transpose pattern with triangular bounds: write[i][j] and read[j][i] with j<i -> SAFE (disjoint)

    For s118-like patterns (i-seq, j-par, reduction):
      a[i] += bb[j][i] * a[i-j-1]
      - Write: a[i] (same for all j -> reduction)
      - Read: a[i-j-1] (different for each j, but all < i)
      - Within same i iteration, j values read from a[0..i-1], write to a[i]
      - No j reads what another j writes -> SAFE for j-parallel reduction!

    For s1119-like patterns (i-seq, j-par, independent):
      aa[i][j] = aa[i-1][j] + bb[i][j]
      - Write: aa[i][j], Read: aa[i-1][j]
      - par_dim (j) is same in both, but seq_dim (i) differs (i vs i-1)
      - Different j values read/write different columns, no conflict -> SAFE

    For s114-like patterns (transpose with triangular bounds):
      aa[i][j] = aa[j][i] + bb[i][j] (with j < i)
      - Write: aa[i][j] (lower triangle, row > col)
      - Read: aa[j][i] (upper triangle, row < col)
      - Regions are DISJOINT -> SAFE for full parallelization
    """
    read_dims = set()
    for dim in dims:
        if re.search(rf'\b{dim}\b', read_expr):
            read_dims.add(dim)

    write_dims = set()
    for dim in dims:
        if re.search(rf'\b{dim}\b', write_expr):
            write_dims.add(dim)

    result = {
        'type': None,
        'safe_to_parallelize': True,
        'explanation': ''
    }

    if par_dim not in read_dims:
        # Write-conflict: if par_dim is ALSO absent from write, then ALL parallel
        # threads read AND write the same location → reduction or race condition.
        # e.g. atax: tmp[i] += A[i][j]*x[j] with j-parallel → all j-threads write tmp[i]
        if par_dim not in write_dims:
            result['type'] = 'write_conflict'
            result['safe_to_parallelize'] = False
            result['explanation'] = (
                f"Write conflict: write [{write_expr}] and read [{read_expr}] both lack "
                f"{par_dim}. All {par_dim}-parallel threads write same location."
            )
            return result
        result['type'] = 'broadcast'
        result['explanation'] = f"Broadcast read: [{read_expr}] doesn't contain {par_dim}"
        return result

    if read_expr == write_expr:
        # Scalar check: if the subscript contains NO loop dims, this is a
        # loop-carried scalar dependency (e.g. ym1, alpha, sum in deriche/durbin).
        expr_dims = set(dim for dim in dims if re.search(rf'\b{dim}\b', read_expr))
        if not expr_dims:
            result['type'] = 'scalar_dep'
            result['safe_to_parallelize'] = False
            result['explanation'] = (
                f"Scalar dependency: [{write_expr or 'scalar'}] has no loop dimensions "
                f"in subscript — loop-carried across iterations"
            )
            return result
        result['type'] = 'same_location'
        result['explanation'] = f"Same location read/write [{read_expr}]"
        return result

    # par_dim is in read_expr but differs from write_expr
    # Need to check if parallel iterations can conflict

    # Key check: does par_dim appear in write_expr?
    if par_dim not in write_dims:
        # Write doesn't use par_dim -> all par_dim iterations write to SAME location
        # This is a REDUCTION pattern

        # For reduction: check if any par_dim iteration reads what another writes
        # If write is a[seq_dim] and read is a[f(seq_dim, par_dim)] where f(...) != seq_dim,
        # then parallel iterations read from DIFFERENT locations than they write
        # -> SAFE as long as reads are from "earlier" seq_dim iterations

        # Check if the read could access the current write location
        # For s118: write a[i], read a[i-j-1]. When j=0, read a[i-1] != a[i] -> safe
        # The reads are always from i - j - 1 < i for j >= 0

        # Heuristic: if read_expr contains subtraction involving par_dim,
        # it likely reads from earlier indices, not the current write location
        if re.search(rf'-\s*{par_dim}', read_expr) or re.search(rf'{par_dim}\s*-', read_expr):
            result['type'] = 'reduction_safe'
            result['explanation'] = (
                f"Reduction pattern: read [{read_expr}] varies with {par_dim}, "
                f"write [{write_expr}] doesn't. Reads from different locations than write -> SAFE"
            )
            return result

        # Check if read could equal write for some par_dim value
        # This requires more sophisticated analysis
        # For now, be conservative only if read could potentially equal write
        result['type'] = 'reduction_uncertain'
        result['explanation'] = (
            f"Reduction: write [{write_expr}], read [{read_expr}]. "
            f"Check if any {par_dim} value causes read to equal write."
        )
        return result

    # par_dim in both read and write
    # Check for shift patterns in 2D arrays

    # Parse as tuple: (first_idx, second_idx) or single idx
    read_parts = [p.strip() for p in read_expr.split(',')]
    write_parts = [p.strip() for p in write_expr.split(',')]

    if len(read_parts) == len(write_parts) == 2:
        # 2D array access - analyze the shift pattern
        # Three key patterns:
        # 1. Row shift: seq_dim differs, par_dim same (s1119: aa[i-1][j] -> aa[i][j])
        # 2. Diagonal shift: BOTH differ by constants (s119: aa[i-1][j-1] -> aa[i][j])
        # 3. Transpose: indices swapped (s114: aa[j][i] -> aa[i][j])

        r_idx0_has_seq = bool(re.search(rf'\b{seq_dim}\b', read_parts[0]))
        w_idx0_has_seq = bool(re.search(rf'\b{seq_dim}\b', write_parts[0]))
        r_idx0_has_par = bool(re.search(rf'\b{par_dim}\b', read_parts[0]))
        w_idx0_has_par = bool(re.search(rf'\b{par_dim}\b', write_parts[0]))

        r_idx1_has_seq = bool(re.search(rf'\b{seq_dim}\b', read_parts[1]))
        w_idx1_has_seq = bool(re.search(rf'\b{seq_dim}\b', write_parts[1]))
        r_idx1_has_par = bool(re.search(rf'\b{par_dim}\b', read_parts[1]))
        w_idx1_has_par = bool(re.search(rf'\b{par_dim}\b', write_parts[1]))

        # Check for row shift pattern: CRITICAL - must verify par_dim is in the IDENTICAL index
        # Row shift is SAFE when: par_dim index is same in read/write (different par values access disjoint slices)
        # Row shift is UNSAFE when: par_dim index shifts (chain dependency along par_dim)

        if read_parts[0] != write_parts[0] and read_parts[1] == write_parts[1]:
            # First index differs, second same
            # Check if par_dim is in the IDENTICAL part (index 1) - that's SAFE
            # If par_dim is in the DIFFERING part (index 0) - that's a chain dependency - UNSAFE
            if re.search(rf'\b{par_dim}\b', write_parts[1]):
                # par_dim is in the identical index - SAFE
                result['type'] = 'row_shift_safe'
                result['explanation'] = (
                    f"Row shift pattern: read [{read_expr}] and write [{write_expr}] "
                    f"differ in first index but {par_dim} index (second) is identical. "
                    f"Different {par_dim} values access disjoint array slices -> SAFE"
                )
                return result
            elif re.search(rf'\b{par_dim}\b', write_parts[0]):
                # par_dim is in the shifting index - CHAIN DEPENDENCY - UNSAFE
                result['type'] = 'chain_dependency_unsafe'
                result['safe_to_parallelize'] = False  # Mark as unsafe!
                result['explanation'] = (
                    f"Chain dependency: read [{read_expr}] and write [{write_expr}] - "
                    f"{par_dim} index shifts (first index differs). "
                    f"Different {par_dim} values have RAW dependencies -> UNSAFE for {par_dim}-parallel"
                )
                return result

        elif read_parts[0] == write_parts[0] and read_parts[1] != write_parts[1]:
            # First index same, second differs
            # Check if par_dim is in the IDENTICAL part (index 0) - that's SAFE
            # If par_dim is in the DIFFERING part (index 1) - that's a chain dependency - UNSAFE
            if re.search(rf'\b{par_dim}\b', write_parts[0]):
                # par_dim is in the identical index - SAFE
                result['type'] = 'row_shift_safe'
                result['explanation'] = (
                    f"Row shift pattern: read [{read_expr}] and write [{write_expr}] "
                    f"differ in second index but {par_dim} index (first) is identical. "
                    f"Different {par_dim} values access disjoint array slices -> SAFE"
                )
                return result
            elif re.search(rf'\b{par_dim}\b', write_parts[1]):
                # par_dim is in the shifting index - CHAIN DEPENDENCY - UNSAFE
                result['type'] = 'chain_dependency_unsafe'
                result['safe_to_parallelize'] = False  # Mark as unsafe!
                result['explanation'] = (
                    f"Chain dependency: read [{read_expr}] and write [{write_expr}] - "
                    f"{par_dim} index shifts (second index differs). "
                    f"Different {par_dim} values have RAW dependencies -> UNSAFE for {par_dim}-parallel"
                )
                return result

        # Check for diagonal shift pattern: BOTH indices differ, but by CONSTANTS
        # e.g., read (i-1, j-1), write (i, j)
        #
        # CRITICAL: Diagonal shift is NOT SAFE for in-kernel loops!
        # When parallelizing j with in-kernel sequential loop over i:
        #   - Thread j at iteration i reads from (i-1, j-1)
        #   - Thread j-1 at iteration i-1 wrote to (i-1, j-1)
        #   - These are DIFFERENT THREADS, not different iterations of the same thread!
        #   - GPU does NOT guarantee thread j sees thread j-1's writes without explicit sync
        #
        # This is different from row-shift aa[i][j] = aa[i-1][j] where:
        #   - Thread j reads from (i-1, j) which the SAME thread j wrote
        #   - GPU guarantees program order for same thread
        #
        if read_parts[0] != write_parts[0] and read_parts[1] != write_parts[1]:
            # Both indices differ - check if they're constant shifts
            # Extract the dimension and check for constant offset pattern
            # E.g., "(-1 + i)" vs "(i)" or "(i - 1)" vs "(i)"

            def is_constant_shift(read_idx, write_idx, dim):
                """Check if read_idx is write_idx minus a constant."""
                # Patterns: "(dim)" vs "(-c + dim)" or "(dim - c)"
                # Strip parens and compare
                r_clean = re.sub(r'[()]', '', read_idx).strip()
                w_clean = re.sub(r'[()]', '', write_idx).strip()

                # If write is just the dimension
                if w_clean == dim:
                    # Check if read is dim - constant
                    if re.match(rf'^-?\d+\s*\+\s*{dim}$', r_clean):  # e.g., -1 + i
                        return True
                    if re.match(rf'^{dim}\s*-\s*\d+$', r_clean):  # e.g., i - 1
                        return True
                return False

            # Check if both indices are constant shifts
            idx0_is_shift = is_constant_shift(read_parts[0], write_parts[0], seq_dim)
            idx1_is_shift = is_constant_shift(read_parts[1], write_parts[1], par_dim)

            if idx0_is_shift and idx1_is_shift:
                # Diagonal shift - NOT SAFE because par_dim index shifts means cross-thread read
                result['type'] = 'diagonal_shift_cross_thread'
                result['safe_to_parallelize'] = False  # Mark as UNSAFE!
                result['explanation'] = (
                    f"Diagonal shift pattern: read [{read_expr}] and write [{write_expr}] - "
                    f"{par_dim} index shifts by constant. Thread {par_dim} reads from thread {par_dim}-1 -> "
                    f"CROSS-THREAD dependency, NOT SAFE for in-kernel loop"
                )
                return result

            # Also check the reverse assignment (in case dims are in different order)
            idx0_is_shift_rev = is_constant_shift(read_parts[0], write_parts[0], par_dim)
            idx1_is_shift_rev = is_constant_shift(read_parts[1], write_parts[1], seq_dim)

            if idx0_is_shift_rev and idx1_is_shift_rev:
                # Diagonal shift - NOT SAFE because par_dim index shifts means cross-thread read
                result['type'] = 'diagonal_shift_cross_thread'
                result['safe_to_parallelize'] = False  # Mark as UNSAFE!
                result['explanation'] = (
                    f"Diagonal shift pattern: read [{read_expr}] and write [{write_expr}] - "
                    f"{par_dim} index shifts by constant. Thread {par_dim} reads from thread {par_dim}-1 -> "
                    f"CROSS-THREAD dependency, NOT SAFE for in-kernel loop"
                )
                return result

    # Check for transpose pattern with triangular bounds (s114-like)
    # Pattern: write[i][j] and read[j][i] with constraint j < i (or i < j)
    # The write and read regions are disjoint (lower vs upper triangle)
    if len(dims) == 2 and triangular_info:
        write_parts = [p.strip() for p in write_expr.split(',')]
        read_parts_tr = [p.strip() for p in read_expr.split(',')]

        if len(write_parts) == 2 and len(read_parts_tr) == 2:
            w0 = re.sub(r'[()]', '', write_parts[0]).strip()
            w1 = re.sub(r'[()]', '', write_parts[1]).strip()
            r0 = re.sub(r'[()]', '', read_parts_tr[0]).strip()
            r1 = re.sub(r'[()]', '', read_parts_tr[1]).strip()

            # Check if indices are swapped: write[i][j] and read[j][i]
            if w0 == r1 and w1 == r0 and w0 != w1:
                # This is a transpose access pattern!
                # With triangular bounds, write region and read region are DISJOINT
                result['type'] = 'transpose_disjoint'
                result['safe_to_parallelize'] = True
                result['explanation'] = (
                    f"Transpose access with triangular bounds: write [{write_expr}] and read [{read_expr}] "
                    f"have swapped indices. With constraint {triangular_info['smaller']} < {triangular_info['larger']}, "
                    f"write region (lower triangle) and read region (upper triangle) are DISJOINT -> SAFE"
                )
                return result

    # Check for chain dependency
    if seq_dim in read_dims and par_dim in write_dims:
        # Write uses par_dim, read uses both -> potential chain
        # e.g., write a[i], read a[j] where j < i
        result['type'] = 'chain_shift'
        result['safe_to_parallelize'] = False
        result['explanation'] = (
            f"Shift dependency: write [{write_expr}] and read [{read_expr}] "
            f"both use {par_dim} differently. {par_dim} iteration k may read what k-1 wrote."
        )
    else:
        result['type'] = 'chain_complex'
        result['safe_to_parallelize'] = False
        result['explanation'] = (
            f"CHAIN DEPENDENCY: read [{read_expr}] differs from write [{write_expr}]. "
            f"Different {par_dim} values may have inter-dependencies."
        )

    return result


def analyze_parallelization_options_full(kernel_name, domain_str, reads, writes, c_code=""):
    """
    Analyze all parallelization options for a 2D loop.
    Returns analysis with valid/invalid options, reasons, and Triton strategy recommendation.

    Key output fields:
    - valid: Whether this ordering satisfies dependencies
    - safe_for_inkernel_loop: Whether single kernel with in-kernel loop is safe
    - triton_strategy: 'SINGLE_KERNEL_INLOOP' or 'MULTI_KERNEL_LAUNCH'

    Args:
        kernel_name: Name of the kernel
        domain_str: ISL domain string
        reads: List of read accesses
        writes: List of write accesses
        c_code: C source code for checking reduction vs overwrite patterns
    """
    dims = get_loop_dimensions(domain_str)
    if len(dims) != 2:
        return None

    # Check triangular bounds
    is_triangular = False
    tri_info = None
    d0, d1 = dims[0], dims[1]
    if re.search(rf'\b{d0}\s*<\s*{d1}\b', domain_str) or re.search(rf'\b{d0}\s*<=\s*{d1}\s*-', domain_str):
        is_triangular = True
        tri_info = {'smaller': d0, 'larger': d1}
    elif re.search(rf'\b{d1}\s*<\s*{d0}\b', domain_str) or re.search(rf'\b{d1}\s*<=\s*{d0}\s*-', domain_str):
        is_triangular = True
        tri_info = {'smaller': d1, 'larger': d0}

    # Find self-dependencies and determine array dimensionality
    self_deps = []
    array_dims = {}  # Track array dimensionality
    for w in writes:
        w_match = re.search(r'->\s*(\w+)\[(.*?)\]', w['index'])
        if not w_match:
            continue
        w_array, w_expr = w_match.group(1), w_match.group(2)
        w_parts = get_index_parts(w_expr)
        array_dims[w_array] = len(w_parts)

        for r in reads:
            r_match = re.search(r'->\s*(\w+)\[(.*?)\]', r['index'])
            if not r_match:
                continue
            r_array, r_expr = r_match.group(1), r_match.group(2)

            if w_array == r_array:
                self_deps.append({
                    'array': w_array,
                    'write_expr': w_expr,
                    'read_expr': r_expr,
                    'is_1d': len(w_parts) == 1
                })

    result = {
        'kernel': kernel_name,
        'dims': dims,
        'domain': domain_str,
        'is_triangular': is_triangular,
        'triangular_info': tri_info,
        'self_dependencies': self_deps,
        'options': []
    }

    # Analyze both orderings
    for seq_idx in range(2):
        seq_dim = dims[seq_idx]
        par_dim = dims[1 - seq_idx]

        option = {
            'sequential_dim': seq_dim,
            'parallel_dim': par_dim,
            'valid': True,
            'parallelism_type': None,
            'safe_for_inkernel_loop': True,  # NEW: in-kernel loop safety
            'triton_strategy': None,  # NEW: SINGLE_KERNEL_INLOOP or MULTI_KERNEL_LAUNCH
            'issues': [],
            'read_patterns': [],
            'inkernel_safety_details': []  # NEW: detailed safety analysis
        }

        # Analyze write pattern
        for w in writes:
            w_match = re.search(r'->\s*(\w+)\[(.*?)\]', w['index'])
            if w_match:
                w_array = w_match.group(1)
                w_expr = w_match.group(2)
                w_dims = set(dim for dim in dims if re.search(rf'\b{dim}\b', w_expr))

                if par_dim in w_dims:
                    option['parallelism_type'] = 'independent'
                else:
                    # Check if it's actual reduction or just overwrite
                    if is_actual_reduction(c_code, w_array):
                        option['parallelism_type'] = 'reduction'
                    else:
                        option['parallelism_type'] = 'overwrite'

        # Analyze read patterns for parallelization validity AND in-kernel loop safety
        for dep in self_deps:
            # Original analysis for parallelization validity
            read_analysis = analyze_read_dependency_type(
                dep['read_expr'], dep['write_expr'], dims, par_dim, seq_dim,
                option['parallelism_type'], tri_info
            )
            option['read_patterns'].append(read_analysis)
            if not read_analysis['safe_to_parallelize']:
                option['valid'] = False
                option['issues'].append(read_analysis['explanation'])

            # NEW: Check in-kernel loop safety
            is_safe, reason, pattern_type = is_safe_for_inkernel_loop(
                dep['write_expr'], dep['read_expr'], dims, seq_dim, par_dim, dep['is_1d'], tri_info
            )
            option['inkernel_safety_details'].append({
                'array': dep['array'],
                'safe': is_safe,
                'reason': reason,
                'pattern_type': pattern_type
            })
            if not is_safe:
                option['safe_for_inkernel_loop'] = False

        # Determine Triton strategy
        if option['valid']:
            if option['safe_for_inkernel_loop']:
                option['triton_strategy'] = 'SINGLE_KERNEL_INLOOP'
            else:
                option['triton_strategy'] = 'MULTI_KERNEL_LAUNCH'
        else:
            option['triton_strategy'] = 'INVALID'

        result['options'].append(option)

    return result


def save_parallelization_analysis(output_file):
    """
    Analyze all kernels and save parallelization analysis to file.
    Format similar to flow_deps_s000_s119.txt
    """
    # Get all kernel files
    kernel_files = sorted([f for f in os.listdir(KERNELS_DIR) if f.endswith('.c')])

    lines = []
    lines.append("=" * 80)
    lines.append("PARALLELIZATION DIMENSION ANALYSIS + TRITON STRATEGY RECOMMENDATION")
    lines.append("=" * 80)
    lines.append("")
    lines.append("This analysis determines:")
    lines.append("1. Which loop dimensions can be parallelized and which must be sequential")
    lines.append("2. Whether in-kernel loop is safe or requires multiple kernel launches")
    lines.append("")
    lines.append("TRITON STRATEGIES:")
    lines.append("  SINGLE_KERNEL_INLOOP: Safe to use single kernel with for-loop inside")
    lines.append("    - Row-shift pattern: par_dim index IDENTICAL in read/write (s1119, s126)")
    lines.append("    - Different parallel blocks access DISJOINT array slices")
    lines.append("")
    lines.append("  MULTI_KERNEL_LAUNCH: Must launch separate kernels for each seq_dim iteration")
    lines.append("    - Cross-block dependency: reads may need values from other blocks (s115, s118)")
    lines.append("    - 1D array where seq_dim reads cross block boundaries")
    lines.append("")
    lines.append("PATTERN TYPES:")
    lines.append("  ✓ row_shift      - 2D array, par_dim index same in read/write -> SAFE")
    lines.append("  ✗ diagonal_shift - 2D array, both indices shift by constants -> UNSAFE (cross-thread dependency)")
    lines.append("  ✓ broadcast      - Read doesn't use par_dim -> SAFE")
    lines.append("  ✓ same_location  - Read == write index -> SAFE")
    lines.append("  ✓ reduction_safe - Reduction where reads don't conflict -> SAFE")
    lines.append("  ✗ cross_block    - Reads may need values from other blocks -> UNSAFE")
    lines.append("  ✗ cross_block_1d - 1D array cross-block dependency -> UNSAFE")
    lines.append("  ✗ diagonal_shift_cross_thread - Thread reads from different thread's write -> UNSAFE")
    lines.append("")

    for kernel_file in kernel_files:
        kernel_name = kernel_file[:-2]  # Remove .c
        kernel_path = os.path.join(KERNELS_DIR, kernel_file)

        pet_output = run_pet(kernel_path)
        if not pet_output:
            continue

        try:
            data = yaml.safe_load(pet_output)
        except:
            continue

        # Read C code
        with open(kernel_path, 'r') as f:
            c_code = f.read()
        scop_match = re.search(r'#pragma scop\s*(.*?)\s*#pragma endscop', c_code, re.DOTALL)
        c_code = scop_match.group(1).strip() if scop_match else ''

        lines.append("=" * 40)
        lines.append(f"Kernel: {kernel_name}")
        lines.append("=" * 40)
        lines.append("")
        lines.append(f"C Code: {c_code}")
        lines.append("")

        for i, stmt in enumerate(data.get('statements', [])):
            domain = stmt.get('domain', '')
            reads_raw, writes_raw = extract_accesses(stmt)
            reads = [{'index': r['index']} for r in reads_raw]
            writes = [{'index': w['index']} for w in writes_raw]

            dims = get_loop_dimensions(domain)

            lines.append(f"Statement S_{i}:")
            lines.append(f"  Domain: {domain}")
            lines.append(f"  Dimensions: {dims}")
            lines.append(f"  Writes: {[w['index'] for w in writes]}")
            lines.append(f"  Reads: {[r['index'] for r in reads]}")
            lines.append("")

            if len(dims) != 2:
                if len(dims) == 1:
                    lines.append("  1D loop - check for shift dependencies")
                lines.append("")
                continue

            # Full parallelization analysis
            analysis = analyze_parallelization_options_full(kernel_name, domain, reads, writes, c_code)
            if not analysis:
                continue

            if analysis['is_triangular']:
                tri = analysis['triangular_info']
                lines.append(f"  Triangular bounds: {tri['smaller']} < {tri['larger']}")

            if analysis['self_dependencies']:
                lines.append("  Self-dependencies:")
                for dep in analysis['self_dependencies']:
                    lines.append(f"    - {dep['array']}: write [{dep['write_expr']}], read [{dep['read_expr']}]")
            lines.append("")

            lines.append("  Parallelization Options:")
            valid_count = sum(1 for opt in analysis['options'] if opt['valid'])

            for opt in analysis['options']:
                status = "VALID" if opt['valid'] else "INVALID"
                triton_strat = opt.get('triton_strategy', 'UNKNOWN')
                lines.append(f"    {opt['sequential_dim']}-sequential, {opt['parallel_dim']}-parallel: {status} ({opt['parallelism_type']})")
                lines.append(f"      Triton Strategy: {triton_strat}")

                # Show in-kernel loop safety details
                for detail in opt.get('inkernel_safety_details', []):
                    safety_icon = "✓" if detail['safe'] else "✗"
                    lines.append(f"      {safety_icon} {detail['array']}: {detail['reason']} [{detail['pattern_type']}]")

                for rp in opt['read_patterns']:
                    if rp['explanation']:
                        prefix = "      " if opt['valid'] else "      !! "
                        lines.append(f"{prefix}{rp['explanation']}")

            lines.append("")

            # Summary with Triton strategy recommendation
            if valid_count == 0:
                lines.append("  SUMMARY: No valid parallelization (fully sequential required)")
            elif valid_count == 1:
                valid_opt = [opt for opt in analysis['options'] if opt['valid']][0]
                lines.append(f"  SUMMARY: MUST use {valid_opt['sequential_dim']}-sequential, {valid_opt['parallel_dim']}-parallel")
                lines.append(f"  TRITON: {valid_opt.get('triton_strategy', 'UNKNOWN')}")
            else:
                lines.append("  SUMMARY: Both orderings valid (choose based on performance)")
                # Show recommended strategy for each valid option
                for opt in analysis['options']:
                    if opt['valid']:
                        lines.append(f"    -> {opt['sequential_dim']}-seq, {opt['parallel_dim']}-par: {opt.get('triton_strategy', 'UNKNOWN')}")

            lines.append("")

        lines.append("")

    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Saved parallelization analysis to: {output_file}")
    return output_file


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze loop parallelization dimensions')
    parser.add_argument('--save', '-s', action='store_true', help='Save analysis to results file')
    parser.add_argument('--output', '-o', default='results/parallel_dims_analysis.txt', help='Output file')
    args = parser.parse_args()

    if args.save:
        output_path = os.path.join(os.path.dirname(__file__), args.output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_parallelization_analysis(output_path)
        return

    # Default: test with specific kernels (including s1119 for row-shift pattern)
    test_kernels = ['s1119', 's115', 's118', 's119', 's112', 's116']

    print("="*80)
    print("LOOP PARALLELISM DIMENSION ANALYSIS")
    print("="*80)

    for kernel in test_kernels:
        kernel_file = os.path.join(KERNELS_DIR, f"{kernel}.c")
        if not os.path.exists(kernel_file):
            continue

        print(f"\n{'='*40}")
        print(f"Kernel: {kernel}")
        print(f"{'='*40}")

        pet_output = run_pet(kernel_file)
        if not pet_output:
            print("Failed to run PET")
            continue

        try:
            data = yaml.safe_load(pet_output)
        except Exception as e:
            print(f"Failed to parse: {e}")
            continue

        # Read C code
        with open(kernel_file, 'r') as f:
            c_code = f.read()
        scop_match = re.search(r'#pragma scop\s*(.*?)\s*#pragma endscop', c_code, re.DOTALL)
        c_code = scop_match.group(1).strip() if scop_match else ''

        for i, stmt in enumerate(data.get('statements', [])):
            domain = stmt.get('domain', '')
            reads_raw, writes_raw = extract_accesses(stmt)

            reads = [{'index': r['index']} for r in reads_raw]
            writes = [{'index': w['index']} for w in writes_raw]

            print(f"\nStatement S_{i}:")
            print(f"  Domain: {domain}")

            dims = get_loop_dimensions(domain)
            print(f"  Loop dimensions: {dims}")

            analysis = analyze_dependency_dimensions(domain, reads, writes)
            if analysis:
                print(f"  Sequential dims: {analysis['dep_dims']}")
                print(f"  Parallel dims: {analysis['parallel_dims']}")
                if analysis.get('is_reduction'):
                    print(f"  Reduction: {analysis['is_reduction']}")

                # Use the new analysis that checks for diagonal shift safety
                options_analysis = analyze_parallelization_options_full(kernel, domain, reads, writes, c_code)
                if options_analysis and options_analysis.get('options'):
                    print(f"\n  Parallelization Options:")
                    for opt in options_analysis['options']:
                        status = "VALID" if opt['valid'] else "INVALID"
                        triton_strat = opt.get('triton_strategy', 'UNKNOWN')
                        print(f"    {opt['sequential_dim']}-sequential, {opt['parallel_dim']}-parallel: {status}")
                        print(f"      Triton Strategy: {triton_strat}")

                        # Show in-kernel loop safety details
                        for detail in opt.get('inkernel_safety_details', []):
                            safety_icon = "✓" if detail['safe'] else "✗"
                            print(f"      {safety_icon} {detail['array']}: {detail['reason']} [{detail['pattern_type']}]")
                else:
                    # Fallback to legacy strategy
                    strategy = analyze_parallelization_strategy(kernel, analysis)
                    if strategy:
                        print(f"\n  Pattern: {strategy['pattern']}")
                        print(f"  Strategy: {strategy['triton_strategy'][:500]}...")


if __name__ == "__main__":
    main()
