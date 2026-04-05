#!/usr/bin/env python3
"""
Compute WAR (Write-After-Read) anti-dependencies for Triton parallelization.

This analysis identifies arrays that are both read and written in a loop where:
- Different iterations access overlapping locations
- Parallel execution could cause race conditions

The solution for WAR in Triton: pass a read-only copy of the array to the kernel.
"""

import subprocess
import yaml
import re
import os
from collections import defaultdict
import islpy as isl

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
    # Fix YAML parsing issues
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


def parse_isl_relation(rel_str):
    """Parse ISL relation string to extract array name and index expression."""
    match = re.search(r'->\s*(\w+)\[(.*?)\]', rel_str)
    if match:
        return {
            'array': match.group(1),
            'index': match.group(2),
            'full': rel_str
        }
    return None


def check_war_vs_raw(domain_str, read_info, write_info, schedule_str=None):
    """
    Determine if a read/write pair is WAR (Write-After-Read) or RAW (Read-After-Write).

    WAR occurs when: read location > write location (read from "future", write to "current")
      Example: a[i] = a[i+1] - in parallel, iteration i+1 might write before iteration i reads

    RAW occurs when: read location < write location (read from "past", write to "current")
      Example: a[i] = a[i-1] or a[i] = a[j] where j < i
      This is a sequential dependency handled by loop ordering, NOT a parallel race

    Same location: read location == write location (read-modify-write in same iteration)
      Example: a[i] += ... - safe for parallelization

    Returns True if there IS a WAR conflict (read >= write possible), False if safe (read < write always).
    """
    try:
        domain = isl.Set(domain_str)
        read_map = isl.Map(read_info['full'])
        write_map = isl.Map(write_info['full'])

        # Build the relation: { iteration -> (read_loc, write_loc) }
        # Then check if read_loc >= write_loc is ever possible

        # For each iteration point, get the read and write locations
        # WAR exists if: exists iter in domain such that read(iter) >= write(iter')
        # for some other iter' that could execute in parallel

        # Simpler approach: check if read range and write range overlap,
        # AND if the read is from a "forward" direction

        read_idx = read_info['index']
        write_idx = write_info['index']
        domain_str_clean = domain_str.strip()

        # Extract loop variables from domain
        var_match = re.search(r'S_\d+\[([^\]]+)\]', domain_str_clean)
        if not var_match:
            return True  # Unknown, assume WAR possible

        loop_vars = [v.strip() for v in var_match.group(1).split(',')]

        # Parse the index expressions to understand the access pattern
        # Common patterns:
        # - (i) or (j) - simple variable
        # - (i + k) or (i - k) - offset from variable
        # - (i - j - 1) - difference of variables

        # For 1D arrays with linear indices, check if read < write always

        # Case 1: Single loop variable
        if len(loop_vars) == 1:
            var = loop_vars[0]
            # Check patterns like a[i+k] vs a[i]
            read_offset = parse_linear_offset(read_idx, var)
            write_offset = parse_linear_offset(write_idx, var)

            if read_offset is not None and write_offset is not None:
                # read = var + read_offset, write = var + write_offset
                # WAR vs RAW depends on loop direction AND offset relationship
                #
                # Forward loop (schedule has [(i)]):
                #   - read_offset > write_offset → WAR (reading ahead of write)
                #   - read_offset < write_offset → RAW (reading behind write)
                # Reverse loop (schedule has [(-i)]):
                #   - read_offset < write_offset → WAR (in reverse, writer comes after reader)
                #   - read_offset > write_offset → RAW (in reverse, writer comes before reader)
                #
                # Check if loop is reverse by looking at schedule pattern
                is_reverse = check_loop_reverse(schedule_str, var) if schedule_str else False

                if is_reverse:
                    # Reverse loop: WAR if read_offset < write_offset
                    if read_offset < write_offset:
                        return True   # WAR in reverse loop
                    elif read_offset > write_offset:
                        return False  # RAW in reverse loop
                else:
                    # Forward loop: WAR if read_offset > write_offset
                    if read_offset > write_offset:
                        return True   # WAR in forward loop
                    elif read_offset < write_offset:
                        return False  # RAW in forward loop
                # read_offset == write_offset means same location (handled elsewhere)

        # Case 2: Two loop variables (nested loops like s115, s118, s119)
        elif len(loop_vars) == 2:
            outer_var, inner_var = loop_vars[0], loop_vars[1]

            # NEW: Handle case where both read/write use only ONE loop variable (s256-like)
            # Pattern: read a[(inner - k)], write a[(inner)] where k > 0
            # The outer loop variable is not used in the array index
            # This is RAW not WAR: within the inner loop, earlier iterations write before later ones read
            #
            # Example s256: for i in range(N): for j in range(1, N): a[j] = 1.0 - a[j-1]
            # Read a[(j-1)] = a[(-1 + j)], write a[(j)]
            # This is RAW because iteration j reads from j-1, which was written by iteration j-1

            # Check if indices use only the inner variable (not the outer)
            read_has_outer = re.search(rf'\b{outer_var}\b', read_idx)
            read_has_inner = re.search(rf'\b{inner_var}\b', read_idx)
            write_has_outer = re.search(rf'\b{outer_var}\b', write_idx)
            write_has_inner = re.search(rf'\b{inner_var}\b', write_idx)

            if not read_has_outer and read_has_inner and not write_has_outer and write_has_inner:
                # Both read and write only use the inner loop variable
                # Check if read is at offset < write (RAW) or offset > write (WAR)
                read_offset = parse_linear_offset(read_idx, inner_var)
                write_offset = parse_linear_offset(write_idx, inner_var)

                if read_offset is not None and write_offset is not None:
                    if read_offset < write_offset:
                        return False  # RAW: read from earlier index, not WAR
                    elif read_offset > write_offset:
                        return True   # WAR: read from later index

            # NEW: Handle case where both read/write use only the OUTER loop variable (s257-like)
            # Pattern: read a[(outer - k)], write a[(outer)] where k > 0
            # The inner loop variable is not used in the array index
            # Example s257: for i in range(1, N): for j in range(N): a[i] = aa[j][i] - a[i-1]
            # Read a[(i-1)], write a[(i)]
            # This is RAW because iteration i reads from i-1, which was written by iteration i-1
            elif read_has_outer and not read_has_inner and write_has_outer and not write_has_inner:
                # Both read and write only use the outer loop variable
                # Check if read is at offset < write (RAW) or offset > write (WAR)
                read_offset = parse_linear_offset(read_idx, outer_var)
                write_offset = parse_linear_offset(write_idx, outer_var)

                if read_offset is not None and write_offset is not None:
                    if read_offset < write_offset:
                        return False  # RAW: read from earlier index, not WAR
                    elif read_offset > write_offset:
                        return True   # WAR: read from later index

            # Check if read index < write index is guaranteed by domain constraints
            # Common pattern: read uses outer var, write uses inner var, outer < inner

            # s115: read a[(j)], write a[(i)], domain has j < i
            # s118: read a[(i - j - 1)], write a[(i)], for j >= 0: i-j-1 < i
            # s119: read aa[(-1 + i), (-1 + j)], write aa[(i), (j)] - 2D diagonal backward shift

            # NEW: Handle 2D diagonal backward shift pattern (s119)
            # Pattern: read aa[(outer - k1), (inner - k2)], write aa[(outer), (inner)]
            # where k1 > 0 and k2 > 0 (or k1 >= 0, k2 > 0 or vice versa with at least one > 0)
            # This is ALWAYS RAW (reading from earlier diagonal), never WAR
            # Because when either dimension is processed sequentially, the "earlier" iteration completes first

            # Check for 2D index patterns (comma-separated)
            if ',' in read_idx and ',' in write_idx:
                # Parse 2D indices: "(-1 + i), (-1 + j)" -> ["-1 + i", "-1 + j"]
                read_parts = [p.strip().strip('()') for p in read_idx.split(',')]
                write_parts = [p.strip().strip('()') for p in write_idx.split(',')]

                if len(read_parts) == 2 and len(write_parts) == 2:
                    # Check if write is just the loop variables: (outer), (inner)
                    write_is_vars = (
                        (write_parts[0].strip() == outer_var and write_parts[1].strip() == inner_var) or
                        (write_parts[0].strip() == inner_var and write_parts[1].strip() == outer_var)
                    )

                    if write_is_vars:
                        # Check if read has backward offsets on BOTH dimensions
                        # Pattern: (-k + var) or (var - k) where k > 0
                        def get_backward_offset(expr, var):
                            """Return offset if expr = var - offset (backward), None otherwise."""
                            expr = expr.strip()
                            # Pattern: -k + var (e.g., "-1 + i")
                            m = re.match(rf'-(\d+)\s*\+\s*{var}$', expr)
                            if m:
                                return int(m.group(1))
                            # Pattern: var - k (e.g., "i - 1")
                            m = re.match(rf'{var}\s*-\s*(\d+)$', expr)
                            if m:
                                return int(m.group(1))
                            # Just the variable (offset = 0)
                            if expr == var:
                                return 0
                            return None

                        # Get offsets for both dimensions
                        if write_parts[0].strip() == outer_var:
                            # Write order is (outer, inner)
                            read_outer_offset = get_backward_offset(read_parts[0], outer_var)
                            read_inner_offset = get_backward_offset(read_parts[1], inner_var)
                        else:
                            # Write order is (inner, outer) - swap
                            read_outer_offset = get_backward_offset(read_parts[1], outer_var)
                            read_inner_offset = get_backward_offset(read_parts[0], inner_var)

                        # If both dimensions have backward (or zero) offsets, and at least one is backward
                        # then it's RAW not WAR (reading from earlier iteration)
                        if read_outer_offset is not None and read_inner_offset is not None:
                            if read_outer_offset >= 0 and read_inner_offset >= 0:
                                if read_outer_offset > 0 or read_inner_offset > 0:
                                    # Diagonal backward shift - this is RAW, not WAR
                                    # When either dimension is sequential, the earlier iteration completes first
                                    return False  # RAW not WAR

            # Use ISL to check: for all points in domain, is read_loc < write_loc?
            # Build: { S_0[vars] : read_loc >= write_loc } intersect domain
            # If empty, then read < write always (RAW, not WAR)

            try:
                # Create a set where read >= write
                # This requires building the constraint from the access maps

                # Get the range (accessed locations) for read and write
                read_range = read_map.range()
                write_range = write_map.range()

                # Check if they can overlap at all
                if read_range.intersect(write_range).is_empty():
                    return False  # No overlap, no conflict

                # For more precise analysis, check the actual index relationship
                # within the domain constraints

                # Heuristic: check common patterns

                # Pattern: read a[(outer)], write a[(inner)], domain has outer < inner
                if f'({outer_var})' in read_idx or read_idx.strip() == f'({outer_var})':
                    if f'({inner_var})' in write_idx or write_idx.strip() == f'({inner_var})':
                        # Check for outer < inner constraint
                        if re.search(rf'{outer_var}\s*<\s*{inner_var}', domain_str_clean) or \
                           re.search(rf'{inner_var}\s*>\s*{outer_var}', domain_str_clean):
                            return False  # read = outer < inner = write, RAW not WAR

                # Pattern: read a[(inner - outer - k)], write a[(inner)]
                # For k >= 0 and outer >= 0: inner - outer - k < inner
                offset_match = re.match(rf'\(\s*{inner_var}\s*-\s*{outer_var}\s*-\s*(\d+)\s*\)', read_idx)
                if offset_match:
                    if f'({inner_var})' in write_idx or write_idx.strip() == f'({inner_var})':
                        # read = inner - outer - k, write = inner
                        # Since outer >= 0 (from domain), read < write
                        return False  # RAW not WAR

                # Pattern: read a[(-k + inner - outer)] (alternative form from PET)
                # e.g., (-1 + i - j) means i - j - 1
                offset_match_alt = re.match(rf'\(\s*-?\s*(\d+)\s*\+\s*{outer_var}\s*-\s*{inner_var}\s*\)', read_idx)
                if offset_match_alt:
                    if f'({outer_var})' in write_idx or write_idx.strip() == f'({outer_var})':
                        # read = outer - inner + const, write = outer
                        # Since inner >= 0 (from domain), read < write (when const <= 0)
                        return False  # RAW not WAR

                # Pattern: read a[(outer - inner - k)] or a[(outer - k - inner)]
                # where outer is the iteration variable for writes
                offset_match2 = re.match(rf'\(\s*{outer_var}\s*-\s*{inner_var}\s*-\s*(\d+)\s*\)', read_idx)
                if offset_match2:
                    if f'({outer_var})' in write_idx or write_idx.strip() == f'({outer_var})':
                        return False  # RAW not WAR

                # Pattern: general form where read index is less than write index
                # For any expression X - Y where Y >= 0, X - Y < X
                # Check if read = f(outer, inner) and write = outer, and read < write
                if f'({outer_var})' in write_idx or write_idx.strip() == f'({outer_var})':
                    # Write is to outer variable
                    # Check if read is outer - something_positive
                    if re.search(rf'{outer_var}\s*-\s*{inner_var}', read_idx) or \
                       re.search(rf'-\s*{inner_var}\s*\+\s*{outer_var}', read_idx) or \
                       re.search(rf'{outer_var}\s*-\s*\d+\s*-\s*{inner_var}', read_idx) or \
                       re.search(rf'-\s*\d+\s*\+\s*{outer_var}\s*-\s*{inner_var}', read_idx):
                        # read has form (outer - inner - k) or similar
                        # Since inner >= 0, read < write
                        return False  # RAW not WAR

            except Exception:
                pass

        # If we can't prove it's safe, check using ISL relation composition
        try:
            write_inv = write_map.reverse()
            conflict_relation = read_map.apply_range(write_inv)
            conflict_relation = conflict_relation.intersect_domain(domain)
            conflict_relation = conflict_relation.intersect_range(domain)

            if conflict_relation.is_empty():
                return False  # No conflict at all

            # Check for cross-iteration conflicts only
            identity = isl.Map.identity(domain.get_space().map_from_set())
            cross_iter = conflict_relation.subtract(identity)

            if cross_iter.is_empty():
                return False  # Only same-iteration access

        except Exception:
            pass

        return True  # Assume WAR possible if we can't prove otherwise

    except Exception:
        return True  # Assume WAR possible on error


def check_loop_reverse(schedule_str, var):
    """
    Check if a loop variable iterates in reverse direction.

    PET encodes reverse loops with negated schedule coefficients:
    - Forward: schedule: "L_0[{ S_0[i] -> [(i)] }]"
    - Reverse: schedule: "L_0[{ S_0[i] -> [(-i)] }]"

    Returns True if the loop is reverse (descending), False if forward (ascending).
    """
    if not schedule_str:
        return False  # Assume forward if no schedule info

    # Look for pattern like [(-var)] indicating reverse iteration
    reverse_pattern = rf'\[\s*\(\s*-\s*{var}\s*\)\s*\]'
    if re.search(reverse_pattern, schedule_str):
        return True

    # Also check for pattern like [(-1 * var)] or similar
    reverse_pattern2 = rf'\[\s*\(\s*-1\s*\*\s*{var}\s*\)\s*\]'
    if re.search(reverse_pattern2, schedule_str):
        return True

    return False


# Global variable to store schedule info during analysis
_current_schedule = None


def parse_linear_offset(index_str, var):
    """
    Parse an index expression to extract the offset from a variable.
    Returns the offset k where index = var + k, or None if not a simple linear form.

    Examples:
      (i) -> 0
      (i + 1) -> 1
      (i - 1) -> -1
      (1 + i) -> 1
    """
    index_str = index_str.strip()

    # Remove outer parentheses
    if index_str.startswith('(') and index_str.endswith(')'):
        index_str = index_str[1:-1].strip()

    # Simple variable: (i) -> offset 0
    if index_str == var:
        return 0

    # Pattern: var + k or var - k
    match = re.match(rf'{var}\s*([+-])\s*(\d+)$', index_str)
    if match:
        sign = 1 if match.group(1) == '+' else -1
        return sign * int(match.group(2))

    # Pattern: k + var
    match = re.match(rf'(\d+)\s*\+\s*{var}$', index_str)
    if match:
        return int(match.group(1))

    # Pattern: -k + var or - k + var (unlikely but possible)
    match = re.match(rf'-\s*(\d+)\s*\+\s*{var}$', index_str)
    if match:
        return -int(match.group(1))

    return None


def compute_war_dependencies(domain_str, reads, writes, schedule_str=None):
    """
    Compute WAR (Write-After-Read) anti-dependencies.

    WAR dependency exists when:
    - Iteration i reads from location L
    - Iteration i' writes to location L
    - i != i' (different iterations)

    In parallel execution, if the write at i' happens before read at i completes,
    iteration i reads the wrong (modified) value.

    Returns list of WAR dependencies with analysis.
    """
    war_deps = []

    try:
        domain = isl.Set(domain_str)
    except:
        return []

    # Group accesses by array
    array_reads = defaultdict(list)
    array_writes = defaultdict(list)

    for read in reads:
        info = parse_isl_relation(read['index'])
        if info:
            array_reads[info['array']].append(info)

    for write in writes:
        info = parse_isl_relation(write['index'])
        if info:
            array_writes[info['array']].append(info)

    # Find arrays that are both read and written
    shared_arrays = set(array_reads.keys()) & set(array_writes.keys())

    for arr in shared_arrays:
        for r_info in array_reads[arr]:
            for w_info in array_writes[arr]:
                read_idx = r_info['index']
                write_idx = w_info['index']

                # Same index means read-modify-write in same iteration - no WAR
                if read_idx == write_idx:
                    continue

                # Check if this is WAR (read ahead) or RAW (read behind)
                # Only WAR needs the copy pattern; RAW is handled by loop ordering
                if not check_war_vs_raw(domain_str, r_info, w_info, schedule_str):
                    continue  # RAW dependency, not WAR - no copy needed

                # Check if different iterations can access the same location
                try:
                    read_map = isl.Map(r_info['full'])
                    write_map = isl.Map(w_info['full'])

                    # For WAR: find { reader -> writer : Read(reader) = Write(writer) }
                    # This means: read_map(reader) intersects write_map(writer)
                    # Compute: write_map^-1 . read_map gives { reader -> writer } where they hit same location

                    write_inv = write_map.reverse()
                    conflict_relation = read_map.apply_range(write_inv)

                    # Intersect with domain
                    conflict_relation = conflict_relation.intersect_domain(domain)
                    conflict_relation = conflict_relation.intersect_range(domain)

                    if conflict_relation.is_empty():
                        continue

                    # Check if there are cross-iteration conflicts (not self-conflicts)
                    identity = isl.Map.identity(domain.get_space().map_from_set())
                    cross_iter_conflicts = conflict_relation.subtract(identity)

                    if cross_iter_conflicts.is_empty():
                        # Only self-conflicts (same iteration) - not a parallelization issue
                        continue

                    # Determine the conflict pattern
                    conflict_str = str(cross_iter_conflicts).strip()

                    # Extract the relationship
                    # e.g., { S_0[i] -> S_0[i'] : i = i' + 1 } means iteration i reads what i-1 writes

                    war_dep = {
                        'array': arr,
                        'read_pattern': r_info['full'],
                        'write_pattern': w_info['full'],
                        'conflict_relation': conflict_str,
                        'read_index': r_info['index'],
                        'write_index': w_info['index'],
                        'description': f"Read {arr}[{r_info['index']}] may conflict with Write {arr}[{w_info['index']}] from different iteration"
                    }

                    # Analyze the offset pattern
                    # Simple pattern matching for common cases
                    if re.match(r'\(\d*\s*\+?\s*i\)', read_idx) and re.match(r'\(\d*\s*\+?\s*i\)', write_idx):
                        # Both are linear in i
                        war_dep['pattern'] = 'linear-offset'
                        war_dep['solution'] = 'Use read-only copy: load from copy, store to original'
                    else:
                        war_dep['pattern'] = 'complex'
                        war_dep['solution'] = 'Use read-only copy for safe parallelization'

                    war_deps.append(war_dep)

                except Exception as e:
                    # Fallback: if indices differ but same array is read/written
                    # Still check if it's RAW (not WAR)
                    if not check_war_vs_raw(domain_str, r_info, w_info, schedule_str):
                        continue

                    war_dep = {
                        'array': arr,
                        'read_pattern': r_info['full'],
                        'write_pattern': w_info['full'],
                        'read_index': r_info['index'],
                        'write_index': w_info['index'],
                        'pattern': 'potential-war',
                        'description': f"Potential WAR: Read {arr}[{r_info['index']}], Write {arr}[{w_info['index']}]",
                        'solution': 'Use read-only copy to avoid race condition',
                        'error': str(e)
                    }
                    war_deps.append(war_dep)

    return war_deps


def parse_linear_index_offset(index_str, var='i'):
    """
    Parse an index expression to extract the offset from a loop variable.
    Handles various ISL output formats.

    Examples:
      (i) -> 0
      (1 + i) -> 1
      (i + 1) -> 1
      (-1 + i) -> -1
      (i - 1) -> -1
      i -> 0

    Returns offset as int, or None if not a simple linear form.
    """
    index_str = index_str.strip()

    # Remove outer parentheses
    while index_str.startswith('(') and index_str.endswith(')'):
        index_str = index_str[1:-1].strip()

    # Simple variable: i -> offset 0
    if index_str == var:
        return 0

    # Pattern: k + var (e.g., "1 + i")
    match = re.match(rf'(-?\d+)\s*\+\s*{var}$', index_str)
    if match:
        return int(match.group(1))

    # Pattern: var + k (e.g., "i + 1")
    match = re.match(rf'{var}\s*\+\s*(\d+)$', index_str)
    if match:
        return int(match.group(1))

    # Pattern: var - k (e.g., "i - 1")
    match = re.match(rf'{var}\s*-\s*(\d+)$', index_str)
    if match:
        return -int(match.group(1))

    # Pattern: -k + var (e.g., "-1 + i")
    match = re.match(rf'-(\d+)\s*\+\s*{var}$', index_str)
    if match:
        return -int(match.group(1))

    return None


def _tokenize_isl_schedule(s):
    """Tokenize an ISL schedule string into a list of tokens.

    Token types: '{', '}', '[', ']', ':', ',', quoted strings ("..."), identifiers.
    Inside quoted strings, all characters (including braces/brackets) are literal.
    """
    tokens = []
    i = 0
    while i < len(s):
        c = s[i]
        if c in ' \t\n\r':
            i += 1
            continue
        if c in '{}[]:,':
            tokens.append(c)
            i += 1
            continue
        if c == '"':
            j = i + 1
            while j < len(s) and s[j] != '"':
                j += 1
            tokens.append(s[i:j + 1])  # include quotes
            i = j + 1
            continue
        # Identifier (key names like domain, child, sequence, filter, schedule, set)
        j = i
        while j < len(s) and s[j] not in ' \t\n\r{}[]:,"':
            j += 1
        tokens.append(s[i:j])
        i = j
    return tokens


def _parse_isl_schedule(tokens, pos):
    """Recursive descent parser for ISL schedule tokens. Returns (value, new_pos)."""
    if pos >= len(tokens):
        return None, pos

    tok = tokens[pos]

    if tok == '{':
        # Parse dict: { key: value, key: value, ... }
        result = {}
        pos += 1
        while pos < len(tokens) and tokens[pos] != '}':
            key = tokens[pos]
            if key.startswith('"'):
                key = key[1:-1]
            pos += 1
            if pos < len(tokens) and tokens[pos] == ':':
                pos += 1
            val, pos = _parse_isl_schedule(tokens, pos)
            result[key] = val
            if pos < len(tokens) and tokens[pos] == ',':
                pos += 1
        if pos < len(tokens):
            pos += 1  # skip '}'
        return result, pos

    elif tok == '[':
        # Parse array: [ value, value, ... ]
        result = []
        pos += 1
        while pos < len(tokens) and tokens[pos] != ']':
            val, pos = _parse_isl_schedule(tokens, pos)
            result.append(val)
            if pos < len(tokens) and tokens[pos] == ',':
                pos += 1
        if pos < len(tokens):
            pos += 1  # skip ']'
        return result, pos

    elif tok.startswith('"'):
        return tok[1:-1], pos + 1

    else:
        return tok, pos + 1


def parse_isl_schedule(schedule_str):
    """Parse an ISL schedule string into a Python dict tree."""
    tokens = _tokenize_isl_schedule(schedule_str)
    result, _ = _parse_isl_schedule(tokens, 0)
    return result


def get_stmt_innermost_loops(tree, current_loop=None):
    """Map each statement to its innermost loop label from the parsed schedule tree.

    Traverses the schedule tree, tracking the current loop label (L_0, L_1, ...).
    When a filter is encountered, statements are mapped to the current loop.
    Deeper filters override shallower ones, so the innermost loop wins.

    Returns dict like {'S_2': 'L_1', 'S_5': 'L_2', 'S_0': None, ...}
    """
    result = {}

    if not isinstance(tree, dict):
        return result

    # Check for schedule key — update current loop label
    if 'schedule' in tree:
        sched_val = tree['schedule']
        if isinstance(sched_val, str):
            loop_match = re.match(r'(L_\d+)', sched_val)
            if loop_match:
                current_loop = loop_match.group(1)

    # Check for filter key — map statements to current loop
    if 'filter' in tree:
        filter_val = tree['filter']
        if isinstance(filter_val, str):
            for s in re.findall(r'(S_\d+)', filter_val):
                result[s] = current_loop

    # Recurse into child, sequence, and set
    if 'child' in tree:
        result.update(get_stmt_innermost_loops(tree['child'], current_loop))
    if 'sequence' in tree and isinstance(tree['sequence'], list):
        for item in tree['sequence']:
            result.update(get_stmt_innermost_loops(item, current_loop))
    if 'set' in tree and isinstance(tree['set'], list):
        for item in tree['set']:
            result.update(get_stmt_innermost_loops(item, current_loop))

    return result


def compute_inter_stmt_war(statements, schedule_str):
    """
    Compute WAR dependencies between different statements in a multi-statement kernel.

    For patterns where multiple statements read/write the same array
    with different offsets (s116, s211, s1213).

    WAR (Write-After-Read) occurs when:
    - Statement A reads arr[i+k] (forward read, k > 0)
    - Statement B writes arr[i]
    - In parallel execution, B's write at iteration i+k may happen before A's read

    This requires passing a read-only copy of the array.
    """
    war_deps = []

    if len(statements) <= 1:
        return war_deps

    # Parse schedule to determine which statements share the same innermost loop
    stmt_loops = {}
    if schedule_str:
        try:
            tree = parse_isl_schedule(schedule_str)
            stmt_loops = get_stmt_innermost_loops(tree)
        except Exception:
            pass  # Fall back to current behavior (no filtering)

    # Collect all read/write patterns across statements
    all_reads = []
    all_writes = []

    for stmt_idx, stmt in enumerate(statements):
        for read in stmt.get('reads', []):
            info = parse_isl_relation(read)
            if info:
                info['stmt'] = f'S_{stmt_idx}'
                info['stmt_idx'] = stmt_idx
                # Parse offset
                info['offset'] = parse_linear_index_offset(info['index'])
                all_reads.append(info)

        for write in stmt.get('writes', []):
            info = parse_isl_relation(write)
            if info:
                info['stmt'] = f'S_{stmt_idx}'
                info['stmt_idx'] = stmt_idx
                # Parse offset
                info['offset'] = parse_linear_index_offset(info['index'])
                all_writes.append(info)

    # Check for WAR conflicts across statements at DIFFERENT iterations
    # WAR occurs when: read_offset > write_offset (forward read)
    # This means iteration i reads arr[i+k] while iteration i+k writes arr[i+k]

    for r_info in all_reads:
        for w_info in all_writes:
            if r_info['array'] != w_info['array']:
                continue

            # Skip same-statement same-index (handled by intra-statement analysis)
            if r_info['stmt'] == w_info['stmt'] and r_info['index'] == w_info['index']:
                continue

            read_idx = r_info['index']
            write_idx = w_info['index']
            read_offset = r_info['offset']
            write_offset = w_info['offset']

            # If we can parse both offsets, check for WAR
            if read_offset is not None and write_offset is not None:
                # WAR condition: read_offset > write_offset (forward read)
                # Example: read a[i+1], write a[i] -> read_offset=1, write_offset=0
                # Iteration i reads a[i+1], iteration i+1 writes a[i+1] -> race!

                if read_offset > write_offset:
                    # Skip WAR between statements in DIFFERENT innermost loops
                    # (they execute sequentially, so no race condition)
                    r_loop = stmt_loops.get(r_info['stmt'])
                    w_loop = stmt_loops.get(w_info['stmt'])
                    if r_loop is not None and w_loop is not None and r_loop != w_loop:
                        continue

                    offset_diff = read_offset - write_offset
                    war_dep = {
                        'array': r_info['array'],
                        'read_stmt': r_info['stmt'],
                        'write_stmt': w_info['stmt'],
                        'read_pattern': r_info['full'],
                        'write_pattern': w_info['full'],
                        'read_offset': read_offset,
                        'write_offset': write_offset,
                        'pattern': 'cross-stmt-forward-read',
                        'offset_diff': offset_diff,
                        'description': f"{r_info['stmt']} reads {r_info['array']}[i{read_offset:+d}], {w_info['stmt']} writes {w_info['array']}[i{write_offset:+d}] - forward read WAR",
                        'solution': f'Use read-only copy of {r_info["array"]}: {r_info["array"]}_copy = {r_info["array"]}.clone()'
                    }
                    war_deps.append(war_dep)

    return war_deps


def analyze_war_from_c_code(kernel_file):
    """
    Fallback WAR analysis directly from C code when PET fails.

    Detects patterns like:
    - a[i] = a[i + inc] + ...  (read ahead with variable offset)
    - a[i] = a[i + k] + ...    (read ahead with constant offset)

    where the read is ahead of the write (WAR dependency).
    """
    with open(kernel_file, 'r') as f:
        content = f.read()

    # Extract scop region
    scop_match = re.search(r'#pragma scop\s*(.*?)\s*#pragma endscop', content, re.DOTALL)
    if not scop_match:
        return None

    scop_code = scop_match.group(1)

    # Find loop variable
    loop_match = re.search(r'for\s*\(\s*(?:int\s+)?(\w+)\s*=', scop_code)
    if not loop_match:
        return None

    loop_var = loop_match.group(1)

    # Find array write: array[loop_var] = ...
    # Pattern: word[i] =
    write_pattern = rf'(\w+)\s*\[\s*{loop_var}\s*\]\s*='
    write_match = re.search(write_pattern, scop_code)
    if not write_match:
        return None

    write_array = write_match.group(1)

    # Find read from same array with forward offset: array[loop_var + offset]
    # Pattern: word[i + something] where something is variable or constant
    read_pattern = rf'{write_array}\s*\[\s*{loop_var}\s*\+\s*(\w+)\s*\]'
    read_match = re.search(read_pattern, scop_code)

    if not read_match:
        return None

    offset_expr = read_match.group(1)

    # Check if offset is positive (constant) or assume positive for variable
    try:
        offset_val = int(offset_expr)
        if offset_val <= 0:
            return None  # Not a forward read
    except ValueError:
        # Variable offset - assume positive (like 'inc' in s175)
        pass

    # This is a WAR dependency!
    return {
        'statements': [],
        'war_dependencies': [{
            'array': write_array,
            'read_pattern': f'{write_array}[{loop_var} + {offset_expr}]',
            'write_pattern': f'{write_array}[{loop_var}]',
            'read_index': f'({loop_var} + {offset_expr})',
            'write_index': f'({loop_var})',
            'pattern': 'forward-read-variable-offset',
            'description': f"Read {write_array}[{loop_var}+{offset_expr}] before Write {write_array}[{loop_var}] - forward read WAR",
            'solution': f'Use read-only copy: {write_array}_copy = {write_array}.clone()'
        }],
        'arrays_needing_copy': [write_array],
        'parallelization_safe': False
    }


def analyze_kernel_war(kernel_file):
    """Analyze a kernel for WAR dependencies."""
    pet_output = run_pet(kernel_file)
    if not pet_output:
        # Fallback to C code analysis
        return analyze_war_from_c_code(kernel_file)

    try:
        data = yaml.safe_load(pet_output)
    except:
        # Fallback to C code analysis
        return analyze_war_from_c_code(kernel_file)

    results = {
        'statements': [],
        'war_dependencies': [],
        'arrays_needing_copy': set(),
        'parallelization_safe': True
    }

    # Get schedule for loop direction analysis
    schedule_str = data.get('schedule', '')

    # First pass: collect statement info
    stmt_infos = []
    for stmt in data.get('statements', []):
        domain = stmt.get('domain', '')
        reads, writes = extract_accesses(stmt)

        stmt_info = {
            'domain': domain,
            'reads': [r['index'] for r in reads],
            'writes': [w['index'] for w in writes]
        }
        results['statements'].append(stmt_info)
        stmt_infos.append(stmt_info)

        # Compute WAR within this statement
        war_deps = compute_war_dependencies(domain, reads, writes, schedule_str)
        results['war_dependencies'].extend(war_deps)

        for dep in war_deps:
            results['arrays_needing_copy'].add(dep['array'])

    # Second pass: inter-statement WAR for multi-statement kernels
    if len(stmt_infos) > 1:
        inter_war = compute_inter_stmt_war(stmt_infos, data.get('schedule', ''))
        results['war_dependencies'].extend(inter_war)

        for dep in inter_war:
            results['arrays_needing_copy'].add(dep['array'])

    results['parallelization_safe'] = len(results['war_dependencies']) == 0
    results['arrays_needing_copy'] = list(results['arrays_needing_copy'])

    return results


def format_war_analysis_for_prompt(kernel_name, war_result):
    """
    Format WAR analysis results for inclusion in LLM prompt.

    If enhanced with loop-level scoping (from LLVM direction vectors),
    provides per-loop-level WAR guidance so the LLM can choose the
    safest parallelization dimension.
    """
    if not war_result:
        return ""

    lines = []
    lines.append(f"WAR (Write-After-Read) Anti-Dependency Analysis for {kernel_name}:")
    lines.append("")

    if war_result['parallelization_safe']:
        lines.append("✓ No WAR dependencies detected - safe for naive parallelization")
    else:
        loop_scoping = war_result.get('loop_level_scoping')

        if loop_scoping:
            # Enhanced format with per-loop-level WAR scoping
            loop_vars = war_result.get('loop_vars', [])
            lines.append("⚠ WAR DEPENDENCIES DETECTED (with loop-level scoping):")
            lines.append("")
            for arr in war_result['arrays_needing_copy']:
                scoping = loop_scoping.get(arr, {})
                carried = scoping.get('carried_by_loops', loop_vars)
                safe = scoping.get('safe_to_parallelize_loops', [])
                dvs = scoping.get('direction_vectors', [])

                lines.append(f"Array `{arr}`:")
                if dvs:
                    # Show a representative direction vector
                    dv_strs = [f"[{' '.join(dv)}]" for dv in dvs[:3]]
                    lines.append(f"  Direction vectors: {', '.join(dv_strs)}")
                seq_ctx = scoping.get('sequential_context_loops', [])
                if carried:
                    lines.append(f"  WAR carried by loop(s): {', '.join(carried)}")
                for var in loop_vars:
                    if var in safe:
                        lines.append(f"  - Parallelizing `{var}`: SAFE (no copy needed for {arr})")
                    elif var in carried:
                        lines.append(f"  - Parallelizing `{var}`: REQUIRES copy ({arr}_copy = {arr}.clone())")
                    elif var in seq_ctx:
                        lines.append(f"  - Loop `{var}`: sequential context (not analyzed for WAR)")
                lines.append("")
        else:
            # Original format without scoping
            lines.append("⚠ WAR RACE CONDITION DETECTED!")
            lines.append("")
            lines.append("Arrays requiring read-only copy for safe parallelization:")
            for arr in war_result['arrays_needing_copy']:
                lines.append(f"  - {arr}")
            lines.append("")

        lines.append("WAR Dependency Details:")
        for i, dep in enumerate(war_result['war_dependencies'], 1):
            lines.append(f"  {i}. {dep['description']}")
            if 'solution' in dep:
                lines.append(f"     Solution: {dep['solution']}")
        lines.append("")

        if not loop_scoping:
            # Only show blanket copy pattern when no scoping available
            lines.append("REQUIRED TRITON PATTERN:")
            lines.append("```python")
            lines.append("# In wrapper function - create read-only copy:")
            for arr in war_result['arrays_needing_copy']:
                lines.append(f"{arr}_readonly = {arr}.clone()  # Read-only copy")
            lines.append("")
            lines.append("# In kernel - load from copy, store to original:")
            for arr in war_result['arrays_needing_copy']:
                lines.append(f"# Load: tl.load({arr}_readonly_ptr + offsets, ...)")
                lines.append(f"# Store: tl.store({arr}_ptr + offsets, result, ...)")
            lines.append("```")

    return "\n".join(lines)


def main():
    kernels = []
    target_nums = set(range(0, 120))
    target_nums.update([1111, 1112, 1113, 1115, 1119, 1161])

    for f in os.listdir(KERNELS_DIR):
        if f.endswith('.c'):
            name = f[:-2]
            if re.match(r's\d{3,4}$', name):
                num_match = re.search(r's(\d+)', name)
                if num_match and int(num_match.group(1)) in target_nums:
                    kernels.append(name)

    kernels.sort()

    print("="*80)
    print("WAR (WRITE-AFTER-READ) ANTI-DEPENDENCY ANALYSIS")
    print("="*80)

    all_results = {}
    for kernel in kernels:
        kernel_file = os.path.join(KERNELS_DIR, f"{kernel}.c")
        if not os.path.exists(kernel_file):
            continue

        result = analyze_kernel_war(kernel_file)
        if not result:
            continue

        all_results[kernel] = result

        print(f"\n{'='*40}")
        print(f"Kernel: {kernel}")
        print(f"{'='*40}")

        if result['parallelization_safe']:
            print("\n✓ No WAR dependencies - safe for naive parallelization")
        else:
            print(f"\n⚠ WAR RACE CONDITION DETECTED!")
            print(f"Arrays needing read-only copy: {result['arrays_needing_copy']}")
            print("\nWAR Dependencies:")
            for dep in result['war_dependencies']:
                print(f"  - {dep['description']}")
                if 'solution' in dep:
                    print(f"    Solution: {dep['solution']}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    safe_kernels = [k for k, r in all_results.items() if r['parallelization_safe']]
    unsafe_kernels = [k for k, r in all_results.items() if not r['parallelization_safe']]

    print(f"\nSafe for naive parallelization ({len(safe_kernels)}):")
    print(f"  {', '.join(sorted(safe_kernels))}")

    print(f"\nRequire read-only copy pattern ({len(unsafe_kernels)}):")
    for k in sorted(unsafe_kernels):
        arrays = all_results[k]['arrays_needing_copy']
        print(f"  {k}: copy {arrays}")


if __name__ == "__main__":
    main()
