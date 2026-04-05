#!/usr/bin/env python3
"""
Compute flow dependencies from PET output using ISL for proper dependence analysis.
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

    # Fix YAML parsing issue: '=' is a reserved YAML tag
    # Quote problematic operation values
    output = result.stdout
    output = re.sub(r'operation: =\s*$', 'operation: "="', output, flags=re.MULTILINE)
    output = re.sub(r'operation: \+\s*$', 'operation: "+"', output, flags=re.MULTILINE)
    output = re.sub(r'operation: -\s*$', 'operation: "-"', output, flags=re.MULTILINE)
    output = re.sub(r'operation: \*\s*$', 'operation: "*"', output, flags=re.MULTILINE)
    output = re.sub(r'operation: /\s*$', 'operation: "/"', output, flags=re.MULTILINE)
    output = re.sub(r'operation: %\s*$', 'operation: "%"', output, flags=re.MULTILINE)
    output = re.sub(r'operation: &&\s*$', 'operation: "&&"', output, flags=re.MULTILINE)
    output = re.sub(r'operation: \|\|\s*$', 'operation: "||"', output, flags=re.MULTILINE)
    output = re.sub(r'operation: <\s*$', 'operation: "<"', output, flags=re.MULTILINE)
    output = re.sub(r'operation: >\s*$', 'operation: ">"', output, flags=re.MULTILINE)
    output = re.sub(r'operation: <=\s*$', 'operation: "<="', output, flags=re.MULTILINE)
    output = re.sub(r'operation: >=\s*$', 'operation: ">="', output, flags=re.MULTILINE)
    output = re.sub(r'operation: ==\s*$', 'operation: "=="', output, flags=re.MULTILINE)
    output = re.sub(r'operation: !=\s*$', 'operation: "!="', output, flags=re.MULTILINE)

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
    # Format: { S_0[i, j] -> array[(idx_expr)] }
    match = re.search(r'->\s*(\w+)\[(.*?)\]', rel_str)
    if match:
        return {
            'array': match.group(1),
            'index': match.group(2),
            'full': rel_str
        }
    return None


def extract_schedule_map(schedule_str, domain, stmt_id=0):
    """
    Extract the schedule map from PET's schedule string.
    Returns a map from iteration space to schedule space.

    For multi-statement kernels with sequence, includes statement order dimension.
    """
    if not schedule_str:
        return None

    # Look for schedule pattern like: schedule: "L_0[{ S_0[i] -> [(expr)] }]"
    # Can be nested for multi-dimensional loops
    match = re.search(r'S_\d+\[[^\]]+\]\s*->\s*\[([^\]]+)\]', schedule_str)
    if match:
        sched_expr = match.group(1).strip()
        # Build a proper ISL map string
        domain_str = str(domain)
        # Extract the iteration variables from domain
        var_match = re.search(r'S_\d+\[([^\]]+)\]', domain_str)
        if var_match:
            vars_str = var_match.group(1)
            try:
                sched_map = isl.Map(f"{{ S_0[{vars_str}] -> [{sched_expr}] }}")
                return sched_map
            except:
                pass
    return None


def parse_sequence_order(schedule_str):
    """
    Parse sequence ordering from PET's schedule string.
    Returns a dict mapping statement name (e.g., 'S_0') to its order in the sequence.
    """
    if not schedule_str:
        return {}

    # Look for sequence pattern in the schedule string
    # The pattern can span multiple lines and may be nested
    # Format: sequence: [ { filter: "{ S_0[i] }" }, { filter: "{ S_1[i] }" }, ... ]

    # Find all filter patterns that define statement ordering
    # These appear as: filter: "{ S_N[...] }"
    filter_matches = re.findall(r'filter:\s*"{\s*(S_\d+)', schedule_str)

    if not filter_matches:
        return {}

    stmt_order = {}
    for i, stmt_name in enumerate(filter_matches):
        stmt_order[stmt_name] = i

    return stmt_order


def compute_inter_stmt_flow_deps(statements, schedule_str):
    """
    Compute flow dependencies between different statements in a multi-statement kernel.

    For flow dependence (RAW): write in S_w -> read in S_r within SAME iteration
    where they access the same memory location and S_w executes before S_r.
    """
    deps = []

    # Parse statement ordering from sequence
    stmt_order = parse_sequence_order(schedule_str)
    if not stmt_order:
        return deps  # No sequence info, can't determine inter-statement ordering

    num_stmts = len(statements)
    if num_stmts <= 1:
        return deps

    # For each pair of statements (writer_stmt, reader_stmt)
    for w_idx, writer_stmt in enumerate(statements):
        writer_name = f"S_{w_idx}"
        writer_order = stmt_order.get(writer_name, w_idx)

        for write_access in writer_stmt.get('writes', []):
            w_info = parse_isl_relation(write_access)
            if not w_info:
                continue

            for r_idx, reader_stmt in enumerate(statements):
                if w_idx == r_idx:
                    continue  # Same statement - handled by intra-statement analysis

                reader_name = f"S_{r_idx}"
                reader_order = stmt_order.get(reader_name, r_idx)

                for read_access in reader_stmt.get('reads', []):
                    r_info = parse_isl_relation(read_access)
                    if not r_info:
                        continue

                    if w_info['array'] != r_info['array']:
                        continue

                    # Same array - check if they access the same location
                    # WITHIN THE SAME ITERATION
                    try:
                        # For inter-statement deps within same iteration:
                        # S_w[i] writes array[f(i)]
                        # S_r[i] reads array[g(i)]  (same i since same iteration)
                        # Flow dep exists when f(i) = g(i) for some i in both domains

                        # Parse access maps
                        write_map = isl.Map(write_access)
                        read_map = isl.Map(read_access)

                        # Normalize both maps to use same iterator name
                        # write: { S_w[i] -> arr[f(i)] }
                        # read:  { S_r[i] -> arr[g(i)] }
                        # We need to find i where f(i) = g(i)

                        # Compose: read_map^-1 . write_map gives us { S_w[i] -> S_r[j] : f(i) = g(j) }
                        # For SAME iteration, we need i = j (after normalizing statement names)

                        # Method: Extract the index expressions and compare
                        # f(i) = g(i) means the offset from i must be the same
                        # e.g., a[(i)] vs a[(i+1)] - different offsets, no same-iteration dep
                        # e.g., a[(i+1)] vs a[(i+1)] - same offset, same-iteration dep possible

                        # Simple approach: check if the index expressions are identical
                        # (more sophisticated would use ISL to compute exact relation)
                        if w_info['index'] == r_info['index']:
                            # Same index expression - they DO access same location in same iteration
                            if writer_order < reader_order:
                                # Writer executes before reader within same iteration
                                # This IS a flow dependence (RAW)
                                dep = {
                                    'type': 'flow',
                                    'array': w_info['array'],
                                    'source': write_access,
                                    'sink': read_access,
                                    'source_idx': w_info['index'],
                                    'sink_idx': r_info['index'],
                                    'kind': 'loop-independent (inter-stmt)',
                                    'description': f"Write {w_info['array']}[{w_info['index']}] in {writer_name}, Read {r_info['array']}[{r_info['index']}] in {reader_name}"
                                }
                                deps.append(dep)
                            # else: reader before writer = WAR, not RAW
                        # Different index expressions within same iteration = different locations
                        # e.g., S_0 writes a[i], S_1 reads a[i+1] -> different locations for same i

                    except Exception as e:
                        # Skip on error
                        pass

    return deps

def compute_flow_deps_isl(domain_str, schedule_str, reads, writes):
    """
    Compute flow dependencies using ISL for proper dependence analysis.

    Flow (RAW) dependence: write in iteration i -> read in iteration i'
    where the same memory location is accessed and i executes before i' in SCHEDULE order.
    """
    deps = []

    try:
        domain = isl.Set(domain_str)
    except:
        return []

    # Extract schedule map
    sched_map = extract_schedule_map(schedule_str, domain)

    for write in writes:
        w_info = parse_isl_relation(write['index'])
        if not w_info:
            continue

        for read in reads:
            r_info = parse_isl_relation(read['index'])
            if not r_info:
                continue

            if w_info['array'] != r_info['array']:
                continue

            # Same array accessed - compute actual dependence using ISL
            try:
                write_map = isl.Map(write['index'])
                read_map = isl.Map(read['index'])

                # For RAW dependence: { i -> i' : W(i) = R(i') }
                # Compute: read_map . write_map^-1, then reverse
                # read_map: S_0[i'] -> array[f(i')]
                # write_map^-1: array[m] -> S_0[i] where m = g(i)
                # Composition gives: S_0[reader] -> S_0[writer]
                # Reverse to get: S_0[writer] -> S_0[reader]
                write_inv = write_map.reverse()
                dep_relation = read_map.apply_range(write_inv).reverse()

                # Filter by domain - both source and sink must be in domain
                dep_relation = dep_relation.intersect_domain(domain)
                dep_relation = dep_relation.intersect_range(domain)

                if dep_relation.is_empty():
                    continue

                # Apply schedule to determine actual execution order
                # Flow dependence only exists if writer executes BEFORE reader in schedule order
                if sched_map is not None:
                    # Transform dep_relation to schedule space
                    # dep_relation: { writer -> reader }
                    # sched_map: { iter -> sched_time }
                    # We need: sched(writer) < sched(reader)
                    sched_dep = dep_relation.apply_domain(sched_map).apply_range(sched_map)
                    # Now sched_dep is { sched(writer) -> sched(reader) }
                    # Filter to only where writer's schedule < reader's schedule
                    sched_lt = isl.Map.lex_lt(sched_map.get_space().range())
                    valid_deps = sched_dep.intersect(sched_lt)

                    if valid_deps.is_empty():
                        continue

                    # Map back to iteration space for reporting
                    # Check loop-carried vs loop-independent in iteration space
                    lex_lt_iter = isl.Map.lex_lt(domain.get_space())
                    loop_carried = dep_relation.intersect(lex_lt_iter)

                    identity = isl.Map.identity(domain.get_space().map_from_set())
                    loop_indep = dep_relation.intersect(identity)

                    # But also check these against schedule
                    if not loop_carried.is_empty():
                        sched_carried = loop_carried.apply_domain(sched_map).apply_range(sched_map)
                        if sched_carried.intersect(sched_lt).is_empty():
                            loop_carried = isl.Map.empty(loop_carried.get_space())

                    if not loop_indep.is_empty():
                        sched_indep = loop_indep.apply_domain(sched_map).apply_range(sched_map)
                        if sched_indep.intersect(sched_lt).is_empty():
                            loop_indep = isl.Map.empty(loop_indep.get_space())
                else:
                    # No schedule info - use lexicographic order as before
                    lex_lt = isl.Map.lex_lt(domain.get_space())
                    loop_carried = dep_relation.intersect(lex_lt)

                    identity = isl.Map.identity(domain.get_space().map_from_set())
                    loop_indep = dep_relation.intersect(identity)

                dep = {
                    'type': 'flow',
                    'array': w_info['array'],
                    'source': write['index'],
                    'sink': read['index'],
                    'source_idx': w_info['index'],
                    'sink_idx': r_info['index']
                }

                if not loop_indep.is_empty() and not loop_carried.is_empty():
                    dep['kind'] = 'loop-carried,loop-independent'
                    dep['description'] = f"Write {w_info['array']}[{w_info['index']}], Read {r_info['array']}[{r_info['index']}]"
                    dep['dep_relation'] = str(dep_relation)
                    deps.append(dep)
                elif not loop_carried.is_empty():
                    dep['kind'] = 'loop-carried'
                    dep['description'] = f"Write {w_info['array']}[{w_info['index']}], Read {r_info['array']}[{r_info['index']}]"
                    dep['dep_relation'] = str(loop_carried)
                    deps.append(dep)
                elif not loop_indep.is_empty():
                    dep['kind'] = 'loop-independent'
                    dep['description'] = f"Same location {w_info['array']}[{w_info['index']}]"
                    dep['dep_relation'] = str(loop_indep)
                    deps.append(dep)

            except Exception as e:
                # Fallback: For same-location access within same statement/iteration,
                # this is typically NOT a flow dependency because in C expression
                # evaluation, reads happen before writes (RHS before LHS assignment).
                # e.g., a[i] = a[i] * b means read a[i] first, then write.
                #
                # Only report as potential dependency if indices differ (loop-carried)
                if w_info['index'] != r_info['index']:
                    dep = {
                        'type': 'flow',
                        'array': w_info['array'],
                        'source': write['index'],
                        'sink': read['index'],
                        'source_idx': w_info['index'],
                        'sink_idx': r_info['index'],
                        'error': str(e),
                        'kind': 'loop-carried (unverified)',
                        'description': f"Write {w_info['array']}[{w_info['index']}], Read {r_info['array']}[{r_info['index']}] (fallback)"
                    }
                    deps.append(dep)
                # Same index within same statement = read-before-write in expression
                # evaluation, NOT a flow dependency

    return deps


def analyze_kernel(kernel_file):
    """Analyze a single kernel and extract dependencies."""
    pet_output = run_pet(kernel_file)
    if not pet_output:
        return None

    try:
        data = yaml.safe_load(pet_output)
    except:
        return None

    schedule = data.get('schedule', '')
    results = {
        'schedule': schedule,
        'arrays': [a.get('extent', '') for a in data.get('arrays', [])],
        'statements': [],
        'dependencies': []
    }

    # First pass: collect statement info
    stmt_infos_for_inter_analysis = []
    for stmt in data.get('statements', []):
        domain = stmt.get('domain', '')
        reads, writes = extract_accesses(stmt)

        stmt_info = {
            'domain': domain,
            'reads': [r['index'] for r in reads],
            'writes': [w['index'] for w in writes]
        }
        results['statements'].append(stmt_info)
        stmt_infos_for_inter_analysis.append(stmt_info)

        # Compute dependencies within this statement using ISL
        deps = compute_flow_deps_isl(domain, schedule, reads, writes)
        results['dependencies'].extend(deps)

    # Second pass: compute inter-statement dependencies for multi-statement kernels
    if len(stmt_infos_for_inter_analysis) > 1:
        inter_deps = compute_inter_stmt_flow_deps(stmt_infos_for_inter_analysis, schedule)
        results['dependencies'].extend(inter_deps)

    return results


def main():
    # List of target kernels (s000-s119 range + s1111, s1112, s1113, s1115, s1119)
    kernels = []
    target_nums = set(range(0, 120))  # s000-s119
    target_nums.update([1111, 1112, 1113, 1115, 1119])  # Add s1111, s1112, s1113, s1115, s1119

    for f in os.listdir(KERNELS_DIR):
        if f.endswith('.c'):
            name = f[:-2]
            if re.match(r's\d{3,4}$', name):
                num_match = re.search(r's(\d+)', name)
                if num_match and int(num_match.group(1)) in target_nums:
                    kernels.append(name)

    kernels.sort()

    print("="*80)
    print("FLOW DEPENDENCE ANALYSIS")
    print("="*80)

    all_results = {}
    for kernel in kernels:
        kernel_file = os.path.join(KERNELS_DIR, f"{kernel}.c")
        if not os.path.exists(kernel_file):
            continue

        result = analyze_kernel(kernel_file)
        if not result:
            continue

        all_results[kernel] = result

        print(f"\n{'='*40}")
        print(f"Kernel: {kernel}")
        print(f"{'='*40}")

        # Extract and display schedule info
        schedule = result.get('schedule', '')
        sched_match = re.search(r'S_\d+\[[^\]]+\]\s*->\s*\[([^\]]+)\]', schedule)
        sched_expr = sched_match.group(1).strip() if sched_match else 'lexicographic'

        # Check for statement sequence ordering
        stmt_order = parse_sequence_order(schedule)
        if stmt_order and len(result['statements']) > 1:
            # Sort by order value and display sequence
            ordered_stmts = sorted(stmt_order.items(), key=lambda x: x[1])
            seq_str = ' -> '.join(s[0] for s in ordered_stmts)
            print(f"\nStatement sequence: {seq_str}")

        for i, stmt in enumerate(result['statements']):
            print(f"\nStatement S_{i}:")
            print(f"  Domain: {stmt['domain']}")
            print(f"  Schedule: ({sched_expr}, order={stmt_order.get(f'S_{i}', i)})" if stmt_order else f"  Schedule: {sched_expr}")
            print(f"  Writes: {stmt['writes']}")
            print(f"  Reads:  {stmt['reads']}")

        if result['dependencies']:
            print(f"\nFlow Dependencies:")
            for dep in result['dependencies']:
                print(f"  [{dep['kind']}] {dep['description']}")
        else:
            print(f"\nNo flow dependencies detected (fully parallel)")

    # Group by dependency pattern
    print("\n" + "="*80)
    print("GROUPING BY DEPENDENCY PATTERN")
    print("="*80)

    groups = defaultdict(list)
    for kernel, result in all_results.items():
        if not result['dependencies']:
            pattern = "independent"
        else:
            dep_kinds = set(d['kind'] for d in result['dependencies'])
            arrays = set(d['array'] for d in result['dependencies'])
            pattern = f"deps_on_{','.join(sorted(arrays))}|{','.join(sorted(dep_kinds))}"
        groups[pattern].append(kernel)

    for pattern, kernels in sorted(groups.items()):
        print(f"\n{pattern}:")
        print(f"  {', '.join(sorted(kernels))}")


if __name__ == "__main__":
    main()
