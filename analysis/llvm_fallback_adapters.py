#!/usr/bin/env python3
"""
LLVM Fallback Adapters

When PET-based analysis modules fail (typically for complex multi-statement
scop regions), these adapters provide equivalent analysis using the LLVM
toolchain as a fallback.

Each adapter returns a dict compatible with the corresponding PET module's
format, so the pipeline can use them interchangeably.
"""

import os
import re
from typing import Optional, Dict, List

from llvm_analyzer import LLVMAnalyzer

_analyzer = None


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = LLVMAnalyzer()
    return _analyzer


def llvm_war_fallback(kernel_file: str) -> Optional[dict]:
    """
    Fallback for compute_war_dependences.analyze_kernel_war().
    Uses LLVM DependenceAnalysis to detect WAR (anti) dependencies.
    """
    analyzer = _get_analyzer()
    return analyzer.analyze_war_dependencies(kernel_file)


def _direction_entry_carries_dep(entry: str) -> bool:
    """Determine if a direction vector entry carries a dependency.

    Returns True if the entry indicates the dependency is carried at this loop level.
    - '0' or '=' → same iteration, no dep carried → False
    - 'S' → sequential context (e.g. timestep loop) → already sequential → False
    - integer (nonzero) → dep carried with that distance → True
    - '<' or '>' → dep carried (forward/backward) → True
    - '*' → unknown → True (conservative)
    """
    entry = entry.strip()
    if entry in ('0', '=', 'S'):
        return False
    return True


def enhance_war_with_llvm_vectors(kernel_file: str, pet_war_result: dict) -> Optional[dict]:
    """
    Combine PET WAR results with LLVM direction vectors to scope WAR
    dependencies to specific loop levels.

    Returns enhanced WAR dict with 'loop_level_scoping' info, or None
    if enhancement fails or adds no value.
    """
    analyzer = _get_analyzer()

    # Get LLVM DA results with direction vectors
    try:
        deps = analyzer.analyze_dependencies(kernel_file)
    except Exception:
        return None
    if not deps:
        return None

    gep_map = deps.get('_gep_map', {})

    # Get loop variable names from source
    loop_vars = analyzer.get_source_loop_vars(kernel_file)
    if not loop_vars:
        return None

    # Collect LLVM anti-dependencies with direction vectors, grouped by array
    llvm_anti_by_array: Dict[str, List[dict]] = {}
    for dep in deps.get('dependencies', []):
        if dep['type'] != 'anti':
            continue
        dv = dep.get('direction_vector')
        if not dv:
            continue

        # Resolve array names from IR references
        src_arr = analyzer._extract_array_from_ir_line(dep['src'], gep_map)
        dst_arr = analyzer._extract_array_from_ir_line(dep['dst'], gep_map)
        arr = src_arr or dst_arr
        if not arr:
            continue

        if arr not in llvm_anti_by_array:
            llvm_anti_by_array[arr] = []
        llvm_anti_by_array[arr].append({
            'direction_vector': dv,
            'detail': dep.get('detail', ''),
        })

    # If no anti-deps with direction vectors found, no enhancement possible
    if not llvm_anti_by_array:
        return None

    # For each PET WAR array, find matching LLVM direction vectors and determine
    # which loop levels carry the WAR
    arrays_needing_copy = pet_war_result.get('arrays_needing_copy', [])
    if not arrays_needing_copy:
        return None

    loop_scoping = {}
    any_scoping_added = False

    for arr in arrays_needing_copy:
        llvm_deps = llvm_anti_by_array.get(arr, [])
        if not llvm_deps:
            # No LLVM info for this array — keep original WAR (conservative)
            loop_scoping[arr] = {
                'carried_by_loops': list(loop_vars),  # conservative: all loops
                'safe_to_parallelize_loops': [],
                'direction_vectors': [],
            }
            continue

        # Merge direction vectors: for each loop level position, determine
        # the worst-case across all LLVM anti-deps for this array.
        # A loop level is "carried" if ANY direction vector entry at that
        # position indicates a carried dep.
        n_levels = len(loop_vars)
        carried_at_level = [False] * n_levels

        dv_details = []
        for ldep in llvm_deps:
            dv = ldep['direction_vector']
            dv_details.append(dv)
            for pos in range(min(len(dv), n_levels)):
                if _direction_entry_carries_dep(dv[pos]):
                    carried_at_level[pos] = True

        # Determine which positions are always 'S' (sequential context)
        # across ALL direction vectors for this array.
        # A position is 'always sequential' only if ALL DVs have 'S' at that position.
        # If a DV is shorter than n_levels, positions beyond the DV are "unknown"
        # (not sequential), since LLVM didn't analyze them.
        always_sequential = [True] * n_levels
        has_any_dv_entry = [False] * n_levels
        for ldep in llvm_deps:
            dv = ldep['direction_vector']
            for pos in range(min(len(dv), n_levels)):
                has_any_dv_entry[pos] = True
                if dv[pos].strip() != 'S':
                    always_sequential[pos] = False
        # Positions with no DV entries are not sequential — mark as unknown
        for pos in range(n_levels):
            if not has_any_dv_entry[pos]:
                always_sequential[pos] = False

        # For positions where no DV has an entry, be conservative: assume carried.
        # This handles cases where LLVM sees fewer loop levels than the source
        # (e.g. due to loop transformations at O1).
        for pos in range(n_levels):
            if not has_any_dv_entry[pos] and not always_sequential[pos]:
                carried_at_level[pos] = True

        carried_loops = [loop_vars[i] for i in range(n_levels) if carried_at_level[i]]
        # A loop is safe to parallelize only if:
        # - No dep is carried at that level (carried_at_level[i] is False)
        # - It's not always marked 'S' (sequential context = outer timestep loop,
        #   which LLVM treats as sequential context, not as analyzed for parallelism)
        # - There was actual DV data for this position (we don't claim safety without evidence)
        safe_loops = [loop_vars[i] for i in range(n_levels)
                      if not carried_at_level[i] and not always_sequential[i]
                      and has_any_dv_entry[i]]

        sequential_loops = [loop_vars[i] for i in range(n_levels) if always_sequential[i]]
        loop_scoping[arr] = {
            'carried_by_loops': carried_loops,
            'safe_to_parallelize_loops': safe_loops,
            'sequential_context_loops': sequential_loops,
            'direction_vectors': dv_details,
        }

        if safe_loops:
            any_scoping_added = True

    if not any_scoping_added:
        return None

    # Build enhanced result: copy original and add scoping
    enhanced = dict(pet_war_result)
    enhanced['loop_level_scoping'] = loop_scoping
    enhanced['loop_vars'] = loop_vars
    return enhanced


def llvm_overwrite_fallback(kernel_file: str) -> Optional[dict]:
    """
    Fallback for compute_statement_overwrites.analyze_kernel_overwrites().
    Uses LLVM AST to detect when multiple stores write to the same location.
    """
    analyzer = _get_analyzer()
    accesses = analyzer.get_array_accesses(kernel_file)
    if not accesses:
        return None

    # Find arrays with multiple writes at different source locations
    write_locations = {}
    for acc in accesses:
        if acc['mode'] == 'w':
            name = acc['name']
            if name not in write_locations:
                write_locations[name] = []
            write_locations[name].append(acc['line'])

    overwrites = []
    for name, lines in write_locations.items():
        unique_lines = sorted(set(lines))
        if len(unique_lines) > 1:
            overwrites.append({
                'overwritten_stmt': 0,
                'overwriting_stmt': 1,
                'array': name,
                'overwritten_offset': 0,
                'overwriting_offset': 0,
                'offset_diff': 0,
                'description': (
                    f"Array '{name}' is written at {len(unique_lines)} different source locations "
                    f"(lines {unique_lines}). Later writes may overwrite earlier ones."
                ),
            })

    applicable = len(overwrites) > 0
    advice = ""
    if applicable:
        advice = "Statement overwrite detected (via LLVM analysis):\n"
        for ow in overwrites:
            advice += f"  - {ow['description']}\n"
        advice += "\nEnsure write ordering is preserved in the parallel implementation."

    return {
        'applicable': applicable,
        'overwrites': overwrites,
        'optimization_advice': advice,
        'statements': [],
        'loop_stride': 1,
        'source': 'llvm',
    }


def llvm_stream_compaction_fallback(kernel_file: str) -> Optional[dict]:
    """
    Fallback for compute_stream_compaction.analyze_kernel_stream_compaction().
    Stream compaction is a very specific pattern (conditional packing/unpacking).
    LLVM can detect conditionals + non-loop-variable indexing.
    """
    analyzer = _get_analyzer()

    # Read source to check for conditional + counter patterns
    with open(kernel_file, 'r') as f:
        code = f.read()

    # Stream compaction: writing to array[counter++] inside a conditional
    # Look for non-affine index variables (not loop iterators)
    has_conditional = bool(re.search(r'\bif\s*\(', code))

    # Look for counter-based indexing (variable that increments independently)
    counter_pattern = re.search(r'(\w+)\s*\+\+|(\w+)\s*\+=\s*1', code)
    has_counter = counter_pattern is not None

    applicable = has_conditional and has_counter
    details = []
    if applicable:
        counter_var = (counter_pattern.group(1) or counter_pattern.group(2))
        details.append({
            'statement': 0,
            'loop_vars': [],
            'counter_var': counter_var,
            'output_array': 'unknown',
            'has_conditional': True,
            'write_pattern': '',
            'direction': 'pack',
        })

    return {
        'applicable': applicable,
        'details': details,
        'advice': "Stream compaction pattern detected (via LLVM analysis)." if applicable else "",
        'output_arrays': [],
        'source': 'llvm',
    }


def llvm_parallel_dims_fallback(kernel_file: str) -> Optional[dict]:
    """
    Fallback for compute_parallel_dims.analyze_kernel_parallelization().
    Uses LLVM SCEV + DependenceAnalysis to determine parallelizable dimensions.
    """
    analyzer = _get_analyzer()

    # Get loop structure
    scev = analyzer.analyze_loops(kernel_file)
    if not scev or not scev.get('loops'):
        return None

    # Get dependencies
    deps = analyzer.analyze_dependencies(kernel_file)

    # Get array accesses
    accesses = analyzer.get_array_accesses(kernel_file)

    # Read source for dimension names
    with open(kernel_file, 'r') as f:
        code = f.read()

    # Use robust loop var extraction (preserves nesting order, deduplicates)
    loop_vars = analyzer.get_source_loop_vars(kernel_file)
    if not loop_vars:
        # Fallback to simple regex
        loop_vars = re.findall(r'for\s*\(\s*(?:int\s+)?(\w+)\s*=', code)

    # Build dimension list from SCEV loops
    dims = loop_vars[:3] if len(loop_vars) >= 3 else loop_vars

    # Analyze dependencies per dimension using direction vectors
    has_flow = deps and deps.get('flow_deps', 0) > 0
    has_anti = deps and deps.get('anti_deps', 0) > 0
    has_real_deps = has_flow or has_anti

    # Extract all direction vectors from flow and anti dependencies
    all_dvs = []
    if deps:
        for dep in deps.get('dependencies', []):
            if dep['type'] in ('flow', 'anti', 'output'):
                dv = dep.get('direction_vector')
                if dv:
                    all_dvs.append(dv)

    # Compute per-dim carried status using direction vectors with outer-safe filtering.
    # A dim at position P is only considered carried if there exists a DV where:
    #   - ALL positions < P have a "safe" entry (0, =, or S), meaning the dep
    #     is NOT already carried by an outer loop, AND
    #   - Position P itself has a carrying entry (*, <, >)
    # This correctly handles cases like floyd_warshall where DVs like ['*','*','0','<']
    # have deps carried by k (pos 0), making i and j safe when k is sequential.
    n_levels = len(loop_vars)
    carried_at_level = [False] * n_levels
    has_any_dv_entry = [False] * n_levels
    always_sequential = [True] * n_levels

    for dv in all_dvs:
        for pos in range(min(len(dv), n_levels)):
            has_any_dv_entry[pos] = True
            if dv[pos].strip() != 'S':
                always_sequential[pos] = False

    # Outer-safe DV filtering: only mark a dim as carried if the dep is NOT
    # already carried by an outer (lower-position) loop.
    for dv in all_dvs:
        dv_len = len(dv)
        for pos in range(min(dv_len, n_levels)):
            if not _direction_entry_carries_dep(dv[pos]):
                continue
            # Cross-phase DVs: shorter DVs where all outer positions are 'S'
            # represent phase-ordering within the sequential outer loop,
            # not spatial loop-carried deps on inner dims. Skip them.
            if dv_len < n_levels:
                all_outer_seq = all(dv[p].strip() == 'S' for p in range(pos))
                if all_outer_seq:
                    continue
            # Check if all outer positions (< pos) are safe (0, =, or S)
            outer_safe = True
            for outer_pos in range(pos):
                if outer_pos < dv_len:
                    entry = dv[outer_pos].strip()
                    if entry not in ('0', '=', 'S'):
                        outer_safe = False
                        break
            if outer_safe:
                carried_at_level[pos] = True

    # Positions with no DV entries are not sequential
    for pos in range(n_levels):
        if not has_any_dv_entry[pos]:
            always_sequential[pos] = False

    # For positions where no DV has an entry, be conservative: assume carried
    # if there are real deps (flow/anti)
    for pos in range(n_levels):
        if not has_any_dv_entry[pos] and has_real_deps:
            carried_at_level[pos] = True

    has_dv_info = any(has_any_dv_entry)

    # Determine self-dependencies from accesses
    self_deps = []
    if accesses:
        read_arrays = set(a['name'] for a in accesses if a['mode'] == 'r')
        write_arrays = set(a['name'] for a in accesses if a['mode'] == 'w')
        rw_arrays = read_arrays & write_arrays
        for arr in rw_arrays:
            self_deps.append({
                'array': arr,
                'write_expr': '(loop_var)',
                'read_expr': '(loop_var)',
            })

    # Build options - for each dimension, determine if parallelizable
    options = []
    for i, dim in enumerate(dims):
        other_dims = [d for j, d in enumerate(dims) if j != i]
        par_dim = dim
        seq_dim = other_dims[0] if other_dims else dim

        # Find the position of this dim in the full loop_vars list
        dim_pos = loop_vars.index(dim) if dim in loop_vars else i

        valid = True
        issues = []

        if has_dv_info:
            # We have direction vectors — use per-dim analysis
            if dim_pos < n_levels and carried_at_level[dim_pos]:
                valid = False
                issues.append(
                    f"Dependencies carried along `{dim}` (direction vectors indicate "
                    f"this dimension is NOT safe to parallelize)"
                )
            elif dim_pos < n_levels and always_sequential[dim_pos]:
                valid = False
                issues.append(
                    f"`{dim}` is a sequential context loop (e.g. timestep) — "
                    f"must remain sequential"
                )
            # else: no dep carried at this level → valid=True
        elif has_real_deps:
            # No direction vectors but deps exist — conservative: mark invalid
            valid = False
            issues.append(
                f"Data dependencies detected but no per-dimension info available — "
                f"cannot confirm `{dim}` is safe to parallelize"
            )

        options.append({
            'sequential_dim': seq_dim,
            'parallel_dim': par_dim,
            'valid': valid,
            'parallelism_type': 'independent',
            'triton_strategy': 'SINGLE_KERNEL_INLOOP',
            'issues': issues,
            'explanations': [],
            'inkernel_safety_details': [],
        })

    # Determine if triangular
    is_triangular = False
    triangular_info = None
    for line in code.split('\n'):
        # Look for bounds like j < i or j <= i-1
        tri_match = re.search(r'for\s*\([^;]+;\s*(\w+)\s*[<]=?\s*(\w+)', line)
        if tri_match and tri_match.group(2) in loop_vars:
            is_triangular = True
            triangular_info = {
                'smaller_dim': tri_match.group(1),
                'larger_dim': tri_match.group(2),
                'smaller': tri_match.group(1),
                'larger': tri_match.group(2),
            }

    # Summary
    loop_info = scev.get('loops', [])
    summary_parts = []
    for loop in loop_info:
        summary_parts.append(f"{loop['name']}: [{loop['start']}..{loop['end']})")
    summary = f"Loop nest with {len(loop_info)} loops: {', '.join(summary_parts)}" if summary_parts else ""

    # Get the code between #pragma scop markers
    scop_match = re.search(r'#pragma\s+scop\s*\n(.*?)#pragma\s+endscop', code, re.DOTALL)
    c_code = scop_match.group(1).strip() if scop_match else ""

    # Count distinct write arrays for multi-phase detection
    n_write_arrays = len(write_arrays) if accesses else 0

    return {
        'kernel': os.path.basename(kernel_file).replace('.c', ''),
        'c_code': c_code,
        'dims': dims,
        'is_triangular': is_triangular,
        'triangular_info': triangular_info,
        'self_dependencies': self_deps,
        'options': options,
        'summary': summary,
        'source': 'llvm',
        'n_write_arrays': n_write_arrays,
    }


def llvm_scalar_expansion_fallback(kernel_file: str) -> Optional[dict]:
    """
    Fallback for compute_scalar_expansion.analyze_kernel_scalar_expansion().
    Uses LLVM AST to detect scalar variables used as accumulators inside loops.
    """
    analyzer = _get_analyzer()

    # Read source code
    with open(kernel_file, 'r') as f:
        code = f.read()

    # Extract the scop region
    scop_match = re.search(r'#pragma\s+scop\s*\n(.*?)#pragma\s+endscop', code, re.DOTALL)
    if not scop_match:
        return None
    scop_code = scop_match.group(1)

    # Find scalar variables: locals that are read and written but not arrays
    accesses = analyzer.get_array_accesses(kernel_file)
    array_names = set(a['name'] for a in accesses) if accesses else set()

    # Extract loop variables
    loop_vars = set(re.findall(r'for\s*\(\s*(?:int\s+)?(\w+)\s*=', scop_code))

    # Find scalar assignments: var = expr or var += expr (where var is not an array)
    scalar_writes = re.findall(r'\b(\w+)\s*(?:\+\=|\-\=|\*\=|\/\=|=)', scop_code)
    scalar_reads = re.findall(r'(?<!=\s)(?<![<>!])\b(\w+)\b(?!\s*(?:\+\=|\-\=|\*\=|\/\=|=[^=]))', scop_code)

    # Filter to potential scalars: written, not a loop var, not an array name,
    # not a type keyword
    type_keywords = {'int', 'float', 'double', 'for', 'if', 'else', 'while', 'return', 'void'}
    potential_scalars = set(scalar_writes) - array_names - loop_vars - type_keywords

    candidates = []
    for var in potential_scalars:
        # Check for accumulator pattern: var += or var *= etc.
        accum_match = re.search(rf'\b{var}\s*(\+\=|\-\=|\*\=|\/\=)', scop_code)
        if accum_match:
            op = accum_match.group(1)
            candidates.append({
                'variable': var,
                'init_value': None,
                'update_info': {
                    'pattern_type': 'accumulator',
                    'update_expressions': [{'operator': op, 'expression': ''}],
                    'is_conditional': bool(re.search(rf'if\s*\([^)]*\).*{var}\s*{re.escape(op)}', scop_code, re.DOTALL)),
                    'depends_on_self': True,
                    'depends_on_loop_var': True,
                    'is_accumulator': True,
                },
                'read_info': {
                    'read_locations': [],
                    'used_in_array_writes': bool(re.search(rf'\w+\s*\[.*\]\s*=.*{var}', scop_code)),
                    'used_in_conditionals': bool(re.search(rf'if\s*\([^)]*{var}', scop_code)),
                },
                'expansion_type': 'prefix_sum' if op == '+=' else 'general_accumulator',
                'strategy': f"Expand scalar '{var}' to per-thread private values, then combine.",
                'reason': f"Scalar '{var}' is updated with '{op}' inside a loop (detected via LLVM).",
            })

    # Get loop info for context
    loop_var = list(loop_vars)[0] if loop_vars else 'i'
    loop_start = '0'
    loop_end = 'N'

    scev = analyzer.analyze_loops(kernel_file)
    if scev and scev.get('loops'):
        first_loop = scev['loops'][0]
        loop_start = str(first_loop.get('start', 0))
        loop_end = str(first_loop.get('end', 'N'))

    return {
        'has_scalar_expansion': len(candidates) > 0,
        'candidates': candidates,
        'loop_var': loop_var,
        'loop_start': loop_start,
        'loop_end': loop_end,
        'source': 'llvm',
    }


# ----------------------------------------------------------------
# Unified fallback interface
# ----------------------------------------------------------------

def try_with_llvm_fallback(pet_func, llvm_fallback_func, *args, **kwargs):
    """
    Try PET analysis first. If it fails or returns None, use LLVM fallback.

    Args:
        pet_func: The PET-based analysis function
        llvm_fallback_func: The LLVM-based fallback function
        *args: Arguments for both functions (typically kernel_file path)

    Returns:
        Analysis result dict from whichever succeeded, or None if both fail.
    """
    # Try PET first
    try:
        if pet_func is not None:
            result = pet_func(*args, **kwargs)
            if result is not None:
                return result
    except Exception:
        pass

    # Fall back to LLVM
    try:
        return llvm_fallback_func(*args)
    except Exception:
        return None
