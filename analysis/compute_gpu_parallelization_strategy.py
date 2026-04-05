#!/usr/bin/env python3
"""
GPU Parallelization Strategy Analysis Module

Detects parallelization patterns in Polybench kernels and produces
Triton code skeletons with concrete strategy recommendations.

Three patterns detected:
1. Inner-Loop Vectorization (trisolv, ludcmp) — sequential outer, vectorizable inner reduction
2. Wavefront/Anti-Diagonal Parallelism (nussinov) — DP with anti-diagonal independence
3. Multi-Kernel Matrix Multiplication (3mm, 2mm) — multiple independent GEMM nests
"""

import re
import os
from typing import Optional, Dict, List, Tuple, Any


KERNELS_DIR = os.path.join(os.path.dirname(__file__), "kernels_polybench")

# Try to import LLVM analyzer for direction vectors
try:
    from llvm_analyzer import LLVMAnalyzer
    HAS_LLVM = True
except ImportError:
    HAS_LLVM = False


# ============================================================================
# Loop structure parser
# ============================================================================

def _parse_loop_structure(c_code: str) -> List[dict]:
    """Parse for-loop nests from C source code.

    Returns a flat list of loop descriptors, each with:
      - var: loop variable name
      - init: init expression string
      - cond: condition expression string
      - incr: increment expression string
      - depth: nesting depth (0 = top-level)
      - direction: 'forward' or 'backward'
      - bound_var: the upper/lower bound variable or number
      - triangular: True if bound depends on an outer loop variable
      - body: raw string of the loop body (without sub-loops stripped)
      - start_pos: character offset in c_code
    """
    # Extract scop region if present
    scop_match = re.search(r'#pragma\s+scop\s*\n(.*?)#pragma\s+endscop', c_code, re.DOTALL)
    text = scop_match.group(1) if scop_match else c_code

    loops = []
    # Match for-loop headers: for (init; cond; incr)
    pattern = re.compile(
        r'for\s*\(\s*'
        r'([^;]*?)\s*;\s*'    # init
        r'([^;]*?)\s*;\s*'    # cond
        r'([^)]*?)\s*\)',     # incr
        re.DOTALL
    )

    def _find_matching_brace_or_stmt(text, pos):
        """Find the body of a for-loop starting at pos (after the closing paren)."""
        # Skip whitespace
        while pos < len(text) and text[pos] in ' \t\n\r':
            pos += 1
        if pos >= len(text):
            return pos, ""
        if text[pos] == '{':
            depth = 1
            start = pos + 1
            pos += 1
            while pos < len(text) and depth > 0:
                if text[pos] == '{':
                    depth += 1
                elif text[pos] == '}':
                    depth -= 1
                pos += 1
            return pos, text[start:pos - 1]
        else:
            # Single statement until semicolon (handling nested parens)
            start = pos
            paren_depth = 0
            while pos < len(text):
                if text[pos] == '(':
                    paren_depth += 1
                elif text[pos] == ')':
                    paren_depth -= 1
                elif text[pos] == ';' and paren_depth == 0:
                    pos += 1
                    break
                pos += 1
            return pos, text[start:pos]

    def _parse_loops_recursive(text, base_depth=0):
        results = []
        for m in pattern.finditer(text):
            init_str = m.group(1).strip()
            cond_str = m.group(2).strip()
            incr_str = m.group(3).strip()

            # Find body
            body_start = m.end()
            body_end, body_text = _find_matching_brace_or_stmt(text, body_start)

            # Parse loop variable from init
            var_match = re.match(r'(?:int\s+)?(\w+)\s*=\s*(.*)', init_str)
            var = var_match.group(1) if var_match else '?'
            init_val = var_match.group(2).strip() if var_match else init_str

            # Determine direction
            direction = 'forward'
            if '--' in incr_str or '-=' in incr_str:
                direction = 'backward'
            elif '++' in incr_str or '+=' in incr_str:
                direction = 'forward'

            # Parse bound from condition
            bound_var = None
            triangular = False
            # Forward: var < BOUND or var <= BOUND
            fwd_match = re.match(rf'{re.escape(var)}\s*(<|<=)\s*(.+)', cond_str)
            # Backward: var >= BOUND or var > BOUND
            bwd_match = re.match(rf'{re.escape(var)}\s*(>=|>)\s*(.+)', cond_str)
            if fwd_match:
                bound_var = fwd_match.group(2).strip()
            elif bwd_match:
                bound_var = bwd_match.group(2).strip()

            # Check if bound references an outer loop variable (triangular)
            if bound_var:
                outer_vars = [l['var'] for l in results]
                # Also check parent context
                for ov in outer_vars:
                    if re.search(rf'\b{re.escape(ov)}\b', bound_var):
                        triangular = True
                        break

            loop_info = {
                'var': var,
                'init': init_val,
                'cond': cond_str,
                'incr': incr_str,
                'depth': base_depth,
                'direction': direction,
                'bound_var': bound_var,
                'triangular': triangular,
                'body': body_text.strip(),
                'start_pos': m.start(),
            }
            results.append(loop_info)

            # Recurse into body for nested loops
            inner = _parse_loops_recursive(body_text, base_depth + 1)
            for il in inner:
                # Fix triangular detection with outer vars from all levels
                if il.get('bound_var'):
                    for outer in results:
                        if re.search(rf'\b{re.escape(outer["var"])}\b', il['bound_var']):
                            il['triangular'] = True
                            break
            results.extend(inner)

        return results

    loops = _parse_loops_recursive(text)
    return loops


def _classify_inner_body(body: str, inner_var: str, outer_var: str) -> str:
    """Classify the body of an inner loop.

    Returns one of: 'dot_product', 'max_reduction', 'matmul_accumulate', 'unknown'
    """
    # Dot-product / triangular solve pattern:
    # x[i] -= L[i][j] * x[j]  or  w -= A[i][k] * A[k][j]
    if re.search(r'\w+\s*-=\s*\w+\[', body):
        return 'dot_product'

    # GEMM accumulate: C[i][j] += A[i][k] * B[k][j]
    if re.search(r'\w+\[.*?\]\[.*?\]\s*\+=\s*\w+\[.*?\]\[.*?\]\s*\*\s*\w+\[.*?\]\[.*?\]', body):
        return 'matmul_accumulate'

    # Max reduction: table[i][j] = max_score(table[i][j], ...)
    if re.search(r'max_score|fmax|MAX\(', body) or re.search(r'=\s*\(.*>=.*\)\s*\?', body):
        return 'max_reduction'

    return 'unknown'


# ============================================================================
# Pattern 1: Inner-Loop Vectorization
# ============================================================================

def _detect_inner_loop_vectorization(c_code: str, loops: List[dict],
                                      deps: Optional[dict] = None) -> Optional[dict]:
    """Detect sequential-outer / vectorizable-inner pattern.

    Targets: trisolv (forward solve), ludcmp (LU decomposition + forward/back solve).

    Signature:
    - Outer loop has loop-carried dependency (reads depend on prior iteration writes)
    - Inner loop is a dot-product/sum reduction with triangular bound (j < i)
    """
    # Find pairs: outer loop (depth 0) with inner loop (depth 1) that has triangular bound
    phases = []

    # Group loops by top-level nest
    top_loops = [l for l in loops if l['depth'] == 0]

    for tl in top_loops:
        # Find inner loops that are children (depth 1, appear after this loop)
        inner_loops = [l for l in loops
                       if l['depth'] == 1
                       and l['triangular']
                       and l['start_pos'] > tl['start_pos']]

        # Filter to only loops whose bound references the outer var
        relevant_inner = []
        for il in inner_loops:
            if il.get('bound_var') and re.search(rf'\b{re.escape(tl["var"])}\b', il['bound_var']):
                relevant_inner.append(il)

        if not relevant_inner:
            continue

        # Classify what the inner loop does
        for il in relevant_inner:
            body_type = _classify_inner_body(il['body'], il['var'], tl['var'])
            if body_type in ('dot_product', 'matmul_accumulate'):
                # Check for accumulation target
                accum_match = re.search(r'(\w+(?:\[\w+\])*)\s*-=', il['body'])
                if not accum_match:
                    accum_match = re.search(r'(\w+)\s*-=', il['body'])

                # Determine what arrays are involved
                array_refs = re.findall(r'(\w+)\[', il['body'])
                arrays_in_inner = list(set(array_refs))

                phases.append({
                    'outer_var': tl['var'],
                    'outer_direction': tl['direction'],
                    'outer_init': tl['init'],
                    'outer_bound': tl['bound_var'],
                    'inner_var': il['var'],
                    'inner_bound': il['bound_var'],
                    'body_type': body_type,
                    'accumulator': accum_match.group(1) if accum_match else None,
                    'arrays': arrays_in_inner,
                    'raw_body': il['body'].strip(),
                })

    if not phases:
        return None

    # Optionally validate with LLVM direction vectors
    llvm_confirmed = False
    if deps and deps.get('dependencies'):
        for dep in deps['dependencies']:
            dv = dep.get('direction_vector', [])
            if len(dv) >= 2:
                # Outer carries dep (non-zero/non-equal), inner is safe (0 or =)
                outer_carries = dv[0] not in ('0', '=')
                inner_safe = len(dv) > 1 and dv[1] in ('0', '=')
                if outer_carries and inner_safe:
                    llvm_confirmed = True
                    break

    return {
        'pattern': 'inner_loop_vectorization',
        'phases': phases,
        'llvm_confirmed': llvm_confirmed,
        'recommendation': 'Outer loop sequential. Vectorize inner reduction using tl.arange() + tl.sum().',
    }


# ============================================================================
# Pattern 2: Wavefront / Anti-Diagonal Parallelism
# ============================================================================

def _detect_wavefront_parallelism(c_code: str, loops: List[dict]) -> Optional[dict]:
    """Detect wavefront/anti-diagonal parallelism pattern.

    Target: nussinov (RNA secondary structure DP).

    Signature:
    - Backward outer loop (i = N-1 → 0)
    - Forward inner loop (j = i+1 → N)
    - Reads from "southwest" neighbors: [i+1][*], [*][j-1], [i+1][j-1]
    - Anti-diagonal elements where j-i = d are independent
    """
    # Find backward-forward triangular pair
    top_loops = [l for l in loops if l['depth'] == 0]

    for tl in top_loops:
        if tl['direction'] != 'backward':
            continue

        # Find forward inner loop at depth 1
        inner_fwd = [l for l in loops
                     if l['depth'] == 1
                     and l['direction'] == 'forward'
                     and l['triangular']
                     and l['start_pos'] > tl['start_pos']]

        for il in inner_fwd:
            outer_var = tl['var']
            inner_var = il['var']

            # Check init of inner loop references outer var: j = i+1
            if not re.search(rf'\b{re.escape(outer_var)}\s*\+\s*1\b', il['init']):
                # Also check: init is outer_var + 1
                if not re.search(rf'\b{re.escape(outer_var)}\b', il['init']):
                    continue

            # Look for DP table accesses in the body region
            # Get the full body of the outer loop
            body = tl['body']

            # Check for southwest neighbor reads:
            # [i+1][j], [i][j-1], [i+1][j-1]
            has_south = bool(re.search(
                rf'\w+\[\s*{re.escape(outer_var)}\s*\+\s*1\s*\]\s*\[\s*{re.escape(inner_var)}\s*\]',
                body
            ))
            has_west = bool(re.search(
                rf'\w+\[\s*{re.escape(outer_var)}\s*\]\s*\[\s*{re.escape(inner_var)}\s*-\s*1\s*\]',
                body
            ))
            has_southwest = bool(re.search(
                rf'\w+\[\s*{re.escape(outer_var)}\s*\+\s*1\s*\]\s*\[\s*{re.escape(inner_var)}\s*-\s*1\s*\]',
                body
            ))

            if has_south or has_west or has_southwest:
                # Determine the DP table name
                table_refs = re.findall(
                    rf'(\w+)\[\s*{re.escape(outer_var)}[^]]*\]\s*\[\s*{re.escape(inner_var)}[^]]*\]',
                    body
                )
                table_name = table_refs[0] if table_refs else 'table'

                # Check for inner k-loop (split point reduction)
                has_k_loop = bool(re.search(
                    rf'for\s*\(\s*\w+\s*=\s*{re.escape(outer_var)}\s*\+\s*1\s*;',
                    body
                ))

                return {
                    'pattern': 'wavefront_anti_diagonal',
                    'outer_var': outer_var,
                    'inner_var': inner_var,
                    'table_name': table_name,
                    'has_south': has_south,
                    'has_west': has_west,
                    'has_southwest': has_southwest,
                    'has_k_reduction': has_k_loop,
                    'recommendation': (
                        f'Wavefront parallelism: process anti-diagonals d=1..N-1 sequentially. '
                        f'Within each diagonal, all ({outer_var},{inner_var}) pairs where '
                        f'{inner_var}-{outer_var}=d are independent. '
                        f'Launch one kernel per diagonal with grid=(n_elements_on_diag,).'
                    ),
                }

    return None


# ============================================================================
# Pattern 3: Multi-Kernel Matrix Multiplication
# ============================================================================

def _detect_multi_kernel_matmul(c_code: str, loops: List[dict],
                                 kernel_name: str) -> Optional[dict]:
    """Detect multiple independent GEMM nests with data flow.

    Target: 3mm (E=A*B, F=C*D, G=E*F), 2mm (D=alpha*A*B*C+beta*D).

    Signature:
    - Multiple top-level triple-nested loops
    - Each has GEMM pattern: output[i][j] = 0; for k: output[i][j] += lhs[i][k] * rhs[k][j]
    """
    scop_match = re.search(r'#pragma\s+scop\s*\n(.*?)#pragma\s+endscop', c_code, re.DOTALL)
    text = scop_match.group(1) if scop_match else c_code

    # Split into top-level loop nests by finding depth-0 for-loops
    top_loops = [l for l in loops if l['depth'] == 0]

    gemm_nests = []
    for tl in top_loops:
        body = tl['body']
        outer_var = tl['var']

        # Find the second-level loop (j-loop)
        j_loops = [l for l in loops
                   if l['depth'] == 1
                   and l['start_pos'] > tl['start_pos']
                   and not l['triangular']]

        for jl in j_loops:
            inner_var = jl['var']
            jbody = jl['body']

            # Find the third-level k-loop
            k_loops = [l for l in loops
                       if l['depth'] == 2
                       and l['start_pos'] > jl['start_pos']
                       and not l['triangular']]

            for kl in k_loops:
                k_var = kl['var']

                # Check for GEMM pattern in k-loop body:
                # output[outer][inner] += lhs[outer][k] * rhs[k][inner]
                accum_match = re.search(
                    rf'(\w+)\[.*?{re.escape(outer_var)}.*?\]\[.*?{re.escape(inner_var)}.*?\]\s*\+=\s*'
                    rf'(\w+)\[.*?{re.escape(outer_var)}.*?\]\[.*?{re.escape(k_var)}.*?\]\s*\*\s*'
                    rf'(\w+)\[.*?{re.escape(k_var)}.*?\]\[.*?{re.escape(inner_var)}.*?\]',
                    kl['body']
                )

                if accum_match:
                    output_arr = accum_match.group(1)
                    lhs_arr = accum_match.group(2)
                    rhs_arr = accum_match.group(3)

                    # Get dimensions from loop bounds
                    gemm_nests.append({
                        'output': output_arr,
                        'lhs': lhs_arr,
                        'rhs': rhs_arr,
                        'outer_var': outer_var,
                        'inner_var': inner_var,
                        'k_var': k_var,
                        'M_bound': tl['bound_var'],
                        'N_bound': jl['bound_var'],
                        'K_bound': kl['bound_var'],
                    })
                    break  # One GEMM per j-loop
            if gemm_nests and gemm_nests[-1]['outer_var'] == outer_var:
                break  # Found GEMM for this outer loop

    if len(gemm_nests) < 2:
        return None

    # Trace data flow: which output feeds which input
    data_flow = []
    for i, g1 in enumerate(gemm_nests):
        for j, g2 in enumerate(gemm_nests):
            if i == j:
                continue
            if g1['output'] in (g2['lhs'], g2['rhs']):
                data_flow.append({
                    'producer': g1['output'],
                    'producer_idx': i,
                    'consumer': g2['output'],
                    'consumer_idx': j,
                    'role': 'lhs' if g1['output'] == g2['lhs'] else 'rhs',
                })

    # Build description
    descriptions = []
    for i, g in enumerate(gemm_nests):
        descriptions.append(f"{g['output']} = {g['lhs']} * {g['rhs']} "
                          f"(M={g['M_bound']}, N={g['N_bound']}, K={g['K_bound']})")

    # Determine independence
    independent_pairs = []
    for i in range(len(gemm_nests)):
        for j in range(i + 1, len(gemm_nests)):
            # Check if neither feeds into the other
            feeds_ij = any(d['producer_idx'] == i and d['consumer_idx'] == j for d in data_flow)
            feeds_ji = any(d['producer_idx'] == j and d['consumer_idx'] == i for d in data_flow)
            if not feeds_ij and not feeds_ji:
                independent_pairs.append((i, j))

    return {
        'pattern': 'multi_kernel_matmul',
        'gemm_nests': gemm_nests,
        'data_flow': data_flow,
        'independent_pairs': independent_pairs,
        'descriptions': descriptions,
        'n_gemms': len(gemm_nests),
        'recommendation': (
            f'{len(gemm_nests)} matrix multiplications detected. '
            f'Use tl.dot() with BLOCK_M=BLOCK_N=BLOCK_K=16 (minimum for tl.dot). '
            f'Launch separate Triton kernels per GEMM for maximum parallelism.'
        ),
    }


# ============================================================================
# Top-level orchestrator
# ============================================================================

def analyze_kernel_gpu_strategy(kernel_name: str,
                                 kernel_file: str = None) -> Optional[dict]:
    """Analyze a kernel and return GPU parallelization strategy.

    Args:
        kernel_name: Name of the kernel (e.g., 'trisolv', 'nussinov')
        kernel_file: Path to C source file. If None, looks in KERNELS_DIR.

    Returns:
        dict with 'pattern', 'recommendation', and pattern-specific details,
        or None if no actionable pattern detected.
    """
    if kernel_file is None:
        kernel_file = os.path.join(KERNELS_DIR, f"{kernel_name}.c")

    if not os.path.exists(kernel_file):
        return None

    with open(kernel_file, 'r') as f:
        c_code = f.read()

    # Parse loop structure
    loops = _parse_loop_structure(c_code)
    if not loops:
        return None

    # Get LLVM dependency info if available
    deps = None
    if HAS_LLVM:
        try:
            analyzer = LLVMAnalyzer()
            deps = analyzer.analyze_dependencies(kernel_file)
        except Exception:
            pass

    # Try each pattern detector in order of specificity
    # Pattern 2: Wavefront (most specific — backward-forward DP)
    result = _detect_wavefront_parallelism(c_code, loops)
    if result:
        return result

    # Pattern 3: Multi-kernel matmul
    result = _detect_multi_kernel_matmul(c_code, loops, kernel_name)
    if result:
        return result

    # Pattern 1: Inner-loop vectorization (most general — sequential + triangular inner)
    result = _detect_inner_loop_vectorization(c_code, loops, deps)
    if result:
        return result

    return None


# ============================================================================
# Prompt formatter with Triton code skeletons
# ============================================================================

def build_gpu_strategy_instructions(kernel_name: str,
                                     strategy_result: dict) -> str:
    """Format GPU strategy analysis into prompt text with Triton code skeletons.

    Args:
        kernel_name: Kernel name
        strategy_result: Output from analyze_kernel_gpu_strategy()

    Returns:
        Formatted markdown string for inclusion in LLM prompt.
    """
    if not strategy_result:
        return ""

    pattern = strategy_result['pattern']

    if pattern == 'inner_loop_vectorization':
        return _format_inner_vectorization(kernel_name, strategy_result)
    elif pattern == 'wavefront_anti_diagonal':
        return _format_wavefront(kernel_name, strategy_result)
    elif pattern == 'multi_kernel_matmul':
        return _format_multi_matmul(kernel_name, strategy_result)

    return ""


def _format_inner_vectorization(kernel_name: str, result: dict) -> str:
    """Format inner-loop vectorization strategy with Triton skeleton."""
    phases = result['phases']

    section = "## GPU Parallelization Strategy: Inner-Loop Vectorization\n\n"
    section += ("**Key insight**: The outer loop has loop-carried dependencies and MUST be sequential. "
                "But the inner triangular reduction can be vectorized with `tl.arange()` + `tl.sum()`.\n\n")

    section += "**CRITICAL**: Do NOT try to parallelize the outer loop. Instead, use a SINGLE program "
    section += "(grid=(1,)) but vectorize all inner-loop reductions using Triton vector operations.\n\n"

    for i, phase in enumerate(phases):
        section += f"### Phase {i + 1}: "
        section += f"Outer `{phase['outer_var']}` ({phase['outer_direction']}, "
        section += f"from {phase['outer_init']} to {phase['outer_bound']}), "
        section += f"inner `{phase['inner_var']}` (up to {phase['inner_bound']})\n\n"
        section += f"- Body type: {phase['body_type']}\n"
        if phase['accumulator']:
            section += f"- Accumulator: `{phase['accumulator']}`\n"
        section += f"- Arrays involved: {', '.join(f'`{a}`' for a in phase['arrays'])}\n\n"

    # Generate Triton skeleton
    section += "### Recommended Triton Pattern\n\n"
    section += "```python\n"
    section += "@triton.jit\n"
    section += f"def {kernel_name}_kernel(\n"
    section += "    # ... array pointers and dimensions ...\n"
    section += "    BLOCK_J: tl.constexpr,  # e.g., 128\n"
    section += "):\n"
    section += "    j_offsets = tl.arange(0, BLOCK_J)  # Define ONCE outside all loops\n"

    # Use first phase as example
    p = phases[0]
    if p['outer_direction'] == 'forward':
        section += f"    for {p['outer_var']} in range({p['outer_bound']}):\n"
    else:
        section += f"    for {p['outer_var']} in range({p['outer_bound']} - 1, -1, -1):\n"

    section += f"        # Load accumulator\n"
    if p['accumulator']:
        section += f"        acc = tl.load(...)  # Load {p['accumulator']}\n"
    else:
        section += f"        acc = tl.load(...)  # Load initial value\n"

    section += f"        # Vectorized inner reduction over {p['inner_var']}\n"
    section += f"        for j_start in range(0, {p['outer_var']}, BLOCK_J):\n"
    section += f"            j = j_start + j_offsets\n"
    section += f"            j_mask = j < {p['outer_var']}\n"
    section += f"            # Load vector of values, multiply, reduce\n"
    section += f"            vals_a = tl.load(ptr_a + ..., mask=j_mask, other=0.0)\n"
    section += f"            vals_b = tl.load(ptr_b + j, mask=j_mask, other=0.0)\n"
    section += f"            acc -= tl.sum(vals_a * vals_b)  # Vectorized dot product\n"
    section += f"        tl.store(...)  # Store result\n"
    section += "```\n\n"

    section += "**Block size**: Use `BLOCK_J=128` for good throughput on triangular reductions.\n"
    section += "**Grid**: `grid=(1,)` — single program instance since outer loop is sequential.\n"

    return section


def _format_wavefront(kernel_name: str, result: dict) -> str:
    """Format wavefront/anti-diagonal strategy with Triton skeleton."""
    outer_var = result['outer_var']
    inner_var = result['inner_var']
    table = result['table_name']

    section = "## GPU Parallelization Strategy: Wavefront (Anti-Diagonal) Parallelism\n\n"
    section += (f"**Key insight**: The nested loop over ({outer_var},{inner_var}) has dependencies on "
                f"'southwest' neighbors ")

    neighbors = []
    if result['has_south']:
        neighbors.append(f"`{table}[{outer_var}+1][{inner_var}]`")
    if result['has_west']:
        neighbors.append(f"`{table}[{outer_var}][{inner_var}-1]`")
    if result['has_southwest']:
        neighbors.append(f"`{table}[{outer_var}+1][{inner_var}-1]`")
    section += f"({', '.join(neighbors)}). "

    section += (f"But elements on the same anti-diagonal (where `{inner_var} - {outer_var} = d`) "
                f"are **independent** and can be processed in parallel.\n\n")

    section += "**Strategy**: Process diagonals d=1..N-1 sequentially. "
    section += "Within each diagonal, launch a parallel Triton kernel over all elements.\n\n"

    if result.get('has_k_reduction'):
        section += (f"**Note**: There is also an inner k-loop reduction (split-point maximization). "
                    f"This can be vectorized within each diagonal element.\n\n")

    # Triton skeleton
    section += "### Recommended Triton Pattern\n\n"
    section += "```python\n"
    section += "@triton.jit\n"
    section += f"def {kernel_name}_diag_kernel(\n"
    section += f"    {table}_ptr, seq_ptr, N, diag_offset, n_elems,\n"
    section += "    BLOCK: tl.constexpr,\n"
    section += "):\n"
    section += "    pid = tl.program_id(0)\n"
    section += "    offsets = pid * BLOCK + tl.arange(0, BLOCK)\n"
    section += "    mask = offsets < n_elems\n"
    section += f"    # Map linear index to ({outer_var}, {inner_var}) on anti-diagonal\n"
    section += f"    {outer_var} = (N - 1 - diag_offset) + offsets  # or appropriate mapping\n"
    section += f"    {inner_var} = {outer_var} + diag_offset\n"
    section += f"    valid = mask & ({outer_var} >= 0) & ({inner_var} < N)\n"
    section += f"    # Load southwest neighbors and compute max\n"
    section += f"    # ... load {table}[{outer_var}+1][{inner_var}], {table}[{outer_var}][{inner_var}-1], etc.\n"
    section += f"    # ... compute and store result\n"
    section += "```\n\n"

    section += "```python\n"
    section += f"def {kernel_name}_triton(...):\n"
    section += "    BLOCK = 64\n"
    section += "    for d in range(1, N):  # Process diagonals sequentially\n"
    section += "        n_elems = min(d, N - d)  # Elements on this diagonal\n"
    section += "        grid = (triton.cdiv(n_elems, BLOCK),)\n"
    section += f"        {kernel_name}_diag_kernel[grid]({table}, seq, N, d, n_elems, BLOCK=BLOCK)\n"
    section += "```\n\n"

    section += "**Grid**: One kernel launch per diagonal, `grid=(ceil(n_elems/BLOCK),)` per launch.\n"
    section += "**Block size**: `BLOCK=64` is a good starting point for diagonal parallelism.\n"

    return section


def _format_multi_matmul(kernel_name: str, result: dict) -> str:
    """Format multi-kernel GEMM strategy with Triton skeleton."""
    gemms = result['gemm_nests']
    data_flow = result['data_flow']
    indep = result['independent_pairs']

    section = "## GPU Parallelization Strategy: Multi-Kernel Matrix Multiplication\n\n"
    section += f"**Key insight**: This kernel contains {result['n_gemms']} matrix multiplications:\n\n"

    for i, desc in enumerate(result['descriptions']):
        section += f"{i + 1}. `{desc}`\n"
    section += "\n"

    # Data flow
    if data_flow:
        section += "**Data flow dependencies**:\n"
        for df in data_flow:
            section += (f"- `{gemms[df['producer_idx']]['output']}` (GEMM {df['producer_idx'] + 1}) "
                       f"feeds into GEMM {df['consumer_idx'] + 1} as {df['role']}\n")
        section += "\n"

    if indep:
        pairs_str = ", ".join(f"GEMM {a+1} and GEMM {b+1}" for a, b in indep)
        section += f"**Independent GEMMs** (can compute in any order): {pairs_str}\n\n"

    section += ("**Strategy**: Implement each GEMM as a separate Triton kernel using `tl.dot()` "
                "for hardware-accelerated matrix multiply. This is the standard tiled GEMM pattern.\n\n")

    # Block size recommendation
    section += "### Recommended Block Sizes\n\n"
    section += ("Use `BLOCK_M = BLOCK_N = BLOCK_K = 16` (minimum required by `tl.dot()`). "
                "For these matrix dimensions, 16 is optimal to avoid wasted computation.\n\n")

    # Triton skeleton
    section += "### Recommended Triton Pattern (per GEMM)\n\n"
    section += "```python\n"
    section += "@triton.jit\n"
    section += "def matmul_kernel(\n"
    section += "    a_ptr, b_ptr, c_ptr,\n"
    section += "    M, N, K,\n"
    section += "    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,\n"
    section += "):\n"
    section += "    pid_m = tl.program_id(0)\n"
    section += "    pid_n = tl.program_id(1)\n"
    section += "    # Tile offsets\n"
    section += "    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)\n"
    section += "    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)\n"
    section += "    # Accumulator\n"
    section += "    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)\n"
    section += "    rk = tl.arange(0, BLOCK_K)\n"
    section += "    for k_start in range(0, K, BLOCK_K):\n"
    section += "        k = k_start + rk\n"
    section += "        # Load tiles\n"
    section += "        a = tl.load(a_ptr + rm[:, None] * K + k[None, :],\n"
    section += "                    mask=(rm[:, None] < M) & (k[None, :] < K), other=0.0)\n"
    section += "        b = tl.load(b_ptr + k[:, None] * N + rn[None, :],\n"
    section += "                    mask=(k[:, None] < K) & (rn[None, :] < N), other=0.0)\n"
    section += "        acc += tl.dot(a, b)\n"
    section += "    # Store result\n"
    section += "    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc,\n"
    section += "             mask=(rm[:, None] < M) & (rn[None, :] < N))\n"
    section += "```\n\n"

    section += "```python\n"

    # Build valid function identifier
    func_id = kernel_name
    if func_id[0].isdigit():
        func_id = "k" + func_id

    section += f"def {func_id}_triton(...):\n"
    section += "    BLOCK_M, BLOCK_N, BLOCK_K = 16, 16, 16\n"
    for i, g in enumerate(gemms):
        section += f"    # GEMM {i + 1}: {g['output']} = {g['lhs']} * {g['rhs']}\n"
        section += (f"    grid_{i + 1} = (triton.cdiv({g['M_bound']}, BLOCK_M), "
                   f"triton.cdiv({g['N_bound']}, BLOCK_N))\n")
        section += (f"    matmul_kernel[grid_{i + 1}]("
                   f"{g['lhs']}, {g['rhs']}, {g['output']}, "
                   f"{g['M_bound']}, {g['N_bound']}, {g['K_bound']}, "
                   f"BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)\n")
    section += "```\n\n"

    section += (f"**Grid**: 2D grid `(ceil(M/BLOCK_M), ceil(N/BLOCK_N))` per GEMM.\n"
                f"**IMPORTANT**: Initialize output arrays to zero before each GEMM "
                f"(use `torch.zeros` or a separate zero-fill kernel).\n")

    return section


# ============================================================================
# CLI test
# ============================================================================

if __name__ == '__main__':
    import sys

    kernels_to_test = sys.argv[1:] if len(sys.argv) > 1 else [
        'trisolv', 'nussinov', 'ludcmp', '3mm'
    ]

    for kname in kernels_to_test:
        kfile = os.path.join(KERNELS_DIR, f"{kname}.c")
        print(f"\n{'='*60}")
        print(f"Kernel: {kname}")
        print(f"{'='*60}")

        if not os.path.exists(kfile):
            print(f"  File not found: {kfile}")
            continue

        result = analyze_kernel_gpu_strategy(kname, kfile)
        if result:
            print(f"  Pattern: {result['pattern']}")
            print(f"  Recommendation: {result.get('recommendation', 'N/A')}")
            print()
            instructions = build_gpu_strategy_instructions(kname, result)
            print(instructions)
        else:
            print("  No pattern detected.")
