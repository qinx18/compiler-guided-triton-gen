#!/usr/bin/env python3
"""
Scalar and Array Expansion Pattern Detection

Detects patterns where a scalar variable is updated in a loop and used in
subsequent iterations, creating a loop-carried dependency. These patterns
can often be parallelized through "scalar expansion" - converting the scalar
to an array indexed by the loop variable.

Example patterns:

Pattern 1: Wrap-around scalar with conditional update (s258)
  s = 0.;
  for (int i = 0; i < N; ++i) {
      if (a[i] > 0.) {
          s = d[i] * d[i];  // Conditionally update
      }
      b[i] = s * c[i] + d[i];  // Use current s value
  }

  After expansion -> s[i] represents the value of s at iteration i
  Can compute s[] as prefix scan, then b[] in parallel

Pattern 2: Accumulator (already handled by reduction detection)
  s = 0.;
  for (int i = 0; i < N; ++i) {
      s += a[i];  // Accumulate
      b[i] = s;   // Use accumulated value
  }

  This is a prefix sum - can be computed with scan operations

Pattern 3: Last-value propagation (s253)
  s = a[0];
  for (int i = 1; i < N; ++i) {
      if (i > k) {
          s = a[i];
      }
      b[i] = s;
  }
"""

import os
import re
from typing import Optional, Dict, List

KERNELS_DIR = "/home/qinxiao/workspace/compiler-guided-triton-gen/analysis/kernels"


def extract_scop_code(c_code: str) -> str:
    """Extract code between #pragma scop and #pragma endscop."""
    match = re.search(r'#pragma scop\s*(.*?)\s*#pragma endscop', c_code, re.DOTALL)
    if match:
        return match.group(1).strip()
    return c_code


def extract_loop_info(c_code: str) -> Optional[Dict]:
    """Extract basic loop information."""
    # Find the main for loop
    loop_match = re.search(
        r'for\s*\(\s*(?:int\s+)?(\w+)\s*=\s*([^;]+);\s*\1\s*<\s*([^;]+);\s*[^)]+\)\s*\{(.*)\}',
        c_code, re.DOTALL
    )
    if not loop_match:
        return None

    return {
        'var': loop_match.group(1),
        'start': loop_match.group(2).strip(),
        'end': loop_match.group(3).strip(),
        'body': loop_match.group(4).strip()
    }


def detect_scalar_variables(c_code: str, loop_body: str) -> List[Dict]:
    """
    Detect scalar variables that are:
    1. Declared/initialized before the loop
    2. Updated inside the loop
    3. Used (read) inside the loop

    Returns list of candidate scalar expansion variables.
    """
    candidates = []

    # Find scalar variable declarations before the loop
    # Pattern: real_t var; or real_t var = init; or real_t var1, var2;
    decl_pattern = r'(?:real_t|float|double|int)\s+([^;]+);'

    # Split code into pre-loop and loop parts
    loop_start = c_code.find('for')
    if loop_start == -1:
        return candidates

    pre_loop = c_code[:loop_start]

    # Collect all declared scalar variables (not arrays)
    declared_scalars = {}  # var_name -> init_value
    for match in re.finditer(decl_pattern, pre_loop):
        decl_body = match.group(1).strip()
        # Parse comma-separated declarations: "t, s" or "x = 0, y = 1"
        for part in decl_body.split(','):
            part = part.strip()
            # Skip array declarations
            if '[' in part:
                continue
            assign_match = re.match(r'(\w+)\s*=\s*(.+)', part)
            if assign_match:
                declared_scalars[assign_match.group(1)] = assign_match.group(2).strip()
            else:
                name_match = re.match(r'(\w+)', part)
                if name_match:
                    declared_scalars[name_match.group(1)] = None

    # Also find pre-loop assignments (initializations before the loop)
    # e.g., t = (real_t) 0.; or x = b[LEN_1D-1];
    # Only consider variables already declared
    scop_code = extract_scop_code(c_code)
    scop_loop_start = scop_code.find('for')
    if scop_loop_start > 0:
        pre_loop_scop = scop_code[:scop_loop_start]
        for var_name in declared_scalars:
            init_match = re.search(rf'\b{var_name}\s*=\s*([^;]+);', pre_loop_scop)
            if init_match:
                init_val = init_match.group(1).strip()
                # Clean up casts like (real_t) 0.
                init_val = re.sub(r'\(\s*(?:real_t|float|double|int)\s*\)\s*', '', init_val)
                declared_scalars[var_name] = init_val

    # Check each declared scalar for loop usage
    for var_name, init_value in declared_scalars.items():
        # Skip common non-scalar names (loop variables, array names, etc.)
        if var_name in ('i', 'j', 'k', 'n', 'N'):
            continue

        # Check if this variable is written to in the loop
        write_pattern = rf'\b{var_name}\s*[\+\-\*\/]?='
        has_writes = bool(re.search(write_pattern, loop_body))

        # Check if this variable is read in the loop (excluding the write itself)
        read_pattern = rf'(?<![=!<>])\b{var_name}\b(?!\s*[\+\-\*\/]?=)'
        has_reads = bool(re.search(read_pattern, loop_body))

        if has_writes and has_reads:
            candidates.append({
                'name': var_name,
                'init_value': init_value,
                'has_writes': True,
                'has_reads': True
            })

    return candidates


def analyze_scalar_updates(var_name: str, loop_body: str, loop_var: str) -> Dict:
    """
    Analyze how a scalar variable is updated in the loop.

    Returns:
        pattern_type: 'conditional_update', 'unconditional_update', 'accumulator'
        update_expr: The expression assigned to the scalar
        is_conditional: Whether update is inside an if statement
        depends_on_self: Whether update uses the scalar variable itself
        depends_on_loop_var: Whether update uses the loop variable
    """
    result = {
        'pattern_type': 'unknown',
        'update_expressions': [],
        'is_conditional': False,
        'depends_on_self': False,
        'depends_on_loop_var': False,
        'is_accumulator': False
    }

    # Find all assignments to this variable
    # Pattern: var = expr; or var op= expr; where op is +=, *=, etc.
    assign_pattern = rf'\b{var_name}\s*([\+\-\*\/]?=)\s*([^;]+);'

    first_plain_assign_pos = None
    first_compound_assign_pos = None
    first_plain_is_conditional = False

    for match in re.finditer(assign_pattern, loop_body):
        op = match.group(1)
        expr = match.group(2).strip()
        assign_pos = match.start()

        # Check if this is an accumulator pattern (+=, *=, etc.)
        if op in ['+=', '-=', '*=', '/=', '|=', '&=', '^=']:
            result['is_accumulator'] = True
            result['pattern_type'] = 'accumulator'
            if first_compound_assign_pos is None:
                first_compound_assign_pos = assign_pos
        else:
            result['pattern_type'] = 'assignment'
            if first_plain_assign_pos is None:
                first_plain_assign_pos = assign_pos

        result['update_expressions'].append({
            'operator': op,
            'expression': expr
        })

        # Check if update depends on the scalar itself
        if re.search(rf'\b{var_name}\b', expr):
            result['depends_on_self'] = True

        # Check if update depends on loop variable
        if re.search(rf'\b{loop_var}\b', expr):
            result['depends_on_loop_var'] = True

        # Check if update is conditional (inside if statement)
        # Look backwards for 'if (' before finding a '}' or ';'
        before_assign = loop_body[:assign_pos]

        # Count braces to determine nesting level
        # If we have unmatched '{' after an 'if', the assignment is conditional
        last_semicolon = before_assign.rfind(';')
        if last_semicolon == -1:
            recent_context = before_assign
        else:
            recent_context = before_assign[last_semicolon:]

        # Count opening and closing braces
        open_braces = recent_context.count('{')
        close_braces = recent_context.count('}')

        # If we have more open than close, and there's an 'if', it's conditional
        if open_braces > close_braces and 'if' in recent_context:
            result['is_conditional'] = True
            if result['pattern_type'] == 'assignment':
                result['pattern_type'] = 'conditional_update'
            # Track if the first plain assignment is conditional
            if assign_pos == first_plain_assign_pos:
                first_plain_is_conditional = True

    # If the variable has both plain '=' and compound 'op=' assignments,
    # and the first plain assignment comes BEFORE the first compound assignment
    # and is unconditional, then the variable is re-initialized each iteration —
    # it's NOT an outer-loop accumulator (the compound assignment is inner-loop only).
    # Example: w = A[i][j]; for(k) { w -= A[i][k]*A[k][j]; } — w is re-initialized.
    if (result['is_accumulator'] and
            first_plain_assign_pos is not None and
            first_compound_assign_pos is not None and
            first_plain_assign_pos < first_compound_assign_pos and
            not first_plain_is_conditional):
        result['is_accumulator'] = False
        result['pattern_type'] = 'assignment'

    return result


def analyze_scalar_reads(var_name: str, loop_body: str) -> Dict:
    """Analyze how a scalar variable is used (read) in the loop."""
    result = {
        'read_locations': [],
        'used_in_array_writes': False,
        'used_in_conditionals': False
    }

    # Find all reads of this variable (not on LHS of assignment)
    read_pattern = rf'(?<![=!<>])\b{var_name}\b(?!\s*=)'

    for match in re.finditer(read_pattern, loop_body):
        read_pos = match.start()
        context = loop_body[max(0, read_pos-50):read_pos+50]

        result['read_locations'].append({
            'position': read_pos,
            'context': context.strip()
        })

        # Check if used in array assignment (array[i] = ... var ...)
        if re.search(r'\w+\[[^\]]+\]\s*=.*\b' + var_name + r'\b', context):
            result['used_in_array_writes'] = True

    return result


def is_read_before_write(var_name: str, loop_body: str) -> bool:
    """
    Check if the first use of var_name in the loop body is a read (not a write).
    If so, the variable carries the previous iteration's value.
    """
    # Find all occurrences of the variable
    # Write: var_name = ... or var_name op= ...
    write_pattern = rf'\b{var_name}\s*[\+\-\*\/]?='
    # Read: var_name used in an expression (not LHS of assignment)
    read_pattern = rf'\b{var_name}\b'

    first_read_pos = None
    first_write_pos = None

    # Find first write position
    write_match = re.search(write_pattern, loop_body)
    if write_match:
        first_write_pos = write_match.start()

    # Find first read position (any occurrence that is not the write)
    for match in re.finditer(read_pattern, loop_body):
        pos = match.start()
        # Check if this position is a write (i.e., followed by =)
        after = loop_body[pos + len(var_name):].lstrip()
        if after and (after[0] == '=' and (len(after) < 2 or after[1] != '=')) or \
           (len(after) >= 2 and after[:2] in ['+=', '-=', '*=', '/=']):
            continue  # This is a write, skip
        first_read_pos = pos
        break

    if first_read_pos is not None and first_write_pos is not None:
        return first_read_pos < first_write_pos

    return False


def resolve_previous_value_expr(var_name: str, loop_body: str, loop_var: str,
                                 all_scalars: Dict[str, Dict], depth: int = 0) -> tuple:
    """
    Resolve what expression a previous-value variable corresponds to.

    Returns (shifted_expr, shift_amount) where shifted_expr is the expression
    with array indices shifted by shift_amount.

    all_scalars maps var_name -> {'assignment_expr': str, 'is_previous_value': bool}
    """
    if depth > 10:
        return (var_name, -1)

    info = all_scalars.get(var_name)
    if not info:
        return (var_name, -1)

    expr = info.get('assignment_expr', var_name)
    # The assignment is e.g. t = s, or x = b[i]
    # If expr is another scalar that is assigned in the loop, resolve recursively
    expr_stripped = expr.strip()

    # Check if the expression is just another scalar variable
    if expr_stripped in all_scalars and expr_stripped != var_name:
        sub_expr, sub_shift = resolve_previous_value_expr(
            expr_stripped, loop_body, loop_var, all_scalars, depth + 1
        )
        # Additional -1 shift since we're reading previous iteration's value of that scalar
        if all_scalars[expr_stripped].get('is_previous_value'):
            return (sub_expr, sub_shift - 1)
        else:
            return (sub_expr, -1)

    # The expression contains array accesses - shift them
    # e.g., b[i] * c[i] -> b[i-1] * c[i-1]
    return (expr_stripped, -1)


def shift_expression(expr: str, loop_var: str, shift: int) -> str:
    """Shift array indices in an expression by the given amount.

    e.g., shift_expression('b[i] * c[i]', 'i', -1) -> 'b[i-1] * c[i-1]'
    """
    def replace_index(m):
        arr_name = m.group(1)
        index_expr = m.group(2).strip()
        if index_expr == loop_var:
            if shift < 0:
                return f'{arr_name}[{loop_var}{shift}]'
            else:
                return f'{arr_name}[{loop_var}+{shift}]'
        else:
            return m.group(0)

    return re.sub(r'(\w+)\[([^\]]+)\]', replace_index, expr)


def determine_expansion_strategy(var_name: str, update_info: Dict, read_info: Dict,
                                  loop_body: str = None, loop_var: str = None,
                                  all_scalars: Dict = None) -> Dict:
    """
    Determine if scalar expansion is beneficial and provide strategy.
    """
    result = {
        'expansion_beneficial': False,
        'expansion_type': None,
        'strategy': None,
        'reason': None
    }

    if all_scalars is None:
        all_scalars = {}

    # Case 1: Accumulator pattern (s += ..., s = s + ...)
    # Check for induction variable (constant increment) first
    if update_info['is_accumulator']:
        # Check if this is an induction variable (constant increment)
        if update_info['update_expressions']:
            upd = update_info['update_expressions'][0]
            increment_expr = upd['expression'].strip()
            # Remove cast: (real_t)2. -> 2.
            clean_incr = re.sub(r'\((?:real_t|float|double|int)\)\s*', '', increment_expr)
            try:
                increment_val = float(clean_incr)
                result['expansion_beneficial'] = True
                result['expansion_type'] = 'induction_variable'
                result['strategy'] = 'Replace with closed-form expression'
                # Format increment nicely
                if increment_val == int(increment_val):
                    incr_str = str(int(increment_val))
                else:
                    incr_str = str(increment_val)
                result['reason'] = (
                    f"{var_name} is an induction variable with constant increment {incr_str}. "
                    f"Closed form: {var_name} = {incr_str}*({loop_var}+1)"
                )
                result['increment'] = incr_str
                result['closed_form'] = f"{incr_str}*({loop_var}+1)"
                return result
            except (ValueError, TypeError):
                pass

        result['expansion_beneficial'] = True
        result['expansion_type'] = 'prefix_sum'
        result['strategy'] = 'Use prefix sum / scan operation'
        result['reason'] = f"{var_name} is an accumulator - can compute as prefix sum then use in parallel"
        return result

    # Case 2: Conditional update that doesn't depend on itself
    is_conditionally_updated = (
        update_info['is_conditional'] or
        update_info['pattern_type'] == 'conditional_update'
    )

    if is_conditionally_updated and not update_info['depends_on_self']:
        if read_info['used_in_array_writes']:
            result['expansion_beneficial'] = True
            result['expansion_type'] = 'conditional_propagation'
            result['strategy'] = 'Expand to array with conditional scan'
            result['reason'] = (
                f"{var_name} is conditionally updated and used in array writes. "
                "Can expand to array s[i] and compute with conditional prefix scan."
            )
            return result

    # Case 2.5: Previous-value forwarding
    # Detect when scalar is read before written in the loop body, carrying
    # the previous iteration's value. This can be trivially replaced with
    # a shifted array expression.
    # Note: depends_on_loop_var may be False if assignment is via another scalar
    # (e.g., t = s where s = b[i]*c[i]), so we check indirect dependency too.
    if (not is_conditionally_updated and
            not update_info['depends_on_self'] and
            read_info['used_in_array_writes'] and
            loop_body is not None and
            is_read_before_write(var_name, loop_body)):

        # Resolve what expression this variable carries from previous iteration
        resolved_expr, shift = resolve_previous_value_expr(
            var_name, loop_body, loop_var or 'i', all_scalars
        )
        shifted = shift_expression(resolved_expr, loop_var or 'i', shift)

        result['expansion_beneficial'] = True
        result['expansion_type'] = 'previous_value'
        result['strategy'] = 'Replace with shifted expression (previous-value forwarding)'
        result['reason'] = (
            f"{var_name} carries the previous iteration's value. "
            f"Can be replaced with {shifted}."
        )
        result['resolved_expr'] = resolved_expr
        result['shifted_expr'] = shifted
        result['shift'] = shift
        return result

    # Case 3: Unconditional assignment that depends on loop variable
    if not is_conditionally_updated and update_info['depends_on_loop_var'] and not update_info['depends_on_self']:
        if read_info['used_in_array_writes']:
            # If variable is written before read (same-iteration temporary),
            # it's a direct expansion, not a wrap-around
            if loop_body and not is_read_before_write(var_name, loop_body):
                result['expansion_beneficial'] = True
                result['expansion_type'] = 'direct_expansion'
                result['strategy'] = 'Direct array expansion (same-iteration temporary)'
                result['reason'] = f"{var_name} = expr[i] is a same-iteration temporary, directly expandable"
                return result
            result['expansion_beneficial'] = True
            result['expansion_type'] = 'conditional_propagation'
            result['strategy'] = 'Expand to array with forward propagation'
            result['reason'] = (
                f"{var_name} value wraps around across iterations. "
                "Expand to array and propagate values forward."
            )
            return result
        else:
            result['expansion_beneficial'] = True
            result['expansion_type'] = 'direct_expansion'
            result['strategy'] = 'Direct array expansion'
            result['reason'] = f"{var_name} = expr[i] can be directly expanded to array"
            return result

    # Case 4: Complex dependency (depends on self, not accumulator)
    # Harder to parallelize
    if update_info['depends_on_self'] and not update_info['is_accumulator']:
        result['expansion_beneficial'] = False
        result['reason'] = f"{var_name} has complex self-dependency"
        return result

    return result


def analyze_kernel_scalar_expansion(kernel_file: str) -> Optional[Dict]:
    """Analyze a kernel for scalar expansion opportunities."""
    if not os.path.exists(kernel_file):
        return None

    with open(kernel_file, 'r') as f:
        c_code = f.read()

    scop_code = extract_scop_code(c_code)
    loop_info = extract_loop_info(scop_code)

    if not loop_info:
        return None

    loop_var = loop_info['var']
    loop_body = loop_info['body']

    # Detect scalar variables that are candidates for expansion
    candidates = detect_scalar_variables(c_code, loop_body)

    if not candidates:
        return {
            'has_scalar_expansion': False,
            'candidates': []
        }

    # Analyze each candidate
    expansion_opportunities = []

    # Build all_scalars info for cross-variable resolution
    all_scalars = {}
    for candidate in candidates:
        var_name = candidate['name']
        update_info = analyze_scalar_updates(var_name, loop_body, loop_var)
        # Find the assignment expression
        assignment_expr = ''
        if update_info['update_expressions']:
            assignment_expr = update_info['update_expressions'][0]['expression']
        all_scalars[var_name] = {
            'assignment_expr': assignment_expr,
            'is_previous_value': is_read_before_write(var_name, loop_body),
            'update_info': update_info,
        }

    for candidate in candidates:
        var_name = candidate['name']

        # Analyze how the scalar is updated
        update_info = all_scalars[var_name]['update_info']

        # Analyze how the scalar is read
        read_info = analyze_scalar_reads(var_name, loop_body)

        # Determine expansion strategy
        strategy = determine_expansion_strategy(var_name, update_info, read_info,
                                                 loop_body=loop_body, loop_var=loop_var,
                                                 all_scalars=all_scalars)

        if strategy['expansion_beneficial']:
            opp = {
                'variable': var_name,
                'init_value': candidate['init_value'],
                'update_info': update_info,
                'read_info': read_info,
                'expansion_type': strategy['expansion_type'],
                'strategy': strategy['strategy'],
                'reason': strategy['reason']
            }
            # Pass through extra fields for specific expansion types
            for key in ('resolved_expr', 'shifted_expr', 'shift',
                        'increment', 'closed_form'):
                if key in strategy:
                    opp[key] = strategy[key]
            expansion_opportunities.append(opp)

    return {
        'has_scalar_expansion': len(expansion_opportunities) > 0,
        'candidates': expansion_opportunities,
        'loop_var': loop_var,
        'loop_start': loop_info['start'],
        'loop_end': loop_info['end']
    }


def format_scalar_expansion_for_prompt(kernel_name: str, expansion_result: Optional[Dict]) -> str:
    """Format scalar expansion analysis for LLM prompt."""
    if not expansion_result or not expansion_result.get('has_scalar_expansion'):
        return ""

    lines = []
    lines.append("=" * 80)
    lines.append("SCALAR EXPANSION PATTERN DETECTED")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Kernel {kernel_name} contains scalar variables that create loop-carried")
    lines.append("dependencies but can be PARALLELIZED through SCALAR EXPANSION.")
    lines.append("")

    for opportunity in expansion_result['candidates']:
        var = opportunity['variable']
        exp_type = opportunity['expansion_type']
        init = opportunity['init_value'] or '0'

        lines.append(f"Variable: {var}")
        lines.append(f"  Initial value: {init}")
        lines.append(f"  Expansion type: {exp_type}")
        lines.append(f"  Reason: {opportunity['reason']}")
        lines.append("")

        if exp_type == 'conditional_propagation':
            lines.append(f"  IMPLEMENTATION STRATEGY for {var}:")
            lines.append("  " + "=" * 70)
            lines.append("")
            lines.append("  Step 1: Expand scalar to array")
            lines.append(f"    Create array: {var}_expanded = torch.zeros(N)")
            lines.append("")
            lines.append("  Step 2: Compute expanded array with conditional scan")
            lines.append("    For each position i:")
            lines.append(f"      - If condition is true: {var}_expanded[i] = new_value")
            lines.append(f"      - Else: {var}_expanded[i] = {var}_expanded[i-1] (carry forward)")
            lines.append("")
            lines.append("    Triton implementation:")
            lines.append("    ```python")
            lines.append("    # Phase 1: Compute expanded scalar array")
            lines.append("    # Use sequential loop (one thread) OR specialized scan algorithm")
            lines.append("    @triton.jit")
            lines.append(f"    def expand_{var}_kernel(...):")
            lines.append("        # Single thread processes all elements sequentially")
            lines.append(f"        {var}_val = 0.0  # Initial value")
            lines.append("        for i in range(N):")
            lines.append("            # Load condition and data")
            lines.append("            cond = ...")
            lines.append("            if cond:")
            lines.append(f"                {var}_val = ...  # Update value")
            lines.append(f"            tl.store({var}_expanded_ptr + i, {var}_val)")
            lines.append("    ")
            lines.append("    # Phase 2: Use expanded array in parallel")
            lines.append("    @triton.jit")
            lines.append("    def compute_kernel(...):")
            lines.append("        # Multiple threads process elements in parallel")
            lines.append("        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)")
            lines.append(f"        {var}_vals = tl.load({var}_expanded_ptr + offsets, mask=mask)")
            lines.append("        # Now use expanded values in parallel")
            lines.append("        ...")
            lines.append("    ```")
            lines.append("")
            lines.append("  Step 3: Use expanded array in original computations")
            lines.append(f"    Replace all uses of scalar '{var}' with '{var}_expanded[i]'")
            lines.append("    Now the array writes can be done in PARALLEL!")
            lines.append("")

        elif exp_type == 'prefix_sum':
            lines.append(f"  IMPLEMENTATION STRATEGY for {var}:")
            lines.append("  " + "=" * 70)
            lines.append("")
            lines.append("  This is a PREFIX SUM / CUMULATIVE SUM pattern.")
            lines.append("  Use Triton's or PyTorch's cumsum operation:")
            lines.append("")
            lines.append("  ```python")
            lines.append("  # Compute prefix sum")
            lines.append(f"  {var}_expanded = torch.cumsum(input_array, dim=0)")
            lines.append("")
            lines.append("  # Then use in parallel kernel")
            lines.append("  @triton.jit")
            lines.append("  def compute_kernel(...):")
            lines.append("      offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)")
            lines.append(f"      {var}_vals = tl.load({var}_expanded_ptr + offsets, mask=mask)")
            lines.append("      # Parallel computation using prefix sum values")
            lines.append("      ...")
            lines.append("  ```")
            lines.append("")

        elif exp_type == 'previous_value':
            shifted_expr = opportunity.get('shifted_expr', '?')
            resolved_expr = opportunity.get('resolved_expr', '?')
            shift = opportunity.get('shift', -1)
            loop_var = expansion_result.get('loop_var', 'i')

            lines.append(f"  Pattern: PREVIOUS-VALUE FORWARDING (trivially eliminable)")
            lines.append(f"  {var} at iteration {loop_var} equals the value of ({shifted_expr}) from the previous iteration.")
            lines.append(f"  Initial value ({loop_var}=0): {var} = {init}")
            lines.append("")
            lines.append(f"  SIMPLIFICATION: Replace {var} with the shifted expression:")
            lines.append(f"    {var} → {shifted_expr}  (for {loop_var} > {-shift - 1})")
            lines.append(f"    {var} → {init}              (for {loop_var} = 0, initial value)")
            lines.append("")
            lines.append(f"  After substitution, the loop becomes FULLY PARALLEL.")
            lines.append(f"  No scan or sequential phase is needed!")
            lines.append("")

        elif exp_type == 'induction_variable':
            closed_form = opportunity.get('closed_form', '?')
            increment = opportunity.get('increment', '?')
            loop_var = expansion_result.get('loop_var', 'i')

            lines.append(f"  Pattern: INDUCTION VARIABLE (trivially eliminable)")
            lines.append(f"  {var} increments by {increment} each iteration.")
            lines.append(f"  Closed form: {var} = {closed_form}")
            lines.append(f"  Initial value: {init}")
            lines.append("")
            lines.append(f"  SIMPLIFICATION: Replace {var} with closed-form expression:")
            lines.append(f"    {var} → {closed_form}")
            lines.append("")
            lines.append(f"  After substitution, the loop becomes FULLY PARALLEL.")
            lines.append(f"  No scan or sequential phase is needed!")
            lines.append("")

        elif exp_type == 'direct_expansion':
            lines.append(f"  IMPLEMENTATION STRATEGY for {var}:")
            lines.append("  " + "=" * 70)
            lines.append("")
            lines.append("  This can be DIRECTLY expanded - no scan needed!")
            lines.append(f"  Simply replace '{var}' with the indexed expression.")
            lines.append("")

    lines.append("=" * 80)
    lines.append("")

    return "\n".join(lines)


def main():
    """Test scalar expansion detection on relevant kernels."""
    # s258: wrap-around scalar with conditional update
    # s253: similar pattern
    test_kernels = ['s258', 's253', 's257', 's251', 's252', 's254', 's255', 's453']

    print("=" * 80)
    print("SCALAR EXPANSION PATTERN DETECTION")
    print("=" * 80)

    for kernel_name in test_kernels:
        kernel_file = os.path.join(KERNELS_DIR, f"{kernel_name}.c")
        if not os.path.exists(kernel_file):
            print(f"\n{kernel_name}: file not found")
            continue

        print(f"\n{'=' * 80}")
        print(f"Kernel: {kernel_name}")
        print(f"{'=' * 80}")

        # Read and display C code
        with open(kernel_file, 'r') as f:
            c_code = f.read()
        scop_match = re.search(r'#pragma scop\s*(.*?)\s*#pragma endscop', c_code, re.DOTALL)
        if scop_match:
            print(f"\nC Code:\n{scop_match.group(1).strip()}")

        result = analyze_kernel_scalar_expansion(kernel_file)
        if result:
            if result['has_scalar_expansion']:
                print(f"\n✓ SCALAR EXPANSION OPPORTUNITIES FOUND!")
                print(f"Number of candidates: {len(result['candidates'])}")

                for opp in result['candidates']:
                    print(f"\n  Variable: {opp['variable']}")
                    print(f"    Expansion type: {opp['expansion_type']}")
                    print(f"    Strategy: {opp['strategy']}")
                    print(f"    Reason: {opp['reason']}")

                # Show formatted prompt
                prompt_text = format_scalar_expansion_for_prompt(kernel_name, result)
                print(f"\n{prompt_text}")
            else:
                print("\n✗ No scalar expansion opportunities found")
        else:
            print("\nFailed to analyze kernel")


if __name__ == "__main__":
    main()
