#!/usr/bin/env python3
"""
Convert goto-based control flow to equivalent if/else structure.

TSVC uses goto patterns for conditional execution. PET cannot analyze goto,
so we convert common patterns to equivalent if/else code that PET can handle.

Common patterns:
1. Forward skip: if (cond) goto L; stmts; L: -> if (!cond) { stmts; }
2. Multi-level: if (c1) goto L2; if (c2) goto L1; A; L1: B; L2: -> nested if/else
"""

import re
import os

KERNELS_DIR = "/home/qinxiao/workspace/compiler-guided-triton-gen/analysis/kernels"


def detect_goto_pattern(c_code: str) -> dict:
    """
    Detect goto patterns in C code and return pattern info.

    Returns:
        dict with:
        - 'has_goto': bool
        - 'pattern_type': str or None
        - 'labels': list of label names
        - 'goto_conditions': list of (condition, label) tuples
    """
    result = {
        'has_goto': False,
        'pattern_type': None,
        'labels': [],
        'goto_conditions': [],
        'convertible': False
    }

    # Check for goto statements
    goto_pattern = r'goto\s+(\w+)\s*;'
    goto_matches = re.findall(goto_pattern, c_code)

    if not goto_matches:
        return result

    result['has_goto'] = True
    result['labels'] = list(set(goto_matches))

    # Extract goto conditions: if (cond) { goto label; } or if (cond) goto label;
    # Use .*? for condition to handle nested parentheses like (real_t)
    # Pattern 1: if (cond) { goto label; }
    cond_goto_pattern1 = r'if\s*\((.*?)\)\s*\{\s*goto\s+(\w+)\s*;\s*\}'
    for match in re.finditer(cond_goto_pattern1, c_code, re.DOTALL):
        cond = match.group(1)
        label = match.group(2)
        result['goto_conditions'].append((cond.strip(), label))

    # Pattern 2: if (cond) goto label; (no braces)
    cond_goto_pattern2 = r'if\s*\((.*?)\)\s*goto\s+(\w+)\s*;'
    for match in re.finditer(cond_goto_pattern2, c_code, re.DOTALL):
        cond = match.group(1)
        label = match.group(2)
        # Avoid duplicates
        entry = (cond.strip(), label)
        if entry not in result['goto_conditions']:
            result['goto_conditions'].append(entry)

    # Detect pattern type
    if len(result['labels']) == 1:
        result['pattern_type'] = 'single_label_forward_skip'
        result['convertible'] = True
    elif len(result['labels']) == 2:
        result['pattern_type'] = 'two_label_nested'
        result['convertible'] = True
    else:
        result['pattern_type'] = 'complex'
        result['convertible'] = False

    return result


def convert_goto_to_ifelse(c_code: str) -> tuple:
    """
    Convert goto-based code to equivalent if/else structure.

    Returns:
        (converted_code, success, description)
    """
    pattern_info = detect_goto_pattern(c_code)

    if not pattern_info['has_goto']:
        return c_code, True, "No goto statements found"

    if not pattern_info['convertible']:
        return c_code, False, f"Complex goto pattern not supported: {pattern_info['pattern_type']}"

    # Handle s277-style pattern:
    # if (a[i] >= 0.) goto L20;
    # if (b[i] >= 0.) goto L30;
    # stmt_A;
    # L30: stmt_B;
    # L20: ;

    if pattern_info['pattern_type'] == 'two_label_nested':
        return _convert_two_label_pattern(c_code, pattern_info)
    elif pattern_info['pattern_type'] == 'single_label_forward_skip':
        return _convert_single_label_pattern(c_code, pattern_info)

    return c_code, False, "Unknown pattern"


def _convert_single_label_pattern(c_code: str, pattern_info: dict) -> tuple:
    """
    Convert: if (cond) goto L; stmts; L: ;
    To: if (!(cond)) { stmts; }
    """
    if not pattern_info['labels']:
        return c_code, False, "No labels found"
    if not pattern_info['goto_conditions']:
        return c_code, False, "No goto conditions found (goto might not be conditional)"

    label = pattern_info['labels'][0]
    cond, _ = pattern_info['goto_conditions'][0]

    # Find the goto statement
    goto_pattern = rf'if\s*\([^)]*\)\s*(?:\{{\s*)?goto\s+{label}\s*;(?:\s*\}})?'

    # Find the label
    label_pattern = rf'{label}\s*:\s*;?'

    # Extract code between goto and label
    match = re.search(goto_pattern, c_code)
    label_match = re.search(label_pattern, c_code)

    if not match or not label_match:
        return c_code, False, "Could not find goto or label"

    # Get the statements between goto and label
    between_start = match.end()
    between_end = label_match.start()
    stmts = c_code[between_start:between_end].strip()

    # Negate the condition
    negated_cond = negate_condition(cond)

    # Build converted code
    before = c_code[:match.start()]
    after = c_code[label_match.end():]

    converted = f"{before}if ({negated_cond}) {{\n    {stmts}\n}}{after}"

    return converted, True, f"Converted single-label pattern with condition: {cond}"


def _convert_two_label_pattern(c_code: str, pattern_info: dict) -> tuple:
    """
    Convert s277-style pattern:
        if (cond1) goto L2;
        if (cond2) goto L1;
        stmt_A;
        L1: stmt_B;
        L2: ;
    To:
        if (!(cond1)) {
            if (!(cond2)) {
                stmt_A;
            }
            stmt_B;
        }
    """
    gotos = pattern_info['goto_conditions']

    if len(gotos) < 2:
        return c_code, False, "Expected 2 goto conditions"

    cond1, label1 = gotos[0]  # First condition (outermost skip)
    cond2, label2 = gotos[1]  # Second condition (inner skip)

    # Determine which label is the outer one (L20 in s277)
    outer_label = label1
    inner_label = label2

    # Find specific goto statements using the exact condition text
    # Escape special regex chars in condition
    cond1_escaped = re.escape(cond1)
    cond2_escaped = re.escape(cond2)

    # Patterns that match the specific conditions
    goto1_pattern = rf'if\s*\({cond1_escaped}\)\s*\{{\s*goto\s+{outer_label}\s*;\s*\}}'
    goto2_pattern = rf'if\s*\({cond2_escaped}\)\s*\{{\s*goto\s+{inner_label}\s*;\s*\}}'
    inner_label_pattern = rf'{inner_label}\s*:'
    outer_label_pattern = rf'{outer_label}\s*:\s*;?'

    goto1_match = re.search(goto1_pattern, c_code, re.DOTALL)
    goto2_match = re.search(goto2_pattern, c_code, re.DOTALL)
    inner_label_match = re.search(inner_label_pattern, c_code)
    outer_label_match = re.search(outer_label_pattern, c_code)

    if not all([goto1_match, goto2_match, inner_label_match, outer_label_match]):
        return c_code, False, "Could not find all pattern components"

    # Extract statements
    # stmt_A is between goto2 and inner_label
    # stmt_B is between inner_label and outer_label
    stmt_a_start = goto2_match.end()
    stmt_a_end = inner_label_match.start()
    stmt_a = c_code[stmt_a_start:stmt_a_end].strip()

    stmt_b_start = inner_label_match.end()
    # Skip the colon
    if c_code[stmt_b_start:stmt_b_start+1] == ':':
        stmt_b_start += 1
    stmt_b_end = outer_label_match.start()
    stmt_b = c_code[stmt_b_start:stmt_b_end].strip()

    # Negate conditions
    neg_cond1 = negate_condition(cond1)
    neg_cond2 = negate_condition(cond2)

    # Build converted code
    before = c_code[:goto1_match.start()]
    after = c_code[outer_label_match.end():]

    # Handle the trailing semicolon after outer label
    after = after.lstrip()
    if after.startswith(';'):
        after = after[1:]

    converted = f"""{before}if ({neg_cond1}) {{
        if ({neg_cond2}) {{
            {stmt_a}
        }}
        {stmt_b}
    }}{after}"""

    return converted, True, f"Converted two-label pattern"


def negate_condition(cond: str) -> str:
    """Negate a C condition expression."""
    cond = cond.strip()

    # Handle common patterns
    if cond.startswith('!'):
        # !(x) -> x or !x -> x
        if cond.startswith('!(') and cond.endswith(')'):
            return cond[2:-1]
        return cond[1:]

    # Handle comparison operators
    if ' >= ' in cond:
        return cond.replace(' >= ', ' < ')
    if ' > ' in cond:
        return cond.replace(' > ', ' <= ')
    if ' <= ' in cond:
        return cond.replace(' <= ', ' > ')
    if ' < ' in cond:
        return cond.replace(' < ', ' >= ')
    if ' == ' in cond:
        return cond.replace(' == ', ' != ')
    if ' != ' in cond:
        return cond.replace(' != ', ' == ')

    # Default: wrap with !()
    return f'!({cond})'


def analyze_goto_parallelization(c_code: str) -> dict:
    """
    Analyze goto-based code for parallelization opportunities.

    Key insight: If the condition for writing to array[i+k] only depends on
    values at index i (not i-1, i+1, etc.), then the loop can be parallelized
    by precomputing effective values.

    Returns:
        dict with parallelization advice
    """
    pattern_info = detect_goto_pattern(c_code)

    if not pattern_info['has_goto']:
        return {'applicable': False, 'reason': 'No goto statements'}

    converted, success, desc = convert_goto_to_ifelse(c_code)

    if not success:
        return {'applicable': False, 'reason': desc}

    # Analyze the converted code for parallelization
    # Look for patterns where:
    # 1. arr[i+1] is written conditionally
    # 2. arr[i] is read in a condition
    # 3. The write condition only depends on other arrays at index i

    result = {
        'applicable': True,
        'original_code': c_code,
        'converted_code': converted,
        'pattern_info': pattern_info,
        'parallelization_strategy': None
    }

    # Check for forward write pattern (arr[i+1] = ...)
    forward_write = re.search(r'(\w+)\[i\s*\+\s*1\]\s*=', converted)
    # Check for same-array condition read (arr[i] in condition)

    if forward_write:
        arr_name = forward_write.group(1)
        # Check if this array is also read in a condition
        cond_read = re.search(rf'if\s*\([^)]*{arr_name}\[i\][^)]*\)', converted)

        if cond_read:
            result['parallelization_strategy'] = f"""
## GOTO-BASED CONDITIONAL PARALLELIZATION

This loop contains goto-based control flow that APPEARS sequential but CAN be parallelized.

### Equivalent if/else structure:
```c
{converted}
```

### Key Insight:
The array `{arr_name}[i+1]` is written conditionally, and `{arr_name}[i]` is read in a later condition.
However, the WRITE CONDITION only depends on other arrays, NOT on previous `{arr_name}` values.

### Parallelization Strategy:
1. **Compute effective_{arr_name}[i]**: For each i, compute what {arr_name}[i] would be AFTER the previous iteration:
   - If the write condition was true at i-1: effective_{arr_name}[i] = computed_value[i-1]
   - Else: effective_{arr_name}[i] = original_{arr_name}[i]

2. **Use effective values for conditions**: Replace {arr_name}[i] reads in conditions with effective_{arr_name}[i]

3. **Parallel execution**: Both the effective value computation and final updates can be done in parallel.

### Implementation Pattern:
```python
# Step 1: Compute write mask (which iterations write to {arr_name})
write_mask = <condition that triggers {arr_name}[i+1] write>  # e.g., a < 0

# Step 2: Compute effective {arr_name} values
effective_{arr_name} = {arr_name}.clone()
new_values = <expression for {arr_name}[i+1]>
effective_{arr_name}[1:] = tl.where(write_mask[:-1], new_values[:-1], {arr_name}[1:])

# Step 3: Use effective values for other conditions and updates
# All updates can now be done in parallel using the precomputed effective values
```
"""

    return result


def format_goto_analysis_for_prompt(analysis_result: dict) -> str:
    """Format goto analysis for inclusion in LLM prompt."""
    if not analysis_result.get('applicable'):
        return ""

    strategy = analysis_result.get('parallelization_strategy')
    if strategy:
        return strategy

    # Basic conversion info
    converted = analysis_result.get('converted_code', '')
    if converted:
        return f"""
## GOTO CONTROL FLOW DETECTED

This code uses goto statements. Equivalent if/else structure:
```c
{converted}
```

Analyze the converted structure for parallelization opportunities.
"""

    return ""


def analyze_kernel_goto(kernel_name: str) -> dict:
    """
    Analyze a kernel for goto patterns and parallelization.

    Args:
        kernel_name: Name of the kernel (e.g., 's277')

    Returns:
        dict with analysis results
    """
    kernel_file = os.path.join(KERNELS_DIR, f"{kernel_name}.c")
    if not os.path.exists(kernel_file):
        return None

    with open(kernel_file, 'r') as f:
        c_code = f.read()

    # Extract the loop body
    scop_match = re.search(r'#pragma scop\s*(.*?)\s*#pragma endscop', c_code, re.DOTALL)
    if scop_match:
        loop_code = scop_match.group(1)
    else:
        loop_code = c_code

    return analyze_goto_parallelization(loop_code)


if __name__ == "__main__":
    # Test with s277
    result = analyze_kernel_goto('s277')
    if result:
        print("Applicable:", result.get('applicable'))
        print("\nParallelization strategy:")
        print(result.get('parallelization_strategy', 'None'))
    else:
        print("Analysis failed")
