#!/usr/bin/env python3
"""
Detect early-exit patterns (break, goto, exit) in loops.

This analysis identifies patterns where loops have data-dependent early exits
that prevent naive parallelization across blocks.

Example s482:
    for (int i = 0; i < LEN_1D; i++) {
        a[i] += b[i] * c[i];
        if (c[i] > b[i]) break;  // Early exit!
    }

Example s481:
    for (int i = 0; i < LEN_1D; i++) {
        if (d[i] < 0.0) exit(0);  // Early exit!
        a[i] += b[i] * c[i];
    }

Key characteristics:
1. Loop has a conditional that can terminate iteration early
2. The condition depends on data values (not just loop bounds)
3. Subsequent iterations should NOT execute after the exit condition

CRITICAL: These patterns CANNOT be naively parallelized because:
- Each Triton block operates independently
- Block N doesn't know if Block 0 encountered the exit condition
- Must find global exit point FIRST, then process only valid elements

Correct implementation:
1. Phase 1: Find the global exit index (where condition first becomes true)
2. Phase 2: Process only elements 0..exit_index (inclusive for break-after-compute)
"""

import re
import os

KERNELS_DIR = "/home/qinxiao/workspace/compiler-guided-triton-gen/analysis/kernels"


def parse_kernel_file(kernel_file):
    """Parse a kernel file to extract the scop region."""
    if not os.path.exists(kernel_file):
        return None

    with open(kernel_file, 'r') as f:
        content = f.read()

    # Extract the scop region
    scop_match = re.search(r'#pragma scop\s*(.*?)\s*#pragma endscop', content, re.DOTALL)
    if not scop_match:
        return None

    return scop_match.group(1).strip()


def detect_break_pattern(code):
    """
    Detect break statements in loops.

    Returns dict with:
    - 'has_break': bool
    - 'condition': str (the condition that triggers break)
    - 'before_break': bool (is computation before break?)
    """
    # Pattern: if (condition) break;
    break_pattern = r'if\s*\(([^)]+)\)\s*break\s*;'
    match = re.search(break_pattern, code)

    if match:
        condition = match.group(1).strip()

        # Check if there's computation before the break
        # Look for assignment before the if statement
        lines = code.split('\n')
        before_break = False
        for i, line in enumerate(lines):
            if 'break' in line:
                # Check previous lines for assignments
                for j in range(i):
                    if '+=' in lines[j] or '=' in lines[j] and 'if' not in lines[j]:
                        before_break = True
                        break
                break

        return {
            'has_break': True,
            'condition': condition,
            'before_break': before_break,
            'exit_type': 'break'
        }

    return {'has_break': False}


def detect_exit_pattern(code):
    """
    Detect exit() calls in loops.

    Returns dict with:
    - 'has_exit': bool
    - 'condition': str (the condition that triggers exit)
    - 'before_exit': bool (is computation before exit?)
    """
    # Check if exit exists in code
    if 'exit' not in code:
        return {'has_exit': False}

    # Find the exit line and look for the corresponding if condition
    lines = code.split('\n')
    condition = None
    exit_line_idx = None

    for i, line in enumerate(lines):
        if 'exit' in line and '(' in line:
            exit_line_idx = i
            # Look back for the if statement
            for j in range(i - 1, -1, -1):
                if 'if' in lines[j] and '(' in lines[j]:
                    # Extract condition from if statement
                    if_line = lines[j]
                    # Find the condition between if ( and ) {
                    # Handle nested parentheses by finding matching parens
                    start = if_line.find('(')
                    if start != -1:
                        depth = 0
                        end = start
                        for k, ch in enumerate(if_line[start:], start):
                            if ch == '(':
                                depth += 1
                            elif ch == ')':
                                depth -= 1
                                if depth == 0:
                                    end = k
                                    break
                        condition = if_line[start + 1:end].strip()
                    break
            break

    if condition:
        # Check if there's computation before the exit check (if statement)
        # We need to find if any assignment happens BEFORE the if statement in the loop body
        before_exit = False

        # Find the if line that contains the condition
        if_line_idx = None
        for i, line in enumerate(lines):
            if 'if' in line and condition.replace(' ', '') in line.replace(' ', ''):
                if_line_idx = i
                break

        if if_line_idx is not None:
            # Check if there's an assignment between the loop start and the if
            for j in range(if_line_idx):
                line = lines[j]
                # Look for assignment that's not the for loop or if statement
                if ('+=' in line or '-=' in line or '*=' in line) and 'for' not in line:
                    before_exit = True
                    break
                if '=' in line and 'for' not in line and 'if' not in line and '==' not in line and '!=' not in line and '<=' not in line and '>=' not in line:
                    before_exit = True
                    break

        return {
            'has_exit': True,
            'condition': condition,
            'before_exit': before_exit,
            'exit_type': 'exit'
        }

    return {'has_exit': False}


def detect_goto_pattern(code):
    """
    Detect goto statements that exit loops.

    Returns dict with:
    - 'has_goto': bool
    - 'condition': str (the condition that triggers goto)
    - 'label': str (the goto target label)
    """
    # Pattern: if (condition) goto LABEL;
    goto_pattern = r'if\s*\(([^)]+)\)\s*goto\s+(\w+)\s*;'
    match = re.search(goto_pattern, code)

    if match:
        condition = match.group(1).strip()
        label = match.group(2).strip()

        return {
            'has_goto': True,
            'condition': condition,
            'label': label,
            'exit_type': 'goto'
        }

    return {'has_goto': False}


def analyze_early_exit(kernel_name):
    """
    Analyze a kernel for early-exit patterns.

    Returns dict with:
    - 'applicable': bool
    - 'exit_type': str ('break', 'exit', 'goto')
    - 'condition': str
    - 'before_exit': bool
    - 'advice': str
    """
    kernel_file = os.path.join(KERNELS_DIR, f"{kernel_name}.c")
    code = parse_kernel_file(kernel_file)

    if not code:
        return {'applicable': False}

    # Check for different exit patterns
    break_info = detect_break_pattern(code)
    exit_info = detect_exit_pattern(code)
    goto_info = detect_goto_pattern(code)

    result = {'applicable': False}

    if break_info['has_break']:
        result = {
            'applicable': True,
            'exit_type': 'break',
            'condition': break_info['condition'],
            'before_exit': break_info['before_break'],
            'advice': generate_early_exit_advice('break', break_info['condition'], break_info['before_break'])
        }
    elif exit_info['has_exit']:
        result = {
            'applicable': True,
            'exit_type': 'exit',
            'condition': exit_info['condition'],
            'before_exit': exit_info['before_exit'],
            'advice': generate_early_exit_advice('exit', exit_info['condition'], exit_info['before_exit'])
        }
    elif goto_info['has_goto']:
        result = {
            'applicable': True,
            'exit_type': 'goto',
            'condition': goto_info['condition'],
            'label': goto_info.get('label'),
            'before_exit': False,  # Assume computation after goto target
            'advice': generate_early_exit_advice('goto', goto_info['condition'], False, goto_info.get('label'))
        }

    return result


def generate_early_exit_advice(exit_type, condition, before_exit, label=None):
    """Generate advice for handling early-exit patterns."""

    exit_desc = {
        'break': 'break statement',
        'exit': 'exit() call',
        'goto': f'goto {label}' if label else 'goto statement'
    }

    advice = f"""DATA-DEPENDENT EARLY EXIT PATTERN DETECTED

This loop has a conditional {exit_desc[exit_type]} that terminates iteration early:
  - Exit condition: {condition}
  - Computation before exit: {'YES' if before_exit else 'NO'}

Original pattern:
```c
for (int i = 0; i < N; i++) {{
    {'// computation here' if before_exit else ''}
    if ({condition}) {exit_type};{'  // exit AFTER computation' if before_exit else '  // exit BEFORE computation'}
    {'// computation here' if not before_exit else ''}
}}
```

CRITICAL: This pattern CANNOT be naively parallelized!

**Why naive parallelization fails:**
- Each Triton block operates independently
- If the exit condition is true at index 5, blocks processing indices 256+, 512+, etc.
  don't know about it and will incorrectly process their elements
- Result: elements after the exit point are incorrectly modified

**Correct implementation (TWO-PHASE approach):**

```python
def kernel_triton(a, b, c, ...):
    # PHASE 1: Find the global exit index
    # Find first index where exit condition is true
    condition_mask = {condition.replace('[i]', '')}  # vectorized condition

    if torch.any(condition_mask):
        exit_idx = torch.argmax(condition_mask.int()).item()

        # PHASE 2: Process only elements 0..exit_idx {'(inclusive)' if before_exit else '(exclusive)'}
        {'# Include exit_idx because computation happens BEFORE exit check' if before_exit else '# Exclude exit_idx because exit check happens BEFORE computation'}
        valid_range = {'exit_idx + 1' if before_exit else 'exit_idx'}
        a[:valid_range] += b[:valid_range] * c[:valid_range]
    else:
        # No exit condition triggered, process all elements
        a[:] += b * c
```

**Key points:**
1. MUST find global exit point FIRST (cannot parallelize this search naively)
2. THEN process only the valid range
3. {'Include' if before_exit else 'Exclude'} the exit index element based on computation order

**DO NOT:**
- Try to handle exit within each Triton block independently
- Use block-local exit detection (cross-block communication needed)
- Ignore the early-exit semantics"""

    return advice


def format_early_exit_for_prompt(result):
    """Format early-exit analysis for inclusion in LLM prompt."""
    if not result or not result.get('applicable'):
        return None

    lines = []
    lines.append("=" * 60)
    lines.append("EARLY EXIT PATTERN DETECTED")
    lines.append("=" * 60)
    lines.append("")

    if result.get('advice'):
        lines.append(result['advice'])

    lines.append("")
    lines.append("=" * 60)

    return '\n'.join(lines)


def analyze_kernel_early_exit(kernel_name):
    """
    Main entry point for early-exit analysis.

    Args:
        kernel_name: Name of the kernel (e.g., 's482')

    Returns:
        dict with analysis results, or None if not applicable
    """
    result = analyze_early_exit(kernel_name)
    if result.get('applicable'):
        return result
    return None


def main():
    """Test early-exit detection on known kernels."""
    test_kernels = ['s481', 's482', 's000', 's111']

    print("=" * 80)
    print("EARLY EXIT PATTERN DETECTION")
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
        code = parse_kernel_file(kernel_file)
        if code:
            lines = code.split('\n')[:10]
            print(f"\nC Code (first 10 lines):")
            for line in lines:
                print(f"  {line}")

        result = analyze_kernel_early_exit(kernel)
        if result:
            print(f"\nDetected: YES")
            print(f"  Exit type: {result.get('exit_type')}")
            print(f"  Condition: {result.get('condition')}")
            print(f"  Before exit: {result.get('before_exit')}")

            formatted = format_early_exit_for_prompt(result)
            if formatted:
                print(f"\n{formatted}")
        else:
            print("\nDetected: NO")


if __name__ == "__main__":
    main()
