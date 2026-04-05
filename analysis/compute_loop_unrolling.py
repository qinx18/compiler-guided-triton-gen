#!/usr/bin/env python3
"""
Detect manually unrolled loop patterns in TSVC kernels.

This analysis identifies patterns where loops have stride > 1 and process
multiple consecutive elements per iteration. These are manually unrolled
loops that should be "re-rolled" for proper GPU parallelization.

Example s353:
    for (int i = 0; i < LEN_1D; i += 5) {
        a[i] += alpha * b[ip[i]];
        a[i + 1] += alpha * b[ip[i + 1]];
        a[i + 2] += alpha * b[ip[i + 2]];
        a[i + 3] += alpha * b[ip[i + 3]];
        a[i + 4] += alpha * b[ip[i + 4]];
    }

After conceptual "re-rolling", this is equivalent to:
    for (int i = 0; i < LEN_1D; i++) {
        a[i] += alpha * b[ip[i]];
    }

This is trivially parallelizable with one element per thread.

Key detection:
1. Loop with stride > 1 (e.g., i += 5, i += 2)
2. Loop body accesses consecutive elements (i, i+1, i+2, ..., i+stride-1)
3. The unrolling is just a manual optimization, not a semantic requirement
"""

import re
import os

KERNELS_DIR = "/home/qinxiao/workspace/compiler-guided-triton-gen/analysis/kernels"


def parse_kernel_file(kernel_file):
    """Parse a kernel file to extract the loop structure."""
    if not os.path.exists(kernel_file):
        return None

    with open(kernel_file, 'r') as f:
        content = f.read()

    # Extract the scop region
    scop_match = re.search(r'#pragma scop\s*(.*?)\s*#pragma endscop', content, re.DOTALL)
    if not scop_match:
        return None

    return scop_match.group(1).strip()


def detect_strided_loop(code):
    """
    Detect loops with stride > 1.

    Returns dict with:
    - 'has_stride': bool
    - 'loop_var': str (e.g., 'i')
    - 'stride': int (e.g., 5)
    - 'stride_expr': str (e.g., '5' or 'inc')
    """
    # Pattern: for (... i += N) or for (... i = i + N)
    stride_patterns = [
        r'for\s*\([^;]*;\s*[^;]*;\s*(\w+)\s*\+=\s*(\d+)\s*\)',  # i += 5
        r'for\s*\([^;]*;\s*[^;]*;\s*(\w+)\s*\+=\s*(\w+)\s*\)',  # i += inc (variable)
        r'for\s*\([^;]*;\s*[^;]*;\s*(\w+)\s*=\s*\1\s*\+\s*(\d+)\s*\)',  # i = i + 5
    ]

    for pattern in stride_patterns:
        match = re.search(pattern, code)
        if match:
            loop_var = match.group(1)
            stride_expr = match.group(2)
            try:
                stride = int(stride_expr)
            except ValueError:
                stride = None  # Variable stride

            if stride is None or stride > 1:
                return {
                    'has_stride': True,
                    'loop_var': loop_var,
                    'stride': stride,
                    'stride_expr': stride_expr
                }

    return {'has_stride': False}


def detect_consecutive_accesses(code, loop_var, stride):
    """
    Detect if the loop body accesses consecutive elements.

    Look for patterns like:
    - a[i], a[i+1], a[i+2], ...
    - a[i], a[i + 1], a[i + 2], ...
    """
    if stride is None:
        return {'has_consecutive': False}

    # Find all array accesses with the loop variable
    # Pattern: array[i + offset] or array[i]
    access_pattern = rf'(\w+)\s*\[\s*{loop_var}\s*(?:\+\s*(\d+))?\s*\]'

    accesses = {}  # array_name -> set of offsets

    for match in re.finditer(access_pattern, code):
        array_name = match.group(1)
        offset = int(match.group(2)) if match.group(2) else 0

        if array_name not in accesses:
            accesses[array_name] = set()
        accesses[array_name].add(offset)

    # Check if any array has consecutive accesses from 0 to stride-1
    consecutive_arrays = []
    for array_name, offsets in accesses.items():
        expected = set(range(stride))
        if expected.issubset(offsets):
            consecutive_arrays.append(array_name)

    if consecutive_arrays:
        return {
            'has_consecutive': True,
            'arrays': consecutive_arrays,
            'all_accesses': accesses
        }

    return {'has_consecutive': False}


def analyze_loop_unrolling(kernel_name):
    """
    Analyze a kernel for manually unrolled loop patterns.

    Returns dict with:
    - 'applicable': bool
    - 'loop_var': str
    - 'stride': int
    - 'arrays': list of arrays with consecutive accesses
    - 'advice': str
    """
    kernel_file = os.path.join(KERNELS_DIR, f"{kernel_name}.c")
    code = parse_kernel_file(kernel_file)

    if not code:
        return {'applicable': False}

    # Detect strided loop
    stride_info = detect_strided_loop(code)
    if not stride_info['has_stride']:
        return {'applicable': False}

    loop_var = stride_info['loop_var']
    stride = stride_info['stride']
    stride_expr = stride_info['stride_expr']

    # For variable stride, this is NOT an unrolled loop pattern
    # Variable stride loops have different semantics (e.g., strided access with WAR deps)
    # Don't report as "loop unrolling" - let WAR analysis handle it
    if stride is None:
        return {'applicable': False}

    # Detect consecutive accesses
    access_info = detect_consecutive_accesses(code, loop_var, stride)

    if not access_info['has_consecutive']:
        return {'applicable': False}

    return {
        'applicable': True,
        'loop_var': loop_var,
        'stride': stride,
        'stride_expr': stride_expr,
        'variable_stride': False,
        'arrays': access_info['arrays'],
        'all_accesses': access_info['all_accesses'],
        'advice': generate_unrolling_advice(loop_var, stride, access_info['arrays'])
    }


def generate_unrolling_advice(loop_var, stride, arrays):
    """Generate advice for handling unrolled loops."""
    arrays_str = ', '.join(arrays)

    advice = f"""MANUALLY UNROLLED LOOP DETECTED

This kernel has a manually unrolled loop with stride {stride}:
  - Loop variable: {loop_var}
  - Stride: {stride}
  - Arrays with consecutive accesses: {arrays_str}

Original pattern:
  for (int {loop_var} = 0; {loop_var} < N; {loop_var} += {stride}) {{
      // {stride} operations on {loop_var}, {loop_var}+1, ..., {loop_var}+{stride-1}
  }}

CRITICAL: This is semantically equivalent to a simple per-element loop:
  for (int {loop_var} = 0; {loop_var} < N; {loop_var}++) {{
      // ONE operation on element {loop_var}
  }}

IMPLEMENTATION REQUIREMENTS:

1. DO NOT preserve the stride-{stride} structure in parallelization
2. Treat this as a simple per-element operation
3. Each thread should handle ONE element, not {stride} elements
4. Standard vectorized Triton patterns apply

CORRECT parallelization:
```python
@triton.jit
def kernel(...):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load, compute, store for EACH element independently
    vals = tl.load(ptr + offsets, mask=mask)
    # ... compute ...
    tl.store(ptr + offsets, result, mask=mask)
```

WRONG parallelization (causes duplicate processing at block boundaries):
```python
# DO NOT DO THIS:
for i in range(0, BLOCK_SIZE, {stride}):  # WRONG!
    # Process {stride} elements per iteration
```

The unrolling in the original C code is a manual optimization for CPU.
For GPU/Triton, ignore the unrolling and parallelize per-element."""

    return advice


def generate_variable_stride_advice(loop_var, stride_expr):
    """Generate advice for variable stride loops."""

    advice = f"""VARIABLE STRIDE LOOP DETECTED

This kernel has a loop with variable stride:
  - Loop variable: {loop_var}
  - Stride expression: {stride_expr}

This pattern requires special handling based on the stride value at runtime.

IMPLEMENTATION REQUIREMENTS:

1. If the stride is known at compile time, treat as unrolled loop
2. Process elements based on actual indices, not block-relative indices
3. Be careful about element coverage - ensure all elements are processed exactly once

Consider using the stride parameter to compute actual element indices:
```python
def kernel_triton(a, b, stride, ...):
    # Process all elements, respecting the stride pattern
    for base in range(0, n, stride):
        # Process elements base, base+1, ..., base+stride-1
```"""

    return advice


def format_unrolling_for_prompt(result):
    """Format loop unrolling analysis for inclusion in LLM prompt."""
    if not result or not result.get('applicable'):
        return None

    lines = []
    lines.append("=" * 60)
    lines.append("LOOP UNROLLING PATTERN DETECTED")
    lines.append("=" * 60)
    lines.append("")

    if result.get('advice'):
        lines.append(result['advice'])

    lines.append("")
    lines.append("=" * 60)

    return '\n'.join(lines)


def analyze_kernel_loop_unrolling(kernel_name):
    """
    Main entry point for loop unrolling analysis.

    Args:
        kernel_name: Name of the kernel (e.g., 's353')

    Returns:
        dict with analysis results, or None if not applicable
    """
    result = analyze_loop_unrolling(kernel_name)
    if result.get('applicable'):
        return result
    return None


def main():
    """Test loop unrolling detection on known unrolled kernels."""
    test_kernels = ['s116', 's351', 's352', 's353', 's119', 's1115']

    print("=" * 80)
    print("LOOP UNROLLING PATTERN DETECTION")
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
            # Show first few lines
            lines = code.split('\n')[:10]
            print(f"\nC Code (first 10 lines):")
            for line in lines:
                print(f"  {line}")

        result = analyze_kernel_loop_unrolling(kernel)
        if result:
            print(f"\nDetected: YES")
            print(f"  Loop var: {result.get('loop_var')}")
            print(f"  Stride: {result.get('stride') or result.get('stride_expr')}")
            if result.get('arrays'):
                print(f"  Arrays: {result.get('arrays')}")

            formatted = format_unrolling_for_prompt(result)
            if formatted:
                print(f"\n{formatted}")
        else:
            print("\nDetected: NO")


if __name__ == "__main__":
    main()
