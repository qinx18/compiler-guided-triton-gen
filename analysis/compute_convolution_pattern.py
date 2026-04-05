#!/usr/bin/env python3
"""
Convolution Pattern Detection Module

Detects convolution-like patterns in nested loops where:
1. An array is accumulated across all iterations of an outer loop
2. The inner loop accesses a shifted version of another array
3. The pattern is: a[i] += b[i + f(j)] * c[j]

These patterns are inherently sequential over the outer loop dimension
but can be rewritten as matrix-vector multiplication for GPU parallelization.

Example patterns detected:
- s176: for j in ...: for i in ...: a[i] += b[i+m-j-1] * c[j]
  This is a 1D convolution/correlation that can be rewritten as a[0:m] += B @ c
  where B is a shifted Toeplitz-like matrix.

Triton strategies:
1. Matrix-vector multiplication (recommended for moderate sizes)
2. FFT-based convolution (for large arrays)
3. Sequential outer loop with vectorized inner loop (fallback)
"""

import os
import re
import yaml
import subprocess
from typing import Optional, Dict, List, Tuple

# Path configuration
KERNELS_DIR = "/home/qinxiao/workspace/compiler-guided-triton-gen/analysis/kernels"
PET_PATH = "/home/qinxiao/workspace/pet/pet"
TSVC_SOURCE = "/home/qinxiao/workspace/compiler-guided-triton-gen/benchmarks_src/TSVC_2/src/archive/tsvc_orig.c"


def run_pet(kernel_file: str) -> Optional[str]:
    """Run PET on a kernel file and return the YAML output."""
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/home/qinxiao/workspace/pet/isl/.libs:' + env.get('LD_LIBRARY_PATH', '')

    try:
        result = subprocess.run(
            [PET_PATH, kernel_file],
            capture_output=True,
            text=True,
            env=env,
            timeout=30
        )
        if result.returncode == 0:
            # Fix YAML parsing issues with operators
            output = result.stdout
            for op in ['=', '+', '-', '*', '/', '%', '&&', '||', '<', '>', '<=', '>=', '==', '!=']:
                output = re.sub(rf'operation: {re.escape(op)}\s*$', f'operation: "{op}"', output, flags=re.MULTILINE)
            return output
    except Exception:
        pass
    return None


def detect_convolution_from_code(c_code: str) -> Dict:
    """
    Detect convolution patterns from C code.

    Patterns detected:
    1. a[i] += b[i + expr(j)] * c[j] - 1D convolution
    2. a[i] += b[f(i,j)] * c[j] where f depends on both i and j

    Returns:
        dict with keys:
            - is_convolution: bool
            - convolution_type: str ('1d_conv', '1d_correlation', None)
            - accumulator_array: str (e.g., 'a')
            - accumulator_index: str (e.g., 'i')
            - input_array: str (e.g., 'b')
            - input_index_expr: str (e.g., 'i+m-j-1')
            - weight_array: str (e.g., 'c')
            - weight_index: str (e.g., 'j')
            - outer_loop_var: str (sequential dimension)
            - inner_loop_var: str (parallel dimension)
            - shift_direction: str ('forward', 'backward', 'mixed')
            - can_rewrite_as_matmul: bool
            - triton_strategy: str
    """
    result = {
        'is_convolution': False,
        'convolution_type': None,
        'accumulator_array': None,
        'accumulator_index': None,
        'input_array': None,
        'input_index_expr': None,
        'weight_array': None,
        'weight_index': None,
        'outer_loop_var': None,
        'inner_loop_var': None,
        'shift_direction': None,
        'can_rewrite_as_matmul': False,
        'triton_strategy': None,
        'estimated_complexity': None,
    }

    # Normalize code
    code = c_code.replace('\n', ' ').replace('\t', ' ')
    code = re.sub(r'\s+', ' ', code)

    # Pattern 1: Nested loops with accumulation
    # for (j = ...) { for (i = ...) { a[i] += b[expr] * c[j]; } }
    # Pattern: accumulator[inner_var] += input[expr_with_both_vars] * weight[outer_var]

    # Look for nested for loops
    nested_loop_match = re.search(
        r'for\s*\([^)]*(\w+)\s*=.*?\)\s*\{?\s*for\s*\([^)]*(\w+)\s*=.*?\)',
        code
    )

    if not nested_loop_match:
        return result

    outer_var = nested_loop_match.group(1)
    inner_var = nested_loop_match.group(2)

    # Look for accumulation pattern: arr[idx] += expr1 * expr2
    # Where one of expr1/expr2 uses outer_var and the other uses both
    accum_pattern = re.search(
        r'(\w+)\[(\w+)\]\s*\+=\s*(\w+)\[([^\]]+)\]\s*\*\s*(\w+)\[(\w+)\]',
        code
    )

    if not accum_pattern:
        # Try alternate form: arr[idx] = arr[idx] + expr1 * expr2
        accum_pattern = re.search(
            r'(\w+)\[(\w+)\]\s*=\s*\1\[\2\]\s*\+\s*(\w+)\[([^\]]+)\]\s*\*\s*(\w+)\[(\w+)\]',
            code
        )

    if not accum_pattern:
        return result

    accum_arr = accum_pattern.group(1)
    accum_idx = accum_pattern.group(2)
    arr1 = accum_pattern.group(3)
    arr1_idx = accum_pattern.group(4)
    arr2 = accum_pattern.group(5)
    arr2_idx = accum_pattern.group(6)

    # Determine which is the input array (shifted) and which is the weight
    # The weight array should be indexed by the outer loop variable only
    # The input array should have an index that depends on both variables

    input_arr = None
    input_idx = None
    weight_arr = None
    weight_idx = None

    # Check if arr1 is indexed by expression containing both vars, arr2 by outer_var only
    arr1_has_outer = outer_var in arr1_idx
    arr1_has_inner = inner_var in arr1_idx
    arr2_has_outer = outer_var == arr2_idx or outer_var in arr2_idx
    arr2_has_inner = inner_var == arr2_idx or inner_var in arr2_idx

    if arr1_has_inner and arr1_has_outer and arr2_has_outer and not arr2_has_inner:
        input_arr = arr1
        input_idx = arr1_idx
        weight_arr = arr2
        weight_idx = arr2_idx
    elif arr2_has_inner and arr2_has_outer and arr1_has_outer and not arr1_has_inner:
        input_arr = arr2
        input_idx = arr2_idx
        weight_arr = arr1
        weight_idx = arr1_idx
    else:
        # Pattern doesn't match expected convolution structure
        return result

    # Verify the accumulator is indexed by inner loop variable
    if accum_idx != inner_var:
        return result

    # This is a convolution pattern!
    result['is_convolution'] = True
    result['accumulator_array'] = accum_arr
    result['accumulator_index'] = accum_idx
    result['input_array'] = input_arr
    result['input_index_expr'] = input_idx
    result['weight_array'] = weight_arr
    result['weight_index'] = weight_idx
    result['outer_loop_var'] = outer_var
    result['inner_loop_var'] = inner_var

    # Determine shift direction from the index expression
    # e.g., i+m-j-1 means backward shift as j increases
    if f'-{outer_var}' in input_idx or f'- {outer_var}' in input_idx:
        result['shift_direction'] = 'backward'
        result['convolution_type'] = '1d_correlation'
    elif f'+{outer_var}' in input_idx or f'+ {outer_var}' in input_idx:
        result['shift_direction'] = 'forward'
        result['convolution_type'] = '1d_conv'
    else:
        result['shift_direction'] = 'mixed'
        result['convolution_type'] = '1d_conv_general'

    # Check if can be rewritten as matrix-vector multiplication
    # This is possible when the shift pattern forms a structured matrix
    result['can_rewrite_as_matmul'] = True

    # Determine best Triton strategy
    result['triton_strategy'] = 'MATRIX_VECTOR_REWRITE'
    result['estimated_complexity'] = 'O(n*m) sequential -> O(n*m) parallel with matmul'

    return result


def analyze_kernel_convolution(kernel_name: str) -> Optional[Dict]:
    """
    Analyze a kernel for convolution patterns.

    Args:
        kernel_name: Name of the kernel (e.g., 's176')

    Returns:
        dict with convolution analysis or None if not found/not convolution
    """
    kernel_file = os.path.join(KERNELS_DIR, f"{kernel_name}.c")
    if not os.path.exists(kernel_file):
        return None

    # Read C code
    with open(kernel_file, 'r') as f:
        c_code = f.read()

    # Extract SCOP region if present
    scop_match = re.search(r'#pragma scop\s*(.*?)\s*#pragma endscop', c_code, re.DOTALL)
    if scop_match:
        c_code = scop_match.group(1)

    # Detect convolution pattern
    conv_result = detect_convolution_from_code(c_code)

    if not conv_result['is_convolution']:
        return None

    # Add kernel name
    conv_result['kernel'] = kernel_name
    conv_result['c_code'] = c_code.strip()

    return conv_result


def build_convolution_instructions(conv_result: Optional[Dict]) -> str:
    """
    Build prompt instructions for detected convolution patterns.

    Returns a string to be included in the LLM prompt.
    """
    if not conv_result or not conv_result.get('is_convolution'):
        return ""

    lines = []
    lines.append("")
    lines.append("## Convolution/Correlation Pattern Detected")
    lines.append("")
    lines.append("**WARNING: This kernel has a convolution-like pattern that is INHERENTLY SEQUENTIAL**")
    lines.append("**over the outer loop dimension. A naive Triton implementation WILL timeout.**")
    lines.append("")

    # Describe the pattern
    lines.append("### Pattern Analysis")
    lines.append(f"- **Accumulator**: `{conv_result['accumulator_array']}[{conv_result['accumulator_index']}]` (accumulated across all `{conv_result['outer_loop_var']}` iterations)")
    lines.append(f"- **Input array**: `{conv_result['input_array']}[{conv_result['input_index_expr']}]` (shifted access)")
    lines.append(f"- **Weight array**: `{conv_result['weight_array']}[{conv_result['weight_index']}]`")
    lines.append(f"- **Outer loop** (`{conv_result['outer_loop_var']}`): MUST be sequential - creates accumulation dependency")
    lines.append(f"- **Inner loop** (`{conv_result['inner_loop_var']}`): Can be vectorized")
    lines.append("")

    # Explain why it's slow
    lines.append("### Why Naive Implementation Times Out")
    lines.append(f"The same `{conv_result['accumulator_array']}[{conv_result['accumulator_index']}]` locations are updated by ALL `{conv_result['outer_loop_var']}` iterations.")
    lines.append(f"This creates a sequential dependency over `{conv_result['outer_loop_var']}` - you CANNOT parallelize this dimension.")
    lines.append(f"With typical TSVC sizes (n/2 = 16000 iterations), a sequential loop inside a Triton kernel will timeout.")
    lines.append("")

    # Recommended approach
    lines.append("### **REQUIRED: Rewrite as Matrix-Vector Multiplication**")
    lines.append("")
    lines.append("This convolution can be expressed as: `a += B @ c`")
    lines.append(f"where `B` is a matrix constructed from shifted views of `{conv_result['input_array']}`.")
    lines.append("")
    lines.append("**Implementation Strategy:**")
    lines.append("```python")
    lines.append(f"def {conv_result['kernel']}_triton({conv_result['accumulator_array']}, {conv_result['input_array']}, {conv_result['weight_array']}):")
    lines.append(f"    n = {conv_result['accumulator_array']}.shape[0]")
    lines.append(f"    m = n // 2  # or appropriate size")
    lines.append(f"    ")
    lines.append(f"    # Build the shifted matrix B where B[i, j] = {conv_result['input_array']}[{conv_result['input_index_expr']}]")
    lines.append(f"    # This creates a Toeplitz-like structure")
    lines.append(f"    ")
    lines.append(f"    # Option 1: Use torch.nn.functional.conv1d (most efficient)")
    lines.append(f"    # Reshape arrays for conv1d: input (batch, channels, length), weight (out_ch, in_ch, kernel)")
    lines.append(f"    {conv_result['input_array']}_reshaped = {conv_result['input_array']}.unsqueeze(0).unsqueeze(0)  # (1, 1, n)")
    lines.append(f"    {conv_result['weight_array']}_flipped = torch.flip({conv_result['weight_array']}[:m], [0]).unsqueeze(0).unsqueeze(0)  # (1, 1, m)")
    lines.append(f"    conv_result = torch.nn.functional.conv1d({conv_result['input_array']}_reshaped, {conv_result['weight_array']}_flipped, padding=m-1)")
    lines.append(f"    {conv_result['accumulator_array']}[:m] += conv_result[0, 0, :m]")
    lines.append(f"    ")
    lines.append(f"    # Option 2: Build explicit matrix and use matmul")
    lines.append(f"    # B = torch.zeros(m, m)")
    lines.append(f"    # for j in range(m):")
    lines.append(f"    #     B[:, j] = {conv_result['input_array']}[m-j-1 : 2*m-j-1]  # Adjust indices based on pattern")
    lines.append(f"    # {conv_result['accumulator_array']}[:m] += B @ {conv_result['weight_array']}[:m]")
    lines.append("```")
    lines.append("")
    lines.append("**Key Points:**")
    lines.append("1. DO NOT use nested loops with sequential outer loop in Triton kernel")
    lines.append("2. Use PyTorch's conv1d or matrix multiplication for the computation")
    lines.append("3. The convolution/correlation can be computed in O(n*m) parallel operations")
    lines.append("4. For very large arrays, consider FFT-based convolution: O(n log n)")
    lines.append("")

    return "\n".join(lines)


def format_convolution_for_prompt(kernel_name: str) -> str:
    """
    Convenience function to analyze and format convolution pattern for prompt.

    Args:
        kernel_name: Name of the kernel

    Returns:
        Formatted string for LLM prompt, or empty string if no convolution detected
    """
    result = analyze_kernel_convolution(kernel_name)
    return build_convolution_instructions(result)


def main():
    """Test the convolution pattern detection."""
    # Test with known convolution kernels
    test_kernels = ['s176']

    for kernel in test_kernels:
        print(f"\n{'=' * 60}")
        print(f"Analyzing: {kernel}")
        print('=' * 60)

        result = analyze_kernel_convolution(kernel)
        if result:
            print(f"  Is convolution: {result['is_convolution']}")
            print(f"  Type: {result['convolution_type']}")
            print(f"  Accumulator: {result['accumulator_array']}[{result['accumulator_index']}]")
            print(f"  Input: {result['input_array']}[{result['input_index_expr']}]")
            print(f"  Weight: {result['weight_array']}[{result['weight_index']}]")
            print(f"  Outer loop (sequential): {result['outer_loop_var']}")
            print(f"  Inner loop (parallel): {result['inner_loop_var']}")
            print(f"  Shift direction: {result['shift_direction']}")
            print(f"  Can rewrite as matmul: {result['can_rewrite_as_matmul']}")
            print(f"  Strategy: {result['triton_strategy']}")
            print()
            print("Prompt instructions:")
            print(build_convolution_instructions(result))
        else:
            print("  No convolution pattern detected")


if __name__ == "__main__":
    main()
