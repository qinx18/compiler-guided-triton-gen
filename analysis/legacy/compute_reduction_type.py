#!/usr/bin/env python3
"""
Reduction Type Analysis Module

Analyzes C code to detect reduction patterns and determine the appropriate
Triton implementation strategy.

Reduction types detected:
- sum: sum += expr (use tl.sum())
- prefix_sum: sum += a[i]; b[i] = sum (use tl.cumsum() for parallel scan)
- product: prod *= expr (use tl.reduce() with custom combiner)
- max: if (expr > max) max = expr (use tl.max())
- min: if (expr < min) min = expr (use tl.min())
- max_abs: max = ABS(expr) (use tl.max(tl.abs()))
- dot: sum += a[i] * b[i] (use tl.sum(a * b))
- argmax: if (expr > max) { max = expr; index = i; } (special handling)
- argmin: if (expr < min) { min = expr; index = i; } (special handling)
"""

import os
import re
import yaml
from typing import Optional, Dict, List, Tuple

# Path configuration
KERNELS_DIR = "/home/qinxiao/workspace/compiler-guided-triton-gen/analysis/kernels"
PET_PATH = "/home/qinxiao/workspace/pet/pet"
TSVC_SOURCE = "/home/qinxiao/workspace/compiler-guided-triton-gen/benchmarks_src/TSVC_2/src/archive/tsvc_orig.c"


def extract_return_statement(kernel_name: str) -> Optional[str]:
    """
    Extract the return statement from the TSVC source for a given kernel.

    Returns the return expression (e.g., "max + xindex+1 + yindex+1") or None.
    """
    if not os.path.exists(TSVC_SOURCE):
        return None

    try:
        with open(TSVC_SOURCE, 'r') as f:
            content = f.read()

        # Find the function definition
        pattern = rf'real_t\s+{kernel_name}\s*\([^)]*\)\s*\{{(.*?)\n\}}'
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            return None

        func_body = match.group(1)

        # Find the return statement
        return_match = re.search(r'return\s+([^;]+);', func_body)
        if return_match:
            return return_match.group(1).strip()

        return None
    except Exception:
        return None


def run_pet(kernel_file: str) -> Optional[str]:
    """Run PET on a kernel file and return the YAML output."""
    import subprocess
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
            return result.stdout
    except Exception:
        pass
    return None


def detect_reduction_from_code(c_code: str) -> Dict:
    """
    Detect reduction type from C code patterns.

    Returns:
        dict with keys:
            - is_reduction: bool
            - reduction_type: str or None
            - reduction_var: str or None (the accumulator variable)
            - triton_strategy: str or None (how to implement in Triton)
            - needs_custom_combiner: bool
            - identity_value: str (identity element for the reduction)
    """
    result = {
        'is_reduction': False,
        'reduction_type': None,
        'reduction_var': None,
        'triton_strategy': None,
        'needs_custom_combiner': False,
        'identity_value': None,
        'has_index_tracking': False,
    }

    # Normalize code for analysis
    code = c_code.replace('\n', ' ').replace('\t', ' ')

    # Pattern 0a: Argmax-abs (max-abs with index tracking)
    # Matches: if (ABS(a[k]) <= max) goto L; index = i; max = ABS(a[k])
    # Or: if (ABS(a[i]) > max) { index = i; max = ABS(a[i]); }
    argmax_abs_match = re.search(r'(ABS|abs)\s*\(.*?\)\s*[<>]=?\s*(\w+).*?index\s*=', code, re.DOTALL)
    if argmax_abs_match:
        result['is_reduction'] = True
        result['reduction_type'] = 'argmax_abs'
        result['reduction_var'] = argmax_abs_match.group(2)
        result['triton_strategy'] = 'torch.argmax(torch.abs()) (not parallelizable in Triton)'
        result['needs_custom_combiner'] = False
        result['identity_value'] = '0.0'
        result['has_index_tracking'] = True
        return result

    # Pattern 0b: Max-abs reduction without index - check FIRST before other patterns
    # Matches: if ((ABS(a[i])) > max) or if (ABS(a[i]) > max)
    maxabs_match = re.search(r'(ABS|abs)\s*\(\s*\w+\[.*?\]\s*\)\s*\)?\s*>\s*(\w+)', code)
    if maxabs_match:
        result['is_reduction'] = True
        result['reduction_type'] = 'max_abs'
        result['reduction_var'] = maxabs_match.group(2)
        result['triton_strategy'] = 'tl.max(tl.abs())'
        result['needs_custom_combiner'] = False
        result['identity_value'] = '0.0'
        return result

    # Pattern 1: Product reduction (prod *= expr)
    prod_match = re.search(r'(\w+)\s*\*=\s*\w+\[', code)
    if prod_match:
        result['is_reduction'] = True
        result['reduction_type'] = 'product'
        result['reduction_var'] = prod_match.group(1)
        result['triton_strategy'] = 'tl.reduce with custom combiner'
        result['needs_custom_combiner'] = True
        result['identity_value'] = '1.0'
        return result

    # Pattern 2: Sum reduction (sum += array_expr) - must be array access, not just any +=
    # Check for dot product first (sum += a[i] * b[i])
    dot_match = re.search(r'(\w+)\s*\+=\s*\w+\[\w+\]\s*\*\s*\w+\[\w+\]', code)
    if dot_match:
        result['is_reduction'] = True
        result['reduction_type'] = 'dot'
        result['reduction_var'] = dot_match.group(1)
        result['triton_strategy'] = 'tl.sum(a * b)'
        result['needs_custom_combiner'] = False
        result['identity_value'] = '0.0'
        return result

    # Check for simple sum (sum += a[i]) - must have array access
    sum_match = re.search(r'(\w+)\s*\+=\s*\w+\[\w+\](?!\s*\*)', code)
    if sum_match:
        sum_var = sum_match.group(1)
        # Check if this is a PREFIX SUM (running sum stored to array)
        # Pattern: sum += a[i]; b[i] = sum (where sum_var is written to an array)
        prefix_sum_pattern = rf'{sum_var}\s*\+=\s*\w+\[\w+\].*?\w+\[\w+\]\s*=\s*{sum_var}'
        if re.search(prefix_sum_pattern, code, re.DOTALL):
            result['is_reduction'] = True
            result['reduction_type'] = 'prefix_sum'
            result['reduction_var'] = sum_var
            result['triton_strategy'] = 'tl.cumsum() for parallel prefix sum'
            result['needs_custom_combiner'] = False
            result['identity_value'] = '0.0'
            return result

        # Regular sum reduction (final value only)
        result['is_reduction'] = True
        result['reduction_type'] = 'sum'
        result['reduction_var'] = sum_var
        result['triton_strategy'] = 'tl.sum()'
        result['needs_custom_combiner'] = False
        result['identity_value'] = '0.0'
        return result

    # Pattern 3: Max/Min with conditional and index tracking (argmax/argmin)
    # if (a[i] > max) { max = a[i]; xindex = i; }
    argmax_match = re.search(
        r'if\s*\(\s*(\w+)\[.*?\]\s*>\s*(\w+)\s*\).*?(\w+)\s*=\s*\1\[.*?\].*?(\w+)\s*=\s*\w+',
        code, re.DOTALL
    )
    if argmax_match:
        result['is_reduction'] = True
        result['reduction_type'] = 'argmax'
        result['reduction_var'] = argmax_match.group(2)
        result['triton_strategy'] = 'torch.argmax (not parallelizable in Triton)'
        result['needs_custom_combiner'] = False
        result['identity_value'] = 'float("-inf")'
        result['has_index_tracking'] = True
        return result

    argmin_match = re.search(
        r'if\s*\(\s*(\w+)\[.*?\]\s*<\s*(\w+)\s*\).*?(\w+)\s*=\s*\1\[.*?\].*?(\w+)\s*=\s*\w+',
        code, re.DOTALL
    )
    if argmin_match:
        result['is_reduction'] = True
        result['reduction_type'] = 'argmin'
        result['reduction_var'] = argmin_match.group(2)
        result['triton_strategy'] = 'torch.argmin (not parallelizable in Triton)'
        result['needs_custom_combiner'] = False
        result['identity_value'] = 'float("inf")'
        result['has_index_tracking'] = True
        return result

    # Pattern 4: Max reduction without index (if (a[i] > x) x = a[i])
    max_match = re.search(r'if\s*\(\s*\w+\[.*?\]\s*>\s*(\w+)\s*\)', code)
    if max_match and 'ABS' not in code and 'abs' not in code:
        result['is_reduction'] = True
        result['reduction_type'] = 'max'
        result['reduction_var'] = max_match.group(1)
        result['triton_strategy'] = 'tl.max()'
        result['needs_custom_combiner'] = False
        result['identity_value'] = 'float("-inf")'
        return result

    # Pattern 5: Min reduction (if (a[i] < x) x = a[i])
    min_match = re.search(r'if\s*\(\s*\w+\[.*?\]\s*<\s*(\w+)\s*\)', code)
    if min_match:
        result['is_reduction'] = True
        result['reduction_type'] = 'min'
        result['reduction_var'] = min_match.group(1)
        result['triton_strategy'] = 'tl.min()'
        result['needs_custom_combiner'] = False
        result['identity_value'] = 'float("inf")'
        return result

    return result


def detect_reduction_from_pet(kernel_file: str) -> Dict:
    """
    Detect reduction type using PET analysis.

    Checks if writes are to scalars (no array index) while reads are from arrays.
    """
    result = {
        'is_reduction': False,
        'writes_to_scalar': False,
        'reads_from_array': False,
        'scalar_vars': [],
        'array_vars': [],
    }

    pet_output = run_pet(kernel_file)
    if not pet_output:
        return result

    try:
        data = yaml.safe_load(pet_output)
    except:
        return result

    scalar_writes = []
    array_reads = []

    for stmt in data.get('statements', []):
        body = stmt.get('body', {})
        expr = body.get('expr', {})

        # Recursively find all accesses
        def find_accesses(node, accesses):
            if isinstance(node, dict):
                if node.get('type') == 'access':
                    index = node.get('index', '')
                    is_read = node.get('read', 0)
                    is_write = node.get('write', 0)

                    # Check if it's a scalar (no array index in access)
                    # Scalar: "{ S_0[] -> prod[] }" (no index variable)
                    # Array: "{ S_1[i] -> a[(i)] }" (has index variable)
                    is_scalar = '[]' in index and not re.search(r'\[\(?\w+\)?\]', index.split('->')[-1])

                    # Extract variable name
                    var_match = re.search(r'->\s*(\w+)', index)
                    var_name = var_match.group(1) if var_match else None

                    if var_name:
                        if is_write and is_scalar:
                            scalar_writes.append(var_name)
                        if is_read and not is_scalar:
                            array_reads.append(var_name)

                for v in node.values():
                    find_accesses(v, accesses)
            elif isinstance(node, list):
                for item in node:
                    find_accesses(item, accesses)

        find_accesses(expr, [])

    result['scalar_vars'] = list(set(scalar_writes))
    result['array_vars'] = list(set(array_reads))
    result['writes_to_scalar'] = len(scalar_writes) > 0
    result['reads_from_array'] = len(array_reads) > 0
    result['is_reduction'] = result['writes_to_scalar'] and result['reads_from_array']

    return result


def analyze_kernel_reduction(kernel_name: str, kernel_file: str = None) -> Optional[Dict]:
    """
    Analyze a kernel for reduction patterns.

    Args:
        kernel_name: Name of the kernel (e.g., 's312')
        kernel_file: Optional full path to kernel .c file. If not provided,
                     looks in KERNELS_DIR (TSVC default).

    Returns:
        dict with complete reduction analysis or None if not found
    """
    if kernel_file is None:
        kernel_file = os.path.join(KERNELS_DIR, f"{kernel_name}.c")
    if not os.path.exists(kernel_file):
        return None

    # Read C code
    with open(kernel_file, 'r') as f:
        c_code = f.read()

    # Code-based detection
    code_result = detect_reduction_from_code(c_code)

    # PET-based detection
    pet_result = detect_reduction_from_pet(kernel_file)

    # Extract return statement from TSVC source
    return_statement = extract_return_statement(kernel_name)

    # Combine results
    result = {
        'kernel': kernel_name,
        'is_reduction': code_result['is_reduction'] or pet_result['is_reduction'],
        'reduction_type': code_result['reduction_type'],
        'reduction_var': code_result['reduction_var'],
        'triton_strategy': code_result['triton_strategy'],
        'needs_custom_combiner': code_result['needs_custom_combiner'],
        'identity_value': code_result['identity_value'],
        'has_index_tracking': code_result['has_index_tracking'],
        'writes_to_scalar': pet_result['writes_to_scalar'],
        'reads_from_array': pet_result['reads_from_array'],
        'scalar_vars': pet_result['scalar_vars'],
        'array_vars': pet_result['array_vars'],
        'return_statement': return_statement,
    }

    return result


def build_reduction_instructions(reduction_result: Optional[Dict]) -> str:
    """
    Build prompt instructions for the detected reduction pattern.

    Returns a string to be included in the LLM prompt.
    """
    if not reduction_result or not reduction_result['is_reduction']:
        return ""

    lines = []
    lines.append("")
    lines.append("## Reduction Pattern Analysis")
    lines.append("")

    # Add return statement warning if available
    return_stmt = reduction_result.get('return_statement')
    if return_stmt:
        lines.append("**⚠️ CRITICAL: Exact Return Value Required**")
        lines.append("")
        lines.append(f"The C function returns: `return {return_stmt};`")
        lines.append("")
        lines.append("Your Triton function MUST return **exactly** this value.")
        lines.append("Pay close attention - the return may differ from intermediate computations (e.g., `chksum`)!")
        lines.append("")

    rtype = reduction_result['reduction_type']

    if rtype == 'product':
        lines.append("**Detected: Product Reduction** (`*=` operator)")
        lines.append("")
        lines.append("**CRITICAL: Triton does NOT have `tl.prod()` or `tl.mul`!**")
        lines.append("")
        lines.append("You MUST use `tl.reduce()` with a **separate @triton.jit combiner function**.")
        lines.append("Lambdas are NOT supported in Triton JIT code.")
        lines.append("")
        lines.append("```python")
        lines.append("@triton.jit")
        lines.append("def _prod_combine(a, b):")
        lines.append("    return a * b")
        lines.append("")
        lines.append("@triton.jit")
        lines.append("def kernel(...):")
        lines.append("    vals = tl.load(a_ptr + offsets, mask=mask, other=1.0)  # Identity=1.0 for product")
        lines.append("    block_prod = tl.reduce(vals, axis=0, combine_fn=_prod_combine)")
        lines.append("```")
        lines.append("")
        lines.append("For large arrays, compute partial products per block, then combine with `torch.prod()`")

    elif rtype == 'prefix_sum':
        lines.append("**Detected: Prefix Sum / Running Sum (Scan Operation)**")
        lines.append("")
        lines.append("This is a **prefix sum** (cumulative sum) where running totals are stored to an output array.")
        lines.append("")
        lines.append("**Use `tl.cumsum()` for parallel prefix sum:**")
        lines.append("```python")
        lines.append("# Load a block of input values")
        lines.append("offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)")
        lines.append("mask = offsets < n_elements")
        lines.append("vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)")
        lines.append("")
        lines.append("# Compute prefix sum within the block")
        lines.append("prefix_sums = tl.cumsum(vals, axis=0)")
        lines.append("")
        lines.append("# Store the prefix sums to output")
        lines.append("tl.store(b_ptr + offsets, prefix_sums, mask=mask)")
        lines.append("```")
        lines.append("")
        lines.append("For multi-block prefix sums, use a two-pass algorithm:")
        lines.append("1. First pass: compute local prefix sums for each block")
        lines.append("2. Extract block totals using tensor indexing (NOT Python loops):")
        lines.append("   ```python")
        lines.append("   # Fast: tensor indexing")
        lines.append("   block_ends = torch.arange(BLOCK_SIZE - 1, n, BLOCK_SIZE, device=b.device)")
        lines.append("   block_ends = block_ends.clamp(max=n-1)")
        lines.append("   block_totals = b[block_ends]")
        lines.append("   block_offsets = torch.cumsum(block_totals[:-1], dim=0)")
        lines.append("   ```")
        lines.append("3. Second pass: add cumulative block offsets to each subsequent block")

    elif rtype == 'sum':
        lines.append("**Detected: Sum Reduction** (`+=` operator)")
        lines.append("")
        lines.append("Use `tl.sum()` for parallel reduction:")
        lines.append("```python")
        lines.append("vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)")
        lines.append("block_sum = tl.sum(vals, axis=0)")
        lines.append("```")

    elif rtype == 'dot':
        lines.append("**Detected: Dot Product Reduction**")
        lines.append("")
        lines.append("Use element-wise multiply then `tl.sum()`:")
        lines.append("```python")
        lines.append("a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)")
        lines.append("b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)")
        lines.append("block_dot = tl.sum(a_vals * b_vals, axis=0)")
        lines.append("```")

    elif rtype == 'max':
        lines.append("**Detected: Max Reduction**")
        lines.append("")
        lines.append("Use `tl.max()` for parallel reduction:")
        lines.append("```python")
        lines.append("vals = tl.load(a_ptr + offsets, mask=mask, other=float('-inf'))")
        lines.append("block_max = tl.max(vals, axis=0)")
        lines.append("```")

    elif rtype == 'min':
        lines.append("**Detected: Min Reduction**")
        lines.append("")
        lines.append("Use `tl.min()` for parallel reduction:")
        lines.append("```python")
        lines.append("vals = tl.load(a_ptr + offsets, mask=mask, other=float('inf'))")
        lines.append("block_min = tl.min(vals, axis=0)")
        lines.append("```")

    elif rtype == 'max_abs':
        lines.append("**Detected: Max-Absolute Reduction**")
        lines.append("")
        lines.append("Use `tl.abs()` then `tl.max()`:")
        lines.append("```python")
        lines.append("vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)")
        lines.append("block_max_abs = tl.max(tl.abs(vals), axis=0)")
        lines.append("```")

    elif rtype in ('argmax_abs', 'argmax', 'argmin'):
        if rtype == 'argmax_abs':
            label = "Argmax-Abs Reduction (max absolute value with index)"
        else:
            label = f"{rtype.title()} Reduction (value + index)"
        lines.append(f"**Detected: {label}**")
        lines.append("")

        # Check if this is a simple 1D global argmax/argmin (not 2D, not early-exit)
        # by looking for a single "index" variable in the return statement
        ret = reduction_result.get('return_statement', '')
        is_simple_1d = (
            re.search(r'\bindex\b', ret) and
            not re.search(r'\b[xy]index\b', ret)
        )

        if is_simple_1d:
            # 1D global argmax/argmin — PyTorch is the best approach
            lines.append("Argmax/argmin with index tracking is not directly parallelizable in Triton.")
            lines.append("**Use PyTorch directly** — a trivial `@triton.jit` stub kernel is fine.")
            lines.append("")
            lines.append("**Key considerations** (read the C code carefully):")
            lines.append("- If the C code uses a **stride** (e.g., `k += inc`), gather elements with that stride first")
            lines.append("- If the C code uses `ABS()`, apply `torch.abs()` before finding the max")
            lines.append("- Check the **exact return expression** from the C code")
            lines.append("")
            lines.append("**General pattern:**")
            lines.append("```python")
            lines.append("@triton.jit")
            lines.append("def kernel_stub(dummy):  # stub — work done in wrapper")
            lines.append("    pass")
            lines.append("")
            lines.append("def kernel_triton(...):")
            lines.append("    # Step 1: Gather elements (apply stride if needed)")
            lines.append("    #   vals = a[::inc]  OR  vals = a  (if no stride)")
            lines.append("    # Step 2: Transform (apply abs() if needed)")
            lines.append("    #   vals = torch.abs(vals)  (if C code uses ABS)")
            lines.append("    # Step 3: Find max/min + index")
            lines.append("    #   max_val = torch.max(vals)")
            lines.append("    #   max_idx = torch.argmax(vals)")
            lines.append("    # Step 4: Return EXACTLY what C code returns")
            lines.append("```")
        else:
            # 2D or early-exit patterns — use existing advice
            lines.append("**Note:** Argmax/argmin with index tracking is complex in Triton.")
            lines.append("Recommended approach: Use PyTorch's `torch.argmax()`/`torch.argmin()` in the wrapper.")
            lines.append("")
            lines.append("For 1D arrays:")
            lines.append("```python")
            lines.append("def kernel_triton(a):")
            lines.append("    max_val = torch.max(a)")
            lines.append("    max_idx = torch.argmax(a)")
            lines.append("    # IMPORTANT: Check the exact return format from C code above!")
            lines.append("```")
            lines.append("")
            lines.append("For 2D arrays:")
            lines.append("```python")
            lines.append("flat_idx = torch.argmax(aa.flatten())")
            lines.append("xindex = flat_idx // aa.shape[1]")
            lines.append("yindex = flat_idx % aa.shape[1]")
            lines.append("# IMPORTANT: Check the exact return format from C code above!")
            lines.append("```")

    lines.append("")
    return "\n".join(lines)


def main():
    """Test the reduction analysis on known kernels."""
    test_kernels = ['s311', 's312', 's313', 's314', 's3110', 's3113', 's318']

    for kernel in test_kernels:
        print(f"\n{'=' * 60}")
        print(f"Analyzing: {kernel}")
        print('=' * 60)

        result = analyze_kernel_reduction(kernel)
        if result:
            print(f"  Is reduction: {result['is_reduction']}")
            print(f"  Type: {result['reduction_type']}")
            print(f"  Strategy: {result['triton_strategy']}")
            print(f"  Needs custom combiner: {result['needs_custom_combiner']}")
            print(f"  Has index tracking: {result['has_index_tracking']}")
            print(f"  Writes to scalar: {result['writes_to_scalar']}")
            print(f"  Reads from array: {result['reads_from_array']}")
        else:
            print("  Analysis failed")


if __name__ == "__main__":
    main()
