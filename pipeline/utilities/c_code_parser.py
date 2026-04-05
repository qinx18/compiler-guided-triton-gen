"""
C Code Parser for extracting function properties from source code.

This module analyzes C loop code to extract:
- Array names and access modes (read/write)
- Scalar parameters
- 2D array patterns
- Offset patterns
- Reduction patterns
"""

import re
from typing import Dict, List, Set, Tuple


def parse_c_code(loop_code: str) -> Dict:
    """
    Parse C loop code and extract properties needed for Triton code generation and testing.

    Args:
        loop_code: The C source code of the loop body

    Returns:
        Dictionary with extracted properties:
        - arrays: Dict[str, str] mapping array name to access mode ('r', 'w', 'rw')
        - scalar_params: Dict[str, str] mapping scalar name to type hint
        - has_2d_arrays: bool
        - has_offset: bool
        - has_reduction: bool
        - has_conditional: bool
    """
    # Extract array accesses and their modes
    arrays = _extract_arrays(loop_code)

    # Extract scalar parameters
    scalars = _extract_scalars(loop_code, set(arrays.keys()))

    # Detect 2D arrays
    has_2d = _detect_2d_arrays(loop_code)

    # Detect offset patterns
    has_offset = _detect_offset(loop_code)

    # Detect reduction patterns
    has_reduction = _detect_reduction(loop_code)

    # Detect conditionals
    has_conditional = _detect_conditional(loop_code)

    return {
        'arrays': arrays,
        'scalar_params': scalars,
        'has_2d_arrays': has_2d,
        'has_offset': has_offset,
        'has_reduction': has_reduction,
        'has_conditional': has_conditional,
    }


def _extract_arrays(code: str) -> Dict[str, str]:
    """
    Extract array names and their access modes from C code.

    Returns dict mapping array name to access mode:
    - 'r': read-only
    - 'w': write-only
    - 'rw': read and write
    """
    # Find all array accesses: identifier followed by [...]
    # Pattern matches: a[i], aa[i][j], b[i + 1], etc.
    array_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\['

    # Find arrays on left side of assignment (written)
    # Pattern: array[...] = or array[...][...] =
    write_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\[[^\]]*\](?:\s*\[[^\]]*\])?\s*='
    # Pattern for compound assignments (+=, -=, etc.) - these are always read-write
    compound_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\[[^\]]*\](?:\s*\[[^\]]*\])?\s*[+\-*/%&|^]='

    all_arrays = set(re.findall(array_pattern, code))
    written_arrays = set(re.findall(write_pattern, code))
    compound_arrays = set(re.findall(compound_pattern, code))  # Always rw

    # Filter out common non-array identifiers
    exclude = {'int', 'float', 'double', 'real_t', 'for', 'if', 'while', 'sizeof'}
    all_arrays -= exclude
    written_arrays -= exclude

    # Determine access mode for each array
    arrays = {}
    for arr in all_arrays:
        if arr in compound_arrays:
            # Compound assignment (+=, -=, etc.) is always read-write
            arrays[arr] = 'rw'
        elif arr in written_arrays:
            # Check if also read (appears on RHS or in conditions)
            # Remove the write occurrences and check if array still appears
            code_without_writes = re.sub(
                rf'\b{arr}\s*\[[^\]]*\](?:\s*\[[^\]]*\])?\s*=',
                '',
                code
            )
            if re.search(rf'\b{arr}\s*\[', code_without_writes):
                arrays[arr] = 'rw'
            else:
                arrays[arr] = 'w'
        else:
            arrays[arr] = 'r'

    return arrays


def _extract_scalars(code: str, array_names: Set[str]) -> Dict[str, str]:
    """
    Extract scalar parameters from C code.

    Looks for variables that are used but not as arrays.
    """
    scalars = {}

    # Check for iteration patterns - various forms used in TSVC
    iteration_patterns = [
        r'nl\s*<\s*iterations',           # nl < iterations
        r'nl\s*<\s*\d*\s*\*?\s*iterations', # nl < 2*iterations
        r'iterations\s*\*\s*\d+',          # iterations * 10
    ]
    for pattern in iteration_patterns:
        if re.search(pattern, code):
            scalars['iterations'] = 'scalar'
            break

    # Look for other scalar usages (common TSVC scalars used in loop conditions)
    scalar_candidates = ['n1', 'n3', 't', 'k']
    for scalar_name in scalar_candidates:
        if re.search(rf'\b{scalar_name}\b', code) and scalar_name not in array_names:
            scalars[scalar_name] = 'scalar'

    return scalars


def _detect_2d_arrays(code: str) -> bool:
    """
    Detect if code uses 2D array access patterns.

    Looks for patterns like: arr[i][j] or arr[expr1][expr2]
    """
    # Pattern for 2D array access
    pattern_2d = r'\b[a-zA-Z_][a-zA-Z0-9_]*\s*\[[^\]]+\]\s*\[[^\]]+\]'
    return bool(re.search(pattern_2d, code))


def _detect_offset(code: str) -> bool:
    """
    Detect if code uses offset patterns in array indexing.

    Looks for patterns like: a[i + 10], a[i - offset], LEN_1D/2, etc.
    """
    # Offset in array index
    offset_patterns = [
        r'\[\s*\w+\s*[+\-]\s*\d+\s*\]',  # [i + 10] or [i - 5]
        r'\[\s*\w+\s*[+\-]\s*\w+\s*\]',  # [i + offset]
        r'LEN_1D\s*/\s*\d+',              # LEN_1D / 2
        r'\[\s*\d+\s*\]',                  # Fixed index like [0], [10]
    ]

    for pattern in offset_patterns:
        if re.search(pattern, code):
            # Exclude simple [0] for initialization
            if pattern == r'\[\s*\d+\s*\]':
                # Only count as offset if there are multiple fixed indices
                matches = re.findall(r'\[\s*(\d+)\s*\]', code)
                unique_indices = set(int(m) for m in matches)
                if len(unique_indices) > 1 or (unique_indices and max(unique_indices) > 1):
                    return True
            else:
                return True

    return False


def _detect_reduction(code: str) -> bool:
    """
    Detect if code performs a reduction operation.

    Looks for patterns like:
    - sum += expr
    - sum = sum + expr
    - x = max(x, expr)
    - Accumulation into a scalar
    """
    reduction_patterns = [
        r'\b\w+\s*\+=',                    # sum +=
        r'\b\w+\s*\*=',                    # prod *=
        r'\b(\w+)\s*=\s*\1\s*[+\-\*/]',   # x = x + ...
        r'\bsum\b',                         # Variable named sum
        r'\bchksum\b',                      # Variable named chksum
        r'>\s*x\s*\)',                      # Comparison for max: > x)
        r'<\s*x\s*\)',                      # Comparison for min: < x)
    ]

    for pattern in reduction_patterns:
        if re.search(pattern, code):
            return True

    return False


def _detect_conditional(code: str) -> bool:
    """
    Detect if code contains conditional statements.

    Looks for if statements (excluding loop conditions).
    """
    # Look for if statements
    # Exclude 'if' that's part of #ifdef or similar
    if_pattern = r'\bif\s*\('
    return bool(re.search(if_pattern, code))


def infer_function_spec(func_name: str, loop_code: str) -> Dict:
    """
    Infer complete function specification from name and loop code.

    This is the main entry point for runtime inference.

    Args:
        func_name: Name of the function
        loop_code: The C source code

    Returns:
        Function specification dictionary compatible with existing framework
    """
    parsed = parse_c_code(loop_code)

    return {
        'name': func_name,
        'loop_code': loop_code,
        'arrays': parsed['arrays'],
        'has_offset': parsed['has_offset'],
        'has_conditional': parsed['has_conditional'],
        'has_reduction': parsed['has_reduction'],
        'has_2d_arrays': parsed['has_2d_arrays'],
        'scalar_params': parsed['scalar_params'],
    }


# Testing
if __name__ == '__main__':
    # Test with some example TSVC code snippets

    # s000: simple array copy
    s000_code = """
for (int nl = 0; nl < 2*iterations; nl++) {
    for (int i = 0; i < LEN_1D; i++) {
        a[i] = b[i] + 1.0f;
    }
}
"""
    print("s000:")
    print(parse_c_code(s000_code))
    print()

    # s315: argmax reduction
    s315_code = """
for (int nl = 0; nl < iterations; nl++) {
        x = a[0];
        index = 0;
        for (int i = 0; i < LEN_1D; ++i) {
            if (a[i] > x) {
                x = a[i];
                index = i;
            }
        }
        chksum = x + (real_t) index;
    }
"""
    print("s315:")
    print(parse_c_code(s315_code))
    print()

    # s2111: 2D wavefront
    s2111_code = """
for (int nl = 0; nl < iterations; nl++) {
    for (int i = 1; i < LEN_2D; i++) {
        for (int j = 1; j < LEN_2D; j++) {
            aa[j][i] = (aa[j][i-1] + aa[j-1][i])/1.9f;
        }
    }
}
"""
    print("s2111:")
    print(parse_c_code(s2111_code))
    print()

    # s31111: sum reduction
    s31111_code = """
for (int nl = 0; nl < iterations*10; nl++) {
    sum = 0.;
    for (int i = 0; i < LEN_1D; i++) {
        sum += a[i];
    }
}
"""
    print("s31111:")
    print(parse_c_code(s31111_code))
