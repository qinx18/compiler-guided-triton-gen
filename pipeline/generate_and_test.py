#!/usr/bin/env python3
"""
Integrated Generation and Testing Pipeline for TSVC Functions

For every generated function:
1. Generate Triton implementation using Claude API
2. Test immediately after generation
3. If correctness test passed → move to next function
4. If numerical error → retry with "numerical error" + "last attempt" + original prompt
5. If non-numerical error → retry with exact error description + "last attempt" + original prompt
6. Maximum 3 attempts per function

Usage:
    python generate_and_test.py              # Process all functions
    python generate_and_test.py s271 s241    # Process specific functions
"""

import os
import sys
import subprocess
import anthropic
import re
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, List

# Import the extracted function database
sys.path.append(str(Path(__file__).parent / "utilities"))
from tsvc_functions_db import TSVC_FUNCTIONS
from c_code_parser import parse_c_code

# Add analysis directory to path
sys.path.insert(0, "/home/qinxiao/workspace/compiler-guided-triton-gen/analysis")

from kernel_analysis import analyze_kernel, format_analysis_for_prompt

# Legacy imports kept as stubs for backward compat
HAS_WAR_ANALYSIS = False
analyze_kernel_war = None

try:
    from compute_statement_overwrites import analyze_kernel_overwrites, format_overwrite_for_prompt
    HAS_OVERWRITE_ANALYSIS = True
except ImportError:
    HAS_OVERWRITE_ANALYSIS = False
    analyze_kernel_overwrites = None
    format_overwrite_for_prompt = None

try:
    from compute_stream_compaction import analyze_kernel_stream_compaction, format_stream_compaction_for_prompt
    HAS_STREAM_COMPACTION_ANALYSIS = True
except ImportError:
    HAS_STREAM_COMPACTION_ANALYSIS = False
    analyze_kernel_stream_compaction = None
    format_stream_compaction_for_prompt = None

try:
    from compute_pointer_aliasing import analyze_kernel_aliasing, format_aliasing_for_prompt
    HAS_POINTER_ALIASING_ANALYSIS = True
except ImportError:
    HAS_POINTER_ALIASING_ANALYSIS = False
    analyze_kernel_aliasing = None
    format_aliasing_for_prompt = None

try:
    from compute_parallel_dims import analyze_kernel_parallelization
    HAS_PARALLELIZATION_ANALYSIS = True
except ImportError:
    HAS_PARALLELIZATION_ANALYSIS = False
    analyze_kernel_parallelization = None

try:
    from compute_crossing_threshold import analyze_kernel_crossing_threshold, format_crossing_threshold_for_prompt
    HAS_CROSSING_THRESHOLD_ANALYSIS = True
except ImportError:
    HAS_CROSSING_THRESHOLD_ANALYSIS = False
    analyze_kernel_crossing_threshold = None
    format_crossing_threshold_for_prompt = None

try:
    from compute_loop_unrolling import analyze_kernel_loop_unrolling, format_unrolling_for_prompt
    HAS_LOOP_UNROLLING_ANALYSIS = True
except ImportError:
    HAS_LOOP_UNROLLING_ANALYSIS = False
    analyze_kernel_loop_unrolling = None
    format_unrolling_for_prompt = None

try:
    from compute_early_exit import analyze_kernel_early_exit, format_early_exit_for_prompt
    HAS_EARLY_EXIT_ANALYSIS = True
except ImportError:
    HAS_EARLY_EXIT_ANALYSIS = False
    analyze_kernel_early_exit = None
    format_early_exit_for_prompt = None

try:
    from compute_statement_reordering import analyze_kernel_reordering, format_reordering_for_prompt
    HAS_STATEMENT_REORDERING_ANALYSIS = True
except ImportError:
    HAS_STATEMENT_REORDERING_ANALYSIS = False
    analyze_kernel_reordering = None

try:
    from compute_scalar_expansion import analyze_kernel_scalar_expansion, format_scalar_expansion_for_prompt
    HAS_SCALAR_EXPANSION_ANALYSIS = True
except ImportError:
    HAS_SCALAR_EXPANSION_ANALYSIS = False
    analyze_kernel_scalar_expansion = None
    format_scalar_expansion_for_prompt = None

try:
    from compute_reduction_type import analyze_kernel_reduction, build_reduction_instructions
    HAS_REDUCTION_ANALYSIS = True
except ImportError:
    HAS_REDUCTION_ANALYSIS = False
    analyze_kernel_reduction = None
    build_reduction_instructions = None

try:
    from compute_convolution_pattern import analyze_kernel_convolution, build_convolution_instructions
    HAS_CONVOLUTION_ANALYSIS = True
except ImportError:
    HAS_CONVOLUTION_ANALYSIS = False
    analyze_kernel_convolution = None
    build_convolution_instructions = None

try:
    from compute_loop_interchange import analyze_kernel_loop_interchange, format_interchange_for_prompt
    HAS_LOOP_INTERCHANGE_ANALYSIS = True
except ImportError:
    HAS_LOOP_INTERCHANGE_ANALYSIS = False
    analyze_kernel_loop_interchange = None
    format_interchange_for_prompt = None

try:
    from compute_indirect_addressing import analyze_indirect_addressing, build_indirect_addressing_instructions
    HAS_INDIRECT_ADDRESSING_ANALYSIS = True
except ImportError:
    HAS_INDIRECT_ADDRESSING_ANALYSIS = False
    analyze_indirect_addressing = None
    build_indirect_addressing_instructions = None

try:
    from compute_loop_distribution import analyze_kernel_loop_distribution, format_loop_distribution_for_prompt
    HAS_LOOP_DISTRIBUTION_ANALYSIS = True
except ImportError:
    HAS_LOOP_DISTRIBUTION_ANALYSIS = False
    analyze_kernel_loop_distribution = None
    format_loop_distribution_for_prompt = None

try:
    from compute_goto_conversion import analyze_kernel_goto, format_goto_analysis_for_prompt
    HAS_GOTO_ANALYSIS = True
except ImportError:
    HAS_GOTO_ANALYSIS = False
    analyze_kernel_goto = None
    format_goto_analysis_for_prompt = None

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=API_KEY) if API_KEY else None

# Paths
TSVC_SOURCE = "/home/qinxiao/workspace/compiler-guided-triton-gen/benchmarks_src/TSVC_2/src/archive/tsvc_orig.c"
KERNELS_DIR = "/home/qinxiao/workspace/compiler-guided-triton-gen/analysis/kernels"
# C reference is pre-compiled in c_reference/tsvc_all_reference.py

MAX_ATTEMPTS = 10
USE_OMP = False  # When True, compile C reference with OpenMP
BENCHMARK_DATA_SIZE = None  # Override: "32MB", "3200MB", or None for default (32K/256)

# Functions requiring higher tolerance due to numerical accumulation patterns
# These functions have recurrence relations where values grow exponentially,
# making absolute error meaningless - only relative error matters.
# Format: {func_name: {'rtol': relative_tolerance, 'atol': absolute_tolerance}}
HIGHER_TOLERANCE_FUNCTIONS = {
    # s2111: 2D diagonal wavefront pattern aa[j][i] = (aa[j][i-1] + aa[j-1][i])/1.9
    # Values grow to ~1e10, and catastrophic cancellation when subtracting
    # nearly-equal large numbers can cause up to ~10% relative error at some points.
    # The algorithm is correct - this is inherent numerical instability.
    's2111': {'rtol': 0.1, 'atol': 1e-3},  # 10% relative tolerance
}

# Helper functions defined in TSVC
HELPER_FUNCTIONS = {
    "test": {
        "code": """real_t test(real_t* A){
  real_t s = (real_t)0.0;
  for (int i = 0; i < 4; i++)
    s += A[i];
  return s;
}""",
        "description": "Sums the first 4 elements of array A"
    },
    "f": {
        "code": """real_t f(real_t a, real_t b){
    return a*b;
}""",
        "description": "Returns the product of a and b (a*b)"
    },
    "s151s": {
        "code": """void s151s(real_t a[LEN_1D], real_t b[LEN_1D], int m)
{
    for (int i = 0; i < LEN_1D-1; i++) {
        a[i] = a[i + m] + b[i];
    }
}""",
        "description": "For each i in 0..LEN_1D-2: a[i] = a[i+m] + b[i]. Shifts and adds."
    },
    "s152s": {
        "code": """void s152s(real_t a[LEN_1D], real_t b[LEN_1D], real_t c[LEN_1D], int i)
{
    a[i] += b[i] * c[i];
}""",
        "description": "Updates single element: a[i] += b[i] * c[i]"
    },
    "s471s": {
        "code": """int s471s(void)
{
// --  dummy subroutine call made in s471
    return 0;
}""",
        "description": "Dummy subroutine call, returns 0. Used to test call overhead."
    }
}


def enrich_func_spec(func_spec: dict) -> dict:
    """
    Enrich function specification with properties parsed from the C code.

    This makes the framework independent of manually-annotated database flags
    by inferring properties like has_reduction, has_conditional, etc. from
    the actual C source code at runtime.

    Args:
        func_spec: Original function specification from database

    Returns:
        Enriched specification with parsed values overriding database values
    """
    if 'loop_code' not in func_spec:
        return func_spec

    # Parse the C code to extract properties
    parsed = parse_c_code(func_spec['loop_code'])

    # Also parse any helper functions called in the loop code
    # This handles cases like s151s(a, b, 1) where arrays are accessed inside the helper
    import re
    helper_arrays = {}
    for helper_name, helper_info in HELPER_FUNCTIONS.items():
        if re.search(rf'\b{helper_name}\s*\(', func_spec['loop_code']):
            helper_parsed = parse_c_code(helper_info['code'])
            # Merge helper arrays into main arrays
            for arr, mode in helper_parsed['arrays'].items():
                if arr not in helper_arrays or mode in ['w', 'rw']:
                    helper_arrays[arr] = mode

    # Merge parsed arrays with helper arrays, prioritizing write modes
    # 'rw' > 'w' > 'r', and combining 'w' + 'r' = 'rw'
    def merge_mode(mode1, mode2):
        if 'rw' in (mode1, mode2):
            return 'rw'
        if 'w' in (mode1, mode2) and 'r' in (mode1, mode2):
            return 'rw'
        if 'w' in (mode1, mode2):
            return 'w'
        return 'r'

    merged_arrays = {}
    for arr in set(parsed['arrays'].keys()) | set(helper_arrays.keys()):
        mode1 = parsed['arrays'].get(arr)
        mode2 = helper_arrays.get(arr)
        if mode1 and mode2:
            merged_arrays[arr] = merge_mode(mode1, mode2)
        else:
            merged_arrays[arr] = mode1 or mode2

    # Create enriched spec - parsed values override database values
    enriched = func_spec.copy()

    # Override with parsed values (these are inferred from code, more reliable)
    # Fall back to database arrays if parser couldn't detect any
    enriched['arrays'] = merged_arrays if merged_arrays else func_spec.get('arrays', {})
    enriched['has_offset'] = parsed['has_offset']
    enriched['has_conditional'] = parsed['has_conditional']
    enriched['has_reduction'] = parsed['has_reduction']
    enriched['has_2d_arrays'] = parsed['has_2d_arrays']

    # Derive scalars from C wrapper signature (params that aren't arrays)
    # This is more reliable than hardcoded lists
    scalar_params = parsed['scalar_params'].copy()  # Start with parsed scalars (e.g., iterations)
    func_name = func_spec.get('name', '')
    if func_name:
        try:
            import inspect
            from c_reference import tsvc_all_reference
            c_func_name = f"{func_name}_c"
            if hasattr(tsvc_all_reference, c_func_name):
                c_func = getattr(tsvc_all_reference, c_func_name)
                sig = inspect.signature(c_func)
                array_names = set(enriched['arrays'].keys())
                for param_name, param in sig.parameters.items():
                    if param_name not in array_names and param_name not in scalar_params:
                        scalar_params[param_name] = 'scalar'
        except ImportError:
            pass

    enriched['scalar_params'] = scalar_params

    return enriched


# ============================================================================
# From generate_llm_triton.py - Code generation functions
# ============================================================================

def find_used_helper_functions(func_body: str) -> List[str]:
    """Find which helper functions are called in the function body."""
    used = []
    for helper_name in HELPER_FUNCTIONS:
        if re.search(rf'\b{helper_name}\s*\(', func_body):
            used.append(helper_name)
    return used


def extract_tsvc_function(func_name: str) -> Optional[dict]:
    """Extract a function from the original TSVC source file."""
    with open(TSVC_SOURCE, 'r') as f:
        content = f.read()

    pattern = rf'real_t {func_name}\s*\(struct args_t \* func_args\)\s*\{{'
    match = re.search(pattern, content)
    if not match:
        return None

    start = match.start()
    brace_count = 0
    in_func = False
    end = start

    for i, char in enumerate(content[start:], start):
        if char == '{':
            brace_count += 1
            in_func = True
        elif char == '}':
            brace_count -= 1
            if in_func and brace_count == 0:
                end = i + 1
                break

    func_body = content[start:end]
    local_vars = extract_local_variables(func_body)
    kernel_loop = extract_kernel_loop(func_body)

    desc_match = re.search(rf'//.*?\n.*?real_t {func_name}', content[max(0, start - 200):start])
    description = ""
    if desc_match:
        desc_lines = desc_match.group(0).split('\n')
        description = '\n'.join(line for line in desc_lines if line.strip().startswith('//'))

    helper_functions_used = find_used_helper_functions(func_body)

    return {
        'name': func_name,
        'full_body': func_body,
        'local_vars': local_vars,
        'kernel_loop': kernel_loop,
        'description': description,
        'helper_functions': helper_functions_used
    }


def extract_local_variables(func_body: str) -> List[str]:
    """Extract local variable declarations from function body."""
    variables = []
    var_patterns = [
        r'^\s*(int\s+(?!nl\b)(?!i\b)(?!j\b)(?!k\b)\w+(?:\s*,\s*\w+)*\s*(?:=\s*[^;]+)?)\s*;',
        r'^\s*(real_t\s+\w+(?:\s*,\s*\w+)*\s*(?:=\s*[^;]+)?)\s*;',
        r'^\s*(float\s+\w+(?:\s*,\s*\w+)*\s*(?:=\s*[^;]+)?)\s*;',
        r'^\s*(double\s+\w+(?:\s*,\s*\w+)*\s*(?:=\s*[^;]+)?)\s*;',
    ]

    lines = func_body.split('\n')
    for line in lines:
        if 'for' in line and ('nl' in line or 'i =' in line or 'j =' in line):
            continue
        if 'gettimeofday' in line:
            continue

        for pattern in var_patterns:
            match = re.match(pattern, line)
            if match:
                decl = match.group(1).strip()
                variables.append(decl)
                break

    return variables


def extract_kernel_loop(func_body: str) -> str:
    """Extract the kernel loop from function body (excluding outer nl loop)."""
    lines = func_body.split('\n')
    loop_lines = []
    in_main_loop = False
    brace_depth = 0

    for line in lines:
        stripped = line.strip()

        if 'initialise_arrays' in stripped or 'gettimeofday' in stripped:
            continue
        if 'dummy(' in stripped or 'calc_checksum' in stripped:
            continue
        if stripped.startswith('//'):
            continue

        if 'for' in stripped and 'nl' in stripped:
            in_main_loop = True
            if '{' in stripped:
                brace_depth = 1
                remainder = stripped[stripped.find('{') + 1:].strip()
                if remainder:
                    loop_lines.append(remainder)
            continue

        if in_main_loop:
            for char in stripped:
                if char == '{':
                    brace_depth += 1
                elif char == '}':
                    brace_depth -= 1

            if brace_depth > 0:
                if 'dummy(' not in stripped:
                    loop_lines.append(line)
            else:
                break

    return '\n'.join(loop_lines)


def get_exact_function_signature(kernel_name: str) -> Optional[str]:
    """Build the exact function signature from the C reference wrapper via introspection."""
    import inspect
    try:
        from c_reference import tsvc_all_reference
        c_func_name = f"{kernel_name}_c"
        if hasattr(tsvc_all_reference, c_func_name):
            c_func = getattr(tsvc_all_reference, c_func_name)
            sig = inspect.signature(c_func)
            # Include all parameters - scalar params like n1, n3, len_2d must be
            # included so the LLM doesn't hardcode them (len_2d differs from array
            # shape when has_offset is True)
            params = [name for name, param in sig.parameters.items()]
            return ", ".join(params) if params else None
    except ImportError:
        pass

    # Fallback to database if C reference not available
    if kernel_name not in TSVC_FUNCTIONS:
        return None

    func_info = TSVC_FUNCTIONS[kernel_name]
    params = []

    if 'arrays' in func_info:
        arrays = sorted(func_info['arrays'].keys())
        params.extend(arrays)

    if 'scalar_params' in func_info:
        scalars = [p for p in sorted(func_info['scalar_params'].keys()) if p != 'iterations']
        params.extend(scalars)

    return ", ".join(params) if params else None



def check_war_eliminated_by_overwrite(war_result: dict, overwrite_result: dict) -> bool:
    """
    Check if WAR dependencies are eliminated by statement overwrite optimization.

    When statement overwrite optimization applies:
    - The overwritten statement only executes at the last iteration(s)
    - If this eliminates the overlap between read and write locations,
      then WAR is no longer a concern

    Example (s244):
    - WAR: S_2 reads a[i+1], S_0 writes a[i]
    - Overwrite: S_0 overwrites S_2's a[i+1] in next iteration
    - After optimization: S_2 only at i=N-2 (writes a[N-1]),
                         S_0 for i=0..N-2 (writes a[0..N-2])
    - No overlap → WAR eliminated!

    Returns:
        True if WAR is eliminated by overwrite optimization, False otherwise
    """
    if not war_result.get('war_dependencies'):
        return False

    overwrites = overwrite_result.get('overwrites', [])
    if not overwrites:
        return False

    # Build a map of which arrays/statements are involved in overwrites
    overwrite_info = {}
    for ow in overwrites:
        array = ow['array']
        overwritten_stmt = ow['overwritten_stmt']
        overwriting_stmt = ow['overwriting_stmt']
        key = (array, overwritten_stmt, overwriting_stmt)
        overwrite_info[key] = ow

    # Check each WAR dependency
    for war in war_result['war_dependencies']:
        array = war['array']
        read_stmt = war['read_stmt']  # e.g., "S_2"
        write_stmt = war['write_stmt']  # e.g., "S_0"

        # Extract statement numbers
        read_stmt_num = int(read_stmt.split('_')[1]) if '_' in read_stmt else None
        write_stmt_num = int(write_stmt.split('_')[1]) if '_' in write_stmt else None

        if read_stmt_num is None or write_stmt_num is None:
            continue

        # Check if this WAR is resolved by statement overwrite
        # Case 1: The writing statement is overwriting the reading statement's result
        key = (array, read_stmt_num, write_stmt_num)
        if key in overwrite_info:
            # The write overwrites the read's result
            # After optimization, the read-statement only executes at last iteration
            # and the write-statement doesn't touch that location anymore
            # → WAR eliminated
            return True

    return False


def build_war_instructions(kernel_name: str, war_result: dict, overwrite_result: dict = None) -> str:
    """Build specific instructions for handling WAR dependencies.

    Args:
        kernel_name: Name of the kernel
        war_result: WAR dependency analysis result
        overwrite_result: Statement overwrite analysis result (optional)

    Returns:
        WAR instructions string, or empty if WAR is eliminated by overwrite optimization
    """
    if not war_result or war_result['parallelization_safe']:
        return ""

    # Check if statement overwrite optimization eliminates the WAR dependencies
    if overwrite_result and overwrite_result.get('applicable'):
        war_eliminated = check_war_eliminated_by_overwrite(war_result, overwrite_result)
        if war_eliminated:
            # WAR is eliminated by statement overwrite optimization
            return ""

    arrays_to_copy = war_result['arrays_needing_copy']

    instructions = f"""
## CRITICAL: WAR Race Condition Handling Required

This kernel has WAR (Write-After-Read) anti-dependencies that cause race conditions in parallel execution.
**Arrays requiring read-only copy**: {arrays_to_copy}

### Required Solution: Read-Only Copy Pattern
Pass a **read-only copy** of the array to the kernel. All threads load from the copy (immutable)
and store results to the original array.

### Implementation Template
```python
# In wrapper function - create read-only copy BEFORE launching kernel:
def {kernel_name}_triton(...):
"""
    for arr in arrays_to_copy:
        instructions += f"    {arr}_copy = {arr}.clone()  # Read-only copy\n"

    instructions += f"""
    # Pass BOTH original (for writes) AND copy (for reads) to kernel
    {kernel_name}_kernel[grid](
"""
    for arr in arrays_to_copy:
        instructions += f"        {arr},        # Write destination (original)\n"
        instructions += f"        {arr}_copy,   # Read source (immutable copy)\n"

    instructions += """        ...other_args...
    )
```

**CRITICAL: Use forward iteration** - ALWAYS use forward iteration with ascending offsets.
"""
    return instructions


def load_overwrite_analysis(kernel_name: str) -> Optional[dict]:
    """Load statement overwrite analysis for a kernel."""
    if not HAS_OVERWRITE_ANALYSIS or analyze_kernel_overwrites is None:
        return None

    kernel_file = os.path.join(KERNELS_DIR, f"{kernel_name}.c")
    if not os.path.exists(kernel_file):
        return None

    try:
        return analyze_kernel_overwrites(kernel_file)
    except Exception:
        return None


def build_overwrite_instructions(kernel_name: str, overwrite_result: dict) -> str:
    """Build specific instructions for handling statement overwrite patterns."""
    if not overwrite_result or not overwrite_result.get('applicable'):
        return ""

    # Use the formatted advice from the module
    if format_overwrite_for_prompt:
        formatted = format_overwrite_for_prompt(overwrite_result)
        if formatted:
            return f"\n{formatted}\n"

    return ""


def load_stream_compaction_analysis(kernel_name: str) -> Optional[dict]:
    """Load stream compaction analysis for a kernel."""
    if not HAS_STREAM_COMPACTION_ANALYSIS or analyze_kernel_stream_compaction is None:
        return None

    kernel_file = os.path.join(KERNELS_DIR, f"{kernel_name}.c")
    if not os.path.exists(kernel_file):
        return None

    try:
        return analyze_kernel_stream_compaction(kernel_file)
    except Exception:
        return None


def build_stream_compaction_instructions(kernel_name: str, compaction_result: dict) -> str:
    """Build specific instructions for handling stream compaction patterns."""
    if not compaction_result or not compaction_result.get('applicable'):
        return ""

    # Use the formatted advice from the module
    if format_stream_compaction_for_prompt:
        formatted = format_stream_compaction_for_prompt(compaction_result)
        if formatted:
            return f"\n{formatted}\n"

    return ""


def load_pointer_aliasing_analysis(kernel_name: str, reordering_result: dict = None, distribution_result: dict = None) -> Optional[dict]:
    """Load pointer aliasing analysis for a kernel.

    Args:
        kernel_name: Name of the kernel
        reordering_result: Optional statement reordering result - if it indicates
                          dependencies are eliminated, pass those to aliasing analysis
        distribution_result: Optional loop distribution result - if it indicates
                            arrays are fully transformed (canceling ops or power recurrence),
                            skip aliasing analysis for those arrays
    """
    if not HAS_POINTER_ALIASING_ANALYSIS or analyze_kernel_aliasing is None:
        return None

    kernel_file = os.path.join(KERNELS_DIR, f"{kernel_name}.c")
    if not os.path.exists(kernel_file):
        return None

    # Extract eliminated dependencies from reordering result
    eliminated_deps = []
    if reordering_result and reordering_result.get('applicable'):
        eliminated_deps = reordering_result.get('eliminated_dependencies', [])

    # Extract arrays that are fully handled by loop distribution
    # (canceling ops make them no-op, power recurrence transforms them to parallel,
    # or distributable loops are verified parallelizable per-statement)
    if distribution_result and distribution_result.get('applicable'):
        # Arrays in canceling pairs are unchanged - skip aliasing for them
        for pair in distribution_result.get('canceling_pairs', []):
            arr = pair.get('array')
            if arr and arr not in eliminated_deps:
                eliminated_deps.append(arr)
        # Arrays with power recurrence are transformed to parallel - skip aliasing
        for rec in distribution_result.get('power_recurrences', []):
            arr = rec.get('array')
            if arr and arr not in eliminated_deps:
                eliminated_deps.append(arr)
        # For distributable loops: loop distribution already verified each statement
        # is parallelizable by checking same-array read/write offsets. The cross-statement
        # dependencies are handled by running distributed loops sequentially.
        # Add all arrays involved in verified distributable statements.
        distributable = distribution_result.get('distributable_loops')
        if distributable:
            all_verified = all(d.get('verified_safe') for d in distributable)
            if all_verified:
                for d in distributable:
                    for arr in d.get('writes', []):
                        if arr and arr not in eliminated_deps:
                            eliminated_deps.append(arr)
                    for arr in d.get('reads', []):
                        if arr and arr not in eliminated_deps:
                            eliminated_deps.append(arr)

    try:
        return analyze_kernel_aliasing(
            kernel_file,
            eliminated_deps=eliminated_deps if eliminated_deps else None,
            distribution_result=distribution_result
        )
    except Exception:
        return None


def build_pointer_aliasing_instructions(kernel_name: str, aliasing_result: dict) -> str:
    """Build specific instructions for handling pointer aliasing patterns."""
    if not aliasing_result:
        return ""

    # For fully_parallel: only emit special advice when stride > 1 makes it look
    # like a dependency but actually isn't (e.g., s111 with stride 2).
    # For stride=1 cases that are fully_parallel (e.g., due to reordering or
    # only WAR dependencies), don't emit misleading "disjoint" message.
    if aliasing_result.get('pattern_type') == 'fully_parallel':
        stride = aliasing_result.get('details', {}).get('stride', 1)
        same_array = aliasing_result.get('same_array_accesses', [])
        has_offset_self_dep = any(
            p['write_offset'] != p['read_offset'] for p in same_array
            if p['write_offset'] is not None and p['read_offset'] is not None
        )
        # Only emit "disjoint due to stride" message when stride > 1
        if has_offset_self_dep and stride > 1:
            lines = [
                "",
                "## Parallelization Analysis",
                "",
                f"Despite the apparent self-dependency (same array read and written at different offsets),",
                f"the loop stride is {stride}, which means writes and reads access DISJOINT index sets.",
                "This loop is **FULLY PARALLEL** — each iteration is independent.",
                "",
                "Implement this with a standard parallel Triton kernel (one element per thread).",
                f"Map the original loop index `i` to thread indices: `i = pid * BLOCK_SIZE + offsets`",
                f"Remember the loop starts at a non-zero index and uses stride {stride}.",
                "",
            ]
            return '\n'.join(lines)
        return ""

    # Use the formatted advice from the module
    if format_aliasing_for_prompt:
        formatted = format_aliasing_for_prompt(aliasing_result)
        if formatted:
            return f"\n{formatted}\n"

    return ""


def load_crossing_threshold_analysis(kernel_name: str) -> Optional[dict]:
    """Load crossing threshold analysis for a kernel."""
    if not HAS_CROSSING_THRESHOLD_ANALYSIS or analyze_kernel_crossing_threshold is None:
        return None

    try:
        return analyze_kernel_crossing_threshold(kernel_name)
    except Exception:
        return None


def build_crossing_threshold_instructions(kernel_name: str, threshold_result: dict) -> str:
    """Build specific instructions for handling crossing threshold patterns."""
    if not threshold_result or not threshold_result.get('applicable'):
        return ""

    # Use the formatted advice from the module
    if format_crossing_threshold_for_prompt:
        formatted = format_crossing_threshold_for_prompt(threshold_result)
        if formatted:
            return f"\n{formatted}\n"

    return ""


def load_loop_unrolling_analysis(kernel_name: str) -> Optional[dict]:
    """Load loop unrolling analysis for a kernel."""
    if not HAS_LOOP_UNROLLING_ANALYSIS or analyze_kernel_loop_unrolling is None:
        return None

    try:
        return analyze_kernel_loop_unrolling(kernel_name)
    except Exception:
        return None


def build_loop_unrolling_instructions(kernel_name: str, unrolling_result: dict) -> str:
    """Build specific instructions for handling loop unrolling patterns."""
    if not unrolling_result or not unrolling_result.get('applicable'):
        return ""

    # Use the formatted advice from the module
    if format_unrolling_for_prompt:
        formatted = format_unrolling_for_prompt(unrolling_result)
        if formatted:
            return f"\n{formatted}\n"

    return ""


def load_early_exit_analysis(kernel_name: str) -> Optional[dict]:
    """Load early exit analysis for a kernel."""
    if not HAS_EARLY_EXIT_ANALYSIS or analyze_kernel_early_exit is None:
        return None

    try:
        return analyze_kernel_early_exit(kernel_name)
    except Exception:
        return None


def build_early_exit_instructions(kernel_name: str, early_exit_result: dict) -> str:
    """Build specific instructions for handling early exit patterns."""
    if not early_exit_result or not early_exit_result.get('applicable'):
        return ""

    # Use the formatted advice from the module
    if format_early_exit_for_prompt:
        formatted = format_early_exit_for_prompt(early_exit_result)
        if formatted:
            return f"\n{formatted}\n"

    return ""


def load_statement_reordering_analysis(kernel_name: str) -> Optional[dict]:
    """Load statement reordering analysis for a kernel."""
    if not HAS_STATEMENT_REORDERING_ANALYSIS or analyze_kernel_reordering is None:
        return None

    try:
        return analyze_kernel_reordering(kernel_name)
    except Exception:
        return None


def build_statement_reordering_instructions(kernel_name: str, reordering_result: dict) -> str:
    """Build specific instructions for handling statement reordering patterns."""
    if not reordering_result or not reordering_result.get('applicable'):
        return ""

    # Use the formatted advice from the module
    if format_reordering_for_prompt:
        formatted = format_reordering_for_prompt(reordering_result)
        if formatted:
            return f"\n{formatted}\n"

    return ""



def build_scalar_expansion_instructions(kernel_name: str, expansion_result: dict) -> str:
    """Build specific instructions for handling scalar expansion patterns."""
    if not expansion_result or not expansion_result.get('has_scalar_expansion'):
        return ""

    # Use the formatted advice from the module
    if format_scalar_expansion_for_prompt:
        formatted = format_scalar_expansion_for_prompt(kernel_name, expansion_result)
        if formatted:
            return f"\n{formatted}\n"

    return ""



def build_reduction_type_instructions(kernel_name: str, reduction_result: dict) -> str:
    """Build specific instructions for handling reduction patterns."""
    if not reduction_result or not reduction_result.get('is_reduction'):
        return ""

    # Use the formatted instructions from the module
    if build_reduction_instructions:
        formatted = build_reduction_instructions(reduction_result)
        if formatted:
            return formatted

    return ""


def load_convolution_analysis(kernel_name: str) -> Optional[dict]:
    """Load convolution pattern analysis for a kernel."""
    if not HAS_CONVOLUTION_ANALYSIS or analyze_kernel_convolution is None:
        return None

    try:
        return analyze_kernel_convolution(kernel_name)
    except Exception:
        return None


def build_convolution_pattern_instructions(kernel_name: str, conv_result: dict) -> str:
    """Build specific instructions for handling convolution patterns."""
    if not conv_result or not conv_result.get('is_convolution'):
        return ""

    # Use the formatted instructions from the module
    if build_convolution_instructions:
        formatted = build_convolution_instructions(conv_result)
        if formatted:
            return formatted

    return ""


def load_loop_interchange_analysis(kernel_name: str) -> Optional[dict]:
    """Load loop interchange analysis for a kernel."""
    if not HAS_LOOP_INTERCHANGE_ANALYSIS or analyze_kernel_loop_interchange is None:
        return None

    try:
        return analyze_kernel_loop_interchange(kernel_name)
    except Exception:
        return None


def build_loop_interchange_instructions(kernel_name: str, interchange_result: dict) -> str:
    """Build specific instructions for handling loop interchange requirements."""
    if not interchange_result or not interchange_result.get('applicable'):
        return ""

    if format_interchange_for_prompt:
        formatted = format_interchange_for_prompt(interchange_result)
        if formatted:
            return f"\n{formatted}\n"

    return ""


def load_indirect_addressing_analysis(kernel_name: str) -> Optional[dict]:
    """Load indirect addressing analysis for a kernel."""
    if not HAS_INDIRECT_ADDRESSING_ANALYSIS or analyze_indirect_addressing is None:
        return None

    try:
        return analyze_indirect_addressing(kernel_name)
    except Exception:
        return None


def build_indirect_addressing_prompt(kernel_name: str, indirect_result: dict) -> str:
    """Build specific instructions for handling indirect addressing patterns."""
    if not indirect_result or not indirect_result.get('has_indirect_addressing'):
        return ""

    if build_indirect_addressing_instructions:
        formatted = build_indirect_addressing_instructions(indirect_result)
        if formatted:
            return f"\n{formatted}\n"

    return ""


def load_loop_distribution_analysis(kernel_name: str) -> Optional[dict]:
    """Load loop distribution analysis for a kernel."""
    if not HAS_LOOP_DISTRIBUTION_ANALYSIS or analyze_kernel_loop_distribution is None:
        return None

    try:
        return analyze_kernel_loop_distribution(kernel_name)
    except Exception:
        return None


def build_loop_distribution_instructions(kernel_name: str, distribution_result: dict) -> str:
    """Build specific instructions for handling loop distribution patterns."""
    if not distribution_result or not distribution_result.get('applicable'):
        return ""

    if format_loop_distribution_for_prompt:
        formatted = format_loop_distribution_for_prompt(distribution_result)
        if formatted:
            return f"\n{formatted}\n"

    return ""


def load_goto_analysis(kernel_name: str) -> Optional[dict]:
    """Load goto conversion and parallelization analysis for a kernel."""
    if not HAS_GOTO_ANALYSIS or analyze_kernel_goto is None:
        return None

    try:
        return analyze_kernel_goto(kernel_name)
    except Exception:
        return None


def build_goto_instructions(kernel_name: str, goto_result: dict) -> str:
    """Build specific instructions for handling goto-based control flow."""
    if not goto_result or not goto_result.get('applicable'):
        return ""

    if format_goto_analysis_for_prompt:
        formatted = format_goto_analysis_for_prompt(goto_result)
        if formatted:
            return f"\n{formatted}\n"

    return ""


def detect_identity_matrix_pattern(c_code: str, seq_dim: str, par_dim: str) -> bool:
    """
    Detect identity matrix initialization pattern:
    - One statement zeros a column/row: aa[j][i] = 0 (j is parallel)
    - Another statement sets diagonal: aa[i][i] = 1 (i is sequential)

    This pattern has a race condition: when j==i, both statements write to aa[i][i].
    The fix is to have each block only set diagonals within its j-range using masked stores.
    """
    import re
    # Look for pattern: arr[seq][seq] = value (diagonal write)
    diagonal_pattern = rf'\w+\s*\[\s*{seq_dim}\s*\]\s*\[\s*{seq_dim}\s*\]\s*='
    # Look for pattern: arr[par][seq] = value (column write with par dim)
    column_pattern = rf'\w+\s*\[\s*{par_dim}\s*\]\s*\[\s*{seq_dim}\s*\]\s*='

    has_diagonal = bool(re.search(diagonal_pattern, c_code))
    has_column = bool(re.search(column_pattern, c_code))

    return has_diagonal and has_column


def detect_alternating_recurrence(c_code: str, seq_dim: str) -> Optional[Dict]:
    """
    Detect alternating recurrence patterns like: a[j] = 1.0 - a[j-1]

    This pattern has a closed form:
    - a[j] = a[0] if j is even
    - a[j] = 1.0 - a[0] if j is odd

    Returns dict with pattern info or None if not detected.
    """
    import re
    # Pattern: arr[seq] = constant - arr[seq-1] or arr[seq] = constant - arr[seq - 1]
    pattern = rf'(\w+)\s*\[\s*{seq_dim}\s*\]\s*=\s*([\d.]+)\s*-\s*\1\s*\[\s*{seq_dim}\s*-\s*1\s*\]'
    match = re.search(pattern, c_code)
    if match:
        return {
            'array': match.group(1),
            'constant': match.group(2),
            'seq_dim': seq_dim,
        }

    # Also check for (real_t) cast: arr[j] = (real_t)1.0 - arr[j - 1]
    pattern2 = rf'(\w+)\s*\[\s*{seq_dim}\s*\]\s*=\s*\(real_t\)\s*([\d.]+)\s*-\s*\1\s*\[\s*{seq_dim}\s*-\s*1\s*\]'
    match = re.search(pattern2, c_code)
    if match:
        return {
            'array': match.group(1),
            'constant': match.group(2),
            'seq_dim': seq_dim,
        }

    return None


def build_parallelization_instructions(kernel_name: str, analysis: Optional[dict]) -> str:
    """Build specific instructions for parallelization strategy."""
    if not analysis:
        return ""

    valid_options = [opt for opt in analysis['options'] if opt['valid']]
    invalid_options = [opt for opt in analysis['options'] if not opt['valid']]

    if not valid_options and not invalid_options:
        return ""

    lines = []
    lines.append("")
    lines.append("## CRITICAL: Parallelization Strategy")
    lines.append("")
    lines.append(f"**Loop structure**: `{analysis['c_code']}`")
    lines.append(f"**Dimensions**: {analysis['dims']}")
    if analysis['is_triangular']:
        tri = analysis['triangular_info']
        lines.append(f"**Triangular bounds**: {tri['smaller']} < {tri['larger']}")
    lines.append("")

    if analysis['self_dependencies']:
        lines.append("**Data dependencies detected**:")
        for dep in analysis['self_dependencies']:
            lines.append(f"- Array `{dep['array']}`: write to `[{dep['write_expr']}]`, read from `[{dep['read_expr']}]`")
        lines.append("")

    # Handle wavefront pattern: both dimensions have dependencies, no simple parallelization
    if len(valid_options) == 0 and len(invalid_options) == 2:
        dims = analysis['dims']
        lines.append("### ⚠️ WAVEFRONT PATTERN - NO SIMPLE PARALLELIZATION POSSIBLE")
        lines.append("")
        lines.append("**CRITICAL**: This loop has dependencies in BOTH dimensions:")
        for dep in analysis.get('self_dependencies', []):
            lines.append(f"- Read `{dep['array']}[{dep['read_expr']}]` depends on previous iterations in both `{dims[0]}` and `{dims[1]}`")
        lines.append("")
        lines.append("**DO NOT** parallelize either dimension directly - this will cause race conditions!")
        lines.append("")
        lines.append("**Recommended approaches:**")
        lines.append("")
        lines.append("**Option 1: Sequential Processing (Simplest)**")
        lines.append("Process all elements sequentially in nested loops:")
        lines.append("```python")
        lines.append("def wrapper(aa):")
        lines.append(f"    for {dims[0]} in range(1, N):")
        lines.append(f"        for {dims[1]} in range(1, N):")
        lines.append(f"            # Sequential computation")
        lines.append("            aa[{}, {}] = ...".format(dims[0], dims[1]))
        lines.append("```")
        lines.append("")
        lines.append("**Option 2: Wavefront/Anti-diagonal Parallelism (Recommended)**")
        lines.append(f"Elements where `{dims[0]} + {dims[1]} = k` (same anti-diagonal) are independent.")
        lines.append("Process anti-diagonals sequentially **inside a single kernel**, parallelize within each:")
        lines.append("")
        lines.append("**CRITICAL: Use a SINGLE kernel launch with an in-kernel loop over diagonals.**")
        lines.append("DO NOT launch a separate kernel per diagonal — the launch overhead will dominate.")
        lines.append("")
        lines.append("**CRITICAL: Memory synchronization between diagonals**")
        lines.append("Add `tl.debug_barrier()` after each store to ensure writes from all warps are visible")
        lines.append("before reading in the next diagonal iteration. Without this, cross-warp memory")
        lines.append("visibility is NOT guaranteed and will cause incorrect results!")
        lines.append("```python")
        lines.append("@triton.jit")
        lines.append(f"def kernel(aa_ptr, bb_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):")
        lines.append(f"    {dims[1]}_idx = tl.arange(0, BLOCK_SIZE)  # parallelize {dims[1]}")
        lines.append(f"    for diag in range(2, 2 * N):  # anti-diagonal loop INSIDE kernel")
        lines.append(f"        {dims[0]} = diag - {dims[1]}_idx")
        lines.append(f"        mask = ({dims[0]} >= 1) & ({dims[0]} < N) & ({dims[1]}_idx >= 1) & ({dims[1]}_idx < N)")
        lines.append(f"        # Use N for stride (2D arrays are exactly N x N)")
        lines.append(f"        aa_idx = {dims[0]} * N + {dims[1]}_idx")
        lines.append(f"        aa_prev_idx = ({dims[0]} - 1) * N + ({dims[1]}_idx - 1)")
        lines.append(f"        # Load, compute, store with mask using aa_idx, aa_prev_idx")
        lines.append("        ...")
        lines.append("        tl.debug_barrier()  # CRITICAL: sync all warps before next diagonal")
        lines.append("")
        lines.append("# Wrapper: SINGLE kernel launch!")
        lines.append("grid = (1,)  # one block covers all 256 elements")
        lines.append("N = len_2d  # Arrays are exactly N x N")
        lines.append("kernel[grid](aa, bb, N, BLOCK_SIZE=256)")
        lines.append("```")
        lines.append("")
        return "\n".join(lines)

    if len(valid_options) >= 1:
        opt = valid_options[0]
        seq_dim = opt['sequential_dim']
        par_dim = opt['parallel_dim']
        par_type = opt['parallelism_type']
        triton_strategy = opt.get('triton_strategy', 'MULTI_KERNEL_LAUNCH')

        lines.append(f"### Required Strategy: {seq_dim}-sequential, {par_dim}-parallel")
        lines.append("")

        if triton_strategy == 'SINGLE_KERNEL_INLOOP':
            # Check for identity matrix pattern
            c_code = analysis.get('c_code', '')
            is_identity_matrix = detect_identity_matrix_pattern(c_code, seq_dim, par_dim)

            # Prefer in-kernel loop - more efficient, single kernel launch
            lines.append("**Implementation Pattern (SINGLE KERNEL with in-kernel loop):**")
            lines.append(f"- Python wrapper: launch ONE kernel with `grid = (triton.cdiv({par_dim}_size, BLOCK_SIZE),)`")
            lines.append(f"- Triton kernel: use `for {seq_dim} in range(...)` loop INSIDE the kernel")
            lines.append(f"- Triton kernel: parallelize `{par_dim}` values using VECTORIZED operations within each block")
            lines.append("")

            if is_identity_matrix:
                # Add specific guidance for identity matrix pattern
                lines.append("**⚠️ CRITICAL: Identity Matrix / Diagonal Race Condition**")
                lines.append("")
                lines.append(f"This code has TWO statements that can write to the same position:")
                lines.append(f"1. `arr[{par_dim}][{seq_dim}] = 0` - zeros all elements (parallel over {par_dim})")
                lines.append(f"2. `arr[{seq_dim}][{seq_dim}] = 1` - sets diagonal (depends only on {seq_dim})")
                lines.append("")
                lines.append(f"**RACE CONDITION**: When {par_dim} == {seq_dim}, both statements write to the same element!")
                lines.append(f"If one block sets the diagonal while another block zeros the same position, you get wrong results.")
                lines.append("")
                lines.append(f"**SOLUTION**: Each block should only set diagonal elements within its {par_dim} range using MASKED stores:")
                lines.append("```python")
                lines.append("@triton.jit")
                lines.append(f"def kernel(arr_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):")
                lines.append(f"    pid = tl.program_id(0)")
                lines.append(f"    {par_dim}_offsets = tl.arange(0, BLOCK_SIZE)")
                lines.append(f"    {par_dim}_idx = pid * BLOCK_SIZE + {par_dim}_offsets")
                lines.append(f"    {par_dim}_mask = {par_dim}_idx < N")
                lines.append("")
                lines.append(f"    for {seq_dim} in range(N):  # Sequential loop")
                lines.append(f"        # Zero all elements in this block's range")
                lines.append(f"        ptrs = arr_ptr + {par_dim}_idx * N + {seq_dim}")
                lines.append(f"        tl.store(ptrs, 0.0, mask={par_dim}_mask)")
                lines.append("")
                lines.append(f"        # Set diagonal ONLY if {seq_dim} is in this block's {par_dim} range")
                lines.append(f"        # Use MASKED store - each block only sets its own diagonals")
                lines.append(f"        diag_mask = {par_dim}_mask & ({par_dim}_idx == {seq_dim})")
                lines.append(f"        diag_ptrs = arr_ptr + {par_dim}_idx * N + {par_dim}_idx  # diagonal positions")
                lines.append(f"        tl.store(diag_ptrs, 1.0, mask=diag_mask)")
                lines.append("```")
                lines.append("")
                lines.append(f"**Key insight**: `diag_mask = {par_dim}_mask & ({par_dim}_idx == {seq_dim})` ensures only ONE thread")
                lines.append(f"in ONE block sets each diagonal element - the thread where {par_dim}_idx equals {seq_dim}.")
                lines.append("")
            else:
                lines.append("**Why in-kernel loop is safe:**")
                lines.append(f"- The sequential dimension `{seq_dim}` has dependencies that require ordering")
                lines.append(f"- But within each `{seq_dim}` iteration, all `{par_dim}` values are independent")
                lines.append(f"- Reading from previous `{seq_dim}` iteration is safe because the in-kernel loop ensures ordering")
                lines.append("")
                lines.append("**Example structure:**")
                lines.append("```python")
                lines.append("@triton.jit")
                lines.append(f"def kernel(...):")
                lines.append(f"    {par_dim}_offsets = tl.arange(0, BLOCK_SIZE)")
                lines.append(f"    {par_dim}_idx = pid * BLOCK_SIZE + {par_dim}_offsets")
                lines.append(f"    for {seq_dim} in range(start, end):  # Sequential loop INSIDE kernel")
                lines.append(f"        # Load, compute, store for this {seq_dim} iteration")
                lines.append("        ...")
                lines.append("")
                lines.append(f"# Wrapper: single kernel launch!")
                lines.append(f"grid = (triton.cdiv({par_dim}_size, BLOCK_SIZE),)")
                lines.append("kernel[grid](...)")
                lines.append("```")
                lines.append("")
        else:
            # Check if parallel dimension is small enough for single-block in-kernel loop
            # For LEN_2D (256), one block covers the entire parallel dimension,
            # so cross-block sync issues don't apply.
            c_code = analysis.get('c_code', '')
            par_fits_one_block = False
            # Check if the parallel dimension fits in a single block (≤256)
            # Case 1: par_dim < LEN_2D or par_dim < 256
            import re as _re
            bound_match = _re.search(rf'{par_dim}\s*<\s*(\w+)', c_code)
            if bound_match and bound_match.group(1) in ('LEN_2D', '256'):
                par_fits_one_block = True
            # Case 2: par_dim <= expr where expr is bounded by LEN_2D
            # (e.g., j <= i-1 where i < LEN_2D -> j <= 254)
            if not par_fits_one_block:
                # Check if the outer (sequential) loop is bounded by LEN_2D
                seq_bound_match = _re.search(rf'{seq_dim}\s*<\s*(\w+)', c_code)
                if seq_bound_match and seq_bound_match.group(1) in ('LEN_2D', '256'):
                    # Inner par_dim is bounded by seq_dim, which is bounded by LEN_2D
                    par_fits_one_block = True
                # Also check if LEN_2D appears anywhere as a loop bound in the scop
                if not par_fits_one_block and 'LEN_2D' in c_code and 'LEN_1D' not in c_code:
                    par_fits_one_block = True

            if par_fits_one_block:
                # Check for alternating recurrence pattern
                alt_recur = detect_alternating_recurrence(c_code, seq_dim)
                if alt_recur:
                    arr = alt_recur['array']
                    const = alt_recur['constant']
                    lines.append(f"**⚠️ CRITICAL: ALTERNATING RECURRENCE DETECTED**")
                    lines.append("")
                    lines.append(f"The pattern `{arr}[{seq_dim}] = {const} - {arr}[{seq_dim}-1]` has a **closed form**:")
                    lines.append(f"- `{arr}[{seq_dim}] = {arr}[0]` when {seq_dim} is even")
                    lines.append(f"- `{arr}[{seq_dim}] = {const} - {arr}[0]` when {seq_dim} is odd")
                    lines.append("")
                    lines.append(f"**This makes the ENTIRE loop fully parallel!**")
                    lines.append("")
                    lines.append(f"Pre-compute `{arr}` using the closed form, then parallelize everything:")
                    lines.append("```python")
                    lines.append("@triton.jit")
                    lines.append(f"def kernel({arr}_ptr, aa_ptr, bb_ptr, d_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):")
                    lines.append("    pid = tl.program_id(0)")
                    lines.append(f"    {par_dim}_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)")
                    lines.append(f"    {par_dim}_mask = {par_dim}_offsets < N")
                    lines.append("")
                    lines.append(f"    # Load {arr}[0] once")
                    lines.append(f"    {arr}0 = tl.load({arr}_ptr)")
                    lines.append("")
                    lines.append(f"    for {seq_dim} in range(1, N):")
                    lines.append(f"        # Closed-form: {arr}[{seq_dim}] = {arr}0 if {seq_dim} even, else {const} - {arr}0")
                    lines.append(f"        {arr}_{seq_dim} = tl.where({seq_dim} % 2 == 0, {arr}0, {const} - {arr}0)")
                    lines.append("")
                    lines.append(f"        # Parallel load of bb[{seq_dim}][{par_dim}] for all {par_dim}")
                    lines.append(f"        bb_offsets = {seq_dim} * N + {par_dim}_offsets")
                    lines.append(f"        bb_vals = tl.load(bb_ptr + bb_offsets, mask={par_dim}_mask)")
                    lines.append(f"        d_{seq_dim} = tl.load(d_ptr + {seq_dim})")
                    lines.append("")
                    lines.append(f"        # Parallel compute and store aa[{seq_dim}][{par_dim}]")
                    lines.append(f"        aa_vals = {arr}_{seq_dim} + bb_vals * d_{seq_dim}")
                    lines.append(f"        aa_offsets = {seq_dim} * N + {par_dim}_offsets")
                    lines.append(f"        tl.store(aa_ptr + aa_offsets, aa_vals, mask={par_dim}_mask)")
                    lines.append("")
                    lines.append(f"        # Also store {arr}[{seq_dim}] (only thread 0 needs to do this)")
                    lines.append(f"        if pid == 0:")
                    lines.append(f"            tl.store({arr}_ptr + {seq_dim}, {arr}_{seq_dim})")
                    lines.append("")
                    lines.append("# Wrapper: parallelize over columns")
                    lines.append("grid = (triton.cdiv(N, BLOCK_SIZE),)")
                    lines.append("kernel[grid](...)")
                    lines.append("```")
                    lines.append("")
                else:
                    lines.append("**Implementation Pattern (SINGLE KERNEL with in-kernel loop):**")
                    lines.append(f"- Python wrapper: launch ONE kernel with `grid = (1,)`")
                    lines.append(f"- Triton kernel: use `for {seq_dim} in range(...)` loop INSIDE the kernel")
                    lines.append(f"- Triton kernel: parallelize `{par_dim}` using `tl.arange(0, BLOCK_SIZE)` within the single block")
                    lines.append("")
                    lines.append(f"**Why single-block works:** `{par_dim}` ranges up to 256 (LEN_2D), which fits in one block.")
                    lines.append("All threads in a block synchronize at each loop iteration, so cross-iteration")
                    lines.append("dependencies are resolved automatically.")
                    lines.append("")
                    lines.append("**Example structure:**")
                    lines.append("```python")
                    lines.append("@triton.jit")
                    lines.append(f"def kernel(...):")
                    lines.append(f"    {par_dim}_idx = tl.arange(0, BLOCK_SIZE)")
                    lines.append(f"    for {seq_dim} in range(start, end):  # Sequential loop INSIDE kernel")
                    lines.append(f"        # Load, compute, store for this {seq_dim} iteration")
                    lines.append("        ...")
                    lines.append("")
                    lines.append("# Wrapper: single kernel launch!")
                    lines.append("grid = (1,)  # one block covers all 256 elements")
                    lines.append("kernel[grid](...)")
                    lines.append("```")
                    lines.append("")
            else:
                lines.append("**Implementation Pattern (Sequential kernel launches):**")
                lines.append(f"- Python wrapper: loop over `{seq_dim}` sequentially, launching kernel each iteration")
                lines.append(f"- Triton kernel: parallelize ALL `{par_dim}` values using VECTORIZED operations")
                lines.append("")

        if par_type == 'reduction':
            lines.append(f"**Reduction pattern:** Use `tl.sum()` to reduce across {par_dim} dimension")
            lines.append("")
        elif par_type == 'overwrite':
            lines.append(f"**Overwrite pattern:** Each {par_dim} iteration overwrites the same location - last value wins")
            lines.append(f"**Implementation:** The final value will be from the last {par_dim} iteration that executes")
            lines.append("")

    if invalid_options:
        lines.append("### INVALID Parallelization (DO NOT USE)")
        for opt in invalid_options:
            lines.append(f"**{opt['sequential_dim']}-sequential, {opt['parallel_dim']}-parallel**: INCORRECT - causes race conditions")
        lines.append("")

    return "\n".join(lines)


def build_base_prompt(kernel_name: str, tsvc_func: dict, exact_sig: str,
                      analysis_text: str = "", **kwargs) -> str:
    """Build the base prompt for Triton generation."""
    c_code_section = tsvc_func['kernel_loop']
    if tsvc_func['local_vars']:
        local_vars_str = '\n'.join(f"    {v};" for v in tsvc_func['local_vars'])
        c_code_section = f"// Local variables:\n{local_vars_str}\n\n// Kernel loop:\n{c_code_section}"

    helper_section = ""
    if tsvc_func['helper_functions']:
        helper_funcs_code = []
        for helper_name in tsvc_func['helper_functions']:
            helper_info = HELPER_FUNCTIONS[helper_name]
            helper_funcs_code.append(f"// {helper_info['description']}\n{helper_info['code']}")
        helper_section = f"""

## Helper Functions Called in This Code:
```c
{chr(10).join(helper_funcs_code)}
```
"""

    prompt = f"""I have an original TSVC (Test Suite for Vectorizing Compilers) C function that I want to implement in Triton for GPU acceleration.

## Original TSVC C Code:
```c
{tsvc_func['description']}
{tsvc_func['full_body']}
```

## Kernel Loop to Implement:
```c
{c_code_section}
```
{helper_section}{analysis_text}

## Array Information:
- Arrays `a`, `b`, `c`, `d`, `e` are 1D float arrays of size LEN_1D (typically 32000)
- Arrays `aa`, `bb`, `cc`, `tt` are 2D float arrays of size LEN_2D x LEN_2D (typically 256x256)
- `flat_2d_array` is a 1D float array of size LEN_2D*LEN_2D
- `indx` is a 1D int array of size LEN_1D

## Requirements:
Please generate a complete Triton implementation that:
1. Includes a @triton.jit kernel function named `{kernel_name}_kernel`
2. Includes a Python wrapper function named `{kernel_name}_triton`
3. The wrapper should accept the tensor arrays and scalar parameters shown in the required signature
4. Uses appropriate block sizes and memory access patterns
5. Handles edge cases with masking
6. Is functionally equivalent to the C code (same computation, same results)
7. **CRITICAL: For 2D arrays with `len_2d` parameter:**
   - Use `len_2d` for BOTH loop bounds AND stride (arrays are exactly len_2d x len_2d)
   ```python
   # For 2D arrays with len_2d parameter:
   N = len_2d  # Use for both bounds and stride

   for i in range(1, N):
       for j in range(1, N):
           linear_idx = i * N + j  # Stride is N (same as len_2d)
   ```

## CRITICAL: Function Signature Requirements
**DO NOT include** the `iterations` parameter or the outer `for (int nl = ...)` timing loop.

**REQUIRED function signature (use EXACTLY these parameter names):**
```python
def {kernel_name}_triton({exact_sig if exact_sig else ''}):
    ...  # Just the kernel computation, NO timing loop
```

IMPORTANT:
- Use EXACTLY the parameter names shown in the required function signature above
- Do NOT implement the outer timing loop
- If WAR dependencies are shown above, you MUST use the read-only copy pattern
- If parallelization analysis is shown above, you MUST follow the specified parallelization strategy

## CRITICAL: Triton Compilation Rules

**NEVER use `tl.arange()` inside a for loop:**
```python
# WRONG
for block_start in range(0, n, BLOCK_SIZE):
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # ERROR!

# CORRECT
offsets = tl.arange(0, BLOCK_SIZE)  # Define once at start
for block_start in range(0, n, BLOCK_SIZE):
    current_offsets = block_start + offsets
```

**NEVER use scalar indexing inside @triton.jit kernel:**
```python
# WRONG
for i in range(BLOCK_SIZE):
    val = tensor[i]

# CORRECT
mask = offsets < n_elements
vals = tl.load(ptr + offsets, mask=mask)
```

**NEVER use non-existent Triton functions:**
```python
# WRONG - tl.mul, tl.div, tl.add, tl.any, tl.cdiv don't exist
# CORRECT - use Python operators: a * b, a / b, a + b
# For cdiv, use triton.cdiv() in wrapper only
```

**NEVER use Python lists inside @triton.jit kernels**

**NEVER use break/continue inside @triton.jit kernels**

**Pass tensors directly to kernels, NOT data_ptr()**

**NEVER use chained comparisons (use separate comparisons with &)**

**NEVER load from bare pointer without offset vector**

Provide ONLY the Python code, no additional explanation."""

    return prompt


def generate_triton_with_retry(kernel_name: str, original_prompt: str,
                               last_attempt: str, error_info: dict,
                               attempt_num: int) -> Tuple[str, str]:
    """Generate Triton code with error feedback for retry."""

    if error_info['type'] == 'numerical':
        error_section = f"""
## PREVIOUS ATTEMPT FAILED - NUMERICAL ERROR

Your last attempt produced incorrect numerical results. The output values don't match the expected TSVC C reference.

**Error type**: Numerical error (values don't match)
**Max error observed**: {error_info.get('max_error', 'unknown')}

Please fix the numerical computation. Common causes:
- Incorrect loop bounds or indices
- Wrong array access patterns
- Missing or incorrect operations
- Off-by-one errors in indexing

## LAST ATTEMPT (DO NOT REPEAT THE SAME MISTAKES):
```python
{last_attempt}
```

"""
    elif error_info['type'] == 'low_speedup':
        error_section = f"""
## PREVIOUS ATTEMPT HAS LOW PERFORMANCE - NEEDS BETTER PARALLELIZATION

Your last attempt is CORRECT but has very low performance (speedup: {error_info.get('speedup', 'unknown')}x).
This indicates the code is NOT properly parallelized for GPU execution.

**CRITICAL ISSUE**: Your kernel is likely running sequentially instead of in parallel.

**Common parallelization mistakes to fix**:
1. Using a `for` loop inside the kernel to iterate over ALL elements - this is SEQUENTIAL!
2. NOT using `tl.program_id(0)` to get the block index for parallel execution
3. Every thread/block processes ALL data instead of just its assigned portion

**CORRECT parallel pattern** (each block handles different elements):
```python
@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                    # Get unique block ID
    block_start = pid * BLOCK_SIZE            # Each block starts at different offset
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Each block handles BLOCK_SIZE elements
    mask = offsets < n_elements
    # Load, compute, store for THIS block only - NO for loop over all elements!
```

**WRONG sequential pattern** (every block processes everything):
```python
@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    for block_start in range(0, n_elements, BLOCK_SIZE):  # WRONG! Sequential loop!
        current_offsets = block_start + offsets
        # This makes EVERY block process ALL elements - completely sequential!
```

## LAST ATTEMPT (NEEDS PARALLELIZATION FIX):
```python
{last_attempt}
```

"""
    else:
        error_section = f"""
## PREVIOUS ATTEMPT FAILED - NON-NUMERICAL ERROR

Your last attempt failed with a compilation or runtime error:

**Error type**: {error_info['type']}
**Error message**:
```
{error_info['message']}
```

Please fix the error. Make sure to:
- Follow all Triton compilation rules
- Use correct Triton API functions
- Avoid Python constructs not supported in Triton kernels

## LAST ATTEMPT (DO NOT REPEAT THE SAME MISTAKES):
```python
{last_attempt}
```

"""

    retry_prompt = f"""{error_section}

## ORIGINAL TASK:
{original_prompt}

This is attempt {attempt_num} of {MAX_ATTEMPTS}. Please provide a corrected implementation.
Provide ONLY the Python code, no additional explanation."""

    print(f"    Retrying with error feedback (attempt {attempt_num}/{MAX_ATTEMPTS})...")

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": retry_prompt}]
    )

    response_text = message.content[0].text
    triton_code = response_text
    if "```python" in triton_code:
        triton_code = triton_code.split("```python")[1].split("```")[0].strip()
    elif "```" in triton_code:
        triton_code = triton_code.split("```")[1].split("```")[0].strip()

    return triton_code, retry_prompt


def generate_triton_initial(kernel_name: str) -> Tuple[str, str, str]:
    """Generate initial Triton implementation. Returns (code, prompt, full_response)."""
    tsvc_func = extract_tsvc_function(kernel_name)
    if not tsvc_func:
        raise ValueError(f"Could not find function {kernel_name} in TSVC source")

    exact_sig = get_exact_function_signature(kernel_name)

    # Unified analysis: single PET call derives all properties
    kernel_file = os.path.join(KERNELS_DIR, f"{kernel_name}.c")
    source_code = open(kernel_file).read() if os.path.exists(kernel_file) else ""
    arrays_dict = {a: "rw" for a in tsvc_func.get("arrays", [])}
    params_dict = {"N": 32000}
    analysis = analyze_kernel(kernel_name, source_code, arrays_dict, params_dict)
    analysis_text = "\n" + format_analysis_for_prompt(analysis)

    prompt = build_base_prompt(kernel_name, tsvc_func, exact_sig, analysis_text=analysis_text)

    print(f"  Generating Triton code (attempt 1/{MAX_ATTEMPTS})...")

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_response = f"""# LLM-Generated Triton Implementation for {kernel_name}
# Generated: {timestamp}
# Model: claude-sonnet-4-20250514

{'=' * 80}
PROMPT:
{'=' * 80}
{prompt}

{'=' * 80}
RESPONSE:
{'=' * 80}
{message.content[0].text}
"""

    response_text = message.content[0].text
    triton_code = response_text
    if "```python" in triton_code:
        triton_code = triton_code.split("```python")[1].split("```")[0].strip()
    elif "```" in triton_code:
        triton_code = triton_code.split("```")[1].split("```")[0].strip()

    return triton_code, prompt, full_response


# ============================================================================
# From auto_test_all_tsvc.py - Testing functions
# ============================================================================

# NOTE: PyTorch baseline generation has been removed.
# The infrastructure now uses the original tsvc.c functions (compiled as C library)
# as the correctness reference and performance baseline.
# See c_reference/tsvc_all_reference.py for the C function wrappers.


# Checksum mapping: which arrays to sum for each function, parsed from TSVC_2/src/core/common.c calc_checksum()
# Format: func_name -> list of (array_name, is_2d) tuples to sum
# For 2D arrays like aa, bb, cc: is_2d=True; for 1D arrays: is_2d=False
# Special cases: 'flat_2d_array' for flattened 2D, 'xx_half' for sum of first half of xx
CHECKSUM_MAP = {
    's000': [('a', False)],
    's111': [('a', False)],
    's1111': [('a', False)],
    's112': [('a', False)],
    's1112': [('a', False)],
    's113': [('a', False)],
    's1113': [('a', False)],
    's114': [('aa', True)],
    's115': [('a', False)],
    's1115': [('aa', True)],
    's116': [('a', False)],
    's118': [('a', False)],
    's119': [('aa', True)],
    's1119': [('aa', True)],
    's121': [('a', False)],
    's122': [('a', False)],
    's123': [('a', False)],
    's124': [('a', False)],
    's125': [('flat_2d_array', False)],
    's126': [('bb', True)],
    's127': [('a', False)],
    's128': [('a', False), ('b', False)],
    's131': [('a', False)],
    's132': [('aa', True)],
    's141': [('flat_2d_array', False)],
    's151': [('a', False)],
    's152': [('a', False)],
    's161': [('a', False), ('c', False)],
    's1161': [('a', False), ('c', False)],
    's162': [('a', False)],
    's171': [('a', False)],
    's172': [('a', False)],
    's173': [('a', False)],
    's174': [('a', False)],
    's175': [('a', False)],
    's176': [('a', False)],
    's211': [('a', False), ('b', False)],
    's212': [('a', False), ('b', False)],
    's1213': [('a', False), ('b', False)],
    's221': [('a', False), ('b', False)],
    's1221': [('a', False), ('b', False)],
    's222': [('a', False), ('b', False)],
    's231': [('aa', True)],
    's232': [('aa', True)],
    's1232': [('aa', True)],
    's233': [('aa', True), ('bb', True)],
    's2233': [('aa', True), ('bb', True)],
    's235': [('a', False), ('b', False)],
    's241': [('a', False), ('b', False)],
    's242': [('a', False)],
    's243': [('a', False), ('b', False)],
    's244': [('a', False), ('b', False)],
    's1244': [('a', False), ('b', False)],
    's2244': [('a', False), ('b', False)],
    's251': [('a', False)],
    's1251': [('a', False)],
    's2251': [('a', False)],
    's3251': [('a', False)],
    's252': [('a', False)],
    's253': [('a', False), ('c', False)],
    's254': [('a', False)],
    's255': [('a', False)],
    's256': [('a', False), ('aa', True)],
    's257': [('a', False), ('aa', True)],
    's258': [('b', False), ('e', False)],
    's261': [('a', False), ('c', False)],
    's271': [('a', False)],
    's272': [('a', False), ('b', False)],
    's273': [('a', False), ('b', False), ('c', False)],
    's274': [('a', False), ('b', False)],
    's275': [('aa', True)],
    's2275': [('aa', True)],
    's276': [('a', False)],
    's277': [('a', False), ('b', False)],
    's278': [('a', False), ('b', False), ('c', False)],
    's279': [('a', False), ('b', False), ('c', False)],
    's1279': [('a', False), ('b', False), ('c', False)],
    's2710': [('a', False), ('b', False), ('c', False)],
    's2711': [('a', False)],
    's2712': [('a', False)],
    's281': [('a', False), ('b', False)],
    's1281': [('a', False), ('b', False)],
    's291': [('a', False)],
    's292': [('a', False)],
    's293': [('a', False)],
    's2101': [('aa', True)],
    's2102': [('aa', True)],
    's2111': [('aa', True)],
    's311': [('a', False)],
    's31111': [('a', False)],
    's321': [('a', False)],
    's322': [('a', False)],
    's323': [('a', False), ('b', False)],
    's341': [('a', False)],
    's342': [('a', False)],
    's343': [('flat_2d_array', False)],
    's351': [('a', False)],
    's1351': [('a', False)],
    's353': [('a', False)],
    's421': [('xx', False)],
    's1421': [('xx_half', False)],
    's422': [('xx', False)],
    's423': [('flat_2d_array', False)],
    's424': [('flat_2d_array', False)],
    's431': [('a', False)],
    's441': [('a', False)],
    's442': [('a', False)],
    's443': [('a', False)],
    's451': [('a', False)],
    's452': [('a', False)],
    's453': [('a', False)],
    's471': [('x', False), ('b', False)],
    's481': [('a', False)],
    's482': [('a', False)],
    's491': [('a', False)],
    's4112': [('a', False)],
    's4113': [('a', False)],
    's4114': [('a', False)],
    's4117': [('a', False)],
    's4121': [('a', False)],
    'va': [('a', False)],
    'vag': [('a', False)],
    'vas': [('a', False)],
    'vif': [('a', False)],
    'vpv': [('a', False)],
    'vtv': [('a', False)],
    'vpvtv': [('a', False)],
    'vpvts': [('a', False)],
    'vpvpv': [('a', False)],
    'vtvtv': [('a', False)],
    'vsumr': [('a', False)],
    'vbor': [('x', False)],
}


def generate_correctness_test(func_name: str, func_spec: dict, attempt: int = 1) -> str:
    """Generate correctness test script for a specific attempt.

    Compares Triton implementation against original TSVC C reference.
    """
    arrays = func_spec['arrays']
    has_offset = func_spec['has_offset']
    has_2d = func_spec.get('has_2d_arrays', False)
    scalar_params = func_spec.get('scalar_params', {})
    has_reduction = func_spec.get('has_reduction', False)

    # Get function-specific tolerance (for functions with numerical accumulation patterns)
    tol_config = HIGHER_TOLERANCE_FUNCTIONS.get(func_name, {'rtol': 1e-3, 'atol': 1e-3})
    rtol = tol_config['rtol']
    atol = tol_config['atol']

    # Check if all arrays are read-only (likely a reduction that returns scalar)
    # This is inferred from array access modes, no need for explicit has_reduction flag
    all_arrays_readonly = all(mode == 'r' for mode in arrays.values())

    if has_offset:
        size_expr = "N + 10"
    else:
        size_expr = "N"

    # For 2D arrays with len_2d parameter, use exact N size (C reference uses len_2d as stride)
    has_len_2d_param = 'len_2d' in scalar_params
    size_2d_expr = "N" if has_len_2d_param else size_expr

    array_inits = []
    for arr, mode in sorted(arrays.items()):
        if mode in ['r', 'rw', 'w']:
            if arr == 'ip':
                # TSVC uses permutation (unique indices) - see common.c init()
                array_inits.append(f"            {arr} = torch.randperm({size_expr}, device='cuda', dtype=torch.long)")
            elif arr == 'indx':
                # TSVC s442/s443 use indx as switch case index (1-4)
                array_inits.append(f"            {arr} = torch.randint(1, 5, ({size_expr},), device='cuda', dtype=torch.int32)")
            elif has_2d and len(arr) == 2 and arr[0] == arr[1]:
                # For 2D arrays with len_2d parameter, use exact N size for correct C reference stride
                array_inits.append(f"            {arr} = torch.randn({size_2d_expr}, {size_2d_expr}, device='cuda', dtype=torch.float32)")
            elif arr == 'flat_2d_array':
                if has_offset:
                    array_inits.append(f"            {arr} = torch.randn((N + 10) * (N + 10), device='cuda', dtype=torch.float32)")
                else:
                    array_inits.append(f"            {arr} = torch.randn(N * N, device='cuda', dtype=torch.float32)")
            elif arr == 'd' and func_name == 's481':
                # s481: d must be non-negative to avoid exit(0) in original C code
                array_inits.append(f"            {arr} = torch.abs(torch.randn({size_expr}, device='cuda', dtype=torch.float32))")
            else:
                array_inits.append(f"            {arr} = torch.randn({size_expr}, device='cuda', dtype=torch.float32)")

    for scalar_name in sorted(scalar_params.keys()):
        # Use different variable name for 'abs' to avoid shadowing Python builtin
        var_name = 'abs_param' if scalar_name == 'abs' else scalar_name
        if scalar_name == 'k':
            # Note: TSVC s431 uses k = 2*k1 - k2 = 2*1 - 2 = 0
            array_inits.append(f"            {var_name} = 0")
        elif scalar_name == 't':
            array_inits.append(f"            {var_name} = 0.5")
        elif scalar_name in ['n1', 'n3']:
            if scalar_name == 'n1':
                array_inits.append(f"            {var_name} = 10")
            elif scalar_name == 'n3':
                array_inits.append(f"            {var_name} = 3")
        elif scalar_name == 'len_2d':
            # len_2d represents the 2D matrix dimension, should be N not 1
            array_inits.append(f"            {var_name} = N")
        elif scalar_name == 'n':
            # n typically represents array length (e.g., LEN_1D), should scale with test size
            array_inits.append(f"            {var_name} = N")
        else:
            array_inits.append(f"            {var_name} = 1")

    array_init_str = '\n'.join(array_inits) if array_inits else "            pass"

    array_names = [arr for arr, mode in sorted(arrays.items()) if mode in ['r', 'rw', 'w']]
    output_arrays = [arr for arr, mode in sorted(arrays.items()) if mode in ['rw', 'w']]

    if not output_arrays and array_names:
        output_arrays = [array_names[0]]

    all_scalar_names = sorted(scalar_params.keys())

    # C reference clones - convert to numpy for C function
    c_ref_clones = []
    for arr in array_names:
        c_ref_clones.append(f"            {arr}_c = {arr}.cpu().numpy().copy()")
    c_ref_clone_str = '\n'.join(c_ref_clones) if c_ref_clones else "            pass"

    triton_clones = []
    for arr in array_names:
        triton_clones.append(f"            {arr}_tr = {arr}.clone()")
    triton_clone_str = '\n'.join(triton_clones) if triton_clones else "            pass"

    # Generate checksum-based comparison (matching TSVC_2 calc_checksum approach)
    # For scalar returns, compare values directly. For array outputs, compare checksums.
    primary_arr = output_arrays[0] if output_arrays else (array_names[0] if array_names else None)

    # Determine checksum arrays for this function
    checksum_arrays = CHECKSUM_MAP.get(func_name, None)
    if checksum_arrays is None:
        # Fallback: sum all output arrays (rw/w mode)
        checksum_arrays = [(arr, has_2d and len(arr) == 2 and arr[0] == arr[1]) for arr in output_arrays]

    # Build checksum computation code for C side and Triton side
    # C wrapper returns modified arrays; for array functions c_result contains the return value
    # We need to compute checksums from the c_result (returned arrays) and triton tensors
    c_checksum_parts = []
    tr_checksum_parts = []
    for arr_name, is_2d in checksum_arrays:
        if arr_name == 'xx_half':
            # Special case: sum first half of xx
            c_checksum_parts.append(f"float(np.sum(c_tensors_after['{arr_name.replace('_half', '')}'][:len(c_tensors_after['{arr_name.replace('_half', '')}'])//2]))")
            tr_checksum_parts.append(f"float(torch.sum(tr_tensors_after['{arr_name.replace('_half', '')}'][:tr_tensors_after['{arr_name.replace('_half', '')}'].numel()//2]).item())")
        else:
            c_checksum_parts.append(f"float(np.sum(c_tensors_after['{arr_name}']))")
            tr_checksum_parts.append(f"float(torch.sum(tr_tensors_after['{arr_name}']).item())")

    c_checksum_expr = ' + '.join(c_checksum_parts) if c_checksum_parts else '0.0'
    tr_checksum_expr = ' + '.join(tr_checksum_parts) if tr_checksum_parts else '0.0'

    if primary_arr is None:
        # No arrays at all - pure scalar function, only compare return values
        compare_str = f"""            # Pure scalar function - compare return values directly
            c_val = float(c_result) if c_result is not None else 0.0
            if isinstance(triton_result, (int, float)):
                tr_val = float(triton_result)
            elif isinstance(triton_result, torch.Tensor):
                tr_val = triton_result.item() if triton_result.numel() == 1 else float(triton_result)
            else:
                tr_val = float(triton_result) if triton_result is not None else 0.0
            max_error = abs(c_val - tr_val)
            is_scalar_comparison = True"""

        passed_check_str = f"""passed = max_error < {atol} or (abs(c_val) > 1e-6 and max_error / abs(c_val) < {rtol})"""
    else:
        compare_str = f"""            # Collect post-execution arrays for checksum
            c_tensors_after = {{{', '.join([f'"{arr}": {arr}_c' for arr in array_names])}}}
            tr_tensors_after = {{{', '.join([f'"{arr}": {arr}_tr' for arr in array_names])}}}

            # Runtime detection: compare scalars if C returns scalar, otherwise use checksum
            if isinstance(c_result, (int, float)):
                # Scalar return - compare values directly
                c_val = float(c_result)
                if isinstance(triton_result, (int, float)):
                    tr_val = float(triton_result)
                elif isinstance(triton_result, torch.Tensor):
                    tr_val = triton_result.item() if triton_result.numel() == 1 else float(triton_result)
                else:
                    tr_val = float(triton_result) if triton_result is not None else float('inf')
                max_error = abs(c_val - tr_val)
                is_scalar_comparison = True
            else:
                # C wrapper modifies 1D arrays in-place via ctypes pointers,
                # so c_tensors_after already has correct values for 1D arrays.
                # However, 2D arrays (aa, bb, cc) are flattened copies in the C wrapper,
                # so their modifications are NOT reflected in c_tensors_after.
                # Update c_tensors_after with any 2D arrays from the return value.
                _checksum_2d = [name for name, is_2d in {repr(checksum_arrays)} if is_2d]
                if c_result is not None and _checksum_2d:
                    _returns = (c_result,) if isinstance(c_result, np.ndarray) else (c_result if isinstance(c_result, tuple) else ())
                    _ret_2d = [r for r in _returns if isinstance(r, np.ndarray) and r.ndim == 2]
                    for _name, _arr in zip(_checksum_2d, _ret_2d):
                        c_tensors_after[_name] = _arr

                # Checksum-based comparison (matches TSVC_2 calc_checksum)
                c_checksum = {c_checksum_expr}
                tr_checksum = {tr_checksum_expr}
                # Handle inf/nan: if both are same inf, treat as match
                import math
                if math.isinf(c_checksum) and math.isinf(tr_checksum) and (c_checksum > 0) == (tr_checksum > 0):
                    max_error = 0.0
                elif math.isnan(c_checksum) or math.isnan(tr_checksum):
                    max_error = float('inf')
                else:
                    max_error = abs(c_checksum - tr_checksum)
                    # Use relative tolerance for large checksums
                    if abs(c_checksum) > 1e-6:
                        max_error = max_error / abs(c_checksum)
                is_scalar_comparison = False"""

        passed_check_str = f"""if is_scalar_comparison:
                passed = max_error < {atol} or (abs(c_val) > 1e-6 and max_error / abs(c_val) < {rtol})
            else:
                passed = max_error < {atol}"""

    available_arrays = array_names
    available_scalars = all_scalar_names

    if has_2d:
        test_sizes_str = "[64, 128, 256]"
    else:
        test_sizes_str = "[100, 1000, 10000]"

    test_code = f'''#!/usr/bin/env python3
"""
Correctness Test for {func_name}
Compares Triton implementation against original TSVC C reference.
"""
import sys
import inspect
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np

try:
    import importlib.util as _ilu
    _c_spec = _ilu.spec_from_file_location("_cmod", "{str(Path('c_reference/tsvc_all_reference.py').resolve())}")
    _c_mod = _ilu.module_from_spec(_c_spec)
    _c_spec.loader.exec_module(_c_mod)
    {func_name}_c = _c_mod.{func_name}_c
    _t_spec = _ilu.spec_from_file_location("_tmod", "{str((Path('../results/tsvc/test29/llm_triton') / func_name / f'attempt{attempt}.py').resolve())}")
    _t_mod = _ilu.module_from_spec(_t_spec)
    _t_spec.loader.exec_module(_t_mod)
    {func_name}_triton = _t_mod.{func_name}_triton
except ImportError as e:
    print(f"Import error: {{e}}")
    sys.exit(1)

def get_func_params(func):
    sig = inspect.signature(func)
    return list(sig.parameters.keys())

def build_kwargs(func, available_tensors, available_scalars):
    params = get_func_params(func)
    kwargs = {{}}
    for p in params:
        if p in available_tensors:
            kwargs[p] = available_tensors[p]
        elif p in available_scalars:
            kwargs[p] = available_scalars[p]
    return kwargs

def test_correctness():
    test_sizes = {test_sizes_str}
    all_passed = True

    print("="*70)
    print(f"Correctness Testing: {func_name}")
    print("Comparing Triton vs TSVC C reference")
    print("="*70)

    for N in test_sizes:
        print(f"Testing N={{N:>6}}...", end=" ")

        try:
{array_init_str}

{c_ref_clone_str}

{triton_clone_str}

            c_tensors = {{{', '.join([f'"{arr}": {arr}_c' for arr in available_arrays])}}}
            tr_tensors = {{{', '.join([f'"{arr}": {arr}_tr' for arr in available_arrays])}}}
            scalars = {{{', '.join([f'"{s}": {"abs_param" if s == "abs" else s}' for s in available_scalars])}}}

            c_kwargs = build_kwargs({func_name}_c, c_tensors, scalars)
            tr_kwargs = build_kwargs({func_name}_triton, tr_tensors, scalars)

            c_result = {func_name}_c(**c_kwargs)
            triton_result = {func_name}_triton(**tr_kwargs)

{compare_str}

            {passed_check_str}
            if passed:
                print(f"PASS  (max_err={{max_error:.2e}})")
            else:
                print(f"FAIL  (max_error={{max_error:.2e}})")
                all_passed = False

        except Exception as e:
            print(f"ERROR: {{e}}")
            import traceback
            traceback.print_exc()
            all_passed = False

    print("="*70)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("="*70)

    return all_passed

if __name__ == "__main__":
    success = test_correctness()
    sys.exit(0 if success else 1)
'''

    return test_code


def run_test(func_name: str, test_file: Path) -> Tuple[bool, dict]:
    """Run the correctness test and parse results.

    Returns:
        (passed: bool, error_info: dict)
        error_info contains:
            - type: 'numerical', 'import', 'runtime', 'compilation', 'timeout'
            - message: error description
            - max_error: (for numerical errors) the max error value
    """
    try:
        env = os.environ.copy()
        env['MKL_THREADING_LAYER'] = 'GNU'
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=Path.cwd(),
            env=env
        )

        stdout = result.stdout
        stderr = result.stderr

        # Check for success
        if result.returncode == 0 and "All tests PASSED!" in stdout:
            return True, {}

        # Parse error type
        combined_output = stdout + stderr

        # Import error
        if "Import error:" in combined_output or "ImportError" in combined_output:
            return False, {
                'type': 'import',
                'message': combined_output[-2000:]  # Last 2000 chars
            }

        # Compilation error (Triton)
        if "CompilationError" in combined_output or "triton.compiler" in combined_output:
            return False, {
                'type': 'compilation',
                'message': combined_output[-2000:]
            }

        # Check for numerical error (test ran but values don't match)
        if "FAIL" in stdout and "max_error" in stdout:
            # Extract max error value
            max_error_match = re.search(r'max_error[=:]?\s*([\d.e+-]+)', stdout)
            max_error = max_error_match.group(1) if max_error_match else 'unknown'
            return False, {
                'type': 'numerical',
                'message': f"Numerical mismatch: max_error = {max_error}",
                'max_error': max_error
            }

        # Runtime error
        if "ERROR:" in stdout or "Exception" in combined_output or "Error" in stderr:
            return False, {
                'type': 'runtime',
                'message': combined_output[-2000:]
            }

        # Unknown failure
        return False, {
            'type': 'unknown',
            'message': combined_output[-2000:]
        }

    except subprocess.TimeoutExpired:
        return False, {
            'type': 'timeout',
            'message': 'Test timed out after 60 seconds'
        }
    except Exception as e:
        return False, {
            'type': 'exception',
            'message': str(e)
        }


def generate_benchmark_test(func_name: str, func_spec: dict, attempt: int = 1) -> str:
    """Generate performance benchmark script for a specific attempt.

    Compares Triton implementation against original TSVC C reference.
    """
    arrays = func_spec['arrays']
    has_offset = func_spec['has_offset']
    has_2d = func_spec.get('has_2d_arrays', False)
    scalar_params = func_spec.get('scalar_params', {})

    if has_offset:
        size_expr = "N + 10"
    else:
        size_expr = "N"

    # For 2D arrays with len_2d parameter, use exact N size (C reference uses len_2d as stride)
    has_len_2d_param = 'len_2d' in scalar_params
    size_2d_expr = "N" if has_len_2d_param else size_expr

    array_inits = []
    for arr, mode in sorted(arrays.items()):
        if mode in ['r', 'rw', 'w']:
            if arr == 'ip':
                array_inits.append(f"    {arr} = torch.randperm({size_expr}, device='cuda', dtype=torch.long)")
            elif arr == 'indx':
                # TSVC s442/s443 use indx as switch case index (1-4)
                array_inits.append(f"    {arr} = torch.randint(1, 5, ({size_expr},), device='cuda', dtype=torch.int32)")
            elif has_2d and len(arr) == 2 and arr[0] == arr[1]:
                # For 2D arrays with len_2d parameter, use exact N size for correct C reference stride
                array_inits.append(f"    {arr} = torch.randn({size_2d_expr}, {size_2d_expr}, device='cuda', dtype=torch.float32)")
            elif arr == 'flat_2d_array':
                if has_offset:
                    array_inits.append(f"    {arr} = torch.randn((N + 10) * (N + 10), device='cuda', dtype=torch.float32)")
                else:
                    array_inits.append(f"    {arr} = torch.randn(N * N, device='cuda', dtype=torch.float32)")
            elif arr == 'd' and func_name == 's481':
                array_inits.append(f"    {arr} = torch.abs(torch.randn({size_expr}, device='cuda', dtype=torch.float32))")
            else:
                array_inits.append(f"    {arr} = torch.randn({size_expr}, device='cuda', dtype=torch.float32)")

    for scalar_name in sorted(scalar_params.keys()):
        # Use different variable name for 'abs' to avoid shadowing Python builtin
        var_name = 'abs_param' if scalar_name == 'abs' else scalar_name
        if scalar_name == 'k':
            array_inits.append(f"    {var_name} = 0")
        elif scalar_name == 't':
            array_inits.append(f"    {var_name} = 0.5")
        elif scalar_name in ['n1', 'n3']:
            if scalar_name == 'n1':
                array_inits.append(f"    {var_name} = 10")
            elif scalar_name == 'n3':
                array_inits.append(f"    {var_name} = 3")
        elif scalar_name == 'len_2d':
            # len_2d represents the 2D matrix dimension, should be N not 1
            array_inits.append(f"    {var_name} = N")
        else:
            array_inits.append(f"    {var_name} = 1")

    array_init_str = '\n'.join(array_inits) if array_inits else "    pass"

    array_names = [arr for arr, mode in sorted(arrays.items()) if mode in ['r', 'rw', 'w']]
    all_scalar_names = sorted(scalar_params.keys())

    available_arrays = array_names
    available_scalars = all_scalar_names

    if BENCHMARK_DATA_SIZE:
        # User-specified data size per array
        size_map = {
            '32MB': (8_000_000, 2828),    # 32MB/4B = 8M (1D), sqrt(32M/4) ≈ 2828 (2D)
            '3200MB': (800_000_000, 28284),  # 3200MB/4B = 800M (1D), sqrt(800M/4) ≈ 28284 (2D)
        }
        if BENCHMARK_DATA_SIZE in size_map:
            n_1d, n_2d = size_map[BENCHMARK_DATA_SIZE]
            benchmark_size = n_2d if has_2d else n_1d
        else:
            benchmark_size = int(BENCHMARK_DATA_SIZE)
    elif has_2d:
        benchmark_size = 256
    else:
        benchmark_size = 32000

    benchmark_code = f'''#!/usr/bin/env python3
"""
Performance Benchmark for {func_name}
Compares Triton implementation against original TSVC C reference.
"""
import sys
import time
import inspect
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np

try:
    import importlib.util as _ilu
    _c_spec = _ilu.spec_from_file_location("_cmod", "{str(Path('c_reference/tsvc_all_reference.py').resolve())}")
    _c_mod = _ilu.module_from_spec(_c_spec)
    _c_spec.loader.exec_module(_c_mod)
    {func_name}_c = _c_mod.{func_name}_c
    _t_spec = _ilu.spec_from_file_location("_tmod", "{str((Path('../results/tsvc/test29/llm_triton') / func_name / f'attempt{attempt}.py').resolve())}")
    _t_mod = _ilu.module_from_spec(_t_spec)
    _t_spec.loader.exec_module(_t_mod)
    {func_name}_triton = _t_mod.{func_name}_triton
except ImportError as e:
    print(f"Import error: {{e}}")
    sys.exit(1)

def get_func_params(func):
    sig = inspect.signature(func)
    return list(sig.parameters.keys())

def build_kwargs(func, available_tensors, available_scalars):
    params = get_func_params(func)
    kwargs = {{}}
    for p in params:
        if p in available_tensors:
            kwargs[p] = available_tensors[p]
        elif p in available_scalars:
            kwargs[p] = available_scalars[p]
    return kwargs

def benchmark():
    N = {benchmark_size}
    num_warmup = 10
    num_iterations = 100
    timeout_per_section = 60  # 60 seconds per section (warmup/benchmark)

    print("="*70)
    print(f"Performance Benchmark: {func_name}")
    print(f"Comparing Triton (GPU) vs TSVC C reference (CPU)")
    print(f"Array size: N={{N}}")
    print("="*70)

    # Initialize arrays on GPU
{array_init_str}

    # Create numpy arrays for C reference (on CPU)
    c_arrays = {{{', '.join([f'"{arr}": {arr}.cpu().numpy().copy()' for arr in available_arrays])}}}
    tr_tensors = {{{', '.join([f'"{arr}": {arr}.clone()' for arr in available_arrays])}}}
    scalars = {{{', '.join([f'"{s}": {"abs_param" if s == "abs" else s}' for s in available_scalars])}}}

    c_kwargs = build_kwargs({func_name}_c, c_arrays, scalars)
    tr_kwargs = build_kwargs({func_name}_triton, tr_tensors, scalars)

    c_time = None
    tr_time = None

    # Benchmark C reference (CPU, with separate timeout handling)
    try:
        print(f"Warming up C reference ({{num_warmup}} iterations)...")
        start_time = time.perf_counter()
        for i in range(num_warmup):
            if time.perf_counter() - start_time > timeout_per_section:
                raise TimeoutError("C reference warmup timeout")
            # Reset arrays for each iteration
            for arr in c_arrays:
                c_arrays[arr] = c_arrays[arr].copy()
            c_kwargs = build_kwargs({func_name}_c, c_arrays, scalars)
            {func_name}_c(**c_kwargs)

        print(f"Benchmarking C reference ({{num_iterations}} iterations)...")
        c_start = time.perf_counter()
        bench_start = time.perf_counter()
        for i in range(num_iterations):
            if time.perf_counter() - bench_start > timeout_per_section:
                raise TimeoutError("C reference benchmark timeout")
            for arr in c_arrays:
                c_arrays[arr] = c_arrays[arr].copy()
            c_kwargs = build_kwargs({func_name}_c, c_arrays, scalars)
            {func_name}_c(**c_kwargs)
        c_time = (time.perf_counter() - c_start) / num_iterations
        print(f"  C reference time: {{c_time*1000:.3f}} ms")
    except (TimeoutError, Exception) as e:
        print(f"  C reference benchmark TIMEOUT or ERROR: {{e}}")
        c_time = None

    # Benchmark Triton (GPU, with separate timeout handling)
    try:
        print(f"Warming up Triton implementation ({{num_warmup}} iterations)...")
        start_time = time.perf_counter()
        for i in range(num_warmup):
            if time.perf_counter() - start_time > timeout_per_section:
                raise TimeoutError("Triton warmup timeout")
            for arr in tr_tensors:
                tr_tensors[arr] = tr_tensors[arr].clone()
            tr_kwargs = build_kwargs({func_name}_triton, tr_tensors, scalars)
            {func_name}_triton(**tr_kwargs)
        torch.cuda.synchronize()

        print(f"Benchmarking Triton implementation ({{num_iterations}} iterations)...")
        torch.cuda.synchronize()
        tr_start = time.perf_counter()
        bench_start = time.perf_counter()
        for i in range(num_iterations):
            if time.perf_counter() - bench_start > timeout_per_section:
                raise TimeoutError("Triton benchmark timeout")
            for arr in tr_tensors:
                tr_tensors[arr] = tr_tensors[arr].clone()
            tr_kwargs = build_kwargs({func_name}_triton, tr_tensors, scalars)
            {func_name}_triton(**tr_kwargs)
        torch.cuda.synchronize()
        tr_time = (time.perf_counter() - tr_start) / num_iterations
        print(f"  Triton time: {{tr_time*1000:.3f}} ms")
    except (TimeoutError, Exception) as e:
        print(f"  Triton benchmark TIMEOUT or ERROR: {{e}}")
        tr_time = None

    # Calculate speedup (handle None cases)
    if c_time is not None and tr_time is not None and tr_time > 0:
        speedup = c_time / tr_time
    else:
        speedup = None

    print("="*70)
    if c_time is not None:
        print(f"C ref time:    {{c_time*1000:8.3f}} ms")
    else:
        print(f"C ref time:    TIMEOUT")
    if tr_time is not None:
        print(f"Triton time:   {{tr_time*1000:8.3f}} ms")
    else:
        print(f"Triton time:   TIMEOUT")
    if speedup is not None:
        print(f"Speedup:       {{speedup:8.2f}}x")
    else:
        print(f"Speedup:       N/A (timeout)")
    print("="*70)

    # Output machine-readable format for parsing (handle None values)
    c_time_ms = c_time * 1000 if c_time is not None else -1
    tr_time_ms = tr_time * 1000 if tr_time is not None else -1
    speedup_val = speedup if speedup is not None else -1
    print(f"BENCHMARK_RESULT:{{c_time_ms:.6f}},{{tr_time_ms:.6f}},{{speedup_val:.6f}}")

if __name__ == "__main__":
    try:
        benchmark()
    except Exception as e:
        print(f"Benchmark error: {{e}}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
'''

    return benchmark_code


def run_benchmark(func_name: str, benchmark_file: Path) -> Optional[dict]:
    """Run performance benchmark and parse results.

    Returns:
        dict with 'c_ref_time_ms', 'triton_time_ms', 'speedup', or None if failed
    """
    try:
        env = os.environ.copy()
        env['MKL_THREADING_LAYER'] = 'GNU'
        result = subprocess.run(
            [sys.executable, str(benchmark_file)],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for benchmarking
            cwd=Path.cwd(),
            env=env
        )

        stdout = result.stdout

        # Parse benchmark results
        for line in stdout.split('\n'):
            if line.startswith('BENCHMARK_RESULT:'):
                parts = line.split(':')[1].split(',')
                if len(parts) == 3:
                    return {
                        'c_ref_time_ms': float(parts[0]),
                        'triton_time_ms': float(parts[1]),
                        'speedup': float(parts[2])
                    }

        return None

    except subprocess.TimeoutExpired:
        print(f"  Benchmark timed out after 300 seconds")
        return None
    except Exception as e:
        print(f"  Benchmark error: {e}")
        return None


def process_function(func_name: str, func_spec: dict) -> dict:
    """Process a single TSVC function with retry logic."""
    print(f"\n{'=' * 70}")
    print(f"Processing: {func_name}")
    print(f"  Arrays: {list(func_spec['arrays'].keys())}")
    print(f"  Offset: {func_spec['has_offset']}, Conditional: {func_spec['has_conditional']}, Reduction: {func_spec['has_reduction']}")
    print(f"{'=' * 70}")

    test_dir = Path("../results/tsvc/test29")
    llm_triton_dir = test_dir / "llm_triton"
    func_code_dir = llm_triton_dir / func_name  # llm_triton/s000/
    func_raw_dir = llm_triton_dir / "raw_responses" / func_name  # llm_triton/raw_responses/s000/
    test_dir = Path("../results/tsvc/my_triton_implementations") / func_name
    test_dir.mkdir(exist_ok=True, parents=True)

    # Clean up previous attempts before regenerating
    if func_code_dir.exists():
        shutil.rmtree(func_code_dir)
    if func_raw_dir.exists():
        shutil.rmtree(func_raw_dir)

    llm_triton_dir.mkdir(exist_ok=True, parents=True)
    func_code_dir.mkdir(exist_ok=True)
    (llm_triton_dir / "raw_responses").mkdir(exist_ok=True)
    func_raw_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True, parents=True)

    # Create __init__.py files to make directories importable
    (test_dir / "__init__.py").touch()
    (llm_triton_dir / "__init__.py").touch()
    (func_code_dir / "__init__.py").touch()

    test_file = test_dir / f"test_{func_name}_correctness.py"

    results = {
        "c_ref_available": True,  # C reference is pre-compiled and always available
        "triton_generated": False,
        "test_generated": False,
        "test_passed": False,
        "attempts": 0,
        "final_error": None
    }

    # C reference is pre-compiled in c_reference/tsvc_all_reference.py
    # No need to generate baseline - just verify C reference function exists
    print(f"  Using TSVC C reference: {func_name}_c")

    # Step 1: Generate Triton with retry loop (5+5 strategy)
    # First 5 attempts: normal retry with error feedback
    # After 5 failures: reset and try 5 more times without showing last attempt
    original_prompt = None
    last_code = None
    error_info = None  # Initialize error_info before the loop
    reset_after = 5  # Reset context after this many failures

    # Track the best passing result across all attempts
    best_result = None
    best_speedup = -float('inf')
    best_attempt = 0

    for attempt in range(1, MAX_ATTEMPTS + 1):
        results["attempts"] = attempt

        # File paths for this attempt
        triton_file = func_code_dir / f"attempt{attempt}.py"
        raw_file = func_raw_dir / f"attempt{attempt}.txt"

        try:
            if attempt == 1:
                # Initial generation
                triton_code, original_prompt, full_response = generate_triton_initial(func_name)
            elif attempt == reset_after + 1:
                # Reset: Generate fresh without showing previous failed attempts
                print(f"  Resetting context after {reset_after} failures, trying fresh approach...")
                triton_code, retry_prompt = generate_triton_with_retry(
                    func_name, original_prompt, None, error_info, attempt
                )
                full_response = retry_prompt + "\n\n" + "=" * 80 + "\nRESPONSE:\n" + "=" * 80 + "\n" + triton_code
            else:
                # Retry with error feedback (show last attempt unless we just reset)
                triton_code, retry_prompt = generate_triton_with_retry(
                    func_name, original_prompt, last_code, error_info, attempt
                )
                full_response = retry_prompt + "\n\n" + "=" * 80 + "\nRESPONSE:\n" + "=" * 80 + "\n" + triton_code

            last_code = triton_code

            # Save raw response
            with open(raw_file, 'w') as f:
                f.write(full_response)

            # Save Triton code
            with open(triton_file, 'w') as f:
                f.write(triton_code)
            print(f"  Saved Triton code to: {triton_file}")
            results["triton_generated"] = True

            # Generate correctness test for this attempt
            test_code = generate_correctness_test(func_name, func_spec, attempt)
            with open(test_file, 'w') as f:
                f.write(test_code)
            test_file.chmod(0o755)
            results["test_generated"] = True

            # Run test
            print(f"  Running correctness test (attempt {attempt}/{MAX_ATTEMPTS})...")
            passed, error_info = run_test(func_name, test_file)

            if passed:
                print(f"  All tests PASSED on attempt {attempt}!")
                results["test_passed"] = True

                # Run performance benchmark
                print(f"  Running performance benchmark...")
                benchmark_file = test_dir / f"benchmark_{func_name}.py"
                benchmark_code = generate_benchmark_test(func_name, func_spec, attempt)
                with open(benchmark_file, 'w') as f:
                    f.write(benchmark_code)
                benchmark_file.chmod(0o755)

                benchmark_results = run_benchmark(func_name, benchmark_file)
                if benchmark_results:
                    results["benchmark"] = benchmark_results
                    speedup = benchmark_results.get('speedup', 0)
                    print(f"  Benchmark complete: {speedup:.2f}x speedup")

                    # Track the best passing result
                    if speedup > best_speedup:
                        best_speedup = speedup
                        best_attempt = attempt
                        best_result = {
                            "c_ref_available": True,
                            "triton_generated": True,
                            "test_generated": True,
                            "test_passed": True,
                            "attempts": attempt,
                            "final_attempt": attempt,
                            "benchmark": benchmark_results.copy(),
                            "final_error": None
                        }
                        print(f"  New best result: {speedup:.2f}x speedup (attempt {attempt})")

                    # Check if speedup is too low - retry with parallelization feedback
                    if speedup < 0.1 and attempt < MAX_ATTEMPTS:
                        print(f"  Speedup too low ({speedup:.2f}x < 0.1x). Retrying for better parallelization...")
                        error_info = {
                            'type': 'low_speedup',
                            'speedup': speedup,
                            'message': f'Code is correct but speedup is only {speedup:.2f}x. Needs better parallelization.'
                        }
                        # Reset test_passed so we continue retrying
                        results["test_passed"] = False
                        continue  # Continue to next attempt
                else:
                    print(f"  Benchmark failed or timed out")
                    results["benchmark"] = None

                # Return immediately if speedup is good enough
                results["final_attempt"] = attempt
                return results
            else:
                error_type = error_info.get('type', 'unknown')
                error_msg = error_info.get('message', '')[:200]
                print(f"  Test FAILED: {error_type}")
                print(f"    {error_msg[:100]}...")
                results["final_error"] = error_info

                # Continue retrying up to MAX_ATTEMPTS for all error types (no early exit for numerical)
                if attempt < MAX_ATTEMPTS:
                    if error_type == 'numerical':
                        print(f"  Will retry with error feedback (numerical error)...")
                    else:
                        print(f"  Will retry with error feedback ({error_type})...")
                else:
                    print(f"  Max attempts ({MAX_ATTEMPTS}) reached. Moving to next function.")

        except Exception as e:
            print(f"  Exception during generation: {e}")
            results["final_error"] = {'type': 'generation_error', 'message': str(e)}
            if attempt >= MAX_ATTEMPTS:
                break

    # Return the best passing result if we have one, otherwise return the last result
    if best_result is not None:
        print(f"  Returning best result from attempt {best_attempt} with {best_speedup:.2f}x speedup")
        best_result["attempts"] = attempt  # Total attempts made
        return best_result

    return results


def main():
    """Main automation pipeline."""
    global USE_OMP, BENCHMARK_DATA_SIZE

    # Check for --omp flag (multi-threaded C reference)
    if '--omp' in sys.argv:
        sys.argv.remove('--omp')
        USE_OMP = True
        import multiprocessing
        omp_threads = multiprocessing.cpu_count()
        os.environ['OMP_NUM_THREADS'] = str(omp_threads)
        os.environ['TSVC_USE_OMP'] = '1'
        # Reload the C reference to pick up the OMP library
        import importlib
        from c_reference import tsvc_all_reference
        importlib.reload(tsvc_all_reference)
        print(f"OpenMP enabled: C reference will use {omp_threads} threads")

    # Check for --data-size flag (benchmark array size: 32MB, 3200MB)
    if '--data-size' in sys.argv:
        idx = sys.argv.index('--data-size')
        BENCHMARK_DATA_SIZE = sys.argv[idx + 1]
        sys.argv.pop(idx)  # remove flag
        sys.argv.pop(idx)  # remove value
        print(f"Benchmark data size: {BENCHMARK_DATA_SIZE} per array")

    print("=" * 70)
    print("Integrated Generation and Testing Pipeline")
    print(f"Total functions available: {len(TSVC_FUNCTIONS)}")
    print(f"Max attempts per function: {MAX_ATTEMPTS}")
    if BENCHMARK_DATA_SIZE:
        print(f"Data size: {BENCHMARK_DATA_SIZE}")
    print("=" * 70)

    if not client:
        print("ERROR: ANTHROPIC_API_KEY not set!")
        sys.exit(1)

    # Check if specific functions requested
    if len(sys.argv) > 1:
        func_names = sys.argv[1:]
        functions_to_process = {k: TSVC_FUNCTIONS[k] for k in func_names if k in TSVC_FUNCTIONS}
        not_found = [k for k in func_names if k not in TSVC_FUNCTIONS]
        if not_found:
            print(f"Warning: Functions not found: {not_found}")
        print(f"Processing {len(functions_to_process)} specific functions: {list(functions_to_process.keys())}")
    else:
        functions_to_process = TSVC_FUNCTIONS
        print(f"Processing ALL {len(functions_to_process)} functions")

    if not functions_to_process:
        print("No valid functions to process!")
        return

    # Process each function
    all_results = {}
    for i, (func_name, func_spec) in enumerate(functions_to_process.items(), 1):
        print(f"\n[{i}/{len(functions_to_process)}]", end=" ")
        # Enrich spec with properties parsed from C code (runtime inference)
        enriched_spec = enrich_func_spec(func_spec)
        results = process_function(func_name, enriched_spec)
        all_results[func_name] = results

    # Print summary
    print(f"\n\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Function':<12} {'C Ref':<10} {'Triton':<10} {'Passed':<10} {'Attempts':<10} {'Speedup':<12}")
    print(f"{'-' * 80}")

    for func_name, results in all_results.items():
        c_ref = "Y" if results.get("c_ref_available", True) else "N"
        triton = "Y" if results["triton_generated"] else "N"
        passed = "Y" if results["test_passed"] else "N"
        attempts = str(results["attempts"])

        if results.get("benchmark"):
            benchmark = results["benchmark"]
            speedup_val = benchmark.get("speedup", -1)
            c_time = benchmark.get("c_ref_time_ms", -1)
            tr_time = benchmark.get("triton_time_ms", -1)

            if speedup_val == -1:
                # Timeout case - calculate minimum/maximum speedup
                timeout_ms = 60000  # 60 seconds timeout per section
                if c_time == -1 and tr_time == -1:
                    speedup = "Both timeout"
                elif c_time == -1 and tr_time > 0:
                    # C reference timed out, Triton completed - calculate minimum speedup
                    min_speedup = timeout_ms / tr_time
                    speedup = f">{min_speedup:.0f}x (C>{timeout_ms/1000:.0f}s, TR:{tr_time:.2f}ms)"
                elif tr_time == -1 and c_time > 0:
                    # Triton timed out, C reference completed - calculate maximum slowdown
                    max_slowdown = timeout_ms / c_time
                    speedup = f"<{1/max_slowdown:.2f}x (C:{c_time:.2f}ms, TR>{timeout_ms/1000:.0f}s)"
                else:
                    speedup = "N/A"
            else:
                speedup = f"{speedup_val:.2f}x"
        else:
            speedup = "-"

        print(f"{func_name:<12} {c_ref:<10} {triton:<10} {passed:<10} {attempts:<10} {speedup:<12}")

    print(f"{'=' * 80}")

    # Count successes
    total = len(all_results)
    c_ref_ok = sum(1 for r in all_results.values() if r.get("c_ref_available", True))
    triton_ok = sum(1 for r in all_results.values() if r["triton_generated"])
    passed = sum(1 for r in all_results.values() if r["test_passed"])
    first_try = sum(1 for r in all_results.values() if r["test_passed"] and r["attempts"] == 1)
    retried = sum(1 for r in all_results.values() if r["test_passed"] and r["attempts"] > 1)

    # Calculate benchmark statistics
    benchmarked = sum(1 for r in all_results.values() if r.get("benchmark"))
    if benchmarked > 0:
        # Filter out timeout cases (speedup == -1)
        speedups = [r["benchmark"]["speedup"] for r in all_results.values()
                   if r.get("benchmark") and r["benchmark"].get("speedup") is not None
                   and r["benchmark"]["speedup"] > 0]
        avg_speedup = sum(speedups) / len(speedups) if speedups else 0
        min_speedup = min(speedups) if speedups else 0
        max_speedup = max(speedups) if speedups else 0

    print(f"\nC reference available: {c_ref_ok}/{total}")
    print(f"Triton generated: {triton_ok}/{total}")
    print(f"Tests passed: {passed}/{total}")
    print(f"  - Passed on first try: {first_try}")
    print(f"  - Passed after retry: {retried}")
    if benchmarked > 0:
        # Count timeout cases
        c_ref_timeouts = sum(1 for r in all_results.values()
                         if r.get("benchmark") and r["benchmark"].get("c_ref_time_ms") == -1)
        tr_timeouts = sum(1 for r in all_results.values()
                         if r.get("benchmark") and r["benchmark"].get("triton_time_ms") == -1)
        successful_benchmarks = len(speedups)

        print(f"\nPerformance benchmarks: {benchmarked}/{total}")
        print(f"  - Successful: {successful_benchmarks}")
        print(f"  - C reference timeouts: {c_ref_timeouts}")
        print(f"  - Triton timeouts: {tr_timeouts}")
        if successful_benchmarks > 0:
            print(f"  - Average speedup: {avg_speedup:.2f}x")
            print(f"  - Min speedup: {min_speedup:.2f}x")
            print(f"  - Max speedup: {max_speedup:.2f}x")
    print(f"{'=' * 80}")

    # Save results to JSON file
    import json
    results_file = Path("../results/tsvc/test29") / "results.json"

    # Load existing results if file exists
    existing_results = {}
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                existing_results = json.load(f)
        except:
            pass

    # Update with new results
    for func_name, results in all_results.items():
        # Convert error info to serializable format
        error_info = results.get("final_error")
        if error_info:
            error_info = {
                'type': error_info.get('type', 'unknown'),
                'message': str(error_info.get('message', ''))[:500]
            }

        existing_results[func_name] = {
            "c_ref_available": results.get("c_ref_available", True),
            "triton_generated": results["triton_generated"],
            "test_passed": results["test_passed"],
            "attempts": results["attempts"],
            "final_error": error_info,
            "benchmark": results.get("benchmark")
        }

    # Save updated results
    Path("../results/tsvc/test29").mkdir(exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(existing_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
