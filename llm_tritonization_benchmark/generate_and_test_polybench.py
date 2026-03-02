#!/usr/bin/env python3
"""
Integrated Generation and Testing Pipeline for Polybench/C Kernels

Analogous to generate_and_test.py but for Polybench/C 4.2.1 (30 kernels).
Uses the same LLM retry strategy and correctness testing approach.

Usage:
    python generate_and_test_polybench.py              # Process all kernels
    python generate_and_test_polybench.py gemm lu atax  # Process specific kernels
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

# Import Polybench function database
sys.path.append(str(Path(__file__).parent / "utilities"))
from polybench_functions_db import POLYBENCH_FUNCTIONS

# Add PET analysis directory to path
sys.path.insert(0, "/home/qinxiao/workspace/pet/isl_analysis")

# Import extraction info (for array sizes)
from extract_polybench_kernels import POLYBENCH_KERNELS

# Import analysis modules (same as generate_and_test.py)
try:
    from compute_war_dependences import analyze_kernel_war
    HAS_WAR_ANALYSIS = True
except ImportError:
    HAS_WAR_ANALYSIS = False
    analyze_kernel_war = None

try:
    from compute_statement_overwrites import analyze_kernel_overwrites, format_overwrite_for_prompt
    HAS_OVERWRITE_ANALYSIS = True
except ImportError:
    HAS_OVERWRITE_ANALYSIS = False

try:
    from compute_stream_compaction import analyze_kernel_stream_compaction
    HAS_STREAM_ANALYSIS = True
except ImportError:
    HAS_STREAM_ANALYSIS = False

try:
    from compute_pointer_aliasing import analyze_kernel_aliasing, format_aliasing_for_prompt
    HAS_ALIASING_ANALYSIS = True
except ImportError:
    HAS_ALIASING_ANALYSIS = False

try:
    from compute_parallel_dims import analyze_kernel_parallelization
    HAS_PARDIMS_ANALYSIS = True
except ImportError:
    HAS_PARDIMS_ANALYSIS = False

try:
    from compute_scalar_expansion import analyze_kernel_scalar_expansion, format_scalar_expansion_for_prompt
    HAS_SCALAR_EXPANSION = True
except ImportError:
    HAS_SCALAR_EXPANSION = False

try:
    from compute_reduction_type import analyze_kernel_reduction, build_reduction_instructions
    HAS_REDUCTION = True
except ImportError:
    HAS_REDUCTION = False

try:
    from compute_gpu_parallelization_strategy import analyze_kernel_gpu_strategy, build_gpu_strategy_instructions
    HAS_GPU_STRATEGY = True
except ImportError:
    HAS_GPU_STRATEGY = False

# LLVM fallback adapters
try:
    from llvm_fallback_adapters import (
        llvm_war_fallback, llvm_overwrite_fallback,
        llvm_stream_compaction_fallback, llvm_parallel_dims_fallback,
        llvm_scalar_expansion_fallback, try_with_llvm_fallback,
        enhance_war_with_llvm_vectors
    )
    HAS_LLVM_FALLBACK = True
except ImportError:
    HAS_LLVM_FALLBACK = False

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=API_KEY) if API_KEY else None

POLYBENCH_KERNELS_DIR = "/home/qinxiao/workspace/pet/isl_analysis/kernels_polybench"
MAX_ATTEMPTS = 10
OUTPUT_DIR = "polybench_results"
ENABLE_ANALYSIS = True

# Kernels requiring higher tolerance due to long sequential dependency chains.
# These are algorithmically correct but FP32 rounding diverges between CPU and GPU
# over many sequential steps. Format: {kernel_name: {'atol': ..., 'rtol': ...}}
HIGHER_TOLERANCE_KERNELS = {
    # durbin: 120-step Levinson-Durbin recurrence, alpha/beta chain across all iterations
    'durbin': {'atol': 0.05, 'rtol': 0.01},
    # gramschmidt: M=60 rows, N=80 cols -> rank-deficient after col 60.
    # R[k][k] becomes ~1e-6 for k>=60, so Q[:,k] = A[:,k]/R[k][k] amplifies
    # tiny FP32 CPU/GPU differences by ~1e6. A and R are correct to 1e-5.
    'gramschmidt': {'atol': 1.0, 'rtol': 2.0},
    # lu: Pivotless LU on 120x120 diag-dominant matrix. 120 sequential row updates,
    # each with inner products of length up to 120. FP32 error compounds to ~2-4 absolute.
    'lu': {'atol': 5.0, 'rtol': 0.05},
    # ludcmp: Pivotless LU decomposition on diag-dominant matrices.
    # A (LU factors) and y (forward-sub) are marked 'temp'. Only x (solution) is checked.
    # x error varies with matrix conditioning: 1e-4 to 0.03 across seeds.
    'ludcmp': {'atol': 0.05, 'rtol': 0.02},
}


# ============================================================================
# Analysis loading with LLVM fallback
# ============================================================================

def load_war_analysis(kernel_name: str) -> Optional[dict]:
    """Load WAR analysis with LLVM fallback and direction vector enhancement."""
    kernel_file = os.path.join(POLYBENCH_KERNELS_DIR, f"{kernel_name}.c")
    if not os.path.exists(kernel_file):
        return None

    pet_result = None
    if HAS_WAR_ANALYSIS and analyze_kernel_war:
        try:
            pet_result = analyze_kernel_war(kernel_file)
        except Exception:
            pass

    # Enhance PET result with LLVM direction vectors for loop-level scoping
    if pet_result and not pet_result.get('parallelization_safe', True) and HAS_LLVM_FALLBACK:
        try:
            enhanced = enhance_war_with_llvm_vectors(kernel_file, pet_result)
            if enhanced:
                return enhanced
        except Exception:
            pass

    if pet_result is not None:
        return pet_result

    # Full LLVM fallback if PET failed entirely
    if HAS_LLVM_FALLBACK:
        try:
            return llvm_war_fallback(kernel_file)
        except Exception:
            pass
    return None


def load_parallelization_analysis(kernel_name: str) -> Optional[dict]:
    """Load parallelization analysis with LLVM fallback."""
    kernel_file = os.path.join(POLYBENCH_KERNELS_DIR, f"{kernel_name}.c")
    if HAS_PARDIMS_ANALYSIS and analyze_kernel_parallelization:
        try:
            result = analyze_kernel_parallelization(kernel_name, kernel_file=kernel_file)
            if result is not None:
                return result
        except Exception:
            pass

    if HAS_LLVM_FALLBACK:
        try:
            return llvm_parallel_dims_fallback(kernel_file)
        except Exception:
            pass
    return None


def load_scalar_expansion_analysis(kernel_name: str) -> Optional[dict]:
    """Load scalar expansion analysis with LLVM fallback."""
    kernel_file = os.path.join(POLYBENCH_KERNELS_DIR, f"{kernel_name}.c")

    if HAS_SCALAR_EXPANSION and analyze_kernel_scalar_expansion:
        try:
            result = analyze_kernel_scalar_expansion(kernel_file)
            if result is not None:
                return result
        except Exception:
            pass

    if HAS_LLVM_FALLBACK:
        try:
            return llvm_scalar_expansion_fallback(kernel_file)
        except Exception:
            pass
    return None


def load_reduction_analysis(kernel_name: str) -> Optional[dict]:
    """Load reduction analysis."""
    if not HAS_REDUCTION or analyze_kernel_reduction is None:
        return None
    kernel_file = os.path.join(POLYBENCH_KERNELS_DIR, f"{kernel_name}.c")
    try:
        return analyze_kernel_reduction(kernel_name, kernel_file=kernel_file)
    except Exception:
        return None


# ============================================================================
# Polybench-specific prompt and code extraction
# ============================================================================

def get_kernel_source(kernel_name: str) -> Optional[str]:
    """Read the standalone kernel .c file."""
    kernel_file = os.path.join(POLYBENCH_KERNELS_DIR, f"{kernel_name}.c")
    if not os.path.exists(kernel_file):
        return None
    with open(kernel_file, 'r') as f:
        return f.read()


def get_kernel_params(kernel_name: str) -> dict:
    """Get the size parameters for a kernel from the extraction database."""
    # Map underscore names back to original
    for orig_name, info in POLYBENCH_KERNELS.items():
        if orig_name.replace("-", "_") == kernel_name:
            return info["params"]
    return {}


def build_polybench_prompt(kernel_name: str, func_spec: dict) -> str:
    """Build the prompt for Polybench kernel Triton generation."""
    source = get_kernel_source(kernel_name)
    if not source:
        raise ValueError(f"Could not read kernel source for {kernel_name}")

    params = get_kernel_params(kernel_name)
    arrays = func_spec.get('arrays', {})
    scalar_params = func_spec.get('scalar_params', {})
    loop_code = func_spec.get('loop_code', '')
    has_2d = func_spec.get('has_2d_arrays', False)
    has_3d = func_spec.get('has_3d_arrays', False)

    # Build array info section
    array_lines = []
    for arr_name, mode in sorted(arrays.items()):
        mode_str = {'r': 'read-only', 'w': 'write-only', 'rw': 'read-write', 'temp': 'temporary scratch (read-write, not checked for correctness)'}[mode]
        array_lines.append(f"- `{arr_name}`: {mode_str}")
    array_info = "\n".join(array_lines)

    # Build dimension info
    dim_lines = []
    for param_name, param_value in sorted(params.items()):
        dim_lines.append(f"- `{param_name}` = {param_value}")
    dim_info = "\n".join(dim_lines)

    # Build function signature
    sig_parts = []
    for arr_name in sorted(arrays.keys()):
        sig_parts.append(arr_name)
    for sp in sorted(scalar_params.keys()):
        sig_parts.append(sp)
    # Add dimension parameters
    for p in sorted(params.keys()):
        if p not in scalar_params:
            sig_parts.append(p)
    exact_sig = ", ".join(sig_parts)

    # Build valid Python identifier for function names
    func_id = kernel_name
    if func_id[0].isdigit():
        func_id = "k" + func_id

    # Load analysis results (skip if analysis disabled)
    analysis_sections = []

    if ENABLE_ANALYSIS:
        war_result = load_war_analysis(kernel_name)
        par_result = load_parallelization_analysis(kernel_name)
        scalar_exp_result = load_scalar_expansion_analysis(kernel_name)
        reduction_result = load_reduction_analysis(kernel_name)

        if war_result and not war_result.get('parallelization_safe', True):
            copies = war_result.get('arrays_needing_copy', [])
            deps = war_result.get('war_dependencies', [])
            loop_scoping = war_result.get('loop_level_scoping')

            section = "\n## WAR (Write-After-Read) Dependencies\n\n"
            section += "**Note**: This kernel has WAR (Write-After-Read) dependencies.\n"
            section += "If you split the computation into **separate Triton kernels** launched sequentially, "
            section += "kernel launch barriers handle these dependencies naturally — no cloning needed.\n"
            section += "Cloning is only needed if reads and writes to the same array happen **within a single kernel**.\n"

            if loop_scoping:
                # Enhanced format with loop-level scoping
                loop_vars = war_result.get('loop_vars', [])
                section += f"\n**Loop variables** (outer to inner): {', '.join(loop_vars)}\n"
                for arr in copies:
                    scoping = loop_scoping.get(arr, {})
                    carried = scoping.get('carried_by_loops', loop_vars)
                    safe = scoping.get('safe_to_parallelize_loops', [])
                    section += f"\n**Array `{arr}`**: WAR carried by loop(s) `{', '.join(carried)}`\n"
                    seq_ctx = scoping.get('sequential_context_loops', [])
                    for var in loop_vars:
                        if var in safe:
                            section += f"- Parallelizing `{var}`: SAFE (no copy needed for `{arr}`)\n"
                        elif var in carried:
                            section += f"- Parallelizing `{var}`: REQUIRES `{arr}_copy = {arr}.clone()`\n"
                        elif var in seq_ctx:
                            section += f"- Loop `{var}`: sequential context (not analyzed for WAR)\n"
            else:
                # Original format without scoping
                if copies:
                    section += f"\n**Arrays with WAR dependencies**: {', '.join(copies)}\n"
                for dep in deps[:5]:
                    section += f"- {dep.get('description', '')}\n"
                if copies:
                    section += "\n**If using a single kernel**: Create read-only copies before the parallel region:\n"
                    section += "```python\n"
                    for arr in copies:
                        section += f"{arr}_copy = {arr}.clone()  # Read from copy, write to original\n"
                    section += "```\n"
                    section += "**If using separate kernels**: No cloning needed — launch one kernel per phase.\n"

            analysis_sections.append(section)

        if par_result and par_result.get('options'):
            valid_opts = [o for o in par_result['options'] if o['valid']]
            # Only include parallelization section if at least one option is valid
            if valid_opts:
                section = "\n## Parallelization Analysis\n\n"
                section += f"**Loop dimensions**: {par_result.get('dims', [])}\n"
                if par_result.get('is_triangular'):
                    tri = par_result['triangular_info']
                    section += f"**Triangular bounds**: {tri.get('smaller', '?')} < {tri.get('larger', '?')}\n"
                for opt in par_result['options']:
                    valid = "VALID" if opt['valid'] else "INVALID"
                    section += f"\n- Parallelize `{opt['parallel_dim']}`, sequential `{opt['sequential_dim']}`: {valid}\n"
                    for issue in opt.get('issues', []):
                        section += f"  - {issue}\n"
                # When both dims are freely parallelizable, recommend 2D grid
                if len(valid_opts) >= 2:
                    d = par_result.get('dims', [])
                    section += f"\n**Both `{d[0]}` and `{d[1]}` are freely parallelizable.** "
                    section += "Use a 2D grid to parallelize both simultaneously for best GPU occupancy.\n"
                analysis_sections.append(section)

        # Cross-reference WAR scoping with parallelization options
        if (war_result and war_result.get('loop_level_scoping')
                and par_result and par_result.get('options')):
            loop_scoping = war_result['loop_level_scoping']
            copies = war_result.get('arrays_needing_copy', [])
            par_dims = par_result.get('dims', [])

            recommendations = []
            for opt in par_result.get('options', []):
                if not opt.get('valid'):
                    continue
                pdim = opt['parallel_dim']
                # Handle multi-dim parallel dims (e.g., "i, j, k" from N-D analysis)
                pdim_list = [d.strip() for d in pdim.split(',')]
                # Check if ALL WAR arrays are safe at ALL parallel dimensions
                all_safe = True
                needs_copy_arrs = []
                for arr in copies:
                    scoping = loop_scoping.get(arr, {})
                    safe = scoping.get('safe_to_parallelize_loops', [])
                    # All parallel dims must be safe for this array
                    if not all(d in safe for d in pdim_list):
                        all_safe = False
                        needs_copy_arrs.append(arr)

                if all_safe:
                    recommendations.append(
                        f"- **RECOMMENDED**: Parallelize `{pdim}` — no WAR copies needed for any array"
                    )
                elif needs_copy_arrs:
                    recommendations.append(
                        f"- Parallelize `{pdim}`: must clone {', '.join(f'`{a}`' for a in needs_copy_arrs)} before parallel region"
                    )

            if recommendations:
                section = "\n## WAR + Parallelization Recommendation\n\n"
                section += "\n".join(recommendations)
                section += "\n"
                analysis_sections.append(section)

        if scalar_exp_result and scalar_exp_result.get('has_scalar_expansion'):
            if HAS_SCALAR_EXPANSION and format_scalar_expansion_for_prompt:
                try:
                    formatted = format_scalar_expansion_for_prompt(kernel_name, scalar_exp_result)
                    if formatted:
                        analysis_sections.append(f"\n{formatted}\n")
                except Exception:
                    pass

        if reduction_result and reduction_result.get('is_reduction'):
            if HAS_REDUCTION and build_reduction_instructions:
                try:
                    formatted = build_reduction_instructions(reduction_result)
                    if formatted:
                        analysis_sections.append(f"\n{formatted}\n")
                except Exception:
                    pass

        # GPU parallelization strategy (wavefront, inner-loop vectorization, multi-GEMM)
        if HAS_GPU_STRATEGY:
            try:
                kernel_file = os.path.join(POLYBENCH_KERNELS_DIR, f"{kernel_name}.c")
                gpu_strategy = analyze_kernel_gpu_strategy(kernel_name, kernel_file)
                if gpu_strategy:
                    formatted = build_gpu_strategy_instructions(kernel_name, gpu_strategy)
                    if formatted:
                        analysis_sections.append(f"\n{formatted}\n")
            except Exception:
                pass

    analysis_text = "\n".join(analysis_sections)

    prompt = f"""I have a Polybench/C kernel that I want to implement in Triton for GPU acceleration.

## Original C Code:
```c
{source}
```

## Kernel Loop to Implement:
```c
{loop_code}
```
{analysis_text}

## Array Information:
{array_info}

## Dimension Parameters (compile-time constants in C, runtime parameters in Triton):
{dim_info}

## Requirements:
Please generate a complete Triton implementation that:
1. Includes a @triton.jit kernel function named `{func_id}_kernel`
2. Includes a Python wrapper function named `{func_id}_triton`
3. The wrapper accepts tensor arrays, scalar parameters, and dimension parameters
4. Uses appropriate block sizes and memory access patterns
5. Handles edge cases with masking
6. Is functionally equivalent to the C code (same computation, same results)
7. For 2D arrays, compute linear index as `row * stride + col`
8. For 3D arrays, compute linear index as `dim0 * (dim1_size * dim2_size) + dim1 * dim2_size + dim2`

## REQUIRED function signature (use EXACTLY these parameter names):
```python
def {func_id}_triton({exact_sig}):
    ...  # kernel computation
```

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
- Use Python operators: `a * b`, `a / b`, `a + b` (not `tl.mul`, `tl.div`, `tl.add`)
- Use `triton.cdiv()` in wrapper only

**NEVER use Python lists, break/continue inside @triton.jit kernels**
**Pass tensors directly to kernels, NOT data_ptr()**
**NEVER use chained comparisons (use separate comparisons with &)**

Provide ONLY the Python code, no additional explanation."""

    return prompt


# ============================================================================
# Test generation for Polybench
# ============================================================================

def generate_correctness_test(kernel_name: str, func_spec: dict, attempt: int = 1) -> str:
    """Generate correctness test for a Polybench kernel."""
    arrays = func_spec['arrays']
    scalar_params = func_spec.get('scalar_params', {})
    params = get_kernel_params(kernel_name)
    has_2d = func_spec.get('has_2d_arrays', False)
    has_3d = func_spec.get('has_3d_arrays', False)

    # Per-kernel tolerance override
    tol = HIGHER_TOLERANCE_KERNELS.get(kernel_name, {'atol': 1e-3, 'rtol': 1e-4})
    atol = tol['atol']
    rtol = tol['rtol']

    # Build array initialization
    array_inits = []
    domain_inits = _get_domain_array_inits(kernel_name, arrays, params, "            ")
    if domain_inits is not None:
        array_inits.extend(domain_inits)
    else:
        for arr_name, mode in sorted(arrays.items()):
            if mode in ['r', 'rw', 'w', 'temp']:
                # Determine array shape from the kernel source
                shape = _get_array_shape(kernel_name, arr_name, params)
                if shape:
                    shape_str = ", ".join(str(s) for s in shape)
                    array_inits.append(
                        f"            {arr_name} = torch.randn({shape_str}, device='cuda', dtype=torch.float32)"
                    )
                else:
                    # Fallback: 1D with first param value
                    first_size = list(params.values())[0] if params else 100
                    array_inits.append(
                        f"            {arr_name} = torch.randn({first_size}, device='cuda', dtype=torch.float32)"
                    )

    # Scalar parameter initialization
    for sp_name in sorted(scalar_params.keys()):
        if sp_name in ('alpha', 'beta'):
            array_inits.append(f"            {sp_name} = 1.5")
        elif sp_name in ('float_n',):
            n_val = params.get('N', 100)
            array_inits.append(f"            {sp_name} = float({n_val})")
        elif sp_name == 'eps':
            array_inits.append(f"            {sp_name} = 0.1")
        else:
            array_inits.append(f"            {sp_name} = 1.0")

    # Dimension parameters
    for p_name, p_val in sorted(params.items()):
        if p_name not in scalar_params:
            array_inits.append(f"            {p_name} = {p_val}")

    array_init_str = "\n".join(array_inits)

    # Build argument lists
    array_names = sorted([a for a, m in arrays.items() if m in ['r', 'rw', 'w', 'temp']])
    output_arrays = sorted([a for a, m in arrays.items() if m in ['rw', 'w']])  # exclude 'temp'
    scalar_names = sorted(scalar_params.keys())
    dim_names = sorted([p for p in params.keys() if p not in scalar_params])

    all_args = array_names + scalar_names + dim_names
    args_str = ", ".join(all_args)

    # C reference clones
    c_ref_clones = [f"            {a}_c = {a}.cpu().numpy().copy()" for a in array_names]
    c_ref_clone_str = "\n".join(c_ref_clones)

    # Triton clones
    triton_clones = [f"            {a}_tr = {a}.clone()" for a in array_names]
    triton_clone_str = "\n".join(triton_clones)

    # C reference call args (numpy arrays + scalars + dims)
    c_args = [f"{a}_c" for a in array_names] + scalar_names + dim_names
    c_call_str = ", ".join(c_args)

    # Triton call args (torch tensors + scalars + dims)
    tr_args = [f"{a}_tr" for a in array_names] + scalar_names + dim_names
    tr_call_str = ", ".join(tr_args)

    # Checksum: sum of output arrays
    c_checksums = [f"float(np.sum({a}_c))" for a in output_arrays]
    tr_checksums = [f"float(torch.sum({a}_tr).item())" for a in output_arrays]

    c_checksum_expr = " + ".join(c_checksums) if c_checksums else "0.0"
    tr_checksum_expr = " + ".join(tr_checksums) if tr_checksums else "0.0"

    # Build function name for C reference
    func_id = kernel_name
    if func_id[0].isdigit():
        func_id = "k" + func_id

    # For digit-starting names, use importlib
    llm_subdir = "llm_triton" if ENABLE_ANALYSIS else "llm_triton_no_analysis"
    if kernel_name[0].isdigit():
        import_block = (
            f"import importlib\n"
            f"    _mod = importlib.import_module(\"{OUTPUT_DIR}.{llm_subdir}.{kernel_name}.attempt{attempt}\")\n"
            f"    {func_id}_triton = _mod.{func_id}_triton"
        )
    else:
        import_block = f"from {OUTPUT_DIR}.{llm_subdir}.{kernel_name}.attempt{attempt} import {func_id}_triton"

    test_code = f'''#!/usr/bin/env python3
"""Correctness test for {kernel_name} (Polybench) - attempt {attempt}"""
import sys
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

# Import Triton implementation
try:
    {import_block}
except ImportError as e:
    print(f"Import error: {{e}}")
    sys.exit(1)

# Load C reference
C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs" / "lib{kernel_name}.so"
if not C_LIB_PATH.exists():
    print(f"C reference library not found: {{C_LIB_PATH}}")
    sys.exit(1)

def run_c_reference({c_call_str}):
    """Run C reference kernel via ctypes."""
    lib = ctypes.CDLL(str(C_LIB_PATH))

    # Set global arrays in the .so
{_gen_ctypes_array_setup(kernel_name, arrays, params)}

    # Set global scalars
{_gen_ctypes_scalar_setup(kernel_name, scalar_params, params)}

    # Run kernel
    func = getattr(lib, "{func_id}_kernel")
    func.argtypes = []
    func.restype = None
    func()

    # Read back output arrays
{_gen_ctypes_array_readback(kernel_name, arrays, params)}

def test_correctness():
    """Test Triton vs C reference."""
    num_tests = 3
    all_passed = True

    for test_idx in range(num_tests):
        try:
            # Initialize arrays
{array_init_str}

            # Clone for C reference
{c_ref_clone_str}

            # Clone for Triton
{triton_clone_str}

            # Run C reference
            run_c_reference({c_call_str})

            # Run Triton
            {func_id}_triton({tr_call_str})

            # Compare output arrays
            max_error = 0.0
{_gen_comparison_code(output_arrays)}

            # Pass if absolute error < atol OR relative error < rtol
            passed = (max_error < {atol}) or (max_rel_error < {rtol})
            if passed:
                print(f"  Test {{test_idx + 1}}: PASS (abs={{max_error:.6e}} rel={{max_rel_error:.6e}})")
            else:
                print(f"  Test {{test_idx + 1}}: FAIL (abs={{max_error:.6e}} rel={{max_rel_error:.6e}})")
                all_passed = False

        except Exception as e:
            print(f"  Test {{test_idx + 1}}: ERROR - {{e}}")
            all_passed = False

    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    return all_passed

if __name__ == "__main__":
    test_correctness()
'''
    return test_code


def _get_array_shape(kernel_name: str, arr_name: str, params: dict) -> Optional[list]:
    """Determine array shape from the kernel source."""
    kernel_file = os.path.join(POLYBENCH_KERNELS_DIR, f"{kernel_name}.c")
    if not os.path.exists(kernel_file):
        return None

    with open(kernel_file, 'r') as f:
        source = f.read()

    # Match: float arr_name[DIM1][DIM2][DIM3]; or float arr_name[DIM1][DIM2]; or float arr_name[DIM1];
    pattern = rf'(?:float|double|int)\s+{re.escape(arr_name)}\s*(\[[^\]]+\](?:\[[^\]]+\])*)\s*;'
    m = re.search(pattern, source)
    if not m:
        return None

    dims_str = m.group(1)
    dims = re.findall(r'\[(\w+)\]', dims_str)

    # Resolve dimension names to values
    shape = []
    for d in dims:
        if d in params:
            shape.append(params[d])
        elif d.isdigit():
            shape.append(int(d))
        else:
            # Try to find #define
            define_match = re.search(rf'#define\s+{d}\s+(\d+)', source)
            if define_match:
                shape.append(int(define_match.group(1)))
            else:
                shape.append(100)  # fallback
    return shape


def _get_array_c_type(kernel_name: str, arr_name: str) -> str:
    """Detect the C type of an array from the kernel source file.

    Returns 'float', 'double', 'int', or 'char'. Defaults to 'float'.
    """
    kernel_file = os.path.join(POLYBENCH_KERNELS_DIR, f"{kernel_name}.c")
    if not os.path.exists(kernel_file):
        return 'float'

    with open(kernel_file, 'r') as f:
        source = f.read()

    pattern = rf'(float|double|int|char|short|long)\s+{re.escape(arr_name)}\s*\['
    m = re.search(pattern, source)
    return m.group(1) if m else 'float'


# C type -> (ctypes type name, numpy dtype string)
_C_TYPE_MAP = {
    'float': ('c_float', 'float32'),
    'double': ('c_double', 'float64'),
    'int': ('c_int', 'int32'),
    'char': ('c_char', 'int8'),
    'short': ('c_short', 'int16'),
    'long': ('c_long', 'int64'),
}


def _get_domain_array_inits(kernel_name: str, arrays: dict, params: dict, indent: str) -> Optional[list]:
    """Return domain-appropriate array init lines for kernels with mathematical preconditions.

    Returns None if default torch.randn is appropriate for all arrays.
    Returns a list of init lines (already indented) covering ALL arrays for the kernel.
    """
    if kernel_name == 'cholesky':
        N = params.get('N', 120)
        return [
            f"{indent}# SPD matrix: A = R^T R + N*I",
            f"{indent}_R = torch.randn({N}, {N}, device='cuda', dtype=torch.float32)",
            f"{indent}A = _R.T @ _R + {N} * torch.eye({N}, device='cuda', dtype=torch.float32)",
        ]

    if kernel_name == 'lu':
        N = params.get('N', 120)
        return [
            f"{indent}# Diagonally dominant for stable pivotless LU",
            f"{indent}A = torch.randn({N}, {N}, device='cuda', dtype=torch.float32) + {N} * torch.eye({N}, device='cuda', dtype=torch.float32)",
        ]

    if kernel_name == 'ludcmp':
        N = params.get('N', 120)
        return [
            f"{indent}# Diagonally dominant for stable pivotless LU; x,y are outputs",
            f"{indent}A = torch.randn({N}, {N}, device='cuda', dtype=torch.float32) + {N} * torch.eye({N}, device='cuda', dtype=torch.float32)",
            f"{indent}b = torch.randn({N}, device='cuda', dtype=torch.float32)",
            f"{indent}x = torch.zeros({N}, device='cuda', dtype=torch.float32)",
            f"{indent}y = torch.zeros({N}, device='cuda', dtype=torch.float32)",
        ]

    if kernel_name == 'trisolv':
        N = params.get('N', 120)
        return [
            f"{indent}# Lower triangular with |diagonal| >= 1",
            f"{indent}L = torch.tril(torch.randn({N}, {N}, device='cuda', dtype=torch.float32))",
            f"{indent}L.diagonal().abs_().clamp_(min=1.0)",
            f"{indent}b = torch.randn({N}, device='cuda', dtype=torch.float32)",
            f"{indent}x = torch.zeros({N}, device='cuda', dtype=torch.float32)",
        ]

    if kernel_name == 'gramschmidt':
        M = params.get('M', 60)
        N = params.get('N', 80)
        return [
            f"{indent}# Well-conditioned A with strong diagonal for stable Gram-Schmidt",
            f"{indent}A = torch.randn({M}, {N}, device='cuda', dtype=torch.float32) + torch.eye({M}, {N}, device='cuda', dtype=torch.float32) * 5.0",
            f"{indent}R = torch.zeros({N}, {N}, device='cuda', dtype=torch.float32)",
            f"{indent}Q = torch.zeros({M}, {N}, device='cuda', dtype=torch.float32)",
        ]

    if kernel_name == 'nussinov':
        N = params.get('N', 180)
        return [
            f"{indent}# Integer base sequence {{0..3}} and zero-initialized score table",
            f"{indent}seq = torch.randint(0, 4, ({N},), device='cuda').float()",
            f"{indent}table = torch.zeros({N}, {N}, device='cuda', dtype=torch.float32)",
        ]

    if kernel_name == 'floyd_warshall':
        N = params.get('N', 120)
        return [
            f"{indent}# Non-negative edge weights for shortest-path",
            f"{indent}path = torch.abs(torch.randn({N}, {N}, device='cuda', dtype=torch.float32)) * 10.0 + 1.0",
        ]

    return None


def _gen_ctypes_array_setup(kernel_name: str, arrays: dict, params: dict) -> str:
    """Generate ctypes code to set global arrays in the .so."""
    lines = []
    for arr_name, mode in sorted(arrays.items()):
        if mode in ['r', 'rw', 'temp']:
            shape = _get_array_shape(kernel_name, arr_name, params)
            if shape:
                total = " * ".join(str(s) for s in shape)
                c_type = _get_array_c_type(kernel_name, arr_name)
                ct, np_dt = _C_TYPE_MAP.get(c_type, ('c_float', 'float32'))
                lines.append(f"    CType_{arr_name} = ctypes.{ct} * ({total})")
                lines.append(f"    c_arr_{arr_name} = CType_{arr_name}.in_dll(lib, '{arr_name}')")
                if np_dt != 'float32':
                    # Convert float32 values to the C array's native type
                    lines.append(f"    src_{arr_name} = np.ascontiguousarray({arr_name}_c.astype(np.{np_dt}), dtype=np.{np_dt})")
                else:
                    lines.append(f"    src_{arr_name} = np.ascontiguousarray({arr_name}_c, dtype=np.float32)")
                lines.append(f"    ctypes.memmove(c_arr_{arr_name}, src_{arr_name}.ctypes.data, src_{arr_name}.nbytes)")
    return "\n".join(lines) if lines else "    pass"


def _gen_ctypes_scalar_setup(kernel_name: str, scalar_params: dict, params: dict) -> str:
    """Generate ctypes code to set global scalars in the .so."""
    lines = []
    # Scalar params from the function spec
    for sp_name in sorted(scalar_params.keys()):
        lines.append(f"    ctypes.c_float.in_dll(lib, '{sp_name}').value = float({sp_name})")
    return "\n".join(lines) if lines else "    pass"


def _gen_ctypes_array_readback(kernel_name: str, arrays: dict, params: dict) -> str:
    """Generate ctypes code to read back modified arrays from the .so."""
    lines = []
    for arr_name, mode in sorted(arrays.items()):
        if mode in ['rw', 'w']:
            shape = _get_array_shape(kernel_name, arr_name, params)
            if shape:
                total = " * ".join(str(s) for s in shape)
                shape_tuple = ", ".join(str(s) for s in shape)
                c_type = _get_array_c_type(kernel_name, arr_name)
                ct, np_dt = _C_TYPE_MAP.get(c_type, ('c_float', 'float32'))
                lines.append(f"    CType_{arr_name} = ctypes.{ct} * ({total})")
                lines.append(f"    c_arr_{arr_name} = CType_{arr_name}.in_dll(lib, '{arr_name}')")
                if np_dt != 'float32':
                    # Read native type and convert back to float32 for comparison
                    lines.append(f"    {arr_name}_c[:] = np.frombuffer(c_arr_{arr_name}, dtype=np.{np_dt}).reshape({shape_tuple}).astype(np.float32).copy()")
                else:
                    lines.append(f"    {arr_name}_c[:] = np.frombuffer(c_arr_{arr_name}, dtype=np.float32).reshape({shape_tuple}).copy()")
    return "\n".join(lines) if lines else "    pass"


def _gen_comparison_code(output_arrays: list) -> str:
    """Generate comparison code for output arrays using combined abs+rel tolerance."""
    lines = []
    lines.append(f"            max_rel_error = 0.0")
    for arr in output_arrays:
        lines.append(f"            c_val = torch.from_numpy({arr}_c).float()")
        lines.append(f"            tr_val = {arr}_tr.cpu().float()")
        lines.append(f"            abs_err = torch.max(torch.abs(c_val - tr_val)).item()")
        lines.append(f"            denom = torch.max(torch.abs(c_val)).item()")
        lines.append(f"            rel_err = abs_err / max(denom, 1e-10)")
        lines.append(f"            max_error = max(max_error, abs_err)")
        lines.append(f"            max_rel_error = max(max_rel_error, rel_err)")
    return "\n".join(lines)


# ============================================================================
# Benchmarking
# ============================================================================

def generate_benchmark_test(kernel_name: str, func_spec: dict, attempt: int = 1) -> str:
    """Generate performance benchmark script for a Polybench kernel."""
    arrays = func_spec['arrays']
    scalar_params = func_spec.get('scalar_params', {})
    params = get_kernel_params(kernel_name)

    func_id = kernel_name
    if func_id[0].isdigit():
        func_id = "k" + func_id

    # Build array initialization
    array_inits = []
    domain_inits = _get_domain_array_inits(kernel_name, arrays, params, "    ")
    if domain_inits is not None:
        array_inits.extend(domain_inits)
    else:
        for arr_name, mode in sorted(arrays.items()):
            shape = _get_array_shape(kernel_name, arr_name, params)
            if shape:
                shape_str = ", ".join(str(s) for s in shape)
                array_inits.append(f"    {arr_name} = torch.randn({shape_str}, device='cuda', dtype=torch.float32)")

    for sp_name in sorted(scalar_params.keys()):
        if sp_name in ('alpha', 'beta'):
            array_inits.append(f"    {sp_name} = 1.5")
        elif sp_name in ('float_n',):
            n_val = params.get('N', 100)
            array_inits.append(f"    {sp_name} = float({n_val})")
        elif sp_name == 'eps':
            array_inits.append(f"    {sp_name} = 0.1")
        else:
            array_inits.append(f"    {sp_name} = 1.0")

    for p_name, p_val in sorted(params.items()):
        if p_name not in scalar_params:
            array_inits.append(f"    {p_name} = {p_val}")

    array_init_str = "\n".join(array_inits)

    array_names = sorted([a for a, m in arrays.items() if m in ['r', 'rw', 'w', 'temp']])
    scalar_names = sorted(scalar_params.keys())
    dim_names = sorted([p for p in params.keys() if p not in scalar_params])

    # C reference args
    c_args = [f"{a}_c" for a in array_names] + scalar_names + dim_names
    c_call_str = ", ".join(c_args)

    # Triton args
    tr_args = [f"{a}_tr" for a in array_names] + scalar_names + dim_names
    tr_call_str = ", ".join(tr_args)

    # Import block
    llm_subdir = "llm_triton" if ENABLE_ANALYSIS else "llm_triton_no_analysis"
    if kernel_name[0].isdigit():
        import_block = (
            f"import importlib\n"
            f"    _mod = importlib.import_module(\"{OUTPUT_DIR}.{llm_subdir}.{kernel_name}.attempt{attempt}\")\n"
            f"    {func_id}_triton = _mod.{func_id}_triton"
        )
    else:
        import_block = f"from {OUTPUT_DIR}.{llm_subdir}.{kernel_name}.attempt{attempt} import {func_id}_triton"

    benchmark_code = f'''#!/usr/bin/env python3
"""Performance Benchmark for {kernel_name} (Polybench)"""
import sys
import time
import ctypes
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

try:
    {import_block}
except ImportError as e:
    print(f"Import error: {{e}}")
    sys.exit(1)

C_LIB_PATH = Path(__file__).parent.parent.parent / "c_reference" / "polybench_libs" / "lib{kernel_name}.so"

def run_c_reference({c_call_str}):
    lib = ctypes.CDLL(str(C_LIB_PATH))
{_gen_ctypes_array_setup(kernel_name, arrays, params)}
{_gen_ctypes_scalar_setup(kernel_name, scalar_params, params)}
    func = getattr(lib, "{func_id}_kernel")
    func.argtypes = []
    func.restype = None
    func()
{_gen_ctypes_array_readback(kernel_name, arrays, params)}

def benchmark():
    num_warmup = 5
    num_iterations = 50

{array_init_str}

    # C reference benchmark
    c_time = None
    try:
        for _ in range(num_warmup):
{chr(10).join(["            " + f"{a}_c = {a}.cpu().numpy().copy()" for a in array_names])}
            run_c_reference({c_call_str})
        start = time.perf_counter()
        for _ in range(num_iterations):
{chr(10).join(["            " + f"{a}_c = {a}.cpu().numpy().copy()" for a in array_names])}
            run_c_reference({c_call_str})
        c_time = (time.perf_counter() - start) / num_iterations
    except Exception as e:
        print(f"C ref error: {{e}}")

    # Triton benchmark
    tr_time = None
    try:
        for _ in range(num_warmup):
{chr(10).join(["            " + f"{a}_tr = {a}.clone()" for a in array_names])}
            {func_id}_triton({tr_call_str})
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
{chr(10).join(["            " + f"{a}_tr = {a}.clone()" for a in array_names])}
            {func_id}_triton({tr_call_str})
        torch.cuda.synchronize()
        tr_time = (time.perf_counter() - start) / num_iterations
    except Exception as e:
        print(f"Triton error: {{e}}")

    # Report
    speedup = c_time / tr_time if c_time and tr_time and tr_time > 0 else None
    c_ms = c_time * 1000 if c_time else -1
    tr_ms = tr_time * 1000 if tr_time else -1
    sp = speedup if speedup else -1

    print(f"C ref:   {{c_ms:8.3f}} ms")
    print(f"Triton:  {{tr_ms:8.3f}} ms")
    if speedup:
        print(f"Speedup: {{speedup:8.2f}}x")
    else:
        print(f"Speedup: N/A")
    print(f"BENCHMARK_RESULT:{{c_ms:.6f}},{{tr_ms:.6f}},{{sp:.6f}}")

if __name__ == "__main__":
    benchmark()
'''
    return benchmark_code


def run_benchmark(kernel_name: str, benchmark_file: Path) -> Optional[dict]:
    """Run performance benchmark and parse results."""
    try:
        env = os.environ.copy()
        env['MKL_THREADING_LAYER'] = 'GNU'
        result = subprocess.run(
            [sys.executable, str(benchmark_file)],
            capture_output=True, text=True, timeout=180,
            cwd=Path.cwd(), env=env
        )

        stdout = result.stdout
        match = re.search(r'BENCHMARK_RESULT:([-\d.]+),([-\d.]+),([-\d.]+)', stdout)
        if match:
            c_ms = float(match.group(1))
            tr_ms = float(match.group(2))
            sp = float(match.group(3))
            return {
                'c_ref_time_ms': c_ms if c_ms > 0 else None,
                'triton_time_ms': tr_ms if tr_ms > 0 else None,
                'speedup': sp if sp > 0 else None,
            }
        return None
    except Exception:
        return None


# ============================================================================
# LLM interaction
# ============================================================================

def generate_triton_initial(kernel_name: str, func_spec: dict) -> Tuple[str, str, str]:
    """Generate initial Triton implementation for a Polybench kernel."""
    prompt = build_polybench_prompt(kernel_name, func_spec)

    print(f"  Generating Triton code (attempt 1/{MAX_ATTEMPTS})...")

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}]
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_response = f"""# LLM-Generated Triton Implementation for {kernel_name} (Polybench)
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


def generate_triton_with_retry(kernel_name: str, original_prompt: str,
                               last_attempt: str, error_info: dict,
                               attempt_num: int) -> Tuple[str, str]:
    """Generate Triton code with error feedback for retry."""
    if error_info['type'] == 'numerical':
        error_section = f"""
## PREVIOUS ATTEMPT FAILED - NUMERICAL ERROR

Your last attempt produced incorrect numerical results.

**Error type**: Numerical error (values don't match)
**Max error observed**: {error_info.get('max_error', 'unknown')}

Please fix the numerical computation. Common causes:
- Incorrect loop bounds or indices
- Wrong array access patterns (row-major vs column-major)
- Missing or incorrect operations
- Off-by-one errors
- Cloning input arrays instead of modifying them in-place (the test checks the original tensors)

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
1. Using `grid=(1,)` with a single thread doing all work — this is SEQUENTIAL on GPU!
2. Using scalar `tl.load`/`tl.store` in a Python for loop instead of vectorized block operations
3. NOT using `tl.program_id(0)` to distribute work across GPU blocks
4. Processing ALL elements in one block instead of splitting across multiple blocks

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

**For 2D kernels**, parallelize the outer loop dimension:
```python
@triton.jit
def kernel(..., N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)  # Each block handles one row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    # Process row 'row' with vectorized column access
```

## LAST ATTEMPT (NEEDS PARALLELIZATION FIX):
```python
{last_attempt}
```
"""
    else:
        error_section = f"""
## PREVIOUS ATTEMPT FAILED

**Error type**: {error_info['type']}
**Error message**:
```
{error_info['message'][:2000]}
```

Please fix the error.

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
        max_tokens=8192,
        messages=[{"role": "user", "content": retry_prompt}]
    )

    response_text = message.content[0].text
    triton_code = response_text
    if "```python" in triton_code:
        triton_code = triton_code.split("```python")[1].split("```")[0].strip()
    elif "```" in triton_code:
        triton_code = triton_code.split("```")[1].split("```")[0].strip()

    return triton_code, retry_prompt


# ============================================================================
# Test execution
# ============================================================================

def run_test(kernel_name: str, test_file: Path) -> Tuple[bool, dict]:
    """Run the correctness test and parse results."""
    try:
        env = os.environ.copy()
        env['MKL_THREADING_LAYER'] = 'GNU'
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True, text=True, timeout=120,
            cwd=Path.cwd(), env=env
        )

        stdout = result.stdout
        stderr = result.stderr
        combined = stdout + stderr

        if result.returncode == 0 and "All tests PASSED!" in stdout:
            return True, {}

        if "Import error:" in combined or "ImportError" in combined:
            return False, {'type': 'import', 'message': combined[-2000:]}

        if "CompilationError" in combined:
            return False, {'type': 'compilation', 'message': combined[-2000:]}

        if "FAIL" in stdout and ("abs=" in stdout or "max_error" in stdout):
            abs_match = re.search(r'abs[=:]\s*([\d.e+-]+)', stdout)
            rel_match = re.search(r'rel[=:]\s*([\d.e+-]+)', stdout)
            max_error = abs_match.group(1) if abs_match else 'unknown'
            rel_error = rel_match.group(1) if rel_match else 'unknown'
            return False, {'type': 'numerical', 'message': f"abs_error={max_error} rel_error={rel_error}", 'max_error': max_error}

        if "ERROR:" in stdout or "Exception" in combined or "Error" in stderr:
            return False, {'type': 'runtime', 'message': combined[-2000:]}

        return False, {'type': 'unknown', 'message': combined[-2000:]}

    except subprocess.TimeoutExpired:
        return False, {'type': 'timeout', 'message': 'Test timed out after 120 seconds'}
    except Exception as e:
        return False, {'type': 'exception', 'message': str(e)}


# ============================================================================
# Main pipeline
# ============================================================================

def process_kernel(kernel_name: str, func_spec: dict) -> dict:
    """Process a single Polybench kernel with retry logic and speedup-based retry."""
    print(f"\n{'=' * 70}")
    print(f"Processing: {kernel_name}")
    print(f"  Arrays: {list(func_spec['arrays'].keys())}")
    print(f"  Params: {list(get_kernel_params(kernel_name).keys())}")
    print(f"{'=' * 70}")

    base_dir = Path(OUTPUT_DIR)
    suffix = "" if ENABLE_ANALYSIS else "_no_analysis"
    llm_dir = base_dir / f"llm_triton{suffix}"
    func_dir = llm_dir / kernel_name
    raw_dir = llm_dir / "raw_responses" / kernel_name
    test_dir = Path("my_polybench_tests") / kernel_name

    # Clean previous attempts
    for d in [func_dir, raw_dir]:
        if d.exists():
            shutil.rmtree(d)

    for d in [func_dir, raw_dir, test_dir]:
        d.mkdir(exist_ok=True, parents=True)

    # Create __init__.py for imports
    (llm_dir / "__init__.py").touch()
    (func_dir / "__init__.py").touch()
    (base_dir / "__init__.py").touch()

    test_file = test_dir / f"test_{kernel_name}_correctness.py"

    results = {
        "triton_generated": False,
        "test_passed": False,
        "attempts": 0,
        "final_error": None
    }

    original_prompt = None
    last_code = None
    error_info = None
    reset_after = 5  # Reset context after this many failures (5+5 strategy)

    # Track the best passing result across all attempts
    best_result = None
    best_speedup = -float('inf')
    best_attempt = 0

    for attempt in range(1, MAX_ATTEMPTS + 1):
        results["attempts"] = attempt

        triton_file = func_dir / f"attempt{attempt}.py"
        raw_file = raw_dir / f"attempt{attempt}.txt"

        try:
            if attempt == 1:
                triton_code, original_prompt, full_response = generate_triton_initial(kernel_name, func_spec)
            elif attempt == reset_after + 1:
                # Reset: generate fresh without showing previous failed code
                print(f"  Resetting context after {reset_after} failures, trying fresh approach...")
                triton_code, retry_prompt = generate_triton_with_retry(
                    kernel_name, original_prompt, None, error_info, attempt
                )
                full_response = retry_prompt
            else:
                triton_code, retry_prompt = generate_triton_with_retry(
                    kernel_name, original_prompt, last_code, error_info, attempt
                )
                full_response = retry_prompt

            last_code = triton_code

            with open(raw_file, 'w') as f:
                f.write(full_response)
            with open(triton_file, 'w') as f:
                f.write(triton_code)

            print(f"  Saved Triton code to: {triton_file}")
            results["triton_generated"] = True

            # Generate and run test
            test_code = generate_correctness_test(kernel_name, func_spec, attempt)
            with open(test_file, 'w') as f:
                f.write(test_code)

            print(f"  Running correctness test...")
            passed, error_info = run_test(kernel_name, test_file)

            if passed:
                print(f"  PASSED on attempt {attempt}!")
                results["test_passed"] = True

                # Run performance benchmark
                print(f"  Running performance benchmark...")
                bench_dir = test_dir
                bench_file = bench_dir / f"benchmark_{kernel_name}.py"
                bench_code = generate_benchmark_test(kernel_name, func_spec, attempt)
                with open(bench_file, 'w') as f:
                    f.write(bench_code)

                benchmark_results = run_benchmark(kernel_name, bench_file)
                if benchmark_results:
                    results["benchmark"] = benchmark_results
                    speedup = benchmark_results.get('speedup', 0) or 0
                    print(f"  Benchmark: {speedup:.2f}x speedup")

                    # Track the best passing result
                    if speedup > best_speedup:
                        best_speedup = speedup
                        best_attempt = attempt
                        best_result = {
                            "triton_generated": True,
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
                    # Use best_result's benchmark if available
                    if best_result and best_result.get("benchmark"):
                        results["benchmark"] = best_result["benchmark"]
                    else:
                        results["benchmark"] = None

                # Return immediately if speedup is good enough
                results["final_attempt"] = attempt
                return results
            else:
                print(f"  FAILED: {error_info.get('type', 'unknown')} - {error_info.get('message', '')[:100]}")
                results["final_error"] = error_info

        except Exception as e:
            print(f"  Exception on attempt {attempt}: {e}")
            error_info = {'type': 'exception', 'message': str(e)}
            results["final_error"] = error_info

    # Return the best passing result if we have one, otherwise return the last result
    if best_result is not None:
        print(f"  Returning best result from attempt {best_attempt} with {best_speedup:.2f}x speedup")
        best_result["attempts"] = results["attempts"]  # Total attempts made
        best_result["test_passed"] = True
        return best_result

    return results


def benchmark_passed_kernels(kernel_names: list = None):
    """Run benchmarks on all passed kernels."""
    import json

    results_file = Path(OUTPUT_DIR) / "results.json"
    if not results_file.exists():
        print("No results.json found — run the pipeline first!")
        return

    with open(results_file) as f:
        all_results = json.load(f)

    # Find passed kernels and their best attempt
    to_benchmark = {}
    for kernel_name, result in all_results.items():
        if kernel_names and kernel_name not in kernel_names:
            continue
        if result.get("test_passed"):
            to_benchmark[kernel_name] = result["attempts"]

    print("=" * 70)
    print(f"Benchmarking {len(to_benchmark)} passed kernels")
    print("=" * 70)

    bench_results = {}
    for i, (kernel_name, attempt) in enumerate(sorted(to_benchmark.items()), 1):
        func_spec = POLYBENCH_FUNCTIONS[kernel_name]
        print(f"\n[{i}/{len(to_benchmark)}] {kernel_name} (attempt {attempt})...")

        bench_dir = Path("my_polybench_tests") / kernel_name
        bench_dir.mkdir(exist_ok=True, parents=True)
        bench_file = bench_dir / f"benchmark_{kernel_name}.py"

        bench_code = generate_benchmark_test(kernel_name, func_spec, attempt)
        with open(bench_file, 'w') as f:
            f.write(bench_code)

        result = run_benchmark(kernel_name, bench_file)
        if result:
            bench_results[kernel_name] = result
            c_ms = result.get('c_ref_time_ms')
            tr_ms = result.get('triton_time_ms')
            sp = result.get('speedup')
            c_str = f"{c_ms:.3f}ms" if c_ms else "N/A"
            tr_str = f"{tr_ms:.3f}ms" if tr_ms else "N/A"
            sp_str = f"{sp:.2f}x" if sp else "N/A"
            print(f"  C: {c_str}  Triton: {tr_str}  Speedup: {sp_str}")
        else:
            print(f"  Benchmark failed")

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"{'Kernel':<18} {'C ref (ms)':<14} {'Triton (ms)':<14} {'Speedup':<10}")
    print(f"{'-' * 56}")
    for kernel_name, result in sorted(bench_results.items()):
        c_ms = result.get('c_ref_time_ms')
        tr_ms = result.get('triton_time_ms')
        sp = result.get('speedup')
        c_str = f"{c_ms:.3f}" if c_ms else "N/A"
        tr_str = f"{tr_ms:.3f}" if tr_ms else "N/A"
        sp_str = f"{sp:.2f}x" if sp else "N/A"
        print(f"{kernel_name:<18} {c_str:<14} {tr_str:<14} {sp_str:<10}")

    # Save benchmark results
    bench_file = Path(OUTPUT_DIR) / "benchmark_results.json"
    with open(bench_file, 'w') as f:
        json.dump(bench_results, f, indent=2)
    print(f"\nBenchmark results saved to: {bench_file}")

    # Stats
    speedups = [r['speedup'] for r in bench_results.values() if r.get('speedup') and r['speedup'] > 0]
    if speedups:
        print(f"\nSpeedup stats ({len(speedups)} kernels):")
        print(f"  Median: {sorted(speedups)[len(speedups)//2]:.2f}x")
        print(f"  Mean:   {sum(speedups)/len(speedups):.2f}x")
        print(f"  Min:    {min(speedups):.2f}x")
        print(f"  Max:    {max(speedups):.2f}x")
        print(f"  >1x:    {sum(1 for s in speedups if s > 1)}/{len(speedups)}")


def main():
    """Main Polybench pipeline."""
    global ENABLE_ANALYSIS

    # Check for --no-analysis flag
    if '--no-analysis' in sys.argv:
        sys.argv.remove('--no-analysis')
        ENABLE_ANALYSIS = False

    # Check for --benchmark flag
    if '--benchmark' in sys.argv:
        sys.argv.remove('--benchmark')
        kernel_names = [k.replace("-", "_") for k in sys.argv[1:]] if len(sys.argv) > 1 else None
        benchmark_passed_kernels(kernel_names)
        return

    analysis_mode = "WITH analysis" if ENABLE_ANALYSIS else "WITHOUT analysis (ablation)"
    print("=" * 70)
    print("Polybench/C Generation and Testing Pipeline")
    print(f"Mode: {analysis_mode}")
    print(f"Total kernels available: {len(POLYBENCH_FUNCTIONS)}")
    print(f"Max attempts per kernel: {MAX_ATTEMPTS}")
    print("=" * 70)

    if not client:
        print("ERROR: ANTHROPIC_API_KEY not set!")
        sys.exit(1)

    # Check if specific kernels requested
    if len(sys.argv) > 1:
        kernel_names = sys.argv[1:]
        kernels_to_process = {}
        for k in kernel_names:
            c_name = k.replace("-", "_")
            if c_name in POLYBENCH_FUNCTIONS:
                kernels_to_process[c_name] = POLYBENCH_FUNCTIONS[c_name]
            else:
                print(f"Warning: Kernel not found: {k}")
        print(f"Processing {len(kernels_to_process)} specific kernels: {list(kernels_to_process.keys())}")
    else:
        kernels_to_process = POLYBENCH_FUNCTIONS
        print(f"Processing ALL {len(kernels_to_process)} kernels")

    if not kernels_to_process:
        print("No valid kernels to process!")
        return

    all_results = {}
    for i, (kernel_name, func_spec) in enumerate(kernels_to_process.items(), 1):
        print(f"\n[{i}/{len(kernels_to_process)}]", end=" ")
        results = process_kernel(kernel_name, func_spec)
        all_results[kernel_name] = results

    # Print summary
    print(f"\n\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Kernel':<18} {'Passed':<8} {'Att':<5} {'Speedup':<10}")
    print(f"{'-' * 41}")

    for kernel_name, results in all_results.items():
        passed = "Y" if results["test_passed"] else "N"
        attempts = str(results["attempts"])
        bench = results.get("benchmark")
        sp_str = f"{bench['speedup']:.2f}x" if bench and bench.get('speedup') else "-"
        print(f"{kernel_name:<18} {passed:<8} {attempts:<5} {sp_str:<10}")

    print(f"{'=' * 70}")

    total = len(all_results)
    triton_ok = sum(1 for r in all_results.values() if r["triton_generated"])
    passed = sum(1 for r in all_results.values() if r["test_passed"])
    first_try = sum(1 for r in all_results.values() if r["test_passed"] and r["attempts"] == 1)

    print(f"\nTriton generated: {triton_ok}/{total}")
    print(f"Tests passed: {passed}/{total} ({100*passed/total:.1f}%)")
    print(f"  - Passed on first try: {first_try}")
    print(f"  - Passed after retry: {passed - first_try}")

    # Print speedup stats from inline benchmarks
    speedups = []
    for r in all_results.values():
        bench = r.get("benchmark")
        if bench and bench.get("speedup") and bench["speedup"] > 0:
            speedups.append(bench["speedup"])
    if speedups:
        speedups_sorted = sorted(speedups)
        print(f"\nSpeedup stats ({len(speedups)} benchmarked kernels):")
        print(f"  Median: {speedups_sorted[len(speedups)//2]:.2f}x")
        print(f"  Mean:   {sum(speedups)/len(speedups):.2f}x")
        print(f"  Min:    {min(speedups):.2f}x")
        print(f"  Max:    {max(speedups):.2f}x")
        print(f"  >1x:    {sum(1 for s in speedups if s > 1)}/{len(speedups)}")
    print(f"{'=' * 70}")

    # Save results to JSON (merge with existing results when running specific kernels)
    import json
    results_filename = "results_no_analysis.json" if not ENABLE_ANALYSIS else "results.json"
    results_file = Path(OUTPUT_DIR) / results_filename
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # Load existing results and merge
    existing_results = {}
    if results_file.exists():
        with open(results_file) as f:
            existing_results = json.load(f)

    # Update with new results (overwrite per-kernel)
    merged = {**existing_results, **{k: {kk: vv for kk, vv in v.items() if kk != 'final_error' or vv is None or isinstance(vv, (str, dict))}
                    for k, v in all_results.items()}}

    with open(results_file, 'w') as f:
        json.dump(merged, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")

    # Also save/merge benchmark results from inline benchmarks
    bench_data = {}
    for kernel_name, r in all_results.items():
        bench = r.get("benchmark")
        if bench:
            bench_data[kernel_name] = bench
    if bench_data:
        bench_file = Path(OUTPUT_DIR) / "benchmark_results.json"
        existing_bench = {}
        if bench_file.exists():
            with open(bench_file) as f:
                existing_bench = json.load(f)
        merged_bench = {**existing_bench, **bench_data}
        with open(bench_file, 'w') as f:
            json.dump(merged_bench, f, indent=2)
        print(f"Benchmark results saved to: {bench_file}")


if __name__ == "__main__":
    main()
