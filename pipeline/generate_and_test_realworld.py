#!/usr/bin/env python3
"""
Integrated Generation and Testing Pipeline for Real-World Application Kernels

Real-world Rodinia kernels with higher algorithmic complexity:
  - srad: Speckle-Reducing Anisotropic Diffusion (multi-phase stencil)
  - lavaMD: Molecular Dynamics N-body with spatial decomposition
  - gaussian: Gaussian Elimination with back-substitution

Usage:
    python generate_and_test_realworld.py                    # Process all 3 kernels
    python generate_and_test_realworld.py srad gaussian      # Process specific kernels
    python generate_and_test_realworld.py --no-analysis      # Ablation: no analysis
    python generate_and_test_realworld.py --benchmark srad   # Re-run benchmarks only
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

# Add analysis directory to path
sys.path.insert(0, "/home/qinxiao/workspace/compiler-guided-triton-gen/analysis")

from kernel_analysis import analyze_kernel, format_analysis_for_prompt

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=API_KEY) if API_KEY else None

REALWORLD_KERNELS_DIR = "/home/qinxiao/workspace/compiler-guided-triton-gen/analysis/kernels_realworld"
MAX_ATTEMPTS = 10
OUTPUT_DIR = "../results/realworld/realworld_results"
ENABLE_ANALYSIS = True


# ============================================================================
# Function database (inline — only 3 kernels)
# ============================================================================

REALWORLD_FUNCTIONS = {
    "srad": {
        "name": "srad",
        "loop_code": """\
    for (iter = 0; iter < NITER; iter++) {
        /* Phase 0: Compute ROI statistics */
        sum = 0.0f;
        sum2 = 0.0f;
        for (i = R1; i <= R2; i++) {
            for (j = C1; j <= C2; j++) {
                float tmp = J[i * COLS + j];
                sum += tmp;
                sum2 += tmp * tmp;
            }
        }
        meanROI = sum / SIZE_R;
        varROI = (sum2 / SIZE_R) - meanROI * meanROI;
        q0sqr = varROI / (meanROI * meanROI);

        /* Phase 1: Directional derivatives + diffusion coefficient */
        for (i = 0; i < ROWS; i++) {
            for (j = 0; j < COLS; j++) {
                k = i * COLS + j;
                Jc = J[k];

                dN[k] = J[iN[i] * COLS + j] - Jc;
                dS[k] = J[iS[i] * COLS + j] - Jc;
                dW[k] = J[i * COLS + jW[j]] - Jc;
                dE[k] = J[i * COLS + jE[j]] - Jc;

                G2 = (dN[k] * dN[k] + dS[k] * dS[k]
                    + dW[k] * dW[k] + dE[k] * dE[k]) / (Jc * Jc);

                L = (dN[k] + dS[k] + dW[k] + dE[k]) / Jc;

                num = (0.5f * G2) - ((1.0f / 16.0f) * (L * L));
                den = 1.0f + (0.25f * L);
                qsqr = num / (den * den);

                den = (qsqr - q0sqr) / (q0sqr * (1.0f + q0sqr));
                c[k] = 1.0f / (1.0f + den);

                if (c[k] < 0.0f) c[k] = 0.0f;
                else if (c[k] > 1.0f) c[k] = 1.0f;
            }
        }

        /* Phase 2: Divergence + image update */
        for (i = 0; i < ROWS; i++) {
            for (j = 0; j < COLS; j++) {
                k = i * COLS + j;

                cN = c[k];
                cS = c[iS[i] * COLS + j];
                cW = c[k];
                cE = c[i * COLS + jE[j]];

                D = cN * dN[k] + cS * dS[k] + cW * dW[k] + cE * dE[k];

                J[k] = J[k] + 0.25f * lambda_val * D;
            }
        }
    }""",
        "arrays": {
            'J': 'rw', 'c': 'rw', 'dN': 'rw', 'dS': 'rw', 'dW': 'rw', 'dE': 'rw',
            'iN': 'r', 'iS': 'r', 'jW': 'r', 'jE': 'r',
        },
        "has_offset": True,
        "has_conditional": True,
        "has_reduction": True,
        "has_2d_arrays": False,
        "has_3d_arrays": False,
        "scalar_params": {'lambda_val': 'scalar'},
    },
    "lavaMD": {
        "name": "lavaMD",
        "loop_code": """\
    a2 = 2.0 * alpha_val * alpha_val;

    for (l = 0; l < NUMBER_BOXES; l++) {
        first_i = box_offset[l];

        for (k = 0; k < (1 + box_nn[l]); k++) {
            if (k == 0) {
                pointer = l;
            } else {
                pointer = box_nei[l * MAX_NEIGHBORS + (k - 1)];
            }
            first_j = box_offset[pointer];

            for (i = 0; i < NUMBER_PAR_PER_BOX; i++) {
                for (j = 0; j < NUMBER_PAR_PER_BOX; j++) {
                    r2 = rv_v[first_i + i] + rv_v[first_j + j]
                       - (rv_x[first_i + i] * rv_x[first_j + j]
                        + rv_y[first_i + i] * rv_y[first_j + j]
                        + rv_z[first_i + i] * rv_z[first_j + j]);

                    u2 = a2 * r2;
                    vij = exp(-u2);
                    fs = 2.0 * vij;

                    dx = rv_x[first_i + i] - rv_x[first_j + j];
                    dy = rv_y[first_i + i] - rv_y[first_j + j];
                    dz = rv_z[first_i + i] - rv_z[first_j + j];

                    fxij = fs * dx;
                    fyij = fs * dy;
                    fzij = fs * dz;

                    fv_v[first_i + i] += qv[first_j + j] * vij;
                    fv_x[first_i + i] += qv[first_j + j] * fxij;
                    fv_y[first_i + i] += qv[first_j + j] * fyij;
                    fv_z[first_i + i] += qv[first_j + j] * fzij;
                }
            }
        }
    }""",
        "arrays": {
            'rv_v': 'r', 'rv_x': 'r', 'rv_y': 'r', 'rv_z': 'r',
            'qv': 'r',
            'fv_v': 'rw', 'fv_x': 'rw', 'fv_y': 'rw', 'fv_z': 'rw',
            'box_offset': 'r', 'box_nn': 'r', 'box_nei': 'r',
        },
        "has_offset": False,
        "has_conditional": True,
        "has_reduction": True,
        "has_2d_arrays": False,
        "has_3d_arrays": False,
        "scalar_params": {'alpha_val': 'scalar'},
    },
    "gaussian": {
        "name": "gaussian",
        "loop_code": """\
    /* Forward elimination */
    for (t = 0; t < N - 1; t++) {
        /* Phase 1: Compute multipliers (Fan1) */
        for (i = t + 1; i < N; i++) {
            m[i * N + t] = a[i * N + t] / a[t * N + t];
        }

        /* Phase 2: Eliminate column t (Fan2) */
        for (i = t + 1; i < N; i++) {
            for (j = t; j < N; j++) {
                a[i * N + j] -= m[i * N + t] * a[t * N + j];
            }
            b[i] -= m[i * N + t] * b[t];
        }
    }

    /* Back substitution */
    x[N - 1] = b[N - 1] / a[(N - 1) * N + (N - 1)];
    for (i = N - 2; i >= 0; i--) {
        float sum = b[i];
        for (j = i + 1; j < N; j++) {
            sum -= a[i * N + j] * x[j];
        }
        x[i] = sum / a[i * N + i];
    }""",
        "arrays": {
            'a': 'rw', 'm': 'rw', 'b': 'rw', 'x': 'w',
        },
        "has_offset": False,
        "has_conditional": False,
        "has_reduction": True,
        "has_2d_arrays": False,
        "has_3d_arrays": False,
        "scalar_params": {},
    },
    "spmv": {
        "name": "spmv",
        "loop_code": """\
    for (row = 0; row < NROWS; row++) {
        sum = 0.0f;
        for (i = row_offsets[row]; i < row_offsets[row + 1]; i++) {
            sum += vals[i] * x[cols[i]];
        }
        y[row] = sum;
    }""",
        "arrays": {
            'row_offsets': 'r', 'cols': 'r', 'vals': 'r',
            'x': 'r', 'y': 'w',
        },
        "has_offset": True,
        "has_conditional": False,
        "has_reduction": True,
        "has_2d_arrays": False,
        "has_3d_arrays": False,
        "scalar_params": {},
    },
    "lj_force": {
        "name": "lj_force",
        "loop_code": """\
    /* Clear forces */
    for (i = 0; i < NLOCAL; i++) {
        f[i * PAD + 0] = 0.0f;
        f[i * PAD + 1] = 0.0f;
        f[i * PAD + 2] = 0.0f;
    }

    /* Compute pairwise forces -- full neighbor list, no Newton 3rd law */
    for (i = 0; i < NLOCAL; i++) {
        xtmp = pos[i * PAD + 0];
        ytmp = pos[i * PAD + 1];
        ztmp = pos[i * PAD + 2];
        fix = 0.0f;
        fiy = 0.0f;
        fiz = 0.0f;

        for (k = 0; k < numneigh[i]; k++) {
            j = neighbors[i * MAXNEIGHS + k];
            delx = xtmp - pos[j * PAD + 0];
            dely = ytmp - pos[j * PAD + 1];
            delz = ztmp - pos[j * PAD + 2];
            rsq = delx * delx + dely * dely + delz * delz;

            if (rsq < cutforcesq_val) {
                sr2 = 1.0f / rsq;
                sr6 = sr2 * sr2 * sr2 * sigma6_val;
                force = 48.0f * sr6 * (sr6 - 0.5f) * sr2 * epsilon_val;
                fix += delx * force;
                fiy += dely * force;
                fiz += delz * force;
            }
        }

        f[i * PAD + 0] += fix;
        f[i * PAD + 1] += fiy;
        f[i * PAD + 2] += fiz;
    }""",
        "arrays": {
            'pos': 'r', 'f': 'w',
            'neighbors': 'r', 'numneigh': 'r',
        },
        "has_offset": True,
        "has_conditional": True,
        "has_reduction": True,
        "has_2d_arrays": False,
        "has_3d_arrays": False,
        "scalar_params": {'cutforcesq_val': 'scalar', 'sigma6_val': 'scalar', 'epsilon_val': 'scalar'},
    },
}

REALWORLD_KERNELS = {
    "srad": {
        "params": {"ROWS": 512, "COLS": 512, "NITER": 10,
                   "R1": 0, "R2": 127, "C1": 0, "C2": 127},
    },
    "lavaMD": {
        "params": {"BOXES1D": 4, "NUMBER_PAR_PER_BOX": 100, "MAX_NEIGHBORS": 26},
    },
    "gaussian": {
        "params": {"N": 512},
    },
    "spmv": {
        "params": {"NROWS": 4096, "NNZ_PER_ROW": 27},
    },
    "lj_force": {
        "params": {"NLOCAL": 4000, "MAXNEIGHS": 128, "PAD": 3},
    },
}

# Kernels requiring higher tolerance
HIGHER_TOLERANCE_KERNELS = {
    'srad': {'atol': 1e-2, 'rtol': 1e-3},       # 10 iterations of stencil
    'lavaMD': {'atol': 1e-6, 'rtol': 1e-6},      # double precision
    'gaussian': {'atol': 5.0, 'rtol': 0.05},     # pivotless elimination, error compounds
    'spmv': {'atol': 1e-4, 'rtol': 1e-4},        # float32 SpMV
    'lj_force': {'atol': 1e-3, 'rtol': 1e-3},    # float32 LJ force, conditional branches
}

# Torch dtype map for different C types
_TORCH_DTYPE_MAP = {
    'float': 'torch.float32',
    'double': 'torch.float64',
    'int': 'torch.int32',
}

# C type to ctypes/numpy mapping
_C_TYPE_MAP = {
    'float': ('c_float', 'float32'),
    'double': ('c_double', 'float64'),
    'int': ('c_int', 'int32'),
    'char': ('c_char', 'int8'),
    'short': ('c_short', 'int16'),
    'long': ('c_long', 'int64'),
}


# ============================================================================
# Kernel source and params
# ============================================================================

def get_kernel_source(kernel_name: str) -> Optional[str]:
    """Read the standalone kernel .c file."""
    kernel_file = os.path.join(REALWORLD_KERNELS_DIR, f"{kernel_name}.c")
    if not os.path.exists(kernel_file):
        return None
    with open(kernel_file, 'r') as f:
        return f.read()


def get_kernel_params(kernel_name: str) -> dict:
    """Get the size parameters for a kernel."""
    if kernel_name in REALWORLD_KERNELS:
        return REALWORLD_KERNELS[kernel_name]["params"]
    return {}


# ============================================================================
# Analysis loading with LLVM fallback
# ============================================================================





# ============================================================================
# Prompt building
# ============================================================================

def build_realworld_prompt(kernel_name: str, func_spec: dict) -> str:
    """Build the prompt for real-world kernel Triton generation."""
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
        mode_str = {'r': 'read-only', 'w': 'write-only', 'rw': 'read-write',
                     'temp': 'temporary scratch (read-write, not checked for correctness)'}[mode]
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
    for p in sorted(params.keys()):
        if p not in scalar_params:
            sig_parts.append(p)
    exact_sig = ", ".join(sig_parts)

    func_id = kernel_name

    # Load analysis results (skip if analysis disabled)
    analysis_text = ""
    if ENABLE_ANALYSIS:
        analysis = analyze_kernel(kernel_name, source, arrays, params,
                                  scalar_params, has_2d)
        analysis_text = "\n" + format_analysis_for_prompt(analysis)


    prompt = f"""I have a real-world application kernel (from Rodinia benchmark suite) that I want to implement in Triton for GPU acceleration.

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

## REQUIRED function signature (use EXACTLY these parameter names):
```python
def {func_id}_triton({exact_sig}):
    ...  # kernel computation
```

## CRITICAL: Triton Compilation Rules

**Pass FIXED dimension parameters as `tl.constexpr`** for best performance:
```python
# GOOD — N, M, BLOCK are constant across ALL kernel launches
def kernel(ptr, N: tl.constexpr, M: tl.constexpr, BLOCK: tl.constexpr):
    for i in range(N):  # compiler can unroll this
        ...
```

**NEVER use `tl.constexpr` for loop-varying parameters** (e.g., a timestep `t` that changes each Python iteration):
```python
# WRONG — causes N recompilations (one per t value)!
def kernel(ptr, t: tl.constexpr, N: tl.constexpr):
    ...
for t in range(N):
    kernel[(grid,)](ptr, t, N)  # recompiles for EACH t!

# CORRECT — t is a regular parameter, no recompilation
def kernel(ptr, t, N: tl.constexpr):
    ...
for t in range(N):
    kernel[(grid,)](ptr, t, N)  # same compiled kernel, different t
```

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

**NEVER use `mask.any()` — Triton tensors don't have `.any()`.** Use `tl.sum(mask) > 0` instead:
```python
# WRONG
if mask.any():  # AttributeError!

# CORRECT
if tl.sum(mask) > 0:
```

**Use `tl.sum(mask) > 0` to skip empty work in shrinking iteration spaces:**
When a CTA has no valid elements (mask all-False), skip the inner loop entirely.
Without this guard, CTAs waste time on useless masked loads/stores.

**Prefer vectorized access over scalar indexing inside @triton.jit kernel:**
```python
# WRONG — scalar indexing into a Triton tensor
for i in range(BLOCK_SIZE):
    val = tensor[i]

# CORRECT — vectorized load
mask = offsets < n_elements
vals = tl.load(ptr + offsets, mask=mask)
```
**Exception**: For variable-length inner loops (e.g., CSR SpMV row processing,
neighbor list traversal), scalar `for` loops with `tl.load`/`tl.store` on
**pointers** are correct and efficient. Store results directly:
```python
for k in range(row_start, row_end):
    sum_val += tl.load(vals_ptr + k) * tl.load(x_ptr + tl.load(cols_ptr + k))
tl.store(y_ptr + row, sum_val)  # Direct scalar store — OK!
```
**For neighbor list traversal**: Iterate using the ACTUAL neighbor count
(`numneigh[i]`), NOT the maximum capacity (`MAXNEIGHS`). Iterating `MAXNEIGHS`
wastes cycles processing empty/padded entries:
```python
num_neighbors = tl.load(numneigh_ptr + i)
for k in range(0, num_neighbors, BLOCK):  # NOT range(0, MAXNEIGHS, BLOCK)
```
Do NOT use `tl.where(arange == idx, val, vec)` to scatter results — this is
O(BLOCK_SIZE) per element vs O(1) for direct `tl.store`.

**NEVER use non-existent Triton functions:**
- Use Python operators: `a * b`, `a / b`, `a + b` (not `tl.mul`, `tl.div`, `tl.add`)
- Use `triton.cdiv()` in wrapper only

**NEVER use `break` or `continue` inside @triton.jit kernels** (unsupported AST). Use masks instead:
```python
# WRONG — break/continue cause "unsupported AST node type" error
for ii in range(BLOCK_SIZE):
    if condition: break  # ERROR!
    if condition: continue  # ERROR!

# CORRECT — use masks to skip invalid elements
offsets = tl.arange(0, BLOCK_SIZE)
mask = offsets < valid_count
vals = tl.load(ptr + offsets, mask=mask)
tl.store(out_ptr + offsets, vals, mask=mask)
```
**NEVER use Python lists inside @triton.jit kernels**
**Pass tensors directly to kernels, NOT data_ptr()**
**NEVER use chained comparisons (use separate comparisons with &)**

Provide ONLY the Python code, no additional explanation."""

    return prompt


# ============================================================================
# Array shape parsing with macro expansion
# ============================================================================

def _resolve_macros(source: str) -> dict:
    """Parse all #define macros from source and resolve them recursively."""
    macros = {}
    for m in re.finditer(r'#define\s+(\w+)\s+(.+)', source):
        name = m.group(1)
        expr = m.group(2).strip()
        # Remove trailing comments
        if '//' in expr:
            expr = expr[:expr.index('//')].strip()
        if '/*' in expr:
            expr = expr[:expr.index('/*')].strip()
        macros[name] = expr

    # Iteratively resolve macros (up to 10 passes for nested defines)
    for _ in range(10):
        changed = False
        for name, expr in list(macros.items()):
            new_expr = expr
            for other_name, other_val in macros.items():
                if other_name != name and other_name in new_expr:
                    # Only replace whole words
                    new_expr = re.sub(r'\b' + re.escape(other_name) + r'\b', f'({other_val})', new_expr)
            if new_expr != expr:
                macros[name] = new_expr
                changed = True
        if not changed:
            break

    return macros


def _eval_size_expr(expr: str, params: dict, macros: dict) -> Optional[int]:
    """Evaluate a size expression using params and resolved macros."""
    result_expr = expr
    # Substitute params first (they override macros)
    for p_name, p_val in params.items():
        result_expr = re.sub(r'\b' + re.escape(p_name) + r'\b', str(p_val), result_expr)
    # Substitute macros
    for m_name, m_val in macros.items():
        result_expr = re.sub(r'\b' + re.escape(m_name) + r'\b', f'({m_val})', result_expr)
    # Try to evaluate
    try:
        # Strip parentheses and clean up for eval
        result_expr = result_expr.replace('(', '(').replace(')', ')')
        val = eval(result_expr)
        return int(val)
    except Exception:
        return None


def _get_array_shape(kernel_name: str, arr_name: str, params: dict) -> Optional[list]:
    """Determine array shape from the kernel source with macro expansion."""
    kernel_file = os.path.join(REALWORLD_KERNELS_DIR, f"{kernel_name}.c")
    if not os.path.exists(kernel_file):
        return None

    with open(kernel_file, 'r') as f:
        source = f.read()

    macros = _resolve_macros(source)

    # Match array declaration: type arr_name[expr1][expr2]...;
    pattern = rf'(?:float|double|int)\s+{re.escape(arr_name)}\s*(\[[^\]]+\](?:\[[^\]]+\])*)\s*;'
    m = re.search(pattern, source)
    if not m:
        return None

    dims_str = m.group(1)
    # Extract each dimension expression (may contain * and macro names)
    dim_exprs = re.findall(r'\[([^\]]+)\]', dims_str)

    shape = []
    for d_expr in dim_exprs:
        d_expr = d_expr.strip()
        # Try direct param lookup
        if d_expr in params:
            shape.append(params[d_expr])
        elif d_expr.isdigit():
            shape.append(int(d_expr))
        else:
            val = _eval_size_expr(d_expr, params, macros)
            if val is not None:
                shape.append(val)
            else:
                shape.append(100)  # fallback
    return shape


def _get_array_c_type(kernel_name: str, arr_name: str) -> str:
    """Detect the C type of an array from the kernel source file."""
    kernel_file = os.path.join(REALWORLD_KERNELS_DIR, f"{kernel_name}.c")
    if not os.path.exists(kernel_file):
        return 'float'
    with open(kernel_file, 'r') as f:
        source = f.read()
    pattern = rf'(float|double|int|char|short|long)\s+{re.escape(arr_name)}\s*\['
    m = re.search(pattern, source)
    return m.group(1) if m else 'float'


def _get_scalar_c_type(kernel_name: str, scalar_name: str) -> str:
    """Detect the C type of a scalar from the kernel source file."""
    kernel_file = os.path.join(REALWORLD_KERNELS_DIR, f"{kernel_name}.c")
    if not os.path.exists(kernel_file):
        return 'float'
    with open(kernel_file, 'r') as f:
        source = f.read()
    pattern = rf'(float|double|int)\s+{re.escape(scalar_name)}\s*='
    m = re.search(pattern, source)
    return m.group(1) if m else 'float'


# ============================================================================
# Domain-specific array initialization
# ============================================================================

def _init_lavaMD_boxes(BOXES1D, MAX_NEIGHBORS, indent):
    """Generate box neighbor metadata for lavaMD."""
    NUMBER_BOXES = BOXES1D ** 3
    I = indent  # base indent
    lines = []
    lines.append(f"{I}NUMBER_BOXES = {BOXES1D}**3")
    lines.append(f"{I}box_nn = torch.zeros(NUMBER_BOXES, dtype=torch.int32, device='cuda')")
    lines.append(f"{I}box_nei = torch.zeros(NUMBER_BOXES * {MAX_NEIGHBORS}, dtype=torch.int32, device='cuda')")
    lines.append(f"{I}for l_idx in range(NUMBER_BOXES):")
    lines.append(f"{I}    bx = l_idx % {BOXES1D}")
    lines.append(f"{I}    by = (l_idx // {BOXES1D}) % {BOXES1D}")
    lines.append(f"{I}    bz = l_idx // ({BOXES1D} * {BOXES1D})")
    lines.append(f"{I}    nn = 0")
    lines.append(f"{I}    for dz in range(-1, 2):")
    lines.append(f"{I}        for dy in range(-1, 2):")
    lines.append(f"{I}            for dx in range(-1, 2):")
    lines.append(f"{I}                if dx == 0 and dy == 0 and dz == 0:")
    lines.append(f"{I}                    continue")
    lines.append(f"{I}                nx, ny, nz = bx + dx, by + dy, bz + dz")
    lines.append(f"{I}                if 0 <= nx < {BOXES1D} and 0 <= ny < {BOXES1D} and 0 <= nz < {BOXES1D}:")
    lines.append(f"{I}                    box_nei[l_idx * {MAX_NEIGHBORS} + nn] = nz * {BOXES1D}*{BOXES1D} + ny * {BOXES1D} + nx")
    lines.append(f"{I}                    nn += 1")
    lines.append(f"{I}    box_nn[l_idx] = nn")
    return lines


def _get_domain_array_inits(kernel_name: str, arrays: dict, params: dict, indent: str) -> Optional[list]:
    """Return domain-appropriate array init lines for kernels with mathematical preconditions."""
    if kernel_name == 'srad':
        ROWS = params.get('ROWS', 512)
        COLS = params.get('COLS', 512)
        SIZE_I = ROWS * COLS
        lines = []
        lines.append(f"{indent}# J: positive image values (exp of random)")
        lines.append(f"{indent}J = torch.exp(torch.randn({SIZE_I}, device='cuda', dtype=torch.float32).clamp(-3, 3))")
        lines.append(f"{indent}# Boundary index arrays (clamp-to-edge)")
        lines.append(f"{indent}iN = torch.arange({ROWS}, device='cuda', dtype=torch.int32)")
        lines.append(f"{indent}iN[0] = 0")
        lines.append(f"{indent}iN[1:] = torch.arange(0, {ROWS - 1}, device='cuda', dtype=torch.int32)")
        lines.append(f"{indent}iS = torch.arange({ROWS}, device='cuda', dtype=torch.int32)")
        lines.append(f"{indent}iS[:-1] = torch.arange(1, {ROWS}, device='cuda', dtype=torch.int32)")
        lines.append(f"{indent}iS[-1] = {ROWS - 1}")
        lines.append(f"{indent}jW = torch.arange({COLS}, device='cuda', dtype=torch.int32)")
        lines.append(f"{indent}jW[0] = 0")
        lines.append(f"{indent}jW[1:] = torch.arange(0, {COLS - 1}, device='cuda', dtype=torch.int32)")
        lines.append(f"{indent}jE = torch.arange({COLS}, device='cuda', dtype=torch.int32)")
        lines.append(f"{indent}jE[:-1] = torch.arange(1, {COLS}, device='cuda', dtype=torch.int32)")
        lines.append(f"{indent}jE[-1] = {COLS - 1}")
        lines.append(f"{indent}# Work arrays initialized to zero")
        lines.append(f"{indent}c = torch.zeros({SIZE_I}, device='cuda', dtype=torch.float32)")
        lines.append(f"{indent}dN = torch.zeros({SIZE_I}, device='cuda', dtype=torch.float32)")
        lines.append(f"{indent}dS = torch.zeros({SIZE_I}, device='cuda', dtype=torch.float32)")
        lines.append(f"{indent}dW = torch.zeros({SIZE_I}, device='cuda', dtype=torch.float32)")
        lines.append(f"{indent}dE = torch.zeros({SIZE_I}, device='cuda', dtype=torch.float32)")
        lines.append(f"{indent}lambda_val = 0.5")
        return lines

    if kernel_name == 'lavaMD':
        BOXES1D = params.get('BOXES1D', 4)
        NUMBER_PAR_PER_BOX = params.get('NUMBER_PAR_PER_BOX', 100)
        MAX_NEIGHBORS = params.get('MAX_NEIGHBORS', 26)
        NUMBER_BOXES = BOXES1D ** 3
        TOTAL = NUMBER_BOXES * NUMBER_PAR_PER_BOX
        lines = []
        lines.append(f"{indent}# Use double precision (fp = double in original)")
        lines.append(f"{indent}NUMBER_BOXES = {BOXES1D}**3")
        lines.append(f"{indent}TOTAL = NUMBER_BOXES * {NUMBER_PAR_PER_BOX}")
        lines.append(f"{indent}# Particle positions")
        lines.append(f"{indent}rv_x = torch.randn(TOTAL, device='cuda', dtype=torch.float64) * 10.0")
        lines.append(f"{indent}rv_y = torch.randn(TOTAL, device='cuda', dtype=torch.float64) * 10.0")
        lines.append(f"{indent}rv_z = torch.randn(TOTAL, device='cuda', dtype=torch.float64) * 10.0")
        lines.append(f"{indent}rv_v = rv_x**2 + rv_y**2 + rv_z**2  # precomputed r^2")
        lines.append(f"{indent}qv = torch.randn(TOTAL, device='cuda', dtype=torch.float64)")
        lines.append(f"{indent}fv_v = torch.zeros(TOTAL, device='cuda', dtype=torch.float64)")
        lines.append(f"{indent}fv_x = torch.zeros(TOTAL, device='cuda', dtype=torch.float64)")
        lines.append(f"{indent}fv_y = torch.zeros(TOTAL, device='cuda', dtype=torch.float64)")
        lines.append(f"{indent}fv_z = torch.zeros(TOTAL, device='cuda', dtype=torch.float64)")
        lines.append(f"{indent}# Box metadata: particle offset per box")
        lines.append(f"{indent}box_offset = torch.arange(0, TOTAL, {NUMBER_PAR_PER_BOX}, device='cuda', dtype=torch.int32)")
        lines.append(f"{indent}# Box neighbor computation")
        box_lines = _init_lavaMD_boxes(BOXES1D, MAX_NEIGHBORS, indent)
        lines.extend(box_lines)
        lines.append(f"{indent}alpha_val = 0.5")
        return lines

    if kernel_name == 'gaussian':
        N = params.get('N', 512)
        lines = []
        lines.append(f"{indent}# Diagonally dominant matrix for stability (no pivoting)")
        lines.append(f"{indent}a = torch.randn({N * N}, device='cuda', dtype=torch.float32)")
        lines.append(f"{indent}# Make diag dominant")
        lines.append(f"{indent}for i_idx in range({N}):")
        lines.append(f"{indent}    a[i_idx * {N} + i_idx] = abs(a[i_idx * {N} + i_idx].item()) + {N}")
        lines.append(f"{indent}m = torch.zeros({N * N}, device='cuda', dtype=torch.float32)")
        lines.append(f"{indent}b = torch.randn({N}, device='cuda', dtype=torch.float32)")
        lines.append(f"{indent}x = torch.zeros({N}, device='cuda', dtype=torch.float32)")
        return lines

    if kernel_name == 'spmv':
        NROWS = params.get('NROWS', 4096)
        NNZ_PER_ROW = params.get('NNZ_PER_ROW', 27)
        NNZ = NROWS * NNZ_PER_ROW
        lines = []
        lines.append(f"{indent}NROWS = {NROWS}")
        lines.append(f"{indent}NNZ_PER_ROW = {NNZ_PER_ROW}")
        lines.append(f"{indent}NNZ = NROWS * NNZ_PER_ROW")
        lines.append(f"{indent}NCOLS = NROWS")
        lines.append(f"{indent}# Build valid CSR structure: each row has exactly NNZ_PER_ROW entries")
        lines.append(f"{indent}row_offsets = torch.arange(0, (NROWS + 1) * NNZ_PER_ROW, NNZ_PER_ROW, device='cuda', dtype=torch.int32)")
        lines.append(f"{indent}# Column indices: centered around diagonal, clamped to [0, NCOLS-1]")
        lines.append(f"{indent}cols_list = []")
        lines.append(f"{indent}half_bw = NNZ_PER_ROW // 2")
        lines.append(f"{indent}for r in range(NROWS):")
        lines.append(f"{indent}    col_start = max(0, r - half_bw)")
        lines.append(f"{indent}    row_cols = list(range(col_start, min(col_start + NNZ_PER_ROW, NCOLS)))")
        lines.append(f"{indent}    while len(row_cols) < NNZ_PER_ROW:")
        lines.append(f"{indent}        row_cols.append(row_cols[-1])")
        lines.append(f"{indent}    cols_list.extend(row_cols[:NNZ_PER_ROW])")
        lines.append(f"{indent}cols = torch.tensor(cols_list, device='cuda', dtype=torch.int32)")
        lines.append(f"{indent}vals = torch.randn(NNZ, device='cuda', dtype=torch.float32) * 0.1")
        lines.append(f"{indent}x = torch.randn(NCOLS, device='cuda', dtype=torch.float32)")
        lines.append(f"{indent}y = torch.zeros(NROWS, device='cuda', dtype=torch.float32)")
        return lines

    if kernel_name == 'lj_force':
        NLOCAL = params.get('NLOCAL', 4000)
        MAXNEIGHS = params.get('MAXNEIGHS', 128)
        PAD = params.get('PAD', 3)
        lines = []
        lines.append(f"{indent}NLOCAL = {NLOCAL}")
        lines.append(f"{indent}MAXNEIGHS = {MAXNEIGHS}")
        lines.append(f"{indent}PAD = {PAD}")
        lines.append(f"{indent}# Atom positions on a lattice in box [0, L]^3, spaced for ~60 neighbors")
        lines.append(f"{indent}# Lattice spacing 2.0, cutoff 4.0 → each atom sees ~60 neighbors")
        lines.append(f"{indent}import numpy as np_init")
        lines.append(f"{indent}lattice_n = int(round(NLOCAL ** (1.0/3.0))) + 1")
        lines.append(f"{indent}spacing = 2.0")
        lines.append(f"{indent}coords = []")
        lines.append(f"{indent}for ix in range(lattice_n):")
        lines.append(f"{indent}    for iy in range(lattice_n):")
        lines.append(f"{indent}        for iz in range(lattice_n):")
        lines.append(f"{indent}            if len(coords) >= NLOCAL:")
        lines.append(f"{indent}                break")
        lines.append(f"{indent}            # Small random perturbation")
        lines.append(f"{indent}            coords.append([ix*spacing + np_init.random.uniform(-0.2, 0.2),")
        lines.append(f"{indent}                           iy*spacing + np_init.random.uniform(-0.2, 0.2),")
        lines.append(f"{indent}                           iz*spacing + np_init.random.uniform(-0.2, 0.2)])")
        lines.append(f"{indent}        if len(coords) >= NLOCAL:")
        lines.append(f"{indent}            break")
        lines.append(f"{indent}    if len(coords) >= NLOCAL:")
        lines.append(f"{indent}        break")
        lines.append(f"{indent}coords = np_init.array(coords[:NLOCAL], dtype=np_init.float32)")
        lines.append(f"{indent}pos = torch.from_numpy(coords.flatten()).to('cuda')")
        lines.append(f"{indent}f = torch.zeros(NLOCAL * PAD, device='cuda', dtype=torch.float32)")
        lines.append(f"{indent}# Build neighbor lists using scipy KDTree for efficiency")
        lines.append(f"{indent}from scipy.spatial import cKDTree")
        lines.append(f"{indent}tree = cKDTree(coords)")
        lines.append(f"{indent}cutoff = 4.0")
        lines.append(f"{indent}numneigh_list = []")
        lines.append(f"{indent}neighbors_flat = []")
        lines.append(f"{indent}for i_atom in range(NLOCAL):")
        lines.append(f"{indent}    nlist = tree.query_ball_point(coords[i_atom], cutoff)")
        lines.append(f"{indent}    nlist = [j for j in nlist if j != i_atom]")
        lines.append(f"{indent}    if len(nlist) > MAXNEIGHS:")
        lines.append(f"{indent}        nlist = nlist[:MAXNEIGHS]")
        lines.append(f"{indent}    numneigh_list.append(len(nlist))")
        lines.append(f"{indent}    padded = nlist + [0] * (MAXNEIGHS - len(nlist))")
        lines.append(f"{indent}    neighbors_flat.extend(padded)")
        lines.append(f"{indent}numneigh = torch.tensor(numneigh_list, device='cuda', dtype=torch.int32)")
        lines.append(f"{indent}neighbors = torch.tensor(neighbors_flat, device='cuda', dtype=torch.int32)")
        lines.append(f"{indent}cutforcesq_val = 16.0  # cutoff^2")
        lines.append(f"{indent}sigma6_val = 1.0")
        lines.append(f"{indent}epsilon_val = 1.0")
        return lines

    return None


# ============================================================================
# Test generation helpers
# ============================================================================

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
                    lines.append(f"    src_{arr_name} = np.ascontiguousarray({arr_name}_c.astype(np.{np_dt}), dtype=np.{np_dt})")
                else:
                    lines.append(f"    src_{arr_name} = np.ascontiguousarray({arr_name}_c, dtype=np.float32)")
                lines.append(f"    ctypes.memmove(c_arr_{arr_name}, src_{arr_name}.ctypes.data, src_{arr_name}.nbytes)")
    return "\n".join(lines) if lines else "    pass"


def _gen_ctypes_scalar_setup(kernel_name: str, scalar_params: dict, params: dict) -> str:
    """Generate ctypes code to set global scalars in the .so."""
    lines = []
    for sp_name in sorted(scalar_params.keys()):
        c_type = _get_scalar_c_type(kernel_name, sp_name)
        ct_type = {'float': 'c_float', 'double': 'c_double', 'int': 'c_int'}.get(c_type, 'c_float')
        lines.append(f"    ctypes.{ct_type}.in_dll(lib, '{sp_name}').value = float({sp_name})")
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


def generate_correctness_test(kernel_name: str, func_spec: dict, attempt: int = 1) -> str:
    """Generate correctness test for a real-world kernel."""
    arrays = func_spec['arrays']
    scalar_params = func_spec.get('scalar_params', {})
    params = get_kernel_params(kernel_name)

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
                shape = _get_array_shape(kernel_name, arr_name, params)
                if shape:
                    shape_str = ", ".join(str(s) for s in shape)
                    c_type = _get_array_c_type(kernel_name, arr_name)
                    torch_dt = _TORCH_DTYPE_MAP.get(c_type, 'torch.float32')
                    array_inits.append(
                        f"            {arr_name} = torch.randn({shape_str}, device='cuda', dtype={torch_dt})"
                    )
                else:
                    first_size = list(params.values())[0] if params else 100
                    array_inits.append(
                        f"            {arr_name} = torch.randn({first_size}, device='cuda', dtype=torch.float32)"
                    )

    # Scalar parameter initialization
    for sp_name in sorted(scalar_params.keys()):
        if sp_name == 'lambda_val':
            array_inits.append(f"            {sp_name} = 0.5")
        elif sp_name == 'alpha_val':
            array_inits.append(f"            {sp_name} = 0.5")
        else:
            array_inits.append(f"            {sp_name} = 1.0")

    # Dimension parameters
    for p_name, p_val in sorted(params.items()):
        if p_name not in scalar_params:
            array_inits.append(f"            {p_name} = {p_val}")

    array_init_str = "\n".join(array_inits)

    # Build argument lists
    array_names = sorted([a for a, m in arrays.items() if m in ['r', 'rw', 'w', 'temp']])
    output_arrays = sorted([a for a, m in arrays.items() if m in ['rw', 'w']])
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

    # C reference call args
    c_args = [f"{a}_c" for a in array_names] + scalar_names + dim_names
    c_call_str = ", ".join(c_args)

    # Triton call args
    tr_args = [f"{a}_tr" for a in array_names] + scalar_names + dim_names
    tr_call_str = ", ".join(tr_args)

    func_id = kernel_name

    llm_subdir = "llm_triton" if ENABLE_ANALYSIS else "llm_triton_no_analysis"
    attempt_file_abs = str((Path(OUTPUT_DIR) / llm_subdir / kernel_name / f"attempt{attempt}.py").resolve())
    import_block = (
        f"import importlib.util\n"
        f"    _spec = importlib.util.spec_from_file_location(\'_kmod\', \'{attempt_file_abs}\')\n"
        f"    _mod = importlib.util.module_from_spec(_spec)\n"
        f"    _spec.loader.exec_module(_mod)\n"
        f"    {func_id}_triton = _mod.{func_id}_triton"
    )

    test_code = f'''#!/usr/bin/env python3
"""Correctness test for {kernel_name} (Real-World) - attempt {attempt}"""
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
C_LIB_PATH = Path("/home/qinxiao/workspace/compiler-guided-triton-gen/pipeline/c_reference") / "realworld_libs" / "lib{kernel_name}.so"
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


# ============================================================================
# Benchmarking
# ============================================================================

def generate_benchmark_test(kernel_name: str, func_spec: dict, attempt: int = 1) -> str:
    """Generate performance benchmark script for a real-world kernel."""
    arrays = func_spec['arrays']
    scalar_params = func_spec.get('scalar_params', {})
    params = get_kernel_params(kernel_name)

    func_id = kernel_name

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
                c_type = _get_array_c_type(kernel_name, arr_name)
                torch_dt = _TORCH_DTYPE_MAP.get(c_type, 'torch.float32')
                array_inits.append(f"    {arr_name} = torch.randn({shape_str}, device='cuda', dtype={torch_dt})")

    for sp_name in sorted(scalar_params.keys()):
        if sp_name == 'lambda_val':
            array_inits.append(f"    {sp_name} = 0.5")
        elif sp_name == 'alpha_val':
            array_inits.append(f"    {sp_name} = 0.5")
        else:
            array_inits.append(f"    {sp_name} = 1.0")

    for p_name, p_val in sorted(params.items()):
        if p_name not in scalar_params:
            array_inits.append(f"    {p_name} = {p_val}")

    array_init_str = "\n".join(array_inits)

    array_names = sorted([a for a, m in arrays.items() if m in ['r', 'rw', 'w', 'temp']])
    scalar_names = sorted(scalar_params.keys())
    dim_names = sorted([p for p in params.keys() if p not in scalar_params])

    c_args = [f"{a}_c" for a in array_names] + scalar_names + dim_names
    c_call_str = ", ".join(c_args)

    tr_args = [f"{a}_tr" for a in array_names] + scalar_names + dim_names
    tr_call_str = ", ".join(tr_args)

    llm_subdir = "llm_triton" if ENABLE_ANALYSIS else "llm_triton_no_analysis"
    attempt_file_abs = str((Path(OUTPUT_DIR) / llm_subdir / kernel_name / f"attempt{attempt}.py").resolve())
    import_block = (
        f"import importlib.util\n"
        f"    _spec = importlib.util.spec_from_file_location(\'_kmod\', \'{attempt_file_abs}\')\n"
        f"    _mod = importlib.util.module_from_spec(_spec)\n"
        f"    _spec.loader.exec_module(_mod)\n"
        f"    {func_id}_triton = _mod.{func_id}_triton"
    )

    benchmark_code = f'''#!/usr/bin/env python3
"""Performance Benchmark for {kernel_name} (Real-World)"""
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

C_LIB_PATH = Path("/home/qinxiao/workspace/compiler-guided-triton-gen/pipeline/c_reference") / "realworld_libs" / "lib{kernel_name}.so"

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
    """Generate initial Triton implementation for a real-world kernel."""
    prompt = build_realworld_prompt(kernel_name, func_spec)

    print(f"  Generating Triton code (attempt 1/{MAX_ATTEMPTS})...")

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}]
    )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_response = f"""# LLM-Generated Triton Implementation for {kernel_name} (Real-World)
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
# C compilation
# ============================================================================

def compile_c_reference(kernel_name: str, params: dict) -> Optional[str]:
    """Compile C kernel with parameter overrides. Returns .so path or None."""
    import tempfile
    c_file = Path(REALWORLD_KERNELS_DIR) / f"{kernel_name}.c"
    if not c_file.exists():
        return None

    lib_dir = Path("c_reference") / "realworld_libs"
    lib_dir.mkdir(parents=True, exist_ok=True)
    so_file = lib_dir / f"lib{kernel_name}.so"

    # Strip source-level #define lines for params we want to override
    source = c_file.read_text()
    param_names = set(params.keys())
    lines = source.split('\n')
    filtered = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#define'):
            parts = stripped.split()
            if len(parts) >= 2 and parts[1] in param_names:
                continue  # skip — will use -D flag instead
        filtered.append(line)

    # Write modified source to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as tmp:
        tmp.write('\n'.join(filtered))
        tmp_path = tmp.name

    d_flags = [f"-D{k}={v}" for k, v in params.items()]
    result = subprocess.run(
        ["clang", "-shared", "-fPIC", "-O2"] + d_flags +
        ["-o", str(so_file), tmp_path, "-lm"],
        capture_output=True, text=True, timeout=30
    )
    os.unlink(tmp_path)

    if result.returncode != 0:
        print(f"  ERROR compiling {kernel_name}: {result.stderr}")
        return None
    return str(so_file)


# ============================================================================
# Main pipeline
# ============================================================================

def process_kernel(kernel_name: str, func_spec: dict) -> dict:
    """Process a single real-world kernel with retry logic and speedup-based retry."""
    print(f"\n{'=' * 70}")
    print(f"Processing: {kernel_name}")
    print(f"  Arrays: {list(func_spec['arrays'].keys())}")
    print(f"  Params: {list(get_kernel_params(kernel_name).keys())}")
    print(f"{'=' * 70}")

    base_dir = Path(OUTPUT_DIR)
    llm_dir = base_dir / ("llm_triton" if ENABLE_ANALYSIS else "llm_triton_no_analysis")
    func_dir = llm_dir / kernel_name
    raw_dir = llm_dir / "raw_responses" / kernel_name
    test_dir = Path("../results/realworld/my_realworld_tests") / kernel_name

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
    reset_after = 5

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

                    if speedup < 0.1 and attempt < MAX_ATTEMPTS:
                        print(f"  Speedup too low ({speedup:.2f}x < 0.1x). Retrying for better parallelization...")
                        error_info = {
                            'type': 'low_speedup',
                            'speedup': speedup,
                            'message': f'Code is correct but speedup is only {speedup:.2f}x. Needs better parallelization.'
                        }
                        results["test_passed"] = False
                        continue
                else:
                    print(f"  Benchmark failed or timed out")
                    if best_result and best_result.get("benchmark"):
                        results["benchmark"] = best_result["benchmark"]
                    else:
                        results["benchmark"] = None

                results["final_attempt"] = attempt
                return results
            else:
                print(f"  FAILED: {error_info.get('type', 'unknown')} - {error_info.get('message', '')[:100]}")
                results["final_error"] = error_info

        except Exception as e:
            print(f"  Exception on attempt {attempt}: {e}")
            error_info = {'type': 'exception', 'message': str(e)}
            results["final_error"] = error_info

    if best_result is not None:
        print(f"  Returning best result from attempt {best_attempt} with {best_speedup:.2f}x speedup")
        best_result["attempts"] = results["attempts"]
        best_result["test_passed"] = True
        return best_result

    return results


def run_benchmark_only(kernel_name: str, func_spec: dict) -> Optional[dict]:
    """Re-run benchmarks for an already-generated kernel."""
    base_dir = Path(OUTPUT_DIR)
    llm_dir = base_dir / ("llm_triton" if ENABLE_ANALYSIS else "llm_triton_no_analysis")
    func_dir = llm_dir / kernel_name
    test_dir = Path("../results/realworld/my_realworld_tests") / kernel_name

    if not func_dir.exists():
        print(f"  No generated code found for {kernel_name}")
        return None

    # Find the latest attempt
    attempts = sorted(func_dir.glob("attempt*.py"))
    if not attempts:
        print(f"  No attempt files found for {kernel_name}")
        return None

    # Find the passing attempt (try latest first)
    for attempt_file in reversed(attempts):
        attempt_num = int(re.search(r'attempt(\d+)', attempt_file.name).group(1))
        test_dir.mkdir(exist_ok=True, parents=True)

        bench_file = test_dir / f"benchmark_{kernel_name}.py"
        bench_code = generate_benchmark_test(kernel_name, func_spec, attempt_num)
        with open(bench_file, 'w') as f:
            f.write(bench_code)

        benchmark_results = run_benchmark(kernel_name, bench_file)
        if benchmark_results and benchmark_results.get('speedup'):
            return benchmark_results

    return None


def main():
    """Main real-world pipeline."""
    global ENABLE_ANALYSIS

    benchmark_only = False

    # Check for flags
    if '--no-analysis' in sys.argv:
        sys.argv.remove('--no-analysis')
        ENABLE_ANALYSIS = False

    if '--benchmark' in sys.argv:
        sys.argv.remove('--benchmark')
        benchmark_only = True

    analysis_mode = "WITH analysis" if ENABLE_ANALYSIS else "WITHOUT analysis (ablation)"
    print("=" * 70)
    print("Real-World Application Kernels Generation and Testing Pipeline")
    print(f"Total kernels available: {len(REALWORLD_FUNCTIONS)}")
    print(f"Max attempts per kernel: {MAX_ATTEMPTS}")
    print(f"Analysis: {analysis_mode}")
    if benchmark_only:
        print("Mode: BENCHMARK ONLY (re-run benchmarks for existing code)")
    print("=" * 70)

    if not benchmark_only and not client:
        print("ERROR: ANTHROPIC_API_KEY not set!")
        sys.exit(1)

    # Check if specific kernels requested
    if len(sys.argv) > 1:
        kernel_names = sys.argv[1:]
        kernels_to_process = {}
        for k in kernel_names:
            if k in REALWORLD_FUNCTIONS:
                kernels_to_process[k] = REALWORLD_FUNCTIONS[k]
            else:
                print(f"Warning: Kernel not found: {k}")
        print(f"Processing {len(kernels_to_process)} specific kernels: {list(kernels_to_process.keys())}")
    else:
        kernels_to_process = REALWORLD_FUNCTIONS
        print(f"Processing ALL {len(kernels_to_process)} kernels")

    if not kernels_to_process:
        print("No valid kernels to process!")
        return

    # Compile C references
    print("\nCompiling C reference libraries...")
    for kernel_name in kernels_to_process:
        params = get_kernel_params(kernel_name)
        so_path = compile_c_reference(kernel_name, params)
        if so_path:
            print(f"  Compiled: {so_path}")

    all_results = {}

    if benchmark_only:
        for i, (kernel_name, func_spec) in enumerate(kernels_to_process.items(), 1):
            print(f"\n[{i}/{len(kernels_to_process)}] Benchmarking: {kernel_name}")
            bench = run_benchmark_only(kernel_name, func_spec)
            if bench:
                all_results[kernel_name] = {
                    "triton_generated": True,
                    "test_passed": True,
                    "attempts": 0,
                    "benchmark": bench,
                    "final_error": None,
                }
                sp = bench.get('speedup', 0) or 0
                print(f"  Speedup: {sp:.2f}x")
            else:
                all_results[kernel_name] = {
                    "triton_generated": False,
                    "test_passed": False,
                    "attempts": 0,
                    "benchmark": None,
                    "final_error": None,
                }
    else:
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
    print(f"Tests passed: {passed}/{total} ({100*passed/total:.1f}%)" if total > 0 else "")
    if not benchmark_only:
        print(f"  - Passed on first try: {first_try}")
        print(f"  - Passed after retry: {passed - first_try}")

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

    # Save results to JSON
    import json
    results_filename = "results.json" if ENABLE_ANALYSIS else "results_no_analysis.json"
    results_file = Path(OUTPUT_DIR) / results_filename
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    existing_results = {}
    if results_file.exists():
        with open(results_file) as f:
            existing_results = json.load(f)

    merged = {**existing_results, **{k: {kk: vv for kk, vv in v.items() if kk != 'final_error' or vv is None or isinstance(vv, (str, dict))}
                    for k, v in all_results.items()}}

    with open(results_file, 'w') as f:
        json.dump(merged, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")

    # Save benchmark results
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
