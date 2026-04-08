#!/usr/bin/env python3
"""
Unified kernel analysis module.

Runs all analysis passes on a C kernel and produces a single structured
KernelAnalysis object. This replaces the pattern of calling 16 separate
modules and assembling their outputs ad-hoc in the prompt builder.

The output is a JSON-serializable dict with a fixed schema, regardless
of what patterns the kernel exhibits. The prompt builder simply renders
this dict — no pattern-specific logic needed.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

# Add analysis dir to path
ANALYSIS_DIR = Path(__file__).parent
sys.path.insert(0, str(ANALYSIS_DIR))

# Import individual analysis modules (graceful fallback if any missing)
def _try_import(module_name, func_name):
    try:
        mod = __import__(module_name)
        return getattr(mod, func_name)
    except (ImportError, AttributeError):
        return None

analyze_war = _try_import('compute_war_dependences', 'analyze_kernel_war')
analyze_parallel = _try_import('compute_parallel_dims', 'analyze_kernel_parallelization')
analyze_scalar_exp = _try_import('compute_scalar_expansion', 'analyze_kernel_scalar_expansion')
analyze_reduction = _try_import('compute_reduction_type', 'analyze_kernel_reduction')
analyze_gpu_strategy = _try_import('compute_gpu_parallelization_strategy', 'analyze_kernel_gpu_strategy')
build_reduction_instr = _try_import('compute_reduction_type', 'build_reduction_instructions')
build_gpu_strategy_instr = _try_import('compute_gpu_parallelization_strategy', 'build_gpu_strategy_instructions')

# LLVM fallback
try:
    from llvm_fallback_adapters import (
        try_with_llvm_fallback, enhance_war_with_llvm_vectors
    )
    HAS_LLVM_FALLBACK = True
except ImportError:
    HAS_LLVM_FALLBACK = False


def analyze_kernel(kernel_name: str, kernel_source: str, arrays: dict,
                   params: dict, scalar_params: dict = None,
                   has_2d_arrays: bool = False) -> dict:
    """
    Run all analysis passes and return a unified KernelAnalysis dict.

    Args:
        kernel_name: Name of the kernel (e.g., 'gemm', 'jacobi_1d')
        kernel_source: C source code of the kernel
        arrays: Dict of {array_name: mode} where mode is 'r'/'w'/'rw'/'temp'
        params: Dict of {param_name: value} for dimensions
        scalar_params: Dict of {param_name: value} for scalar parameters
        has_2d_arrays: Whether the kernel uses 2D arrays

    Returns:
        A dict with the following top-level keys:
        - kernel_name, source, arrays, params, scalar_params
        - dimensions: list of loop dimension info
        - dependencies: WAR and other dependence info
        - parallelism: which dims are safe to parallelize and why
        - reductions: detected reduction patterns
        - memory_access: access pattern info (triangular, stencil, etc.)
        - strategy: recommended high-level GPU strategy
    """
    scalar_params = scalar_params or {}

    analysis = {
        "kernel_name": kernel_name,
        "source": kernel_source,
        "arrays": arrays,
        "params": params,
        "scalar_params": scalar_params,
        "dimensions": [],
        "dependencies": {
            "war": [],
            "war_safe": True,
            "arrays_with_war": [],
            "loop_level_scoping": None,
        },
        "parallelism": {
            "dims": [],
            "options": [],
            "is_triangular": False,
            "triangular_info": None,
            "has_2d_arrays": has_2d_arrays,
            "fully_sequential": False,
        },
        "reductions": {
            "detected": False,
            "type": None,
            "details": None,
        },
        "scalar_expansion": {
            "needed": False,
            "scalars": [],
        },
        "memory_access": {
            "is_stencil": False,
            "is_triangular": False,
            "has_cross_phase": False,
            "phases": [],
        },
        "strategy": {
            "grid_recommendation": None,
            "notes": [],
        },
    }

    # --- Resolve kernel file path ---
    kernel_file = None
    for subdir in ["kernels", "kernels_polybench", "kernels_realworld"]:
        candidate = ANALYSIS_DIR / subdir / f"{kernel_name}.c"
        if candidate.exists():
            kernel_file = str(candidate)
            break

    # --- WAR Dependence Analysis ---
    if analyze_war and kernel_file:
        try:
            war_result = analyze_war(kernel_file)
            if HAS_LLVM_FALLBACK:
                war_result = enhance_war_with_llvm_vectors(kernel_name, war_result)
            if war_result:
                analysis["dependencies"]["war_safe"] = war_result.get('parallelization_safe', True)
                analysis["dependencies"]["arrays_with_war"] = war_result.get('arrays_needing_copy', [])
                analysis["dependencies"]["loop_level_scoping"] = war_result.get('loop_level_scoping')
                for dep in war_result.get('war_dependencies', []):
                    analysis["dependencies"]["war"].append({
                        "description": dep.get('description', ''),
                        "read_array": dep.get('read_array'),
                        "write_array": dep.get('write_array'),
                    })
        except Exception as e:
            analysis["strategy"]["notes"].append(f"WAR analysis failed: {e}")

    # --- Parallelization Analysis ---
    if analyze_parallel and kernel_file:
        try:
            par_result = analyze_parallel(kernel_name, kernel_file)
            if par_result:
                if par_result.get('has_2d_arrays') is None:
                    par_result['has_2d_arrays'] = has_2d_arrays
                analysis["parallelism"]["dims"] = par_result.get('dims', [])
                analysis["parallelism"]["is_triangular"] = par_result.get('is_triangular', False)
                analysis["parallelism"]["triangular_info"] = par_result.get('triangular_info')
                analysis["parallelism"]["has_2d_arrays"] = par_result.get('has_2d_arrays', has_2d_arrays)

                for opt in par_result.get('options', []):
                    analysis["parallelism"]["options"].append({
                        "parallel_dim": opt.get('parallel_dim'),
                        "sequential_dim": opt.get('sequential_dim'),
                        "valid": opt.get('valid', False),
                        "issues": opt.get('issues', []),
                    })

                # Check if fully sequential
                valid_opts = [o for o in analysis["parallelism"]["options"] if o["valid"]]
                if not valid_opts:
                    analysis["parallelism"]["fully_sequential"] = True
        except Exception as e:
            analysis["strategy"]["notes"].append(f"Parallelization analysis failed: {e}")

    # --- Scalar Expansion ---
    if analyze_scalar_exp and kernel_file:
        try:
            se_result = analyze_scalar_exp(kernel_file)
            if se_result and se_result.get('needs_expansion'):
                analysis["scalar_expansion"]["needed"] = True
                analysis["scalar_expansion"]["scalars"] = se_result.get('scalars_to_expand', [])
        except Exception:
            pass

    # --- Reduction Detection ---
    if analyze_reduction and kernel_file:
        try:
            red_result = analyze_reduction(kernel_name, kernel_file)
            if red_result and red_result.get('has_reduction'):
                analysis["reductions"]["detected"] = True
                analysis["reductions"]["type"] = red_result.get('reduction_type')
                analysis["reductions"]["details"] = red_result.get('details')
        except Exception:
            pass

    # --- GPU Strategy ---
    if analyze_gpu_strategy and kernel_file:
        try:
            strat_result = analyze_gpu_strategy(kernel_name, kernel_file)
            if strat_result:
                analysis["strategy"]["grid_recommendation"] = strat_result.get('grid_recommendation')
                if strat_result.get('notes'):
                    analysis["strategy"]["notes"].extend(strat_result['notes'])
        except Exception:
            pass

    # --- Derive memory access patterns ---
    _derive_memory_patterns(analysis)

    return analysis


def _derive_memory_patterns(analysis: dict):
    """Derive high-level memory access patterns from the raw analysis."""
    opts = analysis["parallelism"]["options"]
    arrays_war = analysis["dependencies"]["arrays_with_war"]

    # Detect cross-phase pattern
    for opt in opts:
        if not opt["valid"] and any("Cross-phase" in iss for iss in opt["issues"]):
            analysis["memory_access"]["has_cross_phase"] = True
            break

    # Detect stencil pattern: 1 valid dim, cross-phase, not 2D
    valid_opts = [o for o in opts if o["valid"]]
    if (len(valid_opts) == 1
            and not analysis["parallelism"]["has_2d_arrays"]
            and analysis["memory_access"]["has_cross_phase"]):
        analysis["memory_access"]["is_stencil"] = True

    # Triangular
    if analysis["parallelism"]["is_triangular"]:
        analysis["memory_access"]["is_triangular"] = True


def format_analysis_for_prompt(analysis: dict) -> str:
    """
    Convert a KernelAnalysis dict into a structured text section for the LLM prompt.

    This is the ONLY place that converts analysis → prompt text.
    It is pattern-agnostic: it renders whatever the analysis contains.
    """
    sections = []

    # --- Dependencies ---
    if not analysis["dependencies"]["war_safe"]:
        dep_lines = ["## Data Dependencies", ""]
        dep_lines.append("This kernel has Write-After-Read (WAR) dependencies:")
        for arr in analysis["dependencies"]["arrays_with_war"]:
            dep_lines.append(f"- Array `{arr}` is both read and written")

        scoping = analysis["dependencies"]["loop_level_scoping"]
        if scoping:
            loop_vars = list(scoping.keys()) if isinstance(scoping, dict) else []
            for arr in analysis["dependencies"]["arrays_with_war"]:
                if arr in scoping:
                    s = scoping[arr]
                    carried = s.get('carried_by_loops', [])
                    safe = s.get('safe_to_parallelize_loops', [])
                    dep_lines.append(f"  - WAR on `{arr}` carried by loop(s): {', '.join(carried)}")
                    if safe:
                        dep_lines.append(f"  - Safe to parallelize: {', '.join(safe)}")
        else:
            for dep in analysis["dependencies"]["war"][:5]:
                dep_lines.append(f"- {dep['description']}")

        dep_lines.append("")
        dep_lines.append("Implication: if you parallelize a dimension that carries a WAR dependency,")
        dep_lines.append("you must either clone the array before the parallel region or use separate kernels.")
        sections.append("\n".join(dep_lines))

    # --- Parallelism ---
    opts = analysis["parallelism"]["options"]
    if opts:
        par_lines = ["## Parallelization Analysis", ""]
        par_lines.append(f"Loop dimensions: {analysis['parallelism']['dims']}")

        if analysis["parallelism"]["is_triangular"]:
            tri = analysis["parallelism"]["triangular_info"]
            if tri:
                par_lines.append(f"Triangular loop bounds: {tri.get('smaller', '?')} < {tri.get('larger', '?')}")

        par_lines.append("")
        valid_opts = [o for o in opts if o["valid"]]
        invalid_opts = [o for o in opts if not o["valid"]]

        for opt in opts:
            status = "SAFE" if opt["valid"] else "UNSAFE"
            par_lines.append(f"- Parallelize `{opt['parallel_dim']}`, sequential `{opt['sequential_dim']}`: **{status}**")
            for issue in opt["issues"]:
                par_lines.append(f"  - {issue}")

        if analysis["parallelism"]["fully_sequential"]:
            par_lines.append("")
            par_lines.append("**No dimension can be safely parallelized.** Use `grid=(1,)` with sequential loops.")

        sections.append("\n".join(par_lines))

    # --- Reductions ---
    if analysis["reductions"]["detected"]:
        red_lines = ["## Reduction Pattern", ""]
        red_lines.append(f"Reduction type: {analysis['reductions']['type']}")
        if analysis["reductions"]["details"]:
            red_lines.append(f"Details: {analysis['reductions']['details']}")
        red_lines.append("")
        red_lines.append("Use `tl.sum()`, `tl.max()`, or `tl.dot()` for the reduction dimension.")
        sections.append("\n".join(red_lines))

    # --- Scalar Expansion ---
    if analysis["scalar_expansion"]["needed"]:
        se_lines = ["## Scalar Expansion Needed", ""]
        for s in analysis["scalar_expansion"]["scalars"]:
            se_lines.append(f"- Scalar `{s}` must be expanded to a per-thread register (not shared)")
        sections.append("\n".join(se_lines))

    # --- Memory Access Patterns ---
    mem = analysis["memory_access"]
    if mem["is_stencil"] or mem["is_triangular"] or mem["has_cross_phase"]:
        mem_lines = ["## Memory Access Patterns", ""]
        if mem["is_stencil"]:
            mem_lines.append("- **Stencil pattern detected**: reads neighboring elements of the same array")
            mem_lines.append("  Use `tl.debug_barrier()` between phases if using a single CTA (`grid=(1,)`)")
        if mem["is_triangular"]:
            mem_lines.append("- **Triangular access pattern**: loop bounds depend on outer loop variable")
            mem_lines.append("  Consider `grid=(N,)` with one program per row, masking unused columns")
        if mem["has_cross_phase"]:
            mem_lines.append("- **Cross-phase dependency**: one phase writes what another phase reads")
            mem_lines.append("  Either use separate kernel launches or `tl.debug_barrier()` between phases")
        sections.append("\n".join(mem_lines))

    # --- Strategy Notes ---
    if analysis["strategy"]["notes"]:
        note_lines = ["## Additional Notes", ""]
        for note in analysis["strategy"]["notes"]:
            note_lines.append(f"- {note}")
        sections.append("\n".join(note_lines))

    if not sections:
        return "## Compiler Analysis\n\nNo significant parallelization constraints detected. Standard parallelization should work."

    return "\n\n".join(sections)


def analyze_and_format(kernel_name: str, kernel_source: str, arrays: dict,
                       params: dict, scalar_params: dict = None,
                       has_2d_arrays: bool = False) -> str:
    """Convenience: analyze + format in one call."""
    analysis = analyze_kernel(kernel_name, kernel_source, arrays, params,
                              scalar_params, has_2d_arrays)
    return format_analysis_for_prompt(analysis)


# CLI for testing
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python kernel_analysis.py <kernel_name>")
        sys.exit(1)

    kernel_name = sys.argv[1]
    kernel_file = ANALYSIS_DIR / "kernels" / f"{kernel_name}.c"
    if not kernel_file.exists():
        kernel_file = ANALYSIS_DIR / "kernels_polybench" / f"{kernel_name}.c"
    if not kernel_file.exists():
        print(f"Kernel file not found: {kernel_name}")
        sys.exit(1)

    source = kernel_file.read_text()

    # Minimal test with dummy arrays/params
    analysis = analyze_kernel(kernel_name, source,
                              arrays={"A": "rw"}, params={"N": 100})

    print("=== Raw Analysis (JSON) ===")
    # Remove source from output for readability
    analysis_print = {k: v for k, v in analysis.items() if k != "source"}
    print(json.dumps(analysis_print, indent=2, default=str))

    print("\n=== Formatted for Prompt ===")
    print(format_analysis_for_prompt(analysis))
