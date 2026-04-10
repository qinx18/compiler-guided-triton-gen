#!/usr/bin/env python3
"""
Unified kernel analysis module.

Calls PET once, parses the YAML output once, extracts all accesses once,
then derives all properties (parallelism, WAR, reductions, memory patterns)
from that single representation. Replaces the 5 separate analysis modules.

Usage:
    from kernel_analysis import analyze_kernel, format_analysis_for_prompt

    analysis = analyze_kernel("gemm", source, arrays={"C":"rw","A":"r","B":"r"},
                              params={"NI":60,"NJ":70,"NK":80})
    prompt_text = format_analysis_for_prompt(analysis)
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml

try:
    import islpy as isl
    HAS_ISL = True
except ImportError:
    HAS_ISL = False

ANALYSIS_DIR = Path(__file__).parent
PET_PATH = "/home/qinxiao/workspace/pet/pet"
PET_LD_PATH = "/home/qinxiao/workspace/pet/isl/.libs"


# ============================================================================
# PET invocation and YAML parsing (done ONCE per kernel)
# ============================================================================

def _run_pet(kernel_file: str) -> Optional[dict]:
    """Run PET on a kernel file, parse YAML, return structured dict."""
    if not os.path.exists(PET_PATH):
        return None
    if not os.path.exists(kernel_file):
        return None

    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = PET_LD_PATH + ':' + env.get('LD_LIBRARY_PATH', '')

    try:
        result = subprocess.run(
            [PET_PATH, kernel_file],
            capture_output=True, text=True, timeout=30, env=env
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

    if result.returncode != 0:
        return None

    output = result.stdout
    # Fix YAML parsing: bare operators like "operation: +" break YAML
    for op in ['=', '+', '-', '*', '/', '%', '&&', '||', '<', '>', '<=', '>=', '==', '!=']:
        output = re.sub(rf'operation: {re.escape(op)}\s*$',
                        f'operation: "{op}"', output, flags=re.MULTILINE)

    try:
        return yaml.safe_load(output)
    except yaml.YAMLError:
        return None


def _extract_accesses(stmt: dict) -> tuple:
    """Recursively extract all read/write accesses from a PET statement AST."""
    reads = []
    writes = []

    def traverse(node):
        if not isinstance(node, dict):
            return
        if node.get('type') == 'access':
            info = {
                'index': node.get('index', ''),
                'ref': node.get('reference', ''),
                'read': bool(node.get('read', 0)),
                'write': bool(node.get('write', 0)),
            }
            if info['read']:
                reads.append(info)
            if info['write']:
                writes.append(info)
        for key in ['arguments', 'body', 'expr']:
            if key in node:
                child = node[key]
                if isinstance(child, list):
                    for item in child:
                        traverse(item)
                elif isinstance(child, dict):
                    traverse(child)

    traverse(stmt.get('body', {}))
    return reads, writes


def _parse_isl_relation(rel_str: str) -> Optional[dict]:
    """Parse ISL relation '{ S_0[i,j] -> arr[(expr)] }' into components."""
    m = re.search(r'->\s*(\w+)\[(.*?)\]', rel_str)
    if m:
        return {'array': m.group(1), 'index_expr': m.group(2), 'full': rel_str}
    return None


def _get_loop_dims(domain_str: str) -> list:
    """Extract loop variable names from domain string."""
    m = re.search(r'S_\d+\[([^\]]+)\]', domain_str)
    if m:
        return [d.strip() for d in m.group(1).split(',')]
    return []


def _parse_linear_offset(index_expr: str, var: str) -> Optional[int]:
    """Parse index expression to extract linear offset from a variable.
    E.g., '(i)' -> 0, '(-1 + i)' -> -1, '(i + 2)' -> 2."""
    expr = index_expr.strip().strip('()')
    if expr == var:
        return 0
    # Pattern: -k + var
    m = re.match(rf'(-?\d+)\s*\+\s*{re.escape(var)}$', expr)
    if m:
        return int(m.group(1))
    # Pattern: var + k or var - k
    m = re.match(rf'{re.escape(var)}\s*([+-])\s*(\d+)$', expr)
    if m:
        sign = 1 if m.group(1) == '+' else -1
        return sign * int(m.group(2))
    return None


# ============================================================================
# Core analysis: derive everything from PET's parsed output
# ============================================================================

def _analyze_pet_data(pet_data: dict, kernel_name: str) -> dict:
    """Derive all analysis properties from a single parsed PET output."""

    result = {
        "dimensions": [],
        "dependencies": {
            "war": [],
            "war_safe": True,
            "arrays_with_war": [],
        },
        "parallelism": {
            "dims": [],
            "options": [],
            "is_triangular": False,
            "triangular_info": None,
            "fully_sequential": False,
        },
        "reductions": {
            "detected": False,
            "type": None,
        },
        "memory_access": {
            "is_stencil": False,
            "is_triangular": False,
            "has_cross_phase": False,
        },
    }

    statements = pet_data.get('statements', [])
    if not statements:
        return result

    # Collect all accesses across all statements
    all_reads = []
    all_writes = []
    all_domains = []
    schedule_str = str(pet_data.get('schedule', ''))

    for stmt in statements:
        stmt_domain = stmt.get('domain', '')
        all_domains.append(stmt_domain)
        reads, writes = _extract_accesses(stmt)

        for r in reads:
            parsed = _parse_isl_relation(r['index'])
            if parsed:
                all_reads.append({**parsed, 'domain': stmt_domain, 'stmt': stmt})
        for w in writes:
            parsed = _parse_isl_relation(w['index'])
            if parsed:
                all_writes.append({**parsed, 'domain': stmt_domain, 'stmt': stmt})

    # Get loop dimensions from the statement with the most loop dims
    # (scalar init statements like S_0[] have zero dims)
    dims = []
    domain_str = ''
    for d in all_domains:
        d_dims = _get_loop_dims(d)
        if len(d_dims) > len(dims):
            dims = d_dims
            domain_str = d
    result["parallelism"]["dims"] = dims
    result["dimensions"] = dims

    if not dims:
        return result
    if len(dims) == 2:
        d0, d1 = dims[0], dims[1]
        if re.search(rf'{d0}\s*<\s*{d1}', domain_str) or re.search(rf'{d1}\s*>\s*{d0}', domain_str):
            result["parallelism"]["is_triangular"] = True
            result["parallelism"]["triangular_info"] = {"smaller": d0, "larger": d1}
            result["memory_access"]["is_triangular"] = True
        elif re.search(rf'{d1}\s*<\s*{d0}', domain_str) or re.search(rf'{d0}\s*>\s*{d1}', domain_str):
            result["parallelism"]["is_triangular"] = True
            result["parallelism"]["triangular_info"] = {"smaller": d1, "larger": d0}
            result["memory_access"]["is_triangular"] = True

    # --- Detect cross-phase pattern (multiple statements writing different arrays) ---
    write_arrays_per_stmt = []
    for stmt in statements:
        _, writes = _extract_accesses(stmt)
        warrs = set()
        for w in writes:
            p = _parse_isl_relation(w['index'])
            if p:
                warrs.add(p['array'])
        write_arrays_per_stmt.append(warrs)

    if len(write_arrays_per_stmt) >= 2:
        # Different statements write to different arrays — cross-phase pattern
        all_write_arrays = set()
        for s in write_arrays_per_stmt:
            all_write_arrays.update(s)
        if len(all_write_arrays) >= 2:
            result["memory_access"]["has_cross_phase"] = True

    # --- Detect reductions (scalar write + array read) ---
    scalar_writes = set()
    array_reads = set()
    for w in all_writes:
        # Scalar: index has no loop variable (empty brackets or constant)
        idx = w.get('index_expr', '')
        if not idx or idx.strip() == '' or not any(d in idx for d in dims):
            scalar_writes.add(w['array'])
    for r in all_reads:
        idx = r.get('index_expr', '')
        if idx and any(d in idx for d in dims):
            array_reads.add(r['array'])

    if scalar_writes and array_reads:
        result["reductions"]["detected"] = True
        # Detect type from operation in AST
        result["reductions"]["type"] = _detect_reduction_op(statements)

    # --- Analyze parallelism per dimension using ISL ---
    _analyze_parallelism(result, all_reads, all_writes, all_domains, dims, schedule_str)

    # --- Detect stencil pattern ---
    valid_opts = [o for o in result["parallelism"]["options"] if o["valid"]]
    if (len(valid_opts) == 1
            and result["memory_access"]["has_cross_phase"]
            and len(dims) <= 2):
        result["memory_access"]["is_stencil"] = True

    return result


def _detect_reduction_op(statements: list) -> str:
    """Try to detect the reduction operation type from AST."""
    for stmt in statements:
        body = stmt.get('body', {})
        # Read-modify-write with binary op is a reduction
        if body.get('type') == 'access' and body.get('read') and body.get('write'):
            expr = body.get('expr', {})
            op = expr.get('operation', '')
            if op == '+' or op == '+=':
                return 'sum'
            elif op == '*' or op == '*=':
                return 'product'
        # Check nested expr
        def _find_op(node):
            if not isinstance(node, dict):
                return None
            op = node.get('operation', '')
            if op in ['+', '+=']:
                return 'sum'
            if op in ['*', '*=']:
                return 'product'
            for key in ['arguments', 'body', 'expr']:
                if key in node:
                    child = node[key]
                    if isinstance(child, list):
                        for c in child:
                            r = _find_op(c)
                            if r:
                                return r
                    elif isinstance(child, dict):
                        r = _find_op(child)
                        if r:
                            return r
            return None
        r = _find_op(body)
        if r:
            return r
    return 'unknown'


def _analyze_parallelism(result: dict, all_reads: list, all_writes: list,
                         all_domains: list, dims: list, schedule_str: str):
    """Analyze which dimensions are safe to parallelize using ISL."""
    if not HAS_ISL or not dims:
        return

    # Only consider accesses from statements with the right number of dimensions
    n_dims = len(dims)
    reads = [r for r in all_reads if len(_get_loop_dims(r['domain'])) == n_dims]
    writes = [w for w in all_writes if len(_get_loop_dims(w['domain'])) == n_dims]

    # Find arrays that are both read and written (potential conflicts)
    read_arrays = {r['array'] for r in reads}
    write_arrays = {w['array'] for w in writes}
    rw_arrays = read_arrays & write_arrays

    if not rw_arrays:
        # No array is both read and written — all dims are safe
        if len(dims) == 1:
            result["parallelism"]["options"].append({
                "parallel_dim": dims[0],
                "sequential_dim": None,
                "valid": True,
                "issues": [],
            })
        elif len(dims) == 2:
            result["parallelism"]["options"].append({
                "parallel_dim": dims[0],
                "sequential_dim": dims[1],
                "valid": True,
                "issues": [],
            })
            result["parallelism"]["options"].append({
                "parallel_dim": dims[1],
                "sequential_dim": dims[0],
                "valid": True,
                "issues": [],
            })
        return

    # For each dimension, check if parallelizing it causes cross-iteration conflicts
    domain_str = all_domains[0]
    try:
        domain = isl.Set(domain_str)
    except Exception:
        return

    for par_idx, par_dim in enumerate(dims):
        seq_dims = [d for i, d in enumerate(dims) if i != par_idx]
        seq_dim = seq_dims[0] if seq_dims else None
        issues = []
        has_conflict = False

        for arr in rw_arrays:
            # Get all reads and writes to this array (filtered to correct dims)
            arr_reads = [r for r in reads if r['array'] == arr]
            arr_writes = [w for w in writes if w['array'] == arr]

            for r in arr_reads:
                for w in arr_writes:
                    conflict = _check_cross_iteration_conflict(
                        domain, r, w, par_dim, dims, schedule_str
                    )
                    if conflict:
                        has_conflict = True
                        issues.append(conflict)

        # Check cross-phase: different statements access same arrays
        if result["memory_access"]["has_cross_phase"]:
            # Check if this dim is the "phase" dim (outermost)
            if par_idx == 0 and len(dims) >= 2:
                has_conflict = True
                issues.append(
                    f"Cross-phase dependency: different loop bodies under `{par_dim}` "
                    f"read/write the same arrays"
                )

        result["parallelism"]["options"].append({
            "parallel_dim": par_dim,
            "sequential_dim": seq_dim,
            "valid": not has_conflict,
            "issues": issues,
        })

        if has_conflict and arr in rw_arrays:
            if arr not in result["dependencies"]["arrays_with_war"]:
                result["dependencies"]["arrays_with_war"].append(arr)
                result["dependencies"]["war_safe"] = False
                result["dependencies"]["war"].append({
                    "description": f"WAR on `{arr}`: parallelizing `{par_dim}` may cause race conditions",
                })

    # Check if fully sequential
    valid = [o for o in result["parallelism"]["options"] if o["valid"]]
    if not valid:
        result["parallelism"]["fully_sequential"] = True


def _check_cross_iteration_conflict(domain, read_info, write_info,
                                     par_dim, dims, schedule_str) -> Optional[str]:
    """Check if parallelizing par_dim causes a cross-iteration conflict
    between a read and write to the same array. Returns issue string or None."""
    try:
        read_map = isl.Map(read_info['full'])
        write_map = isl.Map(write_info['full'])

        # Compose: read_map ∘ write_map⁻¹ gives iteration→iteration conflict relation
        write_inv = write_map.reverse()
        conflict = read_map.apply_range(write_inv)
        conflict = conflict.intersect_domain(domain)
        conflict = conflict.intersect_range(domain)

        if conflict.is_empty():
            return None

        # Remove same-iteration (identity) — only care about cross-iteration
        try:
            identity = isl.Map.identity(domain.get_space().map_from_set())
            cross_iter = conflict.subtract(identity)
            if cross_iter.is_empty():
                return None
        except Exception:
            pass

        # There IS a cross-iteration conflict.
        # Check if it's along par_dim specifically.
        # Project out other dimensions to see if conflict exists on par_dim alone.
        par_idx = dims.index(par_dim) if par_dim in dims else -1
        if par_idx < 0:
            return f"Dependencies carried by loop `{par_dim}`"

        # Additional heuristic: check if the conflict is WAR or RAW
        r_idx = read_info.get('index_expr', '')
        w_idx = write_info.get('index_expr', '')
        r_offset = _parse_linear_offset(r_idx, par_dim)
        w_offset = _parse_linear_offset(w_idx, par_dim)

        if r_offset is not None and w_offset is not None:
            if r_offset == w_offset:
                return None  # Same location — read-modify-write, safe for parallelism
            # Check schedule direction
            is_reverse = _check_loop_reverse(schedule_str, par_dim)
            if is_reverse:
                if r_offset < w_offset:
                    return f"WAR on `{read_info['array']}`: read at `{par_dim}{r_offset:+d}`, write at `{par_dim}{w_offset:+d}` (reverse loop)"
                else:
                    return None  # RAW in reverse — safe
            else:
                if r_offset > w_offset:
                    return f"WAR on `{read_info['array']}`: read at `{par_dim}{r_offset:+d}`, write at `{par_dim}{w_offset:+d}`"
                else:
                    return None  # RAW — safe

        return f"Dependencies carried by loop `{par_dim}` on array `{read_info['array']}`"

    except Exception:
        return None


def _check_loop_reverse(schedule_str: str, var: str) -> bool:
    """Check if a loop variable iterates in reverse from the schedule string."""
    if not schedule_str:
        return False
    pattern = rf'\[\s*\(\s*-\s*{re.escape(var)}\s*\)\s*\]'
    return bool(re.search(pattern, schedule_str))


# ============================================================================
# LLVM fallback (when PET fails)
# ============================================================================

def _llvm_fallback(kernel_file: str, kernel_name: str) -> Optional[dict]:
    """Use LLVM DependenceAnalysis as fallback when PET fails."""
    try:
        from llvm_fallback_adapters import (
            llvm_war_fallback, llvm_parallel_dims_fallback
        )
    except ImportError:
        return None

    result = {
        "dimensions": [],
        "dependencies": {"war": [], "war_safe": True, "arrays_with_war": []},
        "parallelism": {"dims": [], "options": [], "is_triangular": False,
                        "triangular_info": None, "fully_sequential": False},
        "reductions": {"detected": False, "type": None},
        "memory_access": {"is_stencil": False, "is_triangular": False, "has_cross_phase": False},
    }

    try:
        par_result = llvm_parallel_dims_fallback(kernel_file)
        if par_result:
            result["parallelism"]["dims"] = par_result.get('dims', [])
            result["parallelism"]["is_triangular"] = par_result.get('is_triangular', False)
            result["parallelism"]["triangular_info"] = par_result.get('triangular_info')
            for opt in par_result.get('options', []):
                result["parallelism"]["options"].append({
                    "parallel_dim": opt.get('parallel_dim'),
                    "sequential_dim": opt.get('sequential_dim'),
                    "valid": opt.get('valid', False),
                    "issues": opt.get('issues', []),
                })
            valid = [o for o in result["parallelism"]["options"] if o["valid"]]
            if not valid:
                result["parallelism"]["fully_sequential"] = True
    except Exception:
        pass

    try:
        war_result = llvm_war_fallback(kernel_file)
        if war_result and not war_result.get('parallelization_safe', True):
            result["dependencies"]["war_safe"] = False
            result["dependencies"]["arrays_with_war"] = war_result.get('arrays_needing_copy', [])
            for dep in war_result.get('war_dependencies', []):
                result["dependencies"]["war"].append({
                    "description": dep.get('description', ''),
                })
    except Exception:
        pass

    return result


# ============================================================================
# Public API
# ============================================================================

def analyze_kernel(kernel_name: str, kernel_source: str, arrays: dict,
                   params: dict, scalar_params: dict = None,
                   has_2d_arrays: bool = False) -> dict:
    """
    Run all analysis passes on a kernel and return a unified dict.

    Calls PET once, parses once, derives everything from that.
    Falls back to LLVM if PET fails.
    """
    scalar_params = scalar_params or {}

    analysis = {
        "kernel_name": kernel_name,
        "source": kernel_source,
        "arrays": arrays,
        "params": params,
        "scalar_params": scalar_params,
        "dimensions": [],
        "dependencies": {"war": [], "war_safe": True, "arrays_with_war": []},
        "parallelism": {"dims": [], "options": [], "is_triangular": False,
                        "triangular_info": None, "has_2d_arrays": has_2d_arrays,
                        "fully_sequential": False},
        "reductions": {"detected": False, "type": None},
        "memory_access": {"is_stencil": False, "is_triangular": False,
                          "has_cross_phase": False},
        "strategy": {"notes": []},
    }

    # Find kernel file
    kernel_file = None
    for subdir in ["kernels", "kernels_polybench", "kernels_realworld"]:
        candidate = ANALYSIS_DIR / subdir / f"{kernel_name}.c"
        if candidate.exists():
            kernel_file = str(candidate)
            break

    if not kernel_file:
        analysis["strategy"]["notes"].append("Kernel file not found for PET analysis")
        return analysis

    # --- Single PET invocation ---
    pet_data = _run_pet(kernel_file)

    if pet_data:
        derived = _analyze_pet_data(pet_data, kernel_name)
        # Merge derived results into analysis
        for key in ["dimensions", "dependencies", "parallelism", "reductions", "memory_access"]:
            analysis[key] = derived[key]
        analysis["parallelism"]["has_2d_arrays"] = has_2d_arrays
    else:
        # Fallback to LLVM
        analysis["strategy"]["notes"].append("PET failed, using LLVM fallback")
        fallback = _llvm_fallback(kernel_file, kernel_name)
        if fallback:
            for key in ["dimensions", "dependencies", "parallelism", "reductions", "memory_access"]:
                analysis[key] = fallback[key]
            analysis["parallelism"]["has_2d_arrays"] = has_2d_arrays

    return analysis


def format_analysis_for_prompt(analysis: dict) -> str:
    """
    Convert analysis dict into structured text for the LLM prompt.
    Pattern-agnostic: renders whatever the analysis contains.
    """
    sections = []

    # --- Dependencies ---
    if not analysis["dependencies"]["war_safe"]:
        lines = ["## Data Dependencies", ""]
        lines.append("This kernel has Write-After-Read (WAR) dependencies:")
        for arr in analysis["dependencies"]["arrays_with_war"]:
            lines.append(f"- Array `{arr}` is both read and written")
        for dep in analysis["dependencies"]["war"][:5]:
            lines.append(f"- {dep['description']}")
        lines.append("")
        lines.append("If you parallelize a dimension that carries a WAR dependency,")
        lines.append("clone the array before the parallel region or use separate kernels.")
        sections.append("\n".join(lines))

    # --- Parallelism ---
    opts = analysis["parallelism"]["options"]
    if opts:
        lines = ["## Parallelization Analysis", ""]
        lines.append(f"Loop dimensions: {analysis['parallelism']['dims']}")
        if analysis["parallelism"]["is_triangular"]:
            tri = analysis["parallelism"]["triangular_info"]
            if tri:
                lines.append(f"Triangular loop bounds: {tri['smaller']} < {tri['larger']}")
        lines.append("")
        for opt in opts:
            status = "SAFE" if opt["valid"] else "UNSAFE"
            lines.append(f"- Parallelize `{opt['parallel_dim']}`, sequential `{opt['sequential_dim']}`: **{status}**")
            for issue in opt["issues"]:
                lines.append(f"  - {issue}")
        if analysis["parallelism"]["fully_sequential"]:
            lines.append("")
            lines.append("**No dimension can be safely parallelized.** Use `grid=(1,)` with sequential loops.")
        sections.append("\n".join(lines))

    # --- Reductions ---
    if analysis["reductions"]["detected"]:
        lines = ["## Reduction Pattern", ""]
        lines.append(f"Reduction type: {analysis['reductions']['type']}")
        lines.append("Use `tl.sum()`, `tl.max()`, or `tl.dot()` for the reduction dimension.")
        sections.append("\n".join(lines))

    # --- Memory Patterns ---
    mem = analysis["memory_access"]
    if mem["is_stencil"] or mem["is_triangular"] or mem["has_cross_phase"]:
        lines = ["## Memory Access Patterns", ""]
        if mem["is_stencil"]:
            lines.append("- **Stencil pattern**: reads neighboring elements of the same array")
            lines.append("  Use `tl.debug_barrier()` between phases if using a single CTA (`grid=(1,)`)")
        if mem["is_triangular"]:
            lines.append("- **Triangular access**: loop bounds depend on outer loop variable")
            lines.append("  Consider `grid=(N,)` with one program per row, masking unused columns")
        if mem["has_cross_phase"]:
            lines.append("- **Cross-phase dependency**: one phase writes what another phase reads")
            lines.append("  Either use separate kernel launches or `tl.debug_barrier()` between phases")
        sections.append("\n".join(lines))

    # --- Notes ---
    if analysis["strategy"]["notes"]:
        lines = ["## Notes", ""]
        for note in analysis["strategy"]["notes"]:
            lines.append(f"- {note}")
        sections.append("\n".join(lines))

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
    kernel_file = None
    for subdir in ["kernels", "kernels_polybench", "kernels_realworld"]:
        candidate = ANALYSIS_DIR / subdir / f"{kernel_name}.c"
        if candidate.exists():
            kernel_file = str(candidate)
            break
    if not kernel_file:
        print(f"Kernel file not found: {kernel_name}")
        sys.exit(1)

    source = Path(kernel_file).read_text()
    analysis = analyze_kernel(kernel_name, source,
                              arrays={"A": "rw"}, params={"N": 100})

    # Print without source
    a = {k: v for k, v in analysis.items() if k != "source"}
    print("=== Raw Analysis (JSON) ===")
    print(json.dumps(a, indent=2, default=str))
    print("\n=== Formatted for Prompt ===")
    print(format_analysis_for_prompt(analysis))
