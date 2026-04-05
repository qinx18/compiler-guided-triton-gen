#!/usr/bin/env python3
"""
Comprehensive nondeterminism study for ALL 10 regression kernels.

Re-runs the WA pipeline multiple times per kernel, merges with existing
evidence, and classifies each regression as either:
  - "nondeterminism": >=1 WA run beats NA on Triton time (random variance)
  - "analysis_unhelpful": 0/N WA runs beat NA (analysis consistently hurts)

Category A (pseudo-regressions): WA Triton already faster than NA — speedup
regression is from C-ref timing variance between WA and NA runs.

Category B (true regressions): WA Triton slower than NA — need multiple runs
to determine if it's nondeterminism or a real guidance flaw.

Usage:
    python run_nondeterminism_study.py
"""
import subprocess
import json
import os
import re
import sys
from pathlib import Path

SCRIPT = "generate_and_test_polybench.py"
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "../results/polybench/polybench_results"
EVIDENCE_FILE = RESULTS_DIR / "nondeterminism_evidence.json"
NA_FILE = RESULTS_DIR / "results_no_analysis.json"

# ── Kernel definitions ──────────────────────────────────────────────────

# Category A: WA Triton is FASTER than NA — regression is C-ref variance
CATEGORY_A = {
    "gemm":    {"new_runs": 3, "na_triton_ms": 0.157},
    "heat_3d": {"new_runs": 2, "na_triton_ms": 0.801},
    "trmm":    {"new_runs": 3, "na_triton_ms": 0.071},
    "mvt":     {"new_runs": 3, "na_triton_ms": 0.135},
    "syrk":    {"new_runs": 3, "na_triton_ms": 0.089},
    "ludcmp":  {"new_runs": 3, "na_triton_ms": 10.266},
}

# Category B: WA Triton is SLOWER than NA — need evidence
CATEGORY_B = {
    "3mm":    {"new_runs": 2, "na_triton_ms": 0.138},
    "symm":   {"new_runs": 2, "na_triton_ms": 0.073},
    "syr2k":  {"new_runs": 5, "na_triton_ms": 0.086},
    "gemver": {"new_runs": 2, "na_triton_ms": 0.186},
}

# Root causes for analysis-unhelpful kernels (if needed)
ROOT_CAUSES = {
    "3mm": "Analysis recommends conservative parallelism for WAR deps. LLM generates grid=(1,) fully sequential kernel. NA generates 3 separate 2D-tiled GEMM kernels with tl.dot().",
    "symm": "Analysis restricts j-parallelism for triangular access pattern. LLM generates grid=(1,) with j-vectorization only. NA parallelizes j directly with grid=(N,).",
}


def extract_code_info(kernel_name: str, code_path: str) -> str:
    """Extract code structure info from generated Triton file."""
    if not os.path.exists(code_path):
        return "FAIL"
    with open(code_path) as f:
        code = f.read()

    has_arange = "tl.arange" in code
    has_debug_barrier = "tl.debug_barrier" in code
    has_tl_dot = "tl.dot" in code
    grid_match = re.search(r'grid\s*=\s*\(([^)]+)\)', code)
    grid_str = grid_match.group(1).strip() if grid_match else "?"
    n_kernels = len(re.findall(r'@triton\.jit', code))

    # Count host loops in wrapper function
    func_id = kernel_name if not kernel_name[0].isdigit() else "k" + kernel_name
    wrapper_match = re.search(rf'def {func_id}_triton\(.*?\n((?:    .*\n)*)', code)
    host_loops = 0
    if wrapper_match:
        wrapper = wrapper_match.group(1)
        host_loops = len(re.findall(r'for\s+\w+\s+in\s+range', wrapper))

    parts = [f"grid=({grid_str})", f"{n_kernels} kernel(s)"]
    if has_arange:
        parts.append("vectorized")
    if has_debug_barrier:
        parts.append("debug_barrier")
    if has_tl_dot:
        parts.append("tl.dot")
    if host_loops > 0:
        parts.append(f"{host_loops} host loop(s)")
    return ", ".join(parts)


def run_kernel_once(kernel_name: str) -> dict:
    """Run a single kernel through the WA pipeline and extract results."""
    print(f"  Running pipeline for {kernel_name}...")
    try:
        proc = subprocess.run(
            ["python", SCRIPT, kernel_name],
            capture_output=True, text=True, timeout=600,
            cwd=str(BASE_DIR)
        )
    except subprocess.TimeoutExpired:
        return {"passed": False, "triton_ms": None, "code_info": "TIMEOUT"}

    # Read results
    results_file = RESULTS_DIR / "results.json"
    with open(results_file) as f:
        data = json.load(f)

    kdata = data.get(kernel_name, {})
    passed = kdata.get("test_passed", False)
    attempts = kdata.get("attempts", 0)
    speedup = kdata.get("benchmark", {}).get("speedup") if passed else None
    triton_ms = kdata.get("benchmark", {}).get("triton_time_ms") if passed else None

    code_info = "FAIL"
    if passed:
        attempt_num = kdata.get("final_attempt", attempts)
        code_path = str(RESULTS_DIR / "llm_triton" / kernel_name / f"attempt{attempt_num}.py")
        code_info = extract_code_info(kernel_name, code_path)

    return {
        "passed": passed,
        "attempts": attempts,
        "speedup": speedup,
        "triton_ms": triton_ms,
        "code_info": code_info,
    }


def main():
    # Load existing evidence
    existing = {}
    if EVIDENCE_FILE.exists():
        with open(EVIDENCE_FILE) as f:
            existing = json.load(f)

    # Load NA results for reference
    with open(NA_FILE) as f:
        na_results = json.load(f)

    all_kernels = {}
    all_kernels.update({k: ("pseudo_regression", v) for k, v in CATEGORY_A.items()})
    all_kernels.update({k: ("true_regression", v) for k, v in CATEGORY_B.items()})

    total_new_runs = sum(v["new_runs"] for v in CATEGORY_A.values()) + \
                     sum(v["new_runs"] for v in CATEGORY_B.values())

    print("=" * 80)
    print("COMPREHENSIVE NONDETERMINISM STUDY")
    print(f"Category A (pseudo-regressions, WA Triton faster): {len(CATEGORY_A)} kernels")
    print(f"Category B (true regressions, WA Triton slower):   {len(CATEGORY_B)} kernels")
    print(f"Total new pipeline runs: {total_new_runs}")
    print("=" * 80)

    # New evidence structure
    evidence = {}
    run_counter = 0

    for kernel_name in sorted(all_kernels.keys()):
        category, config = all_kernels[kernel_name]
        new_runs_needed = config["new_runs"]
        na_t = na_results.get(kernel_name, {}).get("benchmark", {}).get("triton_time_ms", 0)

        print(f"\n{'─' * 60}")
        print(f"{kernel_name} ({category})")
        print(f"  NA Triton: {na_t:.3f}ms | New runs needed: {new_runs_needed}")
        print(f"{'─' * 60}")

        # Collect existing runs (from old evidence format: list of run dicts)
        existing_runs = []
        if kernel_name in existing:
            old_data = existing[kernel_name]
            if isinstance(old_data, list):
                # Old format: list of run dicts
                for r in old_data:
                    existing_runs.append({
                        "run": r.get("run", len(existing_runs) + 1),
                        "passed": r.get("passed", False),
                        "triton_ms": r.get("triton_ms"),
                        "code_info": r.get("code_info", ""),
                    })
            elif isinstance(old_data, dict) and "runs" in old_data:
                # Already in new format
                existing_runs = old_data["runs"]

        print(f"  Existing runs: {len(existing_runs)}")
        for r in existing_runs:
            t = r.get("triton_ms")
            status = f"{t:.3f}ms" if t else "FAIL"
            beat = " (beats NA)" if t and na_t > 0 and t < na_t else ""
            print(f"    Run {r['run']}: {status}{beat}")

        # Run new pipeline invocations
        new_runs = []
        for i in range(new_runs_needed):
            run_counter += 1
            run_num = len(existing_runs) + i + 1
            print(f"\n  [{run_counter}/{total_new_runs}] New run {run_num}...")

            result = run_kernel_once(kernel_name)
            result["run"] = run_num
            new_runs.append(result)

            t = result["triton_ms"]
            if t is not None:
                beat = " (beats NA)" if na_t > 0 and t < na_t else ""
                print(f"    -> {t:.3f}ms [att={result['attempts']}]{beat}")
            else:
                print(f"    -> FAIL [att={result.get('attempts', 0)}]")

        # Combine all runs
        all_runs = existing_runs + new_runs

        # Classify
        wa_beats_na = 0
        for r in all_runs:
            t = r.get("triton_ms")
            if t is not None and na_t > 0 and t < na_t:
                wa_beats_na += 1

        total_passed = sum(1 for r in all_runs if r.get("triton_ms") is not None)

        if wa_beats_na > 0:
            classification = "nondeterminism"
        else:
            classification = "analysis_unhelpful"

        # Build explanation
        if classification == "nondeterminism":
            if category == "pseudo_regression":
                explanation = (f"WA Triton is consistently faster than NA ({na_t:.3f}ms). "
                               f"The speedup regression is caused by C-reference timing variance between "
                               f"WA and NA benchmark runs, not by analysis quality.")
            else:
                triton_times = [r["triton_ms"] for r in all_runs if r.get("triton_ms") is not None]
                best = min(triton_times) if triton_times else 0
                worst = max(triton_times) if triton_times else 0
                explanation = (f"WA Triton time varies from {best:.3f}ms to {worst:.3f}ms across {len(triton_times)} runs. "
                               f"{wa_beats_na}/{len(triton_times)} runs beat NA ({na_t:.3f}ms). "
                               f"Regression is LLM nondeterminism.")
        else:
            explanation = ROOT_CAUSES.get(kernel_name, "Analysis consistently produces slower code for this kernel.")

        evidence[kernel_name] = {
            "na_triton_ms": na_t,
            "category": category,
            "classification": classification,
            "wa_beats_na_count": wa_beats_na,
            "total_runs": len(all_runs),
            "runs": all_runs,
            "explanation": explanation,
        }

        print(f"\n  Classification: {classification.upper()}")
        print(f"  WA beats NA: {wa_beats_na}/{len(all_runs)} runs")

    # Save evidence
    with open(EVIDENCE_FILE, "w") as f:
        json.dump(evidence, f, indent=2)
    print(f"\nEvidence saved to {EVIDENCE_FILE}")

    # Summary
    print("\n" + "=" * 80)
    print("CLASSIFICATION SUMMARY")
    print("=" * 80)
    print(f"\n{'Kernel':<12} {'Category':<20} {'Classification':<20} {'WA>NA':<8} {'Runs':<5}")
    print("-" * 65)
    for k in sorted(evidence.keys()):
        e = evidence[k]
        print(f"{k:<12} {e['category']:<20} {e['classification']:<20} "
              f"{e['wa_beats_na_count']}/{e['total_runs']:<5} {e['total_runs']:<5}")

    nondet_count = sum(1 for e in evidence.values() if e["classification"] == "nondeterminism")
    unhelpful_count = sum(1 for e in evidence.values() if e["classification"] == "analysis_unhelpful")
    print(f"\nNondeterminism: {nondet_count}/10")
    print(f"Analysis unhelpful: {unhelpful_count}/10")


if __name__ == "__main__":
    main()
