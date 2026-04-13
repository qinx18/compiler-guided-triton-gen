#!/usr/bin/env python3
"""
LLM Agent Baseline for PolyBench/C Kernel Generation.

This is a proper agentic baseline: the LLM autonomously generates, tests,
debugs, and optimizes Triton kernels by calling tools (compile, test, benchmark).
It sees the FULL output of each tool — not classified error summaries.

No compiler analysis is provided. The agent relies purely on its own reasoning
about the C source code and feedback from execution.

Usage:
    python agent_baseline.py                    # All 30 kernels
    python agent_baseline.py gemm lu jacobi_1d  # Specific kernels
"""

import os
import sys
import json
import subprocess
import anthropic
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

sys.path.append(str(Path(__file__).parent / "utilities"))
from polybench_functions_db import POLYBENCH_FUNCTIONS

sys.path.insert(0, "/home/qinxiao/workspace/compiler-guided-triton-gen/analysis")
from extract_polybench_kernels import POLYBENCH_KERNELS

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=API_KEY)

POLYBENCH_KERNELS_DIR = "/home/qinxiao/workspace/compiler-guided-triton-gen/analysis/kernels_polybench"
OUTPUT_DIR = "../results/polybench/polybench_results_agent"
MAX_CORRECTNESS_TURNS = 10  # Phase 1: get a correct implementation
MAX_OPTIMIZATION_TURNS = 3   # Phase 2: optimize after correctness passes


def get_kernel_source(kernel_name: str) -> Optional[str]:
    c_name = kernel_name.replace("-", "_")
    path = Path(POLYBENCH_KERNELS_DIR) / f"{c_name}.c"
    return path.read_text() if path.exists() else None


def get_kernel_params(kernel_name: str) -> dict:
    orig = kernel_name.replace("_", "-")
    for entry in POLYBENCH_KERNELS.values():
        if isinstance(entry, dict) and entry.get('name') == orig:
            return dict(entry.get('params', {}))
    return {}


# ============================================================================
# Tools the agent can call
# ============================================================================

TOOLS_CORRECTNESS = [
    {
        "name": "write_triton_code",
        "description": "Write a Triton kernel implementation to a file. The code should be a complete Python file with @triton.jit kernel and a wrapper function named {kernel_name}_triton.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Complete Python code for the Triton kernel implementation"
                }
            },
            "required": ["code"]
        }
    },
    {
        "name": "run_correctness_test",
        "description": "Run the correctness test comparing your Triton implementation against the C reference. Returns PASS/FAIL with detailed error information.",
        "input_schema": {
            "type": "object",
            "properties": {},
        }
    },
]

TOOLS_OPTIMIZATION = TOOLS_CORRECTNESS + [
    {
        "name": "run_benchmark",
        "description": "Run a performance benchmark comparing your Triton implementation against the C reference. Returns execution times and speedup.",
        "input_schema": {
            "type": "object",
            "properties": {},
        }
    },
]


def execute_tool(tool_name: str, tool_input: dict, kernel_name: str,
                 func_spec: dict, attempt: int, func_dir: Path,
                 test_dir: Path) -> str:
    """Execute a tool and return the result as a string."""
    func_id = kernel_name
    if func_id[0].isdigit():
        func_id = "k" + func_id

    if tool_name == "write_triton_code":
        code = tool_input["code"]
        code_file = func_dir / f"attempt{attempt}.py"
        code_file.write_text(code)
        # Quick syntax check
        try:
            compile(code, str(code_file), 'exec')
            return f"Code written to {code_file}. Syntax OK."
        except SyntaxError as e:
            return f"Code written but has SYNTAX ERROR: {e}"

    elif tool_name == "run_correctness_test":
        # Generate and run test using the same infrastructure
        from generate_and_test_polybench import generate_correctness_test, run_test
        test_code = generate_correctness_test(kernel_name, func_spec, attempt)
        test_file = test_dir / f"test_{kernel_name}_correctness.py"
        test_file.write_text(test_code)

        passed, error_info = run_test(kernel_name, test_file)
        if passed:
            return "CORRECTNESS TEST: PASSED. All test cases match the C reference within tolerance."
        else:
            err_type = error_info.get('type', 'unknown')
            err_msg = error_info.get('message', '')[:2000]
            max_err = error_info.get('max_error', '')
            result = f"CORRECTNESS TEST: FAILED\n"
            result += f"Error type: {err_type}\n"
            if max_err:
                result += f"Max numerical error: {max_err}\n"
            result += f"Details:\n{err_msg}"
            return result

    elif tool_name == "run_benchmark":
        from generate_and_test_polybench import generate_benchmark_test, run_benchmark
        bench_code = generate_benchmark_test(kernel_name, func_spec, attempt)
        bench_file = test_dir / f"benchmark_{kernel_name}.py"
        bench_file.write_text(bench_code)

        result = run_benchmark(kernel_name, bench_file)
        if result:
            c_ms = result.get('c_ref_time_ms')
            tr_ms = result.get('triton_time_ms')
            sp = result.get('speedup')
            return (f"BENCHMARK RESULTS:\n"
                    f"  C reference: {c_ms:.3f} ms\n"
                    f"  Triton:      {tr_ms:.3f} ms\n"
                    f"  Speedup:     {sp:.2f}x")
        else:
            return "BENCHMARK: Failed or timed out."

    return f"Unknown tool: {tool_name}"


# ============================================================================
# Agent loop
# ============================================================================

def process_kernel_agent(kernel_name: str, func_spec: dict) -> dict:
    """Process a kernel using the autonomous agent approach."""
    print(f"\n{'=' * 70}")
    print(f"[AGENT] Processing: {kernel_name}")
    print(f"{'=' * 70}")

    source = get_kernel_source(kernel_name)
    if not source:
        return {"test_passed": False, "error": "Source not found"}

    params = get_kernel_params(kernel_name)
    arrays = func_spec.get('arrays', {})
    scalar_params = func_spec.get('scalar_params', {})
    loop_code = func_spec.get('loop_code', '')

    func_id = kernel_name
    if func_id[0].isdigit():
        func_id = "k" + func_id

    # Build array info
    array_lines = []
    for arr_name, mode in sorted(arrays.items()):
        mode_str = {'r': 'read-only', 'w': 'write-only', 'rw': 'read-write',
                    'temp': 'temporary'}[mode]
        array_lines.append(f"- `{arr_name}`: {mode_str}")
    array_info = "\n".join(array_lines)

    dim_lines = [f"- `{k}` = {v}" for k, v in sorted(params.items())]
    dim_info = "\n".join(dim_lines)

    sig_parts = sorted(arrays.keys()) + sorted(scalar_params.keys()) + \
                [p for p in sorted(params.keys()) if p not in scalar_params]
    exact_sig = ", ".join(sig_parts)

    # Setup directories
    base_dir = Path(OUTPUT_DIR)
    func_dir = base_dir / "llm_triton" / kernel_name
    test_dir = Path("../results/polybench/my_polybench_tests_agent") / kernel_name
    for d in [func_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Initial system prompt
    system_prompt = f"""You are a GPU kernel optimization expert. Your task is to translate a C kernel to Triton for GPU acceleration.

You have three tools available:
1. write_triton_code - Write your Triton implementation
2. run_correctness_test - Test if your code produces correct results vs the C reference
3. run_benchmark - Measure performance (only after correctness passes)

Your workflow should be:
1. Analyze the C code and write a Triton implementation
2. Run the correctness test
3. If it fails, analyze the error, fix the code, and retry
4. Once correct, run the benchmark to measure performance
5. If performance is low, optimize and verify correctness again

Important Triton rules:
- Pass dimension parameters as tl.constexpr
- Never use tl.arange() inside a for loop
- Never use Python lists, break/continue inside @triton.jit
- Use Python operators (a * b, not tl.mul)
- For 2D arrays: linear index = row * stride + col"""

    user_msg = f"""Translate this C kernel to Triton:

## C Source Code:
```c
{source}
```

## Kernel Loop:
```c
{loop_code}
```

## Arrays:
{array_info}

## Dimensions:
{dim_info}

## Required function signature:
```python
def {func_id}_triton({exact_sig}):
```

The kernel function should be named `{func_id}_kernel`.
Start by writing your implementation using the write_triton_code tool."""

    messages = [{"role": "user", "content": user_msg}]

    results = {
        "test_passed": False,
        "attempts": 0,
        "correctness_turns": 0,
        "optimization_turns": 0,
        "benchmark": None,
    }

    attempt = 1
    best_speedup = -1
    best_attempt = 0

    # ---- Phase 1: Correctness (up to MAX_CORRECTNESS_TURNS) ----
    print(f"  --- Phase 1: Correctness (up to {MAX_CORRECTNESS_TURNS} turns) ---")
    for turn in range(MAX_CORRECTNESS_TURNS):
        results["correctness_turns"] = turn + 1
        print(f"  [Correctness {turn + 1}/{MAX_CORRECTNESS_TURNS}]", end=" ", flush=True)

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8192,
                system=system_prompt,
                tools=TOOLS_CORRECTNESS,
                messages=messages,
            )
        except Exception as e:
            print(f"API error: {e}")
            break

        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        tool_uses = [block for block in assistant_content if block.type == "tool_use"]

        if not tool_uses:
            text_blocks = [block.text for block in assistant_content if hasattr(block, 'text')]
            print(f"Text: {text_blocks[0][:80]}..." if text_blocks else "No content")
            if response.stop_reason == "end_turn":
                break
            continue

        tool_results = []
        passed_this_turn = False
        for tool_use in tool_uses:
            tool_name = tool_use.name
            tool_input = tool_use.input

            if tool_name == "write_triton_code":
                attempt += 1
                results["attempts"] = attempt
                print(f"Write(att {attempt})", end=" → ", flush=True)
            elif tool_name == "run_correctness_test":
                print(f"Test", end=" → ", flush=True)

            tool_result = execute_tool(
                tool_name, tool_input, kernel_name, func_spec,
                attempt, func_dir, test_dir
            )

            if tool_name == "run_correctness_test" and "PASSED" in tool_result:
                results["test_passed"] = True
                passed_this_turn = True
                print("PASS", end=" ", flush=True)
            elif tool_name == "run_correctness_test":
                print("FAIL", end=" ", flush=True)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": tool_result,
            })

        messages.append({"role": "user", "content": tool_results})
        print()

        if passed_this_turn:
            break

        if response.stop_reason == "end_turn":
            break

    if not results["test_passed"]:
        print(f"  [AGENT] Phase 1 FAILED after {results['correctness_turns']} turns")
        return results

    # ---- Phase 2: Optimization (up to MAX_OPTIMIZATION_TURNS) ----
    print(f"  --- Phase 2: Optimization (up to {MAX_OPTIMIZATION_TURNS} turns) ---")

    # Inject transition message telling the agent to benchmark and optimize
    messages.append({"role": "user", "content": [{"type": "text", "text":
        "Your implementation passes correctness. Now benchmark it and try to optimize for better performance. "
        "You can write improved code, test correctness again, and benchmark. "
        "Keep the best version (highest speedup that still passes correctness)."}]})

    for turn in range(MAX_OPTIMIZATION_TURNS):
        results["optimization_turns"] = turn + 1
        print(f"  [Optimize {turn + 1}/{MAX_OPTIMIZATION_TURNS}]", end=" ", flush=True)

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8192,
                system=system_prompt,
                tools=TOOLS_OPTIMIZATION,
                messages=messages,
            )
        except Exception as e:
            print(f"API error: {e}")
            break

        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        tool_uses = [block for block in assistant_content if block.type == "tool_use"]

        if not tool_uses:
            text_blocks = [block.text for block in assistant_content if hasattr(block, 'text')]
            print(f"Text: {text_blocks[0][:80]}..." if text_blocks else "No content")
            if response.stop_reason == "end_turn":
                break
            continue

        tool_results = []
        for tool_use in tool_uses:
            tool_name = tool_use.name
            tool_input = tool_use.input

            if tool_name == "write_triton_code":
                attempt += 1
                results["attempts"] = attempt
                print(f"Write(att {attempt})", end=" → ", flush=True)
            elif tool_name == "run_correctness_test":
                print(f"Test", end=" → ", flush=True)
            elif tool_name == "run_benchmark":
                print(f"Bench", end=" → ", flush=True)

            tool_result = execute_tool(
                tool_name, tool_input, kernel_name, func_spec,
                attempt, func_dir, test_dir
            )

            if tool_name == "run_correctness_test" and "PASSED" in tool_result:
                print("PASS", end=" ", flush=True)
            elif tool_name == "run_correctness_test":
                print("FAIL", end=" ", flush=True)
            elif tool_name == "run_benchmark" and "Speedup:" in tool_result:
                import re
                sp_match = re.search(r'Speedup:\s+([\d.]+)x', tool_result)
                if sp_match:
                    sp = float(sp_match.group(1))
                    print(f"{sp:.2f}x", end=" ", flush=True)
                    if sp > best_speedup:
                        best_speedup = sp
                        best_attempt = attempt
                        results["benchmark"] = {"speedup": sp}

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": tool_result,
            })

        messages.append({"role": "user", "content": tool_results})
        print()

        if response.stop_reason == "end_turn":
            break

    results["final_attempt"] = best_attempt if best_attempt else attempt
    sp_str = f", speedup: {best_speedup:.2f}x" if best_speedup > 0 else ""
    print(f"  [AGENT] Done: PASS, {results['correctness_turns']}+{results['optimization_turns']} turns{sp_str}")

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    global OUTPUT_DIR

    if not API_KEY:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    # Check for --size-scale flag
    size_scale = 1
    if '--size-scale' in sys.argv:
        idx = sys.argv.index('--size-scale')
        size_scale = int(sys.argv[idx + 1])
        sys.argv.pop(idx)
        sys.argv.pop(idx)
        # Set SIZE_SCALE in the polybench pipeline module so test/bench use scaled sizes
        import generate_and_test_polybench as pb
        pb.SIZE_SCALE = size_scale
        pb.OUTPUT_DIR = f"../results/polybench/polybench_results_scale{size_scale}x"
        OUTPUT_DIR = f"../results/polybench/polybench_results_agent_scale{size_scale}x"

    # Parse kernel names from args
    kernel_names = sys.argv[1:] if len(sys.argv) > 1 else sorted(POLYBENCH_FUNCTIONS.keys())
    kernel_names = [k.replace("-", "_") for k in kernel_names]

    base_dir = Path(OUTPUT_DIR)
    base_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LLM Agent Baseline for PolyBench/C")
    print(f"Kernels: {len(kernel_names)}")
    print(f"Max turns per kernel: {MAX_CORRECTNESS_TURNS} correctness + {MAX_OPTIMIZATION_TURNS} optimization")
    print(f"Model: claude-sonnet-4-20250514 (with tool use)")
    print("=" * 70)

    all_results = {}
    for i, kernel_name in enumerate(kernel_names, 1):
        if kernel_name not in POLYBENCH_FUNCTIONS:
            print(f"Skipping unknown kernel: {kernel_name}")
            continue

        func_spec = POLYBENCH_FUNCTIONS[kernel_name]
        print(f"\n[{i}/{len(kernel_names)}]", end=" ")
        result = process_kernel_agent(kernel_name, func_spec)
        all_results[kernel_name] = result

    # Summary
    passed = sum(1 for r in all_results.values() if r.get('test_passed'))
    total = len(all_results)
    speedups = [r['benchmark']['speedup'] for r in all_results.values()
                if r.get('benchmark') and r['benchmark'].get('speedup')]

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Kernel':<18} {'Passed':<8} {'Turns':<7} {'Speedup':<10}")
    print("-" * 45)
    for k in sorted(all_results):
        r = all_results[k]
        p = "Y" if r.get('test_passed') else "N"
        t = r.get('turns', 0)
        sp = f"{r['benchmark']['speedup']:.2f}x" if r.get('benchmark') and r['benchmark'].get('speedup') else "-"
        print(f"{k:<18} {p:<8} {t:<7} {sp:<10}")

    print(f"\nPassed: {passed}/{total} ({100*passed/total:.1f}%)")
    if speedups:
        speedups.sort()
        print(f"Median speedup: {speedups[len(speedups)//2]:.2f}x")
        print(f"Mean speedup:   {sum(speedups)/len(speedups):.2f}x")
        print(f">1x: {sum(1 for s in speedups if s > 1)}/{len(speedups)}")

    # Save results
    results_file = base_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
