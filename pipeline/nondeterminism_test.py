#!/usr/bin/env python3
"""Run each regression kernel N times to demonstrate LLM nondeterminism.
Collects speedup, Triton time, code structure (grid size, approach) per run.
"""
import subprocess, json, sys, os, re, glob

KERNELS = ["3mm", "jacobi_1d", "correlation", "gemver", "heat_3d", "symm"]
N_RUNS = 3
SCRIPT = "generate_and_test_polybench.py"

results = {k: [] for k in KERNELS}

for run_idx in range(1, N_RUNS + 1):
    for kernel in KERNELS:
        print(f"\n{'='*60}")
        print(f"Run {run_idx}/{N_RUNS}: {kernel}")
        print(f"{'='*60}")

        # Run the kernel
        proc = subprocess.run(
            ["python", SCRIPT, kernel],
            capture_output=True, text=True, timeout=600
        )

        # Read results
        with open("../results/polybench/polybench_results/results.json") as f:
            data = json.load(f)

        kdata = data.get(kernel, {})
        passed = kdata.get("test_passed", False)
        attempts = kdata.get("attempts", 0)
        speedup = kdata.get("benchmark", {}).get("speedup", 0) if passed else None
        triton_ms = kdata.get("benchmark", {}).get("triton_time_ms", 0) if passed else None

        # Analyze the generated code to understand approach
        code_info = "FAIL"
        if passed:
            attempt_num = kdata.get("final_attempt", attempts)
            code_path = f"../results/polybench/polybench_results/llm_triton/{kernel}/attempt{attempt_num}.py"
            if os.path.exists(code_path):
                with open(code_path) as f:
                    code = f.read()
                # Detect approach
                has_arange = "tl.arange" in code
                has_debug_barrier = "tl.debug_barrier" in code
                has_tl_dot = "tl.dot" in code
                grid_match = re.search(r'grid\s*=\s*\(([^)]+)\)', code)
                grid_str = grid_match.group(1).strip() if grid_match else "?"
                # Count kernels
                n_kernels = len(re.findall(r'@triton\.jit', code))
                # Count host-level for loops (kernel launches in loop)
                n_host_loops = len(re.findall(r'for\s+\w+\s+in\s+range', code.split("def " + kernel.replace("-","_"))[0] if ("def " + kernel.replace("-","_")) in code else code))
                # Better: look at the wrapper function
                wrapper_match = re.search(r'def \w+_triton\(.*?\n((?:    .*\n)*)', code)
                host_loops = 0
                if wrapper_match:
                    wrapper = wrapper_match.group(1)
                    host_loops = len(re.findall(r'for\s+\w+\s+in\s+range', wrapper))

                parts = []
                parts.append(f"grid=({grid_str})")
                parts.append(f"{n_kernels} kernel(s)")
                if has_arange: parts.append("vectorized")
                if has_debug_barrier: parts.append("debug_barrier")
                if has_tl_dot: parts.append("tl.dot")
                if host_loops > 0: parts.append(f"{host_loops} host loop(s)")
                code_info = ", ".join(parts)

        results[kernel].append({
            "run": run_idx,
            "passed": passed,
            "attempts": attempts,
            "speedup": speedup,
            "triton_ms": triton_ms,
            "code_info": code_info
        })

        status = f"PASS {speedup:.2f}x ({triton_ms:.3f}ms)" if passed else "FAIL"
        print(f"  -> {status} [att={attempts}] [{code_info}]")

# Print summary
print("\n" + "=" * 100)
print("NONDETERMINISM EVIDENCE SUMMARY")
print("=" * 100)
for kernel in KERNELS:
    runs = results[kernel]
    speedups = [r["speedup"] for r in runs if r["speedup"] is not None]
    triton_times = [r["triton_ms"] for r in runs if r["triton_ms"] is not None]
    pass_count = sum(1 for r in runs if r["passed"])

    print(f"\n{kernel}:")
    print(f"  Pass rate: {pass_count}/{N_RUNS}")
    for r in runs:
        if r["passed"]:
            print(f"  Run {r['run']}: {r['speedup']:.2f}x ({r['triton_ms']:.3f}ms) att={r['attempts']} [{r['code_info']}]")
        else:
            print(f"  Run {r['run']}: FAIL att={r['attempts']} [{r['code_info']}]")
    if len(speedups) >= 2:
        import statistics
        print(f"  Speedup range: {min(speedups):.2f}x - {max(speedups):.2f}x (spread: {max(speedups)-min(speedups):.2f}x)")
        print(f"  Triton time range: {min(triton_times):.3f}ms - {max(triton_times):.3f}ms (ratio: {max(triton_times)/min(triton_times):.1f}x)")

# Save raw results
with open("../results/polybench/polybench_results/nondeterminism_evidence.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nRaw results saved to polybench_results/nondeterminism_evidence.json")
