#!/usr/bin/env python3
"""
Large-size ablation: benchmark BOTH WA and NA Triton code at scaled problem sizes.

For each kernel that passes in both WA and NA pipelines, this script:
1. Loads the WA Triton code and benchmarks at scaled size
2. Loads the NA Triton code and benchmarks at scaled size
3. Compares WA vs NA Triton time at large sizes

This complements the small-size ablation by showing whether analysis-guided
code maintains its advantage (or disadvantage) at realistic GPU workloads.

Usage:
    python benchmark_large_sizes_ablation.py --all
    python benchmark_large_sizes_ablation.py gemm heat_3d trmm
"""
import sys
import json
from pathlib import Path

# Import everything from benchmark_large_sizes
from benchmark_large_sizes import (
    RESULTS_DIR, SCALE_FACTOR, SEQUENTIAL_SCALE, SEQUENTIAL_KERNELS,
    KERNEL_TIMEOUT, load_triton_func, get_scaled_params, get_original_params,
    create_tensors, build_args, POLYBENCH_FUNCTIONS,
    _timeout_handler, TimeoutError,
)
import signal
import time
import torch


def benchmark_kernel_with_mode(kernel_name: str, scale: int, use_analysis: bool,
                                num_warmup=5, num_iterations=20):
    """Benchmark a kernel at scaled size using either WA or NA code."""
    func, code_path = load_triton_func(kernel_name, use_analysis=use_analysis)
    if func is None:
        return None

    func_spec = POLYBENCH_FUNCTIONS.get(kernel_name, {})
    arrays = func_spec.get('arrays', {})
    scalar_params = func_spec.get('scalar_params', {})

    params = get_scaled_params(kernel_name, scale)
    if not params:
        return None

    tensors = create_tensors(kernel_name, params, arrays)
    args = build_args(func, tensors, params, scalar_params)

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(KERNEL_TIMEOUT)

    # Warmup
    try:
        for _ in range(num_warmup):
            func(*args)
        torch.cuda.synchronize()
    except TimeoutError:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return {"error": f"timeout ({KERNEL_TIMEOUT}s)", "phase": "warmup"}
    except Exception as e:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return {"error": str(e), "phase": "warmup"}

    # Timed runs
    try:
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            func(*args)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / num_iterations
        triton_ms = elapsed * 1000
    except TimeoutError:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return {"error": f"timeout ({KERNEL_TIMEOUT}s)", "phase": "benchmark"}
    except Exception as e:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return {"error": str(e), "phase": "benchmark"}

    signal.alarm(0)
    signal.signal(signal.SIGALRM, old_handler)

    return {"triton_time_ms": triton_ms, "params": params, "scale": scale}


def main():
    # Load both WA and NA results to find dual-passing kernels
    with open(RESULTS_DIR / "results.json") as f:
        wa_results = json.load(f)
    with open(RESULTS_DIR / "results_no_analysis.json") as f:
        na_results = json.load(f)

    wa_passed = {k for k, v in wa_results.items() if v.get("test_passed")}
    na_passed = {k for k, v in na_results.items() if v.get("test_passed")}
    dual_passed = sorted(wa_passed & na_passed)

    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        kernels = dual_passed
    elif len(sys.argv) > 1:
        kernels = [k for k in sys.argv[1:] if k in dual_passed]
    else:
        kernels = dual_passed

    print("=" * 120)
    print(f"LARGE-SIZE ABLATION: WA vs NA Triton Code (parallel={SCALE_FACTOR}x, sequential={SEQUENTIAL_SCALE}x)")
    print(f"Dual-passing kernels: {len(dual_passed)} | Benchmarking: {len(kernels)}")
    print(f"Timeout: {KERNEL_TIMEOUT}s per kernel per mode")
    print("=" * 120)
    print()

    header = (f"{'Kernel':<15} {'Scale':>5} "
              f"{'WA_sm':>8} {'NA_sm':>8} "
              f"{'WA_lg':>9} {'NA_lg':>9} "
              f"{'Lg WA/NA':>9} {'Winner':>8}")
    print(header)
    print("-" * len(header))

    all_results = {}

    for kernel_name in kernels:
        scale = SEQUENTIAL_SCALE if kernel_name in SEQUENTIAL_KERNELS else SCALE_FACTOR

        # Small-size Triton times from existing results
        wa_small_t = wa_results[kernel_name].get("benchmark", {}).get("triton_time_ms", 0)
        na_small_t = na_results[kernel_name].get("benchmark", {}).get("triton_time_ms", 0)

        # Benchmark WA at large size
        wa_result = benchmark_kernel_with_mode(kernel_name, scale, use_analysis=True)
        # Benchmark NA at large size
        na_result = benchmark_kernel_with_mode(kernel_name, scale, use_analysis=False)

        wa_large_t = None
        na_large_t = None
        wa_err = None
        na_err = None

        if wa_result and "error" not in wa_result:
            wa_large_t = wa_result["triton_time_ms"]
        elif wa_result:
            wa_err = wa_result.get("error", "")[:30]

        if na_result and "error" not in na_result:
            na_large_t = na_result["triton_time_ms"]
        elif na_result:
            na_err = na_result.get("error", "")[:30]

        # Compute ratio and winner
        ratio_str = ""
        winner = ""
        if wa_large_t is not None and na_large_t is not None:
            ratio = wa_large_t / na_large_t if na_large_t > 0 else float('inf')
            ratio_str = f"{ratio:.2f}x"
            if ratio < 0.95:
                winner = "WA"
            elif ratio > 1.05:
                winner = "NA"
            else:
                winner = "TIE"
        elif wa_large_t is not None:
            winner = "WA-only"
        elif na_large_t is not None:
            winner = "NA-only"
        else:
            winner = "BOTH-ERR"

        wa_lg_str = f"{wa_large_t:.3f}ms" if wa_large_t else (wa_err or "N/A")
        na_lg_str = f"{na_large_t:.3f}ms" if na_large_t else (na_err or "N/A")

        print(f"{kernel_name:<15} {scale:>4}x "
              f"{wa_small_t:>7.3f}  {na_small_t:>7.3f}  "
              f"{wa_lg_str:>9} {na_lg_str:>9} "
              f"{ratio_str:>9} {winner:>8}")

        entry = {
            "scale_factor": scale,
            "wa_small_triton_ms": wa_small_t,
            "na_small_triton_ms": na_small_t,
            "winner": winner,
        }
        if wa_large_t is not None:
            entry["wa_large_triton_ms"] = wa_large_t
        else:
            entry["wa_large_error"] = wa_err or "load failed"
        if na_large_t is not None:
            entry["na_large_triton_ms"] = na_large_t
        else:
            entry["na_large_error"] = na_err or "load failed"
        if wa_large_t is not None and na_large_t is not None and na_large_t > 0:
            entry["large_wa_na_ratio"] = wa_large_t / na_large_t

        all_results[kernel_name] = entry

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    both_ok = {k: v for k, v in all_results.items()
               if "wa_large_triton_ms" in v and "na_large_triton_ms" in v}

    wa_wins = sum(1 for v in both_ok.values() if v["winner"] == "WA")
    na_wins = sum(1 for v in both_ok.values() if v["winner"] == "NA")
    ties = sum(1 for v in both_ok.values() if v["winner"] == "TIE")
    wa_only = sum(1 for v in all_results.values() if v["winner"] == "WA-only")
    na_only = sum(1 for v in all_results.values() if v["winner"] == "NA-only")

    print(f"Both completed: {len(both_ok)}/{len(all_results)}")
    print(f"WA wins (faster): {wa_wins}")
    print(f"NA wins (faster): {na_wins}")
    print(f"Ties (within 5%): {ties}")
    if wa_only:
        print(f"WA-only pass:     {wa_only}")
    if na_only:
        print(f"NA-only pass:     {na_only}")

    # Compare with small-size ratios
    if both_ok:
        print(f"\nSmall vs Large WA/NA ratio comparison:")
        print(f"{'Kernel':<15} {'Small WA/NA':>12} {'Large WA/NA':>12} {'Change':>10}")
        print("-" * 50)
        for k in sorted(both_ok.keys()):
            v = both_ok[k]
            sm_ratio = v["wa_small_triton_ms"] / v["na_small_triton_ms"] if v["na_small_triton_ms"] > 0 else 0
            lg_ratio = v.get("large_wa_na_ratio", 0)
            change = "improved" if lg_ratio < sm_ratio else "worsened"
            print(f"{k:<15} {sm_ratio:>11.2f}x {lg_ratio:>11.2f}x {change:>10}")

    # Save results (merge with existing)
    output_file = RESULTS_DIR / "results_large_sizes_ablation.json"
    existing = {}
    if output_file.exists():
        with open(output_file) as f:
            existing = json.load(f)
    existing.update(all_results)
    with open(output_file, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
