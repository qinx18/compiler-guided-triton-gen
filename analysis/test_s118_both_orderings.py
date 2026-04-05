#!/usr/bin/env python3
"""
Test both parallelization orderings for s118 (dot product recursion).

Original C code:
    for (int i = 1; i < LEN_2D; i++) {
        for (int j = 0; j <= i - 1; j++) {
            a[i] += bb[j][i] * a[i-j-1];
        }
    }

Two valid orderings (both using MULTI-KERNEL for correctness):
1. i-sequential (N-1 kernels), j-parallel: Each kernel reduces over all valid j (EXISTING)
2. j-sequential (N-1 kernels), i-parallel: Each kernel processes all valid i in parallel

Both are CORRECT because kernel launches are synchronous.
Question: Which is FASTER?
"""

import torch
import triton
import triton.language as tl
import time

# ============================================================================
# OPTION 1: i-sequential, j-parallel (Reduction) - EXISTING APPROACH
# ============================================================================
# For each i (sequential), reduce over all j < i in parallel.

@triton.jit
def s118_i_seq_j_par_kernel(
    a_ptr,
    bb_ptr,
    n,
    i_val,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Process one i value, reducing over all j < i in parallel.
    Uses block-level reduction with tl.sum().
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    j_offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask for valid j values (j < i_val)
    mask = j_offsets < i_val

    # Load bb[j, i_val] values
    bb_offsets = j_offsets * n + i_val
    bb_vals = tl.load(bb_ptr + bb_offsets, mask=mask, other=0.0)

    # Load a[i_val - j - 1] values
    a_offsets = i_val - j_offsets - 1
    a_vals = tl.load(a_ptr + a_offsets, mask=mask, other=0.0)

    # Compute partial products and sum
    products = bb_vals * a_vals
    result = tl.sum(products, axis=0)

    # Atomically add to a[i_val]
    tl.atomic_add(a_ptr + i_val, result)


def s118_i_seq_j_par(a, bb, BLOCK_SIZE=128):
    """
    Option 1: i-sequential (kernel launches), j-parallel (reduction).
    Launches N-1 kernels, one per i value.
    """
    n = a.size(0)

    for i in range(1, n):
        num_j = i
        grid_size = triton.cdiv(num_j, BLOCK_SIZE)
        s118_i_seq_j_par_kernel[(grid_size,)](a, bb, n, i, BLOCK_SIZE=BLOCK_SIZE)

    return a


# ============================================================================
# OPTION 2: j-sequential, i-parallel (Independent Writes)
# ============================================================================
# For each j (sequential), all valid i can run in parallel.
# Reordered loop:
#   for j in 0..N-2:         <- sequential kernel launches
#       for i in j+1..N-1:   <- parallel
#           a[i] += bb[j][i] * a[i-j-1]

@triton.jit
def s118_j_seq_i_par_kernel(
    a_ptr,
    bb_ptr,
    n,
    j_val,  # Current j iteration
    BLOCK_SIZE: tl.constexpr,
):
    """
    Process one j iteration with i parallelized across blocks.
    Each block handles a chunk of i values where i > j.
    """
    block_id = tl.program_id(0)
    i_start = j_val + 1 + block_id * BLOCK_SIZE
    i_offsets = i_start + tl.arange(0, BLOCK_SIZE)

    # Mask for valid i values (i < n)
    mask = i_offsets < n

    # Load bb[j_val, i] - bb is [j][i] so offset is j_val * n + i
    bb_offsets = j_val * n + i_offsets
    bb_ji = tl.load(bb_ptr + bb_offsets, mask=mask, other=0.0)

    # Load a[i - j_val - 1]
    a_read_offsets = i_offsets - j_val - 1
    a_read = tl.load(a_ptr + a_read_offsets, mask=mask, other=0.0)

    # Load current a[i]
    a_i = tl.load(a_ptr + i_offsets, mask=mask, other=0.0)

    # Compute and store: a[i] += bb[j][i] * a[i-j-1]
    result = a_i + bb_ji * a_read
    tl.store(a_ptr + i_offsets, result, mask=mask)


def s118_j_seq_i_par(a, bb, BLOCK_SIZE=256):
    """
    Option 2: j-sequential (Python loop), i-parallel (Triton kernel).
    Launches N-1 kernels, one per j iteration.
    """
    n = bb.shape[0]

    for j in range(n - 1):  # j from 0 to N-2
        num_i = n - (j + 1)  # Number of i values to process
        if num_i > 0:
            grid = (triton.cdiv(num_i, BLOCK_SIZE),)
            s118_j_seq_i_par_kernel[grid](a, bb, n, j, BLOCK_SIZE=BLOCK_SIZE)

    return a


# ============================================================================
# Reference Implementation (PyTorch)
# ============================================================================

def s118_reference(a, bb):
    """Reference implementation in PyTorch."""
    n = a.size(0)
    a = a.clone()
    for i in range(1, n):
        for j in range(i):
            a[i] += bb[j, i] * a[i - j - 1]
    return a


# ============================================================================
# Testing and Benchmarking
# ============================================================================

def test_correctness(N=256):
    """Test that both multi-kernel implementations produce correct results."""
    print(f"\n{'='*60}")
    print(f"CORRECTNESS TEST (N={N})")
    print(f"{'='*60}")

    # Initialize with smaller values to avoid numerical overflow
    torch.manual_seed(42)
    a_init = torch.ones(N, device='cuda', dtype=torch.float32)
    bb = torch.randn(N, N, device='cuda', dtype=torch.float32) * 0.01

    # Reference
    a_ref = s118_reference(a_init.clone(), bb)
    print(f"\nReference result range: [{a_ref.min().item():.4f}, {a_ref.max().item():.4f}]")

    tol = 1e-3  # Relaxed tolerance for float32

    # Option 1: i-seq, j-par (reduction) - N-1 kernel launches
    a_opt1 = s118_i_seq_j_par(a_init.clone(), bb)
    diff1 = torch.max(torch.abs(a_opt1 - a_ref)).item()
    print(f"\nOption 1 (i-seq, j-par, {N-1} kernels): max diff = {diff1:.6e} {'PASS' if diff1 < tol else 'FAIL'}")

    # Option 2: j-seq, i-par (independent writes) - N-1 kernel launches
    a_opt2 = s118_j_seq_i_par(a_init.clone(), bb)
    diff2 = torch.max(torch.abs(a_opt2 - a_ref)).item()
    print(f"Option 2 (j-seq, i-par, {N-1} kernels): max diff = {diff2:.6e} {'PASS' if diff2 < tol else 'FAIL'}")

    # Debug: check first few elements if failures
    if diff1 > tol or diff2 > tol:
        print("\nDebug - First 10 elements:")
        print(f"  Reference: {a_ref[:10].cpu().numpy()}")
        print(f"  Option 1:  {a_opt1[:10].cpu().numpy()}")
        print(f"  Option 2:  {a_opt2[:10].cpu().numpy()}")

    print(f"\n{'='*60}")
    print("Both options use sequential kernel launches -> NO RACE CONDITION")
    print("Option 1: parallelize j (reduction into single a[i])")
    print("Option 2: parallelize i (independent writes to different a[i])")
    print(f"{'='*60}")

    return diff1 < tol and diff2 < tol


def benchmark(N=256, warmup=10, iterations=100):
    """Benchmark both multi-kernel implementations."""
    print(f"\n{'='*60}")
    print(f"PERFORMANCE BENCHMARK (N={N})")
    print(f"Both use {N-1} sequential kernel launches")
    print(f"{'='*60}")

    # Initialize
    torch.manual_seed(42)
    a_init = torch.ones(N, device='cuda', dtype=torch.float32)
    bb = torch.randn(N, N, device='cuda', dtype=torch.float32) * 0.01

    results = {}

    # Benchmark Option 1: i-seq, j-par (reduction)
    for _ in range(warmup):
        a = a_init.clone()
        s118_i_seq_j_par(a, bb)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        a = a_init.clone()
        s118_i_seq_j_par(a, bb)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    results['Option1: i-seq, j-par'] = elapsed / iterations * 1000

    # Benchmark Option 2: j-seq, i-par
    for _ in range(warmup):
        a = a_init.clone()
        s118_j_seq_i_par(a, bb)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        a = a_init.clone()
        s118_j_seq_i_par(a, bb)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    results['Option2: j-seq, i-par'] = elapsed / iterations * 1000

    # Print results
    print(f"\nResults (average over {iterations} iterations):")
    print("-" * 50)

    fastest = min(results.values())
    opt1_time = results['Option1: i-seq, j-par']
    opt2_time = results['Option2: j-seq, i-par']

    for name, time_ms in sorted(results.items(), key=lambda x: x[1]):
        marker = " <-- FASTER" if time_ms == fastest else ""
        print(f"  {name}: {time_ms:.4f} ms{marker}")

    ratio = opt1_time / opt2_time
    if ratio > 1:
        print(f"\nOption 2 is {ratio:.2f}x faster than Option 1")
    else:
        print(f"\nOption 1 is {1/ratio:.2f}x faster than Option 2")

    # Analysis
    print(f"\nWork distribution analysis:")
    print(f"  Option 1 (i-seq, j-par - reduction):")
    print(f"    - i=1: reduce over 1 j value")
    print(f"    - i={N//2}: reduce over {N//2} j values")
    print(f"    - i={N-1}: reduce over {N-2} j values")
    print(f"    - Work increases as i increases")

    print(f"  Option 2 (j-seq, i-par - independent writes):")
    print(f"    - j=0: {N-1} parallel i values")
    print(f"    - j={N//2}: {N-1-N//2} parallel i values")
    print(f"    - j={N-2}: 1 parallel i value")
    print(f"    - Work decreases as j increases")

    return results


if __name__ == "__main__":
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit(1)

    print(f"Using GPU: {torch.cuda.get_device_name()}")

    # Test correctness
    test_correctness(N=256)

    print("\n" + "="*60)
    print("IMPORTANT FINDING: s118 vs s115 DEPENDENCY DIFFERENCE")
    print("="*60)
    print("""
s118: a[i] += bb[j][i] * a[i-j-1]

At j=0 (trying to parallelize over i):
  i=1: read a[0], write a[1]
  i=2: read a[1], write a[2]  <-- DEPENDS on i=1's write!
  i=3: read a[2], write a[3]  <-- DEPENDS on i=2's write!

Within a SINGLE j iteration, different i values have RAW dependencies!
This is DIFFERENT from s115 where i values are independent.

s115: a[i] -= aa[j][i] * a[j]
  - All i values read the SAME a[j] (broadcast read)
  - All i values write to DIFFERENT a[i] (independent writes)
  -> Can parallelize over i for fixed j

s118: a[i] += bb[j][i] * a[i-j-1]
  - i=k reads a[k-j-1]
  - i=k+1 reads a[k-j], which was written by some earlier i
  -> CANNOT parallelize over i for fixed j (chain dependency)

CONCLUSION:
- s118 can ONLY be parallelized over j (reduction into a[i])
- s115 can be parallelized over EITHER dimension
- The loop reordering that works for s115 does NOT work for s118
""")
