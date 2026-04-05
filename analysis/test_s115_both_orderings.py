#!/usr/bin/env python3
"""
Test both parallelization orderings for s115 (triangular saxpy loop).

Original C code:
    for (int j = 0; j < LEN_2D; j++) {
        for (int i = j+1; i < LEN_2D; i++) {
            a[i] -= aa[j][i] * a[j];
        }
    }

Two valid orderings (both using MULTI-KERNEL for correctness):
1. j-sequential (N kernels), i-parallel: Each kernel processes all valid i in parallel
2. i-sequential (N kernels), j-parallel: Each kernel reduces over all valid j

Both are CORRECT because kernel launches are synchronous.
Question: Which is FASTER?
"""

import torch
import triton
import triton.language as tl
import time

# ============================================================================
# OPTION 1: j-sequential, i-parallel (Independent Writes)
# ============================================================================
# For each j (sequential), all i > j can run in parallel since they write
# to different a[i] locations.

@triton.jit
def s115_j_seq_i_par_kernel(
    a_ptr,
    aa_ptr,
    N,
    j,  # Current j iteration (passed from Python loop)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Process one j iteration with i parallelized across blocks.
    Each block handles a chunk of i values where i > j.
    """
    block_id = tl.program_id(0)
    i_start = j + 1 + block_id * BLOCK_SIZE
    i_offsets = i_start + tl.arange(0, BLOCK_SIZE)

    # Mask for valid i values
    mask = i_offsets < N

    # Load a[j] - same for all threads
    a_j = tl.load(a_ptr + j)

    # Load aa[j, i] and a[i]
    aa_offsets = j * N + i_offsets
    aa_ji = tl.load(aa_ptr + aa_offsets, mask=mask, other=0.0)
    a_i = tl.load(a_ptr + i_offsets, mask=mask, other=0.0)

    # Compute and store
    result = a_i - aa_ji * a_j
    tl.store(a_ptr + i_offsets, result, mask=mask)


def s115_j_seq_i_par(a, aa, BLOCK_SIZE=256):
    """
    Option 1: j-sequential (Python loop), i-parallel (Triton kernel).
    Launches N kernels, one per j iteration.
    """
    N = aa.shape[0]

    for j in range(N - 1):  # j from 0 to N-2
        num_i = N - (j + 1)  # Number of i values to process
        if num_i > 0:
            grid = (triton.cdiv(num_i, BLOCK_SIZE),)
            s115_j_seq_i_par_kernel[grid](a, aa, N, j, BLOCK_SIZE=BLOCK_SIZE)

    return a


# ============================================================================
# OPTION 1b: j-sequential IN-KERNEL, i-parallel (Single Kernel Launch)
# ============================================================================
# Same logic but with in-kernel loop over j - only ONE kernel launch!

@triton.jit
def s115_j_seq_i_par_single_kernel(
    a_ptr,
    aa_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Single kernel: in-kernel loop over j, blocks parallelize i.
    Each block handles a contiguous chunk of i values.
    """
    block_id = tl.program_id(0)
    i_base = block_id * BLOCK_SIZE
    i_offsets = i_base + tl.arange(0, BLOCK_SIZE)

    # Sequential loop over j
    for j in range(N - 1):
        # Valid mask: i > j and i < N
        mask = (i_offsets > j) & (i_offsets < N)

        # Load a[j] - broadcast to all threads
        a_j = tl.load(a_ptr + j)

        # Load aa[j, i] and a[i]
        aa_offsets = j * N + i_offsets
        aa_ji = tl.load(aa_ptr + aa_offsets, mask=mask, other=0.0)
        a_i = tl.load(a_ptr + i_offsets, mask=mask, other=0.0)

        # Compute and store
        result = a_i - aa_ji * a_j
        tl.store(a_ptr + i_offsets, result, mask=mask)


def s115_j_seq_i_par_single(a, aa, BLOCK_SIZE=256, num_blocks=None):
    """
    Option 1b: Single kernel with in-kernel j loop.

    WARNING: With multiple blocks, there's a race condition!
    - Block 0 writes a[j] at iterations 0..j-1
    - Block 1 reads a[j] at iteration j
    - If Block 1 reaches j before Block 0 finishes j-1, wrong result!

    Use num_blocks=1 for correctness (sacrifices parallelism).
    """
    N = aa.shape[0]
    if num_blocks is None:
        num_blocks = triton.cdiv(N, BLOCK_SIZE)
    grid = (num_blocks,)
    s115_j_seq_i_par_single_kernel[grid](a, aa, N, BLOCK_SIZE=BLOCK_SIZE)
    return a


# ============================================================================
# OPTION 2: i-sequential, j-parallel (Reduction) - MULTI-KERNEL VERSION
# ============================================================================
# For each i (sequential kernel launch), reduce over all j < i in parallel.
# Reordered loop:
#   for i in 1..N-1:       <- sequential kernel launches
#       for j in 0..i-1:   <- parallel reduction
#           a[i] -= aa[j][i] * a[j]

@triton.jit
def s115_i_seq_j_par_kernel(
    a_ptr,
    aa_ptr,
    i,  # Current i value (passed from Python loop)
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Process one i value, reducing over all j < i in parallel.
    Uses block-level reduction with tl.sum().
    """
    # Reduction over j from 0 to i-1
    accumulator = 0.0

    # Process j in chunks (in case i > BLOCK_SIZE)
    for j_base in range(0, i, BLOCK_SIZE):
        j_offsets = j_base + tl.arange(0, BLOCK_SIZE)
        mask = j_offsets < i

        # Load aa[j, i] - note: aa is [j][i] so offset is j * N + i
        aa_offsets = j_offsets * N + i
        aa_ji = tl.load(aa_ptr + aa_offsets, mask=mask, other=0.0)

        # Load a[j]
        a_j = tl.load(a_ptr + j_offsets, mask=mask, other=0.0)

        # Accumulate partial sum using parallel reduction
        partial = tl.sum(aa_ji * a_j)
        accumulator += partial

    # Update a[i] = a[i] - accumulator
    a_i = tl.load(a_ptr + i)
    tl.store(a_ptr + i, a_i - accumulator)


def s115_i_seq_j_par(a, aa, BLOCK_SIZE=256):
    """
    Option 2: i-sequential (kernel launches), j-parallel (reduction).
    Launches N-1 kernels, one per i value.
    Each kernel reduces over j=0..i-1 in parallel.

    CORRECT: No race condition because kernels are launched sequentially.
    """
    N = aa.shape[0]

    for i in range(1, N):  # i from 1 to N-1
        # Launch kernel for this i value
        # Single program since reduction happens within the kernel
        s115_i_seq_j_par_kernel[(1,)](a, aa, i, N, BLOCK_SIZE=BLOCK_SIZE)

    return a


# ============================================================================
# Reference Implementation (PyTorch)
# ============================================================================

def s115_reference(a, aa):
    """Reference implementation in PyTorch."""
    N = aa.shape[0]
    a = a.clone()
    for j in range(N):
        for i in range(j + 1, N):
            a[i] -= aa[j, i] * a[j]
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
    a_init = torch.randn(N, device='cuda', dtype=torch.float32) * 0.1
    aa = torch.randn(N, N, device='cuda', dtype=torch.float32) * 0.1

    # Reference
    a_ref = s115_reference(a_init.clone(), aa)
    print(f"\nReference result range: [{a_ref.min().item():.4f}, {a_ref.max().item():.4f}]")

    tol = 1e-3  # Relaxed tolerance for float32

    # Option 1: j-seq, i-par (multiple kernels) - N-1 kernel launches
    a_opt1 = s115_j_seq_i_par(a_init.clone(), aa)
    diff1 = torch.max(torch.abs(a_opt1 - a_ref)).item()
    print(f"\nOption 1 (j-seq, i-par, {N-1} kernels): max diff = {diff1:.6e} {'PASS' if diff1 < tol else 'FAIL'}")

    # Option 2: i-seq, j-par (reduction, multiple kernels) - N-1 kernel launches
    a_opt2 = s115_i_seq_j_par(a_init.clone(), aa)
    diff2 = torch.max(torch.abs(a_opt2 - a_ref)).item()
    print(f"Option 2 (i-seq, j-par, {N-1} kernels): max diff = {diff2:.6e} {'PASS' if diff2 < tol else 'FAIL'}")

    # Debug: check first few elements if failures
    if diff1 > tol or diff2 > tol:
        print("\nDebug - First 10 elements:")
        print(f"  Reference: {a_ref[:10].cpu().numpy()}")
        print(f"  Option 1:  {a_opt1[:10].cpu().numpy()}")
        print(f"  Option 2:  {a_opt2[:10].cpu().numpy()}")

    print(f"\n{'='*60}")
    print("Both options use sequential kernel launches -> NO RACE CONDITION")
    print("Option 1: parallelize i (independent writes to different a[i])")
    print("Option 2: parallelize j (reduction into single a[i])")
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
    a_init = torch.randn(N, device='cuda', dtype=torch.float32) * 0.1
    aa = torch.randn(N, N, device='cuda', dtype=torch.float32) * 0.1

    results = {}

    # Benchmark Option 1: j-seq, i-par (multiple kernels)
    for _ in range(warmup):
        a = a_init.clone()
        s115_j_seq_i_par(a, aa)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        a = a_init.clone()
        s115_j_seq_i_par(a, aa)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    results['Option1: j-seq, i-par'] = elapsed / iterations * 1000

    # Benchmark Option 2: i-seq, j-par (reduction)
    for _ in range(warmup):
        a = a_init.clone()
        s115_i_seq_j_par(a, aa)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        a = a_init.clone()
        s115_i_seq_j_par(a, aa)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    results['Option2: i-seq, j-par'] = elapsed / iterations * 1000

    # Print results
    print(f"\nResults (average over {iterations} iterations):")
    print("-" * 50)

    fastest = min(results.values())
    opt1_time = results['Option1: j-seq, i-par']
    opt2_time = results['Option2: i-seq, j-par']

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
    print(f"  Option 1 (j-seq, i-par):")
    print(f"    - j=0: {N-1} parallel i values")
    print(f"    - j={N//2}: {N-1-N//2} parallel i values")
    print(f"    - j={N-2}: 1 parallel i value")
    print(f"    - Work decreases as j increases")

    print(f"  Option 2 (i-seq, j-par):")
    print(f"    - i=1: reduce over 1 j value")
    print(f"    - i={N//2}: reduce over {N//2} j values")
    print(f"    - i={N-1}: reduce over {N-2} j values")
    print(f"    - Work increases as i increases")

    return results


def analyze_work_distribution(N=256):
    """Analyze work distribution for both orderings."""
    print(f"\n{'='*60}")
    print(f"WORK DISTRIBUTION ANALYSIS (N={N})")
    print(f"{'='*60}")

    # Option 1: j-sequential, i-parallel
    print("\nOption 1: j-sequential, i-parallel (independent writes)")
    print("-" * 50)
    total_work = 0
    print("  j=0: process i=1..N-1 -> {} parallel elements".format(N-1))
    print("  j=1: process i=2..N-1 -> {} parallel elements".format(N-2))
    print("  ...")
    print("  j=N-2: process i=N-1 -> 1 parallel element")
    for j in range(N-1):
        total_work += (N - j - 1)
    print(f"  Total iterations: {total_work}")
    print(f"  Kernel launches (multi-kernel): {N-1}")
    print(f"  Kernel launches (single-kernel): 1")
    print(f"  Work per j iteration: decreasing from {N-1} to 1")

    # Option 2: i-sequential, j-parallel
    print("\nOption 2: i-sequential, j-parallel (reduction)")
    print("-" * 50)
    print("  i=1: reduce over j=0 -> 1 element reduction")
    print("  i=2: reduce over j=0,1 -> 2 element reduction")
    print("  ...")
    print("  i=N-1: reduce over j=0..N-2 -> {} element reduction".format(N-1))
    print(f"  Total iterations: {total_work}")
    print(f"  Programs launched: {N-1} (one per i)")
    print(f"  Work per program: increasing from 1 to {N-1}")

    print("\n" + "="*60)
    print("KEY INSIGHT:")
    print("="*60)
    print("""
Option 1 (j-seq, i-par):
  - Early j iterations have MORE parallel work
  - Single-kernel version: all blocks active early, fewer active later
  - Good memory coalescing for a[i] writes (consecutive)

Option 2 (i-seq, j-par):
  - Each i is an independent program (N-1 programs total)
  - Early programs (small i) have LESS work (small reductions)
  - Later programs (large i) have MORE work (large reductions)
  - Load imbalance: early programs finish fast, later ones take longer
  - BUT: all programs can run in parallel if enough SMs!

For this triangular pattern:
  - If N is small: Option 2 may win (all i programs fit on GPU simultaneously)
  - If N is large: Option 1 single-kernel may win (better work distribution)
""")


if __name__ == "__main__":
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit(1)

    print(f"Using GPU: {torch.cuda.get_device_name()}")

    # Test correctness
    if not test_correctness(N=256):
        print("\nCORRECTNESS TEST FAILED!")
        exit(1)

    # Benchmark at different sizes
    for N in [64, 128, 256, 512]:
        benchmark(N=N, warmup=5, iterations=50)

    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("""
For s115-like triangular loops with cross-iteration dependencies:

1. BOTH orderings are CORRECT with multi-kernel approach (sequential launches)
   - No race condition because each kernel completes before next launches

2. Performance comparison:
   - Option 1 (j-seq, i-par): More parallel work early, less work later
   - Option 2 (i-seq, j-par): Less parallel work early, more work later
   - Both have same number of kernel launches (N-1)
   - Main difference: independent writes vs reduction overhead

3. The choice depends on:
   - N size
   - Reduction overhead (tl.sum)
   - Memory access patterns
""")
