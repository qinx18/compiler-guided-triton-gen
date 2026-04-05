"""Test both parallelization orderings for s119 to verify the analysis"""
import torch
import triton
import triton.language as tl
import sys
sys.path.insert(0, 'baselines')
from s119_baseline import s119_pytorch

# Option 1: i-sequential, j-parallel
@triton.jit
def s119_kernel_i_seq_j_par(
    aa_ptr, bb_ptr, M, N, i_val,
    stride_aa_0, stride_aa_1,
    stride_bb_0, stride_bb_1,
    BLOCK_SIZE: tl.constexpr,
):
    """i-sequential, j-parallel version"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # j offsets (j starts from 1)
    j_offsets = block_start + tl.arange(0, BLOCK_SIZE) + 1
    mask = j_offsets < N

    # aa[i, j] = aa[i-1, j-1] + bb[i, j]
    aa_curr_addrs = aa_ptr + i_val * stride_aa_0 + j_offsets * stride_aa_1
    aa_prev_addrs = aa_ptr + (i_val - 1) * stride_aa_0 + (j_offsets - 1) * stride_aa_1
    bb_curr_addrs = bb_ptr + i_val * stride_bb_0 + j_offsets * stride_bb_1

    aa_prev = tl.load(aa_prev_addrs, mask=mask, other=0.0)
    bb_curr = tl.load(bb_curr_addrs, mask=mask, other=0.0)
    result = aa_prev + bb_curr
    tl.store(aa_curr_addrs, result, mask=mask)


def s119_triton_option1(aa, bb):
    """i-sequential, j-parallel"""
    aa = aa.contiguous()
    bb = bb.contiguous()
    M, N = aa.shape
    BLOCK_SIZE = 256

    for i in range(1, M):
        num_j = N - 1
        grid_size = triton.cdiv(num_j, BLOCK_SIZE)
        s119_kernel_i_seq_j_par[(grid_size,)](
            aa, bb, M, N, i,
            aa.stride(0), aa.stride(1),
            bb.stride(0), bb.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return aa


# Option 2: j-sequential, i-parallel
@triton.jit
def s119_kernel_j_seq_i_par(
    aa_ptr, bb_ptr, M, N, j_val,
    stride_aa_0, stride_aa_1,
    stride_bb_0, stride_bb_1,
    BLOCK_SIZE: tl.constexpr,
):
    """j-sequential, i-parallel version"""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # i offsets (i starts from 1)
    i_offsets = block_start + tl.arange(0, BLOCK_SIZE) + 1
    mask = i_offsets < M

    # aa[i, j] = aa[i-1, j-1] + bb[i, j]
    aa_curr_addrs = aa_ptr + i_offsets * stride_aa_0 + j_val * stride_aa_1
    aa_prev_addrs = aa_ptr + (i_offsets - 1) * stride_aa_0 + (j_val - 1) * stride_aa_1
    bb_curr_addrs = bb_ptr + i_offsets * stride_bb_0 + j_val * stride_bb_1

    aa_prev = tl.load(aa_prev_addrs, mask=mask, other=0.0)
    bb_curr = tl.load(bb_curr_addrs, mask=mask, other=0.0)
    result = aa_prev + bb_curr
    tl.store(aa_curr_addrs, result, mask=mask)


def s119_triton_option2(aa, bb):
    """j-sequential, i-parallel"""
    aa = aa.contiguous()
    bb = bb.contiguous()
    M, N = aa.shape
    BLOCK_SIZE = 256

    for j in range(1, N):
        num_i = M - 1
        grid_size = triton.cdiv(num_i, BLOCK_SIZE)
        s119_kernel_j_seq_i_par[(grid_size,)](
            aa, bb, M, N, j,
            aa.stride(0), aa.stride(1),
            bb.stride(0), bb.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
        )
    return aa


def test_correctness():
    print("=" * 60)
    print("Testing s119 correctness for both orderings")
    print("=" * 60)
    print()
    print("s119 pattern: aa[i][j] = aa[i-1][j-1] + bb[i][j]")
    print("Diagonal dependency: read (i-1, j-1), write (i, j)")
    print()

    N = 256
    aa_orig = torch.randn(N, N, device='cuda', dtype=torch.float32)
    bb = torch.randn(N, N, device='cuda', dtype=torch.float32)

    # Reference
    result_pytorch = s119_pytorch(aa_orig.clone(), bb)

    # Option 1: i-seq, j-par
    result_opt1 = s119_triton_option1(aa_orig.clone(), bb)
    diff1 = (result_pytorch - result_opt1).abs().max().item()
    match1 = torch.allclose(result_pytorch, result_opt1, atol=1e-5)

    # Option 2: j-seq, i-par
    result_opt2 = s119_triton_option2(aa_orig.clone(), bb)
    diff2 = (result_pytorch - result_opt2).abs().max().item()
    match2 = torch.allclose(result_pytorch, result_opt2, atol=1e-5)

    print("Correctness Results:")
    print(f"  Option 1 (i-seq, j-par): {'PASS' if match1 else 'FAIL'} (max diff: {diff1:.6f})")
    print(f"  Option 2 (j-seq, i-par): {'PASS' if match2 else 'FAIL'} (max diff: {diff2:.6f})")
    print()

    if match1 and match2:
        print("CONCLUSION: Both orderings are VALID for s119!")
        print("(Diagonal dependency allows parallelizing either dimension)")
    elif match1:
        print("CONCLUSION: Only i-seq, j-par is valid for s119")
    elif match2:
        print("CONCLUSION: Only j-seq, i-par is valid for s119")
    else:
        print("CONCLUSION: Neither ordering is valid (unexpected!)")


def test_performance():
    print()
    print("=" * 60)
    print("Performance comparison")
    print("=" * 60)

    N = 256
    aa_orig = torch.randn(N, N, device='cuda', dtype=torch.float32)
    bb = torch.randn(N, N, device='cuda', dtype=torch.float32)

    # Warmup
    for _ in range(3):
        s119_triton_option1(aa_orig.clone(), bb)
        s119_triton_option2(aa_orig.clone(), bb)
    torch.cuda.synchronize()

    # Benchmark
    import time

    n_iter = 100

    start = time.perf_counter()
    for _ in range(n_iter):
        s119_triton_option1(aa_orig.clone(), bb)
    torch.cuda.synchronize()
    time_opt1 = (time.perf_counter() - start) / n_iter * 1000

    start = time.perf_counter()
    for _ in range(n_iter):
        s119_triton_option2(aa_orig.clone(), bb)
    torch.cuda.synchronize()
    time_opt2 = (time.perf_counter() - start) / n_iter * 1000

    print(f"  Option 1 (i-seq, j-par): {time_opt1:.3f} ms")
    print(f"  Option 2 (j-seq, i-par): {time_opt2:.3f} ms")
    print(f"  Difference: {abs(time_opt1 - time_opt2) / min(time_opt1, time_opt2) * 100:.1f}%")


if __name__ == "__main__":
    test_correctness()
    test_performance()
