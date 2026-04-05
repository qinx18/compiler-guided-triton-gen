#!/usr/bin/env python3
"""
Compare different PyTorch baseline implementations for s176
Pattern: a[i] += b[i+m-j-1] * c[j] for i in [0,m), j in [0,m)
"""
import torch
import time


def s176_pytorch_slow(a, b, c):
    """Original slow version with nested Python loops"""
    LEN_1D = a.shape[0]
    m = LEN_1D // 2

    for j in range(m):
        for i in range(m):
            a[i] += b[i + m - j - 1] * c[j]


def s176_pytorch_vectorized_v1(a, b, c):
    """Vectorized version using slicing"""
    LEN_1D = a.shape[0]
    m = LEN_1D // 2

    # Vectorize inner loop: for each j, update all i at once
    for j in range(m):
        # a[0:m] += b[m-j-1:2*m-j-1] * c[j]
        a[:m] += b[m-j-1:2*m-j-1] * c[j]


def s176_pytorch_vectorized_v2(a, b, c):
    """Fully vectorized using broadcasting and matrix ops"""
    LEN_1D = a.shape[0]
    m = LEN_1D // 2

    # Build index matrix: for each (i, j), we need b[i+m-j-1]
    # i ranges [0, m), j ranges [0, m)
    # b_indices[i, j] = i + m - j - 1

    i_idx = torch.arange(m, device=a.device).unsqueeze(1)  # Shape: (m, 1)
    j_idx = torch.arange(m, device=a.device).unsqueeze(0)  # Shape: (1, m)
    b_indices = i_idx + m - j_idx - 1  # Broadcasting: (m, m)

    # Gather b values: b_matrix[i, j] = b[i+m-j-1]
    b_matrix = b[b_indices]  # Shape: (m, m)

    # Matrix-vector multiply: sum over j dimension
    # a[i] += sum_j(b[i+m-j-1] * c[j])
    a[:m] += torch.matmul(b_matrix, c[:m])


def s176_pytorch_conv1d(a, b, c):
    """Use conv1d for maximum efficiency"""
    LEN_1D = a.shape[0]
    m = LEN_1D // 2

    # The pattern is: a[i] += sum_j(b[i+m-j-1] * c[j])
    # This is a correlation: output[i] = sum_j(signal[i+offset-j] * kernel[j])

    # Extract the relevant segment of b: b[m-1:2*m-1] covers indices [m-1, 2m-1)
    # For i in [0,m) and j in [0,m), i+m-j-1 ranges from [m-1, 2m-1)
    b_segment = b[0:2*m-1].unsqueeze(0).unsqueeze(0)  # (1, 1, 2m-1)
    c_kernel = c[:m].flip(0).unsqueeze(0).unsqueeze(0)  # (1, 1, m) - flip for conv

    # Perform convolution
    # Output will have size 2m-1-m+1 = m
    result = torch.nn.functional.conv1d(b_segment, c_kernel, padding=0)
    a[:m] += result[0, 0, :]


def benchmark_implementations():
    """Benchmark all implementations"""
    test_sizes = [100, 1000, 10000]

    implementations = [
        ("Slow (nested loops)", s176_pytorch_slow),
        ("Vectorized v1 (slice)", s176_pytorch_vectorized_v1),
        ("Vectorized v2 (matmul)", s176_pytorch_vectorized_v2),
        ("Conv1d", s176_pytorch_conv1d),
    ]

    print("=" * 80)
    print("s176 PyTorch Baseline Performance Comparison")
    print("=" * 80)

    for N in test_sizes:
        print(f"\nTesting with N={N} (m={N//2}):")
        print("-" * 80)

        # Create reference with slow version on small data
        a_ref = torch.randn(N + 10, device='cuda', dtype=torch.float32)
        b_ref = torch.randn(N + 10, device='cuda', dtype=torch.float32)
        c_ref = torch.randn(N + 10, device='cuda', dtype=torch.float32)

        if N <= 1000:
            a_slow = a_ref.clone()
            s176_pytorch_slow(a_slow, b_ref, c_ref)
            reference = a_slow
        else:
            # Use vectorized v1 as reference for large N
            a_ref_temp = a_ref.clone()
            s176_pytorch_vectorized_v1(a_ref_temp, b_ref, c_ref)
            reference = a_ref_temp

        for name, func in implementations:
            # Skip slow version for large N
            if "Slow" in name and N > 1000:
                print(f"  {name:25s} SKIPPED (too slow)")
                continue

            a = a_ref.clone()
            b = b_ref.clone()
            c = c_ref.clone()

            # Warmup
            if N <= 1000 or "Slow" not in name:
                func(a, b, c)

            # Benchmark
            a = a_ref.clone()
            torch.cuda.synchronize()
            start = time.time()
            func(a, b, c)
            torch.cuda.synchronize()
            elapsed = time.time() - start

            # Check correctness
            max_error = torch.max(torch.abs(a - reference)).item()
            status = "✓" if max_error < 1e-3 else f"✗ (err={max_error:.2e})"

            print(f"  {name:25s} {elapsed*1000:8.2f} ms    {status}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    benchmark_implementations()
