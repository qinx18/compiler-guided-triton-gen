import triton
import triton.language as tl
import torch

@triton.jit
def compute_mean_kernel(data_ptr, mean_ptr, M: tl.constexpr, N: tl.constexpr, float_n):
    j = tl.program_id(0)
    if j >= M:
        return
    
    sum_val = 0.0
    for i in range(N):
        data_idx = i * M + j
        data_val = tl.load(data_ptr + data_idx)
        sum_val += data_val
    
    mean_val = sum_val / float_n
    tl.store(mean_ptr + j, mean_val)

@triton.jit
def compute_stddev_kernel(data_ptr, mean_ptr, stddev_ptr, M: tl.constexpr, N: tl.constexpr, float_n, eps):
    j = tl.program_id(0)
    if j >= M:
        return
    
    mean_val = tl.load(mean_ptr + j)
    sum_sq = 0.0
    
    for i in range(N):
        data_idx = i * M + j
        data_val = tl.load(data_ptr + data_idx)
        diff = data_val - mean_val
        sum_sq += diff * diff
    
    stddev_val = sum_sq / float_n
    stddev_val = tl.sqrt(stddev_val)
    stddev_val = tl.where(stddev_val <= eps, 1.0, stddev_val)
    
    tl.store(stddev_ptr + j, stddev_val)

@triton.jit
def normalize_data_kernel(data_ptr, mean_ptr, stddev_ptr, M: tl.constexpr, N: tl.constexpr, float_n):
    idx = tl.program_id(0)
    if idx >= N * M:
        return
    
    i = idx // M
    j = idx % M
    
    data_idx = i * M + j
    mean_val = tl.load(mean_ptr + j)
    stddev_val = tl.load(stddev_ptr + j)
    data_val = tl.load(data_ptr + data_idx)
    
    normalized = (data_val - mean_val) / (tl.sqrt(float_n) * stddev_val)
    tl.store(data_ptr + data_idx, normalized)

@triton.jit
def compute_correlation_kernel(data_ptr, corr_ptr, M: tl.constexpr, N: tl.constexpr):
    idx = tl.program_id(0)
    
    # Handle diagonal elements
    if idx < M:
        diag_idx = idx * M + idx
        tl.store(corr_ptr + diag_idx, 1.0)
        return
    
    # Handle upper triangular elements
    upper_idx = idx - M
    total_upper = (M * (M - 1)) // 2
    
    if upper_idx >= total_upper:
        return
    
    # Convert linear index to (i, j) coordinates for upper triangle
    i = 0
    remaining = upper_idx
    while remaining >= (M - 1 - i):
        remaining -= (M - 1 - i)
        i += 1
    j = i + 1 + remaining
    
    if i >= M - 1 or j >= M:
        return
    
    corr_val = 0.0
    for k in range(N):
        data_i = tl.load(data_ptr + k * M + i)
        data_j = tl.load(data_ptr + k * M + j)
        corr_val += data_i * data_j
    
    # Store both (i,j) and (j,i)
    corr_idx_ij = i * M + j
    corr_idx_ji = j * M + i
    tl.store(corr_ptr + corr_idx_ij, corr_val)
    tl.store(corr_ptr + corr_idx_ji, corr_val)

def correlation_triton(corr, data, mean, stddev, eps, float_n, M, N):
    # Compute mean
    grid = (M,)
    compute_mean_kernel[grid](data, mean, M, N, float_n)
    
    # Compute standard deviation
    compute_stddev_kernel[grid](data, mean, stddev, M, N, float_n, eps)
    
    # Normalize data
    grid = (N * M,)
    normalize_data_kernel[grid](data, mean, stddev, M, N, float_n)
    
    # Compute correlation matrix
    total_upper = (M * (M - 1)) // 2
    grid = (M + total_upper,)
    compute_correlation_kernel[grid](data, corr, M, N)