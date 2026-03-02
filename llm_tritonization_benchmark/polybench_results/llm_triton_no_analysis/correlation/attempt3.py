import triton
import triton.language as tl
import torch

@triton.jit
def correlation_mean_kernel(
    data_ptr, mean_ptr, M, N,
    BLOCK_SIZE: tl.constexpr
):
    j = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    j_mask = j < M
    
    sum_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for i in range(N):
        data_idx = i * M + j
        data_vals = tl.load(data_ptr + data_idx, mask=j_mask, other=0.0)
        sum_vals += data_vals
    
    mean_vals = sum_vals / N
    tl.store(mean_ptr + j, mean_vals, mask=j_mask)

@triton.jit
def correlation_stddev_kernel(
    data_ptr, mean_ptr, stddev_ptr, eps, M, N,
    BLOCK_SIZE: tl.constexpr
):
    j = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    j_mask = j < M
    
    mean_vals = tl.load(mean_ptr + j, mask=j_mask, other=0.0)
    sum_sq_diff = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for i in range(N):
        data_idx = i * M + j
        data_vals = tl.load(data_ptr + data_idx, mask=j_mask, other=0.0)
        diff = data_vals - mean_vals
        sum_sq_diff += diff * diff
    
    stddev_vals = tl.sqrt(sum_sq_diff / N)
    stddev_vals = tl.where(stddev_vals <= eps, 1.0, stddev_vals)
    
    tl.store(stddev_ptr + j, stddev_vals, mask=j_mask)

@triton.jit
def correlation_normalize_kernel(
    data_ptr, mean_ptr, stddev_ptr, sqrt_n, M, N,
    BLOCK_SIZE: tl.constexpr
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    idx_mask = idx < N * M
    
    i = idx // M
    j = idx % M
    
    mean_val = tl.load(mean_ptr + j, mask=(j < M), other=0.0)
    stddev_val = tl.load(stddev_ptr + j, mask=(j < M), other=1.0)
    
    data_val = tl.load(data_ptr + idx, mask=idx_mask, other=0.0)
    normalized = (data_val - mean_val) / (sqrt_n * stddev_val)
    
    tl.store(data_ptr + idx, normalized, mask=idx_mask)

@triton.jit
def correlation_matrix_kernel(
    corr_ptr, data_ptr, M, N,
    BLOCK_SIZE: tl.constexpr
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_pairs = M * M
    idx_mask = idx < total_pairs
    
    i = idx // M
    j = idx % M
    
    # Set diagonal elements to 1.0
    corr_val = tl.where(i == j, 1.0, 0.0)
    
    # Compute correlation for off-diagonal elements
    is_off_diagonal = i != j
    
    for k in range(N):
        data_ki = tl.load(data_ptr + k * M + i, mask=(idx_mask & is_off_diagonal & (i < M)), other=0.0)
        data_kj = tl.load(data_ptr + k * M + j, mask=(idx_mask & is_off_diagonal & (j < M)), other=0.0)
        product = data_ki * data_kj
        corr_val = tl.where(is_off_diagonal, corr_val + product, corr_val)
    
    tl.store(corr_ptr + idx, corr_val, mask=idx_mask)

def correlation_triton(corr, data, mean, stddev, eps, float_n, M, N):
    BLOCK_SIZE = 64
    
    # Step 1: Compute means
    grid_mean = (triton.cdiv(M, BLOCK_SIZE),)
    correlation_mean_kernel[grid_mean](
        data, mean, M, N, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Step 2: Compute standard deviations
    grid_stddev = (triton.cdiv(M, BLOCK_SIZE),)
    correlation_stddev_kernel[grid_stddev](
        data, mean, stddev, eps, M, N, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Step 3: Normalize data
    sqrt_n = (float_n ** 0.5)
    grid_norm = (triton.cdiv(N * M, BLOCK_SIZE),)
    correlation_normalize_kernel[grid_norm](
        data, mean, stddev, sqrt_n, M, N, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Step 4: Compute correlation matrix
    grid_corr = (triton.cdiv(M * M, BLOCK_SIZE),)
    correlation_matrix_kernel[grid_corr](
        corr, data, M, N, BLOCK_SIZE=BLOCK_SIZE
    )