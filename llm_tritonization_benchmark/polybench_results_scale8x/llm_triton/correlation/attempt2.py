import triton
import triton.language as tl
import torch

@triton.jit
def correlation_mean_kernel(data_ptr, mean_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_N: tl.constexpr):
    j = tl.program_id(0)
    if j >= M:
        return
    
    sum_val = 0.0
    n_offsets = tl.arange(0, BLOCK_N)
    
    for block_start in range(0, N, BLOCK_N):
        current_offsets = block_start + n_offsets
        mask = current_offsets < N
        data_offsets = current_offsets * M + j
        vals = tl.load(data_ptr + data_offsets, mask=mask, other=0.0)
        sum_val += tl.sum(vals)
    
    mean_val = sum_val / N
    tl.store(mean_ptr + j, mean_val)

@triton.jit
def correlation_stddev_kernel(data_ptr, mean_ptr, stddev_ptr, eps, M: tl.constexpr, N: tl.constexpr, BLOCK_N: tl.constexpr):
    j = tl.program_id(0)
    if j >= M:
        return
    
    mean_val = tl.load(mean_ptr + j)
    sum_sq = 0.0
    n_offsets = tl.arange(0, BLOCK_N)
    
    for block_start in range(0, N, BLOCK_N):
        current_offsets = block_start + n_offsets
        mask = current_offsets < N
        data_offsets = current_offsets * M + j
        vals = tl.load(data_ptr + data_offsets, mask=mask, other=0.0)
        diff = vals - mean_val
        sum_sq += tl.sum(diff * diff)
    
    stddev_val = tl.sqrt(sum_sq / N)
    stddev_val = tl.where(stddev_val <= eps, 1.0, stddev_val)
    tl.store(stddev_ptr + j, stddev_val)

@triton.jit
def correlation_normalize_kernel(data_ptr, mean_ptr, stddev_ptr, sqrt_n, M: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr):
    i = tl.program_id(0)
    if i >= N:
        return
    
    m_offsets = tl.arange(0, BLOCK_M)
    
    for block_start in range(0, M, BLOCK_M):
        current_offsets = block_start + m_offsets
        mask = current_offsets < M
        
        data_offsets = i * M + current_offsets
        mean_vals = tl.load(mean_ptr + current_offsets, mask=mask, other=0.0)
        stddev_vals = tl.load(stddev_ptr + current_offsets, mask=mask, other=1.0)
        data_vals = tl.load(data_ptr + data_offsets, mask=mask, other=0.0)
        
        normalized_vals = (data_vals - mean_vals) / (sqrt_n * stddev_vals)
        tl.store(data_ptr + data_offsets, normalized_vals, mask=mask)

@triton.jit
def correlation_matrix_kernel(data_ptr, corr_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_N: tl.constexpr):
    i = tl.program_id(0)
    j = tl.program_id(1)
    
    if i >= M:
        return
    if j >= M:
        return
    
    if i == j:
        tl.store(corr_ptr + i * M + j, 1.0)
        return
    
    if i > j:
        return
    
    sum_val = 0.0
    n_offsets = tl.arange(0, BLOCK_N)
    
    for block_start in range(0, N, BLOCK_N):
        current_offsets = block_start + n_offsets
        mask = current_offsets < N
        
        data_i_offsets = current_offsets * M + i
        data_j_offsets = current_offsets * M + j
        
        vals_i = tl.load(data_ptr + data_i_offsets, mask=mask, other=0.0)
        vals_j = tl.load(data_ptr + data_j_offsets, mask=mask, other=0.0)
        
        sum_val += tl.sum(vals_i * vals_j)
    
    tl.store(corr_ptr + i * M + j, sum_val)
    tl.store(corr_ptr + j * M + i, sum_val)

def correlation_triton(corr, data, mean, stddev, eps, float_n, M, N):
    BLOCK_N = 128
    BLOCK_M = 128
    
    sqrt_n = torch.sqrt(torch.tensor(float_n)).item()
    
    # Phase 1: Compute means
    grid_mean = (M,)
    correlation_mean_kernel[grid_mean](
        data, mean, M, N, BLOCK_N
    )
    
    # Phase 2: Compute standard deviations
    grid_stddev = (M,)
    correlation_stddev_kernel[grid_stddev](
        data, mean, stddev, eps, M, N, BLOCK_N
    )
    
    # Phase 3: Normalize data
    grid_normalize = (N,)
    correlation_normalize_kernel[grid_normalize](
        data, mean, stddev, sqrt_n, M, N, BLOCK_M
    )
    
    # Phase 4: Compute correlation matrix
    grid_corr = (M, M)
    correlation_matrix_kernel[grid_corr](
        data, corr, M, N, BLOCK_N
    )