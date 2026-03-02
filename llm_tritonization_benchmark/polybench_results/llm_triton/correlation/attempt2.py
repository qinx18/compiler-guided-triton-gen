import triton
import triton.language as tl
import torch

@triton.jit
def mean_kernel(
    data_ptr, mean_ptr, M, N, float_n,
    BLOCK_SIZE: tl.constexpr
):
    j = tl.program_id(0)
    if j >= M:
        return
    
    sum_val = 0.0
    for i in range(N):
        data_offset = i * M + j
        data_val = tl.load(data_ptr + data_offset)
        sum_val += data_val
    
    mean_val = sum_val / float_n
    tl.store(mean_ptr + j, mean_val)

@triton.jit
def stddev_kernel(
    data_ptr, mean_ptr, stddev_ptr, M, N, float_n, eps,
    BLOCK_SIZE: tl.constexpr
):
    j = tl.program_id(0)
    if j >= M:
        return
    
    mean_val = tl.load(mean_ptr + j)
    sum_sq_diff = 0.0
    
    for i in range(N):
        data_offset = i * M + j
        data_val = tl.load(data_ptr + data_offset)
        diff = data_val - mean_val
        sum_sq_diff += diff * diff
    
    stddev_val = sum_sq_diff / float_n
    stddev_val = tl.sqrt(stddev_val)
    stddev_val = tl.where(stddev_val <= eps, 1.0, stddev_val)
    
    tl.store(stddev_ptr + j, stddev_val)

@triton.jit
def normalize_kernel(
    data_ptr, mean_ptr, stddev_ptr, M, N, sqrt_float_n,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    col_start = tl.program_id(1) * BLOCK_SIZE
    
    if row >= N:
        return
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    cols = col_start + col_offsets
    col_mask = cols < M
    
    data_offsets = row * M + cols
    mean_vals = tl.load(mean_ptr + cols, mask=col_mask)
    stddev_vals = tl.load(stddev_ptr + cols, mask=col_mask)
    data_vals = tl.load(data_ptr + data_offsets, mask=col_mask)
    
    data_vals = data_vals - mean_vals
    data_vals = data_vals / (sqrt_float_n * stddev_vals)
    
    tl.store(data_ptr + data_offsets, data_vals, mask=col_mask)

@triton.jit
def corr_kernel(
    data_ptr, corr_ptr, M, N,
    BLOCK_SIZE: tl.constexpr
):
    i = tl.program_id(0)
    j = tl.program_id(1)
    
    if i >= M or j >= M:
        return
    
    if i == j:
        tl.store(corr_ptr + i * M + j, 1.0)
        return
    
    if i > j:
        return
    
    corr_val = 0.0
    for k in range(N):
        data_i_offset = k * M + i
        data_j_offset = k * M + j
        data_i_val = tl.load(data_ptr + data_i_offset)
        data_j_val = tl.load(data_ptr + data_j_offset)
        corr_val += data_i_val * data_j_val
    
    tl.store(corr_ptr + i * M + j, corr_val)
    tl.store(corr_ptr + j * M + i, corr_val)

def correlation_triton(corr, data, mean, stddev, eps, float_n, M, N):
    BLOCK_SIZE = 32
    
    # Calculate mean
    grid = (M,)
    mean_kernel[grid](
        data, mean, M, N, float_n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Calculate stddev
    grid = (M,)
    stddev_kernel[grid](
        data, mean, stddev, M, N, float_n, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Normalize data
    sqrt_float_n = torch.sqrt(torch.tensor(float_n)).item()
    grid_x = N
    grid_y = triton.cdiv(M, BLOCK_SIZE)
    grid = (grid_x, grid_y)
    normalize_kernel[grid](
        data, mean, stddev, M, N, sqrt_float_n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Calculate correlation matrix
    grid = (M, M)
    corr_kernel[grid](
        data, corr, M, N,
        BLOCK_SIZE=BLOCK_SIZE
    )