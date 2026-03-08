import triton
import triton.language as tl
import torch

@triton.jit
def correlation_kernel(data_ptr, mean_ptr, stddev_ptr, corr_ptr, 
                      eps, float_n, M: tl.constexpr, N: tl.constexpr,
                      BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate which column this program handles
    j = pid
    
    if j >= M:
        return
    
    # Step 1: Calculate mean for column j
    mean_val = 0.0
    for i in range(N):
        data_idx = i * M + j
        data_val = tl.load(data_ptr + data_idx)
        mean_val += data_val
    mean_val /= float_n
    tl.store(mean_ptr + j, mean_val)
    
    # Step 2: Calculate stddev for column j
    stddev_val = 0.0
    for i in range(N):
        data_idx = i * M + j
        data_val = tl.load(data_ptr + data_idx)
        diff = data_val - mean_val
        stddev_val += diff * diff
    stddev_val /= float_n
    stddev_val = tl.math.sqrt(stddev_val)
    stddev_val = tl.where(stddev_val <= eps, 1.0, stddev_val)
    tl.store(stddev_ptr + j, stddev_val)

@triton.jit
def normalize_kernel(data_ptr, mean_ptr, stddev_ptr, float_n, 
                    M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate which element this program handles
    total_elements = M * N
    element_idx = pid * BLOCK_SIZE
    
    if element_idx >= total_elements:
        return
    
    sqrt_float_n = tl.math.sqrt(float_n)
    
    offsets = tl.arange(0, BLOCK_SIZE)
    element_indices = element_idx + offsets
    mask = element_indices < total_elements
    
    # Calculate i, j from linear index
    i_vals = element_indices // M
    j_vals = element_indices % M
    
    # Load data values
    data_vals = tl.load(data_ptr + element_indices, mask=mask)
    
    # Load corresponding mean and stddev values
    mean_vals = tl.load(mean_ptr + j_vals, mask=mask)
    stddev_vals = tl.load(stddev_ptr + j_vals, mask=mask)
    
    # Normalize
    normalized_vals = (data_vals - mean_vals) / (sqrt_float_n * stddev_vals)
    
    # Store back
    tl.store(data_ptr + element_indices, normalized_vals, mask=mask)

@triton.jit
def correlation_matrix_kernel(data_ptr, corr_ptr, M: tl.constexpr, N: tl.constexpr):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    
    i = pid_x
    j = pid_y
    
    if i >= M or j >= M:
        return
    
    # Set diagonal elements to 1.0
    if i == j:
        corr_idx = i * M + j
        tl.store(corr_ptr + corr_idx, 1.0)
        return
    
    # Only compute upper triangular part
    if j <= i:
        return
    
    # Calculate correlation between columns i and j
    corr_val = 0.0
    for k in range(N):
        data_i_idx = k * M + i
        data_j_idx = k * M + j
        data_i_val = tl.load(data_ptr + data_i_idx)
        data_j_val = tl.load(data_ptr + data_j_idx)
        corr_val += data_i_val * data_j_val
    
    # Store in both symmetric positions
    corr_ij_idx = i * M + j
    corr_ji_idx = j * M + i
    tl.store(corr_ptr + corr_ij_idx, corr_val)
    tl.store(corr_ptr + corr_ji_idx, corr_val)

def correlation_triton(corr, data, mean, stddev, eps, float_n, M, N):
    # Step 1: Calculate mean and stddev for each column
    grid = (M,)
    BLOCK_SIZE = 64
    correlation_kernel[grid](
        data, mean, stddev, corr,
        eps, float_n, M, N, BLOCK_SIZE
    )
    
    # Step 2: Normalize data
    total_elements = M * N
    grid_size = triton.cdiv(total_elements, BLOCK_SIZE)
    grid = (grid_size,)
    normalize_kernel[grid](
        data, mean, stddev, float_n,
        M, N, BLOCK_SIZE
    )
    
    # Step 3: Calculate correlation matrix
    grid = (M, M)
    correlation_matrix_kernel[grid](
        data, corr, M, N
    )