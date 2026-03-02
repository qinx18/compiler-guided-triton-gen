import triton
import triton.language as tl
import torch

@triton.jit
def correlation_kernel(
    corr_ptr, data_ptr, mean_ptr, stddev_ptr,
    eps, float_n, M, N,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate mean for each column
    j_offsets = tl.arange(0, BLOCK_SIZE)
    for j_start in range(0, M, BLOCK_SIZE):
        j_idx = j_start + j_offsets
        j_mask = j_idx < M
        
        # Initialize mean to 0
        mean_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        # Sum over all rows for each column
        for i in range(N):
            data_offsets = i * M + j_idx
            data_vals = tl.load(data_ptr + data_offsets, mask=j_mask, other=0.0)
            mean_vals += data_vals
        
        # Divide by N to get mean
        mean_vals = mean_vals / float_n
        tl.store(mean_ptr + j_idx, mean_vals, mask=j_mask)
    
    # Calculate stddev for each column
    for j_start in range(0, M, BLOCK_SIZE):
        j_idx = j_start + j_offsets
        j_mask = j_idx < M
        
        # Load mean values
        mean_vals = tl.load(mean_ptr + j_idx, mask=j_mask, other=0.0)
        
        # Initialize stddev to 0
        stddev_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        # Sum squared differences
        for i in range(N):
            data_offsets = i * M + j_idx
            data_vals = tl.load(data_ptr + data_offsets, mask=j_mask, other=0.0)
            diff = data_vals - mean_vals
            stddev_vals += diff * diff
        
        # Calculate standard deviation
        stddev_vals = stddev_vals / float_n
        stddev_vals = tl.sqrt(stddev_vals)
        
        # Handle near-zero values
        stddev_vals = tl.where(stddev_vals <= eps, 1.0, stddev_vals)
        tl.store(stddev_ptr + j_idx, stddev_vals, mask=j_mask)
    
    # Center and reduce the column vectors
    sqrt_float_n = tl.sqrt(float_n)
    for i in range(N):
        for j_start in range(0, M, BLOCK_SIZE):
            j_idx = j_start + j_offsets
            j_mask = j_idx < M
            
            data_offsets = i * M + j_idx
            mean_vals = tl.load(mean_ptr + j_idx, mask=j_mask, other=0.0)
            stddev_vals = tl.load(stddev_ptr + j_idx, mask=j_mask, other=1.0)
            data_vals = tl.load(data_ptr + data_offsets, mask=j_mask, other=0.0)
            
            # Center and scale
            data_vals = data_vals - mean_vals
            data_vals = data_vals / (sqrt_float_n * stddev_vals)
            
            tl.store(data_ptr + data_offsets, data_vals, mask=j_mask)
    
    # Calculate correlation matrix
    # Set diagonal elements to 1.0 first
    for i in range(M):
        diag_offset = i * M + i
        tl.store(corr_ptr + diag_offset, 1.0)
    
    # Calculate upper triangular part
    for i in range(M - 1):
        for j in range(i + 1, M):
            corr_val = 0.0
            for k in range(N):
                data_i_offset = k * M + i
                data_j_offset = k * M + j
                data_i_val = tl.load(data_ptr + data_i_offset)
                data_j_val = tl.load(data_ptr + data_j_offset)
                corr_val += data_i_val * data_j_val
            
            # Store symmetric values
            corr_ij_offset = i * M + j
            corr_ji_offset = j * M + i
            tl.store(corr_ptr + corr_ij_offset, corr_val)
            tl.store(corr_ptr + corr_ji_offset, corr_val)

def correlation_triton(corr, data, mean, stddev, eps, float_n, M, N):
    BLOCK_SIZE = 32
    
    # Launch kernel with single thread block
    grid = (1,)
    correlation_kernel[grid](
        corr, data, mean, stddev,
        eps, float_n, M, N,
        BLOCK_SIZE=BLOCK_SIZE
    )