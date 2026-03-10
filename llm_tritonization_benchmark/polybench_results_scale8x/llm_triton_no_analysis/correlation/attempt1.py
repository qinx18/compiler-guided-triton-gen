import triton
import triton.language as tl
import torch

@triton.jit
def correlation_kernel(
    data_ptr, mean_ptr, stddev_ptr, corr_ptr,
    eps, float_n, M: tl.constexpr, N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate mean for each column
    for j in range(M):
        sum_val = 0.0
        offsets = tl.arange(0, BLOCK_SIZE)
        for block_start in range(0, N, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < N
            data_offsets = current_offsets * M + j
            vals = tl.load(data_ptr + data_offsets, mask=mask, other=0.0)
            sum_val += tl.sum(vals)
        
        mean_val = sum_val / float_n
        tl.store(mean_ptr + j, mean_val)
    
    # Calculate stddev for each column
    for j in range(M):
        mean_val = tl.load(mean_ptr + j)
        sum_sq = 0.0
        offsets = tl.arange(0, BLOCK_SIZE)
        for block_start in range(0, N, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < N
            data_offsets = current_offsets * M + j
            vals = tl.load(data_ptr + data_offsets, mask=mask, other=0.0)
            diff = vals - mean_val
            sum_sq += tl.sum(diff * diff, axis=0)
        
        stddev_val = sum_sq / float_n
        stddev_val = tl.sqrt(stddev_val)
        stddev_val = tl.where(stddev_val <= eps, 1.0, stddev_val)
        tl.store(stddev_ptr + j, stddev_val)
    
    # Center and reduce the column vectors
    sqrt_float_n = tl.sqrt(float_n)
    for i in range(N):
        for j in range(M):
            mean_val = tl.load(mean_ptr + j)
            stddev_val = tl.load(stddev_ptr + j)
            data_idx = i * M + j
            val = tl.load(data_ptr + data_idx)
            val = val - mean_val
            val = val / (sqrt_float_n * stddev_val)
            tl.store(data_ptr + data_idx, val)
    
    # Calculate correlation matrix
    for i in range(M - 1):
        # Set diagonal element
        diag_idx = i * M + i
        tl.store(corr_ptr + diag_idx, 1.0)
        
        for j in range(i + 1, M):
            sum_corr = 0.0
            offsets = tl.arange(0, BLOCK_SIZE)
            for block_start in range(0, N, BLOCK_SIZE):
                current_offsets = block_start + offsets
                mask = current_offsets < N
                data_i_offsets = current_offsets * M + i
                data_j_offsets = current_offsets * M + j
                vals_i = tl.load(data_ptr + data_i_offsets, mask=mask, other=0.0)
                vals_j = tl.load(data_ptr + data_j_offsets, mask=mask, other=0.0)
                sum_corr += tl.sum(vals_i * vals_j)
            
            corr_ij_idx = i * M + j
            corr_ji_idx = j * M + i
            tl.store(corr_ptr + corr_ij_idx, sum_corr)
            tl.store(corr_ptr + corr_ji_idx, sum_corr)
    
    # Set last diagonal element
    last_diag_idx = (M - 1) * M + (M - 1)
    tl.store(corr_ptr + last_diag_idx, 1.0)

def correlation_triton(corr, data, mean, stddev, eps, float_n, M, N):
    BLOCK_SIZE = 128
    
    correlation_kernel[(1,)](
        data, mean, stddev, corr,
        eps, float_n, M, N, BLOCK_SIZE
    )