import triton
import triton.language as tl
import torch

@triton.jit
def covariance_kernel(
    cov_ptr, data_ptr, mean_ptr,
    float_n,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    # Step 1: Compute mean for each column
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j_start in range(0, M, BLOCK_SIZE):
        j_indices = j_start + j_offsets
        j_mask = j_indices < M
        
        # Initialize mean values to 0
        mean_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        # Sum over all rows for each column
        for i in range(N):
            data_indices = i * M + j_indices
            data_vals = tl.load(data_ptr + data_indices, mask=j_mask, other=0.0)
            mean_vals += data_vals
        
        # Divide by N to get mean
        mean_vals = mean_vals / float_n
        
        # Store mean values
        tl.store(mean_ptr + j_indices, mean_vals, mask=j_mask)
    
    # Step 2: Subtract mean from data
    for i in range(N):
        for j_start in range(0, M, BLOCK_SIZE):
            j_indices = j_start + j_offsets
            j_mask = j_indices < M
            
            data_indices = i * M + j_indices
            mean_vals = tl.load(mean_ptr + j_indices, mask=j_mask, other=0.0)
            data_vals = tl.load(data_ptr + data_indices, mask=j_mask, other=0.0)
            
            # Subtract mean
            data_vals = data_vals - mean_vals
            tl.store(data_ptr + data_indices, data_vals, mask=j_mask)
    
    # Step 3: Compute covariance matrix (upper triangular)
    for i in range(M):
        for j_start in range(i, M, BLOCK_SIZE):
            j_indices = j_start + j_offsets
            j_mask = (j_indices < M) & (j_indices >= i)
            
            # Initialize covariance values to 0
            cov_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            
            # Sum products over all samples
            for k in range(N):
                data_i_idx = k * M + i
                data_j_indices = k * M + j_indices
                
                data_i_val = tl.load(data_ptr + data_i_idx)
                data_j_vals = tl.load(data_ptr + data_j_indices, mask=j_mask, other=0.0)
                
                cov_vals += data_i_val * data_j_vals
            
            # Divide by (N-1)
            cov_vals = cov_vals / (float_n - 1.0)
            
            # Store upper triangular values
            cov_indices_upper = i * M + j_indices
            tl.store(cov_ptr + cov_indices_upper, cov_vals, mask=j_mask)
            
            # Store lower triangular values (symmetric)
            for offset in range(BLOCK_SIZE):
                j_idx = j_start + offset
                if (j_idx < M) & (j_idx >= i) & (j_idx != i):
                    cov_idx_lower = j_idx * M + i
                    val = tl.load(cov_ptr + (i * M + j_idx))
                    tl.store(cov_ptr + cov_idx_lower, val)

def covariance_triton(cov, data, mean, float_n, M, N):
    BLOCK_SIZE = 32
    
    # Launch kernel with single thread block since we handle all iterations inside
    grid = (1,)
    
    covariance_kernel[grid](
        cov, data, mean,
        float_n,
        M, N,
        BLOCK_SIZE=BLOCK_SIZE,
    )