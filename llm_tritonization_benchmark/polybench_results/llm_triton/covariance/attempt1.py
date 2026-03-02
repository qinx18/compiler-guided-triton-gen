import torch
import triton
import triton.language as tl

@triton.jit
def covariance_kernel(cov_ptr, data_ptr, mean_ptr, float_n, M, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Step 1: Compute means
    if pid == 0:
        j_offsets = tl.arange(0, BLOCK_SIZE)
        for j_start in range(0, M, BLOCK_SIZE):
            j_indices = j_start + j_offsets
            j_mask = j_indices < M
            
            # Initialize mean to 0
            mean_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
            
            # Sum over all rows
            for i in range(N):
                data_indices = i * M + j_indices
                data_vals = tl.load(data_ptr + data_indices, mask=j_mask, other=0.0)
                mean_vals += data_vals
            
            # Divide by N
            mean_vals = mean_vals / float_n
            
            # Store means
            tl.store(mean_ptr + j_indices, mean_vals, mask=j_mask)
    
    # Synchronize
    tl.debug_barrier()
    
    # Step 2: Subtract means from data
    if pid == 0:
        for i in range(N):
            j_offsets = tl.arange(0, BLOCK_SIZE)
            for j_start in range(0, M, BLOCK_SIZE):
                j_indices = j_start + j_offsets
                j_mask = j_indices < M
                
                data_indices = i * M + j_indices
                data_vals = tl.load(data_ptr + data_indices, mask=j_mask, other=0.0)
                mean_vals = tl.load(mean_ptr + j_indices, mask=j_mask, other=0.0)
                
                data_vals = data_vals - mean_vals
                tl.store(data_ptr + data_indices, data_vals, mask=j_mask)
    
    # Synchronize
    tl.debug_barrier()
    
    # Step 3: Compute covariance matrix
    if pid == 0:
        i_offsets = tl.arange(0, BLOCK_SIZE)
        for i_start in range(0, M, BLOCK_SIZE):
            i_indices = i_start + i_offsets
            i_mask = i_indices < M
            
            for i_idx in range(BLOCK_SIZE):
                if i_start + i_idx >= M:
                    break
                    
                i_val = i_start + i_idx
                
                j_offsets = tl.arange(0, BLOCK_SIZE)
                for j_start in range(i_val, M, BLOCK_SIZE):
                    j_indices = j_start + j_offsets
                    j_mask = (j_indices < M) & (j_indices >= i_val)
                    
                    # Initialize covariance to 0
                    cov_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
                    
                    # Sum over all rows
                    for k in range(N):
                        data_i_vals = tl.load(data_ptr + k * M + i_val)
                        data_j_indices = k * M + j_indices
                        data_j_vals = tl.load(data_ptr + data_j_indices, mask=j_mask, other=0.0)
                        cov_vals += data_i_vals * data_j_vals
                    
                    # Divide by (N - 1)
                    cov_vals = cov_vals / (float_n - 1.0)
                    
                    # Store upper triangular
                    cov_indices = i_val * M + j_indices
                    tl.store(cov_ptr + cov_indices, cov_vals, mask=j_mask)
                    
                    # Store lower triangular (symmetric)
                    for j_idx in range(BLOCK_SIZE):
                        if j_start + j_idx >= M or j_start + j_idx < i_val:
                            continue
                        j_val = j_start + j_idx
                        if j_val != i_val:
                            cov_val = tl.load(cov_ptr + i_val * M + j_val)
                            tl.store(cov_ptr + j_val * M + i_val, cov_val)

def covariance_triton(cov, data, mean, float_n, M, N):
    BLOCK_SIZE = 32
    
    grid = (1,)
    
    covariance_kernel[grid](
        cov, data, mean, float_n, M, N,
        BLOCK_SIZE=BLOCK_SIZE
    )