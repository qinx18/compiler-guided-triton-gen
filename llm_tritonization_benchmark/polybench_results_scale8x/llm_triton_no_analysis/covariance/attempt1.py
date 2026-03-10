import triton
import triton.language as tl
import torch

@triton.jit
def covariance_kernel(
    data_ptr, mean_ptr, cov_ptr, float_n,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Step 1: Compute mean for each column
    for j in range(M):
        sum_val = 0.0
        for block_start in range(0, N, BLOCK_SIZE):
            offsets = tl.arange(0, BLOCK_SIZE)
            current_offsets = block_start + offsets
            mask = current_offsets < N
            
            # Load data[i][j] for current block
            data_offsets = current_offsets * M + j
            vals = tl.load(data_ptr + data_offsets, mask=mask, other=0.0)
            sum_val += tl.sum(vals)
        
        # Store mean[j]
        mean_val = sum_val / float_n
        tl.store(mean_ptr + j, mean_val)
    
    # Step 2: Subtract mean from data
    for i in range(N):
        for block_start in range(0, M, BLOCK_SIZE):
            offsets = tl.arange(0, BLOCK_SIZE)
            current_offsets = block_start + offsets
            mask = current_offsets < M
            
            # Load data[i][j] and mean[j]
            data_offsets = i * M + current_offsets
            data_vals = tl.load(data_ptr + data_offsets, mask=mask)
            mean_vals = tl.load(mean_ptr + current_offsets, mask=mask)
            
            # Subtract mean and store back
            new_vals = data_vals - mean_vals
            tl.store(data_ptr + data_offsets, new_vals, mask=mask)
    
    # Step 3: Compute covariance matrix
    for i in range(M):
        for j in range(i, M):
            sum_val = 0.0
            for block_start in range(0, N, BLOCK_SIZE):
                offsets = tl.arange(0, BLOCK_SIZE)
                current_offsets = block_start + offsets
                mask = current_offsets < N
                
                # Load data[k][i] and data[k][j]
                data_i_offsets = current_offsets * M + i
                data_j_offsets = current_offsets * M + j
                data_i_vals = tl.load(data_ptr + data_i_offsets, mask=mask, other=0.0)
                data_j_vals = tl.load(data_ptr + data_j_offsets, mask=mask, other=0.0)
                
                # Accumulate product
                products = data_i_vals * data_j_vals
                sum_val += tl.sum(products)
            
            # Compute final covariance value
            cov_val = sum_val / (float_n - 1.0)
            
            # Store cov[i][j] and cov[j][i]
            cov_ij_offset = i * M + j
            cov_ji_offset = j * M + i
            tl.store(cov_ptr + cov_ij_offset, cov_val)
            tl.store(cov_ptr + cov_ji_offset, cov_val)

def covariance_triton(cov, data, mean, float_n, M, N):
    BLOCK_SIZE = 32
    
    # Launch kernel with single thread block
    grid = (1,)
    covariance_kernel[grid](
        data, mean, cov, float_n,
        M=M, N=N, BLOCK_SIZE=BLOCK_SIZE
    )