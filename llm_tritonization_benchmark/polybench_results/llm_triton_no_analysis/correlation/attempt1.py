import triton
import triton.language as tl
import torch

@triton.jit
def correlation_kernel(
    corr_ptr, data_ptr, mean_ptr, stddev_ptr,
    eps, float_n, M, N,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Step 1: Compute mean for each column j
    m_offsets = tl.arange(0, BLOCK_SIZE_M)
    n_offsets = tl.arange(0, BLOCK_SIZE_N)
    
    for j_start in range(0, M, BLOCK_SIZE_M):
        j_offsets = j_start + m_offsets
        j_mask = j_offsets < M
        
        # Initialize mean values to 0
        mean_vals = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
        
        # Sum over all N elements for each j
        for i_start in range(0, N, BLOCK_SIZE_N):
            i_offsets = i_start + n_offsets
            i_mask = i_offsets < N
            
            # Load data[i][j] values
            for j_idx in range(BLOCK_SIZE_M):
                if j_start + j_idx < M:
                    j_val = j_start + j_idx
                    data_vals = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
                    for i_idx in range(BLOCK_SIZE_N):
                        if i_start + i_idx < N:
                            i_val = i_start + i_idx
                            data_idx = i_val * M + j_val
                            data_vals = tl.where(i_idx == tl.arange(0, BLOCK_SIZE_N), 
                                               tl.load(data_ptr + data_idx), data_vals)
                    
                    # Sum the valid elements
                    valid_mask = (i_start + n_offsets) < N
                    sum_val = tl.sum(tl.where(valid_mask, data_vals, 0.0))
                    mean_vals = tl.where(j_idx == tl.arange(0, BLOCK_SIZE_M), 
                                       tl.where(j_idx == 0, sum_val, 
                                               tl.where(j_idx == 1 and BLOCK_SIZE_M > 1, sum_val, 
                                                       mean_vals)), mean_vals)
        
        # Divide by float_n and store
        mean_vals = mean_vals / float_n
        tl.store(mean_ptr + j_offsets, mean_vals, mask=j_mask)
    
    # Step 2: Compute stddev for each column j
    for j_start in range(0, M, BLOCK_SIZE_M):
        j_offsets = j_start + m_offsets
        j_mask = j_offsets < M
        
        # Load mean values
        mean_vals = tl.load(mean_ptr + j_offsets, mask=j_mask)
        
        # Initialize stddev values to 0
        stddev_vals = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
        
        # Sum squared differences
        for i_start in range(0, N, BLOCK_SIZE_N):
            for j_idx in range(BLOCK_SIZE_M):
                if j_start + j_idx < M:
                    j_val = j_start + j_idx
                    mean_j = tl.load(mean_ptr + j_val)
                    
                    sum_sq_diff = 0.0
                    for i_idx in range(BLOCK_SIZE_N):
                        if i_start + i_idx < N:
                            i_val = i_start + i_idx
                            data_idx = i_val * M + j_val
                            data_val = tl.load(data_ptr + data_idx)
                            diff = data_val - mean_j
                            sum_sq_diff += diff * diff
                    
                    stddev_vals = tl.where(j_idx == tl.arange(0, BLOCK_SIZE_M), 
                                         sum_sq_diff, stddev_vals)
        
        # Finalize stddev computation
        stddev_vals = stddev_vals / float_n
        stddev_vals = tl.sqrt(stddev_vals)
        stddev_vals = tl.where(stddev_vals <= eps, 1.0, stddev_vals)
        
        tl.store(stddev_ptr + j_offsets, stddev_vals, mask=j_mask)
    
    # Step 3: Center and reduce data
    sqrt_float_n = tl.sqrt(float_n)
    
    for i_start in range(0, N, BLOCK_SIZE_N):
        i_offsets = i_start + n_offsets
        i_mask = i_offsets < N
        
        for j_start in range(0, M, BLOCK_SIZE_M):
            j_offsets = j_start + m_offsets
            j_mask = j_offsets < M
            
            # Load mean and stddev values
            mean_vals = tl.load(mean_ptr + j_offsets, mask=j_mask)
            stddev_vals = tl.load(stddev_ptr + j_offsets, mask=j_mask)
            
            for i_idx in range(BLOCK_SIZE_N):
                if i_start + i_idx < N:
                    i_val = i_start + i_idx
                    for j_idx in range(BLOCK_SIZE_M):
                        if j_start + j_idx < M:
                            j_val = j_start + j_idx
                            data_idx = i_val * M + j_val
                            
                            data_val = tl.load(data_ptr + data_idx)
                            mean_j = tl.load(mean_ptr + j_val)
                            stddev_j = tl.load(stddev_ptr + j_val)
                            
                            centered = data_val - mean_j
                            normalized = centered / (sqrt_float_n * stddev_j)
                            
                            tl.store(data_ptr + data_idx, normalized)
    
    # Step 4: Compute correlation matrix
    # Set diagonal elements to 1.0
    for i in range(M):
        corr_idx = i * M + i
        tl.store(corr_ptr + corr_idx, 1.0)
    
    # Compute upper triangle and mirror to lower triangle
    for i in range(M - 1):
        for j in range(i + 1, M):
            corr_val = 0.0
            
            for k_start in range(0, N, BLOCK_SIZE_N):
                for k_idx in range(BLOCK_SIZE_N):
                    if k_start + k_idx < N:
                        k_val = k_start + k_idx
                        data_ki = tl.load(data_ptr + k_val * M + i)
                        data_kj = tl.load(data_ptr + k_val * M + j)
                        corr_val += data_ki * data_kj
            
            # Store symmetric values
            corr_ij_idx = i * M + j
            corr_ji_idx = j * M + i
            tl.store(corr_ptr + corr_ij_idx, corr_val)
            tl.store(corr_ptr + corr_ji_idx, corr_val)


def correlation_triton(corr, data, mean, stddev, eps, float_n, M, N):
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 32
    
    # Launch kernel with single thread block since we handle all computation inside
    grid = (1,)
    
    correlation_kernel[grid](
        corr, data, mean, stddev,
        eps, float_n, M, N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )