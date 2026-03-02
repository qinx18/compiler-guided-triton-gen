import triton
import triton.language as tl
import torch

@triton.jit
def correlation_kernel(
    corr_ptr, data_ptr, mean_ptr, stddev_ptr,
    eps, float_n, M, N,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # Get program ID for parallelization
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Define offsets once
    m_offsets = tl.arange(0, BLOCK_SIZE_M)
    n_offsets = tl.arange(0, BLOCK_SIZE_N)
    
    # Step 1: Compute mean for each column j
    if pid_n == 0:  # Only first program computes mean
        j_start = pid_m * BLOCK_SIZE_M
        j_offsets = j_start + m_offsets
        j_mask = j_offsets < M
        
        mean_vals = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
        
        # Sum over all N elements for each j
        for i_start in range(0, N, BLOCK_SIZE_N):
            i_offsets = i_start + n_offsets
            i_mask = i_offsets < N
            
            # Load data values for current block
            data_vals = tl.zeros([BLOCK_SIZE_N, BLOCK_SIZE_M], dtype=tl.float32)
            for i_idx in range(BLOCK_SIZE_N):
                for j_idx in range(BLOCK_SIZE_M):
                    i_val = i_start + i_idx
                    j_val = j_start + j_idx
                    if i_val < N and j_val < M:
                        data_idx = i_val * M + j_val
                        val = tl.load(data_ptr + data_idx)
                        data_vals = tl.where((i_idx == tl.arange(0, BLOCK_SIZE_N)[:, None]) & 
                                           (j_idx == tl.arange(0, BLOCK_SIZE_M)[None, :]), 
                                           val, data_vals)
            
            # Sum along N dimension
            for j_idx in range(BLOCK_SIZE_M):
                if j_start + j_idx < M:
                    col_sum = 0.0
                    for i_idx in range(BLOCK_SIZE_N):
                        if i_start + i_idx < N:
                            i_val = i_start + i_idx
                            j_val = j_start + j_idx
                            data_idx = i_val * M + j_val
                            val = tl.load(data_ptr + data_idx)
                            col_sum += val
                    mean_vals = tl.where(j_idx == tl.arange(0, BLOCK_SIZE_M), 
                                       mean_vals + col_sum, mean_vals)
        
        # Divide by float_n and store
        mean_vals = mean_vals / float_n
        tl.store(mean_ptr + j_offsets, mean_vals, mask=j_mask)


@triton.jit
def correlation_stddev_kernel(
    data_ptr, mean_ptr, stddev_ptr,
    eps, float_n, M, N,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    
    m_offsets = tl.arange(0, BLOCK_SIZE_M)
    n_offsets = tl.arange(0, BLOCK_SIZE_N)
    
    j_start = pid_m * BLOCK_SIZE_M
    j_offsets = j_start + m_offsets
    j_mask = j_offsets < M
    
    # Load mean values
    mean_vals = tl.load(mean_ptr + j_offsets, mask=j_mask, other=0.0)
    
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
                                     stddev_vals + sum_sq_diff, stddev_vals)
    
    # Finalize stddev computation
    stddev_vals = stddev_vals / float_n
    stddev_vals = tl.sqrt(stddev_vals)
    stddev_vals = tl.where(stddev_vals <= eps, 1.0, stddev_vals)
    
    tl.store(stddev_ptr + j_offsets, stddev_vals, mask=j_mask)


@triton.jit
def correlation_normalize_kernel(
    data_ptr, mean_ptr, stddev_ptr,
    float_n, M, N,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    
    m_offsets = tl.arange(0, BLOCK_SIZE_M)
    n_offsets = tl.arange(0, BLOCK_SIZE_N)
    
    i_start = pid_n * BLOCK_SIZE_N
    j_start = pid_m * BLOCK_SIZE_M
    
    i_offsets = i_start + n_offsets
    j_offsets = j_start + m_offsets
    
    i_mask = i_offsets < N
    j_mask = j_offsets < M
    
    sqrt_float_n = tl.sqrt(float_n)
    
    # Load mean and stddev values
    mean_vals = tl.load(mean_ptr + j_offsets, mask=j_mask, other=0.0)
    stddev_vals = tl.load(stddev_ptr + j_offsets, mask=j_mask, other=1.0)
    
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


@triton.jit
def correlation_matrix_kernel(
    corr_ptr, data_ptr, M, N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Set diagonal to 1.0
    if pid == 0:
        for i in range(M):
            corr_idx = i * M + i
            tl.store(corr_ptr + corr_idx, 1.0)
    
    # Compute upper triangle
    for i in range(M - 1):
        for j in range(i + 1, M):
            corr_val = 0.0
            
            for k_start in range(0, N, BLOCK_SIZE):
                for k_idx in range(BLOCK_SIZE):
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
    
    # Step 1: Compute means
    grid_mean = (triton.cdiv(M, BLOCK_SIZE_M), 1)
    correlation_kernel[grid_mean](
        corr, data, mean, stddev,
        eps, float_n, M, N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    # Step 2: Compute stddev
    grid_stddev = (triton.cdiv(M, BLOCK_SIZE_M),)
    correlation_stddev_kernel[grid_stddev](
        data, mean, stddev,
        eps, float_n, M, N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    # Step 3: Normalize data
    grid_norm = (triton.cdiv(N, BLOCK_SIZE_N), triton.cdiv(M, BLOCK_SIZE_M))
    correlation_normalize_kernel[grid_norm](
        data, mean, stddev,
        float_n, M, N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    # Step 4: Compute correlation matrix
    grid_corr = (1,)
    correlation_matrix_kernel[grid_corr](
        corr, data, M, N,
        BLOCK_SIZE=BLOCK_SIZE_N
    )