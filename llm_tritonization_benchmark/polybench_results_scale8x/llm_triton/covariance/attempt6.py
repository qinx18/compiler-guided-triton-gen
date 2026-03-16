import triton
import triton.language as tl
import torch

@triton.jit
def covariance_phase1_kernel(data_ptr, mean_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_N: tl.constexpr):
    j = tl.program_id(0)
    
    if j >= M:
        return
    
    # Compute mean[j] = sum(data[i][j]) / N for all i
    sum_val = 0.0
    
    n_offsets = tl.arange(0, BLOCK_N)
    for n_start in range(0, N, BLOCK_N):
        current_n = n_start + n_offsets
        mask = current_n < N
        
        # Load data[current_n][j]
        data_offsets = current_n * M + j
        data_vals = tl.load(data_ptr + data_offsets, mask=mask, other=0.0)
        sum_val += tl.sum(data_vals)
    
    mean_val = sum_val / N
    tl.store(mean_ptr + j, mean_val)

@triton.jit
def covariance_phase2_kernel(data_ptr, mean_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr):
    i = tl.program_id(0)
    
    if i >= N:
        return
    
    # Subtract mean from data[i][j] for all j
    m_offsets = tl.arange(0, BLOCK_M)
    for m_start in range(0, M, BLOCK_M):
        current_m = m_start + m_offsets
        mask = current_m < M
        
        # Load data[i][current_m] and mean[current_m]
        data_offsets = i * M + current_m
        data_vals = tl.load(data_ptr + data_offsets, mask=mask)
        mean_vals = tl.load(mean_ptr + current_m, mask=mask)
        
        # Subtract and store back
        result = data_vals - mean_vals
        tl.store(data_ptr + data_offsets, result, mask=mask)

@triton.jit
def covariance_phase3_kernel(data_ptr, cov_ptr, float_n, M: tl.constexpr, N: tl.constexpr, BLOCK_N: tl.constexpr):
    i = tl.program_id(0)
    j_offset = tl.program_id(1)
    j = i + j_offset
    
    if i >= M or j >= M:
        return
    
    # Compute cov[i][j] = sum(data[k][i] * data[k][j]) / (N-1) for all k
    sum_val = 0.0
    
    n_offsets = tl.arange(0, BLOCK_N)
    for n_start in range(0, N, BLOCK_N):
        current_n = n_start + n_offsets
        mask = current_n < N
        
        # Load data[current_n][i] and data[current_n][j]
        data_i_offsets = current_n * M + i
        data_j_offsets = current_n * M + j
        data_i_vals = tl.load(data_ptr + data_i_offsets, mask=mask, other=0.0)
        data_j_vals = tl.load(data_ptr + data_j_offsets, mask=mask, other=0.0)
        
        sum_val += tl.sum(data_i_vals * data_j_vals)
    
    cov_val = sum_val / (float_n - 1.0)
    
    # Store cov[i][j] and cov[j][i]
    cov_ij_offset = i * M + j
    cov_ji_offset = j * M + i
    tl.store(cov_ptr + cov_ij_offset, cov_val)
    tl.store(cov_ptr + cov_ji_offset, cov_val)

def covariance_triton(cov, data, mean, float_n, M, N):
    BLOCK_N = triton.next_power_of_2(min(1024, N))
    BLOCK_M = triton.next_power_of_2(min(1024, M))
    
    # Phase 1: Compute means
    grid1 = (M,)
    covariance_phase1_kernel[grid1](
        data, mean, M, N, BLOCK_N
    )
    
    # Phase 2: Subtract means from data
    grid2 = (N,)
    covariance_phase2_kernel[grid2](
        data, mean, M, N, BLOCK_M
    )
    
    # Phase 3: Compute covariance matrix (upper triangular)
    grid3 = (M, M)
    covariance_phase3_kernel[grid3](
        data, cov, float_n, M, N, BLOCK_N
    )