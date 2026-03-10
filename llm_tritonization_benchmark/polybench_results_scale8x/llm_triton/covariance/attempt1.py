import triton
import triton.language as tl
import torch

@triton.jit
def mean_kernel(data_ptr, mean_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr):
    pid_m = tl.program_id(0)
    
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offset < M
    
    sum_vals = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    for n in range(N):
        data_offsets = n * M + m_offset
        data_vals = tl.load(data_ptr + data_offsets, mask=mask_m, other=0.0)
        sum_vals += data_vals
    
    mean_vals = sum_vals / N
    tl.store(mean_ptr + m_offset, mean_vals, mask=mask_m)

@triton.jit
def subtract_mean_kernel(data_ptr, mean_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    
    n_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    
    mask_n = n_offset < N
    mask_m = m_offset < M
    
    mean_vals = tl.load(mean_ptr + m_offset, mask=mask_m, other=0.0)
    
    for n_idx in range(BLOCK_N):
        n_pos = pid_n * BLOCK_N + n_idx
        if n_pos < N:
            data_offsets = n_pos * M + m_offset
            data_vals = tl.load(data_ptr + data_offsets, mask=mask_m, other=0.0)
            result_vals = data_vals - mean_vals
            tl.store(data_ptr + data_offsets, result_vals, mask=mask_m)

@triton.jit
def covariance_kernel(data_ptr, cov_ptr, float_n, M: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_offset = pid_i * BLOCK_M + tl.arange(0, BLOCK_M)
    j_offset = pid_j * BLOCK_M + tl.arange(0, BLOCK_M)
    
    mask_i = i_offset < M
    mask_j = j_offset < M
    
    for i_idx in range(BLOCK_M):
        i_pos = pid_i * BLOCK_M + i_idx
        if i_pos < M:
            for j_idx in range(BLOCK_M):
                j_pos = pid_j * BLOCK_M + j_idx
                if j_pos < M:
                    if i_pos <= j_pos:
                        sum_val = 0.0
                        
                        for k in range(N):
                            data_i_offset = k * M + i_pos
                            data_j_offset = k * M + j_pos
                            data_i_val = tl.load(data_ptr + data_i_offset)
                            data_j_val = tl.load(data_ptr + data_j_offset)
                            sum_val += data_i_val * data_j_val
                        
                        cov_val = sum_val / (float_n - 1.0)
                        
                        cov_ij_offset = i_pos * M + j_pos
                        cov_ji_offset = j_pos * M + i_pos
                        
                        tl.store(cov_ptr + cov_ij_offset, cov_val)
                        tl.store(cov_ptr + cov_ji_offset, cov_val)

def covariance_triton(cov, data, mean, float_n, M, N):
    BLOCK_M = 128
    BLOCK_N = 128
    
    # Phase 1: Compute mean
    grid_mean = (triton.cdiv(M, BLOCK_M),)
    mean_kernel[grid_mean](data, mean, M, N, BLOCK_M)
    
    # Phase 2: Subtract mean from data
    grid_subtract = (triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M))
    subtract_mean_kernel[grid_subtract](data, mean, M, N, BLOCK_N, BLOCK_M)
    
    # Phase 3: Compute covariance matrix
    grid_cov = (triton.cdiv(M, BLOCK_M), triton.cdiv(M, BLOCK_M))
    covariance_kernel[grid_cov](data, cov, float_n, M, N, BLOCK_M)