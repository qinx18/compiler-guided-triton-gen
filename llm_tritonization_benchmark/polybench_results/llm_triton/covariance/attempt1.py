import triton
import triton.language as tl
import torch

@triton.jit
def compute_mean_kernel(data_ptr, mean_ptr, N: tl.constexpr, M: tl.constexpr, float_n):
    j = tl.program_id(0)
    
    if j >= M:
        return
    
    sum_val = 0.0
    for i in range(N):
        data_idx = i * M + j
        sum_val += tl.load(data_ptr + data_idx)
    
    mean_val = sum_val / float_n
    tl.store(mean_ptr + j, mean_val)

@triton.jit
def subtract_mean_kernel(data_ptr, mean_ptr, N: tl.constexpr, M: tl.constexpr):
    i = tl.program_id(0)
    
    if i >= N:
        return
    
    for j in range(M):
        data_idx = i * M + j
        mean_val = tl.load(mean_ptr + j)
        data_val = tl.load(data_ptr + data_idx)
        tl.store(data_ptr + data_idx, data_val - mean_val)

@triton.jit
def compute_covariance_kernel(data_ptr, cov_ptr, N: tl.constexpr, M: tl.constexpr, float_n):
    i = tl.program_id(0)
    
    if i >= M:
        return
    
    for j in range(i, M):
        cov_val = 0.0
        for k in range(N):
            data_ki = tl.load(data_ptr + k * M + i)
            data_kj = tl.load(data_ptr + k * M + j)
            cov_val += data_ki * data_kj
        
        cov_val = cov_val / (float_n - 1.0)
        
        # Store symmetric values
        tl.store(cov_ptr + i * M + j, cov_val)
        tl.store(cov_ptr + j * M + i, cov_val)

def covariance_triton(cov, data, mean, float_n, M, N):
    # Phase 1: Compute mean
    grid = (M,)
    compute_mean_kernel[grid](data, mean, N, M, float_n)
    
    # Phase 2: Subtract mean from data
    grid = (N,)
    subtract_mean_kernel[grid](data, mean, N, M)
    
    # Phase 3: Compute covariance matrix
    grid = (M,)
    compute_covariance_kernel[grid](data, cov, N, M, float_n)