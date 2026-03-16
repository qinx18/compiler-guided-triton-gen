import triton
import triton.language as tl
import torch

@triton.jit
def compute_mean_kernel(data_ptr, mean_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    j = tl.program_id(0)
    if j >= M:
        return
    
    sum_val = 0.0
    i_offsets = tl.arange(0, BLOCK_SIZE)
    
    for i_start in range(0, N, BLOCK_SIZE):
        i_indices = i_start + i_offsets
        mask = i_indices < N
        data_indices = i_indices * M + j
        vals = tl.load(data_ptr + data_indices, mask=mask, other=0.0)
        sum_val += tl.sum(vals)
    
    mean_val = sum_val / N
    tl.store(mean_ptr + j, mean_val)

@triton.jit
def subtract_mean_kernel(data_ptr, mean_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_start = pid * BLOCK_SIZE
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = i_start + i_offsets
    mask_i = i_indices < N
    
    for j in range(M):
        mean_val = tl.load(mean_ptr + j)
        data_indices = i_indices * M + j
        data_vals = tl.load(data_ptr + data_indices, mask=mask_i, other=0.0)
        new_vals = data_vals - mean_val
        tl.store(data_ptr + data_indices, new_vals, mask=mask_i)

@triton.jit
def compute_covariance_kernel(data_ptr, cov_ptr, M: tl.constexpr, N: tl.constexpr, 
                             float_n, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    i_start = pid_m * BM
    j_start = pid_n * BN
    
    i_offsets = tl.arange(0, BM)
    j_offsets = tl.arange(0, BN)
    
    i_indices = i_start + i_offsets
    j_indices = j_start + j_offsets
    
    mask_i = i_indices < M
    mask_j = j_indices < M
    
    # Only compute upper triangular part
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    
    k_offsets = tl.arange(0, BK)
    
    for k_start in range(0, N, BK):
        k_indices = k_start + k_offsets
        mask_k = k_indices < N
        
        # Load data[k, i] for current block
        data_i_indices = k_indices[:, None] * M + i_indices[None, :]
        data_i = tl.load(data_ptr + data_i_indices, 
                        mask=mask_k[:, None] & mask_i[None, :], other=0.0)
        
        # Load data[k, j] for current block  
        data_j_indices = k_indices[:, None] * M + j_indices[None, :]
        data_j = tl.load(data_ptr + data_j_indices,
                        mask=mask_k[:, None] & mask_j[None, :], other=0.0)
        
        # Accumulate dot product
        acc += tl.dot(data_i.trans(), data_j)
    
    # Scale by (N-1)
    acc = acc / (float_n - 1.0)
    
    # Store results for upper triangular part
    for bi in range(BM):
        for bj in range(BN):
            i = i_start + bi
            j = j_start + bj
            if (i < M) & (j < M) & (i <= j):
                val = acc[bi, bj]
                tl.store(cov_ptr + i * M + j, val)
                if i != j:
                    tl.store(cov_ptr + j * M + i, val)

def covariance_triton(cov, data, mean, float_n, M, N):
    # Phase 1: Compute mean
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    compute_mean_kernel[grid](data, mean, M, N, BLOCK_SIZE)
    
    # Phase 2: Subtract mean from data
    BLOCK_SIZE = 128
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    subtract_mean_kernel[grid](data, mean, M, N, BLOCK_SIZE)
    
    # Phase 3: Compute covariance matrix
    BM = BN = 32
    BK = 64
    grid = (triton.cdiv(M, BM), triton.cdiv(M, BN))
    compute_covariance_kernel[grid](data, cov, M, N, float_n, BM, BN, BK)