import triton
import triton.language as tl
import torch

@triton.jit
def k2mm_kernel_phase1(A_ptr, B_ptr, tmp_ptr, alpha, 
                       NI: tl.constexpr, NJ: tl.constexpr, NK: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, NK, BLOCK_K):
        k_offs = k + offs_k
        
        a_ptrs = A_ptr + offs_m[:, None] * NK + k_offs[None, :]
        b_ptrs = B_ptr + k_offs[:, None] * NJ + offs_n[None, :]
        
        mask_a = (offs_m[:, None] < NI) & (k_offs[None, :] < NK)
        mask_b = (k_offs[:, None] < NK) & (offs_n[None, :] < NJ)
        
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        
        acc += tl.dot(a, b)
    
    acc = alpha * acc
    
    tmp_ptrs = tmp_ptr + offs_m[:, None] * NJ + offs_n[None, :]
    mask_tmp = (offs_m[:, None] < NI) & (offs_n[None, :] < NJ)
    
    tl.store(tmp_ptrs, acc, mask=mask_tmp)

@triton.jit
def k2mm_kernel_phase2(tmp_ptr, C_ptr, D_ptr, beta,
                       NI: tl.constexpr, NJ: tl.constexpr, NL: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, NJ, BLOCK_K):
        k_offs = k + offs_k
        
        tmp_ptrs = tmp_ptr + offs_m[:, None] * NJ + k_offs[None, :]
        c_ptrs = C_ptr + k_offs[:, None] * NL + offs_n[None, :]
        
        mask_tmp = (offs_m[:, None] < NI) & (k_offs[None, :] < NJ)
        mask_c = (k_offs[:, None] < NJ) & (offs_n[None, :] < NL)
        
        tmp_vals = tl.load(tmp_ptrs, mask=mask_tmp, other=0.0)
        c_vals = tl.load(c_ptrs, mask=mask_c, other=0.0)
        
        acc += tl.dot(tmp_vals, c_vals)
    
    d_ptrs = D_ptr + offs_m[:, None] * NL + offs_n[None, :]
    mask_d = (offs_m[:, None] < NI) & (offs_n[None, :] < NL)
    
    d_vals = tl.load(d_ptrs, mask=mask_d, other=0.0)
    d_vals = beta * d_vals + acc
    
    tl.store(d_ptrs, d_vals, mask=mask_d)

def k2mm_triton(A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL):
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16
    
    grid_phase1 = (triton.cdiv(NI, BLOCK_M), triton.cdiv(NJ, BLOCK_N))
    k2mm_kernel_phase1[grid_phase1](
        A, B, tmp, alpha,
        NI, NJ, NK,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    grid_phase2 = (triton.cdiv(NI, BLOCK_M), triton.cdiv(NL, BLOCK_N))
    k2mm_kernel_phase2[grid_phase2](
        tmp, C, D, beta,
        NI, NJ, NL,
        BLOCK_M, BLOCK_N, BLOCK_K
    )

@triton.jit
def k2mm_kernel(A_ptr, B_ptr, C_ptr, D_ptr, tmp_ptr, alpha, beta,
                NI: tl.constexpr, NJ: tl.constexpr, NK: tl.constexpr, NL: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pass