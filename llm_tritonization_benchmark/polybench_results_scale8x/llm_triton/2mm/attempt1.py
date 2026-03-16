import triton
import triton.language as tl
import torch

@triton.jit
def k2mm_phase1_kernel(
    A_ptr, B_ptr, tmp_ptr,
    alpha,
    NI: tl.constexpr, NJ: tl.constexpr, NK: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Matrix multiply: tmp = alpha * A * B
    for k_start in range(0, NK, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # Load A tile
        a_mask = (offs_m[:, None] < NI) & (offs_k[None, :] < NK)
        a_ptrs = A_ptr + offs_m[:, None] * NK + offs_k[None, :]
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load B tile
        b_mask = (offs_k[:, None] < NK) & (offs_n[None, :] < NJ)
        b_ptrs = B_ptr + offs_k[:, None] * NJ + offs_n[None, :]
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Accumulate
        acc += tl.dot(a_tile, b_tile)
    
    # Scale by alpha and store
    acc = alpha * acc
    
    # Store tmp
    tmp_mask = (offs_m[:, None] < NI) & (offs_n[None, :] < NJ)
    tmp_ptrs = tmp_ptr + offs_m[:, None] * NJ + offs_n[None, :]
    tl.store(tmp_ptrs, acc, mask=tmp_mask)

@triton.jit
def k2mm_phase2_kernel(
    tmp_ptr, C_ptr, D_ptr,
    beta,
    NI: tl.constexpr, NJ: tl.constexpr, NL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Load existing D values and scale by beta
    d_mask = (offs_m[:, None] < NI) & (offs_n[None, :] < NL)
    d_ptrs = D_ptr + offs_m[:, None] * NL + offs_n[None, :]
    acc = tl.load(d_ptrs, mask=d_mask, other=0.0) * beta
    
    # Matrix multiply: D += tmp * C
    for k_start in range(0, NJ, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # Load tmp tile
        tmp_mask = (offs_m[:, None] < NI) & (offs_k[None, :] < NJ)
        tmp_ptrs = tmp_ptr + offs_m[:, None] * NJ + offs_k[None, :]
        tmp_tile = tl.load(tmp_ptrs, mask=tmp_mask, other=0.0)
        
        # Load C tile
        c_mask = (offs_k[:, None] < NJ) & (offs_n[None, :] < NL)
        c_ptrs = C_ptr + offs_k[:, None] * NL + offs_n[None, :]
        c_tile = tl.load(c_ptrs, mask=c_mask, other=0.0)
        
        # Accumulate
        acc += tl.dot(tmp_tile, c_tile)
    
    # Store result
    tl.store(d_ptrs, acc, mask=d_mask)

def k2mm_triton(A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL):
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16
    
    # Phase 1: tmp = alpha * A * B
    grid1 = (triton.cdiv(NI, BLOCK_M), triton.cdiv(NJ, BLOCK_N))
    k2mm_phase1_kernel[grid1](
        A, B, tmp,
        alpha,
        NI, NJ, NK,
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    
    # Phase 2: D = beta * D + tmp * C
    grid2 = (triton.cdiv(NI, BLOCK_M), triton.cdiv(NL, BLOCK_N))
    k2mm_phase2_kernel[grid2](
        tmp, C, D,
        beta,
        NI, NJ, NL,
        BLOCK_M, BLOCK_N, BLOCK_K
    )

def k2mm_kernel(A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL):
    k2mm_triton(A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL)