import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    
    A_ptrs = A_ptr + (m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak)
    B_ptrs = B_ptr + (k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        k_mask = (k_start + k_offsets) < K
        
        m_mask = m_offsets < M
        a_mask = m_mask[:, None] & k_mask[None, :]
        a = tl.load(A_ptrs, mask=a_mask, other=0.0)
        
        n_mask = n_offsets < N
        b_mask = k_mask[:, None] & n_mask[None, :]
        b = tl.load(B_ptrs, mask=b_mask, other=0.0)
        
        acc += tl.dot(a, b)
        
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk
    
    m_mask = m_offsets < M
    n_mask = n_offsets < N
    c_mask = m_mask[:, None] & n_mask[None, :]
    
    C_ptrs = C_ptr + (m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn)
    tl.store(C_ptrs, acc, mask=c_mask)

def k3mm_triton(A, B, C, D, E, F, G, NI, NJ, NK, NL, NM):
    BLOCK_SIZE = 32
    
    # E := A*B (NI x NJ = NI x NK * NK x NJ)
    grid = (triton.cdiv(NI, BLOCK_SIZE), triton.cdiv(NJ, BLOCK_SIZE))
    matmul_kernel[grid](
        A, B, E,
        NI, NJ, NK,
        NK, 1, NJ, 1, NJ, 1,
        BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE
    )
    
    # F := C*D (NJ x NL = NJ x NM * NM x NL) 
    grid = (triton.cdiv(NJ, BLOCK_SIZE), triton.cdiv(NL, BLOCK_SIZE))
    matmul_kernel[grid](
        C, D, F,
        NJ, NL, NM,
        NM, 1, NL, 1, NL, 1,
        BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE
    )
    
    # G := E*F (NI x NL = NI x NJ * NJ x NL)
    grid = (triton.cdiv(NI, BLOCK_SIZE), triton.cdiv(NL, BLOCK_SIZE))
    matmul_kernel[grid](
        E, F, G,
        NI, NL, NJ,
        NJ, 1, NL, 1, NL, 1,
        BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE
    )