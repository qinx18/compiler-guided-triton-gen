import triton
import triton.language as tl
import torch

@triton.jit
def k2mm_kernel(A_ptr, B_ptr, C_ptr, D_ptr, tmp_ptr, alpha, beta, NI, NJ, NK, NL, 
                A_stride0, A_stride1, B_stride0, B_stride1, C_stride0, C_stride1, 
                D_stride0, D_stride1, tmp_stride0, tmp_stride1,
                BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr):
    
    # First phase: tmp := alpha * A * B
    # Parallelize over (i, j) blocks
    pid = tl.program_id(0)
    num_blocks_j = (NJ + BLOCK_J - 1) // BLOCK_J
    i_block = (pid // num_blocks_j) * BLOCK_I
    j_block = (pid % num_blocks_j) * BLOCK_J
    
    i_offsets = tl.arange(0, BLOCK_I)
    j_offsets = tl.arange(0, BLOCK_J)
    k_offsets = tl.arange(0, BLOCK_K)
    
    i_idx = i_block + i_offsets
    j_idx = j_block + j_offsets
    
    i_mask = i_idx < NI
    j_mask = j_idx < NJ
    
    # Initialize tmp accumulator
    tmp_val = tl.zeros([BLOCK_I, BLOCK_J], dtype=tl.float32)
    
    # Compute tmp[i,j] = alpha * sum_k(A[i,k] * B[k,j])
    for k_block in range(0, NK, BLOCK_K):
        k_idx = k_block + k_offsets
        k_mask = k_idx < NK
        
        # Load A[i,k]
        A_ptrs = A_ptr + i_idx[:, None] * A_stride0 + k_idx[None, :] * A_stride1
        A_vals = tl.load(A_ptrs, mask=i_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Load B[k,j]
        B_ptrs = B_ptr + k_idx[:, None] * B_stride0 + j_idx[None, :] * B_stride1
        B_vals = tl.load(B_ptrs, mask=k_mask[:, None] & j_mask[None, :], other=0.0)
        
        # Accumulate
        tmp_val += alpha * tl.dot(A_vals, B_vals)
    
    # Store tmp values
    tmp_ptrs = tmp_ptr + i_idx[:, None] * tmp_stride0 + j_idx[None, :] * tmp_stride1
    tl.store(tmp_ptrs, tmp_val, mask=i_mask[:, None] & j_mask[None, :])

@triton.jit
def k2mm_kernel2(D_ptr, tmp_ptr, C_ptr, beta, NI, NJ, NL,
                 D_stride0, D_stride1, tmp_stride0, tmp_stride1, C_stride0, C_stride1,
                 BLOCK_I: tl.constexpr, BLOCK_L: tl.constexpr, BLOCK_J: tl.constexpr):
    
    # Second phase: D := tmp * C + beta * D  
    # Parallelize over (i, l) blocks
    pid = tl.program_id(0)
    num_blocks_l = (NL + BLOCK_L - 1) // BLOCK_L
    i_block = (pid // num_blocks_l) * BLOCK_I
    l_block = (pid % num_blocks_l) * BLOCK_L
    
    i_offsets = tl.arange(0, BLOCK_I)
    l_offsets = tl.arange(0, BLOCK_L)
    j_offsets = tl.arange(0, BLOCK_J)
    
    i_idx = i_block + i_offsets
    l_idx = l_block + l_offsets
    
    i_mask = i_idx < NI
    l_mask = l_idx < NL
    
    # Load and scale D by beta
    D_ptrs = D_ptr + i_idx[:, None] * D_stride0 + l_idx[None, :] * D_stride1
    D_vals = tl.load(D_ptrs, mask=i_mask[:, None] & l_mask[None, :], other=0.0)
    D_vals = D_vals * beta
    
    # Accumulate D[i,l] += sum_j(tmp[i,j] * C[j,l])
    for j_block in range(0, NJ, BLOCK_J):
        j_idx = j_block + j_offsets
        j_mask = j_idx < NJ
        
        # Load tmp[i,j]
        tmp_ptrs = tmp_ptr + i_idx[:, None] * tmp_stride0 + j_idx[None, :] * tmp_stride1
        tmp_vals = tl.load(tmp_ptrs, mask=i_mask[:, None] & j_mask[None, :], other=0.0)
        
        # Load C[j,l]
        C_ptrs = C_ptr + j_idx[:, None] * C_stride0 + l_idx[None, :] * C_stride1
        C_vals = tl.load(C_ptrs, mask=j_mask[:, None] & l_mask[None, :], other=0.0)
        
        # Accumulate
        D_vals += tl.dot(tmp_vals, C_vals)
    
    # Store D values
    tl.store(D_ptrs, D_vals, mask=i_mask[:, None] & l_mask[None, :])

def k2mm_triton(A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL):
    BLOCK_I = 16
    BLOCK_J = 16  
    BLOCK_K = 16
    BLOCK_L = 16
    
    # First phase: compute tmp = alpha * A * B
    grid1 = ((NI + BLOCK_I - 1) // BLOCK_I * (NJ + BLOCK_J - 1) // BLOCK_J,)
    
    k2mm_kernel[grid1](
        A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1), 
        C.stride(0), C.stride(1),
        D.stride(0), D.stride(1),
        tmp.stride(0), tmp.stride(1),
        BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J, BLOCK_K=BLOCK_K
    )
    
    # Second phase: compute D = tmp * C + beta * D
    grid2 = ((NI + BLOCK_I - 1) // BLOCK_I * (NL + BLOCK_L - 1) // BLOCK_L,)
    
    k2mm_kernel2[grid2](
        D, tmp, C, beta, NI, NJ, NL,
        D.stride(0), D.stride(1),
        tmp.stride(0), tmp.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_I=BLOCK_I, BLOCK_L=BLOCK_L, BLOCK_J=BLOCK_J
    )