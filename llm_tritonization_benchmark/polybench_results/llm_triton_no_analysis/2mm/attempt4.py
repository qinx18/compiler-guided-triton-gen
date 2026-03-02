import triton
import triton.language as tl
import torch

@triton.jit
def k2mm_kernel(A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL, 
                BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_L: tl.constexpr):
    
    # First loop: tmp[i][j] = alpha * A[i][k] * B[k][j]
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_start = pid_i * BLOCK_I
    j_start = pid_j * BLOCK_J
    
    i_offsets = tl.arange(0, BLOCK_I)
    j_offsets = tl.arange(0, BLOCK_J)
    k_offsets = tl.arange(0, BLOCK_K)
    
    i_indices = i_start + i_offsets
    j_indices = j_start + j_offsets
    
    i_mask = i_indices < NI
    j_mask = j_indices < NJ
    
    # Initialize accumulator for tmp[i][j]
    acc = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)
    
    # Compute tmp = alpha * A * B
    for k_block in range(0, NK, BLOCK_K):
        k_indices = k_block + k_offsets
        k_mask = k_indices < NK
        
        # Load A[i, k]
        a_ptrs = A + i_indices[:, None] * NK + k_indices[None, :]
        mask_a = i_mask[:, None] & k_mask[None, :]
        a_vals = tl.load(a_ptrs, mask=mask_a, other=0.0)
        
        # Load B[k, j]
        b_ptrs = B + k_indices[:, None] * NJ + j_indices[None, :]
        mask_b = k_mask[:, None] & j_mask[None, :]
        b_vals = tl.load(b_ptrs, mask=mask_b, other=0.0)
        
        # Accumulate A * B
        acc += tl.dot(a_vals, b_vals)
    
    # Store tmp (already scaled by alpha)
    tmp_ptrs = tmp + i_indices[:, None] * NJ + j_indices[None, :]
    mask_tmp = i_mask[:, None] & j_mask[None, :]
    tl.store(tmp_ptrs, alpha * acc, mask=mask_tmp)

@triton.jit
def k2mm_kernel2(C, D, tmp, beta, NI, NJ, NL,
                 BLOCK_I: tl.constexpr, BLOCK_L: tl.constexpr, BLOCK_J: tl.constexpr):
    
    # Second loop: D[i][l] = beta * D[i][l] + tmp[i][j] * C[j][l]
    pid_i = tl.program_id(0)
    pid_l = tl.program_id(1)
    
    i_start = pid_i * BLOCK_I
    l_start = pid_l * BLOCK_L
    
    i_offsets = tl.arange(0, BLOCK_I)
    l_offsets = tl.arange(0, BLOCK_L)
    j_offsets = tl.arange(0, BLOCK_J)
    
    i_indices = i_start + i_offsets
    l_indices = l_start + l_offsets
    
    i_mask = i_indices < NI
    l_mask = l_indices < NL
    
    # Initialize accumulator with beta * D
    d_ptrs = D + i_indices[:, None] * NL + l_indices[None, :]
    mask_d = i_mask[:, None] & l_mask[None, :]
    acc = tl.load(d_ptrs, mask=mask_d, other=0.0) * beta
    
    # Compute tmp * C and add to acc
    for j_block in range(0, NJ, BLOCK_J):
        j_indices = j_block + j_offsets
        j_mask = j_indices < NJ
        
        # Load tmp[i, j]
        tmp_ptrs = tmp + i_indices[:, None] * NJ + j_indices[None, :]
        mask_tmp = i_mask[:, None] & j_mask[None, :]
        tmp_vals = tl.load(tmp_ptrs, mask=mask_tmp, other=0.0)
        
        # Load C[j, l]
        c_ptrs = C + j_indices[:, None] * NL + l_indices[None, :]
        mask_c = j_mask[:, None] & l_mask[None, :]
        c_vals = tl.load(c_ptrs, mask=mask_c, other=0.0)
        
        # Accumulate tmp * C
        acc += tl.dot(tmp_vals, c_vals)
    
    # Store final result in D
    tl.store(d_ptrs, acc, mask=mask_d)

def k2mm_triton(A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL):
    BLOCK_I = 16
    BLOCK_J = 16
    BLOCK_K = 16
    BLOCK_L = 16
    
    # First kernel: compute tmp = alpha * A * B
    grid1 = (triton.cdiv(NI, BLOCK_I), triton.cdiv(NJ, BLOCK_J))
    k2mm_kernel[grid1](
        A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL,
        BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J, BLOCK_K=BLOCK_K, BLOCK_L=BLOCK_L
    )
    
    # Second kernel: compute D = beta * D + tmp * C
    grid2 = (triton.cdiv(NI, BLOCK_I), triton.cdiv(NL, BLOCK_L))
    k2mm_kernel2[grid2](
        C, D, tmp, beta, NI, NJ, NL,
        BLOCK_I=BLOCK_I, BLOCK_L=BLOCK_L, BLOCK_J=BLOCK_J
    )