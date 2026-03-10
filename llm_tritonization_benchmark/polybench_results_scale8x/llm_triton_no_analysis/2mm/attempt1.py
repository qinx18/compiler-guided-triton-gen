import triton
import triton.language as tl
import torch

@triton.jit
def k2mm_kernel(A_ptr, B_ptr, C_ptr, D_ptr, tmp_ptr, alpha, beta,
                NI: tl.constexpr, NJ: tl.constexpr, NK: tl.constexpr, NL: tl.constexpr,
                BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr):
    # Get program IDs
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Calculate block start positions
    i_start = pid_i * BLOCK_I
    j_start = pid_j * BLOCK_J
    
    # Create offset ranges
    i_offsets = tl.arange(0, BLOCK_I)
    j_offsets = tl.arange(0, BLOCK_J)
    k_offsets = tl.arange(0, BLOCK_K)
    
    # First loop: tmp[i][j] = alpha * sum(A[i][k] * B[k][j])
    i_mask = (i_start + i_offsets) < NI
    j_mask = (j_start + j_offsets) < NJ
    
    # Initialize accumulator for tmp
    tmp_acc = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)
    
    # Loop over K dimension in blocks
    for k_block in range(0, NK, BLOCK_K):
        k_mask = (k_block + k_offsets) < NK
        
        # Load A[i_start:i_start+BLOCK_I, k_block:k_block+BLOCK_K]
        A_offsets = (i_start + i_offsets[:, None]) * NK + (k_block + k_offsets[None, :])
        A_mask = i_mask[:, None] & k_mask[None, :]
        A_block = tl.load(A_ptr + A_offsets, mask=A_mask, other=0.0)
        
        # Load B[k_block:k_block+BLOCK_K, j_start:j_start+BLOCK_J]
        B_offsets = (k_block + k_offsets[:, None]) * NJ + (j_start + j_offsets[None, :])
        B_mask = k_mask[:, None] & j_mask[None, :]
        B_block = tl.load(B_ptr + B_offsets, mask=B_mask, other=0.0)
        
        # Accumulate matrix multiplication
        tmp_acc += tl.dot(A_block, B_block)
    
    # Scale by alpha and store tmp
    tmp_result = alpha * tmp_acc
    tmp_offsets = (i_start + i_offsets[:, None]) * NJ + (j_start + j_offsets[None, :])
    tmp_mask = i_mask[:, None] & j_mask[None, :]
    tl.store(tmp_ptr + tmp_offsets, tmp_result, mask=tmp_mask)

@triton.jit
def k2mm_kernel_second(tmp_ptr, C_ptr, D_ptr, beta,
                       NI: tl.constexpr, NJ: tl.constexpr, NL: tl.constexpr,
                       BLOCK_I: tl.constexpr, BLOCK_L: tl.constexpr, BLOCK_J: tl.constexpr):
    # Get program IDs
    pid_i = tl.program_id(0)
    pid_l = tl.program_id(1)
    
    # Calculate block start positions
    i_start = pid_i * BLOCK_I
    l_start = pid_l * BLOCK_L
    
    # Create offset ranges
    i_offsets = tl.arange(0, BLOCK_I)
    l_offsets = tl.arange(0, BLOCK_L)
    j_offsets = tl.arange(0, BLOCK_J)
    
    # Second loop: D[i][j] = beta * D[i][j] + sum(tmp[i][k] * C[k][j])
    i_mask = (i_start + i_offsets) < NI
    l_mask = (l_start + l_offsets) < NL
    
    # Load and scale D by beta
    D_offsets = (i_start + i_offsets[:, None]) * NL + (l_start + l_offsets[None, :])
    D_mask = i_mask[:, None] & l_mask[None, :]
    D_block = tl.load(D_ptr + D_offsets, mask=D_mask, other=0.0)
    D_acc = beta * D_block
    
    # Loop over J dimension in blocks
    for j_block in range(0, NJ, BLOCK_J):
        j_mask = (j_block + j_offsets) < NJ
        
        # Load tmp[i_start:i_start+BLOCK_I, j_block:j_block+BLOCK_J]
        tmp_offsets = (i_start + i_offsets[:, None]) * NJ + (j_block + j_offsets[None, :])
        tmp_mask = i_mask[:, None] & j_mask[None, :]
        tmp_block = tl.load(tmp_ptr + tmp_offsets, mask=tmp_mask, other=0.0)
        
        # Load C[j_block:j_block+BLOCK_J, l_start:l_start+BLOCK_L]
        C_offsets = (j_block + j_offsets[:, None]) * NL + (l_start + l_offsets[None, :])
        C_mask = j_mask[:, None] & l_mask[None, :]
        C_block = tl.load(C_ptr + C_offsets, mask=C_mask, other=0.0)
        
        # Accumulate matrix multiplication
        D_acc += tl.dot(tmp_block, C_block)
    
    # Store final result
    tl.store(D_ptr + D_offsets, D_acc, mask=D_mask)

def k2mm_triton(A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL):
    BLOCK_I = 32
    BLOCK_J = 32
    BLOCK_K = 32
    BLOCK_L = 32
    
    # First kernel: compute tmp = alpha * A @ B
    grid1 = (triton.cdiv(NI, BLOCK_I), triton.cdiv(NJ, BLOCK_J))
    k2mm_kernel[grid1](
        A, B, C, D, tmp, alpha, beta,
        NI, NJ, NK, NL,
        BLOCK_I, BLOCK_J, BLOCK_K
    )
    
    # Second kernel: compute D = beta * D + tmp @ C
    grid2 = (triton.cdiv(NI, BLOCK_I), triton.cdiv(NL, BLOCK_L))
    k2mm_kernel_second[grid2](
        tmp, C, D, beta,
        NI, NJ, NL,
        BLOCK_I, BLOCK_L, BLOCK_J
    )