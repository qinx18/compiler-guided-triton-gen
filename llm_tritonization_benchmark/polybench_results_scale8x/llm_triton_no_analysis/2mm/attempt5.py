import triton
import triton.language as tl
import torch

@triton.jit
def k2mm_kernel(A_ptr, B_ptr, C_ptr, D_ptr, tmp_ptr, alpha, beta,
                NI: tl.constexpr, NJ: tl.constexpr, NK: tl.constexpr, NL: tl.constexpr,
                BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_L: tl.constexpr,
                stage: tl.constexpr):
    
    pid_i = tl.program_id(0)
    pid_second = tl.program_id(1)
    
    if stage == 0:
        # Stage 1: tmp[i][j] = alpha * sum(A[i][k] * B[k][j])
        pid_j = pid_second
        
        i_start = pid_i * BLOCK_I
        j_start = pid_j * BLOCK_J
        
        i_offsets = tl.arange(0, BLOCK_I)
        j_offsets = tl.arange(0, BLOCK_J)
        k_offsets = tl.arange(0, BLOCK_K)
        
        i_mask = (i_start + i_offsets) < NI
        j_mask = (j_start + j_offsets) < NJ
        
        tmp_acc = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)
        
        for k_block in range(0, NK, BLOCK_K):
            k_mask = (k_block + k_offsets) < NK
            
            A_offsets = (i_start + i_offsets[:, None]) * NK + (k_block + k_offsets[None, :])
            A_mask = i_mask[:, None] & k_mask[None, :]
            A_block = tl.load(A_ptr + A_offsets, mask=A_mask, other=0.0)
            
            B_offsets = (k_block + k_offsets[:, None]) * NJ + (j_start + j_offsets[None, :])
            B_mask = k_mask[:, None] & j_mask[None, :]
            B_block = tl.load(B_ptr + B_offsets, mask=B_mask, other=0.0)
            
            tmp_acc += tl.dot(A_block, B_block)
        
        tmp_result = alpha * tmp_acc
        tmp_offsets = (i_start + i_offsets[:, None]) * NJ + (j_start + j_offsets[None, :])
        tmp_mask = i_mask[:, None] & j_mask[None, :]
        tl.store(tmp_ptr + tmp_offsets, tmp_result, mask=tmp_mask)
    
    else:
        # Stage 2: D[i][l] = beta * D[i][l] + sum(tmp[i][j] * C[j][l])
        pid_l = pid_second
        
        i_start = pid_i * BLOCK_I
        l_start = pid_l * BLOCK_L
        
        i_offsets = tl.arange(0, BLOCK_I)
        l_offsets = tl.arange(0, BLOCK_L)
        j_offsets = tl.arange(0, BLOCK_J)
        
        i_mask = (i_start + i_offsets) < NI
        l_mask = (l_start + l_offsets) < NL
        
        D_offsets = (i_start + i_offsets[:, None]) * NL + (l_start + l_offsets[None, :])
        D_mask = i_mask[:, None] & l_mask[None, :]
        D_block = tl.load(D_ptr + D_offsets, mask=D_mask, other=0.0)
        D_acc = beta * D_block
        
        for j_block in range(0, NJ, BLOCK_J):
            j_mask = (j_block + j_offsets) < NJ
            
            tmp_offsets = (i_start + i_offsets[:, None]) * NJ + (j_block + j_offsets[None, :])
            tmp_mask = i_mask[:, None] & j_mask[None, :]
            tmp_block = tl.load(tmp_ptr + tmp_offsets, mask=tmp_mask, other=0.0)
            
            C_offsets = (j_block + j_offsets[:, None]) * NL + (l_start + l_offsets[None, :])
            C_mask = j_mask[:, None] & l_mask[None, :]
            C_block = tl.load(C_ptr + C_offsets, mask=C_mask, other=0.0)
            
            D_acc += tl.dot(tmp_block, C_block)
        
        tl.store(D_ptr + D_offsets, D_acc, mask=D_mask)

def k2mm_triton(A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL):
    BLOCK_I = 32
    BLOCK_J = 32
    BLOCK_K = 32
    BLOCK_L = 32
    
    # Stage 1: compute tmp = alpha * A @ B
    grid1 = (triton.cdiv(NI, BLOCK_I), triton.cdiv(NJ, BLOCK_J))
    k2mm_kernel[grid1](
        A, B, C, D, tmp, 
        alpha, beta,
        NI, NJ, NK, NL,
        BLOCK_I, BLOCK_J, BLOCK_K, BLOCK_L,
        stage=0
    )
    
    # Stage 2: compute D = beta * D + tmp @ C  
    grid2 = (triton.cdiv(NI, BLOCK_I), triton.cdiv(NL, BLOCK_L))
    k2mm_kernel[grid2](
        A, B, C, D, tmp,
        alpha, beta,
        NI, NJ, NK, NL,
        BLOCK_I, BLOCK_J, BLOCK_K, BLOCK_L,
        stage=1
    )