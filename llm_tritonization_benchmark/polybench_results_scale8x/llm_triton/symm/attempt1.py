import torch
import triton
import triton.language as tl

@triton.jit
def symm_kernel(
    A_ptr, B_ptr, C_ptr,
    alpha, beta,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Get column index for this thread block
    j_block_start = tl.program_id(0) * BLOCK_SIZE
    j_offsets = j_block_start + tl.arange(0, BLOCK_SIZE)
    j_mask = j_offsets < N
    
    # Sequential loop over rows
    for i in range(M):
        # Compute temp2 using reduction
        temp2 = 0.0
        
        # Inner k loop for k < i
        for k in range(i):
            # Load B[i, j] for current row i
            b_i_offsets = i * N + j_offsets
            b_i_vals = tl.load(B_ptr + b_i_offsets, mask=j_mask, other=0.0)
            
            # Load A[i, k] (scalar)
            a_i_k = tl.load(A_ptr + i * M + k)
            
            # Update C[k, j] += alpha * B[i, j] * A[i, k]
            c_k_offsets = k * N + j_offsets
            c_k_vals = tl.load(C_ptr + c_k_offsets, mask=j_mask, other=0.0)
            c_k_vals += alpha * b_i_vals * a_i_k
            tl.store(C_ptr + c_k_offsets, c_k_vals, mask=j_mask)
            
            # Load B[k, j] for temp2 accumulation
            b_k_offsets = k * N + j_offsets
            b_k_vals = tl.load(B_ptr + b_k_offsets, mask=j_mask, other=0.0)
            
            # Accumulate temp2 += B[k, j] * A[i, k]
            temp2 += b_k_vals * a_i_k
        
        # Load current C[i, j]
        c_i_offsets = i * N + j_offsets
        c_i_vals = tl.load(C_ptr + c_i_offsets, mask=j_mask, other=0.0)
        
        # Load B[i, j]
        b_i_offsets = i * N + j_offsets
        b_i_vals = tl.load(B_ptr + b_i_offsets, mask=j_mask, other=0.0)
        
        # Load A[i, i] (scalar)
        a_ii = tl.load(A_ptr + i * M + i)
        
        # Update C[i, j] = beta * C[i, j] + alpha * B[i, j] * A[i, i] + alpha * temp2
        c_i_vals = beta * c_i_vals + alpha * b_i_vals * a_ii + alpha * temp2
        
        # Store updated C[i, j]
        tl.store(C_ptr + c_i_offsets, c_i_vals, mask=j_mask)

def symm_triton(A, B, C, alpha, beta, M, N):
    BLOCK_SIZE = 64
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    symm_kernel[grid](
        A, B, C,
        alpha, beta,
        M, N,
        BLOCK_SIZE
    )