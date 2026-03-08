import triton
import triton.language as tl
import torch

@triton.jit
def syr2k_kernel(A_ptr, B_ptr, C_ptr, alpha, beta, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for the i dimension
    i = tl.program_id(0)
    
    if i >= N:
        return
    
    # First, multiply C[i][j] *= beta for j <= i
    j_offsets = tl.arange(0, BLOCK_SIZE)
    for j_block in range(0, i + 1, BLOCK_SIZE):
        j_indices = j_block + j_offsets
        mask = (j_indices <= i) & (j_indices < N)
        
        c_idx = i * N + j_indices
        c_vals = tl.load(C_ptr + c_idx, mask=mask)
        c_vals = c_vals * beta
        tl.store(C_ptr + c_idx, c_vals, mask=mask)
    
    # Then, accumulate the syr2k updates
    for k in range(M):
        # Load B[i][k] (scalar broadcast)
        b_ik = tl.load(B_ptr + i * M + k)
        
        for j_block in range(0, i + 1, BLOCK_SIZE):
            j_indices = j_block + j_offsets
            mask = (j_indices <= i) & (j_indices < N)
            
            # Load A[j][k] and B[j][k]
            a_jk_idx = j_indices * M + k
            b_jk_idx = j_indices * M + k
            
            a_jk = tl.load(A_ptr + a_jk_idx, mask=mask)
            b_jk = tl.load(B_ptr + b_jk_idx, mask=mask)
            
            # Load A[i][k] (scalar broadcast)
            a_ik = tl.load(A_ptr + i * M + k)
            
            # Compute the update: A[j][k]*alpha*B[i][k] + B[j][k]*alpha*A[i][k]
            update = a_jk * alpha * b_ik + b_jk * alpha * a_ik
            
            # Load current C[i][j] values and add update
            c_idx = i * N + j_indices
            c_vals = tl.load(C_ptr + c_idx, mask=mask)
            c_vals = c_vals + update
            tl.store(C_ptr + c_idx, c_vals, mask=mask)

def syr2k_triton(A, B, C, alpha, beta, M, N):
    # Determine block size
    BLOCK_SIZE = 32
    
    # Launch kernel with one program per row (i dimension)
    grid = (N,)
    
    syr2k_kernel[grid](
        A, B, C,
        alpha, beta,
        M, N, BLOCK_SIZE
    )