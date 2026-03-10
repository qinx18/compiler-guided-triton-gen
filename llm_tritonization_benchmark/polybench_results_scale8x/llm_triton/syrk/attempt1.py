import triton
import triton.language as tl
import torch

@triton.jit
def syrk_kernel(A_ptr, C_ptr, alpha, beta, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for i dimension
    pid_i = tl.program_id(0)
    
    # Each program handles one row i
    i = pid_i
    if i >= N:
        return
    
    # First, multiply C[i][j] by beta for j <= i
    j_offsets = tl.arange(0, BLOCK_SIZE)
    for j_block_start in range(0, i + 1, BLOCK_SIZE):
        current_j = j_block_start + j_offsets
        mask_j = (current_j <= i) & (current_j < N)
        
        # Load C[i][j]
        c_offsets = i * N + current_j
        c_vals = tl.load(C_ptr + c_offsets, mask=mask_j, other=0.0)
        
        # Multiply by beta
        c_vals = c_vals * beta
        
        # Store back
        tl.store(C_ptr + c_offsets, c_vals, mask=mask_j)
    
    # Now compute the matrix multiplication part
    for k in range(M):
        # Load A[i][k]
        a_ik = tl.load(A_ptr + i * M + k)
        
        # Process j in blocks
        for j_block_start in range(0, i + 1, BLOCK_SIZE):
            current_j = j_block_start + j_offsets
            mask_j = (current_j <= i) & (current_j < N)
            
            # Load A[j][k] for all j in the block
            a_offsets = current_j * M + k
            a_jk = tl.load(A_ptr + a_offsets, mask=mask_j, other=0.0)
            
            # Load current C[i][j]
            c_offsets = i * N + current_j
            c_vals = tl.load(C_ptr + c_offsets, mask=mask_j, other=0.0)
            
            # Update C[i][j] += alpha * A[i][k] * A[j][k]
            c_vals = c_vals + alpha * a_ik * a_jk
            
            # Store back
            tl.store(C_ptr + c_offsets, c_vals, mask=mask_j)

def syrk_triton(A, C, alpha, beta, M, N):
    # Launch kernel with one program per row
    grid = (N,)
    BLOCK_SIZE = 64
    
    syrk_kernel[grid](
        A, C, alpha, beta, M, N, BLOCK_SIZE
    )