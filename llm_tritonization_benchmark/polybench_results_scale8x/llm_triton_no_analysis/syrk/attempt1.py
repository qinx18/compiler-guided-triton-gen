import triton
import triton.language as tl
import torch

@triton.jit
def syrk_kernel(A_ptr, C_ptr, alpha, beta, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate which row this program handles
    i = pid
    
    if i >= N:
        return
    
    # First loop: multiply C[i][j] by beta for j <= i
    j_offsets = tl.arange(0, BLOCK_SIZE)
    for j_block in range(0, i + 1, BLOCK_SIZE):
        j_indices = j_block + j_offsets
        j_mask = (j_indices <= i) & (j_indices < N)
        
        c_offsets = i * N + j_indices
        c_vals = tl.load(C_ptr + c_offsets, mask=j_mask, other=0.0)
        c_vals = c_vals * beta
        tl.store(C_ptr + c_offsets, c_vals, mask=j_mask)
    
    # Second loop: accumulate alpha * A[i][k] * A[j][k]
    for k in range(M):
        # Load A[i][k]
        a_ik_offset = i * M + k
        a_ik = tl.load(A_ptr + a_ik_offset)
        
        # Process j in blocks
        for j_block in range(0, i + 1, BLOCK_SIZE):
            j_indices = j_block + j_offsets
            j_mask = (j_indices <= i) & (j_indices < N)
            
            # Load A[j][k] for all valid j
            a_jk_offsets = j_indices * M + k
            a_jk_vals = tl.load(A_ptr + a_jk_offsets, mask=j_mask, other=0.0)
            
            # Load current C[i][j] values
            c_offsets = i * N + j_indices
            c_vals = tl.load(C_ptr + c_offsets, mask=j_mask, other=0.0)
            
            # Update C[i][j] += alpha * A[i][k] * A[j][k]
            c_vals = c_vals + alpha * a_ik * a_jk_vals
            
            # Store back
            tl.store(C_ptr + c_offsets, c_vals, mask=j_mask)

def syrk_triton(A, C, alpha, beta, M, N):
    # Determine block size
    BLOCK_SIZE = 64
    
    # Launch kernel with one program per row
    grid = (N,)
    
    syrk_kernel[grid](
        A, C, alpha, beta, M, N, BLOCK_SIZE
    )