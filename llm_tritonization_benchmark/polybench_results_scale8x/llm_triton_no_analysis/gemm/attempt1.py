import triton
import triton.language as tl
import torch

@triton.jit
def gemm_kernel(A_ptr, B_ptr, C_ptr, alpha, beta, 
                NI: tl.constexpr, NJ: tl.constexpr, NK: tl.constexpr,
                BLOCK_SIZE: tl.constexpr):
    
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate row index
    i = pid
    
    if i >= NI:
        return
    
    # Pre-compute offsets for j dimension
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    # First, multiply C[i, :] by beta
    for j_start in range(0, NJ, BLOCK_SIZE):
        j_indices = j_start + j_offsets
        j_mask = j_indices < NJ
        
        # Load C[i, j_start:j_start+BLOCK_SIZE]
        c_offset = i * NJ + j_indices
        c_vals = tl.load(C_ptr + c_offset, mask=j_mask, other=0.0)
        
        # Multiply by beta
        c_vals = c_vals * beta
        
        # Store back to C
        tl.store(C_ptr + c_offset, c_vals, mask=j_mask)
    
    # Then, compute C[i, j] += alpha * A[i, k] * B[k, j]
    for k in range(NK):
        # Load A[i, k] (scalar for this i, k)
        a_val = tl.load(A_ptr + i * NK + k)
        
        for j_start in range(0, NJ, BLOCK_SIZE):
            j_indices = j_start + j_offsets
            j_mask = j_indices < NJ
            
            # Load B[k, j_start:j_start+BLOCK_SIZE]
            b_offset = k * NJ + j_indices
            b_vals = tl.load(B_ptr + b_offset, mask=j_mask, other=0.0)
            
            # Load current C[i, j_start:j_start+BLOCK_SIZE]
            c_offset = i * NJ + j_indices
            c_vals = tl.load(C_ptr + c_offset, mask=j_mask, other=0.0)
            
            # Compute update
            update = alpha * a_val * b_vals
            c_vals = c_vals + update
            
            # Store back to C
            tl.store(C_ptr + c_offset, c_vals, mask=j_mask)

def gemm_triton(A, B, C, alpha, beta, NI, NJ, NK):
    # Block size for j dimension
    BLOCK_SIZE = 64
    
    # Number of programs = number of rows (NI)
    grid = (NI,)
    
    gemm_kernel[grid](
        A, B, C, alpha, beta,
        NI, NJ, NK, BLOCK_SIZE
    )