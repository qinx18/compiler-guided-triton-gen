import triton
import triton.language as tl
import torch

@triton.jit
def gemm_kernel(A, B, C, alpha, beta, NI: tl.constexpr, NJ: tl.constexpr, NK: tl.constexpr, BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr):
    # Get program IDs
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Calculate offsets
    i_offsets = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    j_offsets = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
    
    # Masks
    i_mask = i_offsets < NI
    j_mask = j_offsets < NJ
    
    # Calculate C indices
    c_indices = i_offsets[:, None] * NJ + j_offsets[None, :]
    c_mask = i_mask[:, None] & j_mask[None, :]
    
    # Load and multiply C by beta
    c_vals = tl.load(C + c_indices, mask=c_mask, other=0.0)
    c_vals = c_vals * beta
    
    # Accumulate A * B
    for k in range(NK):
        # Load A[i, k]
        a_indices = i_offsets * NK + k
        a_vals = tl.load(A + a_indices, mask=i_mask, other=0.0)
        
        # Load B[k, j]
        b_indices = k * NJ + j_offsets
        b_vals = tl.load(B + b_indices, mask=j_mask, other=0.0)
        
        # Compute alpha * A[i, k] * B[k, j] and accumulate
        ab_vals = alpha * a_vals[:, None] * b_vals[None, :]
        c_vals = c_vals + ab_vals
    
    # Store result
    tl.store(C + c_indices, c_vals, mask=c_mask)

def gemm_triton(A, B, C, alpha, beta, NI, NJ, NK):
    BLOCK_I = 16
    BLOCK_J = 16
    
    grid_i = triton.cdiv(NI, BLOCK_I)
    grid_j = triton.cdiv(NJ, BLOCK_J)
    
    gemm_kernel[(grid_i, grid_j)](
        A, B, C, alpha, beta, NI, NJ, NK, BLOCK_I, BLOCK_J
    )