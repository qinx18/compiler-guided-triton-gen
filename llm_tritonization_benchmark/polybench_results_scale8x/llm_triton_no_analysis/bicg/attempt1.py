import triton
import triton.language as tl
import torch

@triton.jit
def bicg_kernel(A_ptr, p_ptr, q_ptr, r_ptr, s_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Initialize s[i] = 0 for all i < M
    s_offsets = tl.arange(0, BLOCK_SIZE)
    for s_block_start in range(0, M, BLOCK_SIZE):
        s_current_offsets = s_block_start + s_offsets
        s_mask = s_current_offsets < M
        tl.store(s_ptr + s_current_offsets, 0.0, mask=s_mask)
    
    # Main computation loops
    for i in range(N):
        # Initialize q[i] = 0.0
        q_i = 0.0
        
        # Get r[i]
        r_i = tl.load(r_ptr + i)
        
        # Inner loop over j (columns of A)
        j_offsets = tl.arange(0, BLOCK_SIZE)
        for j_block_start in range(0, M, BLOCK_SIZE):
            j_current_offsets = j_block_start + j_offsets
            j_mask = j_current_offsets < M
            
            # Load A[i][j] values for this block
            A_linear_idx = i * M + j_current_offsets
            A_vals = tl.load(A_ptr + A_linear_idx, mask=j_mask, other=0.0)
            
            # Load p[j] values for this block
            p_vals = tl.load(p_ptr + j_current_offsets, mask=j_mask, other=0.0)
            
            # Load current s[j] values
            s_vals = tl.load(s_ptr + j_current_offsets, mask=j_mask, other=0.0)
            
            # Update s[j] = s[j] + r[i] * A[i][j]
            s_new_vals = s_vals + r_i * A_vals
            tl.store(s_ptr + j_current_offsets, s_new_vals, mask=j_mask)
            
            # Accumulate q[i] = q[i] + A[i][j] * p[j]
            q_i += tl.sum(A_vals * p_vals, axis=0)
        
        # Store final q[i]
        tl.store(q_ptr + i, q_i)

def bicg_triton(A, p, q, r, s, M, N):
    BLOCK_SIZE = 64
    
    # Launch kernel with single program
    bicg_kernel[(1,)](
        A, p, q, r, s,
        M, N, BLOCK_SIZE
    )