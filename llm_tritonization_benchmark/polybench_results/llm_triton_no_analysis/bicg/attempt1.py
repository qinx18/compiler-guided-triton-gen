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
    
    # Main nested loops
    for i in range(N):
        # q[i] = 0.0
        tl.store(q_ptr + i, 0.0)
        
        # Load r[i] once for this iteration
        r_i = tl.load(r_ptr + i)
        
        # Process matrix row A[i] in blocks
        j_offsets = tl.arange(0, BLOCK_SIZE)
        q_i_acc = 0.0
        
        for j_block_start in range(0, M, BLOCK_SIZE):
            j_current_offsets = j_block_start + j_offsets
            j_mask = j_current_offsets < M
            
            # Load A[i][j] values for current block
            A_indices = i * M + j_current_offsets
            A_vals = tl.load(A_ptr + A_indices, mask=j_mask, other=0.0)
            
            # Load p[j] values for current block
            p_vals = tl.load(p_ptr + j_current_offsets, mask=j_mask, other=0.0)
            
            # Load current s[j] values
            s_vals = tl.load(s_ptr + j_current_offsets, mask=j_mask, other=0.0)
            
            # Update s[j] = s[j] + r[i] * A[i][j]
            s_new = s_vals + r_i * A_vals
            tl.store(s_ptr + j_current_offsets, s_new, mask=j_mask)
            
            # Accumulate q[i] += A[i][j] * p[j]
            q_i_acc += tl.sum(A_vals * p_vals, axis=0)
        
        # Store final q[i] value
        q_i_current = tl.load(q_ptr + i)
        tl.store(q_ptr + i, q_i_current + q_i_acc)

def bicg_triton(A, p, q, r, s, M, N):
    BLOCK_SIZE = 64
    
    # Launch kernel with single program instance
    bicg_kernel[(1,)](
        A, p, q, r, s, 
        M=M, N=N, BLOCK_SIZE=BLOCK_SIZE
    )