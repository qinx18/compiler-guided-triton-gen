import triton
import triton.language as tl
import torch

@triton.jit
def bicg_kernel(A_ptr, p_ptr, q_ptr, r_ptr, s_ptr, M, N, A_stride_0, A_stride_1):
    # Initialize s[i] = 0 for all i in [0, M)
    s_offsets = tl.arange(0, 128)
    s_mask = s_offsets < M
    tl.store(s_ptr + s_offsets, 0.0, mask=s_mask)
    
    # Main computation loop over i (rows of A)
    for i in range(N):
        # Load r[i]
        r_i = tl.load(r_ptr + i)
        
        # Initialize q[i] = 0
        q_i = 0.0
        
        # Process columns in blocks
        j_offsets = tl.arange(0, 128)
        for j_start in range(0, M, 128):
            j_current = j_start + j_offsets
            j_mask = j_current < M
            
            # Load A[i][j] values
            A_ptrs = A_ptr + i * A_stride_0 + j_current * A_stride_1
            A_vals = tl.load(A_ptrs, mask=j_mask)
            
            # Load p[j] values
            p_vals = tl.load(p_ptr + j_current, mask=j_mask)
            
            # Load current s[j] values
            s_vals = tl.load(s_ptr + j_current, mask=j_mask)
            
            # Update s[j] = s[j] + r[i] * A[i][j]
            s_new = s_vals + r_i * A_vals
            tl.store(s_ptr + j_current, s_new, mask=j_mask)
            
            # Accumulate q[i] += A[i][j] * p[j]
            q_contrib = A_vals * p_vals
            q_i += tl.sum(tl.where(j_mask, q_contrib, 0.0))
        
        # Store final q[i]
        tl.store(q_ptr + i, q_i)

def bicg_triton(A, p, q, r, s, M, N):
    # Launch kernel with single program
    grid = (1,)
    bicg_kernel[grid](
        A, p, q, r, s, M, N,
        A.stride(0), A.stride(1)
    )