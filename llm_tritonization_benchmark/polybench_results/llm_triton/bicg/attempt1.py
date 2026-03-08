import triton
import triton.language as tl

@triton.jit
def bicg_kernel(A_ptr, p_ptr, q_ptr, r_ptr, s_ptr, 
                M: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    # Initialize offsets for vectorized column access
    col_offsets = tl.arange(0, BLOCK)
    col_mask = col_offsets < M
    
    # Initialize s array to zero
    s_acc = tl.zeros([BLOCK], dtype=tl.float32)
    
    # Main computation loop over rows (N dimension)
    for i in range(N):
        # Load current row of A
        a_row = tl.load(A_ptr + i * M + col_offsets, mask=col_mask)
        
        # Load r[i] (scalar for this row)
        r_i = tl.load(r_ptr + i)
        
        # Load p vector
        p_vec = tl.load(p_ptr + col_offsets, mask=col_mask)
        
        # Accumulate s[j] += r[i] * A[i][j]
        s_acc += r_i * a_row
        
        # Compute q[i] = sum(A[i][j] * p[j])
        q_val = tl.sum(a_row * p_vec)
        tl.store(q_ptr + i, q_val)
    
    # Store accumulated s values
    tl.store(s_ptr + col_offsets, s_acc, mask=col_mask)

def bicg_triton(A, p, q, r, s, M, N):
    # Determine block size
    BLOCK = triton.next_power_of_2(M)
    
    # Launch kernel with single thread block
    grid = (1,)
    
    bicg_kernel[grid](
        A, p, q, r, s,
        M=M, N=N, BLOCK=BLOCK
    )