import triton
import triton.language as tl
import torch

@triton.jit
def bicg_kernel(A_ptr, p_ptr, q_ptr, r_ptr, s_ptr,
                M: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    
    # Initialize s to zero
    mask_s = offsets < M
    tl.store(s_ptr + offsets, tl.zeros([BLOCK], dtype=tl.float32), mask=mask_s)
    
    # Column accumulator for s
    col_acc = tl.zeros([BLOCK], dtype=tl.float32)
    
    for i in range(N):
        # Load row i of A
        mask_a = offsets < M
        a_row = tl.load(A_ptr + i * M + offsets, mask=mask_a)
        
        # Load r[i]
        r_i = tl.load(r_ptr + i)
        
        # Load p vector
        p_vals = tl.load(p_ptr + offsets, mask=mask_a)
        
        # Row reduction for q[i]
        row_sum = tl.sum(a_row * p_vals)
        tl.store(q_ptr + i, row_sum)
        
        # Column accumulation for s[j]
        col_acc += r_i * a_row
    
    # Store final s values
    tl.store(s_ptr + offsets, col_acc, mask=mask_s)

def bicg_triton(A, p, q, r, s, M, N):
    # Zero initialize outputs
    s.zero_()
    q.zero_()
    
    BLOCK_SIZE = triton.next_power_of_2(max(M, 128))
    
    grid = (1,)
    
    bicg_kernel[grid](
        A, p, q, r, s,
        M=M, N=N, BLOCK=BLOCK_SIZE
    )