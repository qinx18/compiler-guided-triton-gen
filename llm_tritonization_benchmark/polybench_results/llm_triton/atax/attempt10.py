import triton
import triton.language as tl

@triton.jit
def atax_kernel(A, tmp, x, y, M, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID for j dimension
    pid = tl.program_id(axis=0)
    
    # Calculate offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    j_offsets = pid * BLOCK_SIZE + offsets
    j_mask = j_offsets < N
    
    # Initialize y[j] = 0
    y_ptrs = y + j_offsets
    tl.store(y_ptrs, 0.0, mask=j_mask)
    
    # Sequential loop over i
    for i in range(M):
        # Initialize tmp[i] = 0.0 (only first program does this)
        if pid == 0:
            tl.store(tmp + i, 0.0)
        
        # Synchronize to ensure tmp[i] is initialized
        tl.debug_barrier()
        
        # Compute tmp[i] += A[i][j] * x[j] for this block of j
        if pid * BLOCK_SIZE < N:
            A_ptrs = A + i * N + j_offsets
            A_vals = tl.load(A_ptrs, mask=j_mask, other=0.0)
            x_ptrs = x + j_offsets
            x_vals = tl.load(x_ptrs, mask=j_mask, other=0.0)
            
            partial_sum = tl.sum(A_vals * x_vals)
            tl.atomic_add(tmp + i, partial_sum)
        
        # Synchronize to ensure tmp[i] is computed
        tl.debug_barrier()
        
        # Load tmp[i] and compute y[j] += A[i][j] * tmp[i]
        if pid * BLOCK_SIZE < N:
            tmp_val = tl.load(tmp + i)
            A_ptrs = A + i * N + j_offsets
            A_vals = tl.load(A_ptrs, mask=j_mask, other=0.0)
            
            y_updates = A_vals * tmp_val
            y_ptrs = y + j_offsets
            tl.atomic_add(y_ptrs, y_updates, mask=j_mask)

def atax_triton(A, tmp, x, y, M, N):
    BLOCK_SIZE = 64
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    atax_kernel[grid](A, tmp, x, y, M, N, BLOCK_SIZE)