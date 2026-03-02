import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_1d_kernel(A_ptr, B_ptr, N, TSTEPS, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for t in range(TSTEPS):
        # First loop: B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1])
        block_start = tl.program_id(0) * BLOCK_SIZE + 1
        current_offsets = block_start + offsets
        mask = (current_offsets >= 1) & (current_offsets < N - 1)
        
        # Load A[i-1], A[i], A[i+1]
        a_prev = tl.load(A_ptr + current_offsets - 1, mask=mask)
        a_curr = tl.load(A_ptr + current_offsets, mask=mask)
        a_next = tl.load(A_ptr + current_offsets + 1, mask=mask)
        
        # Compute B[i]
        b_new = 0.33333 * (a_prev + a_curr + a_next)
        
        # Store B[i]
        tl.store(B_ptr + current_offsets, b_new, mask=mask)
        
        # Synchronization barrier
        tl.debug_barrier()
        
        # Second loop: A[i] = 0.33333 * (B[i-1] + B[i] + B[i+1])
        # Load B[i-1], B[i], B[i+1]
        b_prev = tl.load(B_ptr + current_offsets - 1, mask=mask)
        b_curr = tl.load(B_ptr + current_offsets, mask=mask)
        b_next = tl.load(B_ptr + current_offsets + 1, mask=mask)
        
        # Compute A[i]
        a_new = 0.33333 * (b_prev + b_curr + b_next)
        
        # Store A[i]
        tl.store(A_ptr + current_offsets, a_new, mask=mask)
        
        # Synchronization barrier
        tl.debug_barrier()

def jacobi_1d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 128
    grid_size = triton.cdiv(N - 2, BLOCK_SIZE)
    
    jacobi_1d_kernel[(grid_size,)](
        A, B, N, TSTEPS,
        BLOCK_SIZE=BLOCK_SIZE
    )