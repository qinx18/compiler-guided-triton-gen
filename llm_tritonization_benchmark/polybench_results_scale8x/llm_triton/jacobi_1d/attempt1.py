import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_1d_kernel(A_ptr, B_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Single CTA handles all elements
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    # Pre-compute offsets once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for t in range(TSTEPS):
        # Phase 1: B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1])
        for block_start in range(1, N - 1, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = (current_offsets >= 1) & (current_offsets < N - 1)
            
            # Load A[i-1], A[i], A[i+1]
            a_left = tl.load(A_ptr + current_offsets - 1, mask=mask)
            a_center = tl.load(A_ptr + current_offsets, mask=mask)
            a_right = tl.load(A_ptr + current_offsets + 1, mask=mask)
            
            # Compute B[i]
            b_val = 0.33333 * (a_left + a_center + a_right)
            
            # Store B[i]
            tl.store(B_ptr + current_offsets, b_val, mask=mask)
        
        # Phase 2: A[i] = 0.33333 * (B[i-1] + B[i] + B[i+1])
        for block_start in range(1, N - 1, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = (current_offsets >= 1) & (current_offsets < N - 1)
            
            # Load B[i-1], B[i], B[i+1]
            b_left = tl.load(B_ptr + current_offsets - 1, mask=mask)
            b_center = tl.load(B_ptr + current_offsets, mask=mask)
            b_right = tl.load(B_ptr + current_offsets + 1, mask=mask)
            
            # Compute A[i]
            a_val = 0.33333 * (b_left + b_center + b_right)
            
            # Store A[i]
            tl.store(A_ptr + current_offsets, a_val, mask=mask)

def jacobi_1d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (1,)
    
    jacobi_1d_kernel[grid](
        A, B,
        N=N, TSTEPS=TSTEPS, BLOCK_SIZE=BLOCK_SIZE
    )