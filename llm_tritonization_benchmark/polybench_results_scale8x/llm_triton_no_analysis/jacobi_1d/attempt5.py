import torch
import triton
import triton.language as tl

@triton.jit
def jacobi_1d_kernel(A_ptr, B_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get the starting position for this block
    block_start = tl.program_id(0) * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid indices (1 <= idx < N-1)
    mask = (offsets >= 1) & (offsets < N - 1)
    
    for t in range(TSTEPS):
        # Synchronize across blocks before each step
        tl.debug_barrier()
        
        # First loop: B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1])
        a_prev = tl.load(A_ptr + offsets - 1, mask=mask, other=0.0)
        a_curr = tl.load(A_ptr + offsets, mask=mask, other=0.0)
        a_next = tl.load(A_ptr + offsets + 1, mask=mask, other=0.0)
        b_val = 0.33333 * (a_prev + a_curr + a_next)
        tl.store(B_ptr + offsets, b_val, mask=mask)
        
        # Synchronize across blocks before second step
        tl.debug_barrier()
        
        # Second loop: A[i] = 0.33333 * (B[i-1] + B[i] + B[i+1])
        b_prev = tl.load(B_ptr + offsets - 1, mask=mask, other=0.0)
        b_curr = tl.load(B_ptr + offsets, mask=mask, other=0.0)
        b_next = tl.load(B_ptr + offsets + 1, mask=mask, other=0.0)
        a_val = 0.33333 * (b_prev + b_curr + b_next)
        tl.store(A_ptr + offsets, a_val, mask=mask)

def jacobi_1d_triton(A, B, N, TSTEPS):
    for t in range(TSTEPS):
        # First step: B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1])
        BLOCK_SIZE = 256
        grid_size = triton.cdiv(N, BLOCK_SIZE)
        
        @triton.jit
        def step1_kernel(A_ptr, B_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            block_start = tl.program_id(0) * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = (offsets >= 1) & (offsets < N - 1)
            
            a_prev = tl.load(A_ptr + offsets - 1, mask=mask, other=0.0)
            a_curr = tl.load(A_ptr + offsets, mask=mask, other=0.0)
            a_next = tl.load(A_ptr + offsets + 1, mask=mask, other=0.0)
            b_val = 0.33333 * (a_prev + a_curr + a_next)
            tl.store(B_ptr + offsets, b_val, mask=mask)
        
        step1_kernel[(grid_size,)](A, B, N, BLOCK_SIZE)
        
        # Second step: A[i] = 0.33333 * (B[i-1] + B[i] + B[i+1])
        @triton.jit
        def step2_kernel(A_ptr, B_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
            block_start = tl.program_id(0) * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = (offsets >= 1) & (offsets < N - 1)
            
            b_prev = tl.load(B_ptr + offsets - 1, mask=mask, other=0.0)
            b_curr = tl.load(B_ptr + offsets, mask=mask, other=0.0)
            b_next = tl.load(B_ptr + offsets + 1, mask=mask, other=0.0)
            a_val = 0.33333 * (b_prev + b_curr + b_next)
            tl.store(A_ptr + offsets, a_val, mask=mask)
        
        step2_kernel[(grid_size,)](A, B, N, BLOCK_SIZE)