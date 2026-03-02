import triton
import triton.language as tl

@triton.jit
def jacobi_1d_kernel(A_ptr, B_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + 1
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = (offsets >= 1) & (offsets < N - 1)
    
    # Load A values with offsets
    a_left = tl.load(A_ptr + offsets - 1, mask=mask, other=0.0)
    a_center = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    a_right = tl.load(A_ptr + offsets + 1, mask=mask, other=0.0)
    
    # Compute B values
    b_vals = 0.33333 * (a_left + a_center + a_right)
    
    # Store B values
    tl.store(B_ptr + offsets, b_vals, mask=mask)

@triton.jit
def jacobi_1d_kernel_ba(A_ptr, B_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE + 1
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = (offsets >= 1) & (offsets < N - 1)
    
    # Load B values with offsets
    b_left = tl.load(B_ptr + offsets - 1, mask=mask, other=0.0)
    b_center = tl.load(B_ptr + offsets, mask=mask, other=0.0)
    b_right = tl.load(B_ptr + offsets + 1, mask=mask, other=0.0)
    
    # Compute A values
    a_vals = 0.33333 * (b_left + b_center + b_right)
    
    # Store A values
    tl.store(A_ptr + offsets, a_vals, mask=mask)

def jacobi_1d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 64
    num_elements = N - 2
    grid = (triton.cdiv(num_elements, BLOCK_SIZE),)
    
    for t in range(TSTEPS):
        jacobi_1d_kernel[grid](A, B, N, BLOCK_SIZE=BLOCK_SIZE)
        jacobi_1d_kernel_ba[grid](A, B, N, BLOCK_SIZE=BLOCK_SIZE)