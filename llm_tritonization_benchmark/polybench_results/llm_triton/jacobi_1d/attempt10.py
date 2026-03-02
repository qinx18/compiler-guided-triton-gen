import triton
import triton.language as tl

@triton.jit
def jacobi_1d_kernel(A_ptr, B_ptr, N, t_idx, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_start = 1 + pid * BLOCK_SIZE
    i_offsets = i_start + tl.arange(0, BLOCK_SIZE)
    i_mask = (i_offsets >= 1) & (i_offsets < N - 1)
    
    # First loop: B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1])
    a_left = tl.load(A_ptr + i_offsets - 1, mask=i_mask, other=0.0)
    a_center = tl.load(A_ptr + i_offsets, mask=i_mask, other=0.0)
    a_right = tl.load(A_ptr + i_offsets + 1, mask=i_mask, other=0.0)
    b_vals = 0.33333 * (a_left + a_center + a_right)
    tl.store(B_ptr + i_offsets, b_vals, mask=i_mask)

@triton.jit
def jacobi_1d_kernel_second(A_ptr, B_ptr, N, t_idx, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_start = 1 + pid * BLOCK_SIZE
    i_offsets = i_start + tl.arange(0, BLOCK_SIZE)
    i_mask = (i_offsets >= 1) & (i_offsets < N - 1)
    
    # Second loop: A[i] = 0.33333 * (B[i-1] + B[i] + B[i+1])
    b_left = tl.load(B_ptr + i_offsets - 1, mask=i_mask, other=0.0)
    b_center = tl.load(B_ptr + i_offsets, mask=i_mask, other=0.0)
    b_right = tl.load(B_ptr + i_offsets + 1, mask=i_mask, other=0.0)
    a_vals = 0.33333 * (b_left + b_center + b_right)
    tl.store(A_ptr + i_offsets, a_vals, mask=i_mask)

def jacobi_1d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 32
    
    num_i_elements = N - 2
    grid = (triton.cdiv(num_i_elements, BLOCK_SIZE),)
    
    for t in range(TSTEPS):
        jacobi_1d_kernel[grid](
            A, B, N, t, 
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        jacobi_1d_kernel_second[grid](
            A, B, N, t, 
            BLOCK_SIZE=BLOCK_SIZE
        )