import triton
import triton.language as tl

@triton.jit
def jacobi_1d_kernel(A_ptr, B_ptr, N, T_START, T_END, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_T: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_t = tl.program_id(1)
    
    # Each block handles different time steps
    t_start = T_START + pid_t * BLOCK_SIZE_T
    t_end = tl.minimum(t_start + BLOCK_SIZE_T, T_END)
    
    # Each block handles different spatial indices
    i_start = 1 + pid_i * BLOCK_SIZE_I
    i_offsets = i_start + tl.arange(0, BLOCK_SIZE_I)
    i_mask = (i_offsets >= 1) & (i_offsets < N - 1)
    
    # Process time steps sequentially within this block
    for t in range(t_start, t_end):
        # First loop: B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1])
        a_left = tl.load(A_ptr + i_offsets - 1, mask=i_mask, other=0.0)
        a_center = tl.load(A_ptr + i_offsets, mask=i_mask, other=0.0)
        a_right = tl.load(A_ptr + i_offsets + 1, mask=i_mask, other=0.0)
        b_vals = 0.33333 * (a_left + a_center + a_right)
        tl.store(B_ptr + i_offsets, b_vals, mask=i_mask)
        
        # Second loop: A[i] = 0.33333 * (B[i-1] + B[i] + B[i+1])
        b_left = tl.load(B_ptr + i_offsets - 1, mask=i_mask, other=0.0)
        b_center = tl.load(B_ptr + i_offsets, mask=i_mask, other=0.0)
        b_right = tl.load(B_ptr + i_offsets + 1, mask=i_mask, other=0.0)
        a_vals = 0.33333 * (b_left + b_center + b_right)
        tl.store(A_ptr + i_offsets, a_vals, mask=i_mask)

def jacobi_1d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE_I = 64
    BLOCK_SIZE_T = 8
    
    num_i_elements = N - 2
    grid_i = triton.cdiv(num_i_elements, BLOCK_SIZE_I)
    grid_t = triton.cdiv(TSTEPS, BLOCK_SIZE_T)
    
    jacobi_1d_kernel[(grid_i, grid_t)](
        A, B, N, 0, TSTEPS, 
        BLOCK_SIZE_I=BLOCK_SIZE_I, 
        BLOCK_SIZE_T=BLOCK_SIZE_T
    )