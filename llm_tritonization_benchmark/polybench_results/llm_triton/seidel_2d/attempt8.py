import triton
import triton.language as tl

@triton.jit
def seidel_2d_kernel(A_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    i_base = (pid * BLOCK_SIZE) + 1
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = i_base + i_offsets
    
    i_mask = (i_indices >= 1) & (i_indices < (N - 1))
    
    for j in range(1, N - 1):
        # Calculate linear indices for the 3x3 stencil
        idx_center = i_indices * N + j
        idx_top_left = (i_indices - 1) * N + (j - 1)
        idx_top_center = (i_indices - 1) * N + j
        idx_top_right = (i_indices - 1) * N + (j + 1)
        idx_mid_left = i_indices * N + (j - 1)
        idx_mid_right = i_indices * N + (j + 1)
        idx_bottom_left = (i_indices + 1) * N + (j - 1)
        idx_bottom_center = (i_indices + 1) * N + j
        idx_bottom_right = (i_indices + 1) * N + (j + 1)
        
        # Load all 9 values
        val_center = tl.load(A_ptr + idx_center, mask=i_mask, other=0.0)
        val_top_left = tl.load(A_ptr + idx_top_left, mask=i_mask, other=0.0)
        val_top_center = tl.load(A_ptr + idx_top_center, mask=i_mask, other=0.0)
        val_top_right = tl.load(A_ptr + idx_top_right, mask=i_mask, other=0.0)
        val_mid_left = tl.load(A_ptr + idx_mid_left, mask=i_mask, other=0.0)
        val_mid_right = tl.load(A_ptr + idx_mid_right, mask=i_mask, other=0.0)
        val_bottom_left = tl.load(A_ptr + idx_bottom_left, mask=i_mask, other=0.0)
        val_bottom_center = tl.load(A_ptr + idx_bottom_center, mask=i_mask, other=0.0)
        val_bottom_right = tl.load(A_ptr + idx_bottom_right, mask=i_mask, other=0.0)
        
        # Compute average of 9 values
        result = (val_top_left + val_top_center + val_top_right +
                 val_mid_left + val_center + val_mid_right +
                 val_bottom_left + val_bottom_center + val_bottom_right) / 9.0
        
        # Store result
        tl.store(A_ptr + idx_center, result, mask=i_mask)

def seidel_2d_triton(A, N, TSTEPS):
    BLOCK_SIZE = 32
    
    for t in range(TSTEPS):
        grid_size = triton.cdiv(N - 2, BLOCK_SIZE)
        grid = (grid_size,)
        
        seidel_2d_kernel[grid](A, N, BLOCK_SIZE)