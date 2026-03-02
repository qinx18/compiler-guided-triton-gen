import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_2d_kernel(A_ptr, B_ptr, N, TSTEPS, BLOCK_SIZE: tl.constexpr):
    # Get program ID for 2D grid
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Calculate block starts
    block_start_i = pid_i * BLOCK_SIZE
    block_start_j = pid_j * BLOCK_SIZE
    
    # Create offset arrays once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Time stepping loop
    for t in range(TSTEPS):
        # Calculate current indices
        i_indices = block_start_i + offsets
        j_indices = block_start_j + offsets
        
        # Create masks for valid interior points
        mask_i = (i_indices >= 1) & (i_indices < N - 1)
        mask_j = (j_indices >= 1) & (j_indices < N - 1)
        
        # Expand for 2D indexing
        i_expanded = i_indices[:, None]
        j_expanded = j_indices[None, :]
        mask_2d = mask_i[:, None] & mask_j[None, :]
        
        # Calculate memory indices for all combinations
        center_idx = i_expanded * N + j_expanded
        left_idx = i_expanded * N + (j_expanded - 1)
        right_idx = i_expanded * N + (j_expanded + 1)
        up_idx = (i_expanded - 1) * N + j_expanded
        down_idx = (i_expanded + 1) * N + j_expanded
        
        # First update: A -> B
        a_center = tl.load(A_ptr + center_idx, mask=mask_2d, other=0.0)
        a_left = tl.load(A_ptr + left_idx, mask=mask_2d, other=0.0)
        a_right = tl.load(A_ptr + right_idx, mask=mask_2d, other=0.0)
        a_up = tl.load(A_ptr + up_idx, mask=mask_2d, other=0.0)
        a_down = tl.load(A_ptr + down_idx, mask=mask_2d, other=0.0)
        
        b_new = 0.2 * (a_center + a_left + a_right + a_up + a_down)
        
        tl.store(B_ptr + center_idx, b_new, mask=mask_2d)
        
        # Second update: B -> A
        b_center = tl.load(B_ptr + center_idx, mask=mask_2d, other=0.0)
        b_left = tl.load(B_ptr + left_idx, mask=mask_2d, other=0.0)
        b_right = tl.load(B_ptr + right_idx, mask=mask_2d, other=0.0)
        b_up = tl.load(B_ptr + up_idx, mask=mask_2d, other=0.0)
        b_down = tl.load(B_ptr + down_idx, mask=mask_2d, other=0.0)
        
        a_new = 0.2 * (b_center + b_left + b_right + b_up + b_down)
        
        tl.store(A_ptr + center_idx, a_new, mask=mask_2d)

def jacobi_2d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 16
    
    # Calculate grid size for interior points
    grid_size = triton.cdiv(N - 2, BLOCK_SIZE)
    
    # Launch kernel with 2D grid
    jacobi_2d_kernel[(grid_size, grid_size)](
        A, B, N, TSTEPS, BLOCK_SIZE
    )