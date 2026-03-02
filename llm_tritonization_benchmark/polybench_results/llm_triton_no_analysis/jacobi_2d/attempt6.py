import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_2d_kernel(A_ptr, B_ptr, N, TSTEPS, BLOCK_SIZE: tl.constexpr):
    # Get program ID for 2D grid
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Calculate starting indices
    i_start = pid_i * BLOCK_SIZE + 1
    j_start = pid_j * BLOCK_SIZE + 1
    
    # Create offset arrays once
    i_offsets = tl.arange(0, BLOCK_SIZE)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Time stepping loop
    for t in range(TSTEPS):
        # First update: A -> B
        current_i_offsets = i_start + i_offsets
        current_j_offsets = j_start + j_offsets
        
        # Create masks for valid indices (1 <= i < N-1, 1 <= j < N-1)
        i_mask = (current_i_offsets >= 1) & (current_i_offsets < N - 1)
        j_mask = (current_j_offsets >= 1) & (current_j_offsets < N - 1)
        
        # Expand to 2D masks
        i_mask_2d = i_mask[:, None]
        j_mask_2d = j_mask[None, :]
        mask_2d = i_mask_2d & j_mask_2d
        
        # Calculate 2D indices for each point
        i_indices = current_i_offsets[:, None]
        j_indices = current_j_offsets[None, :]
        
        # Linear indices for accessing memory
        center_idx = i_indices * N + j_indices
        left_idx = i_indices * N + (j_indices - 1)
        right_idx = i_indices * N + (j_indices + 1)
        up_idx = (i_indices - 1) * N + j_indices
        down_idx = (i_indices + 1) * N + j_indices
        
        # Load values from A
        a_center = tl.load(A_ptr + center_idx, mask=mask_2d, other=0.0)
        a_left = tl.load(A_ptr + left_idx, mask=mask_2d, other=0.0)
        a_right = tl.load(A_ptr + right_idx, mask=mask_2d, other=0.0)
        a_up = tl.load(A_ptr + up_idx, mask=mask_2d, other=0.0)
        a_down = tl.load(A_ptr + down_idx, mask=mask_2d, other=0.0)
        
        # Compute B values
        b_new = 0.2 * (a_center + a_left + a_right + a_up + a_down)
        
        # Store to B
        tl.store(B_ptr + center_idx, b_new, mask=mask_2d)
        
        # Second update: B -> A
        # Load values from B
        b_center = tl.load(B_ptr + center_idx, mask=mask_2d, other=0.0)
        b_left = tl.load(B_ptr + left_idx, mask=mask_2d, other=0.0)
        b_right = tl.load(B_ptr + right_idx, mask=mask_2d, other=0.0)
        b_up = tl.load(B_ptr + up_idx, mask=mask_2d, other=0.0)
        b_down = tl.load(B_ptr + down_idx, mask=mask_2d, other=0.0)
        
        # Compute A values
        a_new = 0.2 * (b_center + b_left + b_right + b_up + b_down)
        
        # Store to A
        tl.store(A_ptr + center_idx, a_new, mask=mask_2d)

def jacobi_2d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 32
    
    # Calculate grid size - only need to cover interior points (1 to N-2)
    interior_size = N - 2
    grid_size = triton.cdiv(interior_size, BLOCK_SIZE)
    
    # Launch kernel
    jacobi_2d_kernel[(grid_size, grid_size)](
        A, B, N, TSTEPS, BLOCK_SIZE
    )