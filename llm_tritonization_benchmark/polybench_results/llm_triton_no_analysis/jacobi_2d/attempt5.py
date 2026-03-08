import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_2d_kernel(A_ptr, B_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for 2D grid
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Calculate starting indices for this block
    start_i = 1 + pid_i * BLOCK_SIZE
    start_j = 1 + pid_j * BLOCK_SIZE
    
    # Create offset ranges once
    offset_i = tl.arange(0, BLOCK_SIZE)
    offset_j = tl.arange(0, BLOCK_SIZE)
    
    for t in range(TSTEPS):
        # First phase: Update B from A
        current_i = start_i + offset_i
        current_j = start_j + offset_j
        
        # Create masks for valid indices (1 <= i,j <= N-2)
        mask_i = (current_i >= 1) & (current_i <= N - 2)
        mask_j = (current_j >= 1) & (current_j <= N - 2)
        
        # Create 2D masks
        mask_i_2d = mask_i[:, None]
        mask_j_2d = mask_j[None, :]
        mask_2d = mask_i_2d & mask_j_2d
        
        # Calculate linear indices for all needed positions
        center_idx = current_i[:, None] * N + current_j[None, :]
        left_idx = current_i[:, None] * N + (current_j[None, :] - 1)
        right_idx = current_i[:, None] * N + (current_j[None, :] + 1)
        up_idx = (current_i[:, None] - 1) * N + current_j[None, :]
        down_idx = (current_i[:, None] + 1) * N + current_j[None, :]
        
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
        
        # Second phase: Update A from B
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
    # Block size for processing
    BLOCK_SIZE = 16
    
    # Calculate grid dimensions for interior points (1 to N-2)
    interior_size = N - 2
    grid_dim = triton.cdiv(interior_size, BLOCK_SIZE)
    
    # Launch kernel
    jacobi_2d_kernel[(grid_dim, grid_dim)](
        A, B, N, TSTEPS, BLOCK_SIZE
    )