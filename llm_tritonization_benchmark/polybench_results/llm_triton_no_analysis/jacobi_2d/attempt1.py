import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_2d_kernel(A_ptr, B_ptr, N, TSTEPS, BLOCK_SIZE: tl.constexpr):
    # Get block indices
    block_i = tl.program_id(0)
    block_j = tl.program_id(1)
    
    # Create offset arrays once
    offsets_i = tl.arange(0, BLOCK_SIZE)
    offsets_j = tl.arange(0, BLOCK_SIZE)
    
    # Compute actual indices for this block
    i_indices = block_i * BLOCK_SIZE + offsets_i
    j_indices = block_j * BLOCK_SIZE + offsets_j
    
    # Create 2D index grids
    i_grid = i_indices[:, None]
    j_grid = j_indices[None, :]
    
    # Masks for boundary conditions (exclude boundary elements)
    i_mask = (i_grid >= 1) & (i_grid < N - 1)
    j_mask = (j_grid >= 1) & (j_grid < N - 1)
    valid_mask = i_mask & j_mask
    
    for t in range(TSTEPS):
        # First phase: compute B from A
        # Load A values with stencil pattern
        center_idx = i_grid * N + j_grid
        left_idx = i_grid * N + (j_grid - 1)
        right_idx = i_grid * N + (j_grid + 1)
        up_idx = (i_grid - 1) * N + j_grid
        down_idx = (i_grid + 1) * N + j_grid
        
        # Load stencil values
        a_center = tl.load(A_ptr + center_idx, mask=valid_mask, other=0.0)
        a_left = tl.load(A_ptr + left_idx, mask=valid_mask, other=0.0)
        a_right = tl.load(A_ptr + right_idx, mask=valid_mask, other=0.0)
        a_up = tl.load(A_ptr + up_idx, mask=valid_mask, other=0.0)
        a_down = tl.load(A_ptr + down_idx, mask=valid_mask, other=0.0)
        
        # Compute B values
        b_val = 0.2 * (a_center + a_left + a_right + a_down + a_up)
        
        # Store B values
        tl.store(B_ptr + center_idx, b_val, mask=valid_mask)
        
        # Second phase: compute A from B
        # Load B values with stencil pattern
        b_center = tl.load(B_ptr + center_idx, mask=valid_mask, other=0.0)
        b_left = tl.load(B_ptr + left_idx, mask=valid_mask, other=0.0)
        b_right = tl.load(B_ptr + right_idx, mask=valid_mask, other=0.0)
        b_up = tl.load(B_ptr + up_idx, mask=valid_mask, other=0.0)
        b_down = tl.load(B_ptr + down_idx, mask=valid_mask, other=0.0)
        
        # Compute A values
        a_val = 0.2 * (b_center + b_left + b_right + b_down + b_up)
        
        # Store A values
        tl.store(A_ptr + center_idx, a_val, mask=valid_mask)

def jacobi_2d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 16
    
    # Calculate grid dimensions
    grid_i = triton.cdiv(N, BLOCK_SIZE)
    grid_j = triton.cdiv(N, BLOCK_SIZE)
    
    # Launch kernel
    jacobi_2d_kernel[(grid_i, grid_j)](
        A, B, N, TSTEPS, BLOCK_SIZE
    )