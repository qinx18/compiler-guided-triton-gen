import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_2d_kernel(A_ptr, B_ptr, N, TSTEPS, BLOCK_SIZE: tl.constexpr):
    # Get program ID for 1D grid over all interior points
    pid = tl.program_id(0)
    
    # Calculate block start
    block_start = pid * BLOCK_SIZE
    
    # Create offset array once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate total interior points
    interior_points = (N - 2) * (N - 2)
    
    # Time stepping loop
    for t in range(TSTEPS):
        # Calculate current linear indices
        linear_idx = block_start + offsets
        
        # Convert linear index to 2D coordinates (interior grid)
        interior_i = linear_idx // (N - 2)
        interior_j = linear_idx % (N - 2)
        
        # Convert to actual array indices (add 1 for border)
        i = interior_i + 1
        j = interior_j + 1
        
        # Create mask for valid indices
        mask = linear_idx < interior_points
        
        # Calculate memory indices
        center_idx = i * N + j
        left_idx = i * N + (j - 1)
        right_idx = i * N + (j + 1)
        up_idx = (i - 1) * N + j
        down_idx = (i + 1) * N + j
        
        # First update: A -> B
        a_center = tl.load(A_ptr + center_idx, mask=mask, other=0.0)
        a_left = tl.load(A_ptr + left_idx, mask=mask, other=0.0)
        a_right = tl.load(A_ptr + right_idx, mask=mask, other=0.0)
        a_up = tl.load(A_ptr + up_idx, mask=mask, other=0.0)
        a_down = tl.load(A_ptr + down_idx, mask=mask, other=0.0)
        
        b_new = 0.2 * (a_center + a_left + a_right + a_up + a_down)
        
        tl.store(B_ptr + center_idx, b_new, mask=mask)
        
        # Synchronization barrier
        tl.debug_barrier()
        
        # Second update: B -> A
        b_center = tl.load(B_ptr + center_idx, mask=mask, other=0.0)
        b_left = tl.load(B_ptr + left_idx, mask=mask, other=0.0)
        b_right = tl.load(B_ptr + right_idx, mask=mask, other=0.0)
        b_up = tl.load(B_ptr + up_idx, mask=mask, other=0.0)
        b_down = tl.load(B_ptr + down_idx, mask=mask, other=0.0)
        
        a_new = 0.2 * (b_center + b_left + b_right + b_up + b_down)
        
        tl.store(A_ptr + center_idx, a_new, mask=mask)
        
        # Synchronization barrier
        tl.debug_barrier()

def jacobi_2d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 256
    
    # Calculate total interior points
    interior_points = (N - 2) * (N - 2)
    
    # Calculate grid size
    grid_size = triton.cdiv(interior_points, BLOCK_SIZE)
    
    # Launch kernel
    jacobi_2d_kernel[(grid_size,)](
        A, B, N, TSTEPS, BLOCK_SIZE
    )