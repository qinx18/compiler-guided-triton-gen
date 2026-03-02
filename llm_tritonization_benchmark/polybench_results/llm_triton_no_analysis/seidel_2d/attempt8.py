import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A_ptr, N, TSTEPS, BLOCK_SIZE: tl.constexpr):
    # Get 2D block coordinates
    pid_i = tl.program_id(axis=0)
    pid_j = tl.program_id(axis=1)
    
    # Calculate starting indices for this block
    start_i = pid_i * BLOCK_SIZE + 1
    start_j = pid_j * BLOCK_SIZE + 1
    
    # Create offsets within the block
    offsets_i = tl.arange(0, BLOCK_SIZE)
    offsets_j = tl.arange(0, BLOCK_SIZE)
    
    # Time stepping loop
    for t in range(TSTEPS):
        # Create 2D grid of indices
        i_coords = start_i + offsets_i[:, None]
        j_coords = start_j + offsets_j[None, :]
        
        # Create mask for valid indices
        mask = (i_coords <= N - 2) & (j_coords <= N - 2)
        
        # Calculate linear indices for the 9-point stencil
        idx_center = i_coords * N + j_coords
        
        # Load all 9 stencil points
        val_top_left = tl.load(A_ptr + (i_coords - 1) * N + (j_coords - 1), mask=mask)
        val_top = tl.load(A_ptr + (i_coords - 1) * N + j_coords, mask=mask)
        val_top_right = tl.load(A_ptr + (i_coords - 1) * N + (j_coords + 1), mask=mask)
        
        val_left = tl.load(A_ptr + i_coords * N + (j_coords - 1), mask=mask)
        val_center = tl.load(A_ptr + idx_center, mask=mask)
        val_right = tl.load(A_ptr + i_coords * N + (j_coords + 1), mask=mask)
        
        val_bottom_left = tl.load(A_ptr + (i_coords + 1) * N + (j_coords - 1), mask=mask)
        val_bottom = tl.load(A_ptr + (i_coords + 1) * N + j_coords, mask=mask)
        val_bottom_right = tl.load(A_ptr + (i_coords + 1) * N + (j_coords + 1), mask=mask)
        
        # Compute the average
        new_val = (val_top_left + val_top + val_top_right +
                   val_left + val_center + val_right +
                   val_bottom_left + val_bottom + val_bottom_right) / 9.0
        
        # Store the result
        tl.store(A_ptr + idx_center, new_val, mask=mask)

def seidel_2d_triton(A, N, TSTEPS):
    BLOCK_SIZE = 16
    
    # Calculate grid size for interior points [1, N-2]
    interior_size = N - 2
    grid_i = triton.cdiv(interior_size, BLOCK_SIZE)
    grid_j = triton.cdiv(interior_size, BLOCK_SIZE)
    
    # Launch kernel with 2D grid
    seidel_2d_kernel[(grid_i, grid_j)](
        A, N, TSTEPS, BLOCK_SIZE
    )