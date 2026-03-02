import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A_ptr, N, TSTEPS, BLOCK_SIZE: tl.constexpr):
    # Get program ID for this block
    pid = tl.program_id(axis=0)
    
    # Calculate how many elements we process per block
    elements_per_block = BLOCK_SIZE * BLOCK_SIZE
    total_elements = (N - 2) * (N - 2)
    
    # Calculate starting element for this block
    block_start = pid * elements_per_block
    
    # Create offsets for the block
    offsets = tl.arange(0, BLOCK_SIZE)[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    flat_offsets = tl.reshape(offsets, (BLOCK_SIZE * BLOCK_SIZE,))
    element_ids = block_start + flat_offsets
    
    # Convert flat indices to i, j coordinates (1-based for the valid range)
    valid_range_width = N - 2
    i_coords = element_ids // valid_range_width + 1
    j_coords = element_ids % valid_range_width + 1
    
    # Create mask for valid elements
    mask = element_ids < total_elements
    
    # Time stepping loop
    for t in range(TSTEPS):
        # Load the 9-point stencil values
        idx_center = i_coords * N + j_coords
        
        # Load all 9 stencil points
        val_center = tl.load(A_ptr + idx_center, mask=mask)
        
        val_top_left = tl.load(A_ptr + (i_coords - 1) * N + (j_coords - 1), mask=mask)
        val_top = tl.load(A_ptr + (i_coords - 1) * N + j_coords, mask=mask)
        val_top_right = tl.load(A_ptr + (i_coords - 1) * N + (j_coords + 1), mask=mask)
        
        val_left = tl.load(A_ptr + i_coords * N + (j_coords - 1), mask=mask)
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
    
    # Calculate number of interior points
    total_elements = (N - 2) * (N - 2)
    elements_per_block = BLOCK_SIZE * BLOCK_SIZE
    num_blocks = (total_elements + elements_per_block - 1) // elements_per_block
    
    # Launch kernel
    seidel_2d_kernel[(num_blocks,)](
        A, N, TSTEPS, BLOCK_SIZE
    )