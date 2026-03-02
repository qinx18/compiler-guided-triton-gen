import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_2d_kernel(A_ptr, B_ptr, N, TSTEPS, BLOCK_SIZE: tl.constexpr):
    # Get program ID for this thread block
    pid = tl.program_id(0)
    
    # Calculate which elements this block handles
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    element_ids = block_start + offsets
    
    # Each element_id represents a (i,j) pair in the valid range
    # Valid range: i in [1, N-2], j in [1, N-2]
    valid_size = (N - 2) * (N - 2)
    mask = element_ids < valid_size
    
    # Convert linear element_id to (i,j) coordinates
    i_coords = element_ids // (N - 2) + 1  # Add 1 to skip boundary
    j_coords = element_ids % (N - 2) + 1   # Add 1 to skip boundary
    
    # Time loop
    for t in range(TSTEPS):
        # Phase 1: Update B from A
        # Calculate linear indices for stencil pattern
        center_idx = i_coords * N + j_coords
        left_idx = i_coords * N + (j_coords - 1)
        right_idx = i_coords * N + (j_coords + 1)
        up_idx = (i_coords - 1) * N + j_coords
        down_idx = (i_coords + 1) * N + j_coords
        
        # Load A values
        a_center = tl.load(A_ptr + center_idx, mask=mask)
        a_left = tl.load(A_ptr + left_idx, mask=mask)
        a_right = tl.load(A_ptr + right_idx, mask=mask)
        a_up = tl.load(A_ptr + up_idx, mask=mask)
        a_down = tl.load(A_ptr + down_idx, mask=mask)
        
        # Compute B values
        b_val = 0.2 * (a_center + a_left + a_right + a_up + a_down)
        
        # Store B values
        tl.store(B_ptr + center_idx, b_val, mask=mask)
        
        # Synchronization barrier to ensure B is fully updated before using it
        tl.debug_barrier()
        
        # Phase 2: Update A from B
        # Load B values
        b_center = tl.load(B_ptr + center_idx, mask=mask)
        b_left = tl.load(B_ptr + left_idx, mask=mask)
        b_right = tl.load(B_ptr + right_idx, mask=mask)
        b_up = tl.load(B_ptr + up_idx, mask=mask)
        b_down = tl.load(B_ptr + down_idx, mask=mask)
        
        # Compute A values
        a_val = 0.2 * (b_center + b_left + b_right + b_up + b_down)
        
        # Store A values
        tl.store(A_ptr + center_idx, a_val, mask=mask)
        
        # Synchronization barrier to ensure A is fully updated before next iteration
        tl.debug_barrier()

def jacobi_2d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 64
    
    # Number of interior elements (excluding boundaries)
    valid_size = (N - 2) * (N - 2)
    
    # Calculate grid size
    grid_size = triton.cdiv(valid_size, BLOCK_SIZE)
    
    # Launch kernel with single grid dimension
    jacobi_2d_kernel[(grid_size,)](
        A, B, N, TSTEPS, BLOCK_SIZE
    )