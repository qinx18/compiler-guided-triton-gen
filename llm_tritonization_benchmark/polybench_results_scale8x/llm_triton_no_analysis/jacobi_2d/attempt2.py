import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_2d_kernel(A_ptr, B_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for this block
    pid = tl.program_id(0)
    
    # Calculate total number of interior points
    interior_size = (N - 2) * (N - 2)
    
    # Calculate starting position for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Create mask for valid indices
    mask = indices < interior_size
    
    # Convert linear indices to 2D coordinates (i, j)
    i_coords = (indices // (N - 2)) + 1
    j_coords = (indices % (N - 2)) + 1
    
    # Time stepping loop
    for t in range(TSTEPS):
        # Synchronization barrier to ensure all threads complete before next phase
        tl.debug_barrier()
        
        # First phase: compute B from A
        # Calculate addresses for the 5-point stencil
        center_addr = i_coords * N + j_coords
        left_addr = i_coords * N + (j_coords - 1)
        right_addr = i_coords * N + (j_coords + 1)
        up_addr = (i_coords - 1) * N + j_coords
        down_addr = (i_coords + 1) * N + j_coords
        
        # Load values from A
        a_center = tl.load(A_ptr + center_addr, mask=mask)
        a_left = tl.load(A_ptr + left_addr, mask=mask)
        a_right = tl.load(A_ptr + right_addr, mask=mask)
        a_up = tl.load(A_ptr + up_addr, mask=mask)
        a_down = tl.load(A_ptr + down_addr, mask=mask)
        
        # Compute B values
        b_new = 0.2 * (a_center + a_left + a_right + a_up + a_down)
        
        # Store to B
        tl.store(B_ptr + center_addr, b_new, mask=mask)
        
        # Synchronization barrier to ensure all B values are written before reading
        tl.debug_barrier()
        
        # Second phase: compute A from B
        # Load values from B
        b_center = tl.load(B_ptr + center_addr, mask=mask)
        b_left = tl.load(B_ptr + left_addr, mask=mask)
        b_right = tl.load(B_ptr + right_addr, mask=mask)
        b_up = tl.load(B_ptr + up_addr, mask=mask)
        b_down = tl.load(B_ptr + down_addr, mask=mask)
        
        # Compute A values
        a_new = 0.2 * (b_center + b_left + b_right + b_up + b_down)
        
        # Store to A
        tl.store(A_ptr + center_addr, a_new, mask=mask)

def jacobi_2d_triton(A, B, N, TSTEPS):
    # Calculate number of interior points
    interior_size = (N - 2) * (N - 2)
    
    # Choose block size
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(interior_size, BLOCK_SIZE)
    
    # Launch kernel with single grid dimension to ensure synchronization
    jacobi_2d_kernel[(1,)](
        A, B, N, TSTEPS, interior_size
    )