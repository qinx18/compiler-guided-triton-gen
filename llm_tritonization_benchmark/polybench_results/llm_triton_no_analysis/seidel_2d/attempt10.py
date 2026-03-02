import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A_ptr, N, TSTEPS, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate starting position for this block
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Time stepping loop
    for t in range(TSTEPS):
        # Process elements sequentially within each time step
        for elem_idx in range(BLOCK_SIZE):
            linear_idx = block_start + elem_idx
            
            # Skip if out of bounds - use return instead of continue
            if linear_idx >= (N - 2) * (N - 2):
                return
            
            # Convert linear index to 2D coordinates (for interior points)
            i = 1 + (linear_idx // (N - 2))
            j = 1 + (linear_idx % (N - 2))
            
            # Calculate 2D array index
            center_idx = i * N + j
            
            # Load all 9 stencil points
            val_top_left = tl.load(A_ptr + (i - 1) * N + (j - 1))
            val_top = tl.load(A_ptr + (i - 1) * N + j)
            val_top_right = tl.load(A_ptr + (i - 1) * N + (j + 1))
            
            val_left = tl.load(A_ptr + i * N + (j - 1))
            val_center = tl.load(A_ptr + center_idx)
            val_right = tl.load(A_ptr + i * N + (j + 1))
            
            val_bottom_left = tl.load(A_ptr + (i + 1) * N + (j - 1))
            val_bottom = tl.load(A_ptr + (i + 1) * N + j)
            val_bottom_right = tl.load(A_ptr + (i + 1) * N + (j + 1))
            
            # Compute the average
            new_val = (val_top_left + val_top + val_top_right +
                       val_left + val_center + val_right +
                       val_bottom_left + val_bottom + val_bottom_right) / 9.0
            
            # Store the result
            tl.store(A_ptr + center_idx, new_val)

def seidel_2d_triton(A, N, TSTEPS):
    BLOCK_SIZE = 32
    
    # Total number of interior points
    total_interior = (N - 2) * (N - 2)
    
    # Calculate grid size
    grid_size = triton.cdiv(total_interior, BLOCK_SIZE)
    
    # Launch kernel
    seidel_2d_kernel[(grid_size,)](
        A, N, TSTEPS, BLOCK_SIZE
    )