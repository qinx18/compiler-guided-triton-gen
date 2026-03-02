import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A_ptr, N, TSTEPS, BLOCK_SIZE: tl.constexpr):
    # Sequential time steps - must be done one at a time for Gauss-Seidel
    for t in range(TSTEPS):
        # Process rows sequentially within each time step
        for row in range(1, N - 1):
            # Get program ID for column processing
            pid = tl.program_id(0)
            
            # Calculate column indices for this block
            block_start = pid * BLOCK_SIZE
            offsets = tl.arange(0, BLOCK_SIZE)
            j = block_start + offsets + 1  # +1 because we start from column 1
            mask = j < (N - 1)  # j ranges from 1 to N-2
            
            # Calculate linear indices for 3x3 stencil
            center_idx = row * N + j
            
            # Top row
            top_left = (row - 1) * N + (j - 1)
            top_center = (row - 1) * N + j
            top_right = (row - 1) * N + (j + 1)
            
            # Middle row
            mid_left = row * N + (j - 1)
            mid_right = row * N + (j + 1)
            
            # Bottom row
            bottom_left = (row + 1) * N + (j - 1)
            bottom_center = (row + 1) * N + j
            bottom_right = (row + 1) * N + (j + 1)
            
            # Load all 9 stencil values
            val_tl = tl.load(A_ptr + top_left, mask=mask)
            val_tc = tl.load(A_ptr + top_center, mask=mask)
            val_tr = tl.load(A_ptr + top_right, mask=mask)
            val_ml = tl.load(A_ptr + mid_left, mask=mask)
            val_center = tl.load(A_ptr + center_idx, mask=mask)
            val_mr = tl.load(A_ptr + mid_right, mask=mask)
            val_bl = tl.load(A_ptr + bottom_left, mask=mask)
            val_bc = tl.load(A_ptr + bottom_center, mask=mask)
            val_br = tl.load(A_ptr + bottom_right, mask=mask)
            
            # Compute the 9-point stencil average
            result = (val_tl + val_tc + val_tr + val_ml + val_center + val_mr + val_bl + val_bc + val_br) / 9.0
            
            # Store result back to center position
            tl.store(A_ptr + center_idx, result, mask=mask)

def seidel_2d_triton(A, N, TSTEPS):
    # Number of columns to process (excluding boundaries)
    inner_cols = N - 2
    
    if inner_cols <= 0:
        return
    
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(inner_cols, BLOCK_SIZE)
    
    # Launch kernel
    seidel_2d_kernel[(grid_size,)](
        A, N, TSTEPS, BLOCK_SIZE
    )