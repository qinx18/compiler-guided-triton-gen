import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate which (i, j) pair this program handles
    total_elements = (N - 2) * (N - 2)
    element_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask = element_idx < total_elements
    
    # Convert linear index to (i, j) coordinates
    i = element_idx // (N - 2) + 1
    j = element_idx % (N - 2) + 1
    
    for t in range(TSTEPS):
        # Calculate linear indices for all 9 neighbors
        idx_center = i * N + j
        idx_top_left = (i - 1) * N + (j - 1)
        idx_top = (i - 1) * N + j
        idx_top_right = (i - 1) * N + (j + 1)
        idx_left = i * N + (j - 1)
        idx_right = i * N + (j + 1)
        idx_bottom_left = (i + 1) * N + (j - 1)
        idx_bottom = (i + 1) * N + j
        idx_bottom_right = (i + 1) * N + (j + 1)
        
        # Load all 9 values
        val_center = tl.load(A_ptr + idx_center, mask=mask)
        val_top_left = tl.load(A_ptr + idx_top_left, mask=mask)
        val_top = tl.load(A_ptr + idx_top, mask=mask)
        val_top_right = tl.load(A_ptr + idx_top_right, mask=mask)
        val_left = tl.load(A_ptr + idx_left, mask=mask)
        val_right = tl.load(A_ptr + idx_right, mask=mask)
        val_bottom_left = tl.load(A_ptr + idx_bottom_left, mask=mask)
        val_bottom = tl.load(A_ptr + idx_bottom, mask=mask)
        val_bottom_right = tl.load(A_ptr + idx_bottom_right, mask=mask)
        
        # Compute average
        result = (val_top_left + val_top + val_top_right + 
                 val_left + val_center + val_right + 
                 val_bottom_left + val_bottom + val_bottom_right) / 9.0
        
        # Store result
        tl.store(A_ptr + idx_center, result, mask=mask)

def seidel_2d_triton(A, N, TSTEPS):
    total_elements = (N - 2) * (N - 2)
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(total_elements, BLOCK_SIZE)
    
    seidel_2d_kernel[(grid_size,)](
        A, N, TSTEPS, BLOCK_SIZE
    )