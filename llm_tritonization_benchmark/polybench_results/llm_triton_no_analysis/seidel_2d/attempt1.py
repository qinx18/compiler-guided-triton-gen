import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A_ptr, N, TSTEPS, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate how many elements we need to process (inner region only)
    inner_size = N - 2
    total_elements = inner_size * inner_size
    
    # Calculate starting position for this block
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Sequential time steps
    for t in range(TSTEPS):
        # Process this block
        element_ids = block_start + offsets
        mask = element_ids < total_elements
        
        # Convert linear index to 2D coordinates in inner region
        inner_i = element_ids // inner_size
        inner_j = element_ids % inner_size
        
        # Convert to actual array coordinates (add 1 for boundary)
        i = inner_i + 1
        j = inner_j + 1
        
        # Calculate all 9 neighbor positions
        center_pos = i * N + j
        top_left = (i - 1) * N + (j - 1)
        top_center = (i - 1) * N + j
        top_right = (i - 1) * N + (j + 1)
        mid_left = i * N + (j - 1)
        mid_right = i * N + (j + 1)
        bottom_left = (i + 1) * N + (j - 1)
        bottom_center = (i + 1) * N + j
        bottom_right = (i + 1) * N + (j + 1)
        
        # Load all 9 values
        val_center = tl.load(A_ptr + center_pos, mask=mask)
        val_tl = tl.load(A_ptr + top_left, mask=mask)
        val_tc = tl.load(A_ptr + top_center, mask=mask)
        val_tr = tl.load(A_ptr + top_right, mask=mask)
        val_ml = tl.load(A_ptr + mid_left, mask=mask)
        val_mr = tl.load(A_ptr + mid_right, mask=mask)
        val_bl = tl.load(A_ptr + bottom_left, mask=mask)
        val_bc = tl.load(A_ptr + bottom_center, mask=mask)
        val_br = tl.load(A_ptr + bottom_right, mask=mask)
        
        # Compute average
        result = (val_tl + val_tc + val_tr + val_ml + val_center + val_mr + val_bl + val_bc + val_br) / 9.0
        
        # Store result
        tl.store(A_ptr + center_pos, result, mask=mask)

def seidel_2d_triton(A, N, TSTEPS):
    # Calculate number of inner elements
    inner_size = N - 2
    total_elements = inner_size * inner_size
    
    if total_elements <= 0:
        return
    
    # Block size
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(total_elements, BLOCK_SIZE)
    
    # Launch kernel
    seidel_2d_kernel[(grid_size,)](
        A, N, TSTEPS, BLOCK_SIZE
    )