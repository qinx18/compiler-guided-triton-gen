import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate which (i, j) pair this program handles
    total_elements = (N - 2) * (N - 2)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    element_ids = block_start + offsets
    mask = element_ids < total_elements
    
    # Convert linear element_id to (i, j) coordinates
    i_coords = (element_ids // (N - 2)) + 1
    j_coords = (element_ids % (N - 2)) + 1
    
    for t in range(TSTEPS):
        # Calculate all neighbor offsets
        center_idx = i_coords * N + j_coords
        
        idx_i_minus_1_j_minus_1 = (i_coords - 1) * N + (j_coords - 1)
        idx_i_minus_1_j = (i_coords - 1) * N + j_coords
        idx_i_minus_1_j_plus_1 = (i_coords - 1) * N + (j_coords + 1)
        
        idx_i_j_minus_1 = i_coords * N + (j_coords - 1)
        idx_i_j = i_coords * N + j_coords
        idx_i_j_plus_1 = i_coords * N + (j_coords + 1)
        
        idx_i_plus_1_j_minus_1 = (i_coords + 1) * N + (j_coords - 1)
        idx_i_plus_1_j = (i_coords + 1) * N + j_coords
        idx_i_plus_1_j_plus_1 = (i_coords + 1) * N + (j_coords + 1)
        
        # Load all 9 values
        val_i_minus_1_j_minus_1 = tl.load(A_ptr + idx_i_minus_1_j_minus_1, mask=mask)
        val_i_minus_1_j = tl.load(A_ptr + idx_i_minus_1_j, mask=mask)
        val_i_minus_1_j_plus_1 = tl.load(A_ptr + idx_i_minus_1_j_plus_1, mask=mask)
        
        val_i_j_minus_1 = tl.load(A_ptr + idx_i_j_minus_1, mask=mask)
        val_i_j = tl.load(A_ptr + idx_i_j, mask=mask)
        val_i_j_plus_1 = tl.load(A_ptr + idx_i_j_plus_1, mask=mask)
        
        val_i_plus_1_j_minus_1 = tl.load(A_ptr + idx_i_plus_1_j_minus_1, mask=mask)
        val_i_plus_1_j = tl.load(A_ptr + idx_i_plus_1_j, mask=mask)
        val_i_plus_1_j_plus_1 = tl.load(A_ptr + idx_i_plus_1_j_plus_1, mask=mask)
        
        # Compute the average
        result = (val_i_minus_1_j_minus_1 + val_i_minus_1_j + val_i_minus_1_j_plus_1 +
                 val_i_j_minus_1 + val_i_j + val_i_j_plus_1 +
                 val_i_plus_1_j_minus_1 + val_i_plus_1_j + val_i_plus_1_j_plus_1) / 9.0
        
        # Store the result
        tl.store(A_ptr + center_idx, result, mask=mask)

def seidel_2d_triton(A, N, TSTEPS):
    BLOCK_SIZE = 256
    total_elements = (N - 2) * (N - 2)
    grid_size = triton.cdiv(total_elements, BLOCK_SIZE)
    
    seidel_2d_kernel[(grid_size,)](
        A, N, TSTEPS, BLOCK_SIZE
    )