import triton
import triton.language as tl
import torch

@triton.jit
def heat_3d_kernel(A_ptr, B_ptr, N: tl.constexpr, step: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate starting index for this block
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Total number of interior points
    total_points = (N - 2) * (N - 2) * (N - 2)
    mask = indices < total_points
    
    # Convert linear index to 3D coordinates (i, j, k)
    remaining = indices
    i_coord = remaining // ((N - 2) * (N - 2)) + 1
    remaining = remaining % ((N - 2) * (N - 2))
    j_coord = remaining // (N - 2) + 1
    k_coord = remaining % (N - 2) + 1
    
    # Calculate linear array indices for all stencil points
    center_idx = i_coord * N * N + j_coord * N + k_coord
    i_plus_idx = (i_coord + 1) * N * N + j_coord * N + k_coord
    i_minus_idx = (i_coord - 1) * N * N + j_coord * N + k_coord
    j_plus_idx = i_coord * N * N + (j_coord + 1) * N + k_coord
    j_minus_idx = i_coord * N * N + (j_coord - 1) * N + k_coord
    k_plus_idx = i_coord * N * N + j_coord * N + (k_coord + 1)
    k_minus_idx = i_coord * N * N + j_coord * N + (k_coord - 1)
    
    if step == 0:  # A -> B
        # Load values from A
        center = tl.load(A_ptr + center_idx, mask=mask, other=0.0)
        i_plus = tl.load(A_ptr + i_plus_idx, mask=mask, other=0.0)
        i_minus = tl.load(A_ptr + i_minus_idx, mask=mask, other=0.0)
        j_plus = tl.load(A_ptr + j_plus_idx, mask=mask, other=0.0)
        j_minus = tl.load(A_ptr + j_minus_idx, mask=mask, other=0.0)
        k_plus = tl.load(A_ptr + k_plus_idx, mask=mask, other=0.0)
        k_minus = tl.load(A_ptr + k_minus_idx, mask=mask, other=0.0)
        
        # Compute stencil
        result = (0.125 * (i_plus - 2.0 * center + i_minus) +
                 0.125 * (j_plus - 2.0 * center + j_minus) +
                 0.125 * (k_plus - 2.0 * center + k_minus) +
                 center)
        
        # Store to B
        tl.store(B_ptr + center_idx, result, mask=mask)
    else:  # B -> A
        # Load values from B
        center = tl.load(B_ptr + center_idx, mask=mask, other=0.0)
        i_plus = tl.load(B_ptr + i_plus_idx, mask=mask, other=0.0)
        i_minus = tl.load(B_ptr + i_minus_idx, mask=mask, other=0.0)
        j_plus = tl.load(B_ptr + j_plus_idx, mask=mask, other=0.0)
        j_minus = tl.load(B_ptr + j_minus_idx, mask=mask, other=0.0)
        k_plus = tl.load(B_ptr + k_plus_idx, mask=mask, other=0.0)
        k_minus = tl.load(B_ptr + k_minus_idx, mask=mask, other=0.0)
        
        # Compute stencil
        result = (0.125 * (i_plus - 2.0 * center + i_minus) +
                 0.125 * (j_plus - 2.0 * center + j_minus) +
                 0.125 * (k_plus - 2.0 * center + k_minus) +
                 center)
        
        # Store to A
        tl.store(A_ptr + center_idx, result, mask=mask)

def heat_3d_triton(A, B, N, TSTEPS):
    # Calculate total number of interior points
    total_points = (N - 2) * (N - 2) * (N - 2)
    
    # Block size
    BLOCK_SIZE = 128
    
    # Number of blocks needed
    num_blocks = triton.cdiv(total_points, BLOCK_SIZE)
    
    # Time stepping loop
    for t in range(1, TSTEPS + 1):
        # First step: A -> B
        heat_3d_kernel[(num_blocks,)](
            A, B, N, 0, BLOCK_SIZE
        )
        
        # Second step: B -> A
        heat_3d_kernel[(num_blocks,)](
            A, B, N, 1, BLOCK_SIZE
        )