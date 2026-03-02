import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_2d_kernel(A_ptr, B_ptr, N, TSTEPS, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate starting position for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offset arrays once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Time stepping loop
    for t in range(TSTEPS):
        # First update: A -> B
        for i in range(1, N - 1):
            # Calculate j indices for this block
            j_indices = block_start + offsets
            
            # Create mask for valid j indices
            mask = (j_indices >= 1) & (j_indices < N - 1)
            
            # Calculate memory indices
            center_idx = i * N + j_indices
            left_idx = i * N + (j_indices - 1)
            right_idx = i * N + (j_indices + 1)
            up_idx = (i - 1) * N + j_indices
            down_idx = (i + 1) * N + j_indices
            
            # Load values
            a_center = tl.load(A_ptr + center_idx, mask=mask, other=0.0)
            a_left = tl.load(A_ptr + left_idx, mask=mask, other=0.0)
            a_right = tl.load(A_ptr + right_idx, mask=mask, other=0.0)
            a_up = tl.load(A_ptr + up_idx, mask=mask, other=0.0)
            a_down = tl.load(A_ptr + down_idx, mask=mask, other=0.0)
            
            # Compute new B values
            b_new = 0.2 * (a_center + a_left + a_right + a_up + a_down)
            
            # Store to B
            tl.store(B_ptr + center_idx, b_new, mask=mask)
        
        # Second update: B -> A
        for i in range(1, N - 1):
            # Calculate j indices for this block
            j_indices = block_start + offsets
            
            # Create mask for valid j indices
            mask = (j_indices >= 1) & (j_indices < N - 1)
            
            # Calculate memory indices
            center_idx = i * N + j_indices
            left_idx = i * N + (j_indices - 1)
            right_idx = i * N + (j_indices + 1)
            up_idx = (i - 1) * N + j_indices
            down_idx = (i + 1) * N + j_indices
            
            # Load values
            b_center = tl.load(B_ptr + center_idx, mask=mask, other=0.0)
            b_left = tl.load(B_ptr + left_idx, mask=mask, other=0.0)
            b_right = tl.load(B_ptr + right_idx, mask=mask, other=0.0)
            b_up = tl.load(B_ptr + up_idx, mask=mask, other=0.0)
            b_down = tl.load(B_ptr + down_idx, mask=mask, other=0.0)
            
            # Compute new A values
            a_new = 0.2 * (b_center + b_left + b_right + b_up + b_down)
            
            # Store to A
            tl.store(A_ptr + center_idx, a_new, mask=mask)

def jacobi_2d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 32
    
    # Calculate grid size for j dimension
    grid_size = triton.cdiv(N - 2, BLOCK_SIZE)
    
    # Launch kernel
    jacobi_2d_kernel[grid_size](
        A, B, N, TSTEPS, BLOCK_SIZE
    )