import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_2d_kernel(A_ptr, B_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Compute which block this program handles
    pid = tl.program_id(0)
    
    # Calculate total number of inner elements
    inner_size = (N - 2) * (N - 2)
    
    # Calculate starting position for this block
    start_idx = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = start_idx + offsets
    
    # Mask for valid elements
    mask = idx < inner_size
    
    for t in range(TSTEPS):
        # First phase: update B
        # Convert linear index to 2D coordinates
        valid_idx = tl.where(mask, idx, 0)
        i = 1 + valid_idx // (N - 2)
        j = 1 + valid_idx % (N - 2)
        
        # Calculate memory offsets for stencil pattern
        center_offset = i * N + j
        left_offset = i * N + (j - 1)
        right_offset = i * N + (j + 1)
        up_offset = (i - 1) * N + j
        down_offset = (i + 1) * N + j
        
        # Load stencil values from A
        a_center = tl.load(A_ptr + center_offset, mask=mask)
        a_left = tl.load(A_ptr + left_offset, mask=mask)
        a_right = tl.load(A_ptr + right_offset, mask=mask)
        a_up = tl.load(A_ptr + up_offset, mask=mask)
        a_down = tl.load(A_ptr + down_offset, mask=mask)
        
        # Compute new B values
        b_new = 0.2 * (a_center + a_left + a_right + a_up + a_down)
        
        # Store to B
        tl.store(B_ptr + center_offset, b_new, mask=mask)
        
        # Second phase: update A
        # Load stencil values from B
        b_center = tl.load(B_ptr + center_offset, mask=mask)
        b_left = tl.load(B_ptr + left_offset, mask=mask)
        b_right = tl.load(B_ptr + right_offset, mask=mask)
        b_up = tl.load(B_ptr + up_offset, mask=mask)
        b_down = tl.load(B_ptr + down_offset, mask=mask)
        
        # Compute new A values
        a_new = 0.2 * (b_center + b_left + b_right + b_up + b_down)
        
        # Store to A
        tl.store(A_ptr + center_offset, a_new, mask=mask)

def jacobi_2d_triton(A, B, N, TSTEPS):
    # Calculate number of inner elements to process
    inner_size = (N - 2) * (N - 2)
    
    # Block size
    BLOCK_SIZE = 128
    
    # Number of blocks needed
    num_blocks = triton.cdiv(inner_size, BLOCK_SIZE)
    
    # Launch kernel
    jacobi_2d_kernel[(num_blocks,)](
        A, B, N, TSTEPS, BLOCK_SIZE
    )