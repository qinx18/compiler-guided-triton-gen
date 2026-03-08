import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_2d_kernel(A_ptr, B_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Calculate total number of inner elements
    inner_size = (N - 2) * (N - 2)
    
    # Calculate starting position for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    mask = current_offsets < inner_size
    
    for t in range(TSTEPS):
        # Synchronization barrier between phases
        tl.debug_barrier()
        
        # Convert linear index to 2D coordinates in inner region
        valid_idx = tl.where(mask, current_offsets, 0)
        inner_i = valid_idx // (N - 2)
        inner_j = valid_idx % (N - 2)
        
        # Convert to actual array coordinates (add 1 for boundary offset)
        i = inner_i + 1
        j = inner_j + 1
        
        # Calculate memory offsets for 5-point stencil
        center_off = i * N + j
        left_off = i * N + (j - 1)
        right_off = i * N + (j + 1)
        up_off = (i - 1) * N + j
        down_off = (i + 1) * N + j
        
        # First phase: update B from A
        a_center = tl.load(A_ptr + center_off, mask=mask, other=0.0)
        a_left = tl.load(A_ptr + left_off, mask=mask, other=0.0)
        a_right = tl.load(A_ptr + right_off, mask=mask, other=0.0)
        a_up = tl.load(A_ptr + up_off, mask=mask, other=0.0)
        a_down = tl.load(A_ptr + down_off, mask=mask, other=0.0)
        
        b_new = 0.2 * (a_center + a_left + a_right + a_up + a_down)
        tl.store(B_ptr + center_off, b_new, mask=mask)
        
        # Synchronization barrier between phases
        tl.debug_barrier()
        
        # Second phase: update A from B
        b_center = tl.load(B_ptr + center_off, mask=mask, other=0.0)
        b_left = tl.load(B_ptr + left_off, mask=mask, other=0.0)
        b_right = tl.load(B_ptr + right_off, mask=mask, other=0.0)
        b_up = tl.load(B_ptr + up_off, mask=mask, other=0.0)
        b_down = tl.load(B_ptr + down_off, mask=mask, other=0.0)
        
        a_new = 0.2 * (b_center + b_left + b_right + b_up + b_down)
        tl.store(A_ptr + center_off, a_new, mask=mask)

def jacobi_2d_triton(A, B, N, TSTEPS):
    # Calculate number of inner elements to process
    inner_size = (N - 2) * (N - 2)
    
    # Block size for computation
    BLOCK_SIZE = 64
    
    # Calculate number of blocks needed
    num_blocks = triton.cdiv(inner_size, BLOCK_SIZE)
    
    # Launch the kernel
    jacobi_2d_kernel[(num_blocks,)](
        A, B, N, TSTEPS, BLOCK_SIZE
    )