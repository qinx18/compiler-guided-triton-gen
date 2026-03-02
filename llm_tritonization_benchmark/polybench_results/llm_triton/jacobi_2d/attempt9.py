import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_2d_kernel(A, B, N, t_step: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Calculate i and j indices
    i = pid_i + 1
    j_start = pid_j * BLOCK_SIZE
    
    # Check i bounds
    if i >= N - 1:
        return
    
    # Process j values in blocks
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_indices = j_start + j_offsets + 1
    j_mask = (j_indices < N - 1)
    
    # First phase: compute B[i][j] from A
    if j_mask.any():
        # Load A values
        center_ptrs = A + i * N + j_indices
        left_ptrs = A + i * N + (j_indices - 1)
        right_ptrs = A + i * N + (j_indices + 1)
        up_ptrs = A + (i - 1) * N + j_indices
        down_ptrs = A + (i + 1) * N + j_indices
        
        center = tl.load(center_ptrs, mask=j_mask)
        left = tl.load(left_ptrs, mask=j_mask)
        right = tl.load(right_ptrs, mask=j_mask)
        up = tl.load(up_ptrs, mask=j_mask)
        down = tl.load(down_ptrs, mask=j_mask)
        
        b_val = 0.2 * (center + left + right + up + down)
        
        # Store to B
        b_ptrs = B + i * N + j_indices
        tl.store(b_ptrs, b_val, mask=j_mask)

@triton.jit
def jacobi_2d_kernel_phase2(A, B, N, t_step: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Calculate i and j indices
    i = pid_i + 1
    j_start = pid_j * BLOCK_SIZE
    
    # Check i bounds
    if i >= N - 1:
        return
    
    # Process j values in blocks
    j_offsets = tl.arange(0, BLOCK_SIZE)
    j_indices = j_start + j_offsets + 1
    j_mask = (j_indices < N - 1)
    
    # Second phase: compute A[i][j] from B
    if j_mask.any():
        # Load B values
        center_ptrs = B + i * N + j_indices
        left_ptrs = B + i * N + (j_indices - 1)
        right_ptrs = B + i * N + (j_indices + 1)
        up_ptrs = B + (i - 1) * N + j_indices
        down_ptrs = B + (i + 1) * N + j_indices
        
        center = tl.load(center_ptrs, mask=j_mask)
        left = tl.load(left_ptrs, mask=j_mask)
        right = tl.load(right_ptrs, mask=j_mask)
        up = tl.load(up_ptrs, mask=j_mask)
        down = tl.load(down_ptrs, mask=j_mask)
        
        a_val = 0.2 * (center + left + right + up + down)
        
        # Store to A
        a_ptrs = A + i * N + j_indices
        tl.store(a_ptrs, a_val, mask=j_mask)

def jacobi_2d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 32
    
    for t in range(TSTEPS):
        grid = (N - 2, triton.cdiv(N - 2, BLOCK_SIZE))
        
        # Phase 1: A -> B
        jacobi_2d_kernel[grid](A, B, N, t, BLOCK_SIZE)
        
        # Phase 2: B -> A
        jacobi_2d_kernel_phase2[grid](A, B, N, t, BLOCK_SIZE)