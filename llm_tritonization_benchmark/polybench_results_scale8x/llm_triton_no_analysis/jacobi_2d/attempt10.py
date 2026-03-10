import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_2d_kernel(A_ptr, B_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    start_i = 1 + pid_i * BLOCK_SIZE
    start_j = 1 + pid_j * BLOCK_SIZE
    
    i_offs = tl.arange(0, BLOCK_SIZE)
    j_offs = tl.arange(0, BLOCK_SIZE)
    
    for t in range(TSTEPS):
        # Phase 1: A -> B
        i_indices = start_i + i_offs[:, None]
        j_indices = start_j + j_offs[None, :]
        
        i_mask = (i_indices >= 1) & (i_indices < N - 1)
        j_mask = (j_indices >= 1) & (j_indices < N - 1)
        mask = i_mask & j_mask
        
        # Load center points
        center_idx = i_indices * N + j_indices
        center = tl.load(A_ptr + center_idx, mask=mask)
        
        # Load neighbors
        left = tl.load(A_ptr + i_indices * N + (j_indices - 1), mask=mask)
        right = tl.load(A_ptr + i_indices * N + (j_indices + 1), mask=mask)
        up = tl.load(A_ptr + (i_indices - 1) * N + j_indices, mask=mask)
        down = tl.load(A_ptr + (i_indices + 1) * N + j_indices, mask=mask)
        
        result = 0.2 * (center + left + right + up + down)
        tl.store(B_ptr + center_idx, result, mask=mask)
        
        # Phase 2: B -> A
        # Load center points from B
        center = tl.load(B_ptr + center_idx, mask=mask)
        
        # Load neighbors from B
        left = tl.load(B_ptr + i_indices * N + (j_indices - 1), mask=mask)
        right = tl.load(B_ptr + i_indices * N + (j_indices + 1), mask=mask)
        up = tl.load(B_ptr + (i_indices - 1) * N + j_indices, mask=mask)
        down = tl.load(B_ptr + (i_indices + 1) * N + j_indices, mask=mask)
        
        result = 0.2 * (center + left + right + up + down)
        tl.store(A_ptr + center_idx, result, mask=mask)

def jacobi_2d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 16
    grid_dim = triton.cdiv(N - 2, BLOCK_SIZE)
    grid = (grid_dim, grid_dim)
    
    jacobi_2d_kernel[grid](
        A, B, N, TSTEPS, BLOCK_SIZE
    )