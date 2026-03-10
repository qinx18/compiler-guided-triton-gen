import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_2d_kernel(A_ptr, B_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program IDs
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Calculate starting indices for interior points
    start_i = 1 + pid_i * BLOCK_SIZE
    start_j = 1 + pid_j * BLOCK_SIZE
    
    # Create offset arrays once
    i_offs = tl.arange(0, BLOCK_SIZE)
    j_offs = tl.arange(0, BLOCK_SIZE)
    
    for t in range(TSTEPS):
        # Phase 1: A -> B
        for ii in range(BLOCK_SIZE):
            for jj in range(BLOCK_SIZE):
                i = start_i + ii
                j = start_j + jj
                
                # Check bounds for interior points
                if (i < N - 1) & (j < N - 1):
                    # Load 5-point stencil from A
                    center_idx = i * N + j
                    left_idx = i * N + (j - 1)
                    right_idx = i * N + (j + 1)
                    up_idx = (i - 1) * N + j
                    down_idx = (i + 1) * N + j
                    
                    center = tl.load(A_ptr + center_idx)
                    left = tl.load(A_ptr + left_idx)
                    right = tl.load(A_ptr + right_idx)
                    up = tl.load(A_ptr + up_idx)
                    down = tl.load(A_ptr + down_idx)
                    
                    result = 0.2 * (center + left + right + up + down)
                    tl.store(B_ptr + center_idx, result)
        
        # Phase 2: B -> A
        for ii in range(BLOCK_SIZE):
            for jj in range(BLOCK_SIZE):
                i = start_i + ii
                j = start_j + jj
                
                # Check bounds for interior points
                if (i < N - 1) & (j < N - 1):
                    # Load 5-point stencil from B
                    center_idx = i * N + j
                    left_idx = i * N + (j - 1)
                    right_idx = i * N + (j + 1)
                    up_idx = (i - 1) * N + j
                    down_idx = (i + 1) * N + j
                    
                    center = tl.load(B_ptr + center_idx)
                    left = tl.load(B_ptr + left_idx)
                    right = tl.load(B_ptr + right_idx)
                    up = tl.load(B_ptr + up_idx)
                    down = tl.load(B_ptr + down_idx)
                    
                    result = 0.2 * (center + left + right + up + down)
                    tl.store(A_ptr + center_idx, result)

def jacobi_2d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 32
    # Grid covers interior points only
    grid_dim = triton.cdiv(N - 2, BLOCK_SIZE)
    grid = (grid_dim, grid_dim)
    
    jacobi_2d_kernel[grid](
        A, B, N, TSTEPS, BLOCK_SIZE
    )