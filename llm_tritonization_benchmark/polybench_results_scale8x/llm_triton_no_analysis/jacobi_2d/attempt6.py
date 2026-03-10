import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_2d_kernel(A_ptr, B_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get block indices
    block_i = tl.program_id(0)
    block_j = tl.program_id(1)
    
    # Calculate base indices for this block
    base_i = 1 + block_i * BLOCK_SIZE
    base_j = 1 + block_j * BLOCK_SIZE
    
    # Create offset arrays
    i_offsets = tl.arange(0, BLOCK_SIZE)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for t in range(TSTEPS):
        # First phase: A -> B
        for ii in range(BLOCK_SIZE):
            for jj in range(BLOCK_SIZE):
                i = base_i + ii
                j = base_j + jj
                
                if i < N - 1 and j < N - 1:
                    # Load the 5-point stencil from A
                    center = tl.load(A_ptr + i * N + j)
                    left = tl.load(A_ptr + i * N + (j - 1))
                    right = tl.load(A_ptr + i * N + (j + 1))
                    up = tl.load(A_ptr + (i - 1) * N + j)
                    down = tl.load(A_ptr + (i + 1) * N + j)
                    
                    # Compute and store result
                    result = 0.2 * (center + left + right + up + down)
                    tl.store(B_ptr + i * N + j, result)
        
        # Second phase: B -> A
        for ii in range(BLOCK_SIZE):
            for jj in range(BLOCK_SIZE):
                i = base_i + ii
                j = base_j + jj
                
                if i < N - 1 and j < N - 1:
                    # Load the 5-point stencil from B
                    center = tl.load(B_ptr + i * N + j)
                    left = tl.load(B_ptr + i * N + (j - 1))
                    right = tl.load(B_ptr + i * N + (j + 1))
                    up = tl.load(B_ptr + (i - 1) * N + j)
                    down = tl.load(B_ptr + (i + 1) * N + j)
                    
                    # Compute and store result
                    result = 0.2 * (center + left + right + up + down)
                    tl.store(A_ptr + i * N + j, result)

def jacobi_2d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE = 16
    grid_size = triton.cdiv(N - 2, BLOCK_SIZE)
    grid = (grid_size, grid_size)
    
    jacobi_2d_kernel[grid](
        A, B, N, TSTEPS, BLOCK_SIZE
    )