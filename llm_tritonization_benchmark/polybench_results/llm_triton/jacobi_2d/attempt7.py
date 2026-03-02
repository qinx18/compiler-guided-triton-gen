import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_2d_kernel(A, B, N, TSTEPS, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Each program handles one (i,j) position
    total_points = (N - 2) * (N - 2)
    
    if pid >= total_points:
        return
    
    # Convert linear pid to (i,j) coordinates
    i = (pid // (N - 2)) + 1
    j = (pid % (N - 2)) + 1
    
    # Time loop - must be sequential
    for t in range(TSTEPS):
        # First phase: compute B[i][j] from A
        center = tl.load(A + i * N + j)
        left = tl.load(A + i * N + (j - 1))
        right = tl.load(A + i * N + (j + 1))
        up = tl.load(A + (i - 1) * N + j)
        down = tl.load(A + (i + 1) * N + j)
        
        b_val = 0.2 * (center + left + right + up + down)
        tl.store(B + i * N + j, b_val)
        
        # Wait for all threads to complete B computation
        tl.debug_barrier()
        
        # Second phase: compute A[i][j] from B
        center = tl.load(B + i * N + j)
        left = tl.load(B + i * N + (j - 1))
        right = tl.load(B + i * N + (j + 1))
        up = tl.load(B + (i - 1) * N + j)
        down = tl.load(B + (i + 1) * N + j)
        
        a_val = 0.2 * (center + left + right + up + down)
        tl.store(A + i * N + j, a_val)
        
        # Wait for all threads to complete A computation
        tl.debug_barrier()

def jacobi_2d_triton(A, B, N, TSTEPS):
    # Number of interior points to update
    total_points = (N - 2) * (N - 2)
    
    # Launch one thread per interior point
    BLOCK_SIZE = 1
    grid = (total_points,)
    
    jacobi_2d_kernel[grid](A, B, N, TSTEPS, BLOCK_SIZE)