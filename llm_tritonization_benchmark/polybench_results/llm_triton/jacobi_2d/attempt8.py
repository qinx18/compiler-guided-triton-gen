import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_2d_kernel(A, B, N, TSTEPS, t_start: tl.constexpr, BLOCK_SIZE_T: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_i = tl.program_id(1)
    
    # Calculate actual t and i indices
    t = t_start + pid_t
    i = pid_i + 1
    
    # Check bounds
    if t >= TSTEPS or i >= N - 1:
        return
    
    # Process all j values for this (t, i) pair
    for j in range(1, N - 1):
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
    
    for j in range(1, N - 1):
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
    # Process one timestep at a time to ensure correctness
    for t_start in range(TSTEPS):
        grid = (1, N - 2)  # 1 timestep, (N-2) i values
        BLOCK_SIZE_T = 1
        
        jacobi_2d_kernel[grid](A, B, N, TSTEPS, t_start, BLOCK_SIZE_T)