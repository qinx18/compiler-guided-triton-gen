import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_2d_kernel(A, B, N, TSTEPS, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_T: tl.constexpr):
    # Get program IDs
    pid_i = tl.program_id(0)
    pid_t = tl.program_id(1)
    
    # Calculate base indices
    base_i = pid_i * BLOCK_SIZE_I
    base_t = pid_t * BLOCK_SIZE_T
    
    # Create offset arrays once
    offsets_i = tl.arange(0, BLOCK_SIZE_I)
    offsets_t = tl.arange(0, BLOCK_SIZE_T)
    
    # Process time steps in blocks
    for t_block in range(BLOCK_SIZE_T):
        t_idx = base_t + t_block
        t_valid = t_idx < TSTEPS
        
        if t_valid:
            # First stencil: A -> B
            for i_block in range(BLOCK_SIZE_I):
                i_idx = base_i + i_block
                i_valid = (i_idx >= 1) & (i_idx < N - 1)
                
                if i_valid:
                    for j in range(1, N - 1):
                        # Load stencil points for A -> B
                        center = tl.load(A + i_idx * N + j)
                        left = tl.load(A + i_idx * N + (j - 1))
                        right = tl.load(A + i_idx * N + (j + 1))
                        up = tl.load(A + (i_idx - 1) * N + j)
                        down = tl.load(A + (i_idx + 1) * N + j)
                        
                        # Compute and store result
                        result = 0.2 * (center + left + right + up + down)
                        tl.store(B + i_idx * N + j, result)
            
            # Synchronization barrier
            tl.debug_barrier()
            
            # Second stencil: B -> A
            for i_block in range(BLOCK_SIZE_I):
                i_idx = base_i + i_block
                i_valid = (i_idx >= 1) & (i_idx < N - 1)
                
                if i_valid:
                    for j in range(1, N - 1):
                        # Load stencil points for B -> A
                        center = tl.load(B + i_idx * N + j)
                        left = tl.load(B + i_idx * N + (j - 1))
                        right = tl.load(B + i_idx * N + (j + 1))
                        up = tl.load(B + (i_idx - 1) * N + j)
                        down = tl.load(B + (i_idx + 1) * N + j)
                        
                        # Compute and store result
                        result = 0.2 * (center + left + right + up + down)
                        tl.store(A + i_idx * N + j, result)

def jacobi_2d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE_I = 32
    BLOCK_SIZE_T = 4
    
    grid_i = triton.cdiv(N - 2, BLOCK_SIZE_I)
    grid_t = triton.cdiv(TSTEPS, BLOCK_SIZE_T)
    
    jacobi_2d_kernel[(grid_i, grid_t)](
        A, B, N, TSTEPS,
        BLOCK_SIZE_I=BLOCK_SIZE_I,
        BLOCK_SIZE_T=BLOCK_SIZE_T
    )