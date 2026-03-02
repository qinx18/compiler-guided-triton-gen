import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A, N, TSTEPS, t_start, t_end, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_offsets = tl.arange(0, BLOCK_SIZE_I)
    j_offsets = tl.arange(0, BLOCK_SIZE_J)
    
    i_base = 1 + pid_i * BLOCK_SIZE_I
    j_base = 1 + pid_j * BLOCK_SIZE_J
    
    i_indices = i_base + i_offsets
    j_indices = j_base + j_offsets
    
    i_mask = (i_indices >= 1) & (i_indices <= N - 2)
    j_mask = (j_indices >= 1) & (j_indices <= N - 2)
    
    for t in range(t_start, t_end):
        # Load 9-point stencil values for each point
        for ii in range(BLOCK_SIZE_I):
            for jj in range(BLOCK_SIZE_J):
                i_curr = i_base + ii
                j_curr = j_base + jj
                
                if i_curr >= 1 and i_curr <= N - 2 and j_curr >= 1 and j_curr <= N - 2:
                    # Compute 9-point stencil
                    val = 0.0
                    for di in range(-1, 2):
                        for dj in range(-1, 2):
                            idx = (i_curr + di) * N + (j_curr + dj)
                            val += tl.load(A + idx)
                    
                    val = val / 9.0
                    store_idx = i_curr * N + j_curr
                    tl.store(A + store_idx, val)

def seidel_2d_triton(A, N, TSTEPS):
    BLOCK_SIZE_I = 16
    BLOCK_SIZE_J = 16
    
    grid_i = triton.cdiv(N - 2, BLOCK_SIZE_I)
    grid_j = triton.cdiv(N - 2, BLOCK_SIZE_J)
    
    # Execute time steps sequentially to handle dependencies
    for t in range(TSTEPS):
        grid = (grid_i, grid_j)
        seidel_2d_kernel[grid](
            A, N, TSTEPS, t, t + 1,
            BLOCK_SIZE_I=BLOCK_SIZE_I,
            BLOCK_SIZE_J=BLOCK_SIZE_J
        )