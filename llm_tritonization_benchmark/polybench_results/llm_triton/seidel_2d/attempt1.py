import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A, N, TSTEPS, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr):
    pid_t = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_j = tl.program_id(2)
    
    i_offsets = tl.arange(0, BLOCK_SIZE_I)
    j_offsets = tl.arange(0, BLOCK_SIZE_J)
    
    for t in range(TSTEPS):
        i_base = 1 + pid_i * BLOCK_SIZE_I
        j_base = 1 + pid_j * BLOCK_SIZE_J
        
        i_indices = i_base + i_offsets
        j_indices = j_base + j_offsets
        
        i_mask = (i_indices >= 1) & (i_indices <= N - 2)
        j_mask = (j_indices >= 1) & (j_indices <= N - 2)
        
        # Load 9-point stencil values
        vals = tl.zeros((BLOCK_SIZE_I, BLOCK_SIZE_J), dtype=tl.float32)
        
        for di in range(-1, 2):
            for dj in range(-1, 2):
                i_load = i_indices[:, None] + di
                j_load = j_indices[None, :] + dj
                linear_idx = i_load * N + j_load
                mask_2d = i_mask[:, None] & j_mask[None, :]
                val = tl.load(A + linear_idx, mask=mask_2d, other=0.0)
                vals += val
        
        vals = vals / 9.0
        
        # Store result
        i_store = i_indices[:, None]
        j_store = j_indices[None, :]
        linear_idx = i_store * N + j_store
        mask_2d = i_mask[:, None] & j_mask[None, :]
        tl.store(A + linear_idx, vals, mask=mask_2d)

def seidel_2d_triton(A, N, TSTEPS):
    BLOCK_SIZE_I = 16
    BLOCK_SIZE_J = 16
    
    grid_i = triton.cdiv(N - 2, BLOCK_SIZE_I)
    grid_j = triton.cdiv(N - 2, BLOCK_SIZE_J)
    
    grid = (1, grid_i, grid_j)
    
    seidel_2d_kernel[grid](
        A, N, TSTEPS,
        BLOCK_SIZE_I=BLOCK_SIZE_I,
        BLOCK_SIZE_J=BLOCK_SIZE_J
    )