import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_start = pid_i * BLOCK_SIZE + 1
    j_start = pid_j * BLOCK_SIZE + 1
    
    i_offsets = tl.arange(0, BLOCK_SIZE)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for t in range(TSTEPS):
        i_indices = i_start + i_offsets
        j_indices = j_start + j_offsets
        
        i_mask = i_indices < (N - 1)
        j_mask = j_indices < (N - 1)
        
        # Create 2D mask
        mask = i_mask[:, None] & j_mask[None, :]
        
        # Load 9-point stencil values
        stencil_sum = 0.0
        for di in range(-1, 2):
            for dj in range(-1, 2):
                row_idx = i_indices[:, None] + di
                col_idx = j_indices[None, :] + dj
                addr = row_idx * N + col_idx
                val = tl.load(A_ptr + addr, mask=mask, other=0.0)
                stencil_sum = stencil_sum + val
        
        result = stencil_sum / 9.0
        
        # Store result
        store_addr = i_indices[:, None] * N + j_indices[None, :]
        tl.store(A_ptr + store_addr, result, mask=mask)

def seidel_2d_triton(A, N, TSTEPS):
    BLOCK_SIZE = 16
    
    grid_i = triton.cdiv(N - 2, BLOCK_SIZE)
    grid_j = triton.cdiv(N - 2, BLOCK_SIZE)
    
    seidel_2d_kernel[(grid_i, grid_j)](A, N, TSTEPS, BLOCK_SIZE)