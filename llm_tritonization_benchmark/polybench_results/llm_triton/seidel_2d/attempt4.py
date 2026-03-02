import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A, N, TSTEPS, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr):
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
    
    for t in range(TSTEPS):
        i_expanded = i_indices[:, None]
        j_expanded = j_indices[None, :]
        
        mask = i_mask[:, None] & j_mask[None, :]
        
        # Create pointers for the 9-point stencil
        base_ptr = A + i_expanded * N + j_expanded
        
        # Load 9-point stencil
        v00 = tl.load(base_ptr + (-1) * N + (-1), mask=mask, other=0.0)
        v01 = tl.load(base_ptr + (-1) * N + 0, mask=mask, other=0.0)
        v02 = tl.load(base_ptr + (-1) * N + 1, mask=mask, other=0.0)
        v10 = tl.load(base_ptr + 0 * N + (-1), mask=mask, other=0.0)
        v11 = tl.load(base_ptr + 0 * N + 0, mask=mask, other=0.0)
        v12 = tl.load(base_ptr + 0 * N + 1, mask=mask, other=0.0)
        v20 = tl.load(base_ptr + 1 * N + (-1), mask=mask, other=0.0)
        v21 = tl.load(base_ptr + 1 * N + 0, mask=mask, other=0.0)
        v22 = tl.load(base_ptr + 1 * N + 1, mask=mask, other=0.0)
        
        # Compute average
        result = (v00 + v01 + v02 + v10 + v11 + v12 + v20 + v21 + v22) / 9.0
        
        # Store result
        tl.store(base_ptr, result, mask=mask)

def seidel_2d_triton(A, N, TSTEPS):
    BLOCK_SIZE_I = 16
    BLOCK_SIZE_J = 16
    
    grid_i = triton.cdiv(N - 2, BLOCK_SIZE_I)
    grid_j = triton.cdiv(N - 2, BLOCK_SIZE_J)
    
    grid = (grid_i, grid_j)
    seidel_2d_kernel[grid](
        A, N, TSTEPS,
        BLOCK_SIZE_I=BLOCK_SIZE_I,
        BLOCK_SIZE_J=BLOCK_SIZE_J
    )