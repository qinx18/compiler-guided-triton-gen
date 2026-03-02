import triton
import triton.language as tl
import torch

@triton.jit
def jacobi_2d_kernel(A, B, N, TSTEPS, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_offsets = tl.arange(0, BLOCK_SIZE_I)
    j_offsets = tl.arange(0, BLOCK_SIZE_J)
    
    base_i = pid_i * BLOCK_SIZE_I + 1
    base_j = pid_j * BLOCK_SIZE_J + 1
    
    for t in range(TSTEPS):
        # First pass: A -> B
        i_indices = base_i + i_offsets
        j_indices = base_j + j_offsets
        
        i_mask = i_indices < (N - 1)
        j_mask = j_indices < (N - 1)
        
        i_expanded = i_indices[:, None]
        j_expanded = j_indices[None, :]
        
        mask = i_mask[:, None] & j_mask[None, :]
        
        center = tl.load(A + i_expanded * N + j_expanded, mask=mask)
        left = tl.load(A + i_expanded * N + (j_expanded - 1), mask=mask)
        right = tl.load(A + i_expanded * N + (j_expanded + 1), mask=mask)
        up = tl.load(A + (i_expanded - 1) * N + j_expanded, mask=mask)
        down = tl.load(A + (i_expanded + 1) * N + j_expanded, mask=mask)
        
        result = 0.2 * (center + left + right + up + down)
        tl.store(B + i_expanded * N + j_expanded, result, mask=mask)
        
        # Second pass: B -> A
        center = tl.load(B + i_expanded * N + j_expanded, mask=mask)
        left = tl.load(B + i_expanded * N + (j_expanded - 1), mask=mask)
        right = tl.load(B + i_expanded * N + (j_expanded + 1), mask=mask)
        up = tl.load(B + (i_expanded - 1) * N + j_expanded, mask=mask)
        down = tl.load(B + (i_expanded + 1) * N + j_expanded, mask=mask)
        
        result = 0.2 * (center + left + right + up + down)
        tl.store(A + i_expanded * N + j_expanded, result, mask=mask)

def jacobi_2d_triton(A, B, N, TSTEPS):
    BLOCK_SIZE_I = 16
    BLOCK_SIZE_J = 16
    
    grid_i = triton.cdiv(N - 2, BLOCK_SIZE_I)
    grid_j = triton.cdiv(N - 2, BLOCK_SIZE_J)
    
    jacobi_2d_kernel[(grid_i, grid_j)](
        A, B, N, TSTEPS,
        BLOCK_SIZE_I=BLOCK_SIZE_I,
        BLOCK_SIZE_J=BLOCK_SIZE_J
    )