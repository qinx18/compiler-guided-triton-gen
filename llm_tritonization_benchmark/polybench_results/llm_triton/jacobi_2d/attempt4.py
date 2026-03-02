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
    
    i_indices = base_i + i_offsets
    j_indices = base_j + j_offsets
    
    i_mask = i_indices < (N - 1)
    j_mask = j_indices < (N - 1)
    
    for t in range(TSTEPS):
        # First pass: A -> B
        for ii in range(BLOCK_SIZE_I):
            i_idx = base_i + ii
            if i_idx < (N - 1):
                for jj in range(BLOCK_SIZE_J):
                    j_idx = base_j + jj
                    if j_idx < (N - 1):
                        center = tl.load(A + i_idx * N + j_idx)
                        left = tl.load(A + i_idx * N + (j_idx - 1))
                        right = tl.load(A + i_idx * N + (j_idx + 1))
                        up = tl.load(A + (i_idx - 1) * N + j_idx)
                        down = tl.load(A + (i_idx + 1) * N + j_idx)
                        result = 0.2 * (center + left + right + up + down)
                        tl.store(B + i_idx * N + j_idx, result)
        
        # Second pass: B -> A
        for ii in range(BLOCK_SIZE_I):
            i_idx = base_i + ii
            if i_idx < (N - 1):
                for jj in range(BLOCK_SIZE_J):
                    j_idx = base_j + jj
                    if j_idx < (N - 1):
                        center = tl.load(B + i_idx * N + j_idx)
                        left = tl.load(B + i_idx * N + (j_idx - 1))
                        right = tl.load(B + i_idx * N + (j_idx + 1))
                        up = tl.load(B + (i_idx - 1) * N + j_idx)
                        down = tl.load(B + (i_idx + 1) * N + j_idx)
                        result = 0.2 * (center + left + right + up + down)
                        tl.store(A + i_idx * N + j_idx, result)

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