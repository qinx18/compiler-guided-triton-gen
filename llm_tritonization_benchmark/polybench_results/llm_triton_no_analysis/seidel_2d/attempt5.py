import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A_ptr, N: tl.constexpr, i: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    j_coords = block_start + offsets + 1
    
    mask = j_coords <= N - 2
    
    idx_i_minus_1_j_minus_1 = (i - 1) * N + (j_coords - 1)
    idx_i_minus_1_j = (i - 1) * N + j_coords
    idx_i_minus_1_j_plus_1 = (i - 1) * N + (j_coords + 1)
    
    idx_i_j_minus_1 = i * N + (j_coords - 1)
    idx_i_j = i * N + j_coords
    idx_i_j_plus_1 = i * N + (j_coords + 1)
    
    idx_i_plus_1_j_minus_1 = (i + 1) * N + (j_coords - 1)
    idx_i_plus_1_j = (i + 1) * N + j_coords
    idx_i_plus_1_j_plus_1 = (i + 1) * N + (j_coords + 1)
    
    val_i_minus_1_j_minus_1 = tl.load(A_ptr + idx_i_minus_1_j_minus_1, mask=mask)
    val_i_minus_1_j = tl.load(A_ptr + idx_i_minus_1_j, mask=mask)
    val_i_minus_1_j_plus_1 = tl.load(A_ptr + idx_i_minus_1_j_plus_1, mask=mask)
    
    val_i_j_minus_1 = tl.load(A_ptr + idx_i_j_minus_1, mask=mask)
    val_i_j = tl.load(A_ptr + idx_i_j, mask=mask)
    val_i_j_plus_1 = tl.load(A_ptr + idx_i_j_plus_1, mask=mask)
    
    val_i_plus_1_j_minus_1 = tl.load(A_ptr + idx_i_plus_1_j_minus_1, mask=mask)
    val_i_plus_1_j = tl.load(A_ptr + idx_i_plus_1_j, mask=mask)
    val_i_plus_1_j_plus_1 = tl.load(A_ptr + idx_i_plus_1_j_plus_1, mask=mask)
    
    result = (val_i_minus_1_j_minus_1 + val_i_minus_1_j + val_i_minus_1_j_plus_1 +
             val_i_j_minus_1 + val_i_j + val_i_j_plus_1 +
             val_i_plus_1_j_minus_1 + val_i_plus_1_j + val_i_plus_1_j_plus_1) / 9.0
    
    tl.store(A_ptr + idx_i_j, result, mask=mask)

def seidel_2d_triton(A, N, TSTEPS):
    BLOCK_SIZE = 256
    
    for t in range(TSTEPS):
        for i in range(1, N - 1):
            num_elements = N - 2
            grid_size = triton.cdiv(num_elements, BLOCK_SIZE)
            seidel_2d_kernel[(grid_size,)](A, N, i, BLOCK_SIZE)