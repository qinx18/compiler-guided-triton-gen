import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_start = pid_i * BLOCK_SIZE + 1
    j_start = pid_j * BLOCK_SIZE + 1
    
    i_offsets = i_start + tl.arange(0, BLOCK_SIZE)
    j_offsets = j_start + tl.arange(0, BLOCK_SIZE)
    
    i_mask = i_offsets < (N - 1)
    j_mask = j_offsets < (N - 1)
    
    for t in range(TSTEPS):
        for i_idx in range(BLOCK_SIZE):
            i = i_start + i_idx
            if i >= (N - 1):
                break
            
            j_indices = j_start + tl.arange(0, BLOCK_SIZE)
            mask = j_mask
            
            # Load 9-point stencil values
            addr_base = (i-1) * N + j_indices
            val1 = tl.load(A_ptr + addr_base - 1, mask=mask, other=0.0)
            val2 = tl.load(A_ptr + addr_base, mask=mask, other=0.0)
            val3 = tl.load(A_ptr + addr_base + 1, mask=mask, other=0.0)
            
            addr_base = i * N + j_indices
            val4 = tl.load(A_ptr + addr_base - 1, mask=mask, other=0.0)
            val5 = tl.load(A_ptr + addr_base, mask=mask, other=0.0)
            val6 = tl.load(A_ptr + addr_base + 1, mask=mask, other=0.0)
            
            addr_base = (i+1) * N + j_indices
            val7 = tl.load(A_ptr + addr_base - 1, mask=mask, other=0.0)
            val8 = tl.load(A_ptr + addr_base, mask=mask, other=0.0)
            val9 = tl.load(A_ptr + addr_base + 1, mask=mask, other=0.0)
            
            result = (val1 + val2 + val3 + val4 + val5 + val6 + val7 + val8 + val9) / 9.0
            
            tl.store(A_ptr + i * N + j_indices, result, mask=mask)

def seidel_2d_triton(A, N, TSTEPS):
    BLOCK_SIZE = 16
    
    grid_i = triton.cdiv(N - 2, BLOCK_SIZE)
    grid_j = triton.cdiv(N - 2, BLOCK_SIZE)
    
    seidel_2d_kernel[(grid_i, grid_j)](A, N, TSTEPS, BLOCK_SIZE)