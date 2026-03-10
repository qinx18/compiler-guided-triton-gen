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
        for i_idx in range(BLOCK_SIZE):
            i = i_start + i_idx
            if i < N - 1:
                j_indices = j_start + j_offsets
                mask = j_indices < N - 1
                
                # Calculate addresses for the 9-point stencil
                center_addr = i * N + j_indices
                top_addr = (i - 1) * N + j_indices
                bottom_addr = (i + 1) * N + j_indices
                
                # Load 9 neighbors
                top_left = tl.load(A_ptr + top_addr - 1, mask=mask, other=0.0)
                top_center = tl.load(A_ptr + top_addr, mask=mask, other=0.0)
                top_right = tl.load(A_ptr + top_addr + 1, mask=mask, other=0.0)
                
                center_left = tl.load(A_ptr + center_addr - 1, mask=mask, other=0.0)
                center = tl.load(A_ptr + center_addr, mask=mask, other=0.0)
                center_right = tl.load(A_ptr + center_addr + 1, mask=mask, other=0.0)
                
                bottom_left = tl.load(A_ptr + bottom_addr - 1, mask=mask, other=0.0)
                bottom_center = tl.load(A_ptr + bottom_addr, mask=mask, other=0.0)
                bottom_right = tl.load(A_ptr + bottom_addr + 1, mask=mask, other=0.0)
                
                # Compute average
                result = (top_left + top_center + top_right +
                         center_left + center + center_right +
                         bottom_left + bottom_center + bottom_right) / 9.0
                
                # Store result
                tl.store(A_ptr + center_addr, result, mask=mask)

def seidel_2d_triton(A, N, TSTEPS):
    BLOCK_SIZE = 32
    grid_i = triton.cdiv(N - 2, BLOCK_SIZE)
    grid_j = triton.cdiv(N - 2, BLOCK_SIZE)
    
    seidel_2d_kernel[(grid_i, grid_j)](
        A, N, TSTEPS, BLOCK_SIZE
    )