import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    for t in range(TSTEPS):
        for i in range(1, N - 1):
            j_start = pid * BLOCK_SIZE + 1
            j_offsets = tl.arange(0, BLOCK_SIZE)
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
    grid = triton.cdiv(N - 2, BLOCK_SIZE)
    
    seidel_2d_kernel[(grid,)](
        A, N, TSTEPS, BLOCK_SIZE
    )