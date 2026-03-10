import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A_ptr, N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for t in range(TSTEPS):
        for i in range(1, N - 1):
            indices = block_start + offsets
            j_coords = indices + 1
            
            mask = (indices < N - 2) & (j_coords >= 1) & (j_coords <= N - 2)
            
            if tl.sum(mask.to(tl.int32)) == 0:
                continue
                
            # Calculate memory addresses
            addr_center = i * N + j_coords
            addr_top = (i - 1) * N + j_coords
            addr_bottom = (i + 1) * N + j_coords
            
            # Load 9 neighbors
            top_left = tl.load(A_ptr + addr_top - 1, mask=mask, other=0.0)
            top_center = tl.load(A_ptr + addr_top, mask=mask, other=0.0)
            top_right = tl.load(A_ptr + addr_top + 1, mask=mask, other=0.0)
            
            center_left = tl.load(A_ptr + addr_center - 1, mask=mask, other=0.0)
            center = tl.load(A_ptr + addr_center, mask=mask, other=0.0)
            center_right = tl.load(A_ptr + addr_center + 1, mask=mask, other=0.0)
            
            bottom_left = tl.load(A_ptr + addr_bottom - 1, mask=mask, other=0.0)
            bottom_center = tl.load(A_ptr + addr_bottom, mask=mask, other=0.0)
            bottom_right = tl.load(A_ptr + addr_bottom + 1, mask=mask, other=0.0)
            
            # Compute average
            result = (top_left + top_center + top_right +
                     center_left + center + center_right +
                     bottom_left + bottom_center + bottom_right) / 9.0
            
            # Store result
            tl.store(A_ptr + addr_center, result, mask=mask)

def seidel_2d_triton(A, N, TSTEPS):
    BLOCK_SIZE = 32
    grid_size = triton.cdiv(N - 2, BLOCK_SIZE)
    
    seidel_2d_kernel[(grid_size,)](
        A, N, TSTEPS, BLOCK_SIZE
    )