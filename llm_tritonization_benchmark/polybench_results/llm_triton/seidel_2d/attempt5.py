import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A, N, t, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = pid * BLOCK_SIZE + offsets
    
    # Convert linear index to i, j coordinates
    i = idx // (N - 2) + 1
    j = idx % (N - 2) + 1
    
    # Mask for valid indices
    mask = (idx < (N - 2) * (N - 2)) & (i >= 1) & (i <= N - 2) & (j >= 1) & (j <= N - 2)
    
    # Calculate linear indices for the 9-point stencil
    base_idx = i * N + j
    
    # Load 9-point stencil values
    v00 = tl.load(A + base_idx - N - 1, mask=mask, other=0.0)
    v01 = tl.load(A + base_idx - N, mask=mask, other=0.0)
    v02 = tl.load(A + base_idx - N + 1, mask=mask, other=0.0)
    v10 = tl.load(A + base_idx - 1, mask=mask, other=0.0)
    v11 = tl.load(A + base_idx, mask=mask, other=0.0)
    v12 = tl.load(A + base_idx + 1, mask=mask, other=0.0)
    v20 = tl.load(A + base_idx + N - 1, mask=mask, other=0.0)
    v21 = tl.load(A + base_idx + N, mask=mask, other=0.0)
    v22 = tl.load(A + base_idx + N + 1, mask=mask, other=0.0)
    
    # Compute average
    result = (v00 + v01 + v02 + v10 + v11 + v12 + v20 + v21 + v22) / 9.0
    
    # Store result
    tl.store(A + base_idx, result, mask=mask)

def seidel_2d_triton(A, N, TSTEPS):
    BLOCK_SIZE = 256
    
    total_elements = (N - 2) * (N - 2)
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    for t in range(TSTEPS):
        seidel_2d_kernel[grid](
            A, N, t,
            BLOCK_SIZE=BLOCK_SIZE
        )