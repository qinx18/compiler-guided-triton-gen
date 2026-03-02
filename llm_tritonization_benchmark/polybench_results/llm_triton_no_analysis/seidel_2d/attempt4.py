import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A_ptr, N, TSTEPS, BLOCK_SIZE: tl.constexpr):
    # Get column block ID
    pid = tl.program_id(0)
    
    # Pre-compute offsets once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Sequential time steps
    for t in range(TSTEPS):
        # Sequential rows
        for i in range(1, N - 1):
            # Calculate column indices for this block
            j = pid * BLOCK_SIZE + offsets + 1
            mask = j < (N - 1)
            
            # Calculate all stencil positions
            center = i * N + j
            
            # Load 9-point stencil
            val_tl = tl.load(A_ptr + (i-1) * N + (j-1), mask=mask)
            val_tc = tl.load(A_ptr + (i-1) * N + j, mask=mask)
            val_tr = tl.load(A_ptr + (i-1) * N + (j+1), mask=mask)
            
            val_ml = tl.load(A_ptr + i * N + (j-1), mask=mask)
            val_mc = tl.load(A_ptr + center, mask=mask)
            val_mr = tl.load(A_ptr + i * N + (j+1), mask=mask)
            
            val_bl = tl.load(A_ptr + (i+1) * N + (j-1), mask=mask)
            val_bc = tl.load(A_ptr + (i+1) * N + j, mask=mask)
            val_br = tl.load(A_ptr + (i+1) * N + (j+1), mask=mask)
            
            # Compute stencil result
            result = (val_tl + val_tc + val_tr + val_ml + val_mc + val_mr + val_bl + val_bc + val_br) / 9.0
            
            # Store result
            tl.store(A_ptr + center, result, mask=mask)

def seidel_2d_triton(A, N, TSTEPS):
    inner_cols = N - 2
    if inner_cols <= 0:
        return
    
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(inner_cols, BLOCK_SIZE)
    
    seidel_2d_kernel[(grid_size,)](
        A, N, TSTEPS, BLOCK_SIZE
    )