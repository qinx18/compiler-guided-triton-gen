import triton
import triton.language as tl
import torch

@triton.jit
def seidel_2d_kernel(A_ptr, N, TSTEPS, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    # Pre-compute offsets once
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for t in range(TSTEPS):
        for i in range(1, N - 1):
            # Calculate starting column for this block
            j_start = pid * BLOCK_SIZE + 1
            j_offsets = j_start + offsets
            mask = j_offsets < (N - 1)
            
            # Calculate base addresses for each row
            row_above = (i - 1) * N
            row_current = i * N
            row_below = (i + 1) * N
            
            # Load the 9-point stencil values
            # Row above (i-1)
            val_tl = tl.load(A_ptr + row_above + j_offsets - 1, mask=mask)
            val_tc = tl.load(A_ptr + row_above + j_offsets, mask=mask)
            val_tr = tl.load(A_ptr + row_above + j_offsets + 1, mask=mask)
            
            # Current row (i)
            val_ml = tl.load(A_ptr + row_current + j_offsets - 1, mask=mask)
            val_mc = tl.load(A_ptr + row_current + j_offsets, mask=mask)
            val_mr = tl.load(A_ptr + row_current + j_offsets + 1, mask=mask)
            
            # Row below (i+1)
            val_bl = tl.load(A_ptr + row_below + j_offsets - 1, mask=mask)
            val_bc = tl.load(A_ptr + row_below + j_offsets, mask=mask)
            val_br = tl.load(A_ptr + row_below + j_offsets + 1, mask=mask)
            
            # Compute the average of the 9 points
            result = (val_tl + val_tc + val_tr + 
                     val_ml + val_mc + val_mr + 
                     val_bl + val_bc + val_br) / 9.0
            
            # Store the result back
            tl.store(A_ptr + row_current + j_offsets, result, mask=mask)

def seidel_2d_triton(A, N, TSTEPS):
    inner_cols = N - 2
    if inner_cols <= 0:
        return
    
    BLOCK_SIZE = 64
    grid_size = triton.cdiv(inner_cols, BLOCK_SIZE)
    
    seidel_2d_kernel[(grid_size,)](
        A, N, TSTEPS, BLOCK_SIZE
    )