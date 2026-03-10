import triton
import triton.language as tl
import torch

@triton.jit
def mvt_kernel(A_ptr, x1_ptr, x2_ptr, y_1_ptr, y_2_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate which row this program handles
    row = pid
    
    if row < N:
        # First loop: x1[i] = x1[i] + A[i][j] * y_1[j]
        x1_val = tl.load(x1_ptr + row)
        
        # Process in blocks
        offsets = tl.arange(0, BLOCK_SIZE)
        for block_start in range(0, N, BLOCK_SIZE):
            j_offsets = block_start + offsets
            mask = j_offsets < N
            
            # Load A[row, j_offsets] and y_1[j_offsets]
            a_offsets = row * N + j_offsets
            a_vals = tl.load(A_ptr + a_offsets, mask=mask, other=0.0)
            y1_vals = tl.load(y_1_ptr + j_offsets, mask=mask, other=0.0)
            
            # Compute A[i][j] * y_1[j] and accumulate
            products = a_vals * y1_vals
            x1_val = x1_val + tl.sum(products)
        
        # Store updated x1[row]
        tl.store(x1_ptr + row, x1_val)
        
        # Second loop: x2[i] = x2[i] + A[j][i] * y_2[j]
        x2_val = tl.load(x2_ptr + row)
        
        # Process in blocks
        for block_start in range(0, N, BLOCK_SIZE):
            j_offsets = block_start + offsets
            mask = j_offsets < N
            
            # Load A[j_offsets, row] and y_2[j_offsets]
            a_offsets = j_offsets * N + row
            a_vals = tl.load(A_ptr + a_offsets, mask=mask, other=0.0)
            y2_vals = tl.load(y_2_ptr + j_offsets, mask=mask, other=0.0)
            
            # Compute A[j][i] * y_2[j] and accumulate
            products = a_vals * y2_vals
            x2_val = x2_val + tl.sum(products)
        
        # Store updated x2[row]
        tl.store(x2_ptr + row, x2_val)

def mvt_triton(A, x1, x2, y_1, y_2, N):
    BLOCK_SIZE = 32
    grid = (N,)
    
    mvt_kernel[grid](
        A, x1, x2, y_1, y_2,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )