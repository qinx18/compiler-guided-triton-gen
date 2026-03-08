import triton
import triton.language as tl
import torch

@triton.jit
def gesummv_kernel(A, B, tmp, x, y, alpha, beta, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID for the i dimension (parallelized)
    pid = tl.program_id(0)
    
    # Calculate the row index for this thread block
    i = pid
    
    # Initialize tmp[i] and y[i] to 0
    tmp_val = 0.0
    y_val = 0.0
    
    # Sequential loop over j dimension with blocking
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for j_start in range(0, N, BLOCK_SIZE):
        current_j_offsets = j_start + j_offsets
        mask = current_j_offsets < N
        
        # Load x[j] values
        x_vals = tl.load(x + current_j_offsets, mask=mask, other=0.0)
        
        # Load A[i][j] values
        a_offsets = i * N + current_j_offsets
        a_vals = tl.load(A + a_offsets, mask=mask, other=0.0)
        
        # Load B[i][j] values
        b_offsets = i * N + current_j_offsets
        b_vals = tl.load(B + b_offsets, mask=mask, other=0.0)
        
        # Accumulate tmp[i] = A[i][j] * x[j] + tmp[i]
        tmp_val += tl.sum(a_vals * x_vals)
        
        # Accumulate y[i] = B[i][j] * x[j] + y[i]
        y_val += tl.sum(b_vals * x_vals)
    
    # Final computation: y[i] = alpha * tmp[i] + beta * y[i]
    final_y = alpha * tmp_val + beta * y_val
    
    # Store results
    tl.store(tmp + i, tmp_val)
    tl.store(y + i, final_y)

def gesummv_triton(A, B, tmp, x, y, alpha, beta, N):
    # Define block size for the j dimension
    BLOCK_SIZE = 32
    
    # Number of thread blocks (one per row i)
    grid = (N,)
    
    # Launch kernel
    gesummv_kernel[grid](
        A, B, tmp, x, y, alpha, beta, N, BLOCK_SIZE
    )