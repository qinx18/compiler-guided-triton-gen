import triton
import triton.language as tl
import torch

@triton.jit
def atax_kernel(A_ptr, x_ptr, tmp_ptr, y_ptr, 
                M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get column offsets for vectorized operations
    col_offsets = tl.arange(0, BLOCK_SIZE)
    col_mask = col_offsets < N
    
    # Initialize y[j] = 0 (column-wise accumulator)
    y_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Iterate over rows (M dimension)
    for i in range(M):
        # Load row i of matrix A (coalesced access)
        a_row = tl.load(A_ptr + i * N + col_offsets, mask=col_mask, other=0.0)
        
        # Load x vector for row reduction
        x_vals = tl.load(x_ptr + col_offsets, mask=col_mask, other=0.0)
        
        # Compute tmp[i] = sum(A[i][j] * x[j]) - row reduction
        tmp_val = tl.sum(a_row * x_vals)
        tl.store(tmp_ptr + i, tmp_val)
        
        # Accumulate y[j] += A[i][j] * tmp[i] - column accumulation
        y_acc += a_row * tmp_val
    
    # Store final y values
    tl.store(y_ptr + col_offsets, y_acc, mask=col_mask)

def atax_triton(A, tmp, x, y, M, N):
    # Zero initialize y
    y.zero_()
    
    # Determine block size based on N
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    # Launch single kernel instance
    grid = (1,)
    
    atax_kernel[grid](
        A, x, tmp, y,
        M=M, N=N, BLOCK_SIZE=BLOCK_SIZE
    )