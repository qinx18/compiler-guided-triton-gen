import triton
import triton.language as tl
import torch

@triton.jit
def atax_kernel(A_ptr, x_ptr, y_ptr, tmp_ptr, 
                M: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N
    
    # Initialize y[j] = 0
    y_acc = tl.zeros([BLOCK], dtype=tl.float32)
    
    # Outer loop over rows (M dimension)
    for i in range(M):
        # Load row A[i][:]
        a_row = tl.load(A_ptr + i * N + offsets, mask=mask, other=0.0)
        
        # Load x vector
        x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # Compute tmp[i] = sum(A[i][j] * x[j])
        tmp_val = tl.sum(a_row * x_vals)
        
        # Store tmp[i]
        tl.store(tmp_ptr + i, tmp_val)
        
        # Accumulate y[j] += A[i][j] * tmp[i]
        y_acc += a_row * tmp_val
    
    # Store final y values
    tl.store(y_ptr + offsets, y_acc, mask=mask)

def atax_triton(A, tmp, x, y, M, N):
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    grid = (1,)
    
    atax_kernel[grid](
        A, x, y, tmp,
        M=M, N=N, BLOCK=BLOCK_SIZE
    )