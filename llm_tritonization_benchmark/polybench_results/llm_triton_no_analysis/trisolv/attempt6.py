import triton
import triton.language as tl
import torch

@triton.jit
def trisolv_kernel(L_ptr, b_ptr, x_ptr, N: tl.constexpr):
    # Each program handles one row
    i = tl.program_id(0)
    
    if i >= N:
        return
    
    # Load b[i] into x[i]
    x_i = tl.load(b_ptr + i)
    
    # Sequential reduction for j from 0 to i-1
    for j in range(i):
        L_ij = tl.load(L_ptr + i * N + j)
        x_j = tl.load(x_ptr + j)
        x_i = x_i - L_ij * x_j
    
    # Divide by L[i][i]
    L_ii = tl.load(L_ptr + i * N + i)
    x_i = x_i / L_ii
    
    # Store result
    tl.store(x_ptr + i, x_i)

def trisolv_triton(L, b, x, N):
    # Copy b to x first
    x.copy_(b)
    
    # Launch kernel sequentially for each row to maintain dependencies
    for i in range(N):
        trisolv_kernel[(1,)](
            L, b, x, N,
            num_warps=1
        )