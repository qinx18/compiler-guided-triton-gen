import triton
import triton.language as tl
import torch

@triton.jit
def trisolv_kernel(L_ptr, b_ptr, x_ptr, N: tl.constexpr, stride_L: tl.constexpr, i: tl.constexpr):
    # Initialize x[i] = b[i]
    b_val = tl.load(b_ptr + i)
    x_val = b_val
    
    # Compute sum of L[i][j] * x[j] for j < i
    for j in range(i):
        L_ij = tl.load(L_ptr + i * stride_L + j)
        x_j = tl.load(x_ptr + j)
        x_val = x_val - L_ij * x_j
    
    # Divide by diagonal element L[i][i]
    L_ii = tl.load(L_ptr + i * stride_L + i)
    x_val = x_val / L_ii
    
    # Store result
    tl.store(x_ptr + i, x_val)

def trisolv_triton(L, b, x, N):
    # Process each row sequentially to maintain dependency
    for i in range(N):
        grid = (1,)
        trisolv_kernel[grid](
            L, b, x,
            N=N,
            stride_L=L.stride(0),
            i=i
        )