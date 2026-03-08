import triton
import triton.language as tl
import torch

@triton.jit
def trisolv_kernel(L_ptr, b_ptr, x_ptr, N: tl.constexpr):
    # Sequential implementation - no parallelization due to WAR dependencies
    for i in range(N):
        # x[i] = b[i]
        b_val = tl.load(b_ptr + i)
        x_val = b_val
        
        # for j in range(i): x[i] -= L[i][j] * x[j]
        for j in range(i):
            l_idx = i * N + j
            l_val = tl.load(L_ptr + l_idx)
            x_j = tl.load(x_ptr + j)
            x_val = x_val - l_val * x_j
        
        # x[i] = x[i] / L[i][i]
        l_diag_idx = i * N + i
        l_diag = tl.load(L_ptr + l_diag_idx)
        x_val = x_val / l_diag
        
        # Store result
        tl.store(x_ptr + i, x_val)

def trisolv_triton(L, b, x, N):
    grid = (1,)
    trisolv_kernel[grid](L, b, x, N)