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
        
        # for (j = 0; j < i; j++)
        #     x[i] -= L[i][j] * x[j];
        for j in range(i):
            L_idx = i * N + j
            L_val = tl.load(L_ptr + L_idx)
            x_j = tl.load(x_ptr + j)
            x_val -= L_val * x_j
        
        # x[i] = x[i] / L[i][i];
        L_diag_idx = i * N + i
        L_diag = tl.load(L_ptr + L_diag_idx)
        x_val = x_val / L_diag
        
        tl.store(x_ptr + i, x_val)

def trisolv_triton(L, b, x, N):
    # Launch single thread since computation is inherently sequential
    grid = (1,)
    
    trisolv_kernel[grid](
        L, b, x,
        N=N
    )