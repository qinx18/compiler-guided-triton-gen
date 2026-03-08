import triton
import triton.language as tl
import torch

@triton.jit
def durbin_kernel(r_ptr, y_ptr, z_ptr, N: tl.constexpr):
    # Initialize y[0], beta, alpha
    r0 = tl.load(r_ptr)
    tl.store(y_ptr, -r0)
    
    beta = 1.0
    alpha = -r0
    
    # Main loop over k
    for k in range(1, N):
        # Update beta
        beta = (1.0 - alpha * alpha) * beta
        
        # Compute sum = sum(r[k-i-1] * y[i]) for i in [0, k)
        sum_val = 0.0
        for i in range(k):
            r_val = tl.load(r_ptr + (k - i - 1))
            y_val = tl.load(y_ptr + i)
            sum_val += r_val * y_val
        
        # Compute new alpha
        r_k = tl.load(r_ptr + k)
        alpha = -(r_k + sum_val) / beta
        
        # Update z[i] = y[i] + alpha * y[k-i-1] for i in [0, k)
        for i in range(k):
            y_i = tl.load(y_ptr + i)
            y_k_i_1 = tl.load(y_ptr + (k - i - 1))
            z_i = y_i + alpha * y_k_i_1
            tl.store(z_ptr + i, z_i)
        
        # Copy z[i] to y[i] for i in [0, k)
        for i in range(k):
            z_i = tl.load(z_ptr + i)
            tl.store(y_ptr + i, z_i)
        
        # Set y[k] = alpha
        tl.store(y_ptr + k, alpha)

def durbin_triton(r, y, z, N):
    grid = (1,)
    durbin_kernel[grid](r, y, z, N)