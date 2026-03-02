import triton
import triton.language as tl
import torch

@triton.jit
def durbin_kernel(r_ptr, y_ptr, z_ptr, N: tl.constexpr):
    # Initialize
    r0 = tl.load(r_ptr)
    tl.store(y_ptr, -r0)
    beta = 1.0
    alpha = -r0
    
    # Main loop - must be sequential due to dependencies
    for k in range(1, N):
        # Update beta
        beta = (1.0 - alpha * alpha) * beta
        
        # Compute sum
        sum_val = 0.0
        for i in range(k):
            r_val = tl.load(r_ptr + (k - i - 1))
            y_val = tl.load(y_ptr + i)
            sum_val += r_val * y_val
        
        # Update alpha
        rk = tl.load(r_ptr + k)
        alpha = -(rk + sum_val) / beta
        
        # Update z array
        for i in range(k):
            y_i = tl.load(y_ptr + i)
            y_rev = tl.load(y_ptr + (k - i - 1))
            z_val = y_i + alpha * y_rev
            tl.store(z_ptr + i, z_val)
        
        # Copy z to y
        for i in range(k):
            z_val = tl.load(z_ptr + i)
            tl.store(y_ptr + i, z_val)
        
        # Set y[k] = alpha
        tl.store(y_ptr + k, alpha)

def durbin_triton(r, y, z, N):
    grid = (1,)
    durbin_kernel[grid](r, y, z, N)