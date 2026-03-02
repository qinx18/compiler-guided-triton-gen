import triton
import triton.language as tl
import torch

@triton.jit
def durbin_kernel(r_ptr, y_ptr, z_ptr, k, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < k
    
    # Calculate beta for current k
    beta = 1.0
    alpha = -tl.load(r_ptr)
    
    for iter_k in range(1, k + 1):
        beta = (1.0 - alpha * alpha) * beta
        sum_val = 0.0
        for i in range(iter_k):
            r_val = tl.load(r_ptr + iter_k - i - 1)
            y_val = tl.load(y_ptr + i)
            sum_val += r_val * y_val
        alpha = -(tl.load(r_ptr + iter_k) + sum_val) / beta
    
    # Update z[i] = y[i] + alpha * y[k-i-1] for valid indices
    y_vals = tl.load(y_ptr + indices, mask=mask)
    y_rev_indices = k - indices - 1
    y_rev_vals = tl.load(y_ptr + y_rev_indices, mask=mask)
    z_vals = y_vals + alpha * y_rev_vals
    tl.store(z_ptr + indices, z_vals, mask=mask)

def durbin_triton(r, y, z, N):
    # Initialize
    y[0] = -r[0]
    
    BLOCK_SIZE = 128
    
    # Sequential over k
    for k in range(1, N):
        # Launch kernel for current k
        grid = (triton.cdiv(k, BLOCK_SIZE),)
        durbin_kernel[grid](r, y, z, k, N, BLOCK_SIZE)
        
        # Copy z to y and calculate alpha for y[k]
        beta = 1.0
        alpha = -r[0].item()
        
        for iter_k in range(1, k + 1):
            beta = (1.0 - alpha * alpha) * beta
            sum_val = 0.0
            for i in range(iter_k):
                sum_val += r[iter_k - i - 1].item() * y[i].item()
            alpha = -(r[iter_k].item() + sum_val) / beta
        
        # Copy z to y for positions 0 to k-1
        y[:k] = z[:k]
        y[k] = alpha