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
    
    # Calculate beta for iteration k using previous alpha from y[k-1]
    if k == 1:
        prev_alpha = -tl.load(r_ptr)  # alpha from k=0
    else:
        prev_alpha = tl.load(y_ptr + k - 1)  # alpha from k-1
    
    # Calculate beta recursively
    beta = 1.0
    current_alpha = -tl.load(r_ptr)
    for iter_k in range(1, k):
        beta = (1.0 - current_alpha * current_alpha) * beta
        iter_sum = 0.0
        for i in range(iter_k):
            r_val = tl.load(r_ptr + iter_k - i - 1)
            y_val = tl.load(y_ptr + i)
            iter_sum += r_val * y_val
        current_alpha = -(tl.load(r_ptr + iter_k) + iter_sum) / beta
    
    beta = (1.0 - current_alpha * current_alpha) * beta
    
    # Calculate sum for current k
    sum_val = 0.0
    for i in range(k):
        r_val = tl.load(r_ptr + k - i - 1)
        y_val = tl.load(y_ptr + i)
        sum_val += r_val * y_val
    
    # Calculate alpha for current k
    r_k = tl.load(r_ptr + k)
    alpha = -(r_k + sum_val) / beta
    
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
    
    # Sequential over k, parallel over i
    for k in range(1, N):
        # Launch kernel for current k
        grid = (triton.cdiv(k, BLOCK_SIZE),)
        durbin_kernel[grid](r, y, z, k, N, BLOCK_SIZE)
        
        # Copy z to y for positions 0 to k-1
        y[:k] = z[:k]
        
        # Calculate alpha for y[k] using current state
        beta = 1.0
        current_alpha = -r[0].item()
        
        for iter_k in range(1, k):
            beta = (1.0 - current_alpha * current_alpha) * beta
            iter_sum = 0.0
            for i in range(iter_k):
                iter_sum += r[iter_k - i - 1].item() * y[i].item()
            current_alpha = -(r[iter_k].item() + iter_sum) / beta
        
        beta = (1.0 - current_alpha * current_alpha) * beta
        
        sum_val = 0.0
        for i in range(k):
            sum_val += r[k - i - 1].item() * y[i].item()
        
        alpha = -(r[k].item() + sum_val) / beta
        y[k] = alpha