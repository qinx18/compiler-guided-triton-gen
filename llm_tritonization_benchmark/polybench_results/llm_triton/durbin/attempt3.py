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
    
    # Calculate beta for iteration k
    beta = 1.0
    if k > 1:
        # Need previous alpha values
        prev_alpha = -tl.load(r_ptr)
        prev_beta = 1.0
        for prev_k in range(1, k):
            prev_beta = (1.0 - prev_alpha * prev_alpha) * prev_beta
            prev_sum = 0.0
            for i in range(prev_k):
                r_val = tl.load(r_ptr + prev_k - i - 1)
                y_val = tl.load(y_ptr + i)
                prev_sum += r_val * y_val
            prev_alpha = -(tl.load(r_ptr + prev_k) + prev_sum) / prev_beta
        beta = (1.0 - prev_alpha * prev_alpha) * prev_beta
    else:
        r0 = tl.load(r_ptr)
        alpha_init = -r0
        beta = (1.0 - alpha_init * alpha_init) * 1.0
    
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
    valid_mask = mask > 0
    if tl.sum(valid_mask.to(tl.int32)) > 0:
        y_vals = tl.load(y_ptr + indices, mask=mask)
        y_rev_indices = k - indices - 1
        y_rev_mask = mask & (y_rev_indices >= 0)
        y_rev_vals = tl.load(y_ptr + y_rev_indices, mask=y_rev_mask)
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
        
        # Copy z to y sequentially
        for i in range(k):
            y[i] = z[i]
        
        # Calculate alpha again for y[k]
        beta = 1.0
        if k > 1:
            prev_alpha = -r[0].item()
            prev_beta = 1.0
            for prev_k in range(1, k):
                prev_beta = (1.0 - prev_alpha * prev_alpha) * prev_beta
                prev_sum = 0.0
                for i in range(prev_k):
                    prev_sum += r[prev_k - i - 1].item() * y[i].item()
                prev_alpha = -(r[prev_k].item() + prev_sum) / prev_beta
            beta = (1.0 - prev_alpha * prev_alpha) * prev_beta
        else:
            alpha_init = -r[0].item()
            beta = (1.0 - alpha_init * alpha_init) * 1.0
        
        sum_val = 0.0
        for i in range(k):
            sum_val += r[k - i - 1].item() * y[i].item()
        
        alpha = -(r[k].item() + sum_val) / beta
        y[k] = alpha