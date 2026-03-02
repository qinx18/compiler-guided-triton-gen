import triton
import triton.language as tl
import torch

@triton.jit
def durbin_kernel(r_ptr, y_ptr, z_ptr, N: tl.constexpr):
    # Initialize
    tl.store(y_ptr, -tl.load(r_ptr))
    beta = 1.0
    alpha = -tl.load(r_ptr)
    
    # Main loop - must be sequential due to dependencies
    for k in range(1, N):
        # Update beta
        beta = (1.0 - alpha * alpha) * beta
        
        # Compute sum using vectorized operations
        sum_val = 0.0
        BLOCK_SIZE = 128
        offsets = tl.arange(0, BLOCK_SIZE)
        
        for block_start in range(0, k, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < k
            
            r_indices = k - current_offsets - 1
            r_vals = tl.load(r_ptr + r_indices, mask=mask, other=0.0)
            y_vals = tl.load(y_ptr + current_offsets, mask=mask, other=0.0)
            
            products = r_vals * y_vals
            sum_val += tl.sum(tl.where(mask, products, 0.0))
        
        # Update alpha
        alpha = -(tl.load(r_ptr + k) + sum_val) / beta
        
        # Update z array
        for block_start in range(0, k, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < k
            
            y_vals = tl.load(y_ptr + current_offsets, mask=mask, other=0.0)
            reverse_indices = k - current_offsets - 1
            y_reverse_vals = tl.load(y_ptr + reverse_indices, mask=mask, other=0.0)
            
            z_vals = y_vals + alpha * y_reverse_vals
            tl.store(z_ptr + current_offsets, z_vals, mask=mask)
        
        # Copy z to y
        for block_start in range(0, k, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < k
            
            z_vals = tl.load(z_ptr + current_offsets, mask=mask, other=0.0)
            tl.store(y_ptr + current_offsets, z_vals, mask=mask)
        
        # Set y[k] = alpha
        tl.store(y_ptr + k, alpha)

def durbin_triton(r, y, z, N):
    grid = (1,)
    durbin_kernel[grid](r, y, z, N)