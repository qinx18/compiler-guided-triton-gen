import triton
import triton.language as tl
import torch

@triton.jit
def durbin_kernel(r_ptr, y_ptr, z_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Initialize
    r_0 = tl.load(r_ptr)
    tl.store(y_ptr, -r_0)
    
    beta = 1.0
    alpha = -r_0
    
    # Main loop - must be sequential
    for k in range(1, N):
        # Update beta
        beta = (1.0 - alpha * alpha) * beta
        
        # Compute sum
        sum_val = 0.0
        offsets = tl.arange(0, BLOCK_SIZE)
        
        for block_start in range(0, k, BLOCK_SIZE):
            actual_size = min(BLOCK_SIZE, k - block_start)
            i_indices = block_start + offsets
            r_indices = k - i_indices - 1
            
            mask = offsets < actual_size
            
            r_vals = tl.load(r_ptr + r_indices, mask=mask, other=0.0)
            y_vals = tl.load(y_ptr + i_indices, mask=mask, other=0.0)
            
            products = r_vals * y_vals
            block_sum = tl.sum(tl.where(mask, products, 0.0))
            sum_val += block_sum
        
        # Update alpha
        r_k = tl.load(r_ptr + k)
        alpha = -(r_k + sum_val) / beta
        
        # Update z[i] = y[i] + alpha * y[k-i-1]
        for block_start in range(0, k, BLOCK_SIZE):
            actual_size = min(BLOCK_SIZE, k - block_start)
            i_indices = block_start + offsets
            reverse_indices = k - i_indices - 1
            
            mask = offsets < actual_size
            
            y_i = tl.load(y_ptr + i_indices, mask=mask, other=0.0)
            y_reverse = tl.load(y_ptr + reverse_indices, mask=mask, other=0.0)
            
            z_vals = y_i + alpha * y_reverse
            tl.store(z_ptr + i_indices, z_vals, mask=mask)
        
        # Copy z[i] to y[i]
        for block_start in range(0, k, BLOCK_SIZE):
            actual_size = min(BLOCK_SIZE, k - block_start)
            i_indices = block_start + offsets
            
            mask = offsets < actual_size
            
            z_vals = tl.load(z_ptr + i_indices, mask=mask)
            tl.store(y_ptr + i_indices, z_vals, mask=mask)
        
        # Set y[k] = alpha
        tl.store(y_ptr + k, alpha)

def durbin_triton(r, y, z, N):
    BLOCK_SIZE = 64
    
    grid = (1,)
    durbin_kernel[grid](r, y, z, N, BLOCK_SIZE)