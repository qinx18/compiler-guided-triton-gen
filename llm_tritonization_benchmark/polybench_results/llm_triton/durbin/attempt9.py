import triton
import triton.language as tl
import torch

@triton.jit
def durbin_kernel(r_ptr, y_ptr, z_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Initialize y[0] = -r[0]
    if tl.program_id(0) == 0:
        r0 = tl.load(r_ptr)
        tl.store(y_ptr, -r0)
    
    # Sequential k loop from 1 to N-1
    for k in range(1, N):
        # Single thread computes alpha
        if tl.program_id(0) == 0:
            # Compute beta iteratively
            beta = 1.0
            prev_alpha = -tl.load(r_ptr)  # Initial alpha = -r[0]
            
            for prev_k in range(1, k + 1):
                beta = (1.0 - prev_alpha * prev_alpha) * beta
                
                # Compute sum for prev_k
                sum_val = 0.0
                for i in range(prev_k):
                    r_val = tl.load(r_ptr + (prev_k - i - 1))
                    y_val = tl.load(y_ptr + i)
                    sum_val += r_val * y_val
                
                # Compute alpha for prev_k
                r_prev_k = tl.load(r_ptr + prev_k)
                prev_alpha = -(r_prev_k + sum_val) / beta
            
            # Store alpha at y[k]
            tl.store(y_ptr + k, prev_alpha)
        
        # Parallel update of y array
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < k
        
        if tl.any(mask):
            # Load current alpha
            alpha = tl.load(y_ptr + k)
            
            # Load current y values
            y_vals = tl.load(y_ptr + offsets, mask=mask)
            
            # Load y[k-i-1] values
            reverse_offsets = k - offsets - 1
            reverse_mask = mask & (reverse_offsets >= 0) & (reverse_offsets < N)
            y_reverse_vals = tl.load(y_ptr + reverse_offsets, mask=reverse_mask, other=0.0)
            
            # Compute z[i] = y[i] + alpha * y[k-i-1]
            z_vals = y_vals + alpha * y_reverse_vals
            
            # Store z values and update y
            tl.store(z_ptr + offsets, z_vals, mask=mask)
            tl.store(y_ptr + offsets, z_vals, mask=mask)

def durbin_triton(r, y, z, N):
    BLOCK_SIZE = 32
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    durbin_kernel[grid](
        r, y, z, N,
        BLOCK_SIZE=BLOCK_SIZE
    )