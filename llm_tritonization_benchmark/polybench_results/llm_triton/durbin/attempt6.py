import triton
import triton.language as tl
import torch

@triton.jit
def durbin_kernel(r_ptr, y_ptr, z_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    if block_start == 0:
        # Initialize y[0], beta, alpha for first thread only
        if pid == 0:
            r0 = tl.load(r_ptr)
            tl.store(y_ptr, -r0)
    
    tl.debug_barrier()
    
    # Sequential k loop
    for k in range(1, N):
        if block_start == 0:
            if pid == 0:
                # Compute previous alpha
                if k == 1:
                    prev_alpha = -tl.load(r_ptr)
                else:
                    prev_alpha = tl.load(y_ptr + (k-1))
                
                # Compute beta
                beta = 1.0
                temp_alpha = -tl.load(r_ptr)
                for iter in range(1, k):
                    beta = (1.0 - temp_alpha * temp_alpha) * beta
                    
                    # Compute sum for this iteration
                    sum_val = 0.0
                    for i in range(iter):
                        r_val = tl.load(r_ptr + (iter - i - 1))
                        y_val = tl.load(y_ptr + i)
                        sum_val += r_val * y_val
                    
                    r_iter = tl.load(r_ptr + iter)
                    temp_alpha = -(r_iter + sum_val) / beta
                
                # Now compute for current k
                beta = (1.0 - prev_alpha * prev_alpha) * beta
                
                # Compute sum
                sum_val = 0.0
                for i in range(k):
                    r_val = tl.load(r_ptr + (k - i - 1))
                    y_val = tl.load(y_ptr + i)
                    sum_val += r_val * y_val
                
                r_k = tl.load(r_ptr + k)
                alpha = -(r_k + sum_val) / beta
                
                # Store alpha at y[k]
                tl.store(y_ptr + k, alpha)
        
        tl.debug_barrier()
        
        # Parallel update of z and y arrays
        current_offsets = block_start + offsets
        mask = current_offsets < k
        
        if tl.any(mask):
            # Load current y values
            y_vals = tl.load(y_ptr + current_offsets, mask=mask)
            
            # Load alpha
            alpha = tl.load(y_ptr + k)
            
            # Load y[k-i-1] values
            reverse_offsets = k - current_offsets - 1
            reverse_mask = mask & (reverse_offsets >= 0) & (reverse_offsets < N)
            y_reverse_vals = tl.load(y_ptr + reverse_offsets, mask=reverse_mask, other=0.0)
            
            # Compute z[i] = y[i] + alpha * y[k-i-1]
            z_vals = y_vals + alpha * y_reverse_vals
            
            # Store z values
            tl.store(z_ptr + current_offsets, z_vals, mask=mask)
            
            # Copy z to y
            tl.store(y_ptr + current_offsets, z_vals, mask=mask)
        
        tl.debug_barrier()

def durbin_triton(r, y, z, N):
    BLOCK_SIZE = 128
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    durbin_kernel[grid](
        r, y, z, N,
        BLOCK_SIZE=BLOCK_SIZE
    )