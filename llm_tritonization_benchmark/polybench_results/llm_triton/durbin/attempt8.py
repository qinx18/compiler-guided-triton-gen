import triton
import triton.language as tl
import torch

@triton.jit
def durbin_kernel(r_ptr, y_ptr, z_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Initialize y[0] = -r[0]
    if pid == 0:
        r0 = tl.load(r_ptr)
        tl.store(y_ptr, -r0)
    
    tl.debug_barrier()
    
    # Sequential k loop from 1 to N-1
    for k in range(1, N):
        # Single thread computes beta, sum, alpha
        if pid == 0:
            # Get previous alpha
            prev_alpha = tl.load(y_ptr + (k-1))
            
            # Compute beta
            beta = (1.0 - prev_alpha * prev_alpha)
            for prev_k in range(1, k):
                # Need to recompute previous alpha values
                prev_sum = 0.0
                for prev_i in range(prev_k):
                    r_val = tl.load(r_ptr + (prev_k - prev_i - 1))
                    y_val = tl.load(y_ptr + prev_i)
                    prev_sum += r_val * y_val
                
                # Recompute previous beta
                if prev_k == 1:
                    temp_beta = 1.0 - tl.load(r_ptr) * tl.load(r_ptr)
                else:
                    temp_beta = 1.0
                    temp_alpha = -tl.load(r_ptr)
                    for temp_k in range(1, prev_k):
                        temp_beta = (1.0 - temp_alpha * temp_alpha) * temp_beta
                        temp_sum = 0.0
                        for temp_i in range(temp_k):
                            temp_r = tl.load(r_ptr + (temp_k - temp_i - 1))
                            temp_y = tl.load(y_ptr + temp_i)
                            temp_sum += temp_r * temp_y
                        r_temp_k = tl.load(r_ptr + temp_k)
                        temp_alpha = -(r_temp_k + temp_sum) / temp_beta
                    temp_beta = (1.0 - temp_alpha * temp_alpha) * temp_beta
                
                r_prev_k = tl.load(r_ptr + prev_k)
                prev_alpha_val = -(r_prev_k + prev_sum) / temp_beta
                beta = beta * (1.0 - prev_alpha_val * prev_alpha_val)
            
            # Compute sum for current k
            sum_val = 0.0
            for i in range(k):
                r_val = tl.load(r_ptr + (k - i - 1))
                y_val = tl.load(y_ptr + i)
                sum_val += r_val * y_val
            
            # Compute alpha
            r_k = tl.load(r_ptr + k)
            alpha = -(r_k + sum_val) / beta
            
            # Store alpha at y[k]
            tl.store(y_ptr + k, alpha)
        
        tl.debug_barrier()
        
        # Parallel update of z and y arrays
        mask = offsets < k
        
        if tl.sum(mask.to(tl.int32)) > 0:
            # Load current y values
            y_vals = tl.load(y_ptr + offsets, mask=mask)
            
            # Load alpha
            alpha = tl.load(y_ptr + k)
            
            # Load y[k-i-1] values
            reverse_offsets = k - offsets - 1
            reverse_mask = mask & (reverse_offsets >= 0)
            y_reverse_vals = tl.load(y_ptr + reverse_offsets, mask=reverse_mask, other=0.0)
            
            # Compute z[i] = y[i] + alpha * y[k-i-1]
            z_vals = y_vals + alpha * y_reverse_vals
            
            # Store z values
            tl.store(z_ptr + offsets, z_vals, mask=mask)
            
            # Copy z to y
            tl.store(y_ptr + offsets, z_vals, mask=mask)
        
        tl.debug_barrier()

def durbin_triton(r, y, z, N):
    BLOCK_SIZE = 128
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    durbin_kernel[grid](
        r, y, z, N,
        BLOCK_SIZE=BLOCK_SIZE
    )