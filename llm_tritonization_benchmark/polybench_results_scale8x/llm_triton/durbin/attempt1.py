import torch
import triton
import triton.language as tl

@triton.jit
def durbin_kernel(r_ptr, y_ptr, z_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    # Initialize
    if pid == 0:
        r0 = tl.load(r_ptr)
        tl.store(y_ptr, -r0)
    
    tl.debug_barrier()
    
    # Sequential loop over k
    for k in range(1, N):
        if pid == 0:
            # Load current alpha from y[k-1] (computed in previous iteration)
            alpha = tl.load(y_ptr + (k - 1))
            
            # beta = (1 - alpha*alpha) * beta
            if k == 1:
                beta = (1.0 - alpha * alpha) * 1.0
            else:
                # For k > 1, we need to recompute beta from scratch
                beta_val = 1.0
                for prev_k in range(1, k):
                    prev_alpha = tl.load(y_ptr + prev_k)
                    beta_val = (1.0 - prev_alpha * prev_alpha) * beta_val
                beta = beta_val
            
            # Compute sum
            sum_val = 0.0
            for i in range(k):
                r_val = tl.load(r_ptr + (k - i - 1))
                y_val = tl.load(y_ptr + i)
                sum_val += r_val * y_val
            
            # Compute new alpha
            r_k = tl.load(r_ptr + k)
            new_alpha = -(r_k + sum_val) / beta
            
            # Store new alpha temporarily in a known location
            tl.store(z_ptr + N, new_alpha)
        
        tl.debug_barrier()
        
        # Load the new alpha computed by thread 0
        new_alpha = tl.load(z_ptr + N)
        
        # Parallel update: z[i] = y[i] + alpha*y[k-i-1] for i in [0, k)
        i_offsets = tl.arange(0, BLOCK_SIZE)
        i_base = pid * BLOCK_SIZE
        current_i = i_base + i_offsets
        mask = (current_i < k)
        
        if tl.sum(mask.to(tl.int32)) > 0:
            y_i = tl.load(y_ptr + current_i, mask=mask)
            y_k_minus_i_minus_1 = tl.load(y_ptr + (k - current_i - 1), mask=mask)
            z_i = y_i + new_alpha * y_k_minus_i_minus_1
            tl.store(z_ptr + current_i, z_i, mask=mask)
        
        tl.debug_barrier()
        
        # Parallel copy: y[i] = z[i] for i in [0, k)
        if tl.sum(mask.to(tl.int32)) > 0:
            z_i = tl.load(z_ptr + current_i, mask=mask)
            tl.store(y_ptr + current_i, z_i, mask=mask)
        
        tl.debug_barrier()
        
        # Set y[k] = alpha
        if pid == 0:
            tl.store(y_ptr + k, new_alpha)
        
        tl.debug_barrier()

def durbin_triton(r, y, z, N):
    BLOCK_SIZE = 64
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    durbin_kernel[grid](r, y, z, N, BLOCK_SIZE)