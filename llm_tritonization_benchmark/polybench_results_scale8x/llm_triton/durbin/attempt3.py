import torch
import triton
import triton.language as tl

@triton.jit
def durbin_kernel(r_ptr, y_ptr, z_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    # Initialize y[0] = -r[0]
    if pid == 0:
        r0 = tl.load(r_ptr)
        tl.store(y_ptr, -r0)
    
    tl.debug_barrier()
    
    # Sequential loop over k
    for k in range(1, N):
        # Thread 0 computes the scalar values
        if pid == 0:
            # Compute beta from scratch
            beta = 1.0
            for prev_k in range(1, k):
                # Get alpha from previous iteration (k-1)
                prev_alpha = tl.load(y_ptr + prev_k - 1)
                beta = (1.0 - prev_alpha * prev_alpha) * beta
            
            # Get the current alpha (from k-1)
            if k > 1:
                alpha = tl.load(y_ptr + k - 1)
                beta = (1.0 - alpha * alpha) * beta
            else:
                # k == 1, alpha is y[0] = -r[0]
                alpha = tl.load(y_ptr)
                beta = (1.0 - alpha * alpha) * beta
            
            # Compute sum
            sum_val = 0.0
            for i in range(k):
                r_val = tl.load(r_ptr + k - i - 1)
                y_val = tl.load(y_ptr + i)
                sum_val += r_val * y_val
            
            # Compute new alpha
            r_k = tl.load(r_ptr + k)
            new_alpha = -(r_k + sum_val) / beta
            
            # Store new alpha in temporary location
            tl.store(z_ptr + N, new_alpha)
        
        tl.debug_barrier()
        
        # All threads read the new alpha
        new_alpha = tl.load(z_ptr + N)
        
        # Parallel computation of z[i] = y[i] + new_alpha*y[k-i-1] for i in [0, k)
        offsets = tl.arange(0, BLOCK_SIZE)
        i_base = pid * BLOCK_SIZE
        current_i = i_base + offsets
        mask = current_i < k
        
        if tl.sum(mask.to(tl.int32)) > 0:
            y_i = tl.load(y_ptr + current_i, mask=mask)
            y_k_minus_i_minus_1 = tl.load(y_ptr + k - current_i - 1, mask=mask)
            z_i = y_i + new_alpha * y_k_minus_i_minus_1
            tl.store(z_ptr + current_i, z_i, mask=mask)
        
        tl.debug_barrier()
        
        # Parallel copy z[i] -> y[i] for i in [0, k)
        if tl.sum(mask.to(tl.int32)) > 0:
            z_i = tl.load(z_ptr + current_i, mask=mask)
            tl.store(y_ptr + current_i, z_i, mask=mask)
        
        tl.debug_barrier()
        
        # Thread 0 sets y[k] = new_alpha
        if pid == 0:
            tl.store(y_ptr + k, new_alpha)
        
        tl.debug_barrier()

def durbin_triton(r, y, z, N):
    BLOCK_SIZE = 64
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    durbin_kernel[grid](r, y, z, N, BLOCK_SIZE)