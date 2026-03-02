import triton
import triton.language as tl
import torch

@triton.jit
def durbin_kernel(r_ptr, y_ptr, z_ptr, N, BLOCK_SIZE: tl.constexpr):
    # Get thread position
    pid = tl.program_id(0)
    
    # Each thread handles one value of k
    k = pid + 1
    if k >= N:
        return
    
    # Load r[0] for initial alpha calculation
    r0 = tl.load(r_ptr)
    
    # Calculate alpha for current k
    # For k=1: alpha = -r[0] (previous value)
    # For k>1: alpha = -(r[k-1] + sum_prev)/beta_prev
    if k == 1:
        alpha = -r0
    else:
        # Calculate previous iteration's values
        prev_k = k - 1
        
        # Calculate previous beta
        prev_alpha = -r0
        prev_beta = 1.0
        for prev_iter in range(1, prev_k):
            prev_beta = (1.0 - prev_alpha * prev_alpha) * prev_beta
            
            # Calculate sum for prev_iter
            prev_sum = 0.0
            for i in range(prev_iter):
                r_val = tl.load(r_ptr + prev_iter - i - 1)
                y_val = tl.load(y_ptr + i)
                prev_sum += r_val * y_val
            
            prev_alpha = -(tl.load(r_ptr + prev_iter) + prev_sum) / prev_beta
        
        # Now calculate sum for current k-1
        sum_val = 0.0
        for i in range(prev_k):
            r_val = tl.load(r_ptr + prev_k - i - 1)
            y_val = tl.load(y_ptr + i)
            sum_val += r_val * y_val
        
        alpha = -(tl.load(r_ptr + prev_k) + sum_val) / prev_beta
    
    # Calculate current beta
    beta = 1.0
    curr_alpha = -r0
    for iter in range(1, k + 1):
        if iter == k:
            curr_alpha = alpha
        else:
            # Recalculate alpha for this iteration
            iter_beta = 1.0
            iter_alpha = -r0
            for sub_iter in range(1, iter):
                iter_beta = (1.0 - iter_alpha * iter_alpha) * iter_beta
                iter_sum = 0.0
                for j in range(sub_iter):
                    r_val = tl.load(r_ptr + sub_iter - j - 1)
                    y_val = tl.load(y_ptr + j)
                    iter_sum += r_val * y_val
                iter_alpha = -(tl.load(r_ptr + sub_iter) + iter_sum) / iter_beta
            curr_alpha = iter_alpha
        
        beta = (1.0 - curr_alpha * curr_alpha) * beta
    
    # Calculate sum for current k
    sum_val = 0.0
    for i in range(k):
        r_val = tl.load(r_ptr + k - i - 1)
        y_val = tl.load(y_ptr + i)
        sum_val += r_val * y_val
    
    # Recalculate alpha for current k
    alpha = -(tl.load(r_ptr + k) + sum_val) / beta
    
    # Update z and y arrays
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < k
    
    # Load y values
    y_vals = tl.load(y_ptr + offsets, mask=mask)
    y_rev_vals = tl.load(y_ptr + k - offsets - 1, mask=mask)
    
    # Calculate z[i] = y[i] + alpha * y[k-i-1]
    z_vals = y_vals + alpha * y_rev_vals
    
    # Store z values
    tl.store(z_ptr + offsets, z_vals, mask=mask)
    
    # Copy z to y
    tl.store(y_ptr + offsets, z_vals, mask=mask)
    
    # Store y[k] = alpha
    if k < N:
        tl.store(y_ptr + k, alpha)

def durbin_triton(r, y, z, N):
    # Initialize y[0] = -r[0]
    y[0] = -r[0]
    
    # Launch kernel for k = 1 to N-1
    BLOCK_SIZE = 128
    grid = (triton.cdiv(N - 1, 1),)
    
    # Sequential execution since each k depends on previous iterations
    for k in range(1, N):
        # Calculate beta
        alpha_val = -r[0].item()
        beta = 1.0
        
        for iter in range(1, k + 1):
            if iter < k:
                # Calculate sum for iter
                sum_val = 0.0
                for i in range(iter):
                    sum_val += r[iter - i - 1].item() * y[i].item()
                alpha_val = -(r[iter].item() + sum_val) / beta
            
            beta = (1.0 - alpha_val * alpha_val) * beta
        
        # Calculate sum for current k
        sum_val = 0.0
        for i in range(k):
            sum_val += r[k - i - 1].item() * y[i].item()
        
        alpha_val = -(r[k].item() + sum_val) / beta
        
        # Update arrays
        for i in range(k):
            z[i] = y[i] + alpha_val * y[k - i - 1]
        
        for i in range(k):
            y[i] = z[i]
        
        y[k] = alpha_val