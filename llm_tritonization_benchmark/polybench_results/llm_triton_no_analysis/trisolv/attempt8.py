import triton
import triton.language as tl
import torch

@triton.jit
def trisolv_kernel(L, b, x, N, L_stride):
    i = tl.program_id(0)
    
    if i >= N:
        return
    
    # x[i] = b[i]
    b_ptr = b + i
    x_ptr = x + i
    b_val = tl.load(b_ptr)
    tl.store(x_ptr, b_val)
    
    # Load current x[i] value
    x_val = b_val
    
    # for (j = 0; j < i; j++)
    #   x[i] -= L[i][j] * x[j];
    if i > 0:
        BLOCK_SIZE = 32
        j_offsets = tl.arange(0, BLOCK_SIZE)
        
        for j_start in range(0, i, BLOCK_SIZE):
            j_end = min(j_start + BLOCK_SIZE, i)
            actual_block_size = j_end - j_start
            
            if actual_block_size > 0:
                j_indices = j_start + j_offsets
                mask = j_offsets < actual_block_size
                
                # Load L[i][j] values
                L_ptrs = L + i * L_stride + j_indices
                L_vals = tl.load(L_ptrs, mask=mask)
                
                # Load x[j] values
                x_ptrs = x + j_indices
                x_vals = tl.load(x_ptrs, mask=mask)
                
                # Compute products and sum
                products = L_vals * x_vals
                sum_val = tl.sum(products, axis=0)
                x_val = x_val - sum_val
    
    # x[i] = x[i] / L[i][i];
    L_diag_ptr = L + i * L_stride + i
    L_diag = tl.load(L_diag_ptr)
    x_val = x_val / L_diag
    
    # Store final result
    tl.store(x_ptr, x_val)

def trisolv_triton(L, b, x, N):
    L_stride = L.shape[1]
    
    # Launch kernel with one thread per row
    grid = (N,)
    trisolv_kernel[grid](L, b, x, N, L_stride)