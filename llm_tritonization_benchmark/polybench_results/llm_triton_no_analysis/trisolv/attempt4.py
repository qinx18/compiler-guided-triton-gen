import triton
import triton.language as tl
import torch

@triton.jit
def trisolv_kernel(L_ptr, b_ptr, x_ptr, N, L_stride):
    BLOCK_SIZE = 128
    
    # Define offsets once outside the loop
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process each row sequentially
    for i in range(N):
        # Initialize x[i] with b[i]
        b_i = tl.load(b_ptr + i)
        x_i = b_i
        
        # Compute sum of L[i][j] * x[j] for j < i
        if i > 0:
            # Process in blocks
            sum_val = 0.0
            for j_start in range(0, i, BLOCK_SIZE):
                current_j_offsets = j_start + j_offsets
                j_mask = (current_j_offsets < i) & (current_j_offsets >= 0)
                
                # Load L[i][j_start:j_start+BLOCK_SIZE]
                L_row_ptr = L_ptr + i * L_stride
                L_vals = tl.load(L_row_ptr + current_j_offsets, mask=j_mask, other=0.0)
                
                # Load x[j_start:j_start+BLOCK_SIZE]
                x_vals = tl.load(x_ptr + current_j_offsets, mask=j_mask, other=0.0)
                
                # Compute partial sum
                products = L_vals * x_vals
                sum_val += tl.sum(products)
            
            x_i = x_i - sum_val
        
        # Divide by diagonal element
        L_diag = tl.load(L_ptr + i * L_stride + i)
        x_i = x_i / L_diag
        
        # Store result
        tl.store(x_ptr + i, x_i)

def trisolv_triton(L, b, x, N):
    x.copy_(b)
    
    grid = (1,)
    trisolv_kernel[grid](
        L.data_ptr(), b.data_ptr(), x.data_ptr(), N, L.stride(0)
    )