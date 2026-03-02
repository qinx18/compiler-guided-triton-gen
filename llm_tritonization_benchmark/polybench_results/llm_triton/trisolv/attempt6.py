import triton
import triton.language as tl
import torch

@triton.jit
def trisolv_kernel(L_ptr, b_ptr, x_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid_i = tl.program_id(0)
    
    if pid_i >= N:
        return

    i = pid_i

    # Start with x[i] = b[i]
    b_val = tl.load(b_ptr + i)
    x_val = b_val

    # Compute sum of L[i][j] * x[j] for j < i
    if i > 0:
        # Process in blocks
        for block_start in range(0, i, BLOCK_SIZE):
            j_offsets = tl.arange(0, BLOCK_SIZE) + block_start
            j_mask = (j_offsets < i) & (j_offsets < N)
            
            # Load L[i][j] values
            L_indices = i * N + j_offsets
            L_vals = tl.load(L_ptr + L_indices, mask=j_mask, other=0.0)
            
            # Load x[j] values
            x_vals = tl.load(x_ptr + j_offsets, mask=j_mask, other=0.0)
            
            # Compute products and sum
            products = L_vals * x_vals
            x_val -= tl.sum(products)

    # Divide by L[i][i]
    L_ii = tl.load(L_ptr + i * N + i)
    x_val = x_val / L_ii

    # Store result
    tl.store(x_ptr + i, x_val)

def trisolv_triton(L, b, x, N):
    BLOCK_SIZE = 64
    
    # Launch kernel with one thread per row
    grid = (N,)
    
    trisolv_kernel[grid](
        L, b, x, N,
        BLOCK_SIZE=BLOCK_SIZE
    )