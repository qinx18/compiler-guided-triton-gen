import triton
import triton.language as tl
import torch

@triton.jit
def trisolv_kernel(L_ptr, b_ptr, x_ptr, N, L_stride, i, BLOCK_SIZE: tl.constexpr):
    col_offsets = tl.arange(0, BLOCK_SIZE)
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    current_offsets = block_start + col_offsets
    mask = (current_offsets < i) & (current_offsets >= 0)
    
    # Load L[i][j] for current block
    L_ptrs = L_ptr + i * L_stride + current_offsets
    L_vals = tl.load(L_ptrs, mask=mask, other=0.0)
    
    # Load x[j] for current block
    x_ptrs = x_ptr + current_offsets
    x_vals = tl.load(x_ptrs, mask=mask, other=0.0)
    
    # Compute partial sum: L[i][j] * x[j]
    products = L_vals * x_vals
    partial_sum = tl.sum(products)
    
    # Store partial sum in a temporary buffer
    temp_ptr = x_ptr + N + pid
    tl.store(temp_ptr, partial_sum)

@triton.jit
def trisolv_reduce_kernel(b_ptr, x_ptr, L_ptr, N, L_stride, i, num_blocks):
    # Sum all partial results
    total_sum = 0.0
    for block_id in range(num_blocks):
        temp_ptr = x_ptr + N + block_id
        partial_sum = tl.load(temp_ptr)
        total_sum += partial_sum
    
    # x[i] = b[i] - sum
    b_val = tl.load(b_ptr + i)
    x_val = b_val - total_sum
    
    # x[i] = x[i] / L[i][i]
    L_diag = tl.load(L_ptr + i * L_stride + i)
    x_val = x_val / L_diag
    
    tl.store(x_ptr + i, x_val)

def trisolv_triton(L, b, x, N):
    BLOCK_SIZE = 64
    
    # Create temporary buffer for partial sums
    temp_size = triton.cdiv(N, BLOCK_SIZE)
    x_extended = torch.zeros(N + temp_size, dtype=x.dtype, device=x.device)
    x_extended[:N] = x
    
    for i in range(N):
        if i > 0:
            num_blocks = triton.cdiv(i, BLOCK_SIZE)
            grid = (num_blocks,)
            trisolv_kernel[grid](L, b, x_extended, N, L.stride(0), i, BLOCK_SIZE)
            
            # Reduce partial sums
            trisolv_reduce_kernel[(1,)](b, x_extended, L, N, L.stride(0), i, num_blocks)
        else:
            # For i=0, just x[0] = b[0] / L[0][0]
            b_val = b[0]
            L_diag = L[0, 0]
            x_extended[0] = b_val / L_diag
    
    x.copy_(x_extended[:N])