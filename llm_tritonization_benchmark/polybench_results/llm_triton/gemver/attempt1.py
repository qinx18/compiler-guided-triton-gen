import triton
import triton.language as tl
import torch

@triton.jit
def gemver_kernel(A_ptr, u1_ptr, u2_ptr, v1_ptr, v2_ptr, w_ptr, x_ptr, y_ptr, z_ptr, alpha, beta, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID for this block
    pid = tl.program_id(0)
    
    # Each block handles BLOCK_SIZE rows
    row_start = pid * BLOCK_SIZE
    row_offsets = tl.arange(0, BLOCK_SIZE)
    row_indices = row_start + row_offsets
    row_mask = row_indices < N
    
    # First loop: A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j]
    u1_vals = tl.load(u1_ptr + row_indices, mask=row_mask, other=0.0)
    u2_vals = tl.load(u2_ptr + row_indices, mask=row_mask, other=0.0)
    
    for j in range(N):
        v1_val = tl.load(v1_ptr + j)
        v2_val = tl.load(v2_ptr + j)
        
        # Compute linear indices for A[i][j]
        linear_indices = row_indices * N + j
        
        # Load current A values
        a_vals = tl.load(A_ptr + linear_indices, mask=row_mask, other=0.0)
        
        # Compute new A values
        new_a_vals = a_vals + u1_vals * v1_val + u2_vals * v2_val
        
        # Store back to A
        tl.store(A_ptr + linear_indices, new_a_vals, mask=row_mask)
    
    # Second loop: x[i] = x[i] + beta * A[j][i] * y[j]
    # Load current x values for this block
    x_vals = tl.load(x_ptr + row_indices, mask=row_mask, other=0.0)
    
    for j in range(N):
        y_val = tl.load(y_ptr + j)
        
        # Linear indices for A[j][i] (transposed access)
        linear_indices = j * N + row_indices
        
        # Load A[j][i] values
        a_vals = tl.load(A_ptr + linear_indices, mask=row_mask, other=0.0)
        
        # Accumulate to x
        x_vals = x_vals + beta * a_vals * y_val
    
    # Third loop: x[i] = x[i] + z[i]
    z_vals = tl.load(z_ptr + row_indices, mask=row_mask, other=0.0)
    x_vals = x_vals + z_vals
    
    # Store updated x values
    tl.store(x_ptr + row_indices, x_vals, mask=row_mask)
    
    # Fourth loop: w[i] = w[i] + alpha * A[i][j] * x[j]
    # Load current w values for this block
    w_vals = tl.load(w_ptr + row_indices, mask=row_mask, other=0.0)
    
    for j in range(N):
        x_val = tl.load(x_ptr + j)
        
        # Linear indices for A[i][j]
        linear_indices = row_indices * N + j
        
        # Load A[i][j] values
        a_vals = tl.load(A_ptr + linear_indices, mask=row_mask, other=0.0)
        
        # Accumulate to w
        w_vals = w_vals + alpha * a_vals * x_val
    
    # Store updated w values
    tl.store(w_ptr + row_indices, w_vals, mask=row_mask)

def gemver_triton(A, u1, u2, v1, v2, w, x, y, z, alpha, beta, N):
    BLOCK_SIZE = 32
    
    # Calculate grid size
    grid_size = triton.cdiv(N, BLOCK_SIZE)
    
    # Launch kernel
    gemver_kernel[(grid_size,)](
        A, u1, u2, v1, v2, w, x, y, z,
        alpha, beta, N,
        BLOCK_SIZE=BLOCK_SIZE
    )