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

def gemver_triton(A, u1, u2, v1, v2, w, x, y, z, alpha, beta, N):
    BLOCK_SIZE = 32
    
    # Calculate grid size
    grid_size = triton.cdiv(N, BLOCK_SIZE)
    
    # Launch first kernel for A update
    gemver_kernel[(grid_size,)](
        A, u1, u2, v1, v2, w, x, y, z,
        alpha, beta, N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Second loop: x[i] = x[i] + beta * A[j][i] * y[j] - sequential on CPU
    for j in range(N):
        y_val = y[j].item()
        # Vectorized update for all i
        x[:] = x + beta * A[j, :] * y_val
    
    # Third loop: x[i] = x[i] + z[i] - vectorized
    x[:] = x + z
    
    # Fourth loop: w[i] = w[i] + alpha * A[i][j] * x[j] - sequential on CPU  
    for j in range(N):
        x_val = x[j].item()
        # Vectorized update for all i
        w[:] = w + alpha * A[:, j] * x_val