import triton
import triton.language as tl
import torch

@triton.jit
def gemver_kernel(A_ptr, u1_ptr, u2_ptr, v1_ptr, v2_ptr, w_ptr, x_ptr, y_ptr, z_ptr, 
                  alpha, beta, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate which row this program handles
    row_start = pid * BLOCK_SIZE
    row_offsets = tl.arange(0, BLOCK_SIZE)
    row_indices = row_start + row_offsets
    row_mask = row_indices < N
    
    # Column offsets for inner loops
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # First loop: A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j]
    for i in range(N):
        if (i >= row_start) & (i < row_start + BLOCK_SIZE) & (i < N):
            local_i = i - row_start
            
            # Load u1[i] and u2[i]
            u1_i = tl.load(u1_ptr + i)
            u2_i = tl.load(u2_ptr + i)
            
            # Process columns in blocks
            for col_start in range(0, N, BLOCK_SIZE):
                col_indices = col_start + col_offsets
                col_mask = col_indices < N
                
                # Load A[i][j], v1[j], v2[j]
                A_offsets = i * N + col_indices
                A_vals = tl.load(A_ptr + A_offsets, mask=col_mask, other=0.0)
                v1_vals = tl.load(v1_ptr + col_indices, mask=col_mask, other=0.0)
                v2_vals = tl.load(v2_ptr + col_indices, mask=col_mask, other=0.0)
                
                # Compute A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j]
                new_A_vals = A_vals + u1_i * v1_vals + u2_i * v2_vals
                
                # Store back
                tl.store(A_ptr + A_offsets, new_A_vals, mask=col_mask)

    # Second loop: x[i] = x[i] + beta * A[j][i] * y[j]
    for i in range(N):
        if (i >= row_start) & (i < row_start + BLOCK_SIZE) & (i < N):
            x_accumulator = tl.load(x_ptr + i)
            
            # Process rows in blocks
            for row_blk_start in range(0, N, BLOCK_SIZE):
                row_blk_indices = row_blk_start + col_offsets
                row_blk_mask = row_blk_indices < N
                
                # Load A[j][i] and y[j]
                A_ji_offsets = row_blk_indices * N + i
                A_ji_vals = tl.load(A_ptr + A_ji_offsets, mask=row_blk_mask, other=0.0)
                y_vals = tl.load(y_ptr + row_blk_indices, mask=row_blk_mask, other=0.0)
                
                # Accumulate beta * A[j][i] * y[j]
                contrib = beta * A_ji_vals * y_vals
                x_accumulator = x_accumulator + tl.sum(contrib)
            
            # Store accumulated result
            tl.store(x_ptr + i, x_accumulator)

    # Third loop: x[i] = x[i] + z[i]
    for i in range(N):
        if (i >= row_start) & (i < row_start + BLOCK_SIZE) & (i < N):
            x_val = tl.load(x_ptr + i)
            z_val = tl.load(z_ptr + i)
            tl.store(x_ptr + i, x_val + z_val)

    # Fourth loop: w[i] = w[i] + alpha * A[i][j] * x[j]
    for i in range(N):
        if (i >= row_start) & (i < row_start + BLOCK_SIZE) & (i < N):
            w_accumulator = tl.load(w_ptr + i)
            
            # Process columns in blocks
            for col_blk_start in range(0, N, BLOCK_SIZE):
                col_blk_indices = col_blk_start + col_offsets
                col_blk_mask = col_blk_indices < N
                
                # Load A[i][j] and x[j]
                A_ij_offsets = i * N + col_blk_indices
                A_ij_vals = tl.load(A_ptr + A_ij_offsets, mask=col_blk_mask, other=0.0)
                x_vals = tl.load(x_ptr + col_blk_indices, mask=col_blk_mask, other=0.0)
                
                # Accumulate alpha * A[i][j] * x[j]
                contrib = alpha * A_ij_vals * x_vals
                w_accumulator = w_accumulator + tl.sum(contrib)
            
            # Store accumulated result
            tl.store(w_ptr + i, w_accumulator)

def gemver_triton(A, u1, u2, v1, v2, w, x, y, z, alpha, beta, N):
    BLOCK_SIZE = 32
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    gemver_kernel[grid](A, u1, u2, v1, v2, w, x, y, z, 
                        alpha, beta, N, BLOCK_SIZE)