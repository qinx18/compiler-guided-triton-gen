import triton
import triton.language as tl
import torch

@triton.jit
def gemver_kernel(A_ptr, u1_ptr, u2_ptr, v1_ptr, v2_ptr, w_ptr, x_ptr, y_ptr, z_ptr, 
                  alpha, beta, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    row_start = pid * BLOCK_SIZE
    row_offsets = tl.arange(0, BLOCK_SIZE)
    row_indices = row_start + row_offsets
    row_mask = row_indices < N
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # First loop: A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j]
    if pid * BLOCK_SIZE < N:
        u1_vals = tl.load(u1_ptr + row_indices, mask=row_mask)
        u2_vals = tl.load(u2_ptr + row_indices, mask=row_mask)
        
        for col_start in range(0, N, BLOCK_SIZE):
            col_indices = col_start + col_offsets
            col_mask = col_indices < N
            
            v1_vals = tl.load(v1_ptr + col_indices, mask=col_mask)
            v2_vals = tl.load(v2_ptr + col_indices, mask=col_mask)
            
            for r in range(BLOCK_SIZE):
                if row_start + r < N:
                    row_idx = row_start + r
                    a_row_offset = row_idx * N
                    
                    a_offsets = a_row_offset + col_indices
                    a_vals = tl.load(A_ptr + a_offsets, mask=col_mask)
                    
                    u1_val = tl.load(u1_ptr + row_idx)
                    u2_val = tl.load(u2_ptr + row_idx)
                    update = u1_val * v1_vals + u2_val * v2_vals
                    new_a_vals = a_vals + update
                    
                    tl.store(A_ptr + a_offsets, new_a_vals, mask=col_mask)

    # Second loop: x[i] = x[i] + beta * A[j][i] * y[j]
    if pid * BLOCK_SIZE < N:
        x_vals = tl.load(x_ptr + row_indices, mask=row_mask)
        
        for col_start in range(0, N, BLOCK_SIZE):
            col_indices = col_start + col_offsets
            col_mask = col_indices < N
            
            y_vals = tl.load(y_ptr + col_indices, mask=col_mask)
            
            for c in range(BLOCK_SIZE):
                if col_start + c < N:
                    j = col_start + c
                    y_val = tl.load(y_ptr + j)
                    
                    a_offsets = j * N + row_indices
                    a_vals = tl.load(A_ptr + a_offsets, mask=row_mask)
                    
                    x_vals = x_vals + beta * a_vals * y_val
        
        tl.store(x_ptr + row_indices, x_vals, mask=row_mask)

    # Third loop: x[i] = x[i] + z[i]
    if pid * BLOCK_SIZE < N:
        x_vals = tl.load(x_ptr + row_indices, mask=row_mask)
        z_vals = tl.load(z_ptr + row_indices, mask=row_mask)
        x_vals = x_vals + z_vals
        tl.store(x_ptr + row_indices, x_vals, mask=row_mask)

    # Fourth loop: w[i] = w[i] + alpha * A[i][j] * x[j]
    if pid * BLOCK_SIZE < N:
        w_vals = tl.load(w_ptr + row_indices, mask=row_mask)
        
        for col_start in range(0, N, BLOCK_SIZE):
            col_indices = col_start + col_offsets
            col_mask = col_indices < N
            
            x_vals = tl.load(x_ptr + col_indices, mask=col_mask)
            
            for r in range(BLOCK_SIZE):
                if row_start + r < N:
                    row_idx = row_start + r
                    a_row_offset = row_idx * N
                    
                    a_offsets = a_row_offset + col_indices
                    a_vals = tl.load(A_ptr + a_offsets, mask=col_mask)
                    
                    contrib = tl.sum(alpha * a_vals * x_vals)
                    current_w = tl.load(w_ptr + row_idx)
                    tl.store(w_ptr + row_idx, current_w + contrib)


def gemver_triton(A, u1, u2, v1, v2, w, x, y, z, alpha, beta, N):
    BLOCK_SIZE = 32
    num_blocks = triton.cdiv(N, BLOCK_SIZE)
    
    gemver_kernel[(num_blocks,)](
        A, u1, u2, v1, v2, w, x, y, z,
        alpha, beta, N, BLOCK_SIZE
    )