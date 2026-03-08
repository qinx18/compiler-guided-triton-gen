import triton
import triton.language as tl
import torch

@triton.jit
def gemver_kernel(A_ptr, u1_ptr, u2_ptr, v1_ptr, v2_ptr, w_ptr, x_ptr, y_ptr, z_ptr, 
                  alpha, beta, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    
    row_start = pid * BLOCK_SIZE
    row_offsets = tl.arange(0, BLOCK_SIZE)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # First loop: A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j]
    for r in range(BLOCK_SIZE):
        i = row_start + r
        if i < N:
            u1_val = tl.load(u1_ptr + i)
            u2_val = tl.load(u2_ptr + i)
            
            for col_block in range(0, N, BLOCK_SIZE):
                j_indices = col_block + col_offsets
                j_mask = j_indices < N
                
                v1_vals = tl.load(v1_ptr + j_indices, mask=j_mask)
                v2_vals = tl.load(v2_ptr + j_indices, mask=j_mask)
                
                a_indices = i * N + j_indices
                a_vals = tl.load(A_ptr + a_indices, mask=j_mask)
                
                new_a_vals = a_vals + u1_val * v1_vals + u2_val * v2_vals
                tl.store(A_ptr + a_indices, new_a_vals, mask=j_mask)

    tl.debug_barrier()

    # Second loop: x[i] = x[i] + beta * A[j][i] * y[j]
    for r in range(BLOCK_SIZE):
        i = row_start + r
        if i < N:
            x_val = tl.load(x_ptr + i)
            
            for j in range(N):
                a_val = tl.load(A_ptr + j * N + i)
                y_val = tl.load(y_ptr + j)
                x_val = x_val + beta * a_val * y_val
            
            tl.store(x_ptr + i, x_val)

    tl.debug_barrier()

    # Third loop: x[i] = x[i] + z[i]
    for r in range(BLOCK_SIZE):
        i = row_start + r
        if i < N:
            x_val = tl.load(x_ptr + i)
            z_val = tl.load(z_ptr + i)
            tl.store(x_ptr + i, x_val + z_val)

    tl.debug_barrier()

    # Fourth loop: w[i] = w[i] + alpha * A[i][j] * x[j]
    for r in range(BLOCK_SIZE):
        i = row_start + r
        if i < N:
            w_val = tl.load(w_ptr + i)
            
            for col_block in range(0, N, BLOCK_SIZE):
                j_indices = col_block + col_offsets
                j_mask = j_indices < N
                
                a_indices = i * N + j_indices
                a_vals = tl.load(A_ptr + a_indices, mask=j_mask)
                x_vals = tl.load(x_ptr + j_indices, mask=j_mask)
                
                contrib = tl.sum(tl.where(j_mask, alpha * a_vals * x_vals, 0.0))
                w_val = w_val + contrib
            
            tl.store(w_ptr + i, w_val)


def gemver_triton(A, u1, u2, v1, v2, w, x, y, z, alpha, beta, N):
    BLOCK_SIZE = 32
    num_blocks = triton.cdiv(N, BLOCK_SIZE)
    
    gemver_kernel[(num_blocks,)](
        A, u1, u2, v1, v2, w, x, y, z,
        alpha, beta, N, BLOCK_SIZE
    )