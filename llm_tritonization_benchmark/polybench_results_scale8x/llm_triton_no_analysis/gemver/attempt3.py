import triton
import triton.language as tl
import torch

@triton.jit
def gemver_kernel(A_ptr, u1_ptr, u2_ptr, v1_ptr, v2_ptr, w_ptr, x_ptr, y_ptr, z_ptr, 
                  alpha, beta, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate which elements this program handles
    start_idx = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = start_idx + offsets
    mask = indices < N
    
    # First loop: A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j]
    for i in range(N):
        u1_i = tl.load(u1_ptr + i)
        u2_i = tl.load(u2_ptr + i)
        
        for j_start in range(0, N, BLOCK_SIZE):
            j_indices = j_start + offsets
            j_mask = j_indices < N
            
            A_offsets = i * N + j_indices
            A_vals = tl.load(A_ptr + A_offsets, mask=j_mask, other=0.0)
            v1_vals = tl.load(v1_ptr + j_indices, mask=j_mask, other=0.0)
            v2_vals = tl.load(v2_ptr + j_indices, mask=j_mask, other=0.0)
            
            new_A_vals = A_vals + u1_i * v1_vals + u2_i * v2_vals
            tl.store(A_ptr + A_offsets, new_A_vals, mask=j_mask)

    # Second loop: x[i] = x[i] + beta * A[j][i] * y[j]
    for i in range(N):
        x_accumulator = 0.0
        
        for j_start in range(0, N, BLOCK_SIZE):
            j_indices = j_start + offsets
            j_mask = j_indices < N
            
            A_ji_offsets = j_indices * N + i
            A_ji_vals = tl.load(A_ptr + A_ji_offsets, mask=j_mask, other=0.0)
            y_vals = tl.load(y_ptr + j_indices, mask=j_mask, other=0.0)
            
            contrib = beta * A_ji_vals * y_vals
            x_accumulator = x_accumulator + tl.sum(contrib)
        
        x_val = tl.load(x_ptr + i)
        tl.store(x_ptr + i, x_val + x_accumulator)

    # Third loop: x[i] = x[i] + z[i]
    for i in range(N):
        x_val = tl.load(x_ptr + i)
        z_val = tl.load(z_ptr + i)
        tl.store(x_ptr + i, x_val + z_val)

    # Fourth loop: w[i] = w[i] + alpha * A[i][j] * x[j]
    for i in range(N):
        w_accumulator = 0.0
        
        for j_start in range(0, N, BLOCK_SIZE):
            j_indices = j_start + offsets
            j_mask = j_indices < N
            
            A_ij_offsets = i * N + j_indices
            A_ij_vals = tl.load(A_ptr + A_ij_offsets, mask=j_mask, other=0.0)
            x_vals = tl.load(x_ptr + j_indices, mask=j_mask, other=0.0)
            
            contrib = alpha * A_ij_vals * x_vals
            w_accumulator = w_accumulator + tl.sum(contrib)
        
        w_val = tl.load(w_ptr + i)
        tl.store(w_ptr + i, w_val + w_accumulator)

def gemver_triton(A, u1, u2, v1, v2, w, x, y, z, alpha, beta, N):
    BLOCK_SIZE = 32
    grid = (1,)
    
    gemver_kernel[grid](A, u1, u2, v1, v2, w, x, y, z, 
                        alpha, beta, N, BLOCK_SIZE)