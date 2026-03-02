import triton
import triton.language as tl
import torch

@triton.jit
def gemver_kernel_a(A_ptr, u1_ptr, u2_ptr, v1_ptr, v2_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE
    row_offsets = tl.arange(0, BLOCK_SIZE)
    row_indices = row_start + row_offsets
    row_mask = row_indices < N
    
    u1_vals = tl.load(u1_ptr + row_indices, mask=row_mask, other=0.0)
    u2_vals = tl.load(u2_ptr + row_indices, mask=row_mask, other=0.0)
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    for col_start in range(0, N, BLOCK_SIZE):
        col_indices = col_start + col_offsets
        col_mask = col_indices < N
        
        v1_vals = tl.load(v1_ptr + col_indices, mask=col_mask, other=0.0)
        v2_vals = tl.load(v2_ptr + col_indices, mask=col_mask, other=0.0)
        
        row_expanded = tl.expand_dims(row_indices, 1)
        col_expanded = tl.expand_dims(col_indices, 0)
        linear_indices = row_expanded * N + col_expanded
        
        mask_2d = tl.expand_dims(row_mask, 1) & tl.expand_dims(col_mask, 0)
        
        a_vals = tl.load(A_ptr + linear_indices, mask=mask_2d, other=0.0)
        
        u1_expanded = tl.expand_dims(u1_vals, 1)
        u2_expanded = tl.expand_dims(u2_vals, 1)
        v1_expanded = tl.expand_dims(v1_vals, 0)
        v2_expanded = tl.expand_dims(v2_vals, 0)
        
        new_a_vals = a_vals + u1_expanded * v1_expanded + u2_expanded * v2_expanded
        
        tl.store(A_ptr + linear_indices, new_a_vals, mask=mask_2d)

@triton.jit
def gemver_kernel_x(A_ptr, x_ptr, y_ptr, beta, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_start = pid * BLOCK_SIZE
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = i_start + i_offsets
    i_mask = i_indices < N
    
    x_vals = tl.load(x_ptr + i_indices, mask=i_mask, other=0.0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    for j_start in range(0, N, BLOCK_SIZE):
        j_indices = j_start + j_offsets
        j_mask = j_indices < N
        
        y_vals = tl.load(y_ptr + j_indices, mask=j_mask, other=0.0)
        
        j_expanded = tl.expand_dims(j_indices, 1)
        i_expanded = tl.expand_dims(i_indices, 0)
        linear_indices = j_expanded * N + i_expanded
        
        mask_2d = tl.expand_dims(j_mask, 1) & tl.expand_dims(i_mask, 0)
        
        a_vals = tl.load(A_ptr + linear_indices, mask=mask_2d, other=0.0)
        
        y_expanded = tl.expand_dims(y_vals, 1)
        
        contrib = beta * a_vals * y_expanded
        x_vals = x_vals + tl.sum(contrib, axis=0)
    
    tl.store(x_ptr + i_indices, x_vals, mask=i_mask)

@triton.jit
def gemver_kernel_w(A_ptr, w_ptr, x_ptr, alpha, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    i_start = pid * BLOCK_SIZE
    i_offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = i_start + i_offsets
    i_mask = i_indices < N
    
    w_vals = tl.load(w_ptr + i_indices, mask=i_mask, other=0.0)
    
    j_offsets = tl.arange(0, BLOCK_SIZE)
    for j_start in range(0, N, BLOCK_SIZE):
        j_indices = j_start + j_offsets
        j_mask = j_indices < N
        
        x_vals = tl.load(x_ptr + j_indices, mask=j_mask, other=0.0)
        
        i_expanded = tl.expand_dims(i_indices, 1)
        j_expanded = tl.expand_dims(j_indices, 0)
        linear_indices = i_expanded * N + j_expanded
        
        mask_2d = tl.expand_dims(i_mask, 1) & tl.expand_dims(j_mask, 0)
        
        a_vals = tl.load(A_ptr + linear_indices, mask=mask_2d, other=0.0)
        
        x_expanded = tl.expand_dims(x_vals, 0)
        
        contrib = alpha * a_vals * x_expanded
        w_vals = w_vals + tl.sum(contrib, axis=1)
    
    tl.store(w_ptr + i_indices, w_vals, mask=i_mask)

def gemver_triton(A, u1, u2, v1, v2, w, x, y, z, alpha, beta, N):
    BLOCK_SIZE = 32
    grid_size = triton.cdiv(N, BLOCK_SIZE)
    
    gemver_kernel_a[(grid_size,)](A, u1, u2, v1, v2, N, BLOCK_SIZE=BLOCK_SIZE)
    
    gemver_kernel_x[(grid_size,)](A, x, y, beta, N, BLOCK_SIZE=BLOCK_SIZE)
    
    x[:] = x + z
    
    gemver_kernel_w[(grid_size,)](A, w, x, alpha, N, BLOCK_SIZE=BLOCK_SIZE)