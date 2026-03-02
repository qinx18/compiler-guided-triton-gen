import triton
import triton.language as tl

@triton.jit
def gemver_kernel(A_ptr, u1_ptr, u2_ptr, v1_ptr, v2_ptr, w_ptr, x_ptr, y_ptr, z_ptr, alpha, beta, N, BLOCK_SIZE: tl.constexpr):
    # First loop: A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j]
    i_offsets = tl.arange(0, BLOCK_SIZE)
    j_offsets = tl.arange(0, BLOCK_SIZE)
    
    for i_block in range(0, N, BLOCK_SIZE):
        i_current = i_block + i_offsets
        i_mask = i_current < N
        
        u1_vals = tl.load(u1_ptr + i_current, mask=i_mask, other=0.0)
        u2_vals = tl.load(u2_ptr + i_current, mask=i_mask, other=0.0)
        
        for j_block in range(0, N, BLOCK_SIZE):
            j_current = j_block + j_offsets
            j_mask = j_current < N
            
            v1_vals = tl.load(v1_ptr + j_current, mask=j_mask, other=0.0)
            v2_vals = tl.load(v2_ptr + j_current, mask=j_mask, other=0.0)
            
            # Compute outer product updates for this block
            for i_local in range(BLOCK_SIZE):
                i_val = i_block + i_local
                if i_val >= N:
                    continue
                
                u1_i = u1_vals[i_local] if i_val < N else 0.0
                u2_i = u2_vals[i_local] if i_val < N else 0.0
                
                A_row_ptr = A_ptr + i_val * N
                A_vals = tl.load(A_row_ptr + j_current, mask=j_mask, other=0.0)
                
                update_vals = u1_i * v1_vals + u2_i * v2_vals
                new_A_vals = A_vals + update_vals
                
                tl.store(A_row_ptr + j_current, new_A_vals, mask=j_mask)
    
    # Second loop: x[i] = x[i] + beta * A[j][i] * y[j]
    for i_block in range(0, N, BLOCK_SIZE):
        i_current = i_block + i_offsets
        i_mask = i_current < N
        
        x_vals = tl.load(x_ptr + i_current, mask=i_mask, other=0.0)
        
        for j_block in range(0, N, BLOCK_SIZE):
            j_current = j_block + j_offsets
            j_mask = j_current < N
            
            y_vals = tl.load(y_ptr + j_current, mask=j_mask, other=0.0)
            
            for j_local in range(BLOCK_SIZE):
                j_val = j_block + j_local
                if j_val >= N:
                    continue
                
                y_j = y_vals[j_local] if j_val < N else 0.0
                
                A_col_indices = j_val * N + i_current
                col_mask = i_mask
                A_col_vals = tl.load(A_ptr + A_col_indices, mask=col_mask, other=0.0)
                
                x_vals = x_vals + beta * A_col_vals * y_j
        
        tl.store(x_ptr + i_current, x_vals, mask=i_mask)
    
    # Third loop: x[i] = x[i] + z[i]
    for i_block in range(0, N, BLOCK_SIZE):
        i_current = i_block + i_offsets
        i_mask = i_current < N
        
        x_vals = tl.load(x_ptr + i_current, mask=i_mask, other=0.0)
        z_vals = tl.load(z_ptr + i_current, mask=i_mask, other=0.0)
        
        new_x_vals = x_vals + z_vals
        
        tl.store(x_ptr + i_current, new_x_vals, mask=i_mask)
    
    # Fourth loop: w[i] = w[i] + alpha * A[i][j] * x[j]
    for i_block in range(0, N, BLOCK_SIZE):
        i_current = i_block + i_offsets
        i_mask = i_current < N
        
        w_vals = tl.load(w_ptr + i_current, mask=i_mask, other=0.0)
        
        for j_block in range(0, N, BLOCK_SIZE):
            j_current = j_block + j_offsets
            j_mask = j_current < N
            
            x_vals = tl.load(x_ptr + j_current, mask=j_mask, other=0.0)
            
            for i_local in range(BLOCK_SIZE):
                i_val = i_block + i_local
                if i_val >= N:
                    continue
                
                A_row_ptr = A_ptr + i_val * N
                A_vals = tl.load(A_row_ptr + j_current, mask=j_mask, other=0.0)
                
                w_i = w_vals[i_local] if i_val < N else 0.0
                dot_product = tl.sum(alpha * A_vals * x_vals)
                new_w_i = w_i + dot_product
                
                w_vals = tl.where(i_local == tl.arange(0, BLOCK_SIZE), new_w_i, w_vals)
        
        tl.store(w_ptr + i_current, w_vals, mask=i_mask)

def gemver_triton(A, u1, u2, v1, v2, w, x, y, z, alpha, beta, N):
    BLOCK_SIZE = 32
    
    grid = (1,)
    
    gemver_kernel[grid](
        A, u1, u2, v1, v2, w, x, y, z,
        alpha, beta, N,
        BLOCK_SIZE=BLOCK_SIZE
    )