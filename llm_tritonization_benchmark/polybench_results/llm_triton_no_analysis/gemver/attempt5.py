import triton
import triton.language as tl

@triton.jit
def gemver_kernel(A_ptr, u1_ptr, u2_ptr, v1_ptr, v2_ptr, w_ptr, x_ptr, y_ptr, z_ptr, alpha, beta, N, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # First loop: A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j]
    for i_block in range(0, N, BLOCK_SIZE):
        i_current = i_block + offsets
        i_mask = i_current < N
        
        u1_vals = tl.load(u1_ptr + i_current, mask=i_mask, other=0.0)
        u2_vals = tl.load(u2_ptr + i_current, mask=i_mask, other=0.0)
        
        for j_block in range(0, N, BLOCK_SIZE):
            j_current = j_block + offsets
            j_mask = j_current < N
            
            v1_vals = tl.load(v1_ptr + j_current, mask=j_mask, other=0.0)
            v2_vals = tl.load(v2_ptr + j_current, mask=j_mask, other=0.0)
            
            # Process each row in the i_block
            for i_idx in range(BLOCK_SIZE):
                i_val = i_block + i_idx
                if i_val < N:
                    A_row_ptr = A_ptr + i_val * N
                    A_vals = tl.load(A_row_ptr + j_current, mask=j_mask, other=0.0)
                    
                    # Extract scalar values for broadcasting
                    u1_scalar = tl.load(u1_ptr + i_val)
                    u2_scalar = tl.load(u2_ptr + i_val)
                    
                    update_vals = u1_scalar * v1_vals + u2_scalar * v2_vals
                    new_A_vals = A_vals + update_vals
                    
                    tl.store(A_row_ptr + j_current, new_A_vals, mask=j_mask)
    
    # Second loop: x[i] = x[i] + beta * A[j][i] * y[j]
    for i_block in range(0, N, BLOCK_SIZE):
        i_current = i_block + offsets
        i_mask = i_current < N
        
        x_vals = tl.load(x_ptr + i_current, mask=i_mask, other=0.0)
        
        for j in range(N):
            y_scalar = tl.load(y_ptr + j)
            A_col_ptrs = A_ptr + j * N + i_current
            A_col_vals = tl.load(A_col_ptrs, mask=i_mask, other=0.0)
            x_vals = x_vals + beta * A_col_vals * y_scalar
        
        tl.store(x_ptr + i_current, x_vals, mask=i_mask)
    
    # Third loop: x[i] = x[i] + z[i]
    for i_block in range(0, N, BLOCK_SIZE):
        i_current = i_block + offsets
        i_mask = i_current < N
        
        x_vals = tl.load(x_ptr + i_current, mask=i_mask, other=0.0)
        z_vals = tl.load(z_ptr + i_current, mask=i_mask, other=0.0)
        
        new_x_vals = x_vals + z_vals
        
        tl.store(x_ptr + i_current, new_x_vals, mask=i_mask)
    
    # Fourth loop: w[i] = w[i] + alpha * A[i][j] * x[j]
    for i_block in range(0, N, BLOCK_SIZE):
        i_current = i_block + offsets
        i_mask = i_current < N
        
        w_vals = tl.load(w_ptr + i_current, mask=i_mask, other=0.0)
        
        for i_idx in range(BLOCK_SIZE):
            i_val = i_block + i_idx
            if i_val < N:
                A_row_ptr = A_ptr + i_val * N
                dot_sum = 0.0
                for j_block in range(0, N, BLOCK_SIZE):
                    j_current = j_block + offsets
                    j_mask = j_current < N
                    
                    A_vals = tl.load(A_row_ptr + j_current, mask=j_mask, other=0.0)
                    x_vals = tl.load(x_ptr + j_current, mask=j_mask, other=0.0)
                    
                    dot_sum += tl.sum(A_vals * x_vals)
                
                # Update single element
                current_w = tl.load(w_ptr + i_val)
                new_w = current_w + alpha * dot_sum
                tl.store(w_ptr + i_val, new_w)

def gemver_triton(A, u1, u2, v1, v2, w, x, y, z, alpha, beta, N):
    BLOCK_SIZE = 32
    
    grid = (1,)
    
    gemver_kernel[grid](
        A, u1, u2, v1, v2, w, x, y, z,
        alpha, beta, N,
        BLOCK_SIZE=BLOCK_SIZE
    )