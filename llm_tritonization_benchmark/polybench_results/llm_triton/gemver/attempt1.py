import triton
import triton.language as tl
import torch

@triton.jit
def gemver_kernel_1(A_ptr, u1_ptr, u2_ptr, v1_ptr, v2_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Phase 1: A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j]
    row = tl.program_id(0)
    
    if row < N:
        u1_val = tl.load(u1_ptr + row)
        u2_val = tl.load(u2_ptr + row)
        
        for col_start in range(0, N, BLOCK_SIZE):
            col_offsets = tl.arange(0, BLOCK_SIZE) + col_start
            col_mask = col_offsets < N
            
            # Load current A values
            a_offsets = row * N + col_offsets
            a_vals = tl.load(A_ptr + a_offsets, mask=col_mask)
            
            # Load v1 and v2 values
            v1_vals = tl.load(v1_ptr + col_offsets, mask=col_mask)
            v2_vals = tl.load(v2_ptr + col_offsets, mask=col_mask)
            
            # Update A values
            new_a_vals = a_vals + u1_val * v1_vals + u2_val * v2_vals
            
            # Store back
            tl.store(A_ptr + a_offsets, new_a_vals, mask=col_mask)

@triton.jit
def gemver_kernel_2(A_ptr, x_ptr, y_ptr, beta, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Phase 2: x[i] = x[i] + beta * A[j][i] * y[j]
    col = tl.program_id(0)
    
    if col < N:
        x_val = tl.load(x_ptr + col)
        
        for row_start in range(0, N, BLOCK_SIZE):
            row_offsets = tl.arange(0, BLOCK_SIZE) + row_start
            row_mask = row_offsets < N
            
            # Load A column values (A[j][i] = A[row][col])
            a_offsets = row_offsets * N + col
            a_vals = tl.load(A_ptr + a_offsets, mask=row_mask)
            
            # Load y values
            y_vals = tl.load(y_ptr + row_offsets, mask=row_mask)
            
            # Accumulate
            products = beta * a_vals * y_vals
            x_val += tl.sum(tl.where(row_mask, products, 0.0))
        
        tl.store(x_ptr + col, x_val)

@triton.jit
def gemver_kernel_3(x_ptr, z_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Phase 3: x[i] = x[i] + z[i]
    idx_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE) + idx_start
    mask = offsets < N
    
    x_vals = tl.load(x_ptr + offsets, mask=mask)
    z_vals = tl.load(z_ptr + offsets, mask=mask)
    
    result = x_vals + z_vals
    tl.store(x_ptr + offsets, result, mask=mask)

@triton.jit
def gemver_kernel_4(A_ptr, w_ptr, x_ptr, alpha, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Phase 4: w[i] = w[i] + alpha * A[i][j] * x[j]
    row = tl.program_id(0)
    
    if row < N:
        w_val = tl.load(w_ptr + row)
        
        for col_start in range(0, N, BLOCK_SIZE):
            col_offsets = tl.arange(0, BLOCK_SIZE) + col_start
            col_mask = col_offsets < N
            
            # Load A row values
            a_offsets = row * N + col_offsets
            a_vals = tl.load(A_ptr + a_offsets, mask=col_mask)
            
            # Load x values
            x_vals = tl.load(x_ptr + col_offsets, mask=col_mask)
            
            # Accumulate
            products = alpha * a_vals * x_vals
            w_val += tl.sum(tl.where(col_mask, products, 0.0))
        
        tl.store(w_ptr + row, w_val)

def gemver_triton(A, u1, u2, v1, v2, w, x, y, z, alpha, beta, N):
    BLOCK_SIZE = 128
    
    # Phase 1: Update A matrix
    grid1 = (N,)
    gemver_kernel_1[grid1](A, u1, u2, v1, v2, N, BLOCK_SIZE)
    
    # Phase 2: Update x with A^T * y
    grid2 = (N,)
    gemver_kernel_2[grid2](A, x, y, beta, N, BLOCK_SIZE)
    
    # Phase 3: Add z to x
    grid3 = (triton.cdiv(N, BLOCK_SIZE),)
    gemver_kernel_3[grid3](x, z, N, BLOCK_SIZE)
    
    # Phase 4: Update w with A * x
    grid4 = (N,)
    gemver_kernel_4[grid4](A, w, x, alpha, N, BLOCK_SIZE)