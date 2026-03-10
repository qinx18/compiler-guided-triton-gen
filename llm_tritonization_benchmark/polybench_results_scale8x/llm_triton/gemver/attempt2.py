import triton
import triton.language as tl
import torch

@triton.jit
def gemver_kernel1(A, u1, u2, v1, v2, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    
    if i < N:
        j_offsets = tl.arange(0, BLOCK_SIZE)
        u1_i = tl.load(u1 + i)
        u2_i = tl.load(u2 + i)
        
        for j_start in range(0, N, BLOCK_SIZE):
            j_indices = j_start + j_offsets
            mask = j_indices < N
            
            A_indices = i * N + j_indices
            v1_vals = tl.load(v1 + j_indices, mask=mask)
            v2_vals = tl.load(v2 + j_indices, mask=mask)
            A_vals = tl.load(A + A_indices, mask=mask)
            
            new_A_vals = A_vals + u1_i * v1_vals + u2_i * v2_vals
            tl.store(A + A_indices, new_A_vals, mask=mask)

@triton.jit
def gemver_kernel2(A, x, y, beta, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    
    if i < N:
        j_offsets = tl.arange(0, BLOCK_SIZE)
        
        x_acc = 0.0
        
        for j_start in range(0, N, BLOCK_SIZE):
            j_indices = j_start + j_offsets
            mask = j_indices < N
            
            A_indices = j_indices * N + i
            A_vals = tl.load(A + A_indices, mask=mask)
            y_block = tl.load(y + j_indices, mask=mask)
            
            products = beta * A_vals * y_block
            x_acc += tl.sum(products)
        
        x_old = tl.load(x + i)
        tl.store(x + i, x_old + x_acc)

@triton.jit
def gemver_kernel3(x, z, N: tl.constexpr):
    i = tl.program_id(0)
    
    if i < N:
        x_val = tl.load(x + i)
        z_val = tl.load(z + i)
        tl.store(x + i, x_val + z_val)

@triton.jit
def gemver_kernel4(A, w, x, alpha, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    
    if i < N:
        j_offsets = tl.arange(0, BLOCK_SIZE)
        
        w_acc = 0.0
        
        for j_start in range(0, N, BLOCK_SIZE):
            j_indices = j_start + j_offsets
            mask = j_indices < N
            
            A_indices = i * N + j_indices
            A_vals = tl.load(A + A_indices, mask=mask)
            x_vals = tl.load(x + j_indices, mask=mask)
            
            products = alpha * A_vals * x_vals
            w_acc += tl.sum(products)
        
        w_old = tl.load(w + i)
        tl.store(w + i, w_old + w_acc)

def gemver_triton(A, u1, u2, v1, v2, w, x, y, z, alpha, beta, N):
    BLOCK_SIZE = 128
    grid = (N,)
    
    # Phase 1: Update A matrix
    gemver_kernel1[grid](A, u1, u2, v1, v2, N, BLOCK_SIZE)
    
    # Phase 2: Update x with matrix-vector product
    gemver_kernel2[grid](A, x, y, beta, N, BLOCK_SIZE)
    
    # Phase 3: Add z to x
    gemver_kernel3[grid](x, z, N)
    
    # Phase 4: Update w with matrix-vector product
    gemver_kernel4[grid](A, w, x, alpha, N, BLOCK_SIZE)