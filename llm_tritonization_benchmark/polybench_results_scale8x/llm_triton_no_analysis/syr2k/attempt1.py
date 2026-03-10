import triton
import triton.language as tl
import torch

@triton.jit
def syr2k_kernel(A_ptr, B_ptr, C_ptr, alpha, beta, M: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    
    if row >= N:
        return
    
    # First loop: C[row][j] *= beta for j <= row
    j_offsets = tl.arange(0, BLOCK_SIZE)
    for j_start in range(0, row + 1, BLOCK_SIZE):
        current_j_offsets = j_start + j_offsets
        j_mask = (current_j_offsets <= row) & (current_j_offsets < N)
        
        if tl.sum(j_mask.to(tl.int32)) > 0:
            c_indices = row * N + current_j_offsets
            c_vals = tl.load(C_ptr + c_indices, mask=j_mask, other=0.0)
            c_vals = c_vals * beta
            tl.store(C_ptr + c_indices, c_vals, mask=j_mask)
    
    # Second nested loop: for k in range(M), for j <= row
    for k in range(M):
        # Load B[row][k] and A[row][k] once per k iteration
        b_row_k = tl.load(B_ptr + row * M + k)
        a_row_k = tl.load(A_ptr + row * M + k)
        
        for j_start in range(0, row + 1, BLOCK_SIZE):
            current_j_offsets = j_start + j_offsets
            j_mask = (current_j_offsets <= row) & (current_j_offsets < N)
            
            if tl.sum(j_mask.to(tl.int32)) > 0:
                # Load A[j][k] and B[j][k]
                a_indices = current_j_offsets * M + k
                b_indices = current_j_offsets * M + k
                
                a_j_k = tl.load(A_ptr + a_indices, mask=j_mask, other=0.0)
                b_j_k = tl.load(B_ptr + b_indices, mask=j_mask, other=0.0)
                
                # Load current C[row][j]
                c_indices = row * N + current_j_offsets
                c_vals = tl.load(C_ptr + c_indices, mask=j_mask, other=0.0)
                
                # Compute: C[row][j] += A[j][k]*alpha*B[row][k] + B[j][k]*alpha*A[row][k]
                update = a_j_k * alpha * b_row_k + b_j_k * alpha * a_row_k
                c_vals = c_vals + update
                
                # Store back
                tl.store(C_ptr + c_indices, c_vals, mask=j_mask)

def syr2k_triton(A, B, C, alpha, beta, M, N):
    BLOCK_SIZE = 64
    
    grid = (N,)
    
    syr2k_kernel[grid](
        A, B, C,
        alpha, beta,
        M, N, BLOCK_SIZE
    )