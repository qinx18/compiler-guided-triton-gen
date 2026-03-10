import triton
import triton.language as tl
import torch

@triton.jit
def cholesky_kernel(A_ptr, N: tl.constexpr):
    # Sequential implementation - single thread processes entire matrix
    for i in range(N):
        # j < i case
        for j in range(i):
            # Compute A[i][j] -= sum(A[i][k] * A[j][k] for k in range(j))
            for k in range(j):
                i_k_offset = i * N + k
                j_k_offset = j * N + k
                i_j_offset = i * N + j
                
                a_i_k = tl.load(A_ptr + i_k_offset)
                a_j_k = tl.load(A_ptr + j_k_offset)
                a_i_j = tl.load(A_ptr + i_j_offset)
                
                a_i_j = a_i_j - a_i_k * a_j_k
                tl.store(A_ptr + i_j_offset, a_i_j)
            
            # A[i][j] /= A[j][j]
            i_j_offset = i * N + j
            j_j_offset = j * N + j
            
            a_i_j = tl.load(A_ptr + i_j_offset)
            a_j_j = tl.load(A_ptr + j_j_offset)
            a_i_j = a_i_j / a_j_j
            tl.store(A_ptr + i_j_offset, a_i_j)
        
        # i == j case
        # Compute A[i][i] -= sum(A[i][k] * A[i][k] for k in range(i))
        for k in range(i):
            i_k_offset = i * N + k
            i_i_offset = i * N + i
            
            a_i_k = tl.load(A_ptr + i_k_offset)
            a_i_i = tl.load(A_ptr + i_i_offset)
            
            a_i_i = a_i_i - a_i_k * a_i_k
            tl.store(A_ptr + i_i_offset, a_i_i)
        
        # A[i][i] = sqrt(A[i][i])
        i_i_offset = i * N + i
        a_i_i = tl.load(A_ptr + i_i_offset)
        a_i_i = tl.sqrt(a_i_i)
        tl.store(A_ptr + i_i_offset, a_i_i)

def cholesky_triton(A, N):
    # Launch single thread since computation is inherently sequential
    grid = (1,)
    cholesky_kernel[grid](A, N)