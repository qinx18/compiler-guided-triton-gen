import triton
import triton.language as tl
import torch

@triton.jit
def cholesky_kernel(A_ptr, N: tl.constexpr):
    # Single thread processes entire matrix sequentially
    tid = tl.program_id(0)
    
    if tid != 0:
        return
    
    for i in range(N):
        # j < i case
        for j in range(i):
            for k in range(j):
                # A[i][j] -= A[i][k] * A[j][k]
                a_ik = tl.load(A_ptr + i * N + k)
                a_jk = tl.load(A_ptr + j * N + k)
                a_ij = tl.load(A_ptr + i * N + j)
                a_ij -= a_ik * a_jk
                tl.store(A_ptr + i * N + j, a_ij)
            
            # A[i][j] /= A[j][j]
            a_ij = tl.load(A_ptr + i * N + j)
            a_jj = tl.load(A_ptr + j * N + j)
            a_ij /= a_jj
            tl.store(A_ptr + i * N + j, a_ij)
        
        # i == j case (diagonal)
        for k in range(i):
            # A[i][i] -= A[i][k] * A[i][k]
            a_ik = tl.load(A_ptr + i * N + k)
            a_ii = tl.load(A_ptr + i * N + i)
            a_ii -= a_ik * a_ik
            tl.store(A_ptr + i * N + i, a_ii)
        
        # A[i][i] = sqrt(A[i][i])
        a_ii = tl.load(A_ptr + i * N + i)
        a_ii = tl.sqrt(a_ii)
        tl.store(A_ptr + i * N + i, a_ii)

def cholesky_triton(A, N):
    cholesky_kernel[(1,)](A, N)