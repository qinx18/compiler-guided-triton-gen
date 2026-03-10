import triton
import triton.language as tl
import torch

@triton.jit
def gramschmidt_kernel(A_ptr, Q_ptr, R_ptr, M: tl.constexpr, N: tl.constexpr):
    for k in range(N):
        # Compute norm squared
        nrm = 0.0
        for i in range(M):
            a_val = tl.load(A_ptr + i * N + k)
            nrm += a_val * a_val
        
        # Compute R[k][k] = sqrt(nrm)
        r_kk = tl.sqrt(nrm)
        tl.store(R_ptr + k * N + k, r_kk)
        
        # Compute Q[:, k] = A[:, k] / R[k][k]
        for i in range(M):
            a_val = tl.load(A_ptr + i * N + k)
            q_val = a_val / r_kk
            tl.store(Q_ptr + i * N + k, q_val)
        
        # Update remaining columns
        for j in range(k + 1, N):
            # Compute R[k][j] = Q[:, k]^T * A[:, j]
            r_kj = 0.0
            for i in range(M):
                q_val = tl.load(Q_ptr + i * N + k)
                a_val = tl.load(A_ptr + i * N + j)
                r_kj += q_val * a_val
            
            tl.store(R_ptr + k * N + j, r_kj)
            
            # Update A[:, j] = A[:, j] - Q[:, k] * R[k][j]
            for i in range(M):
                q_val = tl.load(Q_ptr + i * N + k)
                a_val = tl.load(A_ptr + i * N + j)
                new_a_val = a_val - q_val * r_kj
                tl.store(A_ptr + i * N + j, new_a_val)

def gramschmidt_triton(A, Q, R, M, N):
    grid = (1,)
    gramschmidt_kernel[grid](A, Q, R, M, N)