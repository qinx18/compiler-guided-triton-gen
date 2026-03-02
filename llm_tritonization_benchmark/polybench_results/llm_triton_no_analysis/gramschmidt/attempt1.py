import triton
import triton.language as tl
import torch

@triton.jit
def gramschmidt_kernel(
    A_ptr, Q_ptr, R_ptr,
    M, N,
    stride_A_m, stride_A_n,
    stride_Q_m, stride_Q_n,
    stride_R_m, stride_R_n,
):
    for k in range(N):
        # Compute norm: nrm = sum(A[i][k] * A[i][k])
        nrm = 0.0
        for i in range(M):
            a_val = tl.load(A_ptr + i * stride_A_m + k * stride_A_n)
            nrm += a_val * a_val
        
        # R[k][k] = sqrt(nrm)
        r_kk = tl.sqrt(nrm)
        tl.store(R_ptr + k * stride_R_m + k * stride_R_n, r_kk)
        
        # Q[i][k] = A[i][k] / R[k][k]
        for i in range(M):
            a_val = tl.load(A_ptr + i * stride_A_m + k * stride_A_n)
            q_val = a_val / r_kk
            tl.store(Q_ptr + i * stride_Q_m + k * stride_Q_n, q_val)
        
        # For each subsequent column j
        for j in range(k + 1, N):
            # R[k][j] = sum(Q[i][k] * A[i][j])
            r_kj = 0.0
            for i in range(M):
                q_val = tl.load(Q_ptr + i * stride_Q_m + k * stride_Q_n)
                a_val = tl.load(A_ptr + i * stride_A_m + j * stride_A_n)
                r_kj += q_val * a_val
            tl.store(R_ptr + k * stride_R_m + j * stride_R_n, r_kj)
            
            # A[i][j] = A[i][j] - Q[i][k] * R[k][j]
            for i in range(M):
                q_val = tl.load(Q_ptr + i * stride_Q_m + k * stride_Q_n)
                a_val = tl.load(A_ptr + i * stride_A_m + j * stride_A_n)
                new_a_val = a_val - q_val * r_kj
                tl.store(A_ptr + i * stride_A_m + j * stride_A_n, new_a_val)

def gramschmidt_triton(A, Q, R, M, N):
    grid = (1,)
    
    gramschmidt_kernel[grid](
        A, Q, R,
        M, N,
        A.stride(0), A.stride(1),
        Q.stride(0), Q.stride(1),
        R.stride(0), R.stride(1),
    )