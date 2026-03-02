import triton
import triton.language as tl
import torch

@triton.jit
def gramschmidt_kernel(A, Q, R, M, N, BLOCK_SIZE: tl.constexpr):
    for k in range(N):
        # Compute norm squared using reduction
        block_offsets = tl.arange(0, BLOCK_SIZE)
        nrm = 0.0
        for block_start in range(0, M, BLOCK_SIZE):
            i_offsets = block_start + block_offsets
            mask = i_offsets < M
            a_vals = tl.load(A + i_offsets * N + k, mask=mask, other=0.0)
            nrm += tl.sum(a_vals * a_vals)
        
        # Compute R[k][k] = sqrt(nrm)
        r_kk = tl.sqrt(nrm)
        tl.store(R + k * N + k, r_kk)
        
        # Compute Q[:, k] = A[:, k] / R[k][k]
        for block_start in range(0, M, BLOCK_SIZE):
            i_offsets = block_start + block_offsets
            mask = i_offsets < M
            a_vals = tl.load(A + i_offsets * N + k, mask=mask)
            q_vals = a_vals / r_kk
            tl.store(Q + i_offsets * N + k, q_vals, mask=mask)
        
        # For each j > k
        for j in range(k + 1, N):
            # Compute R[k][j] = sum(Q[:, k] * A[:, j])
            r_kj = 0.0
            for block_start in range(0, M, BLOCK_SIZE):
                i_offsets = block_start + block_offsets
                mask = i_offsets < M
                q_vals = tl.load(Q + i_offsets * N + k, mask=mask, other=0.0)
                a_vals = tl.load(A + i_offsets * N + j, mask=mask, other=0.0)
                r_kj += tl.sum(q_vals * a_vals)
            
            # Store R[k][j]
            tl.store(R + k * N + j, r_kj)
            
            # Update A[:, j] = A[:, j] - Q[:, k] * R[k][j]
            for block_start in range(0, M, BLOCK_SIZE):
                i_offsets = block_start + block_offsets
                mask = i_offsets < M
                a_vals = tl.load(A + i_offsets * N + j, mask=mask)
                q_vals = tl.load(Q + i_offsets * N + k, mask=mask, other=0.0)
                updated_a = a_vals - q_vals * r_kj
                tl.store(A + i_offsets * N + j, updated_a, mask=mask)

def gramschmidt_triton(A, Q, R, M, N):
    BLOCK_SIZE = 64
    
    # Launch kernel with single thread block since algorithm is inherently sequential
    grid = (1,)
    gramschmidt_kernel[grid](A, Q, R, M, N, BLOCK_SIZE)