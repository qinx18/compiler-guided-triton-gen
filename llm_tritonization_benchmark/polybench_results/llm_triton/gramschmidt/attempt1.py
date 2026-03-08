import triton
import triton.language as tl
import torch

@triton.jit
def gramschmidt_kernel(A_ptr, Q_ptr, R_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr):
    for k in range(N):
        # Compute norm: nrm = sum(A[i][k] * A[i][k]) for i in [0, M)
        nrm = 0.0
        for block_start in range(0, M, BLOCK_M):
            i_offsets = tl.arange(0, BLOCK_M)
            i_indices = block_start + i_offsets
            mask = i_indices < M
            
            # Load A[i][k]
            a_offsets = i_indices * N + k
            a_vals = tl.load(A_ptr + a_offsets, mask=mask, other=0.0)
            
            # Sum a_vals * a_vals
            block_sum = tl.sum(a_vals * a_vals)
            nrm += block_sum
        
        # R[k][k] = sqrt(nrm)
        r_kk = tl.sqrt(nrm)
        r_kk_offset = k * N + k
        tl.store(R_ptr + r_kk_offset, r_kk)
        
        # Q[i][k] = A[i][k] / R[k][k] for i in [0, M)
        for block_start in range(0, M, BLOCK_M):
            i_offsets = tl.arange(0, BLOCK_M)
            i_indices = block_start + i_offsets
            mask = i_indices < M
            
            # Load A[i][k]
            a_offsets = i_indices * N + k
            a_vals = tl.load(A_ptr + a_offsets, mask=mask, other=0.0)
            
            # Compute Q[i][k] = A[i][k] / R[k][k]
            q_vals = a_vals / r_kk
            
            # Store Q[i][k]
            q_offsets = i_indices * N + k
            tl.store(Q_ptr + q_offsets, q_vals, mask=mask)
        
        # For j in [k+1, N)
        for j in range(k + 1, N):
            # Compute R[k][j] = sum(Q[i][k] * A[i][j]) for i in [0, M)
            r_kj = 0.0
            for block_start in range(0, M, BLOCK_M):
                i_offsets = tl.arange(0, BLOCK_M)
                i_indices = block_start + i_offsets
                mask = i_indices < M
                
                # Load Q[i][k] and A[i][j]
                q_offsets = i_indices * N + k
                a_offsets = i_indices * N + j
                q_vals = tl.load(Q_ptr + q_offsets, mask=mask, other=0.0)
                a_vals = tl.load(A_ptr + a_offsets, mask=mask, other=0.0)
                
                # Sum Q[i][k] * A[i][j]
                block_sum = tl.sum(q_vals * a_vals)
                r_kj += block_sum
            
            # Store R[k][j]
            r_kj_offset = k * N + j
            tl.store(R_ptr + r_kj_offset, r_kj)
            
            # Update A[i][j] = A[i][j] - Q[i][k] * R[k][j] for i in [0, M)
            for block_start in range(0, M, BLOCK_M):
                i_offsets = tl.arange(0, BLOCK_M)
                i_indices = block_start + i_offsets
                mask = i_indices < M
                
                # Load Q[i][k] and A[i][j]
                q_offsets = i_indices * N + k
                a_offsets = i_indices * N + j
                q_vals = tl.load(Q_ptr + q_offsets, mask=mask, other=0.0)
                a_vals = tl.load(A_ptr + a_offsets, mask=mask, other=0.0)
                
                # Update A[i][j]
                new_a_vals = a_vals - q_vals * r_kj
                tl.store(A_ptr + a_offsets, new_a_vals, mask=mask)

def gramschmidt_triton(A, Q, R, M, N):
    BLOCK_M = 64
    
    # Launch single kernel with grid=(1,)
    grid = (1,)
    gramschmidt_kernel[grid](
        A, Q, R,
        M=M, N=N, BLOCK_M=BLOCK_M
    )