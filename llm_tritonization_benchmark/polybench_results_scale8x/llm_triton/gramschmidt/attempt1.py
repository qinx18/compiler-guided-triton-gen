import triton
import triton.language as tl
import torch

@triton.jit
def gramschmidt_kernel(A, Q, R, M: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr):
    # Define offsets once at start
    m_offsets = tl.arange(0, BLOCK_M)
    
    for k in range(N):
        # Compute norm of column k
        nrm = 0.0
        for m_start in range(0, M, BLOCK_M):
            current_m_offsets = m_start + m_offsets
            mask_m = current_m_offsets < M
            
            # Load A[i][k] for current block
            a_ptrs = A + current_m_offsets * N + k
            a_vals = tl.load(a_ptrs, mask=mask_m, other=0.0)
            
            # Accumulate squared values
            nrm += tl.sum(a_vals * a_vals)
        
        # R[k][k] = sqrt(nrm)
        r_kk = tl.sqrt(nrm)
        r_ptr = R + k * N + k
        tl.store(r_ptr, r_kk)
        
        # Compute Q[i][k] = A[i][k] / R[k][k]
        for m_start in range(0, M, BLOCK_M):
            current_m_offsets = m_start + m_offsets
            mask_m = current_m_offsets < M
            
            # Load A[i][k]
            a_ptrs = A + current_m_offsets * N + k
            a_vals = tl.load(a_ptrs, mask=mask_m, other=0.0)
            
            # Compute and store Q[i][k]
            q_vals = a_vals / r_kk
            q_ptrs = Q + current_m_offsets * N + k
            tl.store(q_ptrs, q_vals, mask=mask_m)
        
        # Process columns j > k
        for j in range(k + 1, N):
            # Compute R[k][j] = sum(Q[i][k] * A[i][j])
            r_kj = 0.0
            for m_start in range(0, M, BLOCK_M):
                current_m_offsets = m_start + m_offsets
                mask_m = current_m_offsets < M
                
                # Load Q[i][k] and A[i][j]
                q_ptrs = Q + current_m_offsets * N + k
                a_ptrs = A + current_m_offsets * N + j
                
                q_vals = tl.load(q_ptrs, mask=mask_m, other=0.0)
                a_vals = tl.load(a_ptrs, mask=mask_m, other=0.0)
                
                # Accumulate dot product
                r_kj += tl.sum(q_vals * a_vals)
            
            # Store R[k][j]
            r_ptr = R + k * N + j
            tl.store(r_ptr, r_kj)
            
            # Update A[i][j] = A[i][j] - Q[i][k] * R[k][j]
            for m_start in range(0, M, BLOCK_M):
                current_m_offsets = m_start + m_offsets
                mask_m = current_m_offsets < M
                
                # Load Q[i][k] and A[i][j]
                q_ptrs = Q + current_m_offsets * N + k
                a_ptrs = A + current_m_offsets * N + j
                
                q_vals = tl.load(q_ptrs, mask=mask_m, other=0.0)
                a_vals = tl.load(a_ptrs, mask=mask_m, other=0.0)
                
                # Update and store A[i][j]
                new_a_vals = a_vals - q_vals * r_kj
                tl.store(a_ptrs, new_a_vals, mask=mask_m)

def gramschmidt_triton(A, Q, R, M, N):
    BLOCK_M = 128
    
    # Initialize R to zero
    R.fill_(0.0)
    
    # Launch single kernel instance
    grid = (1,)
    gramschmidt_kernel[grid](A, Q, R, M, N, BLOCK_M)