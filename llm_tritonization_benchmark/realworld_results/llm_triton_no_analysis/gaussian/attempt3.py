import triton
import triton.language as tl
import torch

@triton.jit
def gaussian_kernel(a_ptr, b_ptr, m_ptr, x_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Forward elimination
    for t in range(N - 1):
        # Phase 1: Compute multipliers
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        
        offsets = tl.arange(0, BLOCK_SIZE)
        i_vals = block_start + offsets + (t + 1)
        mask = i_vals < N
        
        if tl.sum(mask) > 0:
            # Load a[i][t] and a[t][t]
            a_it_offsets = i_vals * N + t
            a_it = tl.load(a_ptr + a_it_offsets, mask=mask)
            
            a_tt = tl.load(a_ptr + t * N + t)
            
            # Compute multipliers m[i][t] = a[i][t] / a[t][t]
            m_it = a_it / a_tt
            
            # Store multipliers
            m_it_offsets = i_vals * N + t
            tl.store(m_ptr + m_it_offsets, m_it, mask=mask)
        
        # Phase 2: Eliminate column t
        for j in range(t, N):
            if tl.sum(mask) > 0:
                # Load multipliers and a[t][j]
                m_it = tl.load(m_ptr + i_vals * N + t, mask=mask)
                a_tj = tl.load(a_ptr + t * N + j)
                
                # Update a[i][j] -= m[i][t] * a[t][j]
                a_ij_offsets = i_vals * N + j
                a_ij = tl.load(a_ptr + a_ij_offsets, mask=mask)
                a_ij_new = a_ij - m_it * a_tj
                tl.store(a_ptr + a_ij_offsets, a_ij_new, mask=mask)
        
        # Update b[i] -= m[i][t] * b[t]
        if tl.sum(mask) > 0:
            m_it = tl.load(m_ptr + i_vals * N + t, mask=mask)
            b_t = tl.load(b_ptr + t)
            b_i = tl.load(b_ptr + i_vals, mask=mask)
            b_i_new = b_i - m_it * b_t
            tl.store(b_ptr + i_vals, b_i_new, mask=mask)
    
    # Back substitution (only first thread does this)
    if tl.program_id(0) == 0:
        # x[N-1] = b[N-1] / a[N-1][N-1]
        last_idx = N - 1
        b_last = tl.load(b_ptr + last_idx)
        a_last = tl.load(a_ptr + last_idx * N + last_idx)
        x_last = b_last / a_last
        tl.store(x_ptr + last_idx, x_last)
        
        # Process remaining rows
        for i in range(N - 2, -1, -1):
            sum_val = tl.load(b_ptr + i)
            
            for j in range(i + 1, N):
                a_ij = tl.load(a_ptr + i * N + j)
                x_j = tl.load(x_ptr + j)
                sum_val = sum_val - a_ij * x_j
            
            a_ii = tl.load(a_ptr + i * N + i)
            x_i = sum_val / a_ii
            tl.store(x_ptr + i, x_i)

def gaussian_triton(a, b, m, x, N):
    BLOCK_SIZE = 64
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    gaussian_kernel[grid](
        a, b, m, x, N, BLOCK_SIZE
    )