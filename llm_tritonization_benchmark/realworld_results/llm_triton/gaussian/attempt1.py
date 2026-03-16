import triton
import triton.language as tl

@triton.jit
def gaussian_kernel(a_ptr, b_ptr, m_ptr, x_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Forward elimination
    for t in range(N - 1):
        # Phase 1: Compute multipliers (Fan1)
        i_vals = pid * BLOCK_SIZE + offsets + t + 1
        mask = i_vals < N
        
        if tl.sum(mask) > 0:
            # Load pivot element
            pivot = tl.load(a_ptr + t * N + t)
            
            # Load a[i][t] values for multiplier computation
            a_indices = i_vals * N + t
            a_vals = tl.load(a_ptr + a_indices, mask=mask)
            
            # Compute multipliers m[i][t] = a[i][t] / a[t][t]
            multipliers = a_vals / pivot
            
            # Store multipliers
            m_indices = i_vals * N + t
            tl.store(m_ptr + m_indices, multipliers, mask=mask)
            
            # Phase 2: Eliminate column t (Fan2)
            for j in range(t, N):
                # Load a[t][j] (pivot row element)
                pivot_row_val = tl.load(a_ptr + t * N + j)
                
                # Load a[i][j] values to be updated
                a_ij_indices = i_vals * N + j
                a_ij_vals = tl.load(a_ptr + a_ij_indices, mask=mask)
                
                # Update: a[i][j] -= m[i][t] * a[t][j]
                updated_vals = a_ij_vals - multipliers * pivot_row_val
                
                # Store updated values
                tl.store(a_ptr + a_ij_indices, updated_vals, mask=mask)
            
            # Update b[i] -= m[i][t] * b[t]
            pivot_b_val = tl.load(b_ptr + t)
            b_vals = tl.load(b_ptr + i_vals, mask=mask)
            updated_b_vals = b_vals - multipliers * pivot_b_val
            tl.store(b_ptr + i_vals, updated_b_vals, mask=mask)
    
    # Back substitution (only CTA 0)
    if tl.program_id(0) == 0:
        # x[N-1] = b[N-1] / a[N-1][N-1]
        last_idx = N - 1
        last_b = tl.load(b_ptr + last_idx)
        last_a = tl.load(a_ptr + last_idx * N + last_idx)
        tl.store(x_ptr + last_idx, last_b / last_a)
        
        # Back substitution loop
        for i in range(N - 2, -1, -1):
            sum_val = tl.load(b_ptr + i)
            for j in range(i + 1, N):
                a_ij = tl.load(a_ptr + i * N + j)
                x_j = tl.load(x_ptr + j)
                sum_val -= a_ij * x_j
            diagonal_val = tl.load(a_ptr + i * N + i)
            tl.store(x_ptr + i, sum_val / diagonal_val)

def gaussian_triton(a, b, m, x, N):
    BLOCK_SIZE = 64
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    gaussian_kernel[grid](a, b, m, x, N=N, BLOCK_SIZE=BLOCK_SIZE)