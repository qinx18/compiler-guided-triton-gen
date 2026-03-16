import triton
import triton.language as tl
import torch

@triton.jit
def gaussian_elimination_phase1_kernel(a_ptr, m_ptr, t: tl.constexpr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Phase 1: Compute multipliers for column t
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets + (t + 1)
    
    mask = i_offsets < N
    
    # Load a[i][t] and a[t][t]
    a_it_offsets = i_offsets * N + t
    a_it = tl.load(a_ptr + a_it_offsets, mask=mask)
    
    a_tt_offset = t * N + t
    a_tt = tl.load(a_ptr + a_tt_offset)
    
    # Compute multipliers m[i][t] = a[i][t] / a[t][t]
    m_it = a_it / a_tt
    
    # Store multipliers
    m_it_offsets = i_offsets * N + t
    tl.store(m_ptr + m_it_offsets, m_it, mask=mask)

@triton.jit
def gaussian_elimination_phase2_kernel(a_ptr, b_ptr, m_ptr, t: tl.constexpr, N: tl.constexpr, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr):
    # Phase 2: Eliminate column t
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    block_start_i = pid_i * BLOCK_SIZE_I
    block_start_j = pid_j * BLOCK_SIZE_J
    
    offsets_i = tl.arange(0, BLOCK_SIZE_I)
    offsets_j = tl.arange(0, BLOCK_SIZE_J)
    
    i_vals = block_start_i + offsets_i + (t + 1)
    j_vals = block_start_j + offsets_j + t
    
    mask_i = i_vals < N
    mask_j = j_vals < N
    
    # Load multipliers m[i][t]
    m_offsets = i_vals * N + t
    m_vals = tl.load(m_ptr + m_offsets, mask=mask_i)
    
    # Load a[t][j] values  
    a_t_offsets = t * N + j_vals
    a_t_vals = tl.load(a_ptr + a_t_offsets, mask=mask_j)
    
    # Update a[i][j] -= m[i][t] * a[t][j]
    i_expanded = i_vals[:, None]
    j_expanded = j_vals[None, :]
    mask_expanded = mask_i[:, None] & mask_j[None, :]
    
    a_ij_offsets = i_expanded * N + j_expanded
    a_ij_vals = tl.load(a_ptr + a_ij_offsets, mask=mask_expanded)
    
    m_expanded = m_vals[:, None]
    a_t_expanded = a_t_vals[None, :]
    
    a_ij_new = a_ij_vals - m_expanded * a_t_expanded
    tl.store(a_ptr + a_ij_offsets, a_ij_new, mask=mask_expanded)
    
    # Update b[i] -= m[i][t] * b[t] (only once per i)
    if pid_j == 0:
        b_t = tl.load(b_ptr + t)
        b_i_vals = tl.load(b_ptr + i_vals, mask=mask_i)
        b_i_new = b_i_vals - m_vals * b_t
        tl.store(b_ptr + i_vals, b_i_new, mask=mask_i)

@triton.jit
def back_substitution_kernel(a_ptr, b_ptr, x_ptr, N: tl.constexpr):
    # Back substitution is sequential, process one row at a time
    pid = tl.program_id(0)
    
    if pid == 0:
        # x[N-1] = b[N-1] / a[N-1][N-1]
        last_idx = N - 1
        b_last = tl.load(b_ptr + last_idx)
        a_last = tl.load(a_ptr + last_idx * N + last_idx)
        x_last = b_last / a_last
        tl.store(x_ptr + last_idx, x_last)
        
        # Process remaining rows from N-2 down to 0
        for i in range(N - 2, -1, -1):
            sum_val = tl.load(b_ptr + i)
            
            # Compute sum -= a[i][j] * x[j] for j = i+1 to N-1
            for j in range(i + 1, N):
                a_ij = tl.load(a_ptr + i * N + j)
                x_j = tl.load(x_ptr + j)
                sum_val -= a_ij * x_j
            
            # x[i] = sum / a[i][i]
            a_ii = tl.load(a_ptr + i * N + i)
            x_i = sum_val / a_ii
            tl.store(x_ptr + i, x_i)

def gaussian_triton(a, b, m, x, N):
    BLOCK_SIZE_I = 32
    BLOCK_SIZE_J = 32
    
    # Forward elimination
    for t in range(N - 1):
        # Phase 1: Compute multipliers
        num_rows = N - (t + 1)
        if num_rows > 0:
            grid_phase1 = (triton.cdiv(num_rows, BLOCK_SIZE_I),)
            gaussian_elimination_phase1_kernel[grid_phase1](
                a, m, t, N, BLOCK_SIZE_I
            )
            
            # Phase 2: Eliminate column t
            num_cols = N - t
            grid_phase2 = (triton.cdiv(num_rows, BLOCK_SIZE_I), triton.cdiv(num_cols, BLOCK_SIZE_J))
            gaussian_elimination_phase2_kernel[grid_phase2](
                a, b, m, t, N, BLOCK_SIZE_I, BLOCK_SIZE_J
            )
    
    # Back substitution
    grid_back = (1,)
    back_substitution_kernel[grid_back](
        a, b, x, N
    )