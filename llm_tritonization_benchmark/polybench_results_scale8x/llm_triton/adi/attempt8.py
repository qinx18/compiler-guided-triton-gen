import triton
import triton.language as tl
import torch

@triton.jit
def adi_kernel_column_sweep(u_ptr, v_ptr, p_ptr, q_ptr, N: tl.constexpr, a, b, c, d, f):
    i = tl.program_id(0) + 1
    
    if i >= N - 1:
        return
    
    # v[0][i] = 1.0
    v_offset = 0 * N + i
    tl.store(v_ptr + v_offset, 1.0)
    
    # p[i][0] = 0.0, q[i][0] = v[0][i] = 1.0
    p_offset_base = i * N
    q_offset_base = i * N
    tl.store(p_ptr + p_offset_base + 0, 0.0)
    tl.store(q_ptr + q_offset_base + 0, 1.0)
    
    # Forward sweep
    for j in range(1, N - 1):
        p_prev = tl.load(p_ptr + p_offset_base + (j - 1))
        q_prev = tl.load(q_ptr + q_offset_base + (j - 1))
        
        u_val_prev = tl.load(u_ptr + j * N + (i - 1))
        u_val_curr = tl.load(u_ptr + j * N + i)
        u_val_next = tl.load(u_ptr + j * N + (i + 1))
        
        p_val = -c / (a * p_prev + b)
        q_val = (-d * u_val_prev + (1.0 + 2.0 * d) * u_val_curr - f * u_val_next - a * q_prev) / (a * p_prev + b)
        
        tl.store(p_ptr + p_offset_base + j, p_val)
        tl.store(q_ptr + q_offset_base + j, q_val)
    
    # v[N-1][i] = 1.0
    v_offset_end = (N - 1) * N + i
    tl.store(v_ptr + v_offset_end, 1.0)
    
    # Backward sweep
    for j in range(N - 2, 0, -1):
        p_val = tl.load(p_ptr + p_offset_base + j)
        q_val = tl.load(q_ptr + q_offset_base + j)
        v_next = tl.load(v_ptr + (j + 1) * N + i)
        
        v_val = p_val * v_next + q_val
        tl.store(v_ptr + j * N + i, v_val)

@triton.jit
def adi_kernel_row_sweep(u_ptr, v_ptr, p_ptr, q_ptr, N: tl.constexpr, a, c, d, e, f):
    i = tl.program_id(0) + 1
    
    if i >= N - 1:
        return
    
    # u[i][0] = 1.0
    u_offset = i * N + 0
    tl.store(u_ptr + u_offset, 1.0)
    
    # p[i][0] = 0.0, q[i][0] = u[i][0] = 1.0
    p_offset_base = i * N
    q_offset_base = i * N
    tl.store(p_ptr + p_offset_base + 0, 0.0)
    tl.store(q_ptr + q_offset_base + 0, 1.0)
    
    # Forward sweep
    for j in range(1, N - 1):
        p_prev = tl.load(p_ptr + p_offset_base + (j - 1))
        q_prev = tl.load(q_ptr + q_offset_base + (j - 1))
        
        v_val_prev = tl.load(v_ptr + (i - 1) * N + j)
        v_val_curr = tl.load(v_ptr + i * N + j)
        v_val_next = tl.load(v_ptr + (i + 1) * N + j)
        
        p_val = -f / (d * p_prev + e)
        q_val = (-a * v_val_prev + (1.0 + 2.0 * a) * v_val_curr - c * v_val_next - d * q_prev) / (d * p_prev + e)
        
        tl.store(p_ptr + p_offset_base + j, p_val)
        tl.store(q_ptr + q_offset_base + j, q_val)
    
    # u[i][N-1] = 1.0
    u_offset_end = i * N + (N - 1)
    tl.store(u_ptr + u_offset_end, 1.0)
    
    # Backward sweep
    for j in range(N - 2, 0, -1):
        p_val = tl.load(p_ptr + p_offset_base + j)
        q_val = tl.load(q_ptr + q_offset_base + j)
        u_next = tl.load(u_ptr + i * N + (j + 1))
        
        u_val = p_val * u_next + q_val
        tl.store(u_ptr + i * N + j, u_val)

def adi_triton(p, q, u, v, N, TSTEPS):
    DX = 1.0 / N
    DY = 1.0 / N
    DT = 1.0 / TSTEPS
    B1 = 2.0
    B2 = 1.0
    mul1 = B1 * DT / (DX * DX)
    mul2 = B2 * DT / (DY * DY)
    
    a = -mul1 / 2.0
    b = 1.0 + mul1
    c = a
    d = -mul2 / 2.0
    e = 1.0 + mul2
    f = d
    
    grid_size = (N - 2,)
    
    for t in range(1, TSTEPS + 1):
        # Column Sweep
        adi_kernel_column_sweep[grid_size](u, v, p, q, N, a, b, c, d, f)
        
        # Row Sweep  
        adi_kernel_row_sweep[grid_size](u, v, p, q, N, a, c, d, e, f)