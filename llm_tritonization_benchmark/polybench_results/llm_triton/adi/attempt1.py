import triton
import triton.language as tl
import torch

@triton.jit
def adi_column_sweep_kernel(
    u_ptr, v_ptr, p_ptr, q_ptr,
    a, b, c, d, e, f,
    N: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask = (i >= 1) & (i < N - 1)
    
    for idx in range(BLOCK_SIZE):
        if pid * BLOCK_SIZE + idx >= 1 and pid * BLOCK_SIZE + idx < N - 1:
            i_val = pid * BLOCK_SIZE + idx
            
            # v[0][i] = 1.0
            tl.store(v_ptr + 0 * N + i_val, 1.0)
            
            # p[i][0] = 0.0, q[i][0] = v[0][i]
            tl.store(p_ptr + i_val * N + 0, 0.0)
            tl.store(q_ptr + i_val * N + 0, 1.0)
            
            # Forward sweep
            for j in range(1, N - 1):
                p_prev = tl.load(p_ptr + i_val * N + (j - 1))
                q_prev = tl.load(q_ptr + i_val * N + (j - 1))
                
                u_prev = tl.load(u_ptr + j * N + (i_val - 1))
                u_curr = tl.load(u_ptr + j * N + i_val)
                u_next = tl.load(u_ptr + j * N + (i_val + 1))
                
                p_val = -c / (a * p_prev + b)
                q_val = (-d * u_prev + (1.0 + 2.0 * d) * u_curr - f * u_next - a * q_prev) / (a * p_prev + b)
                
                tl.store(p_ptr + i_val * N + j, p_val)
                tl.store(q_ptr + i_val * N + j, q_val)
            
            # v[N-1][i] = 1.0
            tl.store(v_ptr + (N - 1) * N + i_val, 1.0)
            
            # Backward sweep
            for j in range(N - 2, 0, -1):
                p_val = tl.load(p_ptr + i_val * N + j)
                q_val = tl.load(q_ptr + i_val * N + j)
                v_next = tl.load(v_ptr + (j + 1) * N + i_val)
                
                v_val = p_val * v_next + q_val
                tl.store(v_ptr + j * N + i_val, v_val)

@triton.jit
def adi_row_sweep_kernel(
    u_ptr, v_ptr, p_ptr, q_ptr,
    a, b, c, d, e, f,
    N: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    i = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    mask = (i >= 1) & (i < N - 1)
    
    for idx in range(BLOCK_SIZE):
        if pid * BLOCK_SIZE + idx >= 1 and pid * BLOCK_SIZE + idx < N - 1:
            i_val = pid * BLOCK_SIZE + idx
            
            # u[i][0] = 1.0
            tl.store(u_ptr + i_val * N + 0, 1.0)
            
            # p[i][0] = 0.0, q[i][0] = u[i][0]
            tl.store(p_ptr + i_val * N + 0, 0.0)
            tl.store(q_ptr + i_val * N + 0, 1.0)
            
            # Forward sweep
            for j in range(1, N - 1):
                p_prev = tl.load(p_ptr + i_val * N + (j - 1))
                q_prev = tl.load(q_ptr + i_val * N + (j - 1))
                
                v_prev = tl.load(v_ptr + (i_val - 1) * N + j)
                v_curr = tl.load(v_ptr + i_val * N + j)
                v_next = tl.load(v_ptr + (i_val + 1) * N + j)
                
                p_val = -f / (d * p_prev + e)
                q_val = (-a * v_prev + (1.0 + 2.0 * a) * v_curr - c * v_next - d * q_prev) / (d * p_prev + e)
                
                tl.store(p_ptr + i_val * N + j, p_val)
                tl.store(q_ptr + i_val * N + j, q_val)
            
            # u[i][N-1] = 1.0
            tl.store(u_ptr + i_val * N + (N - 1), 1.0)
            
            # Backward sweep
            for j in range(N - 2, 0, -1):
                p_val = tl.load(p_ptr + i_val * N + j)
                q_val = tl.load(q_ptr + i_val * N + j)
                u_next = tl.load(u_ptr + i_val * N + (j + 1))
                
                u_val = p_val * u_next + q_val
                tl.store(u_ptr + i_val * N + j, u_val)

def adi_triton(p, q, u, v, N, TSTEPS):
    device = u.device
    
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
    
    BLOCK_SIZE = 32
    grid_size = triton.cdiv(N - 2, BLOCK_SIZE)
    
    for t in range(1, TSTEPS + 1):
        # Column Sweep
        adi_column_sweep_kernel[(grid_size,)](
            u, v, p, q,
            a, b, c, d, e, f,
            N, BLOCK_SIZE
        )
        
        # Row Sweep
        adi_row_sweep_kernel[(grid_size,)](
            u, v, p, q,
            a, b, c, d, e, f,
            N, BLOCK_SIZE
        )