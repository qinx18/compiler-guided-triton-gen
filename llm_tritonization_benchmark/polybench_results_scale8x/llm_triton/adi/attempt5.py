import torch
import triton
import triton.language as tl

@triton.jit
def adi_column_sweep_kernel(
    u_ptr, v_ptr, p_ptr, q_ptr,
    N: tl.constexpr,
    a: tl.constexpr, b: tl.constexpr, c: tl.constexpr,
    d: tl.constexpr, f: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    i = pid + 1
    
    if i >= N - 1:
        return
    
    # Set boundary conditions
    tl.store(v_ptr + 0 * N + i, 1.0)
    tl.store(p_ptr + i * N + 0, 0.0)
    tl.store(q_ptr + i * N + 0, 1.0)
    
    # Forward sweep
    for j in range(1, N - 1):
        p_prev = tl.load(p_ptr + i * N + (j - 1))
        q_prev = tl.load(q_ptr + i * N + (j - 1))
        
        u_j_im1 = tl.load(u_ptr + j * N + (i - 1))
        u_j_i = tl.load(u_ptr + j * N + i)
        u_j_ip1 = tl.load(u_ptr + j * N + (i + 1))
        
        p_val = -c / (a * p_prev + b)
        q_val = (-d * u_j_im1 + (1.0 + 2.0 * d) * u_j_i - f * u_j_ip1 - a * q_prev) / (a * p_prev + b)
        
        tl.store(p_ptr + i * N + j, p_val)
        tl.store(q_ptr + i * N + j, q_val)
    
    # Set boundary condition
    tl.store(v_ptr + (N - 1) * N + i, 1.0)
    
    # Backward sweep
    for j in range(N - 2, 0, -1):
        p_val = tl.load(p_ptr + i * N + j)
        q_val = tl.load(q_ptr + i * N + j)
        v_next = tl.load(v_ptr + (j + 1) * N + i)
        
        v_val = p_val * v_next + q_val
        tl.store(v_ptr + j * N + i, v_val)

@triton.jit
def adi_row_sweep_kernel(
    u_ptr, v_ptr, p_ptr, q_ptr,
    N: tl.constexpr,
    a: tl.constexpr, c: tl.constexpr,
    d: tl.constexpr, e: tl.constexpr, f: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    i = pid + 1
    
    if i >= N - 1:
        return
    
    # Set boundary conditions
    tl.store(u_ptr + i * N + 0, 1.0)
    tl.store(p_ptr + i * N + 0, 0.0)
    tl.store(q_ptr + i * N + 0, 1.0)
    
    # Forward sweep
    for j in range(1, N - 1):
        p_prev = tl.load(p_ptr + i * N + (j - 1))
        q_prev = tl.load(q_ptr + i * N + (j - 1))
        
        v_im1_j = tl.load(v_ptr + (i - 1) * N + j)
        v_i_j = tl.load(v_ptr + i * N + j)
        v_ip1_j = tl.load(v_ptr + (i + 1) * N + j)
        
        p_val = -f / (d * p_prev + e)
        q_val = (-a * v_im1_j + (1.0 + 2.0 * a) * v_i_j - c * v_ip1_j - d * q_prev) / (d * p_prev + e)
        
        tl.store(p_ptr + i * N + j, p_val)
        tl.store(q_ptr + i * N + j, q_val)
    
    # Set boundary condition
    tl.store(u_ptr + i * N + (N - 1), 1.0)
    
    # Backward sweep
    for j in range(N - 2, 0, -1):
        p_val = tl.load(p_ptr + i * N + j)
        q_val = tl.load(q_ptr + i * N + j)
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
    
    BLOCK_SIZE = 16
    
    for t in range(1, TSTEPS + 1):
        # Column Sweep
        grid = (N - 2,)
        adi_column_sweep_kernel[grid](
            u, v, p, q,
            N, a, b, c, d, f, BLOCK_SIZE
        )
        
        # Row Sweep
        grid = (N - 2,)
        adi_row_sweep_kernel[grid](
            u, v, p, q,
            N, a, c, d, e, f, BLOCK_SIZE
        )