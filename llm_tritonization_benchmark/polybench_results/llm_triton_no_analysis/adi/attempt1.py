import torch
import triton
import triton.language as tl

@triton.jit
def adi_kernel(
    p_ptr, q_ptr, u_ptr, v_ptr,
    N: tl.constexpr, TSTEPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate constants
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
    
    # Time stepping loop
    for t in range(1, TSTEPS + 1):
        # Column Sweep
        for i in range(1, N - 1):
            # Set boundary conditions
            tl.store(v_ptr + 0 * N + i, 1.0)
            tl.store(p_ptr + i * N + 0, 0.0)
            v_0_i = tl.load(v_ptr + 0 * N + i)
            tl.store(q_ptr + i * N + 0, v_0_i)
            
            # Forward sweep
            for j in range(1, N - 1):
                p_i_j_minus_1 = tl.load(p_ptr + i * N + (j - 1))
                q_i_j_minus_1 = tl.load(q_ptr + i * N + (j - 1))
                
                u_j_i_minus_1 = tl.load(u_ptr + j * N + (i - 1))
                u_j_i = tl.load(u_ptr + j * N + i)
                u_j_i_plus_1 = tl.load(u_ptr + j * N + (i + 1))
                
                p_val = -c / (a * p_i_j_minus_1 + b)
                q_val = (-d * u_j_i_minus_1 + (1.0 + 2.0 * d) * u_j_i - f * u_j_i_plus_1 - a * q_i_j_minus_1) / (a * p_i_j_minus_1 + b)
                
                tl.store(p_ptr + i * N + j, p_val)
                tl.store(q_ptr + i * N + j, q_val)
            
            # Set boundary condition
            tl.store(v_ptr + (N - 1) * N + i, 1.0)
            
            # Backward sweep
            for j_idx in range(N - 2):
                j = N - 2 - j_idx
                if j >= 1:
                    p_i_j = tl.load(p_ptr + i * N + j)
                    q_i_j = tl.load(q_ptr + i * N + j)
                    v_j_plus_1_i = tl.load(v_ptr + (j + 1) * N + i)
                    
                    v_val = p_i_j * v_j_plus_1_i + q_i_j
                    tl.store(v_ptr + j * N + i, v_val)
        
        # Row Sweep
        for i in range(1, N - 1):
            # Set boundary conditions
            tl.store(u_ptr + i * N + 0, 1.0)
            tl.store(p_ptr + i * N + 0, 0.0)
            u_i_0 = tl.load(u_ptr + i * N + 0)
            tl.store(q_ptr + i * N + 0, u_i_0)
            
            # Forward sweep
            for j in range(1, N - 1):
                p_i_j_minus_1 = tl.load(p_ptr + i * N + (j - 1))
                q_i_j_minus_1 = tl.load(q_ptr + i * N + (j - 1))
                
                v_i_minus_1_j = tl.load(v_ptr + (i - 1) * N + j)
                v_i_j = tl.load(v_ptr + i * N + j)
                v_i_plus_1_j = tl.load(v_ptr + (i + 1) * N + j)
                
                p_val = -f / (d * p_i_j_minus_1 + e)
                q_val = (-a * v_i_minus_1_j + (1.0 + 2.0 * a) * v_i_j - c * v_i_plus_1_j - d * q_i_j_minus_1) / (d * p_i_j_minus_1 + e)
                
                tl.store(p_ptr + i * N + j, p_val)
                tl.store(q_ptr + i * N + j, q_val)
            
            # Set boundary condition
            tl.store(u_ptr + i * N + (N - 1), 1.0)
            
            # Backward sweep
            for j_idx in range(N - 2):
                j = N - 2 - j_idx
                if j >= 1:
                    p_i_j = tl.load(p_ptr + i * N + j)
                    q_i_j = tl.load(q_ptr + i * N + j)
                    u_i_j_plus_1 = tl.load(u_ptr + i * N + (j + 1))
                    
                    u_val = p_i_j * u_i_j_plus_1 + q_i_j
                    tl.store(u_ptr + i * N + j, u_val)

def adi_triton(p, q, u, v, N, TSTEPS):
    BLOCK_SIZE = 1
    
    grid = (1,)
    adi_kernel[grid](
        p, q, u, v,
        N, TSTEPS,
        BLOCK_SIZE
    )