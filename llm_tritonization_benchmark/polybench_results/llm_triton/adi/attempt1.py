import torch
import triton
import triton.language as tl

@triton.jit
def adi_kernel(p_ptr, q_ptr, u_ptr, v_ptr, N, TSTEPS, 
               BLOCK_SIZE: tl.constexpr):
    # Get program id for time step
    t_id = tl.program_id(0)
    t = t_id + 1
    
    if t > TSTEPS:
        return
    
    # Calculate coefficients
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
    
    # Column Sweep
    for i in range(1, N-1):
        # Set boundary conditions
        v_idx = 0 * N + i
        tl.store(v_ptr + v_idx, 1.0)
        
        p_idx = i * N + 0
        q_idx = i * N + 0
        tl.store(p_ptr + p_idx, 0.0)
        v_val = tl.load(v_ptr + v_idx)
        tl.store(q_ptr + q_idx, v_val)
        
        # Forward sweep
        for j in range(1, N-1):
            p_curr_idx = i * N + j
            p_prev_idx = i * N + (j-1)
            q_curr_idx = i * N + j
            q_prev_idx = i * N + (j-1)
            
            u_idx1 = j * N + (i-1)
            u_idx2 = j * N + i
            u_idx3 = j * N + (i+1)
            
            p_prev = tl.load(p_ptr + p_prev_idx)
            q_prev = tl.load(q_ptr + q_prev_idx)
            u1 = tl.load(u_ptr + u_idx1)
            u2 = tl.load(u_ptr + u_idx2)
            u3 = tl.load(u_ptr + u_idx3)
            
            p_val = -c / (a * p_prev + b)
            q_val = (-d * u1 + (1.0 + 2.0 * d) * u2 - f * u3 - a * q_prev) / (a * p_prev + b)
            
            tl.store(p_ptr + p_curr_idx, p_val)
            tl.store(q_ptr + q_curr_idx, q_val)
        
        # Set boundary condition
        v_end_idx = (N-1) * N + i
        tl.store(v_ptr + v_end_idx, 1.0)
        
        # Backward sweep
        for j in range(N-2, 0, -1):
            v_curr_idx = j * N + i
            v_next_idx = (j+1) * N + i
            p_idx = i * N + j
            q_idx = i * N + j
            
            v_next = tl.load(v_ptr + v_next_idx)
            p_val = tl.load(p_ptr + p_idx)
            q_val = tl.load(q_ptr + q_idx)
            
            v_val = p_val * v_next + q_val
            tl.store(v_ptr + v_curr_idx, v_val)
    
    # Row Sweep
    for i in range(1, N-1):
        # Set boundary conditions
        u_idx = i * N + 0
        tl.store(u_ptr + u_idx, 1.0)
        
        p_idx = i * N + 0
        q_idx = i * N + 0
        tl.store(p_ptr + p_idx, 0.0)
        u_val = tl.load(u_ptr + u_idx)
        tl.store(q_ptr + q_idx, u_val)
        
        # Forward sweep
        for j in range(1, N-1):
            p_curr_idx = i * N + j
            p_prev_idx = i * N + (j-1)
            q_curr_idx = i * N + j
            q_prev_idx = i * N + (j-1)
            
            v_idx1 = (i-1) * N + j
            v_idx2 = i * N + j
            v_idx3 = (i+1) * N + j
            
            p_prev = tl.load(p_ptr + p_prev_idx)
            q_prev = tl.load(q_ptr + q_prev_idx)
            v1 = tl.load(v_ptr + v_idx1)
            v2 = tl.load(v_ptr + v_idx2)
            v3 = tl.load(v_ptr + v_idx3)
            
            p_val = -f / (d * p_prev + e)
            q_val = (-a * v1 + (1.0 + 2.0 * a) * v2 - c * v3 - d * q_prev) / (d * p_prev + e)
            
            tl.store(p_ptr + p_curr_idx, p_val)
            tl.store(q_ptr + q_curr_idx, q_val)
        
        # Set boundary condition
        u_end_idx = i * N + (N-1)
        tl.store(u_ptr + u_end_idx, 1.0)
        
        # Backward sweep
        for j in range(N-2, 0, -1):
            u_curr_idx = i * N + j
            u_next_idx = i * N + (j+1)
            p_idx = i * N + j
            q_idx = i * N + j
            
            u_next = tl.load(u_ptr + u_next_idx)
            p_val = tl.load(p_ptr + p_idx)
            q_val = tl.load(q_ptr + q_idx)
            
            u_val = p_val * u_next + q_val
            tl.store(u_ptr + u_curr_idx, u_val)

def adi_triton(p, q, u, v, N, TSTEPS):
    BLOCK_SIZE = 1
    grid = (TSTEPS,)
    
    adi_kernel[grid](
        p, q, u, v,
        N, TSTEPS,
        BLOCK_SIZE=BLOCK_SIZE
    )