import triton
import triton.language as tl
import torch

@triton.jit
def adi_kernel(p_ptr, q_ptr, u_ptr, v_ptr, p_copy_ptr, q_copy_ptr, u_copy_ptr, v_copy_ptr, 
               N: tl.constexpr, TSTEPS: tl.constexpr, BLOCK_I: tl.constexpr):
    # Constants
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
    
    # Time loop
    for t in range(1, TSTEPS + 1):
        # Column Sweep
        for i in range(1, N - 1):
            # Set boundary conditions
            tl.store(v_ptr + 0 * N + i, 1.0)
            tl.store(p_ptr + i * N + 0, 0.0)
            tl.store(q_ptr + i * N + 0, tl.load(v_copy_ptr + 0 * N + i))
            
            # Forward sweep
            for j in range(1, N - 1):
                p_prev = tl.load(p_copy_ptr + i * N + (j - 1))
                q_prev = tl.load(q_copy_ptr + i * N + (j - 1))
                
                u_val1 = tl.load(u_copy_ptr + j * N + (i - 1))
                u_val2 = tl.load(u_copy_ptr + j * N + i)
                u_val3 = tl.load(u_copy_ptr + j * N + (i + 1))
                
                p_val = -c / (a * p_prev + b)
                q_val = (-d * u_val1 + (1.0 + 2.0 * d) * u_val2 - f * u_val3 - a * q_prev) / (a * p_prev + b)
                
                tl.store(p_ptr + i * N + j, p_val)
                tl.store(q_ptr + i * N + j, q_val)
            
            # Set boundary condition
            tl.store(v_ptr + (N - 1) * N + i, 1.0)
            
            # Backward sweep
            for j in range(N - 2, 0, -1):
                p_val = tl.load(p_copy_ptr + i * N + j)
                q_val = tl.load(q_copy_ptr + i * N + j)
                v_next = tl.load(v_copy_ptr + (j + 1) * N + i)
                
                v_val = p_val * v_next + q_val
                tl.store(v_ptr + j * N + i, v_val)
        
        # Update copies after column sweep
        for idx in range(N * N):
            tl.store(p_copy_ptr + idx, tl.load(p_ptr + idx))
            tl.store(q_copy_ptr + idx, tl.load(q_ptr + idx))
            tl.store(v_copy_ptr + idx, tl.load(v_ptr + idx))
        
        # Row Sweep
        for i in range(1, N - 1):
            # Set boundary conditions
            tl.store(u_ptr + i * N + 0, 1.0)
            tl.store(p_ptr + i * N + 0, 0.0)
            tl.store(q_ptr + i * N + 0, tl.load(u_copy_ptr + i * N + 0))
            
            # Forward sweep
            for j in range(1, N - 1):
                p_prev = tl.load(p_copy_ptr + i * N + (j - 1))
                q_prev = tl.load(q_copy_ptr + i * N + (j - 1))
                
                v_val1 = tl.load(v_copy_ptr + (i - 1) * N + j)
                v_val2 = tl.load(v_copy_ptr + i * N + j)
                v_val3 = tl.load(v_copy_ptr + (i + 1) * N + j)
                
                p_val = -f / (d * p_prev + e)
                q_val = (-a * v_val1 + (1.0 + 2.0 * a) * v_val2 - c * v_val3 - d * q_prev) / (d * p_prev + e)
                
                tl.store(p_ptr + i * N + j, p_val)
                tl.store(q_ptr + i * N + j, q_val)
            
            # Set boundary condition
            tl.store(u_ptr + i * N + (N - 1), 1.0)
            
            # Backward sweep
            for j in range(N - 2, 0, -1):
                p_val = tl.load(p_copy_ptr + i * N + j)
                q_val = tl.load(q_copy_ptr + i * N + j)
                u_next = tl.load(u_copy_ptr + i * N + (j + 1))
                
                u_val = p_val * u_next + q_val
                tl.store(u_ptr + i * N + j, u_val)
        
        # Update copies after row sweep
        for idx in range(N * N):
            tl.store(p_copy_ptr + idx, tl.load(p_ptr + idx))
            tl.store(q_copy_ptr + idx, tl.load(q_ptr + idx))
            tl.store(u_copy_ptr + idx, tl.load(u_ptr + idx))

def adi_triton(p, q, u, v, N, TSTEPS):
    # Clone arrays to handle WAR dependencies
    p_copy = p.clone()
    q_copy = q.clone()
    u_copy = u.clone()
    v_copy = v.clone()
    
    BLOCK_I = triton.next_power_of_2(N)
    
    # Single grid for sequential execution
    grid = (1,)
    
    adi_kernel[grid](
        p, q, u, v,
        p_copy, q_copy, u_copy, v_copy,
        N, TSTEPS, BLOCK_I
    )