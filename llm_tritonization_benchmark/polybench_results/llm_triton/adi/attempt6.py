import triton
import triton.language as tl
import torch

@triton.jit
def adi_kernel(p_ptr, q_ptr, u_ptr, v_ptr, N, TSTEPS, t_start, i_start, BLOCK_T: tl.constexpr, BLOCK_I: tl.constexpr):
    t_idx = tl.program_id(0)
    i_idx = tl.program_id(1)
    
    t_offsets = t_start + t_idx * BLOCK_T + tl.arange(0, BLOCK_T)
    i_offsets = i_start + i_idx * BLOCK_I + tl.arange(0, BLOCK_I)
    
    t_mask = (t_offsets >= 1) & (t_offsets <= TSTEPS)
    i_mask = (i_offsets >= 1) & (i_offsets < N - 1)
    
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
    
    for t_block in range(BLOCK_T):
        t_val = t_start + t_idx * BLOCK_T + t_block
        if t_val < 1 or t_val > TSTEPS:
            continue
            
        for i_block in range(BLOCK_I):
            i_val = i_start + i_idx * BLOCK_I + i_block
            if i_val < 1 or i_val >= N - 1:
                continue
                
            # Column Sweep
            # v[0][i] = 1.0
            v_idx = 0 * N + i_val
            tl.store(v_ptr + v_idx, 1.0)
            
            # p[i][0] = 0.0
            p_idx = i_val * N + 0
            tl.store(p_ptr + p_idx, 0.0)
            
            # q[i][0] = v[0][i]
            q_idx = i_val * N + 0
            tl.store(q_ptr + q_idx, 1.0)
            
            # Forward sweep
            for j in range(1, N - 1):
                p_idx_curr = i_val * N + j
                p_idx_prev = i_val * N + (j - 1)
                q_idx_curr = i_val * N + j
                q_idx_prev = i_val * N + (j - 1)
                
                p_prev = tl.load(p_ptr + p_idx_prev)
                q_prev = tl.load(q_ptr + q_idx_prev)
                
                # p[i][j] = -c / (a*p[i][j-1]+b)
                p_val = -c / (a * p_prev + b)
                tl.store(p_ptr + p_idx_curr, p_val)
                
                # q[i][j] = (-d*u[j][i-1]+(1.0+2.0*d)*u[j][i] - f*u[j][i+1]-a*q[i][j-1])/(a*p[i][j-1]+b)
                u_idx_prev = j * N + (i_val - 1)
                u_idx_curr = j * N + i_val
                u_idx_next = j * N + (i_val + 1)
                
                u_prev = tl.load(u_ptr + u_idx_prev)
                u_curr = tl.load(u_ptr + u_idx_curr)
                u_next = tl.load(u_ptr + u_idx_next)
                
                q_val = (-d * u_prev + (1.0 + 2.0 * d) * u_curr - f * u_next - a * q_prev) / (a * p_prev + b)
                tl.store(q_ptr + q_idx_curr, q_val)
            
            # v[N-1][i] = 1.0
            v_idx = (N - 1) * N + i_val
            tl.store(v_ptr + v_idx, 1.0)
            
            # Backward sweep
            for j in range(N - 2, 0, -1):
                v_idx_curr = j * N + i_val
                v_idx_next = (j + 1) * N + i_val
                p_idx = i_val * N + j
                q_idx = i_val * N + j
                
                v_next = tl.load(v_ptr + v_idx_next)
                p_val = tl.load(p_ptr + p_idx)
                q_val = tl.load(q_ptr + q_idx)
                
                # v[j][i] = p[i][j] * v[j+1][i] + q[i][j]
                v_val = p_val * v_next + q_val
                tl.store(v_ptr + v_idx_curr, v_val)
            
            # Row Sweep
            # u[i][0] = 1.0
            u_idx = i_val * N + 0
            tl.store(u_ptr + u_idx, 1.0)
            
            # p[i][0] = 0.0
            p_idx = i_val * N + 0
            tl.store(p_ptr + p_idx, 0.0)
            
            # q[i][0] = u[i][0]
            q_idx = i_val * N + 0
            tl.store(q_ptr + q_idx, 1.0)
            
            # Forward sweep
            for j in range(1, N - 1):
                p_idx_curr = i_val * N + j
                p_idx_prev = i_val * N + (j - 1)
                q_idx_curr = i_val * N + j
                q_idx_prev = i_val * N + (j - 1)
                
                p_prev = tl.load(p_ptr + p_idx_prev)
                q_prev = tl.load(q_ptr + q_idx_prev)
                
                # p[i][j] = -f / (d*p[i][j-1]+e)
                p_val = -f / (d * p_prev + e)
                tl.store(p_ptr + p_idx_curr, p_val)
                
                # q[i][j] = (-a*v[i-1][j]+(1.0+2.0*a)*v[i][j] - c*v[i+1][j]-d*q[i][j-1])/(d*p[i][j-1]+e)
                v_idx_prev = (i_val - 1) * N + j
                v_idx_curr = i_val * N + j
                v_idx_next = (i_val + 1) * N + j
                
                v_prev = tl.load(v_ptr + v_idx_prev)
                v_curr = tl.load(v_ptr + v_idx_curr)
                v_next = tl.load(v_ptr + v_idx_next)
                
                q_val = (-a * v_prev + (1.0 + 2.0 * a) * v_curr - c * v_next - d * q_prev) / (d * p_prev + e)
                tl.store(q_ptr + q_idx_curr, q_val)
            
            # u[i][N-1] = 1.0
            u_idx = i_val * N + (N - 1)
            tl.store(u_ptr + u_idx, 1.0)
            
            # Backward sweep
            for j in range(N - 2, 0, -1):
                u_idx_curr = i_val * N + j
                u_idx_next = i_val * N + (j + 1)
                p_idx = i_val * N + j
                q_idx = i_val * N + j
                
                u_next = tl.load(u_ptr + u_idx_next)
                p_val = tl.load(p_ptr + p_idx)
                q_val = tl.load(q_ptr + q_idx)
                
                # u[i][j] = p[i][j] * u[i][j+1] + q[i][j]
                u_val = p_val * u_next + q_val
                tl.store(u_ptr + u_idx_curr, u_val)

def adi_triton(p, q, u, v, N, TSTEPS):
    BLOCK_T = 1
    BLOCK_I = 1
    
    # Process each timestep and i sequentially due to dependencies
    for t_start in range(1, TSTEPS + 1, BLOCK_T):
        for i_start in range(1, N - 1, BLOCK_I):
            grid = (triton.cdiv(BLOCK_T, BLOCK_T), triton.cdiv(BLOCK_I, BLOCK_I))
            
            adi_kernel[grid](
                p, q, u, v,
                N, TSTEPS,
                t_start, i_start,
                BLOCK_T=BLOCK_T,
                BLOCK_I=BLOCK_I
            )