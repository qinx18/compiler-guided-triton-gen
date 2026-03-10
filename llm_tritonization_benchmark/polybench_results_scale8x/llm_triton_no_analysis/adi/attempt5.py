import torch
import triton
import triton.language as tl

@triton.jit
def adi_kernel(p_ptr, q_ptr, u_ptr, v_ptr, N: tl.constexpr, TSTEPS: tl.constexpr):
    # Compute constants
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
    
    for t in range(1, TSTEPS + 1):
        # Column Sweep
        for i in range(1, N - 1):
            # v[0][i] = 1.0
            tl.store(v_ptr + 0 * N + i, 1.0)
            
            # p[i][0] = 0.0
            tl.store(p_ptr + i * N + 0, 0.0)
            
            # q[i][0] = v[0][i]
            tl.store(q_ptr + i * N + 0, 1.0)
            
            for j in range(1, N - 1):
                # p[i][j] = -c / (a*p[i][j-1]+b)
                p_i_j_minus_1 = tl.load(p_ptr + i * N + (j - 1))
                p_i_j = -c / (a * p_i_j_minus_1 + b)
                tl.store(p_ptr + i * N + j, p_i_j)
                
                # q[i][j] = (-d*u[j][i-1]+(1.0+2.0*d)*u[j][i] - f*u[j][i+1]-a*q[i][j-1])/(a*p[i][j-1]+b)
                u_j_i_minus_1 = tl.load(u_ptr + j * N + (i - 1))
                u_j_i = tl.load(u_ptr + j * N + i)
                u_j_i_plus_1 = tl.load(u_ptr + j * N + (i + 1))
                q_i_j_minus_1 = tl.load(q_ptr + i * N + (j - 1))
                
                numerator = (-d * u_j_i_minus_1 + (1.0 + 2.0 * d) * u_j_i - f * u_j_i_plus_1 - a * q_i_j_minus_1)
                denominator = a * p_i_j_minus_1 + b
                q_i_j = numerator / denominator
                tl.store(q_ptr + i * N + j, q_i_j)
            
            # v[N-1][i] = 1.0
            tl.store(v_ptr + (N - 1) * N + i, 1.0)
            
            for j in range(N - 2, 0, -1):
                # v[j][i] = p[i][j] * v[j+1][i] + q[i][j]
                p_i_j = tl.load(p_ptr + i * N + j)
                v_j_plus_1_i = tl.load(v_ptr + (j + 1) * N + i)
                q_i_j = tl.load(q_ptr + i * N + j)
                v_j_i = p_i_j * v_j_plus_1_i + q_i_j
                tl.store(v_ptr + j * N + i, v_j_i)
        
        # Row Sweep  
        for i in range(1, N - 1):
            # u[i][0] = 1.0
            tl.store(u_ptr + i * N + 0, 1.0)
            
            # p[i][0] = 0.0
            tl.store(p_ptr + i * N + 0, 0.0)
            
            # q[i][0] = u[i][0]
            tl.store(q_ptr + i * N + 0, 1.0)
            
            for j in range(1, N - 1):
                # p[i][j] = -f / (d*p[i][j-1]+e)
                p_i_j_minus_1 = tl.load(p_ptr + i * N + (j - 1))
                p_i_j = -f / (d * p_i_j_minus_1 + e)
                tl.store(p_ptr + i * N + j, p_i_j)
                
                # q[i][j] = (-a*v[i-1][j]+(1.0+2.0*a)*v[i][j] - c*v[i+1][j]-d*q[i][j-1])/(d*p[i][j-1]+e)
                v_i_minus_1_j = tl.load(v_ptr + (i - 1) * N + j)
                v_i_j = tl.load(v_ptr + i * N + j)
                v_i_plus_1_j = tl.load(v_ptr + (i + 1) * N + j)
                q_i_j_minus_1 = tl.load(q_ptr + i * N + (j - 1))
                
                numerator = (-a * v_i_minus_1_j + (1.0 + 2.0 * a) * v_i_j - c * v_i_plus_1_j - d * q_i_j_minus_1)
                denominator = d * p_i_j_minus_1 + e
                q_i_j = numerator / denominator
                tl.store(q_ptr + i * N + j, q_i_j)
            
            # u[i][N-1] = 1.0
            tl.store(u_ptr + i * N + (N - 1), 1.0)
            
            for j in range(N - 2, 0, -1):
                # u[i][j] = p[i][j] * u[i][j+1] + q[i][j]
                p_i_j = tl.load(p_ptr + i * N + j)
                u_i_j_plus_1 = tl.load(u_ptr + i * N + (j + 1))
                q_i_j = tl.load(q_ptr + i * N + j)
                u_i_j = p_i_j * u_i_j_plus_1 + q_i_j
                tl.store(u_ptr + i * N + j, u_i_j)

def adi_triton(p, q, u, v, N, TSTEPS):
    grid = (1,)
    adi_kernel[grid](p, q, u, v, N, TSTEPS)