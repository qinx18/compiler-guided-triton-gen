import triton
import triton.language as tl
import torch

@triton.jit
def adi_kernel(p_ptr, q_ptr, u_ptr, v_ptr, N: tl.constexpr, TSTEPS: tl.constexpr):
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
            # v[0][i] = 1.0
            tl.store(v_ptr + 0 * N + i, 1.0)
            # p[i][0] = 0.0
            tl.store(p_ptr + i * N + 0, 0.0)
            # q[i][0] = v[0][i] = 1.0
            tl.store(q_ptr + i * N + 0, 1.0)
            
            for j in range(1, N - 1):
                # p[i][j] = -c / (a*p[i][j-1]+b)
                p_prev = tl.load(p_ptr + i * N + (j - 1))
                p_val = -c / (a * p_prev + b)
                tl.store(p_ptr + i * N + j, p_val)
                
                # q[i][j] = (-d*u[j][i-1]+(1.0+2.0*d)*u[j][i] - f*u[j][i+1]-a*q[i][j-1])/(a*p[i][j-1]+b)
                u_jmi1 = tl.load(u_ptr + j * N + (i - 1))
                u_ji = tl.load(u_ptr + j * N + i)
                u_jpi1 = tl.load(u_ptr + j * N + (i + 1))
                q_prev = tl.load(q_ptr + i * N + (j - 1))
                
                numerator = -d * u_jmi1 + (1.0 + 2.0 * d) * u_ji - f * u_jpi1 - a * q_prev
                denominator = a * p_prev + b
                q_val = numerator / denominator
                tl.store(q_ptr + i * N + j, q_val)
            
            # v[N-1][i] = 1.0
            tl.store(v_ptr + (N - 1) * N + i, 1.0)
            
            # Backward sweep for v
            for j in range(N - 2, 0, -1):
                # v[j][i] = p[i][j] * v[j+1][i] + q[i][j]
                p_val = tl.load(p_ptr + i * N + j)
                v_next = tl.load(v_ptr + (j + 1) * N + i)
                q_val = tl.load(q_ptr + i * N + j)
                v_val = p_val * v_next + q_val
                tl.store(v_ptr + j * N + i, v_val)
        
        # Row Sweep
        for i in range(1, N - 1):
            # u[i][0] = 1.0
            tl.store(u_ptr + i * N + 0, 1.0)
            # p[i][0] = 0.0
            tl.store(p_ptr + i * N + 0, 0.0)
            # q[i][0] = u[i][0] = 1.0
            tl.store(q_ptr + i * N + 0, 1.0)
            
            for j in range(1, N - 1):
                # p[i][j] = -f / (d*p[i][j-1]+e)
                p_prev = tl.load(p_ptr + i * N + (j - 1))
                p_val = -f / (d * p_prev + e)
                tl.store(p_ptr + i * N + j, p_val)
                
                # q[i][j] = (-a*v[i-1][j]+(1.0+2.0*a)*v[i][j] - c*v[i+1][j]-d*q[i][j-1])/(d*p[i][j-1]+e)
                v_mi1j = tl.load(v_ptr + (i - 1) * N + j)
                v_ij = tl.load(v_ptr + i * N + j)
                v_pi1j = tl.load(v_ptr + (i + 1) * N + j)
                q_prev = tl.load(q_ptr + i * N + (j - 1))
                
                numerator = -a * v_mi1j + (1.0 + 2.0 * a) * v_ij - c * v_pi1j - d * q_prev
                denominator = d * p_prev + e
                q_val = numerator / denominator
                tl.store(q_ptr + i * N + j, q_val)
            
            # u[i][N-1] = 1.0
            tl.store(u_ptr + i * N + (N - 1), 1.0)
            
            # Backward sweep for u
            for j in range(N - 2, 0, -1):
                # u[i][j] = p[i][j] * u[i][j+1] + q[i][j]
                p_val = tl.load(p_ptr + i * N + j)
                u_next = tl.load(u_ptr + i * N + (j + 1))
                q_val = tl.load(q_ptr + i * N + j)
                u_val = p_val * u_next + q_val
                tl.store(u_ptr + i * N + j, u_val)

def adi_triton(p, q, u, v, N, TSTEPS):
    adi_kernel[(1,)](p, q, u, v, N, TSTEPS)