import triton
import triton.language as tl
import torch

@triton.jit
def gramschmidt_kernel_column_ops(
    A_ptr, Q_ptr, R_ptr,
    M, N, k,
    stride_A_m, stride_A_n,
    stride_Q_m, stride_Q_n,
    stride_R_m, stride_R_n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Compute norm for column k
    if pid == 0:
        nrm = 0.0
        for i in range(M):
            a_val = tl.load(A_ptr + i * stride_A_m + k * stride_A_n)
            nrm += a_val * a_val
        
        r_kk = tl.sqrt(nrm)
        tl.store(R_ptr + k * stride_R_m + k * stride_R_n, r_kk)

@triton.jit
def gramschmidt_kernel_q_column(
    A_ptr, Q_ptr, R_ptr,
    M, N, k,
    stride_A_m, stride_A_n,
    stride_Q_m, stride_Q_n,
    stride_R_m, stride_R_n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    mask = i_offsets < M
    
    r_kk = tl.load(R_ptr + k * stride_R_m + k * stride_R_n)
    
    a_ptrs = A_ptr + i_offsets * stride_A_m + k * stride_A_n
    q_ptrs = Q_ptr + i_offsets * stride_Q_m + k * stride_Q_n
    
    a_vals = tl.load(a_ptrs, mask=mask)
    q_vals = a_vals / r_kk
    tl.store(q_ptrs, q_vals, mask=mask)

@triton.jit
def gramschmidt_kernel_update(
    A_ptr, Q_ptr, R_ptr,
    M, N, k, j,
    stride_A_m, stride_A_n,
    stride_Q_m, stride_Q_n,
    stride_R_m, stride_R_n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Compute R[k][j]
    if pid == 0:
        r_kj = 0.0
        for i in range(M):
            q_val = tl.load(Q_ptr + i * stride_Q_m + k * stride_Q_n)
            a_val = tl.load(A_ptr + i * stride_A_m + j * stride_A_n)
            r_kj += q_val * a_val
        tl.store(R_ptr + k * stride_R_m + j * stride_R_n, r_kj)

@triton.jit
def gramschmidt_kernel_subtract(
    A_ptr, Q_ptr, R_ptr,
    M, N, k, j,
    stride_A_m, stride_A_n,
    stride_Q_m, stride_Q_n,
    stride_R_m, stride_R_n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    mask = i_offsets < M
    
    r_kj = tl.load(R_ptr + k * stride_R_m + j * stride_R_n)
    
    q_ptrs = Q_ptr + i_offsets * stride_Q_m + k * stride_Q_n
    a_ptrs = A_ptr + i_offsets * stride_A_m + j * stride_A_n
    
    q_vals = tl.load(q_ptrs, mask=mask)
    a_vals = tl.load(a_ptrs, mask=mask)
    
    new_a_vals = a_vals - q_vals * r_kj
    tl.store(a_ptrs, new_a_vals, mask=mask)

def gramschmidt_triton(A, Q, R, M, N):
    BLOCK_SIZE = 64
    
    for k in range(N):
        # Compute norm and R[k][k]
        gramschmidt_kernel_column_ops[(1,)](
            A, Q, R,
            M, N, k,
            A.stride(0), A.stride(1),
            Q.stride(0), Q.stride(1),
            R.stride(0), R.stride(1),
            BLOCK_SIZE,
        )
        
        # Compute Q column k
        grid_size = triton.cdiv(M, BLOCK_SIZE)
        gramschmidt_kernel_q_column[(grid_size,)](
            A, Q, R,
            M, N, k,
            A.stride(0), A.stride(1),
            Q.stride(0), Q.stride(1),
            R.stride(0), R.stride(1),
            BLOCK_SIZE,
        )
        
        # Update remaining columns
        for j in range(k + 1, N):
            # Compute R[k][j]
            gramschmidt_kernel_update[(1,)](
                A, Q, R,
                M, N, k, j,
                A.stride(0), A.stride(1),
                Q.stride(0), Q.stride(1),
                R.stride(0), R.stride(1),
                BLOCK_SIZE,
            )
            
            # Update A column j
            grid_size = triton.cdiv(M, BLOCK_SIZE)
            gramschmidt_kernel_subtract[(grid_size,)](
                A, Q, R,
                M, N, k, j,
                A.stride(0), A.stride(1),
                Q.stride(0), Q.stride(1),
                R.stride(0), R.stride(1),
                BLOCK_SIZE,
            )