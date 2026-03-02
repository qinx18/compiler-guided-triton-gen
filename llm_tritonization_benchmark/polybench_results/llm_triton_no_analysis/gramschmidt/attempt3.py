import triton
import triton.language as tl
import torch

@triton.jit
def gramschmidt_kernel_norm(
    A_ptr, R_ptr,
    M, N, k,
    stride_A_m, stride_A_n,
    stride_R_m, stride_R_n,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_offsets = block_start + offsets
    mask = i_offsets < M
    
    a_ptrs = A_ptr + i_offsets * stride_A_m + k * stride_A_n
    a_vals = tl.load(a_ptrs, mask=mask, other=0.0)
    
    squared_vals = a_vals * a_vals
    block_sum = tl.sum(squared_vals)
    
    tl.atomic_add(R_ptr + k * stride_R_m + k * stride_R_n, block_sum)

@triton.jit
def gramschmidt_kernel_finalize_norm(
    R_ptr,
    k,
    stride_R_m, stride_R_n,
):
    pid = tl.program_id(0)
    if pid == 0:
        r_kk_addr = R_ptr + k * stride_R_m + k * stride_R_n
        r_kk_squared = tl.load(r_kk_addr)
        r_kk = tl.sqrt(r_kk_squared)
        tl.store(r_kk_addr, r_kk)

@triton.jit
def gramschmidt_kernel_compute_q(
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
def gramschmidt_kernel_compute_r(
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
    
    q_ptrs = Q_ptr + i_offsets * stride_Q_m + k * stride_Q_n
    a_ptrs = A_ptr + i_offsets * stride_A_m + j * stride_A_n
    
    q_vals = tl.load(q_ptrs, mask=mask, other=0.0)
    a_vals = tl.load(a_ptrs, mask=mask, other=0.0)
    
    products = q_vals * a_vals
    block_sum = tl.sum(products)
    
    tl.atomic_add(R_ptr + k * stride_R_m + j * stride_R_n, block_sum)

@triton.jit
def gramschmidt_kernel_update_a(
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
        # Zero out R[k][k] for atomic accumulation
        R[k, k] = 0.0
        
        # Compute norm: sum of A[i][k]^2
        grid_size = triton.cdiv(M, BLOCK_SIZE)
        gramschmidt_kernel_norm[(grid_size,)](
            A, R,
            M, N, k,
            A.stride(0), A.stride(1),
            R.stride(0), R.stride(1),
            BLOCK_SIZE,
        )
        
        # Finalize norm: R[k][k] = sqrt(sum)
        gramschmidt_kernel_finalize_norm[(1,)](
            R,
            k,
            R.stride(0), R.stride(1),
        )
        
        # Compute Q[:, k] = A[:, k] / R[k][k]
        gramschmidt_kernel_compute_q[(grid_size,)](
            A, Q, R,
            M, N, k,
            A.stride(0), A.stride(1),
            Q.stride(0), Q.stride(1),
            R.stride(0), R.stride(1),
            BLOCK_SIZE,
        )
        
        # Update remaining columns
        for j in range(k + 1, N):
            # Zero out R[k][j] for atomic accumulation
            R[k, j] = 0.0
            
            # Compute R[k][j] = sum(Q[i][k] * A[i][j])
            gramschmidt_kernel_compute_r[(grid_size,)](
                A, Q, R,
                M, N, k, j,
                A.stride(0), A.stride(1),
                Q.stride(0), Q.stride(1),
                R.stride(0), R.stride(1),
                BLOCK_SIZE,
            )
            
            # Update A[:, j] = A[:, j] - Q[:, k] * R[k][j]
            gramschmidt_kernel_update_a[(grid_size,)](
                A, Q, R,
                M, N, k, j,
                A.stride(0), A.stride(1),
                Q.stride(0), Q.stride(1),
                R.stride(0), R.stride(1),
                BLOCK_SIZE,
            )