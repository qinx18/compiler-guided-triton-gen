import triton
import triton.language as tl
import torch

@triton.jit
def k2mm_kernel(A_ptr, B_ptr, C_ptr, D_ptr, tmp_ptr, alpha, beta, NI, NJ, NK, NL, 
                stride_A_i, stride_A_k, stride_B_k, stride_B_j, 
                stride_C_j, stride_C_l, stride_D_i, stride_D_l, 
                stride_tmp_i, stride_tmp_j, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr):
    
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # First loop: tmp[i][j] = alpha * A[i][k] * B[k][j]
    offsets_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    offsets_j = pid_j * BLOCK_SIZE_J + tl.arange(0, BLOCK_SIZE_J)
    
    mask_i = offsets_i < NI
    mask_j = offsets_j < NJ
    
    for i in range(BLOCK_SIZE_I):
        if pid_i * BLOCK_SIZE_I + i >= NI:
            break
        for j in range(BLOCK_SIZE_J):
            if pid_j * BLOCK_SIZE_J + j >= NJ:
                break
            
            actual_i = pid_i * BLOCK_SIZE_I + i
            actual_j = pid_j * BLOCK_SIZE_J + j
            
            acc = 0.0
            for k in range(NK):
                a_val = tl.load(A_ptr + actual_i * stride_A_i + k * stride_A_k)
                b_val = tl.load(B_ptr + k * stride_B_k + actual_j * stride_B_j)
                acc += alpha * a_val * b_val
            
            tl.store(tmp_ptr + actual_i * stride_tmp_i + actual_j * stride_tmp_j, acc)

@triton.jit
def k2mm_kernel_second(A_ptr, B_ptr, C_ptr, D_ptr, tmp_ptr, alpha, beta, NI, NJ, NK, NL, 
                       stride_A_i, stride_A_k, stride_B_k, stride_B_j, 
                       stride_C_j, stride_C_l, stride_D_i, stride_D_l, 
                       stride_tmp_i, stride_tmp_j, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_L: tl.constexpr):
    
    pid_i = tl.program_id(0)
    pid_l = tl.program_id(1)
    
    # Second loop: D[i][j] = beta * D[i][j] + tmp[i][k] * C[k][j]
    for i in range(BLOCK_SIZE_I):
        if pid_i * BLOCK_SIZE_I + i >= NI:
            break
        for l in range(BLOCK_SIZE_L):
            if pid_l * BLOCK_SIZE_L + l >= NL:
                break
            
            actual_i = pid_i * BLOCK_SIZE_I + i
            actual_l = pid_l * BLOCK_SIZE_L + l
            
            d_val = tl.load(D_ptr + actual_i * stride_D_i + actual_l * stride_D_l)
            acc = beta * d_val
            
            for k in range(NJ):
                tmp_val = tl.load(tmp_ptr + actual_i * stride_tmp_i + k * stride_tmp_j)
                c_val = tl.load(C_ptr + k * stride_C_j + actual_l * stride_C_l)
                acc += tmp_val * c_val
            
            tl.store(D_ptr + actual_i * stride_D_i + actual_l * stride_D_l, acc)

def k2mm_triton(A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL):
    BLOCK_SIZE_I = 8
    BLOCK_SIZE_J = 8
    BLOCK_SIZE_L = 8
    
    grid_i = triton.cdiv(NI, BLOCK_SIZE_I)
    grid_j = triton.cdiv(NJ, BLOCK_SIZE_J)
    grid_l = triton.cdiv(NL, BLOCK_SIZE_L)
    
    # First kernel launch for tmp computation
    k2mm_kernel[(grid_i, grid_j)](
        A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1),
        C.stride(0), C.stride(1), D.stride(0), D.stride(1),
        tmp.stride(0), tmp.stride(1), BLOCK_SIZE_I, BLOCK_SIZE_J
    )
    
    # Second kernel launch for D computation
    k2mm_kernel_second[(grid_i, grid_l)](
        A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL,
        A.stride(0), A.stride(1), B.stride(0), B.stride(1),
        C.stride(0), C.stride(1), D.stride(0), D.stride(1),
        tmp.stride(0), tmp.stride(1), BLOCK_SIZE_I, BLOCK_SIZE_L
    )