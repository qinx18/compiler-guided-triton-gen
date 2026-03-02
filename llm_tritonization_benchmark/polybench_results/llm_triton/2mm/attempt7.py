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
    
    # Initialize tmp block
    tmp_block = tl.zeros((BLOCK_SIZE_I, BLOCK_SIZE_J), dtype=tl.float32)
    
    # Compute tmp[i][j] = alpha * sum_k(A[i][k] * B[k][j])
    for k in range(NK):
        # Load A[i][k] for all i in block
        A_ptrs = A_ptr + offsets_i[:, None] * stride_A_i + k * stride_A_k
        A_vals = tl.load(A_ptrs, mask=mask_i[:, None], other=0.0)
        
        # Load B[k][j] for all j in block  
        B_ptrs = B_ptr + k * stride_B_k + offsets_j[None, :] * stride_B_j
        B_vals = tl.load(B_ptrs, mask=mask_j[None, :], other=0.0)
        
        # Accumulate alpha * A[i][k] * B[k][j]
        tmp_block += alpha * A_vals * B_vals
    
    # Store tmp block
    tmp_ptrs = tmp_ptr + offsets_i[:, None] * stride_tmp_i + offsets_j[None, :] * stride_tmp_j
    mask_2d = mask_i[:, None] & mask_j[None, :]
    tl.store(tmp_ptrs, tmp_block, mask=mask_2d)

@triton.jit
def k2mm_kernel_second(A_ptr, B_ptr, C_ptr, D_ptr, tmp_ptr, alpha, beta, NI, NJ, NK, NL, 
                       stride_A_i, stride_A_k, stride_B_k, stride_B_j, 
                       stride_C_j, stride_C_l, stride_D_i, stride_D_l, 
                       stride_tmp_i, stride_tmp_j, BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_L: tl.constexpr):
    
    pid_i = tl.program_id(0)
    pid_l = tl.program_id(1)
    
    # Second loop: D[i][l] = beta * D[i][l] + sum_k(tmp[i][k] * C[k][l])
    offsets_i = pid_i * BLOCK_SIZE_I + tl.arange(0, BLOCK_SIZE_I)
    offsets_l = pid_l * BLOCK_SIZE_L + tl.arange(0, BLOCK_SIZE_L)
    
    mask_i = offsets_i < NI
    mask_l = offsets_l < NL
    
    # Load and scale D by beta
    D_ptrs = D_ptr + offsets_i[:, None] * stride_D_i + offsets_l[None, :] * stride_D_l
    mask_2d = mask_i[:, None] & mask_l[None, :]
    D_vals = tl.load(D_ptrs, mask=mask_2d, other=0.0)
    D_block = beta * D_vals
    
    # Compute sum_k(tmp[i][k] * C[k][l])
    for k in range(NJ):
        # Load tmp[i][k] for all i in block
        tmp_ptrs = tmp_ptr + offsets_i[:, None] * stride_tmp_i + k * stride_tmp_j
        tmp_vals = tl.load(tmp_ptrs, mask=mask_i[:, None], other=0.0)
        
        # Load C[k][l] for all l in block
        C_ptrs = C_ptr + k * stride_C_j + offsets_l[None, :] * stride_C_l
        C_vals = tl.load(C_ptrs, mask=mask_l[None, :], other=0.0)
        
        # Accumulate tmp[i][k] * C[k][l]
        D_block += tmp_vals * C_vals
    
    # Store D block
    tl.store(D_ptrs, D_block, mask=mask_2d)

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