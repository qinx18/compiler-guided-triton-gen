import triton
import triton.language as tl
import torch

@triton.jit
def symm_kernel(A_ptr, B_ptr, C_ptr, alpha, beta, M, N, 
                A_stride_0, A_stride_1, B_stride_0, B_stride_1, C_stride_0, C_stride_1,
                BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr):
    
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_start = pid_i * BLOCK_SIZE_I
    j_start = pid_j * BLOCK_SIZE_J
    
    i_offsets = i_start + tl.arange(0, BLOCK_SIZE_I)
    j_offsets = j_start + tl.arange(0, BLOCK_SIZE_J)
    
    i_mask = i_offsets < M
    j_mask = j_offsets < N
    
    # Process each i in the block sequentially
    for i_block_idx in range(BLOCK_SIZE_I):
        i = i_start + i_block_idx
        
        if i < M:
            # Load B[i,j] for all j in block
            b_ij_ptr = B_ptr + i * B_stride_0 + j_offsets * B_stride_1
            b_ij = tl.load(b_ij_ptr, mask=j_mask, other=0.0)
            
            # Load A[i,i]
            a_ii = tl.load(A_ptr + i * A_stride_0 + i * A_stride_1)
            
            # Initialize temp2 for this i
            temp2 = tl.zeros([BLOCK_SIZE_J], dtype=tl.float32)
            
            # Inner k loop: for k in range(i)
            for k in range(i):
                # Load A[i,k]
                a_ik = tl.load(A_ptr + i * A_stride_0 + k * A_stride_1)
                
                # Load B[k,j] for all j
                b_kj_ptr = B_ptr + k * B_stride_0 + j_offsets * B_stride_1
                b_kj = tl.load(b_kj_ptr, mask=j_mask, other=0.0)
                
                # Update C[k,j] += alpha * B[i,j] * A[i,k]
                c_kj_ptr = C_ptr + k * C_stride_0 + j_offsets * C_stride_1
                c_kj = tl.load(c_kj_ptr, mask=j_mask, other=0.0)
                c_kj = c_kj + alpha * b_ij * a_ik
                tl.store(c_kj_ptr, c_kj, mask=j_mask)
                
                # Accumulate temp2 += B[k,j] * A[i,k]
                temp2 = temp2 + b_kj * a_ik
            
            # Update C[i,j] = beta * C[i,j] + alpha * B[i,j] * A[i,i] + alpha * temp2
            c_ij_ptr = C_ptr + i * C_stride_0 + j_offsets * C_stride_1
            c_ij = tl.load(c_ij_ptr, mask=j_mask, other=0.0)
            c_ij = beta * c_ij + alpha * b_ij * a_ii + alpha * temp2
            tl.store(c_ij_ptr, c_ij, mask=j_mask)

def symm_triton(A, B, C, alpha, beta, M, N):
    BLOCK_SIZE_I = 4
    BLOCK_SIZE_J = 16
    
    grid = (triton.cdiv(M, BLOCK_SIZE_I), triton.cdiv(N, BLOCK_SIZE_J))
    
    symm_kernel[grid](
        A, B, C, alpha, beta, M, N,
        A.stride(0), A.stride(1), 
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_I=BLOCK_SIZE_I, 
        BLOCK_SIZE_J=BLOCK_SIZE_J
    )