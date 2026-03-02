import triton
import triton.language as tl
import torch

@triton.jit
def k2mm_kernel(A_ptr, B_ptr, C_ptr, D_ptr, tmp_ptr, alpha, beta, NI, NJ, NK, NL,
                A_stride_0, A_stride_1, B_stride_0, B_stride_1,
                C_stride_0, C_stride_1, D_stride_0, D_stride_1,
                tmp_stride_0, tmp_stride_1,
                BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr,
                BLOCK_SIZE_K: tl.constexpr, PHASE: tl.constexpr):
    
    # Get program IDs
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    # Define offsets once at the start
    offsets_i = tl.arange(0, BLOCK_SIZE_I)
    offsets_j = tl.arange(0, BLOCK_SIZE_J)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Calculate current row and column blocks
    i_start = pid_i * BLOCK_SIZE_I
    j_start = pid_j * BLOCK_SIZE_J
    
    current_i_offsets = i_start + offsets_i
    current_j_offsets = j_start + offsets_j
    
    # Masks for boundary checking
    i_mask = current_i_offsets < NI
    
    if PHASE == 0:
        # First phase: compute tmp = alpha * A * B
        j_mask = current_j_offsets < NJ
        
        # Initialize tmp block to zero
        tmp_block = tl.zeros((BLOCK_SIZE_I, BLOCK_SIZE_J), dtype=tl.float32)
        
        # Loop over K dimension in blocks
        for k_start in range(0, NK, BLOCK_SIZE_K):
            current_k_offsets = k_start + offsets_k
            k_mask = current_k_offsets < NK
            
            # Load A block [i, k]
            A_ptrs = A_ptr + current_i_offsets[:, None] * A_stride_0 + current_k_offsets[None, :] * A_stride_1
            A_mask = i_mask[:, None] & k_mask[None, :]
            A_block = tl.load(A_ptrs, mask=A_mask, other=0.0)
            
            # Load B block [k, j]
            B_ptrs = B_ptr + current_k_offsets[:, None] * B_stride_0 + current_j_offsets[None, :] * B_stride_1
            B_mask = k_mask[:, None] & j_mask[None, :]
            B_block = tl.load(B_ptrs, mask=B_mask, other=0.0)
            
            # Accumulate matrix multiplication
            tmp_block += tl.dot(A_block, B_block)
        
        # Scale by alpha and store tmp block
        tmp_block = tmp_block * alpha
        tmp_ptrs = tmp_ptr + current_i_offsets[:, None] * tmp_stride_0 + current_j_offsets[None, :] * tmp_stride_1
        tmp_mask = i_mask[:, None] & j_mask[None, :]
        tl.store(tmp_ptrs, tmp_block, mask=tmp_mask)
        
    else:
        # Second phase: compute D = beta * D + tmp * C
        j_mask = current_j_offsets < NL
        
        # Load and scale D by beta first
        D_ptrs = D_ptr + current_i_offsets[:, None] * D_stride_0 + current_j_offsets[None, :] * D_stride_1
        D_mask = i_mask[:, None] & j_mask[None, :]
        D_block = tl.load(D_ptrs, mask=D_mask, other=0.0)
        result_block = D_block * beta
        
        # Loop over NJ dimension for tmp * C
        for k_start in range(0, NJ, BLOCK_SIZE_K):
            current_k_offsets = k_start + offsets_k
            k_mask = current_k_offsets < NJ
            
            # Load tmp block [i, k]
            tmp_ptrs = tmp_ptr + current_i_offsets[:, None] * tmp_stride_0 + current_k_offsets[None, :] * tmp_stride_1
            tmp_mask = i_mask[:, None] & k_mask[None, :]
            tmp_block = tl.load(tmp_ptrs, mask=tmp_mask, other=0.0)
            
            # Load C block [k, j]
            C_ptrs = C_ptr + current_k_offsets[:, None] * C_stride_0 + current_j_offsets[None, :] * C_stride_1
            C_mask = k_mask[:, None] & j_mask[None, :]
            C_block = tl.load(C_ptrs, mask=C_mask, other=0.0)
            
            # Accumulate matrix multiplication
            result_block += tl.dot(tmp_block, C_block)
        
        # Store final result to D
        tl.store(D_ptrs, result_block, mask=D_mask)

def k2mm_triton(A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL):
    # Block sizes
    BLOCK_SIZE_I = 16
    BLOCK_SIZE_J = 16
    BLOCK_SIZE_K = 16
    
    # First phase: compute tmp = alpha * A * B
    grid_i = triton.cdiv(NI, BLOCK_SIZE_I)
    grid_j = triton.cdiv(NJ, BLOCK_SIZE_J)
    grid = (grid_i, grid_j)
    
    k2mm_kernel[grid](
        A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        D.stride(0), D.stride(1),
        tmp.stride(0), tmp.stride(1),
        BLOCK_SIZE_I=BLOCK_SIZE_I,
        BLOCK_SIZE_J=BLOCK_SIZE_J,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        PHASE=0
    )
    
    # Second phase: compute D = beta * D + tmp * C
    grid_j = triton.cdiv(NL, BLOCK_SIZE_J)
    grid = (grid_i, grid_j)
    
    k2mm_kernel[grid](
        A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        D.stride(0), D.stride(1),
        tmp.stride(0), tmp.stride(1),
        BLOCK_SIZE_I=BLOCK_SIZE_I,
        BLOCK_SIZE_J=BLOCK_SIZE_J,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        PHASE=1
    )