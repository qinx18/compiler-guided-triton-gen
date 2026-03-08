import torch
import triton
import triton.language as tl

@triton.jit
def k3mm_kernel(A_ptr, B_ptr, C_ptr, D_ptr, E_ptr, F_ptr, G_ptr,
                NI: tl.constexpr, NJ: tl.constexpr, NK: tl.constexpr, 
                NL: tl.constexpr, NM: tl.constexpr,
                BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr):
    
    # Get block indices
    block_i = tl.program_id(0)
    block_j = tl.program_id(1)
    
    # Create offset ranges
    i_offsets = tl.arange(0, BLOCK_I)
    j_offsets = tl.arange(0, BLOCK_J)
    k_offsets = tl.arange(0, BLOCK_K)
    
    # E := A*B computation
    if block_i * BLOCK_I < NI and block_j * BLOCK_J < NJ:
        i_base = block_i * BLOCK_I
        j_base = block_j * BLOCK_J
        
        i_idx = i_base + i_offsets
        j_idx = j_base + j_offsets
        
        i_mask = i_idx < NI
        j_mask = j_idx < NJ
        
        # Initialize accumulator
        acc = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)
        
        # Matrix multiplication A*B
        for k in range(0, NK, BLOCK_K):
            k_idx = k + k_offsets
            k_mask = k_idx < NK
            
            # Load A[i, k]
            A_ptrs = A_ptr + i_idx[:, None] * NK + k_idx[None, :]
            A_vals = tl.load(A_ptrs, mask=i_mask[:, None] & k_mask[None, :], other=0.0)
            
            # Load B[k, j]
            B_ptrs = B_ptr + k_idx[:, None] * NJ + j_idx[None, :]
            B_vals = tl.load(B_ptrs, mask=k_mask[:, None] & j_mask[None, :], other=0.0)
            
            # Accumulate
            acc += tl.dot(A_vals, B_vals)
        
        # Store E[i, j]
        E_ptrs = E_ptr + i_idx[:, None] * NJ + j_idx[None, :]
        tl.store(E_ptrs, acc, mask=i_mask[:, None] & j_mask[None, :])
    
    # F := C*D computation
    if block_i * BLOCK_I < NJ and block_j * BLOCK_J < NL:
        i_base = block_i * BLOCK_I
        j_base = block_j * BLOCK_J
        
        i_idx = i_base + i_offsets
        j_idx = j_base + j_offsets
        
        i_mask = i_idx < NJ
        j_mask = j_idx < NL
        
        # Initialize accumulator
        acc = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)
        
        # Matrix multiplication C*D
        for k in range(0, NM, BLOCK_K):
            k_idx = k + k_offsets
            k_mask = k_idx < NM
            
            # Load C[i, k]
            C_ptrs = C_ptr + i_idx[:, None] * NM + k_idx[None, :]
            C_vals = tl.load(C_ptrs, mask=i_mask[:, None] & k_mask[None, :], other=0.0)
            
            # Load D[k, j]
            D_ptrs = D_ptr + k_idx[:, None] * NL + j_idx[None, :]
            D_vals = tl.load(D_ptrs, mask=k_mask[:, None] & j_mask[None, :], other=0.0)
            
            # Accumulate
            acc += tl.dot(C_vals, D_vals)
        
        # Store F[i, j]
        F_ptrs = F_ptr + i_idx[:, None] * NL + j_idx[None, :]
        tl.store(F_ptrs, acc, mask=i_mask[:, None] & j_mask[None, :])

@triton.jit
def k3mm_kernel_final(E_ptr, F_ptr, G_ptr,
                      NI: tl.constexpr, NJ: tl.constexpr, NL: tl.constexpr,
                      BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr):
    
    # Get block indices
    block_i = tl.program_id(0)
    block_j = tl.program_id(1)
    
    # Create offset ranges
    i_offsets = tl.arange(0, BLOCK_I)
    j_offsets = tl.arange(0, BLOCK_J)
    k_offsets = tl.arange(0, BLOCK_K)
    
    # G := E*F computation
    if block_i * BLOCK_I < NI and block_j * BLOCK_J < NL:
        i_base = block_i * BLOCK_I
        j_base = block_j * BLOCK_J
        
        i_idx = i_base + i_offsets
        j_idx = j_base + j_offsets
        
        i_mask = i_idx < NI
        j_mask = j_idx < NL
        
        # Initialize accumulator
        acc = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)
        
        # Matrix multiplication E*F
        for k in range(0, NJ, BLOCK_K):
            k_idx = k + k_offsets
            k_mask = k_idx < NJ
            
            # Load E[i, k]
            E_ptrs = E_ptr + i_idx[:, None] * NJ + k_idx[None, :]
            E_vals = tl.load(E_ptrs, mask=i_mask[:, None] & k_mask[None, :], other=0.0)
            
            # Load F[k, j]
            F_ptrs = F_ptr + k_idx[:, None] * NL + j_idx[None, :]
            F_vals = tl.load(F_ptrs, mask=k_mask[:, None] & j_mask[None, :], other=0.0)
            
            # Accumulate
            acc += tl.dot(E_vals, F_vals)
        
        # Store G[i, j]
        G_ptrs = G_ptr + i_idx[:, None] * NL + j_idx[None, :]
        tl.store(G_ptrs, acc, mask=i_mask[:, None] & j_mask[None, :])

def k3mm_triton(A, B, C, D, E, F, G, NI, NJ, NK, NL, NM):
    BLOCK_I = 16
    BLOCK_J = 16
    BLOCK_K = 16
    
    # First two matrix multiplications: E := A*B and F := C*D
    grid_E = (triton.cdiv(NI, BLOCK_I), triton.cdiv(NJ, BLOCK_J))
    k3mm_kernel[grid_E](
        A.data_ptr(), B.data_ptr(), C.data_ptr(), D.data_ptr(), 
        E.data_ptr(), F.data_ptr(), G.data_ptr(),
        NI, NJ, NK, NL, NM,
        BLOCK_I, BLOCK_J, BLOCK_K
    )
    
    # Final matrix multiplication: G := E*F
    grid_G = (triton.cdiv(NI, BLOCK_I), triton.cdiv(NL, BLOCK_J))
    k3mm_kernel_final[grid_G](
        E.data_ptr(), F.data_ptr(), G.data_ptr(),
        NI, NJ, NL,
        BLOCK_I, BLOCK_J, BLOCK_K
    )