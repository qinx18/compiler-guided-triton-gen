import triton
import triton.language as tl
import torch

@triton.jit
def k3mm_kernel(A, B, C, D, E, F, G, NI, NJ, NK, NL, NM, 
                BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr):
    # Get program IDs
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    pid_stage = tl.program_id(2)
    
    # Define offsets once
    i_offsets = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    j_offsets = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
    k_offsets = tl.arange(0, BLOCK_K)
    
    if pid_stage == 0:
        # E := A*B (NI x NJ)
        i_mask = i_offsets < NI
        j_mask = j_offsets < NJ
        
        acc = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)
        
        for k_start in range(0, NK, BLOCK_K):
            k_current = k_start + k_offsets
            k_mask = k_current < NK
            
            # Load A[i, k]
            a_ptrs = A + i_offsets[:, None] * NK + k_current[None, :]
            a_vals = tl.load(a_ptrs, mask=i_mask[:, None] & k_mask[None, :], other=0.0)
            
            # Load B[k, j]
            b_ptrs = B + k_current[:, None] * NJ + j_offsets[None, :]
            b_vals = tl.load(b_ptrs, mask=k_mask[:, None] & j_mask[None, :], other=0.0)
            
            # Accumulate
            acc += tl.dot(a_vals, b_vals)
        
        # Store E[i, j]
        e_ptrs = E + i_offsets[:, None] * NJ + j_offsets[None, :]
        tl.store(e_ptrs, acc, mask=i_mask[:, None] & j_mask[None, :])
    
    elif pid_stage == 1:
        # F := C*D (NJ x NL)
        # Reuse i_offsets for NJ dimension, j_offsets for NL dimension
        i_mask = i_offsets < NJ
        j_mask = j_offsets < NL
        
        acc = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)
        
        for k_start in range(0, NM, BLOCK_K):
            k_current = k_start + k_offsets
            k_mask = k_current < NM
            
            # Load C[i, k] where i is in NJ range
            c_ptrs = C + i_offsets[:, None] * NM + k_current[None, :]
            c_vals = tl.load(c_ptrs, mask=i_mask[:, None] & k_mask[None, :], other=0.0)
            
            # Load D[k, j] where j is in NL range
            d_ptrs = D + k_current[:, None] * NL + j_offsets[None, :]
            d_vals = tl.load(d_ptrs, mask=k_mask[:, None] & j_mask[None, :], other=0.0)
            
            # Accumulate
            acc += tl.dot(c_vals, d_vals)
        
        # Store F[i, j] where i is NJ, j is NL
        f_ptrs = F + i_offsets[:, None] * NL + j_offsets[None, :]
        tl.store(f_ptrs, acc, mask=i_mask[:, None] & j_mask[None, :])
    
    else:  # pid_stage == 2
        # G := E*F (NI x NL)
        # i_offsets for NI dimension, j_offsets for NL dimension
        i_mask = i_offsets < NI
        j_mask = j_offsets < NL
        
        acc = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)
        
        for k_start in range(0, NJ, BLOCK_K):
            k_current = k_start + k_offsets
            k_mask = k_current < NJ
            
            # Load E[i, k] where i is in NI range, k is in NJ range
            e_ptrs = E + i_offsets[:, None] * NJ + k_current[None, :]
            e_vals = tl.load(e_ptrs, mask=i_mask[:, None] & k_mask[None, :], other=0.0)
            
            # Load F[k, j] where k is in NJ range, j is in NL range
            f_ptrs = F + k_current[:, None] * NL + j_offsets[None, :]
            f_vals = tl.load(f_ptrs, mask=k_mask[:, None] & j_mask[None, :], other=0.0)
            
            # Accumulate
            acc += tl.dot(e_vals, f_vals)
        
        # Store G[i, j] where i is NI, j is NL
        g_ptrs = G + i_offsets[:, None] * NL + j_offsets[None, :]
        tl.store(g_ptrs, acc, mask=i_mask[:, None] & j_mask[None, :])

def k3mm_triton(A, B, C, D, E, F, G, NI, NJ, NK, NL, NM):
    BLOCK_I = 16
    BLOCK_J = 16
    BLOCK_K = 32
    
    # Stage 0: E := A*B (NI x NJ)
    grid_0 = (triton.cdiv(NI, BLOCK_I), triton.cdiv(NJ, BLOCK_J), 1)
    k3mm_kernel[grid_0](
        A, B, C, D, E, F, G, NI, NJ, NK, NL, NM,
        BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J, BLOCK_K=BLOCK_K
    )
    
    # Stage 1: F := C*D (NJ x NL)
    grid_1 = (triton.cdiv(NJ, BLOCK_I), triton.cdiv(NL, BLOCK_J), 1)
    k3mm_kernel[grid_1](
        A, B, C, D, E, F, G, NI, NJ, NK, NL, NM,
        BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J, BLOCK_K=BLOCK_K
    )
    
    # Stage 2: G := E*F (NI x NL)
    grid_2 = (triton.cdiv(NI, BLOCK_I), triton.cdiv(NL, BLOCK_J), 1)
    k3mm_kernel[grid_2](
        A, B, C, D, E, F, G, NI, NJ, NK, NL, NM,
        BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J, BLOCK_K=BLOCK_K
    )