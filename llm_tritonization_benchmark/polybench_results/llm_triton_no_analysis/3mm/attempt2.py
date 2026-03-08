import torch
import triton
import triton.language as tl

@triton.jit
def k3mm_kernel_e(A_ptr, B_ptr, E_ptr,
                  NI: tl.constexpr, NJ: tl.constexpr, NK: tl.constexpr,
                  BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr):
    
    block_i = tl.program_id(0)
    block_j = tl.program_id(1)
    
    i_offsets = tl.arange(0, BLOCK_I)
    j_offsets = tl.arange(0, BLOCK_J)
    k_offsets = tl.arange(0, BLOCK_K)
    
    i_idx = block_i * BLOCK_I + i_offsets
    j_idx = block_j * BLOCK_J + j_offsets
    
    i_mask = i_idx < NI
    j_mask = j_idx < NJ
    
    acc = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)
    
    for k in range(0, NK, BLOCK_K):
        k_idx = k + k_offsets
        k_mask = k_idx < NK
        
        A_ptrs = A_ptr + i_idx[:, None] * NK + k_idx[None, :]
        A_vals = tl.load(A_ptrs, mask=i_mask[:, None] & k_mask[None, :], other=0.0)
        
        B_ptrs = B_ptr + k_idx[:, None] * NJ + j_idx[None, :]
        B_vals = tl.load(B_ptrs, mask=k_mask[:, None] & j_mask[None, :], other=0.0)
        
        acc += tl.dot(A_vals, B_vals)
    
    E_ptrs = E_ptr + i_idx[:, None] * NJ + j_idx[None, :]
    tl.store(E_ptrs, acc, mask=i_mask[:, None] & j_mask[None, :])

@triton.jit
def k3mm_kernel_f(C_ptr, D_ptr, F_ptr,
                  NJ: tl.constexpr, NL: tl.constexpr, NM: tl.constexpr,
                  BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr):
    
    block_i = tl.program_id(0)
    block_j = tl.program_id(1)
    
    i_offsets = tl.arange(0, BLOCK_I)
    j_offsets = tl.arange(0, BLOCK_J)
    k_offsets = tl.arange(0, BLOCK_K)
    
    i_idx = block_i * BLOCK_I + i_offsets
    j_idx = block_j * BLOCK_J + j_offsets
    
    i_mask = i_idx < NJ
    j_mask = j_idx < NL
    
    acc = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)
    
    for k in range(0, NM, BLOCK_K):
        k_idx = k + k_offsets
        k_mask = k_idx < NM
        
        C_ptrs = C_ptr + i_idx[:, None] * NM + k_idx[None, :]
        C_vals = tl.load(C_ptrs, mask=i_mask[:, None] & k_mask[None, :], other=0.0)
        
        D_ptrs = D_ptr + k_idx[:, None] * NL + j_idx[None, :]
        D_vals = tl.load(D_ptrs, mask=k_mask[:, None] & j_mask[None, :], other=0.0)
        
        acc += tl.dot(C_vals, D_vals)
    
    F_ptrs = F_ptr + i_idx[:, None] * NL + j_idx[None, :]
    tl.store(F_ptrs, acc, mask=i_mask[:, None] & j_mask[None, :])

@triton.jit
def k3mm_kernel_g(E_ptr, F_ptr, G_ptr,
                  NI: tl.constexpr, NJ: tl.constexpr, NL: tl.constexpr,
                  BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr):
    
    block_i = tl.program_id(0)
    block_j = tl.program_id(1)
    
    i_offsets = tl.arange(0, BLOCK_I)
    j_offsets = tl.arange(0, BLOCK_J)
    k_offsets = tl.arange(0, BLOCK_K)
    
    i_idx = block_i * BLOCK_I + i_offsets
    j_idx = block_j * BLOCK_J + j_offsets
    
    i_mask = i_idx < NI
    j_mask = j_idx < NL
    
    acc = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)
    
    for k in range(0, NJ, BLOCK_K):
        k_idx = k + k_offsets
        k_mask = k_idx < NJ
        
        E_ptrs = E_ptr + i_idx[:, None] * NJ + k_idx[None, :]
        E_vals = tl.load(E_ptrs, mask=i_mask[:, None] & k_mask[None, :], other=0.0)
        
        F_ptrs = F_ptr + k_idx[:, None] * NL + j_idx[None, :]
        F_vals = tl.load(F_ptrs, mask=k_mask[:, None] & j_mask[None, :], other=0.0)
        
        acc += tl.dot(E_vals, F_vals)
    
    G_ptrs = G_ptr + i_idx[:, None] * NL + j_idx[None, :]
    tl.store(G_ptrs, acc, mask=i_mask[:, None] & j_mask[None, :])

def k3mm_triton(A, B, C, D, E, F, G, NI, NJ, NK, NL, NM):
    BLOCK_I = 16
    BLOCK_J = 16
    BLOCK_K = 16
    
    grid_E = (triton.cdiv(NI, BLOCK_I), triton.cdiv(NJ, BLOCK_J))
    k3mm_kernel_e[grid_E](
        A, B, E,
        NI, NJ, NK,
        BLOCK_I, BLOCK_J, BLOCK_K
    )
    
    grid_F = (triton.cdiv(NJ, BLOCK_I), triton.cdiv(NL, BLOCK_J))
    k3mm_kernel_f[grid_F](
        C, D, F,
        NJ, NL, NM,
        BLOCK_I, BLOCK_J, BLOCK_K
    )
    
    grid_G = (triton.cdiv(NI, BLOCK_I), triton.cdiv(NL, BLOCK_J))
    k3mm_kernel_g[grid_G](
        E, F, G,
        NI, NJ, NL,
        BLOCK_I, BLOCK_J, BLOCK_K
    )