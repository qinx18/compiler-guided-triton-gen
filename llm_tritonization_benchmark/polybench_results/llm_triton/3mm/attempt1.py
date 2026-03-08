import triton
import triton.language as tl

@triton.jit
def k3mm_kernel_1(A_ptr, B_ptr, E_ptr, NI: tl.constexpr, NJ: tl.constexpr, NK: tl.constexpr, 
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    m_offsets = tl.arange(0, BLOCK_M)
    n_offsets = tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, NK, BLOCK_K):
        m_indices = m_start + m_offsets
        k_indices = k_start + k_offsets
        n_indices = n_start + n_offsets
        
        m_mask = m_indices < NI
        k_mask = k_indices < NK
        n_mask = n_indices < NJ
        
        a_ptrs = A_ptr + m_indices[:, None] * NK + k_indices[None, :]
        a_mask = m_mask[:, None] & k_mask[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        b_ptrs = B_ptr + k_indices[:, None] * NJ + n_indices[None, :]
        b_mask = k_mask[:, None] & n_mask[None, :]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        acc += tl.dot(a, b)
    
    m_indices = m_start + m_offsets
    n_indices = n_start + n_offsets
    e_ptrs = E_ptr + m_indices[:, None] * NJ + n_indices[None, :]
    e_mask = (m_indices < NI)[:, None] & (n_indices < NJ)[None, :]
    tl.store(e_ptrs, acc, mask=e_mask)

@triton.jit
def k3mm_kernel_2(C_ptr, D_ptr, F_ptr, NJ: tl.constexpr, NL: tl.constexpr, NM: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    m_offsets = tl.arange(0, BLOCK_M)
    n_offsets = tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, NM, BLOCK_K):
        m_indices = m_start + m_offsets
        k_indices = k_start + k_offsets
        n_indices = n_start + n_offsets
        
        m_mask = m_indices < NJ
        k_mask = k_indices < NM
        n_mask = n_indices < NL
        
        c_ptrs = C_ptr + m_indices[:, None] * NM + k_indices[None, :]
        c_mask = m_mask[:, None] & k_mask[None, :]
        c = tl.load(c_ptrs, mask=c_mask, other=0.0)
        
        d_ptrs = D_ptr + k_indices[:, None] * NL + n_indices[None, :]
        d_mask = k_mask[:, None] & n_mask[None, :]
        d = tl.load(d_ptrs, mask=d_mask, other=0.0)
        
        acc += tl.dot(c, d)
    
    m_indices = m_start + m_offsets
    n_indices = n_start + n_offsets
    f_ptrs = F_ptr + m_indices[:, None] * NL + n_indices[None, :]
    f_mask = (m_indices < NJ)[:, None] & (n_indices < NL)[None, :]
    tl.store(f_ptrs, acc, mask=f_mask)

@triton.jit
def k3mm_kernel_3(E_ptr, F_ptr, G_ptr, NI: tl.constexpr, NL: tl.constexpr, NJ: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    m_offsets = tl.arange(0, BLOCK_M)
    n_offsets = tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, NJ, BLOCK_K):
        m_indices = m_start + m_offsets
        k_indices = k_start + k_offsets
        n_indices = n_start + n_offsets
        
        m_mask = m_indices < NI
        k_mask = k_indices < NJ
        n_mask = n_indices < NL
        
        e_ptrs = E_ptr + m_indices[:, None] * NJ + k_indices[None, :]
        e_mask = m_mask[:, None] & k_mask[None, :]
        e = tl.load(e_ptrs, mask=e_mask, other=0.0)
        
        f_ptrs = F_ptr + k_indices[:, None] * NL + n_indices[None, :]
        f_mask = k_mask[:, None] & n_mask[None, :]
        f = tl.load(f_ptrs, mask=f_mask, other=0.0)
        
        acc += tl.dot(e, f)
    
    m_indices = m_start + m_offsets
    n_indices = n_start + n_offsets
    g_ptrs = G_ptr + m_indices[:, None] * NL + n_indices[None, :]
    g_mask = (m_indices < NI)[:, None] & (n_indices < NL)[None, :]
    tl.store(g_ptrs, acc, mask=g_mask)

def k3mm_triton(A, B, C, D, E, F, G, NI, NJ, NK, NL, NM):
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 16
    
    grid_1 = (triton.cdiv(NI, BLOCK_M), triton.cdiv(NJ, BLOCK_N))
    k3mm_kernel_1[grid_1](A, B, E, NI, NJ, NK, BLOCK_M, BLOCK_N, BLOCK_K)
    
    grid_2 = (triton.cdiv(NJ, BLOCK_M), triton.cdiv(NL, BLOCK_N))
    k3mm_kernel_2[grid_2](C, D, F, NJ, NL, NM, BLOCK_M, BLOCK_N, BLOCK_K)
    
    grid_3 = (triton.cdiv(NI, BLOCK_M), triton.cdiv(NL, BLOCK_N))
    k3mm_kernel_3[grid_3](E, F, G, NI, NL, NJ, BLOCK_M, BLOCK_N, BLOCK_K)