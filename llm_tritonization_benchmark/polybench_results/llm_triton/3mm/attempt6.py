import triton
import triton.language as tl

@triton.jit
def k3mm_kernel(A_ptr, B_ptr, C_ptr, D_ptr, E_ptr, F_ptr, G_ptr, 
                NI, NJ, NK, NL, NM,
                BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr, BLOCK_SIZE_L: tl.constexpr):
    
    # First kernel: E := A*B
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_start = pid_i * BLOCK_SIZE_I
    j_start = pid_j * BLOCK_SIZE_J
    
    i_offsets = i_start + tl.arange(0, BLOCK_SIZE_I)
    j_offsets = j_start + tl.arange(0, BLOCK_SIZE_J)
    
    i_mask = i_offsets < NI
    j_mask = j_offsets < NJ
    
    # Initialize E[i,j] = 0
    e_acc = tl.zeros((BLOCK_SIZE_I, BLOCK_SIZE_J), dtype=tl.float32)
    
    # E[i,j] += A[i,k] * B[k,j]
    for k in range(0, NK):
        a_ptrs = A_ptr + i_offsets[:, None] * NK + k
        b_ptrs = B_ptr + k * NJ + j_offsets[None, :]
        
        a_vals = tl.load(a_ptrs, mask=i_mask[:, None], other=0.0)
        b_vals = tl.load(b_ptrs, mask=j_mask[None, :], other=0.0)
        
        e_acc += a_vals * b_vals
    
    # Store E
    e_ptrs = E_ptr + i_offsets[:, None] * NJ + j_offsets[None, :]
    mask = i_mask[:, None] & j_mask[None, :]
    tl.store(e_ptrs, e_acc, mask=mask)

@triton.jit
def k3mm_kernel_f(C_ptr, D_ptr, F_ptr, NJ, NL, NM,
                  BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr):
    
    # Second kernel: F := C*D
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_start = pid_i * BLOCK_SIZE_I
    j_start = pid_j * BLOCK_SIZE_J
    
    i_offsets = i_start + tl.arange(0, BLOCK_SIZE_I)
    j_offsets = j_start + tl.arange(0, BLOCK_SIZE_J)
    
    i_mask = i_offsets < NJ
    j_mask = j_offsets < NL
    
    # Initialize F[i,j] = 0
    f_acc = tl.zeros((BLOCK_SIZE_I, BLOCK_SIZE_J), dtype=tl.float32)
    
    # F[i,j] += C[i,k] * D[k,j]
    for k in range(0, NM):
        c_ptrs = C_ptr + i_offsets[:, None] * NM + k
        d_ptrs = D_ptr + k * NL + j_offsets[None, :]
        
        c_vals = tl.load(c_ptrs, mask=i_mask[:, None], other=0.0)
        d_vals = tl.load(d_ptrs, mask=j_mask[None, :], other=0.0)
        
        f_acc += c_vals * d_vals
    
    # Store F
    f_ptrs = F_ptr + i_offsets[:, None] * NL + j_offsets[None, :]
    mask = i_mask[:, None] & j_mask[None, :]
    tl.store(f_ptrs, f_acc, mask=mask)

@triton.jit
def k3mm_kernel_g(E_ptr, F_ptr, G_ptr, NI, NJ, NL,
                  BLOCK_SIZE_I: tl.constexpr, BLOCK_SIZE_J: tl.constexpr):
    
    # Third kernel: G := E*F
    pid_i = tl.program_id(0)
    pid_j = tl.program_id(1)
    
    i_start = pid_i * BLOCK_SIZE_I
    j_start = pid_j * BLOCK_SIZE_J
    
    i_offsets = i_start + tl.arange(0, BLOCK_SIZE_I)
    j_offsets = j_start + tl.arange(0, BLOCK_SIZE_J)
    
    i_mask = i_offsets < NI
    j_mask = j_offsets < NL
    
    # Initialize G[i,j] = 0
    g_acc = tl.zeros((BLOCK_SIZE_I, BLOCK_SIZE_J), dtype=tl.float32)
    
    # G[i,j] += E[i,k] * F[k,j]
    for k in range(0, NJ):
        e_ptrs = E_ptr + i_offsets[:, None] * NJ + k
        f_ptrs = F_ptr + k * NL + j_offsets[None, :]
        
        e_vals = tl.load(e_ptrs, mask=i_mask[:, None], other=0.0)
        f_vals = tl.load(f_ptrs, mask=j_mask[None, :], other=0.0)
        
        g_acc += e_vals * f_vals
    
    # Store G
    g_ptrs = G_ptr + i_offsets[:, None] * NL + j_offsets[None, :]
    mask = i_mask[:, None] & j_mask[None, :]
    tl.store(g_ptrs, g_acc, mask=mask)

def k3mm_triton(A, B, C, D, E, F, G, NI, NJ, NK, NL, NM):
    BLOCK_SIZE_I = 16
    BLOCK_SIZE_J = 16
    BLOCK_SIZE_L = 16
    
    # Launch first kernel: E := A*B
    grid1 = (triton.cdiv(NI, BLOCK_SIZE_I), triton.cdiv(NJ, BLOCK_SIZE_J))
    k3mm_kernel[grid1](A, B, C, D, E, F, G, NI, NJ, NK, NL, NM, BLOCK_SIZE_I, BLOCK_SIZE_J, BLOCK_SIZE_L)
    
    # Launch second kernel: F := C*D  
    grid2 = (triton.cdiv(NJ, BLOCK_SIZE_I), triton.cdiv(NL, BLOCK_SIZE_J))
    k3mm_kernel_f[grid2](C, D, F, NJ, NL, NM, BLOCK_SIZE_I, BLOCK_SIZE_J)
    
    # Launch third kernel: G := E*F
    grid3 = (triton.cdiv(NI, BLOCK_SIZE_I), triton.cdiv(NL, BLOCK_SIZE_J))
    k3mm_kernel_g[grid3](E, F, G, NI, NJ, NL, BLOCK_SIZE_I, BLOCK_SIZE_J)