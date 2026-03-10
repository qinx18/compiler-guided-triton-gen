import triton
import triton.language as tl
import torch

@triton.jit
def k3mm_kernel_e(A, B, E, 
                  NI: tl.constexpr, NJ: tl.constexpr, NK: tl.constexpr,
                  BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < NI * NJ
    
    i_coords = indices // NJ
    j_coords = indices % NJ
    
    e_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for k in range(NK):
        a_indices = i_coords * NK + k
        b_indices = k * NJ + j_coords
        
        a_vals = tl.load(A + a_indices, mask=mask, other=0.0)
        b_vals = tl.load(B + b_indices, mask=mask, other=0.0)
        
        e_vals += a_vals * b_vals
    
    e_indices = i_coords * NJ + j_coords
    tl.store(E + e_indices, e_vals, mask=mask)

@triton.jit
def k3mm_kernel_f(C, D, F,
                  NJ: tl.constexpr, NL: tl.constexpr, NM: tl.constexpr,
                  BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < NJ * NL
    
    i_coords = indices // NL
    j_coords = indices % NL
    
    f_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for k in range(NM):
        c_indices = i_coords * NM + k
        d_indices = k * NL + j_coords
        
        c_vals = tl.load(C + c_indices, mask=mask, other=0.0)
        d_vals = tl.load(D + d_indices, mask=mask, other=0.0)
        
        f_vals += c_vals * d_vals
    
    f_indices = i_coords * NL + j_coords
    tl.store(F + f_indices, f_vals, mask=mask)

@triton.jit
def k3mm_kernel_g(E, F, G,
                  NI: tl.constexpr, NJ: tl.constexpr, NL: tl.constexpr,
                  BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < NI * NL
    
    i_coords = indices // NL
    j_coords = indices % NL
    
    g_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for k in range(NJ):
        e_indices = i_coords * NJ + k
        f_indices = k * NL + j_coords
        
        e_vals = tl.load(E + e_indices, mask=mask, other=0.0)
        f_vals = tl.load(F + f_indices, mask=mask, other=0.0)
        
        g_vals += e_vals * f_vals
    
    g_indices = i_coords * NL + j_coords
    tl.store(G + g_indices, g_vals, mask=mask)

def k3mm_triton(A, B, C, D, E, F, G, NI, NJ, NK, NL, NM):
    BLOCK_SIZE = 256
    
    # E := A*B
    grid_e = triton.cdiv(NI * NJ, BLOCK_SIZE)
    k3mm_kernel_e[(grid_e,)](A, B, E, NI, NJ, NK, BLOCK_SIZE)
    
    # F := C*D
    grid_f = triton.cdiv(NJ * NL, BLOCK_SIZE)
    k3mm_kernel_f[(grid_f,)](C, D, F, NJ, NL, NM, BLOCK_SIZE)
    
    # G := E*F
    grid_g = triton.cdiv(NI * NL, BLOCK_SIZE)
    k3mm_kernel_g[(grid_g,)](E, F, G, NI, NJ, NL, BLOCK_SIZE)