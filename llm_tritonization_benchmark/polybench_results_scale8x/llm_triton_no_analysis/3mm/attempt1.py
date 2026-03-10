import triton
import triton.language as tl
import torch

@triton.jit
def k3mm_kernel(A, B, C, D, E, F, G, 
                NI: tl.constexpr, NJ: tl.constexpr, NK: tl.constexpr, 
                NL: tl.constexpr, NM: tl.constexpr,
                BLOCK_SIZE: tl.constexpr):
    
    pid = tl.program_id(0)
    
    # Calculate which computation this program handles
    total_e = NI * NJ
    total_f = NJ * NL
    total_g = NI * NL
    
    if pid < triton.cdiv(total_e, BLOCK_SIZE):
        # E := A*B computation
        block_start = pid * BLOCK_SIZE
        offsets = tl.arange(0, BLOCK_SIZE)
        indices = block_start + offsets
        mask = indices < total_e
        
        # Convert linear indices to 2D coordinates
        i_coords = indices // NJ
        j_coords = indices % NJ
        
        # Initialize E values
        e_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        # Compute dot product for each (i,j) pair
        for k in range(NK):
            a_indices = i_coords * NK + k
            b_indices = k * NJ + j_coords
            
            a_mask = mask & (i_coords < NI)
            b_mask = mask & (j_coords < NJ)
            combined_mask = a_mask & b_mask
            
            a_vals = tl.load(A + a_indices, mask=combined_mask, other=0.0)
            b_vals = tl.load(B + b_indices, mask=combined_mask, other=0.0)
            
            e_vals += a_vals * b_vals
        
        # Store E results
        e_indices = i_coords * NJ + j_coords
        tl.store(E + e_indices, e_vals, mask=mask)
    
    elif pid < triton.cdiv(total_e, BLOCK_SIZE) + triton.cdiv(total_f, BLOCK_SIZE):
        # F := C*D computation
        f_pid = pid - triton.cdiv(total_e, BLOCK_SIZE)
        block_start = f_pid * BLOCK_SIZE
        offsets = tl.arange(0, BLOCK_SIZE)
        indices = block_start + offsets
        mask = indices < total_f
        
        # Convert linear indices to 2D coordinates
        i_coords = indices // NL
        j_coords = indices % NL
        
        # Initialize F values
        f_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        # Compute dot product for each (i,j) pair
        for k in range(NM):
            c_indices = i_coords * NM + k
            d_indices = k * NL + j_coords
            
            c_mask = mask & (i_coords < NJ)
            d_mask = mask & (j_coords < NL)
            combined_mask = c_mask & d_mask
            
            c_vals = tl.load(C + c_indices, mask=combined_mask, other=0.0)
            d_vals = tl.load(D + d_indices, mask=combined_mask, other=0.0)
            
            f_vals += c_vals * d_vals
        
        # Store F results
        f_indices = i_coords * NL + j_coords
        tl.store(F + f_indices, f_vals, mask=mask)
    
    else:
        # G := E*F computation
        g_pid = pid - triton.cdiv(total_e, BLOCK_SIZE) - triton.cdiv(total_f, BLOCK_SIZE)
        block_start = g_pid * BLOCK_SIZE
        offsets = tl.arange(0, BLOCK_SIZE)
        indices = block_start + offsets
        mask = indices < total_g
        
        # Convert linear indices to 2D coordinates
        i_coords = indices // NL
        j_coords = indices % NL
        
        # Initialize G values
        g_vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        
        # Compute dot product for each (i,j) pair
        for k in range(NJ):
            e_indices = i_coords * NJ + k
            f_indices = k * NL + j_coords
            
            e_mask = mask & (i_coords < NI)
            f_mask = mask & (j_coords < NL)
            combined_mask = e_mask & f_mask
            
            e_vals = tl.load(E + e_indices, mask=combined_mask, other=0.0)
            f_vals = tl.load(F + f_indices, mask=combined_mask, other=0.0)
            
            g_vals += e_vals * f_vals
        
        # Store G results
        g_indices = i_coords * NL + j_coords
        tl.store(G + g_indices, g_vals, mask=mask)

def k3mm_triton(A, B, C, D, E, F, G, NI, NJ, NK, NL, NM):
    BLOCK_SIZE = 256
    
    # Calculate total work for all three matrix multiplications
    total_e = NI * NJ
    total_f = NJ * NL  
    total_g = NI * NL
    
    # Total number of thread blocks needed
    num_programs = (triton.cdiv(total_e, BLOCK_SIZE) + 
                   triton.cdiv(total_f, BLOCK_SIZE) + 
                   triton.cdiv(total_g, BLOCK_SIZE))
    
    # Launch kernel
    k3mm_kernel[(num_programs,)](
        A, B, C, D, E, F, G,
        NI, NJ, NK, NL, NM,
        BLOCK_SIZE
    )