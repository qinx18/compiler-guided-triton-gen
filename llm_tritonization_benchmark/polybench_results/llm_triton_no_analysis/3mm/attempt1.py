import triton
import triton.language as tl
import torch

@triton.jit
def k3mm_kernel(A_ptr, B_ptr, C_ptr, D_ptr, E_ptr, F_ptr, G_ptr,
                NI, NJ, NK, NL, NM,
                BLOCK_SIZE: tl.constexpr):
    
    # E := A*B (NI x NJ = NI x NK * NK x NJ)
    for i_block_start in range(0, NI, BLOCK_SIZE):
        for j_block_start in range(0, NJ, BLOCK_SIZE):
            i_offsets = tl.arange(0, BLOCK_SIZE)
            j_offsets = tl.arange(0, BLOCK_SIZE)
            i_indices = i_block_start + i_offsets
            j_indices = j_block_start + j_offsets
            
            i_mask = i_indices < NI
            j_mask = j_indices < NL
            
            # Initialize accumulator
            acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
            
            # Compute dot product
            for k in range(NK):
                # Load A[i, k]
                a_ptrs = A_ptr + i_indices[:, None] * NK + k
                a_vals = tl.load(a_ptrs, mask=i_mask[:, None], other=0.0)
                
                # Load B[k, j]
                b_ptrs = B_ptr + k * NJ + j_indices[None, :]
                b_vals = tl.load(b_ptrs, mask=j_mask[None, :], other=0.0)
                
                # Accumulate
                acc += a_vals * b_vals
            
            # Store E[i, j]
            e_ptrs = E_ptr + i_indices[:, None] * NJ + j_indices[None, :]
            mask_2d = i_mask[:, None] & j_mask[None, :]
            tl.store(e_ptrs, acc, mask=mask_2d)
    
    # F := C*D (NJ x NL = NJ x NM * NM x NL)
    for i_block_start in range(0, NJ, BLOCK_SIZE):
        for j_block_start in range(0, NL, BLOCK_SIZE):
            i_offsets = tl.arange(0, BLOCK_SIZE)
            j_offsets = tl.arange(0, BLOCK_SIZE)
            i_indices = i_block_start + i_offsets
            j_indices = j_block_start + j_offsets
            
            i_mask = i_indices < NJ
            j_mask = j_indices < NL
            
            # Initialize accumulator
            acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
            
            # Compute dot product
            for k in range(NM):
                # Load C[i, k]
                c_ptrs = C_ptr + i_indices[:, None] * NM + k
                c_vals = tl.load(c_ptrs, mask=i_mask[:, None], other=0.0)
                
                # Load D[k, j]
                d_ptrs = D_ptr + k * NL + j_indices[None, :]
                d_vals = tl.load(d_ptrs, mask=j_mask[None, :], other=0.0)
                
                # Accumulate
                acc += c_vals * d_vals
            
            # Store F[i, j]
            f_ptrs = F_ptr + i_indices[:, None] * NL + j_indices[None, :]
            mask_2d = i_mask[:, None] & j_mask[None, :]
            tl.store(f_ptrs, acc, mask=mask_2d)
    
    # G := E*F (NI x NL = NI x NJ * NJ x NL)
    for i_block_start in range(0, NI, BLOCK_SIZE):
        for j_block_start in range(0, NL, BLOCK_SIZE):
            i_offsets = tl.arange(0, BLOCK_SIZE)
            j_offsets = tl.arange(0, BLOCK_SIZE)
            i_indices = i_block_start + i_offsets
            j_indices = j_block_start + j_offsets
            
            i_mask = i_indices < NI
            j_mask = j_indices < NL
            
            # Initialize accumulator
            acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
            
            # Compute dot product
            for k in range(NJ):
                # Load E[i, k]
                e_ptrs = E_ptr + i_indices[:, None] * NJ + k
                e_vals = tl.load(e_ptrs, mask=i_mask[:, None], other=0.0)
                
                # Load F[k, j]
                f_ptrs = F_ptr + k * NL + j_indices[None, :]
                f_vals = tl.load(f_ptrs, mask=j_mask[None, :], other=0.0)
                
                # Accumulate
                acc += e_vals * f_vals
            
            # Store G[i, j]
            g_ptrs = G_ptr + i_indices[:, None] * NL + j_indices[None, :]
            mask_2d = i_mask[:, None] & j_mask[None, :]
            tl.store(g_ptrs, acc, mask=mask_2d)

def k3mm_triton(A, B, C, D, E, F, G, NI, NJ, NK, NL, NM):
    BLOCK_SIZE = 32
    
    grid = (1,)
    
    k3mm_kernel[grid](
        A, B, C, D, E, F, G,
        NI, NJ, NK, NL, NM,
        BLOCK_SIZE=BLOCK_SIZE
    )