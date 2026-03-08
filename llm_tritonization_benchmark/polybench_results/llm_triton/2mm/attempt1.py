import triton
import triton.language as tl

@triton.jit
def k2mm_kernel(A_ptr, B_ptr, C_ptr, D_ptr, tmp_ptr, alpha, beta, 
                NI: tl.constexpr, NJ: tl.constexpr, NK: tl.constexpr, NL: tl.constexpr,
                BLOCK_SIZE: tl.constexpr):
    
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # First phase: tmp = alpha * A * B
    for i in range(NI):
        # Vectorized computation over j dimension
        j_mask = offsets < NJ
        j_offsets = offsets
        
        # Initialize tmp[i][j] = 0.0
        tmp_indices = i * NJ + j_offsets
        tl.store(tmp_ptr + tmp_indices, 0.0, mask=j_mask)
        
        # Accumulate: tmp[i][j] += alpha * A[i][k] * B[k][j]
        for k in range(NK):
            a_val = tl.load(A_ptr + i * NK + k)  # A[i][k] - scalar
            b_indices = k * NJ + j_offsets  # B[k][j] - vector
            b_vals = tl.load(B_ptr + b_indices, mask=j_mask)
            
            # Load current tmp values
            tmp_vals = tl.load(tmp_ptr + tmp_indices, mask=j_mask)
            
            # Update tmp[i][j] += alpha * A[i][k] * B[k][j]
            new_tmp_vals = tmp_vals + alpha * a_val * b_vals
            tl.store(tmp_ptr + tmp_indices, new_tmp_vals, mask=j_mask)
    
    tl.debug_barrier()
    
    # Second phase: D = tmp * C + beta * D
    for i in range(NI):
        # Vectorized computation over j dimension (now for NL)
        j_mask = offsets < NL
        j_offsets = offsets
        
        # D[i][j] *= beta
        d_indices = i * NL + j_offsets
        d_vals = tl.load(D_ptr + d_indices, mask=j_mask)
        d_vals = d_vals * beta
        
        # Accumulate: D[i][j] += tmp[i][k] * C[k][j]
        for k in range(NJ):
            tmp_val = tl.load(tmp_ptr + i * NJ + k)  # tmp[i][k] - scalar
            c_indices = k * NL + j_offsets  # C[k][j] - vector
            c_vals = tl.load(C_ptr + c_indices, mask=j_mask)
            
            # Update D[i][j] += tmp[i][k] * C[k][j]
            d_vals = d_vals + tmp_val * c_vals
        
        tl.store(D_ptr + d_indices, d_vals, mask=j_mask)

def k2mm_triton(A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL):
    BLOCK_SIZE = triton.next_power_of_2(max(NJ, NL))
    
    k2mm_kernel[(1,)](
        A, B, C, D, tmp, alpha, beta,
        NI, NJ, NK, NL, BLOCK_SIZE
    )