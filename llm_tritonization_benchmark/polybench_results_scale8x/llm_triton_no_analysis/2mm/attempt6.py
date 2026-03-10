import triton
import triton.language as tl

@triton.jit
def k2mm_kernel(A_ptr, B_ptr, C_ptr, D_ptr, tmp_ptr, alpha, beta, 
                NI: tl.constexpr, NJ: tl.constexpr, NK: tl.constexpr, NL: tl.constexpr,
                BLOCK_SIZE: tl.constexpr):
    
    pid = tl.program_id(0)
    
    # First loop: tmp[i][j] = sum(alpha * A[i][k] * B[k][j])
    for i in range(NI):
        for j in range(NJ):
            tmp_val = 0.0
            for k in range(NK):
                a_val = tl.load(A_ptr + i * NK + k)
                b_val = tl.load(B_ptr + k * NJ + j)
                tmp_val += alpha * a_val * b_val
            tl.store(tmp_ptr + i * NJ + j, tmp_val)
    
    # Second loop: D[i][j] = beta * D[i][j] + sum(tmp[i][k] * C[k][j])
    for i in range(NI):
        for j in range(NL):
            d_val = tl.load(D_ptr + i * NL + j)
            d_val *= beta
            for k in range(NJ):
                tmp_val = tl.load(tmp_ptr + i * NJ + k)
                c_val = tl.load(C_ptr + k * NL + j)
                d_val += tmp_val * c_val
            tl.store(D_ptr + i * NL + j, d_val)

def k2mm_triton(A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL):
    grid = (1,)
    k2mm_kernel[grid](A, B, C, D, tmp, alpha, beta, NI, NJ, NK, NL, BLOCK_SIZE=16)