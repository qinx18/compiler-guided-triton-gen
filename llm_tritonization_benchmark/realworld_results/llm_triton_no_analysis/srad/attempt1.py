import torch
import triton
import triton.language as tl

@triton.jit
def srad_kernel(
    J_ptr, c_ptr, dE_ptr, dN_ptr, dS_ptr, dW_ptr,
    iN_ptr, iS_ptr, jE_ptr, jW_ptr,
    lambda_val,
    C1: tl.constexpr, C2: tl.constexpr, COLS: tl.constexpr,
    NITER: tl.constexpr, R1: tl.constexpr, R2: tl.constexpr,
    ROWS: tl.constexpr, SIZE_I: tl.constexpr, SIZE_R: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # Compute offsets once at start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    for iter in range(NITER):
        # Phase 0: ROI statistics reduction
        sum_val = 0.0
        sum2_val = 0.0
        
        for i in range(R1, R2 + 1):
            for j in range(C1, C2 + 1):
                idx = i * COLS + j
                tmp = tl.load(J_ptr + idx)
                sum_val += tmp
                sum2_val += tmp * tmp
        
        meanROI = sum_val / SIZE_R
        varROI = (sum2_val / SIZE_R) - meanROI * meanROI
        q0sqr = varROI / (meanROI * meanROI)
        
        # Phase 1: Process in blocks
        for block_start in range(0, SIZE_I, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < SIZE_I
            
            # Load current J values
            Jc_vals = tl.load(J_ptr + current_offsets, mask=mask)
            
            # Compute i, j for each offset
            i_vals = current_offsets // COLS
            j_vals = current_offsets % COLS
            
            # Load boundary indices
            iN_vals = tl.load(iN_ptr + i_vals, mask=mask)
            iS_vals = tl.load(iS_ptr + i_vals, mask=mask)
            jW_vals = tl.load(jW_ptr + j_vals, mask=mask)
            jE_vals = tl.load(jE_ptr + j_vals, mask=mask)
            
            # Load neighbor values
            J_N = tl.load(J_ptr + iN_vals * COLS + j_vals, mask=mask)
            J_S = tl.load(J_ptr + iS_vals * COLS + j_vals, mask=mask)
            J_W = tl.load(J_ptr + i_vals * COLS + jW_vals, mask=mask)
            J_E = tl.load(J_ptr + i_vals * COLS + jE_vals, mask=mask)
            
            # Compute derivatives
            dN_vals = J_N - Jc_vals
            dS_vals = J_S - Jc_vals
            dW_vals = J_W - Jc_vals
            dE_vals = J_E - Jc_vals
            
            # Store derivatives
            tl.store(dN_ptr + current_offsets, dN_vals, mask=mask)
            tl.store(dS_ptr + current_offsets, dS_vals, mask=mask)
            tl.store(dW_ptr + current_offsets, dW_vals, mask=mask)
            tl.store(dE_ptr + current_offsets, dE_vals, mask=mask)
            
            # Compute diffusion coefficient
            G2 = (dN_vals * dN_vals + dS_vals * dS_vals + 
                  dW_vals * dW_vals + dE_vals * dE_vals) / (Jc_vals * Jc_vals)
            
            L = (dN_vals + dS_vals + dW_vals + dE_vals) / Jc_vals
            
            num = (0.5 * G2) - ((1.0 / 16.0) * (L * L))
            den = 1.0 + (0.25 * L)
            qsqr = num / (den * den)
            
            den = (qsqr - q0sqr) / (q0sqr * (1.0 + q0sqr))
            c_vals = 1.0 / (1.0 + den)
            
            # Clamp to [0, 1]
            c_vals = tl.where(c_vals < 0.0, 0.0, c_vals)
            c_vals = tl.where(c_vals > 1.0, 1.0, c_vals)
            
            tl.store(c_ptr + current_offsets, c_vals, mask=mask)
        
        # Phase 2: Image update in blocks
        for block_start in range(0, SIZE_I, BLOCK_SIZE):
            current_offsets = block_start + offsets
            mask = current_offsets < SIZE_I
            
            # Compute i, j for each offset
            i_vals = current_offsets // COLS
            j_vals = current_offsets % COLS
            
            # Load boundary indices
            iS_vals = tl.load(iS_ptr + i_vals, mask=mask)
            jE_vals = tl.load(jE_ptr + j_vals, mask=mask)
            
            # Load diffusion coefficients
            cN_vals = tl.load(c_ptr + current_offsets, mask=mask)
            cS_vals = tl.load(c_ptr + iS_vals * COLS + j_vals, mask=mask)
            cW_vals = tl.load(c_ptr + current_offsets, mask=mask)
            cE_vals = tl.load(c_ptr + i_vals * COLS + jE_vals, mask=mask)
            
            # Load derivatives
            dN_vals = tl.load(dN_ptr + current_offsets, mask=mask)
            dS_vals = tl.load(dS_ptr + current_offsets, mask=mask)
            dW_vals = tl.load(dW_ptr + current_offsets, mask=mask)
            dE_vals = tl.load(dE_ptr + current_offsets, mask=mask)
            
            # Compute divergence
            D = cN_vals * dN_vals + cS_vals * dS_vals + cW_vals * dW_vals + cE_vals * dE_vals
            
            # Load current J and update
            J_vals = tl.load(J_ptr + current_offsets, mask=mask)
            J_new = J_vals + 0.25 * lambda_val * D
            
            tl.store(J_ptr + current_offsets, J_new, mask=mask)

def srad_triton(J, c, dE, dN, dS, dW, iN, iS, jE, jW, lambda_val, C1, C2, COLS, NITER, R1, R2, ROWS):
    SIZE_I = ROWS * COLS
    SIZE_R = (R2 - R1 + 1) * (C2 - C1 + 1)
    BLOCK_SIZE = 256
    
    grid = (1,)
    
    srad_kernel[grid](
        J, c, dE, dN, dS, dW,
        iN, iS, jE, jW,
        lambda_val,
        C1, C2, COLS, NITER, R1, R2, ROWS,
        SIZE_I, SIZE_R, BLOCK_SIZE
    )