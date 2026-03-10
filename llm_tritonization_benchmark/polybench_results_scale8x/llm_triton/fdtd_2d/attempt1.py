import triton
import triton.language as tl

@triton.jit
def fdtd_2d_kernel(_fict_, ex, ey, hz, NX: tl.constexpr, NY: tl.constexpr, TMAX: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Define offsets once at the start
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Time loop (sequential)
    for t in range(TMAX):
        # Phase 1: ey[0][j] = _fict_[t]
        for j_start in range(0, NY, BLOCK_SIZE):
            j_offsets = j_start + offsets
            j_mask = j_offsets < NY
            
            # Load _fict_[t]
            fict_val = tl.load(_fict_ + t)
            
            # Store to ey[0][j]
            ey_ptr = ey + j_offsets  # ey[0][j] = 0 * NY + j
            tl.store(ey_ptr, fict_val, mask=j_mask)
        
        # Phase 2: ey[i][j] = ey[i][j] - 0.5*(hz[i][j]-hz[i-1][j]) for i >= 1
        for i in range(1, NX):
            for j_start in range(0, NY, BLOCK_SIZE):
                j_offsets = j_start + offsets
                j_mask = j_offsets < NY
                
                # Linear indices
                ey_indices = i * NY + j_offsets
                hz_indices = i * NY + j_offsets
                hz_prev_indices = (i - 1) * NY + j_offsets
                
                # Load values
                ey_vals = tl.load(ey + ey_indices, mask=j_mask)
                hz_vals = tl.load(hz + hz_indices, mask=j_mask)
                hz_prev_vals = tl.load(hz + hz_prev_indices, mask=j_mask)
                
                # Compute and store
                result = ey_vals - 0.5 * (hz_vals - hz_prev_vals)
                tl.store(ey + ey_indices, result, mask=j_mask)
        
        # Phase 3: ex[i][j] = ex[i][j] - 0.5*(hz[i][j]-hz[i][j-1]) for j >= 1
        for i in range(NX):
            for j_start in range(1, NY, BLOCK_SIZE):
                j_offsets = j_start + offsets
                j_mask = j_offsets < NY
                
                # Linear indices
                ex_indices = i * NY + j_offsets
                hz_indices = i * NY + j_offsets
                hz_prev_indices = i * NY + (j_offsets - 1)
                
                # Load values
                ex_vals = tl.load(ex + ex_indices, mask=j_mask)
                hz_vals = tl.load(hz + hz_indices, mask=j_mask)
                hz_prev_vals = tl.load(hz + hz_prev_indices, mask=j_mask)
                
                # Compute and store
                result = ex_vals - 0.5 * (hz_vals - hz_prev_vals)
                tl.store(ex + ex_indices, result, mask=j_mask)
        
        # Phase 4: hz update for i < NX-1, j < NY-1
        for i in range(NX - 1):
            for j_start in range(0, NY - 1, BLOCK_SIZE):
                j_offsets = j_start + offsets
                j_mask = j_offsets < (NY - 1)
                
                # Linear indices for hz[i][j]
                hz_indices = i * NY + j_offsets
                
                # Linear indices for ex[i][j+1] and ex[i][j]
                ex_next_indices = i * NY + (j_offsets + 1)
                ex_curr_indices = i * NY + j_offsets
                
                # Linear indices for ey[i+1][j] and ey[i][j]
                ey_next_indices = (i + 1) * NY + j_offsets
                ey_curr_indices = i * NY + j_offsets
                
                # Load values
                hz_vals = tl.load(hz + hz_indices, mask=j_mask)
                ex_next_vals = tl.load(ex + ex_next_indices, mask=j_mask)
                ex_curr_vals = tl.load(ex + ex_curr_indices, mask=j_mask)
                ey_next_vals = tl.load(ey + ey_next_indices, mask=j_mask)
                ey_curr_vals = tl.load(ey + ey_curr_indices, mask=j_mask)
                
                # Compute and store
                result = hz_vals - 0.7 * ((ex_next_vals - ex_curr_vals) + (ey_next_vals - ey_curr_vals))
                tl.store(hz + hz_indices, result, mask=j_mask)

def fdtd_2d_triton(_fict_, ex, ey, hz, NX, NY, TMAX):
    BLOCK_SIZE = 128
    
    # Single kernel launch with t loop inside
    fdtd_2d_kernel[1,](_fict_, ex, ey, hz, NX, NY, TMAX, BLOCK_SIZE)