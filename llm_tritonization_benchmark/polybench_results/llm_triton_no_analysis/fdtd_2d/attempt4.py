import triton
import triton.language as tl
import torch

@triton.jit
def fdtd_2d_kernel_step1(_fict_ptr, ey_ptr, t: tl.constexpr, NY: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    j_offsets = block_start + offsets
    mask = j_offsets < NY
    
    fict_val = tl.load(_fict_ptr + t)
    ey_idx = j_offsets  # row 0, so index is just j
    tl.store(ey_ptr + ey_idx, fict_val, mask=mask)

@triton.jit
def fdtd_2d_kernel_step2(ey_ptr, hz_ptr, i: tl.constexpr, NY: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    j_offsets = block_start + offsets
    mask = j_offsets < NY
    
    ey_idx = i * NY + j_offsets
    hz_idx = i * NY + j_offsets
    hz_prev_idx = (i - 1) * NY + j_offsets
    
    ey_val = tl.load(ey_ptr + ey_idx, mask=mask)
    hz_val = tl.load(hz_ptr + hz_idx, mask=mask)
    hz_prev_val = tl.load(hz_ptr + hz_prev_idx, mask=mask)
    
    new_ey = ey_val - 0.5 * (hz_val - hz_prev_val)
    tl.store(ey_ptr + ey_idx, new_ey, mask=mask)

@triton.jit
def fdtd_2d_kernel_step3(ex_ptr, hz_ptr, i: tl.constexpr, NY: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    j_offsets = block_start + offsets + 1  # j starts from 1
    mask = j_offsets < NY
    
    ex_idx = i * NY + j_offsets
    hz_idx = i * NY + j_offsets
    hz_prev_idx = i * NY + (j_offsets - 1)
    
    ex_val = tl.load(ex_ptr + ex_idx, mask=mask)
    hz_val = tl.load(hz_ptr + hz_idx, mask=mask)
    hz_prev_val = tl.load(hz_ptr + hz_prev_idx, mask=mask)
    
    new_ex = ex_val - 0.5 * (hz_val - hz_prev_val)
    tl.store(ex_ptr + ex_idx, new_ex, mask=mask)

@triton.jit
def fdtd_2d_kernel_step4(ex_ptr, ey_ptr, hz_ptr, i: tl.constexpr, NY: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    j_offsets = block_start + offsets
    mask = j_offsets < (NY - 1)
    
    hz_idx = i * NY + j_offsets
    ex_next_idx = i * NY + (j_offsets + 1)
    ex_curr_idx = i * NY + j_offsets
    ey_next_idx = (i + 1) * NY + j_offsets
    ey_curr_idx = i * NY + j_offsets
    
    hz_val = tl.load(hz_ptr + hz_idx, mask=mask)
    ex_next = tl.load(ex_ptr + ex_next_idx, mask=mask)
    ex_curr = tl.load(ex_ptr + ex_curr_idx, mask=mask)
    ey_next = tl.load(ey_ptr + ey_next_idx, mask=mask)
    ey_curr = tl.load(ey_ptr + ey_curr_idx, mask=mask)
    
    new_hz = hz_val - 0.7 * ((ex_next - ex_curr) + (ey_next - ey_curr))
    tl.store(hz_ptr + hz_idx, new_hz, mask=mask)

def fdtd_2d_triton(_fict_, ex, ey, hz, NX, NY, TMAX):
    BLOCK_SIZE = 128
    
    for t in range(TMAX):
        # Step 1: Set boundary condition ey[0][j] = _fict_[t]
        grid1 = (triton.cdiv(NY, BLOCK_SIZE),)
        fdtd_2d_kernel_step1[grid1](_fict_, ey, t, NY, BLOCK_SIZE)
        
        # Step 2: Update ey[i][j] for i=1 to NX-1
        grid2 = (triton.cdiv(NY, BLOCK_SIZE),)
        for i in range(1, NX):
            fdtd_2d_kernel_step2[grid2](ey, hz, i, NY, BLOCK_SIZE)
        
        # Step 3: Update ex[i][j] for i=0 to NX-1, j=1 to NY-1
        grid3 = (triton.cdiv(NY - 1, BLOCK_SIZE),)
        for i in range(NX):
            fdtd_2d_kernel_step3[grid3](ex, hz, i, NY, BLOCK_SIZE)
        
        # Step 4: Update hz[i][j] for i=0 to NX-2, j=0 to NY-2
        grid4 = (triton.cdiv(NY - 1, BLOCK_SIZE),)
        for i in range(NX - 1):
            fdtd_2d_kernel_step4[grid4](ex, ey, hz, i, NY, BLOCK_SIZE)