import torch
import triton
import triton.language as tl

@triton.jit
def lavaMD_kernel(
    box_nei_ptr, box_nn_ptr, box_offset_ptr,
    fv_v_ptr, fv_x_ptr, fv_y_ptr, fv_z_ptr,
    qv_ptr, rv_v_ptr, rv_x_ptr, rv_y_ptr, rv_z_ptr,
    alpha_val,
    NUMBER_BOXES: tl.constexpr,
    MAX_NEIGHBORS: tl.constexpr, 
    NUMBER_PAR_PER_BOX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    i_indices = block_start + offsets
    mask = i_indices < NUMBER_PAR_PER_BOX
    
    a2 = 2.0 * alpha_val * alpha_val
    
    for l in range(NUMBER_BOXES):
        first_i = tl.load(box_offset_ptr + l)
        nn = tl.load(box_nn_ptr + l)
        
        for k in range(1 + nn):
            if k == 0:
                pointer = l
            else:
                pointer = tl.load(box_nei_ptr + l * MAX_NEIGHBORS + (k - 1))
            
            first_j = tl.load(box_offset_ptr + pointer)
            
            # Load particle data for current block of i particles
            i_abs = first_i + i_indices
            rv_v_i = tl.load(rv_v_ptr + i_abs, mask=mask)
            rv_x_i = tl.load(rv_x_ptr + i_abs, mask=mask)
            rv_y_i = tl.load(rv_y_ptr + i_abs, mask=mask)
            rv_z_i = tl.load(rv_z_ptr + i_abs, mask=mask)
            
            # Initialize accumulators
            fv_v_acc = tl.zeros_like(rv_v_i)
            fv_x_acc = tl.zeros_like(rv_x_i)
            fv_y_acc = tl.zeros_like(rv_y_i)
            fv_z_acc = tl.zeros_like(rv_z_i)
            
            for j in range(NUMBER_PAR_PER_BOX):
                j_abs = first_j + j
                
                # Load j particle data (scalar)
                rv_v_j = tl.load(rv_v_ptr + j_abs)
                rv_x_j = tl.load(rv_x_ptr + j_abs)
                rv_y_j = tl.load(rv_y_ptr + j_abs)
                rv_z_j = tl.load(rv_z_ptr + j_abs)
                qv_j = tl.load(qv_ptr + j_abs)
                
                # Compute r2 (distance squared)
                r2 = rv_v_i + rv_v_j - (rv_x_i * rv_x_j + rv_y_i * rv_y_j + rv_z_i * rv_z_j)
                
                u2 = a2 * r2
                vij = tl.exp(-u2)
                fs = 2.0 * vij
                
                dx = rv_x_i - rv_x_j
                dy = rv_y_i - rv_y_j
                dz = rv_z_i - rv_z_j
                
                fxij = fs * dx
                fyij = fs * dy
                fzij = fs * dz
                
                # Accumulate forces
                fv_v_acc += qv_j * vij
                fv_x_acc += qv_j * fxij
                fv_y_acc += qv_j * fyij
                fv_z_acc += qv_j * fzij
            
            # Store accumulated forces
            current_fv_v = tl.load(fv_v_ptr + i_abs, mask=mask)
            current_fv_x = tl.load(fv_x_ptr + i_abs, mask=mask)
            current_fv_y = tl.load(fv_y_ptr + i_abs, mask=mask)
            current_fv_z = tl.load(fv_z_ptr + i_abs, mask=mask)
            
            tl.store(fv_v_ptr + i_abs, current_fv_v + fv_v_acc, mask=mask)
            tl.store(fv_x_ptr + i_abs, current_fv_x + fv_x_acc, mask=mask)
            tl.store(fv_y_ptr + i_abs, current_fv_y + fv_y_acc, mask=mask)
            tl.store(fv_z_ptr + i_abs, current_fv_z + fv_z_acc, mask=mask)

def lavaMD_triton(box_nei, box_nn, box_offset, fv_v, fv_x, fv_y, fv_z, qv, rv_v, rv_x, rv_y, rv_z, alpha_val, BOXES1D, MAX_NEIGHBORS, NUMBER_PAR_PER_BOX):
    NUMBER_BOXES = BOXES1D * BOXES1D * BOXES1D
    BLOCK_SIZE = 32
    grid = (triton.cdiv(NUMBER_PAR_PER_BOX, BLOCK_SIZE),)
    
    lavaMD_kernel[grid](
        box_nei, box_nn, box_offset,
        fv_v, fv_x, fv_y, fv_z,
        qv, rv_v, rv_x, rv_y, rv_z,
        alpha_val,
        NUMBER_BOXES, MAX_NEIGHBORS, NUMBER_PAR_PER_BOX, BLOCK_SIZE
    )