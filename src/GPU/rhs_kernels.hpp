#ifndef SHUD_GPU_RHS_KERNELS_HPP
#define SHUD_GPU_RHS_KERNELS_HPP

#ifdef _CUDA_ON

#include <cuda_runtime_api.h>
#include <sundials/sundials_types.h>

struct DeviceModel;

/* Placeholder kernel pipeline for CUDA RHS evaluation.
 *
 * NOTE: This is intentionally minimal for Issue #12 (E4-1). The first
 * implementation simply writes `ydot = 0` on the GPU to validate the end-to-end
 * NVECTOR_CUDA callback path without touching host pointers.
 */
void launch_rhs_kernels(realtype t,
                        const realtype *dY,
                        realtype *dYdot,
                        const DeviceModel *d_model,
                        cudaStream_t stream);

#endif /* _CUDA_ON */

#endif /* SHUD_GPU_RHS_KERNELS_HPP */
