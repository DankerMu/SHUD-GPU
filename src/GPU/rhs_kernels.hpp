#ifndef SHUD_GPU_RHS_KERNELS_HPP
#define SHUD_GPU_RHS_KERNELS_HPP

#ifdef _CUDA_ON

#include <cuda_runtime_api.h>
#include <sundials/sundials_types.h>

struct DeviceModel;

#ifdef DEBUG_GPU_VERIFY
struct GpuVerifyContext;
#endif

void launch_rhs_kernels(realtype t,
                        const realtype *dY,
                        realtype *dYdot,
                        const DeviceModel *d_model,
                        const DeviceModel *h_model,
                        cudaStream_t stream
#ifdef DEBUG_GPU_VERIFY
                        ,
                        const GpuVerifyContext *verify
#endif
);

#endif /* _CUDA_ON */

#endif /* SHUD_GPU_RHS_KERNELS_HPP */
