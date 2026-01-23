#include "rhs_kernels.hpp"

#ifdef _CUDA_ON

#include "DeviceContext.hpp"

#include <cuda_runtime.h>

namespace {

__global__ void rhs_zero_kernel(realtype *dYdot, const DeviceModel *d_model)
{
    if (dYdot == nullptr || d_model == nullptr) {
        return;
    }
    const int n = 3 * d_model->NumEle + d_model->NumRiv + d_model->NumLake;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        dYdot[idx] = (realtype)0;
    }
}

} // namespace

void launch_rhs_kernels(realtype t,
                        const realtype *dY,
                        realtype *dYdot,
                        const DeviceModel *d_model,
                        cudaStream_t stream)
{
    (void)t;
    (void)dY;
    constexpr int kBlockSize = 256;
    constexpr int kNumBlocks = 256;
    rhs_zero_kernel<<<kNumBlocks, kBlockSize, 0, stream>>>(dYdot, d_model);
}

#endif /* _CUDA_ON */
