#include "Model_Data.hpp"

#ifdef _CUDA_ON

#include "rhs_kernels.hpp"

#include <cuda_runtime_api.h>
#include <cstdio>

int f_gpu(double t, N_Vector y, N_Vector ydot, void *user_data)
{
    Model_Data *md = static_cast<Model_Data *>(user_data);
    if (md == nullptr) {
        return -1;
    }

    if (N_VGetVectorID(y) != SUNDIALS_NVEC_CUDA || N_VGetVectorID(ydot) != SUNDIALS_NVEC_CUDA) {
        fprintf(stderr, "ERROR: f_gpu requires NVECTOR_CUDA for y and ydot.\n");
        return -1;
    }

    if (md->d_model == nullptr) {
        fprintf(stderr, "ERROR: f_gpu called before gpuInit().\n");
        return -1;
    }

    realtype *dY = N_VGetDeviceArrayPointer_Cuda(y);
    realtype *dYdot = N_VGetDeviceArrayPointer_Cuda(ydot);
    if (dY == nullptr || dYdot == nullptr) {
        fprintf(stderr, "ERROR: f_gpu: N_VGetDeviceArrayPointer_Cuda returned NULL.\n");
        return -1;
    }

    /*
     * Stream strategy:
     * - Launch RHS kernels on the same stream associated with the NVECTOR_CUDA
     *   vectors so that CVODE/NVector GPU ops and our RHS evaluation are
     *   correctly ordered without per-call synchronizations.
     * - Model forcing H2D copies run on md->cuda_stream; if the RHS stream
     *   differs, wait on md->forcing_copy_event to ensure forcing data is ready.
     */
    const cudaStream_t rhs_stream = N_VGetCudaStream_Cuda(ydot);
    const cudaStream_t y_stream = N_VGetCudaStream_Cuda(y);
    if (y_stream != rhs_stream) {
        const cudaError_t err = cudaStreamSynchronize(y_stream);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "CUDA_ERROR: f_gpu: cudaStreamSynchronize(y_stream) failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
    }
    if (md->cuda_stream != rhs_stream && md->forcing_copy_event != nullptr && md->nGpuForcingCopy > 0) {
        const cudaError_t err = cudaStreamWaitEvent(rhs_stream, md->forcing_copy_event, 0);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "CUDA_ERROR: f_gpu: cudaStreamWaitEvent(forcing_copy_event) failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
    }

    launch_rhs_kernels((realtype)t, dY, dYdot, md->d_model, rhs_stream);
    {
        const cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA_ERROR: f_gpu: kernel launch failed: %s\n", cudaGetErrorString(err));
            return -1;
        }
    }

    {
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA_ERROR: f_gpu: CUDA error before return: %s\n", cudaGetErrorString(err));
            return -1;
        }
    }

    return 0;
}

int f_cuda(double t, N_Vector y, N_Vector ydot, void *user_data)
{
    return f_gpu(t, y, ydot, user_data);
}

#else

int f_gpu(double t, N_Vector y, N_Vector ydot, void *user_data)
{
    (void)t;
    (void)y;
    (void)ydot;
    (void)user_data;
    return -1;
}

int f_cuda(double t, N_Vector y, N_Vector ydot, void *user_data)
{
    (void)t;
    (void)y;
    (void)ydot;
    (void)user_data;
    return -1;
}

#endif /* _CUDA_ON */
