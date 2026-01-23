#include "Model_Data.hpp"

#ifdef _CUDA_ON

#include "rhs_kernels.hpp"

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

    /* Use the model-owned stream (also used by forcing H2D copies). */
    cudaStream_t stream = md->cuda_stream;
    launch_rhs_kernels((realtype)t, dY, dYdot, md->d_model, stream);

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
