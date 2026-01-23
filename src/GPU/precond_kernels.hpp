#ifndef SHUD_GPU_PRECOND_KERNELS_HPP
#define SHUD_GPU_PRECOND_KERNELS_HPP

#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>

#ifdef _CUDA_ON
int PSetup_cuda(realtype t,
                N_Vector y,
                N_Vector fy,
                booleantype jok,
                booleantype *jcurPtr,
                realtype gamma,
                void *user_data);

int PSolve_cuda(realtype t,
                N_Vector y,
                N_Vector fy,
                N_Vector r,
                N_Vector z,
                realtype gamma,
                realtype delta,
                int lr,
                void *user_data);
#else
static inline int PSetup_cuda(realtype t,
                              N_Vector y,
                              N_Vector fy,
                              booleantype jok,
                              booleantype *jcurPtr,
                              realtype gamma,
                              void *user_data)
{
    (void)t;
    (void)y;
    (void)fy;
    (void)jok;
    (void)jcurPtr;
    (void)gamma;
    (void)user_data;
    return -1;
}

static inline int PSolve_cuda(realtype t,
                              N_Vector y,
                              N_Vector fy,
                              N_Vector r,
                              N_Vector z,
                              realtype gamma,
                              realtype delta,
                              int lr,
                              void *user_data)
{
    (void)t;
    (void)y;
    (void)fy;
    (void)r;
    (void)z;
    (void)gamma;
    (void)delta;
    (void)lr;
    (void)user_data;
    return -1;
}
#endif /* _CUDA_ON */

#endif /* SHUD_GPU_PRECOND_KERNELS_HPP */

