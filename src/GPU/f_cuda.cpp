#include "Model_Data.hpp"

#ifdef _CUDA_ON

#include "rhs_kernels.hpp"
#include "Nvtx.hpp"

#include <cuda_runtime_api.h>
#include <cstdio>

#ifdef DEBUG_GPU_VERIFY
#include "gpu_verify.hpp"

#include <vector>
#endif

int f_gpu(double t, N_Vector y, N_Vector ydot, void *user_data)
{
    shud_nvtx::scoped_range range("f_cuda");
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
        shud_nvtx::scoped_range sync_range("f_cuda/sync_y_stream");
        const cudaError_t err = cudaStreamSynchronize(y_stream);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "CUDA_ERROR: f_gpu: cudaStreamSynchronize(y_stream) failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
    }
    if (md->cuda_stream != rhs_stream && md->forcing_copy_event != nullptr && md->nGpuForcingCopy > 0) {
        shud_nvtx::scoped_range wait_range("f_cuda/wait_forcing_event");
        const cudaError_t err = cudaStreamWaitEvent(rhs_stream, md->forcing_copy_event, 0);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "CUDA_ERROR: f_gpu: cudaStreamWaitEvent(forcing_copy_event) failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
    }

#ifdef DEBUG_GPU_VERIFY
    const GpuVerifySettings verify_settings = gpuVerifySettingsFromEnv();
    const unsigned long step = md->nFCall;
    const bool do_verify = verify_settings.enabled && (verify_settings.interval > 0) && (step % verify_settings.interval == 0);

    std::vector<double> cpu_Y;
    std::vector<double> cpu_DY;
    std::vector<double> cpu_QeleSurf_flat;
    std::vector<double> cpu_QeleSub_flat;
    std::vector<double> cpu_ele_satn;
    std::vector<double> cpu_ele_effKH;
    std::vector<double> cpu_riv_topWidth;
    std::vector<double> cpu_riv_CSarea;
    std::vector<double> cpu_riv_CSperem;
    GpuVerifyContext verify_ctx{};
    const GpuVerifyContext *verify_ptr = nullptr;

    if (do_verify) {
        md->gpuSyncStateFromDevice(y);
        const realtype *hY = N_VGetHostArrayPointer_Cuda(y);
        if (hY == nullptr) {
            fprintf(stderr, "ERROR: f_gpu: N_VGetHostArrayPointer_Cuda returned NULL in DEBUG_GPU_VERIFY.\n");
            return -1;
        }

        const int nY = md->NumY;
        cpu_Y.resize(static_cast<size_t>(nY));
        cpu_DY.assign(static_cast<size_t>(nY), 0.0);
        for (int i = 0; i < nY; i++) {
            cpu_Y[static_cast<size_t>(i)] = static_cast<double>(hY[i]);
        }

        md->f_update(cpu_Y.data(), cpu_DY.data(), t);
        md->f_loop(t);
        md->f_applyDY(cpu_DY.data(), t);

        const int nEle = md->NumEle;
        cpu_QeleSurf_flat.resize(static_cast<size_t>(nEle) * 3);
        cpu_QeleSub_flat.resize(static_cast<size_t>(nEle) * 3);
        for (int i = 0; i < nEle; i++) {
            for (int j = 0; j < 3; j++) {
                cpu_QeleSurf_flat[static_cast<size_t>(i) * 3 + static_cast<size_t>(j)] = md->QeleSurf[i][j];
                cpu_QeleSub_flat[static_cast<size_t>(i) * 3 + static_cast<size_t>(j)] = md->QeleSub[i][j];
            }
        }

        cpu_ele_satn.resize(static_cast<size_t>(nEle));
        cpu_ele_effKH.resize(static_cast<size_t>(nEle));
        for (int i = 0; i < nEle; i++) {
            cpu_ele_satn[static_cast<size_t>(i)] = md->Ele[i].u_satn;
            cpu_ele_effKH[static_cast<size_t>(i)] = md->Ele[i].u_effKH;
        }

        const int nRiv = md->NumRiv;
        cpu_riv_topWidth.resize(static_cast<size_t>(nRiv));
        cpu_riv_CSarea.resize(static_cast<size_t>(nRiv));
        cpu_riv_CSperem.resize(static_cast<size_t>(nRiv));
        for (int i = 0; i < nRiv; i++) {
            cpu_riv_topWidth[static_cast<size_t>(i)] = md->Riv[i].u_topWidth;
            cpu_riv_CSarea[static_cast<size_t>(i)] = md->Riv[i].u_CSarea;
            cpu_riv_CSperem[static_cast<size_t>(i)] = md->Riv[i].u_CSperem;
        }

        verify_ctx.step = step;
        verify_ctx.t = t;
        verify_ctx.NumEle = md->NumEle;
        verify_ctx.NumRiv = md->NumRiv;
        verify_ctx.NumLake = md->NumLake;
        verify_ctx.NumY = md->NumY;
        verify_ctx.settings = verify_settings;

        verify_ctx.cpu_uYsf = uYsf;
        verify_ctx.cpu_uYus = uYus;
        verify_ctx.cpu_uYgw = uYgw;
        verify_ctx.cpu_uYriv = uYriv;
        verify_ctx.cpu_yLakeStg = md->yLakeStg;

        verify_ctx.cpu_ele_satn = cpu_ele_satn.data();
        verify_ctx.cpu_ele_effKH = cpu_ele_effKH.data();

        verify_ctx.cpu_qEleInfil = md->qEleInfil;
        verify_ctx.cpu_qEleExfil = md->qEleExfil;
        verify_ctx.cpu_qEleRecharge = md->qEleRecharge;
        verify_ctx.cpu_qEs = md->qEs;
        verify_ctx.cpu_qEu = md->qEu;
        verify_ctx.cpu_qEg = md->qEg;
        verify_ctx.cpu_qTu = md->qTu;
        verify_ctx.cpu_qTg = md->qTg;

        verify_ctx.cpu_QeleSurf = cpu_QeleSurf_flat.data();
        verify_ctx.cpu_QeleSub = cpu_QeleSub_flat.data();

        verify_ctx.cpu_Qe2r_Surf = md->Qe2r_Surf;
        verify_ctx.cpu_Qe2r_Sub = md->Qe2r_Sub;

        verify_ctx.cpu_QrivSurf = md->QrivSurf;
        verify_ctx.cpu_QrivSub = md->QrivSub;
        verify_ctx.cpu_QrivUp = md->QrivUp;
        verify_ctx.cpu_QrivDown = md->QrivDown;
        verify_ctx.cpu_riv_topWidth = cpu_riv_topWidth.data();
        verify_ctx.cpu_riv_CSarea = cpu_riv_CSarea.data();
        verify_ctx.cpu_riv_CSperem = cpu_riv_CSperem.data();

        verify_ctx.cpu_QLakeSurf = md->QLakeSurf;
        verify_ctx.cpu_QLakeSub = md->QLakeSub;
        verify_ctx.cpu_QLakeRivIn = md->QLakeRivIn;
        verify_ctx.cpu_QLakeRivOut = md->QLakeRivOut;
        verify_ctx.cpu_qLakePrcp = md->qLakePrcp;
        verify_ctx.cpu_qLakeEvap = md->qLakeEvap;
        verify_ctx.cpu_y2LakeArea = md->y2LakeArea;

        verify_ctx.cpu_dYdot = cpu_DY.data();

        verify_ptr = &verify_ctx;
    }

    launch_rhs_kernels((realtype)t, dY, dYdot, md->d_model, md->h_model, rhs_stream, verify_ptr);
#else
    launch_rhs_kernels((realtype)t, dY, dYdot, md->d_model, md->h_model, rhs_stream);
#endif
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
