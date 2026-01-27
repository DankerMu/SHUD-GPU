#include "DeviceContext.hpp"

#ifdef _CUDA_ON

#include "Model_Data.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

inline void cudaDie(cudaError_t err, const char *what)
{
    if (err == cudaSuccess) {
        return;
    }
    fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(err));
    std::abort();
}

template <typename T>
cudaError_t cudaAllocAndUpload(T **d_out, const T *h_in, size_t count)
{
    *d_out = nullptr;
    if (count == 0) {
        return cudaSuccess;
    }
    cudaError_t err = cudaMalloc(reinterpret_cast<void **>(d_out), sizeof(T) * count);
    if (err != cudaSuccess) {
        *d_out = nullptr;
        return err;
    }
    if (h_in != nullptr) {
        err = cudaMemcpy(*d_out, h_in, sizeof(T) * count, cudaMemcpyHostToDevice);
    } else {
        err = cudaMemset(*d_out, 0, sizeof(T) * count);
    }
    if (err != cudaSuccess) {
        cudaFree(*d_out);
        *d_out = nullptr;
    }
    return err;
}

void cudaFreeIfNotNull(void *p)
{
    if (p == nullptr) {
        return;
    }
    (void)cudaFree(p);
}

void tryHostRegister(Model_Data *md, void *ptr, size_t bytes, const char *what)
{
    if (md == nullptr || ptr == nullptr || bytes == 0) {
        return;
    }
    const cudaError_t err = cudaHostRegister(ptr, bytes, cudaHostRegisterDefault);
    if (err == cudaSuccess) {
        md->pinned_host_buffers.push_back(ptr);
        return;
    }
    if (err == cudaErrorHostMemoryAlreadyRegistered) {
        return;
    }
    fprintf(stderr, "gpuInit: failed to pin host buffer for %s: %s (continuing)\n", what, cudaGetErrorString(err));
}

void unregisterPinnedHostBuffers(Model_Data *md)
{
    if (md == nullptr) {
        return;
    }
    for (void *ptr : md->pinned_host_buffers) {
        if (ptr == nullptr) {
            continue;
        }
        const cudaError_t err = cudaHostUnregister(ptr);
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuFree: failed to unpin host buffer: %s (continuing)\n", cudaGetErrorString(err));
        }
    }
    md->pinned_host_buffers.clear();
}

void freeDeviceBuffers(DeviceModel &h)
{
    cudaFreeIfNotNull(h.ele_area);
    cudaFreeIfNotNull(h.ele_z_surf);
    cudaFreeIfNotNull(h.ele_z_bottom);
    cudaFreeIfNotNull(h.ele_depression);
    cudaFreeIfNotNull(h.ele_Rough);

    cudaFreeIfNotNull(h.ele_nabr);
    cudaFreeIfNotNull(h.ele_lakenabr);
    cudaFreeIfNotNull(h.ele_nabrToMe);
    cudaFreeIfNotNull(h.ele_edge);
    cudaFreeIfNotNull(h.ele_Dist2Nabor);
    cudaFreeIfNotNull(h.ele_Dist2Edge);
    cudaFreeIfNotNull(h.ele_avgRough);

    cudaFreeIfNotNull(h.ele_AquiferDepth);
    cudaFreeIfNotNull(h.ele_Sy);
    cudaFreeIfNotNull(h.ele_infD);
    cudaFreeIfNotNull(h.ele_infKsatV);
    cudaFreeIfNotNull(h.ele_ThetaS);
    cudaFreeIfNotNull(h.ele_ThetaR);
    cudaFreeIfNotNull(h.ele_ThetaFC);
    cudaFreeIfNotNull(h.ele_Alpha);
    cudaFreeIfNotNull(h.ele_Beta);
    cudaFreeIfNotNull(h.ele_hAreaF);
    cudaFreeIfNotNull(h.ele_macKsatV);
    cudaFreeIfNotNull(h.ele_KsatH);
    cudaFreeIfNotNull(h.ele_KsatV);
    cudaFreeIfNotNull(h.ele_geo_vAreaF);
    cudaFreeIfNotNull(h.ele_macKsatH);
    cudaFreeIfNotNull(h.ele_macD);
    cudaFreeIfNotNull(h.ele_RzD);
    cudaFreeIfNotNull(h.ele_VegFrac);
    cudaFreeIfNotNull(h.ele_ImpAF);
    cudaFreeIfNotNull(h.ele_iLake);
    cudaFreeIfNotNull(h.ele_iBC);
    cudaFreeIfNotNull(h.ele_iSS);
    cudaFreeIfNotNull(h.ele_yBC);
    cudaFreeIfNotNull(h.ele_QBC);
    cudaFreeIfNotNull(h.ele_QSS);

    cudaFreeIfNotNull(h.riv_down_raw);
    cudaFreeIfNotNull(h.riv_toLake);
    cudaFreeIfNotNull(h.riv_BC);
    cudaFreeIfNotNull(h.riv_Length);
    cudaFreeIfNotNull(h.riv_depth);
    cudaFreeIfNotNull(h.riv_BankSlope);
    cudaFreeIfNotNull(h.riv_BottomWidth);
    cudaFreeIfNotNull(h.riv_BedSlope);
    cudaFreeIfNotNull(h.riv_rivRough);
    cudaFreeIfNotNull(h.riv_avgRough);
    cudaFreeIfNotNull(h.riv_Dist2DownStream);
    cudaFreeIfNotNull(h.riv_KsatH);
    cudaFreeIfNotNull(h.riv_BedThick);
    cudaFreeIfNotNull(h.riv_yBC);
    cudaFreeIfNotNull(h.riv_qBC);
    cudaFreeIfNotNull(h.riv_zbed);
    cudaFreeIfNotNull(h.riv_zbank);

    cudaFreeIfNotNull(h.seg_iEle);
    cudaFreeIfNotNull(h.seg_iRiv);
    cudaFreeIfNotNull(h.seg_length);
    cudaFreeIfNotNull(h.seg_Cwr);
    cudaFreeIfNotNull(h.seg_KsatH);
    cudaFreeIfNotNull(h.seg_eqDistance);

    cudaFreeIfNotNull(h.lake_zmin);
    cudaFreeIfNotNull(h.lake_invNumEle);
    cudaFreeIfNotNull(h.lake_bathy_off);
    cudaFreeIfNotNull(h.lake_bathy_n);
    cudaFreeIfNotNull(h.bathy_yi);
    cudaFreeIfNotNull(h.bathy_ai);

    cudaFreeIfNotNull(h.qElePrep);
    cudaFreeIfNotNull(h.qEleNetPrep);
    cudaFreeIfNotNull(h.qPotEvap);
    cudaFreeIfNotNull(h.qPotTran);
    cudaFreeIfNotNull(h.qEleE_IC);
    cudaFreeIfNotNull(h.t_lai);
    cudaFreeIfNotNull(h.fu_Surf);
    cudaFreeIfNotNull(h.fu_Sub);

    cudaFreeIfNotNull(h.uYsf);
    cudaFreeIfNotNull(h.uYus);
    cudaFreeIfNotNull(h.uYgw);
    cudaFreeIfNotNull(h.uYriv);
    cudaFreeIfNotNull(h.uYlake);
    cudaFreeIfNotNull(h.ele_satn);
    cudaFreeIfNotNull(h.ele_effKH);

    cudaFreeIfNotNull(h.qEleInfil);
    cudaFreeIfNotNull(h.qEleExfil);
    cudaFreeIfNotNull(h.qEleRecharge);
    cudaFreeIfNotNull(h.qEs);
    cudaFreeIfNotNull(h.qEu);
    cudaFreeIfNotNull(h.qEg);
    cudaFreeIfNotNull(h.qTu);
    cudaFreeIfNotNull(h.qTg);
    cudaFreeIfNotNull(h.qEleTrans);
    cudaFreeIfNotNull(h.qEleEvapo);
    cudaFreeIfNotNull(h.qEleETA);

    cudaFreeIfNotNull(h.QeleSurf);
    cudaFreeIfNotNull(h.QeleSub);
    cudaFreeIfNotNull(h.QeleSurfTot);
    cudaFreeIfNotNull(h.QeleSubTot);

    cudaFreeIfNotNull(h.QrivSurf);
    cudaFreeIfNotNull(h.QrivSub);
    cudaFreeIfNotNull(h.QrivUp);
    cudaFreeIfNotNull(h.QrivDown);
    cudaFreeIfNotNull(h.Qe2r_Surf);
    cudaFreeIfNotNull(h.Qe2r_Sub);

    cudaFreeIfNotNull(h.QLakeSurf);
    cudaFreeIfNotNull(h.QLakeSub);
    cudaFreeIfNotNull(h.QLakeRivIn);
    cudaFreeIfNotNull(h.QLakeRivOut);
    cudaFreeIfNotNull(h.qLakePrcp);
    cudaFreeIfNotNull(h.qLakeEvap);
    cudaFreeIfNotNull(h.y2LakeArea);

    cudaFreeIfNotNull(h.riv_CSarea);
    cudaFreeIfNotNull(h.riv_CSperem);
    cudaFreeIfNotNull(h.riv_topWidth);
    cudaFreeIfNotNull(h.prec_inv);

    h = DeviceModel{};
}

} // namespace

void gpuInit(Model_Data *md)
{
    if (md == nullptr) {
        return;
    }

    if (md->d_model != nullptr) {
        gpuFree(md);
    }

    cudaStream_t stream = nullptr;
    cudaEvent_t forcing_event = nullptr;
    cudaEvent_t precond_event = nullptr;
    DeviceModel h{};
    h.NumEle = md->NumEle;
    h.NumRiv = md->NumRiv;
    h.NumSeg = md->NumSegmt;
    h.NumLake = md->NumLake;
    h.CloseBoundary = md->CS.CloseBoundary;

    cudaError_t err = cudaSuccess;

    err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (err != cudaSuccess) {
        stream = nullptr;
        fprintf(stderr, "gpuInit: failed to create CUDA stream\n");
        delete md->h_model;
        md->h_model = nullptr;
        cudaDie(err, "gpuInit(cudaStreamCreateWithFlags)");
        return;
    }
    err = cudaEventCreateWithFlags(&forcing_event, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        forcing_event = nullptr;
        fprintf(stderr, "gpuInit: failed to create CUDA event\n");
        if (stream != nullptr) {
            (void)cudaStreamDestroy(stream);
        }
        delete md->h_model;
        md->h_model = nullptr;
        cudaDie(err, "gpuInit(cudaEventCreateWithFlags forcing_event)");
        return;
    }
    err = cudaEventCreateWithFlags(&precond_event, cudaEventDisableTiming);
    if (err != cudaSuccess) {
        precond_event = nullptr;
        fprintf(stderr, "gpuInit: failed to create CUDA event for preconditioner\n");
        if (forcing_event != nullptr) {
            (void)cudaEventDestroy(forcing_event);
        }
        if (stream != nullptr) {
            (void)cudaStreamDestroy(stream);
        }
        delete md->h_model;
        md->h_model = nullptr;
        cudaDie(err, "gpuInit(cudaEventCreateWithFlags precond_event)");
        return;
    }
    md->cuda_stream = stream;
    md->forcing_copy_event = forcing_event;
    md->precond_setup_event = precond_event;
    md->nGpuForcingCopy = 0;
    md->nGpuPrecSetup = 0;
    md->nGpuRhsCalls = 0;
    md->nGpuRhsKernelNodes = 0;
    md->nGpuRhsLaunchCalls = 0;
    md->gpuRhsLaunchCpu_s = 0.0;
    md->rhs_graph_failed = 0;
    md->rhs_graph_kernel_nodes = 0;
    md->rhs_graph_clamp_policy = 0;
    md->rhs_graph_dY = nullptr;
    md->rhs_graph_dYdot = nullptr;
    if (md->rhs_graph_exec != nullptr) {
        (void)cudaGraphExecDestroy(md->rhs_graph_exec);
        md->rhs_graph_exec = nullptr;
    }
    if (md->rhs_graph != nullptr) {
        (void)cudaGraphDestroy(md->rhs_graph);
        md->rhs_graph = nullptr;
    }

    /* ------------------------------ Element static parameters ------------------------------ */
    std::vector<double> ele_area(h.NumEle);
    std::vector<double> ele_z_surf(h.NumEle);
    std::vector<double> ele_z_bottom(h.NumEle);
    std::vector<double> ele_depression(h.NumEle);
    std::vector<double> ele_Rough(h.NumEle);

    std::vector<int> ele_nabr(static_cast<size_t>(h.NumEle) * 3);
    std::vector<int> ele_lakenabr(static_cast<size_t>(h.NumEle) * 3);
    std::vector<int> ele_nabrToMe(static_cast<size_t>(h.NumEle) * 3);
    std::vector<double> ele_edge(static_cast<size_t>(h.NumEle) * 3);
    std::vector<double> ele_Dist2Nabor(static_cast<size_t>(h.NumEle) * 3);
    std::vector<double> ele_Dist2Edge(static_cast<size_t>(h.NumEle) * 3);
    std::vector<double> ele_avgRough(static_cast<size_t>(h.NumEle) * 3);

    std::vector<double> ele_AquiferDepth(h.NumEle);
    std::vector<double> ele_Sy(h.NumEle);
    std::vector<double> ele_infD(h.NumEle);
    std::vector<double> ele_infKsatV(h.NumEle);
    std::vector<double> ele_ThetaS(h.NumEle);
    std::vector<double> ele_ThetaR(h.NumEle);
    std::vector<double> ele_ThetaFC(h.NumEle);
    std::vector<double> ele_Alpha(h.NumEle);
    std::vector<double> ele_Beta(h.NumEle);
    std::vector<double> ele_hAreaF(h.NumEle);
    std::vector<double> ele_macKsatV(h.NumEle);
    std::vector<double> ele_KsatH(h.NumEle);
    std::vector<double> ele_KsatV(h.NumEle);
    std::vector<double> ele_geo_vAreaF(h.NumEle);
    std::vector<double> ele_macKsatH(h.NumEle);
    std::vector<double> ele_macD(h.NumEle);
    std::vector<double> ele_RzD(h.NumEle);
    std::vector<double> ele_VegFrac(h.NumEle);
    std::vector<double> ele_ImpAF(h.NumEle);
    std::vector<int> ele_iLake(h.NumEle);
    std::vector<int> ele_iBC(h.NumEle);
    std::vector<int> ele_iSS(h.NumEle);
    std::vector<double> ele_yBC(h.NumEle);
    std::vector<double> ele_QBC(h.NumEle);
    std::vector<double> ele_QSS(h.NumEle);

    for (int i = 0; i < h.NumEle; i++) {
        ele_area[i] = md->Ele[i].area;
        ele_z_surf[i] = md->Ele[i].z_surf;
        ele_z_bottom[i] = md->Ele[i].z_bottom;
        ele_depression[i] = md->Ele[i].depression;
        ele_Rough[i] = md->Ele[i].Rough;

        ele_AquiferDepth[i] = md->Ele[i].AquiferDepth;
        ele_Sy[i] = md->Ele[i].Sy;
        ele_infD[i] = md->Ele[i].infD;
        ele_infKsatV[i] = md->Ele[i].infKsatV;
        ele_ThetaS[i] = md->Ele[i].ThetaS;
        ele_ThetaR[i] = md->Ele[i].ThetaR;
        ele_ThetaFC[i] = md->Ele[i].ThetaFC;
        ele_Alpha[i] = md->Ele[i].Alpha;
        ele_Beta[i] = md->Ele[i].Beta;
        ele_hAreaF[i] = md->Ele[i].hAreaF;
        ele_macKsatV[i] = md->Ele[i].macKsatV;
        ele_KsatH[i] = md->Ele[i].KsatH;
        ele_KsatV[i] = md->Ele[i].KsatV;
        ele_geo_vAreaF[i] = md->Ele[i].geo_vAreaF;
        ele_macKsatH[i] = md->Ele[i].macKsatH;
        ele_macD[i] = md->Ele[i].macD;
        ele_RzD[i] = md->Ele[i].RzD;
        ele_VegFrac[i] = md->Ele[i].VegFrac;
        ele_ImpAF[i] = md->Ele[i].ImpAF;
        ele_iLake[i] = md->Ele[i].iLake;
        ele_iBC[i] = md->Ele[i].iBC;
        ele_iSS[i] = md->Ele[i].iSS;
        ele_yBC[i] = md->Ele[i].yBC;
        ele_QBC[i] = md->Ele[i].QBC;
        ele_QSS[i] = md->Ele[i].QSS;

        for (int j = 0; j < 3; j++) {
            const size_t idx = static_cast<size_t>(i) * 3 + j;
            ele_nabr[idx] = md->Ele[i].nabr[j];
            ele_lakenabr[idx] = md->Ele[i].lakenabr[j];
            ele_nabrToMe[idx] = md->Ele[i].nabrToMe[j];
            ele_edge[idx] = md->Ele[i].edge[j];
            ele_Dist2Nabor[idx] = md->Ele[i].Dist2Nabor[j];
            ele_Dist2Edge[idx] = md->Ele[i].Dist2Edge[j];
            ele_avgRough[idx] = md->Ele[i].avgRough[j];
        }
    }

    err = cudaAllocAndUpload(&h.ele_area, ele_area.data(), ele_area.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_area\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_z_surf, ele_z_surf.data(), ele_z_surf.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_z_surf\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_z_bottom, ele_z_bottom.data(), ele_z_bottom.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_z_bottom\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_depression, ele_depression.data(), ele_depression.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_depression\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_Rough, ele_Rough.data(), ele_Rough.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_Rough\n");
        goto fail;
    }

    err = cudaAllocAndUpload(&h.ele_nabr, ele_nabr.data(), ele_nabr.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_nabr\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_lakenabr, ele_lakenabr.data(), ele_lakenabr.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_lakenabr\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_nabrToMe, ele_nabrToMe.data(), ele_nabrToMe.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_nabrToMe\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_edge, ele_edge.data(), ele_edge.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_edge\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_Dist2Nabor, ele_Dist2Nabor.data(), ele_Dist2Nabor.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_Dist2Nabor\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_Dist2Edge, ele_Dist2Edge.data(), ele_Dist2Edge.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_Dist2Edge\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_avgRough, ele_avgRough.data(), ele_avgRough.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_avgRough\n");
        goto fail;
    }

    err = cudaAllocAndUpload(&h.ele_AquiferDepth, ele_AquiferDepth.data(), ele_AquiferDepth.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_AquiferDepth\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_Sy, ele_Sy.data(), ele_Sy.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_Sy\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_infD, ele_infD.data(), ele_infD.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_infD\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_infKsatV, ele_infKsatV.data(), ele_infKsatV.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_infKsatV\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_ThetaS, ele_ThetaS.data(), ele_ThetaS.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_ThetaS\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_ThetaR, ele_ThetaR.data(), ele_ThetaR.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_ThetaR\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_ThetaFC, ele_ThetaFC.data(), ele_ThetaFC.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_ThetaFC\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_Alpha, ele_Alpha.data(), ele_Alpha.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_Alpha\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_Beta, ele_Beta.data(), ele_Beta.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_Beta\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_hAreaF, ele_hAreaF.data(), ele_hAreaF.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_hAreaF\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_macKsatV, ele_macKsatV.data(), ele_macKsatV.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_macKsatV\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_KsatH, ele_KsatH.data(), ele_KsatH.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_KsatH\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_KsatV, ele_KsatV.data(), ele_KsatV.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_KsatV\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_geo_vAreaF, ele_geo_vAreaF.data(), ele_geo_vAreaF.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_geo_vAreaF\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_macKsatH, ele_macKsatH.data(), ele_macKsatH.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_macKsatH\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_macD, ele_macD.data(), ele_macD.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_macD\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_RzD, ele_RzD.data(), ele_RzD.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_RzD\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_VegFrac, ele_VegFrac.data(), ele_VegFrac.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_VegFrac\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_ImpAF, ele_ImpAF.data(), ele_ImpAF.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_ImpAF\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_iLake, ele_iLake.data(), ele_iLake.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_iLake\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_iBC, ele_iBC.data(), ele_iBC.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_iBC\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_iSS, ele_iSS.data(), ele_iSS.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_iSS\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_yBC, ele_yBC.data(), ele_yBC.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_yBC\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_QBC, ele_QBC.data(), ele_QBC.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_QBC\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_QSS, ele_QSS.data(), ele_QSS.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_QSS\n");
        goto fail;
    }

    /* ------------------------------ River static parameters ------------------------------ */
    if (h.NumRiv > 0) {
        std::vector<int> riv_down_raw(h.NumRiv);
        std::vector<int> riv_toLake(h.NumRiv);
        std::vector<int> riv_BC(h.NumRiv);
        std::vector<double> riv_Length(h.NumRiv);
        std::vector<double> riv_depth(h.NumRiv);
        std::vector<double> riv_BankSlope(h.NumRiv);
        std::vector<double> riv_BottomWidth(h.NumRiv);
        std::vector<double> riv_BedSlope(h.NumRiv);
        std::vector<double> riv_rivRough(h.NumRiv);
        std::vector<double> riv_avgRough(h.NumRiv);
        std::vector<double> riv_Dist2DownStream(h.NumRiv);
        std::vector<double> riv_KsatH(h.NumRiv);
        std::vector<double> riv_BedThick(h.NumRiv);
        std::vector<double> riv_yBC(h.NumRiv);
        std::vector<double> riv_qBC(h.NumRiv);
        std::vector<double> riv_zbed(h.NumRiv);
        std::vector<double> riv_zbank(h.NumRiv);

        for (int i = 0; i < h.NumRiv; i++) {
            riv_down_raw[i] = md->Riv[i].down;
            riv_toLake[i] = md->Riv[i].toLake;
            riv_BC[i] = md->Riv[i].BC;
            riv_Length[i] = md->Riv[i].Length;
            riv_depth[i] = md->Riv[i].depth;
            riv_BankSlope[i] = md->Riv[i].bankslope;
            riv_BottomWidth[i] = md->Riv[i].BottomWidth;
            riv_BedSlope[i] = md->Riv[i].BedSlope;
            riv_rivRough[i] = md->Riv[i].rivRough;
            riv_avgRough[i] = md->Riv[i].avgRough;
            riv_Dist2DownStream[i] = md->Riv[i].Dist2DownStream;
            riv_KsatH[i] = md->Riv[i].KsatH;
            riv_BedThick[i] = md->Riv[i].BedThick;
            riv_yBC[i] = md->Riv[i].yBC;
            riv_qBC[i] = md->Riv[i].qBC;
            riv_zbed[i] = md->Riv[i].zbed;
            riv_zbank[i] = md->Riv[i].zbank;
        }

        err = cudaAllocAndUpload(&h.riv_down_raw, riv_down_raw.data(), riv_down_raw.size());
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload riv_down_raw\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.riv_toLake, riv_toLake.data(), riv_toLake.size());
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload riv_toLake\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.riv_BC, riv_BC.data(), riv_BC.size());
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload riv_BC\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.riv_Length, riv_Length.data(), riv_Length.size());
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload riv_Length\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.riv_depth, riv_depth.data(), riv_depth.size());
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload riv_depth\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.riv_BankSlope, riv_BankSlope.data(), riv_BankSlope.size());
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload riv_BankSlope\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.riv_BottomWidth, riv_BottomWidth.data(), riv_BottomWidth.size());
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload riv_BottomWidth\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.riv_BedSlope, riv_BedSlope.data(), riv_BedSlope.size());
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload riv_BedSlope\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.riv_rivRough, riv_rivRough.data(), riv_rivRough.size());
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload riv_rivRough\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.riv_avgRough, riv_avgRough.data(), riv_avgRough.size());
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload riv_avgRough\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.riv_Dist2DownStream, riv_Dist2DownStream.data(), riv_Dist2DownStream.size());
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload riv_Dist2DownStream\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.riv_KsatH, riv_KsatH.data(), riv_KsatH.size());
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload riv_KsatH\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.riv_BedThick, riv_BedThick.data(), riv_BedThick.size());
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload riv_BedThick\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.riv_yBC, riv_yBC.data(), riv_yBC.size());
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload riv_yBC\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.riv_qBC, riv_qBC.data(), riv_qBC.size());
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload riv_qBC\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.riv_zbed, riv_zbed.data(), riv_zbed.size());
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload riv_zbed\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.riv_zbank, riv_zbank.data(), riv_zbank.size());
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload riv_zbank\n");
            goto fail;
        }
    }

    /* ------------------------------ Segment parameters ------------------------------ */
    if (h.NumSeg > 0) {
        std::vector<int> seg_iEle(h.NumSeg);
        std::vector<int> seg_iRiv(h.NumSeg);
        std::vector<double> seg_length(h.NumSeg);
        std::vector<double> seg_Cwr(h.NumSeg);
        std::vector<double> seg_KsatH(h.NumSeg);
        std::vector<double> seg_eqDistance(h.NumSeg);

        for (int i = 0; i < h.NumSeg; i++) {
            seg_iEle[i] = md->RivSeg[i].iEle;
            seg_iRiv[i] = md->RivSeg[i].iRiv;
            seg_length[i] = md->RivSeg[i].length;
            seg_Cwr[i] = md->RivSeg[i].Cwr;
            seg_KsatH[i] = md->RivSeg[i].KsatH;
            seg_eqDistance[i] = md->RivSeg[i].eqDistance;
        }

        err = cudaAllocAndUpload(&h.seg_iEle, seg_iEle.data(), seg_iEle.size());
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload seg_iEle\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.seg_iRiv, seg_iRiv.data(), seg_iRiv.size());
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload seg_iRiv\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.seg_length, seg_length.data(), seg_length.size());
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload seg_length\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.seg_Cwr, seg_Cwr.data(), seg_Cwr.size());
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload seg_Cwr\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.seg_KsatH, seg_KsatH.data(), seg_KsatH.size());
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload seg_KsatH\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.seg_eqDistance, seg_eqDistance.data(), seg_eqDistance.size());
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload seg_eqDistance\n");
            goto fail;
        }
    }

    /* ------------------------------ Lake parameters ------------------------------ */
    if (h.NumLake > 0) {
        std::vector<double> lake_zmin(h.NumLake);
        std::vector<double> lake_invNumEle(h.NumLake);
        std::vector<int> lake_bathy_off(h.NumLake);
        std::vector<int> lake_bathy_n(h.NumLake);

        int bathy_total = 0;
        for (int i = 0; i < h.NumLake; i++) {
            const int n = (md->lake[i].bathymetry.nvalue > 0 && md->lake[i].bathymetry.yi != nullptr &&
                           md->lake[i].bathymetry.ai != nullptr)
                              ? md->lake[i].bathymetry.nvalue
                              : 0;
            lake_bathy_off[i] = bathy_total;
            lake_bathy_n[i] = n;
            bathy_total += n;
        }
        h.bathy_nTotal = bathy_total;

        std::vector<double> bathy_yi(static_cast<size_t>(bathy_total));
        std::vector<double> bathy_ai(static_cast<size_t>(bathy_total));
        for (int i = 0; i < h.NumLake; i++) {
            lake_zmin[i] = md->lake[i].zmin;
            lake_invNumEle[i] = (md->lake[i].NumEleLake > 0) ? (1.0 / static_cast<double>(md->lake[i].NumEleLake)) : 0.0;

            const int off = lake_bathy_off[i];
            const int n = lake_bathy_n[i];
            for (int j = 0; j < n; j++) {
                bathy_yi[static_cast<size_t>(off + j)] = md->lake[i].bathymetry.yi[j];
                bathy_ai[static_cast<size_t>(off + j)] = md->lake[i].bathymetry.ai[j];
            }
        }

        err = cudaAllocAndUpload(&h.lake_zmin, lake_zmin.data(), lake_zmin.size());
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload lake_zmin\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.lake_invNumEle, lake_invNumEle.data(), lake_invNumEle.size());
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload lake_invNumEle\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.lake_bathy_off, lake_bathy_off.data(), lake_bathy_off.size());
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload lake_bathy_off\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.lake_bathy_n, lake_bathy_n.data(), lake_bathy_n.size());
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload lake_bathy_n\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.bathy_yi, bathy_yi.data(), bathy_yi.size());
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload bathy_yi\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.bathy_ai, bathy_ai.data(), bathy_ai.size());
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload bathy_ai\n");
            goto fail;
        }
    }

    /* ------------------------------ Scratch arrays ------------------------------ */
    err = cudaAllocAndUpload(&h.uYsf, static_cast<const double *>(nullptr), static_cast<size_t>(h.NumEle));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to allocate uYsf\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.uYus, static_cast<const double *>(nullptr), static_cast<size_t>(h.NumEle));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to allocate uYus\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.uYgw, static_cast<const double *>(nullptr), static_cast<size_t>(h.NumEle));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to allocate uYgw\n");
        goto fail;
    }
    if (h.NumRiv > 0) {
        err = cudaAllocAndUpload(&h.uYriv, static_cast<const double *>(nullptr), static_cast<size_t>(h.NumRiv));
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to allocate uYriv\n");
            goto fail;
        }
    }
    if (h.NumLake > 0) {
        err = cudaAllocAndUpload(&h.uYlake, static_cast<const double *>(nullptr), static_cast<size_t>(h.NumLake));
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to allocate uYlake\n");
            goto fail;
        }
    }
    err = cudaAllocAndUpload(&h.ele_satn, static_cast<const double *>(nullptr), static_cast<size_t>(h.NumEle));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to allocate ele_satn\n");
        goto fail;
    }
    err = cudaMemset(h.ele_satn, 0xFF, static_cast<size_t>(h.NumEle) * sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to initialize ele_satn\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.ele_effKH, static_cast<const double *>(nullptr), static_cast<size_t>(h.NumEle));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to allocate ele_effKH\n");
        goto fail;
    }

    err = cudaAllocAndUpload(&h.qEleInfil, md->qEleInfil, static_cast<size_t>(h.NumEle));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload qEleInfil\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.qEleExfil, md->qEleExfil, static_cast<size_t>(h.NumEle));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload qEleExfil\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.qEleRecharge, md->qEleRecharge, static_cast<size_t>(h.NumEle));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload qEleRecharge\n");
        goto fail;
    }

    err = cudaAllocAndUpload(&h.qEs, md->qEs, static_cast<size_t>(h.NumEle));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload qEs\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.qEu, md->qEu, static_cast<size_t>(h.NumEle));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload qEu\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.qEg, md->qEg, static_cast<size_t>(h.NumEle));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload qEg\n");
        goto fail;
    }

    err = cudaAllocAndUpload(&h.qTu, md->qTu, static_cast<size_t>(h.NumEle));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload qTu\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.qTg, md->qTg, static_cast<size_t>(h.NumEle));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload qTg\n");
        goto fail;
    }

    {
        std::vector<double> QeleSurf(static_cast<size_t>(h.NumEle) * 3);
        std::vector<double> QeleSub(static_cast<size_t>(h.NumEle) * 3);
        for (int i = 0; i < h.NumEle; i++) {
            for (int j = 0; j < 3; j++) {
                const size_t idx = static_cast<size_t>(i) * 3 + j;
                QeleSurf[idx] = (md->QeleSurf != nullptr && md->QeleSurf[i] != nullptr) ? md->QeleSurf[i][j] : 0.0;
                QeleSub[idx] = (md->QeleSub != nullptr && md->QeleSub[i] != nullptr) ? md->QeleSub[i][j] : 0.0;
            }
        }
        err = cudaAllocAndUpload(&h.QeleSurf, QeleSurf.data(), QeleSurf.size());
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload QeleSurf\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.QeleSub, QeleSub.data(), QeleSub.size());
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload QeleSub\n");
            goto fail;
        }
    }

    err = cudaAllocAndUpload(&h.Qe2r_Surf, md->Qe2r_Surf, static_cast<size_t>(h.NumEle));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload Qe2r_Surf\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.Qe2r_Sub, md->Qe2r_Sub, static_cast<size_t>(h.NumEle));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload Qe2r_Sub\n");
        goto fail;
    }

    /* Derived element diagnostics for host-side output (computed on demand). */
    {
        const bool need_ele_et = (md->CS.dt_qe_et > 0);
        const bool need_ele_eta = (md->CS.dt_qe_eta > 0);
        const bool need_ele_Q_surfTot = (md->CS.dt_Qe_surf > 0);
        const bool need_ele_Q_subTot = (md->CS.dt_Qe_sub > 0);

        if (need_ele_et) {
            err = cudaAllocAndUpload(&h.qEleTrans, static_cast<const double *>(nullptr), static_cast<size_t>(h.NumEle));
            if (err != cudaSuccess) {
                fprintf(stderr, "gpuInit: failed to allocate qEleTrans\n");
                goto fail;
            }
            err = cudaAllocAndUpload(&h.qEleEvapo, static_cast<const double *>(nullptr), static_cast<size_t>(h.NumEle));
            if (err != cudaSuccess) {
                fprintf(stderr, "gpuInit: failed to allocate qEleEvapo\n");
                goto fail;
            }
        }
        if (need_ele_eta) {
            err = cudaAllocAndUpload(&h.qEleETA, static_cast<const double *>(nullptr), static_cast<size_t>(h.NumEle));
            if (err != cudaSuccess) {
                fprintf(stderr, "gpuInit: failed to allocate qEleETA\n");
                goto fail;
            }
        }
        if (need_ele_Q_surfTot) {
            err = cudaAllocAndUpload(&h.QeleSurfTot, static_cast<const double *>(nullptr), static_cast<size_t>(h.NumEle));
            if (err != cudaSuccess) {
                fprintf(stderr, "gpuInit: failed to allocate QeleSurfTot\n");
                goto fail;
            }
        }
        if (need_ele_Q_subTot) {
            err = cudaAllocAndUpload(&h.QeleSubTot, static_cast<const double *>(nullptr), static_cast<size_t>(h.NumEle));
            if (err != cudaSuccess) {
                fprintf(stderr, "gpuInit: failed to allocate QeleSubTot\n");
                goto fail;
            }
        }
    }

    if (h.NumRiv > 0) {
        err = cudaAllocAndUpload(&h.QrivSurf, md->QrivSurf, static_cast<size_t>(h.NumRiv));
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload QrivSurf\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.QrivSub, md->QrivSub, static_cast<size_t>(h.NumRiv));
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload QrivSub\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.QrivUp, md->QrivUp, static_cast<size_t>(h.NumRiv));
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload QrivUp\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.QrivDown, md->QrivDown, static_cast<size_t>(h.NumRiv));
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload QrivDown\n");
            goto fail;
        }

        err = cudaAllocAndUpload(&h.riv_CSarea, static_cast<const double *>(nullptr), static_cast<size_t>(h.NumRiv));
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to allocate riv_CSarea\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.riv_CSperem, static_cast<const double *>(nullptr), static_cast<size_t>(h.NumRiv));
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to allocate riv_CSperem\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.riv_topWidth, static_cast<const double *>(nullptr), static_cast<size_t>(h.NumRiv));
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to allocate riv_topWidth\n");
            goto fail;
        }
    }

    if (h.NumLake > 0) {
        err = cudaAllocAndUpload(&h.QLakeSurf, md->QLakeSurf, static_cast<size_t>(h.NumLake));
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload QLakeSurf\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.QLakeSub, md->QLakeSub, static_cast<size_t>(h.NumLake));
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload QLakeSub\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.QLakeRivIn, md->QLakeRivIn, static_cast<size_t>(h.NumLake));
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload QLakeRivIn\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.QLakeRivOut, md->QLakeRivOut, static_cast<size_t>(h.NumLake));
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload QLakeRivOut\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.qLakePrcp, md->qLakePrcp, static_cast<size_t>(h.NumLake));
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload qLakePrcp\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.qLakeEvap, md->qLakeEvap, static_cast<size_t>(h.NumLake));
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload qLakeEvap\n");
            goto fail;
        }
        err = cudaAllocAndUpload(&h.y2LakeArea, md->y2LakeArea, static_cast<size_t>(h.NumLake));
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to upload y2LakeArea\n");
            goto fail;
        }
    }

    /* ------------------------------ Forcing arrays ------------------------------ */
    err = cudaAllocAndUpload(&h.qElePrep, static_cast<const double *>(nullptr), static_cast<size_t>(h.NumEle));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to allocate qElePrep\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.qEleNetPrep, static_cast<const double *>(nullptr), static_cast<size_t>(h.NumEle));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to allocate qEleNetPrep\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.qPotEvap, static_cast<const double *>(nullptr), static_cast<size_t>(h.NumEle));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to allocate qPotEvap\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.qPotTran, static_cast<const double *>(nullptr), static_cast<size_t>(h.NumEle));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to allocate qPotTran\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.qEleE_IC, static_cast<const double *>(nullptr), static_cast<size_t>(h.NumEle));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to allocate qEleE_IC\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.t_lai, static_cast<const double *>(nullptr), static_cast<size_t>(h.NumEle));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to allocate t_lai\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.fu_Surf, static_cast<const double *>(nullptr), static_cast<size_t>(h.NumEle));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to allocate fu_Surf\n");
        goto fail;
    }
    err = cudaAllocAndUpload(&h.fu_Sub, static_cast<const double *>(nullptr), static_cast<size_t>(h.NumEle));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to allocate fu_Sub\n");
        goto fail;
    }

    /* ------------------------------ Preconditioner storage ------------------------------ */
    {
        const size_t prec_count = static_cast<size_t>(h.NumEle) * 9u + static_cast<size_t>(h.NumRiv) +
                                  static_cast<size_t>(h.NumLake);
        err = cudaAllocAndUpload(&h.prec_inv, static_cast<const double *>(nullptr), prec_count);
        if (err != cudaSuccess) {
            fprintf(stderr, "gpuInit: failed to allocate prec_inv\n");
            goto fail;
        }
    }

    md->d_qElePrep = h.qElePrep;
    md->d_qEleNetPrep = h.qEleNetPrep;
    md->d_qPotEvap = h.qPotEvap;
    md->d_qPotTran = h.qPotTran;
    md->d_qEleE_IC = h.qEleE_IC;
    md->d_t_lai = h.t_lai;
    md->d_fu_Surf = h.fu_Surf;
    md->d_fu_Sub = h.fu_Sub;
    md->d_ele_yBC = h.ele_yBC;
    md->d_ele_QBC = h.ele_QBC;
    md->d_riv_yBC = h.riv_yBC;
    md->d_riv_qBC = h.riv_qBC;

    /* Finally: allocate & upload the DeviceModel itself. */
    err = cudaMalloc(reinterpret_cast<void **>(&md->d_model), sizeof(DeviceModel));
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to allocate md->d_model\n");
        goto fail;
    }
    err = cudaMemcpy(md->d_model, &h, sizeof(DeviceModel), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload md->d_model\n");
        goto fail;
    }
    delete md->h_model;
    md->h_model = new DeviceModel(h);

    /* Pin host output buffers for truly async D2H copies during ExportResults. */
    md->pinned_host_buffers.clear();
    {
        const size_t nEle = static_cast<size_t>(h.NumEle);
        const size_t nRiv = static_cast<size_t>(h.NumRiv);
        const size_t nLake = static_cast<size_t>(h.NumLake);

        if (md->CS.dt_qe_infil > 0) {
            tryHostRegister(md, md->qEleInfil, nEle * sizeof(double), "qEleInfil");
            tryHostRegister(md, md->qEleExfil, nEle * sizeof(double), "qEleExfil");
        }
        if (md->CS.dt_qe_rech > 0) {
            tryHostRegister(md, md->qEleRecharge, nEle * sizeof(double), "qEleRecharge");
        }
        if (md->CS.dt_qe_et > 0) {
            tryHostRegister(md, md->qEleTrans, nEle * sizeof(double), "qEleTrans");
            tryHostRegister(md, md->qEleEvapo, nEle * sizeof(double), "qEleEvapo");
        }
        if (md->CS.dt_qe_eta > 0) {
            tryHostRegister(md, md->qEleETA, nEle * sizeof(double), "qEleETA");
        }

        if (md->CS.dt_Qe_surf > 0) {
            tryHostRegister(md, md->QeleSurfTot, nEle * sizeof(double), "QeleSurfTot");
        }
        if (md->CS.dt_Qe_sub > 0) {
            tryHostRegister(md, md->QeleSubTot, nEle * sizeof(double), "QeleSubTot");
        }
        if (md->CS.dt_Qe_rsurf > 0) {
            tryHostRegister(md, md->Qe2r_Surf, nEle * sizeof(double), "Qe2r_Surf");
        }
        if (md->CS.dt_Qe_rsub > 0) {
            tryHostRegister(md, md->Qe2r_Sub, nEle * sizeof(double), "Qe2r_Sub");
        }

        if (md->CS.dt_Qe_surfx > 0 && h.NumEle > 0) {
            md->d2h_QeleSurf_flat.resize(nEle * 3u);
            tryHostRegister(md,
                            md->d2h_QeleSurf_flat.data(),
                            md->d2h_QeleSurf_flat.size() * sizeof(double),
                            "d2h_QeleSurf_flat");
        }
        if (md->CS.dt_Qe_subx > 0 && h.NumEle > 0) {
            md->d2h_QeleSub_flat.resize(nEle * 3u);
            tryHostRegister(md,
                            md->d2h_QeleSub_flat.data(),
                            md->d2h_QeleSub_flat.size() * sizeof(double),
                            "d2h_QeleSub_flat");
        }

        if (h.NumRiv > 0) {
            if (md->CS.dt_Qr_surf > 0) tryHostRegister(md, md->QrivSurf, nRiv * sizeof(double), "QrivSurf");
            if (md->CS.dt_Qr_sub > 0) tryHostRegister(md, md->QrivSub, nRiv * sizeof(double), "QrivSub");
            if (md->CS.dt_Qr_up > 0) tryHostRegister(md, md->QrivUp, nRiv * sizeof(double), "QrivUp");
            if (md->CS.dt_Qr_down > 0) tryHostRegister(md, md->QrivDown, nRiv * sizeof(double), "QrivDown");
        }

        if (md->CS.dt_lake > 0 && h.NumLake > 0) {
            tryHostRegister(md, md->yLakeStg, nLake * sizeof(double), "yLakeStg");
            tryHostRegister(md, md->QLakeSurf, nLake * sizeof(double), "QLakeSurf");
            tryHostRegister(md, md->QLakeSub, nLake * sizeof(double), "QLakeSub");
            tryHostRegister(md, md->QLakeRivIn, nLake * sizeof(double), "QLakeRivIn");
            tryHostRegister(md, md->QLakeRivOut, nLake * sizeof(double), "QLakeRivOut");
            tryHostRegister(md, md->qLakePrcp, nLake * sizeof(double), "qLakePrcp");
            tryHostRegister(md, md->qLakeEvap, nLake * sizeof(double), "qLakeEvap");
            tryHostRegister(md, md->y2LakeArea, nLake * sizeof(double), "y2LakeArea");
        }
    }

    return;

fail:
    if (precond_event != nullptr) {
        (void)cudaEventDestroy(precond_event);
        md->precond_setup_event = nullptr;
    }
    if (forcing_event != nullptr) {
        (void)cudaEventDestroy(forcing_event);
        md->forcing_copy_event = nullptr;
    }
    if (stream != nullptr) {
        (void)cudaStreamDestroy(stream);
        md->cuda_stream = nullptr;
    }
    freeDeviceBuffers(h);
    if (md->d_model != nullptr) {
        cudaFreeIfNotNull(md->d_model);
        md->d_model = nullptr;
    }
    delete md->h_model;
    md->h_model = nullptr;
    cudaDie(err, "gpuInit");
}

void gpuFree(Model_Data *md)
{
    if (md == nullptr) {
        return;
    }

    if (md->rhs_graph_exec != nullptr) {
        (void)cudaGraphExecDestroy(md->rhs_graph_exec);
        md->rhs_graph_exec = nullptr;
    }
    if (md->rhs_graph != nullptr) {
        (void)cudaGraphDestroy(md->rhs_graph);
        md->rhs_graph = nullptr;
    }
    md->rhs_graph_dY = nullptr;
    md->rhs_graph_dYdot = nullptr;
    md->rhs_graph_kernel_nodes = 0;
    md->rhs_graph_failed = 0;

    delete md->h_model;
    md->h_model = nullptr;

    if (md->cuda_stream != nullptr) {
        (void)cudaStreamSynchronize(md->cuda_stream);
    }

    unregisterPinnedHostBuffers(md);

    if (md->forcing_copy_event != nullptr) {
        (void)cudaEventDestroy(md->forcing_copy_event);
        md->forcing_copy_event = nullptr;
    }
    if (md->precond_setup_event != nullptr) {
        (void)cudaEventDestroy(md->precond_setup_event);
        md->precond_setup_event = nullptr;
    }
    if (md->cuda_stream != nullptr) {
        (void)cudaStreamDestroy(md->cuda_stream);
        md->cuda_stream = nullptr;
    }

    md->d_qEleNetPrep = nullptr;
    md->d_qPotEvap = nullptr;
    md->d_qPotTran = nullptr;
    md->d_qEleE_IC = nullptr;
    md->d_t_lai = nullptr;
    md->d_fu_Surf = nullptr;
    md->d_fu_Sub = nullptr;
    md->d_qElePrep = nullptr;
    md->d_ele_yBC = nullptr;
    md->d_ele_QBC = nullptr;
    md->d_riv_yBC = nullptr;
    md->d_riv_qBC = nullptr;

    md->nGpuPrecSetup = 0;

    if (md->d_model == nullptr) {
        return;
    }

    DeviceModel h{};
    const cudaError_t err = cudaMemcpy(&h, md->d_model, sizeof(DeviceModel), cudaMemcpyDeviceToHost);
    cudaDie(err, "gpuFree(cudaMemcpy D2H)");

    freeDeviceBuffers(h);
    cudaFreeIfNotNull(md->d_model);
    md->d_model = nullptr;
}

void Model_Data::gpuUpdateForcing()
{
    if (d_model == nullptr) {
        return;
    }
    if (NumEle <= 0) {
        return;
    }
    if (cuda_stream == nullptr) {
        fprintf(stderr, "gpuUpdateForcing: cuda_stream is not initialized\n");
        std::abort();
    }
    if (forcing_copy_event == nullptr) {
        fprintf(stderr, "gpuUpdateForcing: forcing_copy_event is not initialized\n");
        std::abort();
    }

    cudaStream_t stream = cuda_stream;
    const size_t bytes = static_cast<size_t>(NumEle) * sizeof(double);

    cudaError_t err = cudaMemcpyAsync(d_qElePrep, qElePrep, bytes, cudaMemcpyHostToDevice, stream);
    cudaDie(err, "gpuUpdateForcing(qElePrep)");

    err = cudaMemcpyAsync(d_qEleNetPrep, qEleNetPrep, bytes, cudaMemcpyHostToDevice, stream);
    cudaDie(err, "gpuUpdateForcing(qEleNetPrep)");
    err = cudaMemcpyAsync(d_qPotEvap, qPotEvap, bytes, cudaMemcpyHostToDevice, stream);
    cudaDie(err, "gpuUpdateForcing(qPotEvap)");
    err = cudaMemcpyAsync(d_qPotTran, qPotTran, bytes, cudaMemcpyHostToDevice, stream);
    cudaDie(err, "gpuUpdateForcing(qPotTran)");
    err = cudaMemcpyAsync(d_qEleE_IC, qEleE_IC, bytes, cudaMemcpyHostToDevice, stream);
    cudaDie(err, "gpuUpdateForcing(qEleE_IC)");
    err = cudaMemcpyAsync(d_t_lai, t_lai, bytes, cudaMemcpyHostToDevice, stream);
    cudaDie(err, "gpuUpdateForcing(t_lai)");
    err = cudaMemcpyAsync(d_fu_Surf, fu_Surf, bytes, cudaMemcpyHostToDevice, stream);
    cudaDie(err, "gpuUpdateForcing(fu_Surf)");
    err = cudaMemcpyAsync(d_fu_Sub, fu_Sub, bytes, cudaMemcpyHostToDevice, stream);
    cudaDie(err, "gpuUpdateForcing(fu_Sub)");

    if (d_ele_yBC != nullptr && d_ele_QBC != nullptr) {
        std::vector<double> h_ele_yBC(static_cast<size_t>(NumEle));
        std::vector<double> h_ele_QBC(static_cast<size_t>(NumEle));
        for (int i = 0; i < NumEle; i++) {
            h_ele_yBC[static_cast<size_t>(i)] = Ele[i].yBC;
            h_ele_QBC[static_cast<size_t>(i)] = Ele[i].QBC;
        }
        err = cudaMemcpyAsync(d_ele_yBC, h_ele_yBC.data(), bytes, cudaMemcpyHostToDevice, stream);
        cudaDie(err, "gpuUpdateForcing(ele_yBC)");
        err = cudaMemcpyAsync(d_ele_QBC, h_ele_QBC.data(), bytes, cudaMemcpyHostToDevice, stream);
        cudaDie(err, "gpuUpdateForcing(ele_QBC)");
    }

    if (NumRiv > 0 && d_riv_yBC != nullptr && d_riv_qBC != nullptr) {
        const size_t bytes_riv = static_cast<size_t>(NumRiv) * sizeof(double);
        std::vector<double> h_riv_yBC(static_cast<size_t>(NumRiv));
        std::vector<double> h_riv_qBC(static_cast<size_t>(NumRiv));
        for (int i = 0; i < NumRiv; i++) {
            h_riv_yBC[static_cast<size_t>(i)] = Riv[i].yBC;
            h_riv_qBC[static_cast<size_t>(i)] = Riv[i].qBC;
        }
        err = cudaMemcpyAsync(d_riv_yBC, h_riv_yBC.data(), bytes_riv, cudaMemcpyHostToDevice, stream);
        cudaDie(err, "gpuUpdateForcing(riv_yBC)");
        err = cudaMemcpyAsync(d_riv_qBC, h_riv_qBC.data(), bytes_riv, cudaMemcpyHostToDevice, stream);
        cudaDie(err, "gpuUpdateForcing(riv_qBC)");
    }

    err = cudaEventRecord(forcing_copy_event, stream);
    cudaDie(err, "gpuUpdateForcing(cudaEventRecord)");

    nGpuForcingCopy++;
}

void Model_Data::gpuWaitForcingCopy()
{
    if (d_model == nullptr) {
        return;
    }
    if (nGpuForcingCopy == 0) {
        return;
    }
    if (forcing_copy_event == nullptr) {
        fprintf(stderr, "gpuWaitForcingCopy: forcing_copy_event is not initialized\n");
        std::abort();
    }

    const cudaError_t err = cudaEventSynchronize(forcing_copy_event);
    cudaDie(err, "gpuWaitForcingCopy(cudaEventSynchronize)");
}

void Model_Data::gpuSyncStateFromDevice(N_Vector y)
{
    if (y == nullptr) {
        return;
    }

    if (N_VGetVectorID(y) != SUNDIALS_NVEC_CUDA) {
        return;
    }

    N_VCopyFromDevice_Cuda(y);
    (void)N_VGetHostArrayPointer_Cuda(y);
}

namespace {

inline int cap_blocks(int blocks) { return (blocks > 65535) ? 65535 : blocks; }

__global__ void k_derive_element_diagnostics(const DeviceModel *m)
{
    if (m == nullptr || m->NumEle <= 0) {
        return;
    }

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m->NumEle; i += blockDim.x * gridDim.x) {
        const int iLake = (m->NumLake > 0 && m->ele_iLake != nullptr) ? m->ele_iLake[i] : 0;
        const bool is_lake_ele = (iLake > 0);

        if (m->QeleSurfTot != nullptr) {
            double total = (m->Qe2r_Surf != nullptr) ? m->Qe2r_Surf[i] : 0.0;
            if (m->QeleSurf != nullptr) {
                const int base = i * 3;
                total += m->QeleSurf[base + 0] + m->QeleSurf[base + 1] + m->QeleSurf[base + 2];
            }
            m->QeleSurfTot[i] = total;
        }

        if (m->QeleSubTot != nullptr) {
            double total = (m->Qe2r_Sub != nullptr) ? m->Qe2r_Sub[i] : 0.0;
            if (m->QeleSub != nullptr) {
                const int base = i * 3;
                total += m->QeleSub[base + 0] + m->QeleSub[base + 1] + m->QeleSub[base + 2];
            }
            m->QeleSubTot[i] = total;
        }

        const double trans =
            is_lake_ele ? 0.0 : ((m->qTu != nullptr) ? m->qTu[i] : 0.0) + ((m->qTg != nullptr) ? m->qTg[i] : 0.0);
        const double evapo =
            is_lake_ele ? ((m->qPotEvap != nullptr) ? m->qPotEvap[i] : 0.0)
                        : ((m->qEs != nullptr) ? m->qEs[i] : 0.0) + ((m->qEu != nullptr) ? m->qEu[i] : 0.0) +
                              ((m->qEg != nullptr) ? m->qEg[i] : 0.0);

        if (m->qEleTrans != nullptr) {
            m->qEleTrans[i] = trans;
        }
        if (m->qEleEvapo != nullptr) {
            m->qEleEvapo[i] = evapo;
        }
        if (m->qEleETA != nullptr) {
            const double e_ic = (is_lake_ele || m->qEleE_IC == nullptr) ? 0.0 : m->qEleE_IC[i];
            m->qEleETA[i] = e_ic + evapo + trans;
        }
    }
}

} // namespace

void Model_Data::gpuSyncDiagnosticsFromDevice(N_Vector y)
{
    if (y == nullptr) {
        return;
    }
    if (d_model == nullptr || h_model == nullptr) {
        return;
    }
    if (N_VGetVectorID(y) != SUNDIALS_NVEC_CUDA) {
        return;
    }

    cudaStream_t stream = SHUD_NVecCudaStream(y);

    bool did_copy = false;
    auto d2h = [&](double *host, const double *dev, size_t count, const char *what) {
        if (host == nullptr || dev == nullptr || count == 0) {
            return;
        }
        const cudaError_t err =
            cudaMemcpyAsync(host, dev, count * sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaDie(err, what);
        did_copy = true;
    };

    const bool need_ele_infil = (CS.dt_qe_infil > 0);
    const bool need_ele_rech = (CS.dt_qe_rech > 0);
    const bool need_ele_eta = (CS.dt_qe_eta > 0);
    const bool need_ele_et = (CS.dt_qe_et > 0);

    const bool need_ele_Q_surfTot = (CS.dt_Qe_surf > 0);
    const bool need_ele_Q_surfx = (CS.dt_Qe_surfx > 0);
    const bool need_ele_Q_subTot = (CS.dt_Qe_sub > 0);
    const bool need_ele_Q_subx = (CS.dt_Qe_subx > 0);
    const bool need_ele_Q_rsurf = (CS.dt_Qe_rsurf > 0);
    const bool need_ele_Q_rsub = (CS.dt_Qe_rsub > 0);

    const bool need_ele_QeleSurf = need_ele_Q_surfx;
    const bool need_ele_QeleSub = need_ele_Q_subx;
    const bool need_ele_Qe2r_Surf = need_ele_Q_rsurf;
    const bool need_ele_Qe2r_Sub = need_ele_Q_rsub;

    const bool need_riv_outputs =
        (CS.dt_Qr_up > 0) || (CS.dt_Qr_down > 0) || (CS.dt_Qr_sub > 0) || (CS.dt_Qr_surf > 0);
    const bool need_lake_outputs = (CS.dt_lake > 0) && (NumLake > 0);

    const size_t nEle = static_cast<size_t>(NumEle);
    const size_t nRiv = static_cast<size_t>(NumRiv);
    const size_t nLake = static_cast<size_t>(NumLake);

    if (NumEle > 0) {
        const bool need_ele_derived = need_ele_Q_surfTot || need_ele_Q_subTot || need_ele_et || need_ele_eta;
        if (need_ele_derived) {
            constexpr int kBlockSize = 256;
            const int blocks = cap_blocks((NumEle + kBlockSize - 1) / kBlockSize);
            k_derive_element_diagnostics<<<blocks, kBlockSize, 0, stream>>>(d_model);
            cudaDie(cudaPeekAtLastError(), "gpuSyncDiagnosticsFromDevice(k_derive_element_diagnostics launch)");
        }

        if (need_ele_infil) {
            d2h(qEleInfil, h_model->qEleInfil, nEle, "gpuSyncDiagnosticsFromDevice(qEleInfil)");
            d2h(qEleExfil, h_model->qEleExfil, nEle, "gpuSyncDiagnosticsFromDevice(qEleExfil)");
        }
        if (need_ele_rech) {
            d2h(qEleRecharge, h_model->qEleRecharge, nEle, "gpuSyncDiagnosticsFromDevice(qEleRecharge)");
        }
        if (need_ele_et) {
            d2h(qEleTrans, h_model->qEleTrans, nEle, "gpuSyncDiagnosticsFromDevice(qEleTrans)");
            d2h(qEleEvapo, h_model->qEleEvapo, nEle, "gpuSyncDiagnosticsFromDevice(qEleEvapo)");
        }
        if (need_ele_eta) {
            d2h(qEleETA, h_model->qEleETA, nEle, "gpuSyncDiagnosticsFromDevice(qEleETA)");
        }
        if (need_ele_Q_surfTot) {
            d2h(QeleSurfTot, h_model->QeleSurfTot, nEle, "gpuSyncDiagnosticsFromDevice(QeleSurfTot)");
        }
        if (need_ele_Q_subTot) {
            d2h(QeleSubTot, h_model->QeleSubTot, nEle, "gpuSyncDiagnosticsFromDevice(QeleSubTot)");
        }

        if (need_ele_Qe2r_Surf) {
            d2h(Qe2r_Surf, h_model->Qe2r_Surf, nEle, "gpuSyncDiagnosticsFromDevice(Qe2r_Surf)");
        }
        if (need_ele_Qe2r_Sub) {
            d2h(Qe2r_Sub, h_model->Qe2r_Sub, nEle, "gpuSyncDiagnosticsFromDevice(Qe2r_Sub)");
        }
        if (need_ele_QeleSurf) {
            d2h_QeleSurf_flat.resize(nEle * 3u);
            d2h(d2h_QeleSurf_flat.data(),
                h_model->QeleSurf,
                d2h_QeleSurf_flat.size(),
                "gpuSyncDiagnosticsFromDevice(QeleSurf)");
        }
        if (need_ele_QeleSub) {
            d2h_QeleSub_flat.resize(nEle * 3u);
            d2h(d2h_QeleSub_flat.data(),
                h_model->QeleSub,
                d2h_QeleSub_flat.size(),
                "gpuSyncDiagnosticsFromDevice(QeleSub)");
        }
    }

    if (need_riv_outputs && NumRiv > 0) {
        if (CS.dt_Qr_surf > 0) {
            d2h(QrivSurf, h_model->QrivSurf, nRiv, "gpuSyncDiagnosticsFromDevice(QrivSurf)");
        }
        if (CS.dt_Qr_sub > 0) {
            d2h(QrivSub, h_model->QrivSub, nRiv, "gpuSyncDiagnosticsFromDevice(QrivSub)");
        }
        if (CS.dt_Qr_up > 0) {
            d2h(QrivUp, h_model->QrivUp, nRiv, "gpuSyncDiagnosticsFromDevice(QrivUp)");
        }
        if (CS.dt_Qr_down > 0) {
            d2h(QrivDown, h_model->QrivDown, nRiv, "gpuSyncDiagnosticsFromDevice(QrivDown)");
        }
    }

    if (need_lake_outputs) {
        d2h(yLakeStg, h_model->uYlake, nLake, "gpuSyncDiagnosticsFromDevice(yLakeStg)");
        d2h(QLakeSurf, h_model->QLakeSurf, nLake, "gpuSyncDiagnosticsFromDevice(QLakeSurf)");
        d2h(QLakeSub, h_model->QLakeSub, nLake, "gpuSyncDiagnosticsFromDevice(QLakeSub)");
        d2h(QLakeRivIn, h_model->QLakeRivIn, nLake, "gpuSyncDiagnosticsFromDevice(QLakeRivIn)");
        d2h(QLakeRivOut, h_model->QLakeRivOut, nLake, "gpuSyncDiagnosticsFromDevice(QLakeRivOut)");
        d2h(qLakePrcp, h_model->qLakePrcp, nLake, "gpuSyncDiagnosticsFromDevice(qLakePrcp)");
        d2h(qLakeEvap, h_model->qLakeEvap, nLake, "gpuSyncDiagnosticsFromDevice(qLakeEvap)");
        d2h(y2LakeArea, h_model->y2LakeArea, nLake, "gpuSyncDiagnosticsFromDevice(y2LakeArea)");
    }

    if (!did_copy) {
        return;
    }

    cudaDie(cudaStreamSynchronize(stream), "gpuSyncDiagnosticsFromDevice(cudaStreamSynchronize)");

    if (need_ele_QeleSurf && NumEle > 0 && QeleSurf != nullptr) {
        for (int i = 0; i < NumEle; i++) {
            if (QeleSurf[i] == nullptr) {
                continue;
            }
            const size_t base = static_cast<size_t>(i) * 3u;
            QeleSurf[i][0] = d2h_QeleSurf_flat[base + 0];
            QeleSurf[i][1] = d2h_QeleSurf_flat[base + 1];
            QeleSurf[i][2] = d2h_QeleSurf_flat[base + 2];
        }
    }
    if (need_ele_QeleSub && NumEle > 0 && QeleSub != nullptr) {
        for (int i = 0; i < NumEle; i++) {
            if (QeleSub[i] == nullptr) {
                continue;
            }
            const size_t base = static_cast<size_t>(i) * 3u;
            QeleSub[i][0] = d2h_QeleSub_flat[base + 0];
            QeleSub[i][1] = d2h_QeleSub_flat[base + 1];
            QeleSub[i][2] = d2h_QeleSub_flat[base + 2];
        }
    }
}

#endif /* _CUDA_ON */
