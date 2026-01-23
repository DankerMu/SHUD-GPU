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

void freeDeviceBuffers(DeviceModel &h)
{
    cudaFreeIfNotNull(h.ele_area);
    cudaFreeIfNotNull(h.ele_z_surf);
    cudaFreeIfNotNull(h.ele_z_bottom);
    cudaFreeIfNotNull(h.ele_depression);
    cudaFreeIfNotNull(h.ele_Rough);

    cudaFreeIfNotNull(h.ele_nabr);
    cudaFreeIfNotNull(h.ele_lakenabr);
    cudaFreeIfNotNull(h.ele_edge);

    cudaFreeIfNotNull(h.ele_AquiferDepth);
    cudaFreeIfNotNull(h.ele_Sy);
    cudaFreeIfNotNull(h.ele_infD);
    cudaFreeIfNotNull(h.ele_infKsatV);
    cudaFreeIfNotNull(h.ele_ThetaS);
    cudaFreeIfNotNull(h.ele_ThetaR);
    cudaFreeIfNotNull(h.ele_ThetaFC);
    cudaFreeIfNotNull(h.ele_KsatH);
    cudaFreeIfNotNull(h.ele_macKsatH);
    cudaFreeIfNotNull(h.ele_macD);
    cudaFreeIfNotNull(h.ele_VegFrac);
    cudaFreeIfNotNull(h.ele_ImpAF);

    cudaFreeIfNotNull(h.riv_down_raw);
    cudaFreeIfNotNull(h.riv_toLake);
    cudaFreeIfNotNull(h.riv_Length);
    cudaFreeIfNotNull(h.riv_depth);
    cudaFreeIfNotNull(h.riv_BankSlope);
    cudaFreeIfNotNull(h.riv_BottomWidth);
    cudaFreeIfNotNull(h.riv_BedSlope);
    cudaFreeIfNotNull(h.riv_rivRough);

    cudaFreeIfNotNull(h.seg_iEle);
    cudaFreeIfNotNull(h.seg_iRiv);
    cudaFreeIfNotNull(h.seg_length);
    cudaFreeIfNotNull(h.seg_Cwr);

    cudaFreeIfNotNull(h.lake_zmin);
    cudaFreeIfNotNull(h.lake_invNumEle);
    cudaFreeIfNotNull(h.lake_bathy_off);
    cudaFreeIfNotNull(h.lake_bathy_n);
    cudaFreeIfNotNull(h.bathy_yi);
    cudaFreeIfNotNull(h.bathy_ai);

    cudaFreeIfNotNull(h.qEleInfil);
    cudaFreeIfNotNull(h.qEleExfil);
    cudaFreeIfNotNull(h.qEleRecharge);
    cudaFreeIfNotNull(h.qEs);
    cudaFreeIfNotNull(h.qEu);
    cudaFreeIfNotNull(h.qEg);
    cudaFreeIfNotNull(h.qTu);
    cudaFreeIfNotNull(h.qTg);

    cudaFreeIfNotNull(h.QeleSurf);
    cudaFreeIfNotNull(h.QeleSub);

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

    DeviceModel h{};
    h.NumEle = md->NumEle;
    h.NumRiv = md->NumRiv;
    h.NumSeg = md->NumSegmt;
    h.NumLake = md->NumLake;

    cudaError_t err = cudaSuccess;

    /* ------------------------------ Element static parameters ------------------------------ */
    std::vector<double> ele_area(h.NumEle);
    std::vector<double> ele_z_surf(h.NumEle);
    std::vector<double> ele_z_bottom(h.NumEle);
    std::vector<double> ele_depression(h.NumEle);
    std::vector<double> ele_Rough(h.NumEle);

    std::vector<int> ele_nabr(static_cast<size_t>(h.NumEle) * 3);
    std::vector<int> ele_lakenabr(static_cast<size_t>(h.NumEle) * 3);
    std::vector<double> ele_edge(static_cast<size_t>(h.NumEle) * 3);

    std::vector<double> ele_AquiferDepth(h.NumEle);
    std::vector<double> ele_Sy(h.NumEle);
    std::vector<double> ele_infD(h.NumEle);
    std::vector<double> ele_infKsatV(h.NumEle);
    std::vector<double> ele_ThetaS(h.NumEle);
    std::vector<double> ele_ThetaR(h.NumEle);
    std::vector<double> ele_ThetaFC(h.NumEle);
    std::vector<double> ele_KsatH(h.NumEle);
    std::vector<double> ele_macKsatH(h.NumEle);
    std::vector<double> ele_macD(h.NumEle);
    std::vector<double> ele_VegFrac(h.NumEle);
    std::vector<double> ele_ImpAF(h.NumEle);

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
        ele_KsatH[i] = md->Ele[i].KsatH;
        ele_macKsatH[i] = md->Ele[i].macKsatH;
        ele_macD[i] = md->Ele[i].macD;
        ele_VegFrac[i] = md->Ele[i].VegFrac;
        ele_ImpAF[i] = md->Ele[i].ImpAF;

        for (int j = 0; j < 3; j++) {
            const size_t idx = static_cast<size_t>(i) * 3 + j;
            ele_nabr[idx] = md->Ele[i].nabr[j];
            ele_lakenabr[idx] = md->Ele[i].lakenabr[j];
            ele_edge[idx] = md->Ele[i].edge[j];
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
    err = cudaAllocAndUpload(&h.ele_edge, ele_edge.data(), ele_edge.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_edge\n");
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
    err = cudaAllocAndUpload(&h.ele_KsatH, ele_KsatH.data(), ele_KsatH.size());
    if (err != cudaSuccess) {
        fprintf(stderr, "gpuInit: failed to upload ele_KsatH\n");
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

    /* ------------------------------ River static parameters ------------------------------ */
    if (h.NumRiv > 0) {
        std::vector<int> riv_down_raw(h.NumRiv);
        std::vector<int> riv_toLake(h.NumRiv);
        std::vector<double> riv_Length(h.NumRiv);
        std::vector<double> riv_depth(h.NumRiv);
        std::vector<double> riv_BankSlope(h.NumRiv);
        std::vector<double> riv_BottomWidth(h.NumRiv);
        std::vector<double> riv_BedSlope(h.NumRiv);
        std::vector<double> riv_rivRough(h.NumRiv);

        for (int i = 0; i < h.NumRiv; i++) {
            riv_down_raw[i] = md->Riv[i].down;
            riv_toLake[i] = md->Riv[i].toLake;
            riv_Length[i] = md->Riv[i].Length;
            riv_depth[i] = md->Riv[i].depth;
            riv_BankSlope[i] = md->Riv[i].bankslope;
            riv_BottomWidth[i] = md->Riv[i].BottomWidth;
            riv_BedSlope[i] = md->Riv[i].BedSlope;
            riv_rivRough[i] = md->Riv[i].rivRough;
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
    }

    /* ------------------------------ Segment parameters ------------------------------ */
    if (h.NumSeg > 0) {
        std::vector<int> seg_iEle(h.NumSeg);
        std::vector<int> seg_iRiv(h.NumSeg);
        std::vector<double> seg_length(h.NumSeg);
        std::vector<double> seg_Cwr(h.NumSeg);

        for (int i = 0; i < h.NumSeg; i++) {
            seg_iEle[i] = md->RivSeg[i].iEle;
            seg_iRiv[i] = md->RivSeg[i].iRiv;
            seg_length[i] = md->RivSeg[i].length;
            seg_Cwr[i] = md->RivSeg[i].Cwr;
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
    return;

fail:
    freeDeviceBuffers(h);
    if (md->d_model != nullptr) {
        cudaFreeIfNotNull(md->d_model);
        md->d_model = nullptr;
    }
    cudaDie(err, "gpuInit");
}

void gpuFree(Model_Data *md)
{
    if (md == nullptr || md->d_model == nullptr) {
        return;
    }

    DeviceModel h{};
    const cudaError_t err = cudaMemcpy(&h, md->d_model, sizeof(DeviceModel), cudaMemcpyDeviceToHost);
    cudaDie(err, "gpuFree(cudaMemcpy D2H)");

    freeDeviceBuffers(h);
    cudaFreeIfNotNull(md->d_model);
    md->d_model = nullptr;
}

#endif /* _CUDA_ON */
