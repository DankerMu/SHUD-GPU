#ifndef SHUD_GPU_DEVICECONTEXT_HPP
#define SHUD_GPU_DEVICECONTEXT_HPP

#include <cstddef>

struct DeviceModel {
    int NumEle = 0;
    int NumRiv = 0;
    int NumSeg = 0;
    int NumLake = 0;
    int bathy_nTotal = 0;
    int CloseBoundary = 0;

    /* Element static parameters */
    double *ele_area = nullptr;
    double *ele_z_surf = nullptr;
    double *ele_z_bottom = nullptr;
    double *ele_depression = nullptr;
    double *ele_Rough = nullptr;

    int *ele_nabr = nullptr;      /* [NumEle * 3] */
    int *ele_lakenabr = nullptr;  /* [NumEle * 3] */
    int *ele_nabrToMe = nullptr;  /* [NumEle * 3] */
    double *ele_edge = nullptr;   /* [NumEle * 3] */
    double *ele_Dist2Nabor = nullptr; /* [NumEle * 3] */
    double *ele_Dist2Edge = nullptr;  /* [NumEle * 3] */
    double *ele_avgRough = nullptr;   /* [NumEle * 3] */

    double *ele_AquiferDepth = nullptr;
    double *ele_Sy = nullptr;
    double *ele_infD = nullptr;
    double *ele_infKsatV = nullptr;
    double *ele_ThetaS = nullptr;
    double *ele_ThetaR = nullptr;
    double *ele_ThetaFC = nullptr;
    double *ele_Alpha = nullptr;
    double *ele_Beta = nullptr;
    double *ele_hAreaF = nullptr;
    double *ele_macKsatV = nullptr;
    double *ele_KsatH = nullptr;
    double *ele_KsatV = nullptr;
    double *ele_geo_vAreaF = nullptr;
    double *ele_macKsatH = nullptr;
    double *ele_macD = nullptr;
    double *ele_RzD = nullptr;
    double *ele_VegFrac = nullptr;
    double *ele_ImpAF = nullptr;
    int *ele_iLake = nullptr;
    int *ele_iBC = nullptr;
    int *ele_iSS = nullptr;
    double *ele_yBC = nullptr;
    double *ele_QBC = nullptr;
    double *ele_QSS = nullptr;

    /* River static parameters */
    int *riv_down_raw = nullptr;
    int *riv_toLake = nullptr;
    int *riv_BC = nullptr;
    double *riv_Length = nullptr;
    double *riv_depth = nullptr;
    double *riv_BankSlope = nullptr;
    double *riv_BottomWidth = nullptr;
    double *riv_BedSlope = nullptr;
    double *riv_rivRough = nullptr;
    double *riv_avgRough = nullptr;
    double *riv_Dist2DownStream = nullptr;
    double *riv_KsatH = nullptr;
    double *riv_BedThick = nullptr;
    double *riv_yBC = nullptr;
    double *riv_qBC = nullptr;
    double *riv_zbed = nullptr;
    double *riv_zbank = nullptr;

    /* Segment parameters */
    int *seg_iEle = nullptr;
    int *seg_iRiv = nullptr;
    double *seg_length = nullptr;
    double *seg_Cwr = nullptr;
    double *seg_KsatH = nullptr;
    double *seg_eqDistance = nullptr;

    /* Lake parameters */
    double *lake_zmin = nullptr;
    double *lake_invNumEle = nullptr;
    int *lake_bathy_off = nullptr;
    int *lake_bathy_n = nullptr;
    double *bathy_yi = nullptr;
    double *bathy_ai = nullptr;

    /* Forcing arrays (updated per forcing step) */
    double *qElePrep = nullptr;
    double *qEleNetPrep = nullptr;
    double *qPotEvap = nullptr;
    double *qPotTran = nullptr;
    double *qEleE_IC = nullptr;
    double *t_lai = nullptr;
    double *fu_Surf = nullptr;
    double *fu_Sub = nullptr;

    /* Scratch arrays (reused for each RHS evaluation) */
    double *uYsf = nullptr;
    double *uYus = nullptr;
    double *uYgw = nullptr;
    double *uYriv = nullptr;
    double *uYlake = nullptr;
    double *ele_satn = nullptr;
    double *ele_effKH = nullptr;

    double *qEleInfil = nullptr;
    double *qEleExfil = nullptr;
    double *qEleRecharge = nullptr;
    double *qEs = nullptr;
    double *qEu = nullptr;
    double *qEg = nullptr;
    double *qTu = nullptr;
    double *qTg = nullptr;

    double *QeleSurf = nullptr; /* [NumEle * 3] */
    double *QeleSub = nullptr;  /* [NumEle * 3] */

    double *QrivSurf = nullptr;
    double *QrivSub = nullptr;
    double *QrivUp = nullptr;
    double *QrivDown = nullptr;
    double *Qe2r_Surf = nullptr;
    double *Qe2r_Sub = nullptr;

    double *QLakeSurf = nullptr;
    double *QLakeSub = nullptr;
    double *QLakeRivIn = nullptr;
    double *QLakeRivOut = nullptr;
    double *qLakePrcp = nullptr;
    double *qLakeEvap = nullptr;
    double *y2LakeArea = nullptr;

    double *riv_CSarea = nullptr;
    double *riv_CSperem = nullptr;
    double *riv_topWidth = nullptr;
};

#ifdef _CUDA_ON
class Model_Data;
void gpuInit(Model_Data *md);
void gpuFree(Model_Data *md);
#endif

#endif /* SHUD_GPU_DEVICECONTEXT_HPP */
