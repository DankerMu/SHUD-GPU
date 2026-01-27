#include "DeviceContext.hpp"
#include "Macros.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

__global__ void k_ele_local(const DeviceModel *m);

namespace {

double h_min(double a, double b) { return a < b ? a : b; }
double h_max(double a, double b) { return a > b ? a : b; }

double soilMoistureStress(double ThetaS, double ThetaR, double SatRatio)
{
    const double fc = ThetaS * FieldCapacityRatio;
    double beta_s = (SatRatio * (ThetaS - ThetaR) - ThetaR) / (fc - ThetaR);
    beta_s = h_min(h_max(beta_s, 0.0), 1.0);
    beta_s = 0.5 * (1.0 - std::cos(PI * beta_s));
    return beta_s;
}

double effKH(double Ygw, double aqDepth, double MacD, double Kmac, double AF, double Kmx)
{
    if (MacD <= ZERO || Ygw < aqDepth - MacD) {
        return Kmx;
    }
    if (Ygw > aqDepth) {
        return (Kmac * MacD * AF + Kmx * (aqDepth - MacD * AF)) / aqDepth;
    }
    return (Kmac * (Ygw - (aqDepth - MacD)) * AF +
            Kmx * (aqDepth - MacD + (Ygw - (aqDepth - MacD)) * (1.0 - AF))) /
           Ygw;
}

struct EtOut {
    double qEs = 0.0;
    double qEu = 0.0;
    double qEg = 0.0;
    double qTu = 0.0;
    double qTg = 0.0;
    double satn_new = 0.0;
    double effKH_val = 0.0;
};

EtOut host_reference(double ysf,
                     double yus,
                     double ygw,
                     double satn_prev,
                     double AquiferDepth,
                     double infD,
                     double ThetaS,
                     double ThetaR,
                     double Beta,
                     double macD,
                     double macKsatH,
                     double geo_vAreaF,
                     double KsatH,
                     double VegFrac,
                     double ImpAF,
                     double RzD,
                     double qPotEvap,
                     double qPotTran,
                     double lai,
                     double qEIC)
{
    EtOut out{};
    out.effKH_val = effKH(ygw, AquiferDepth, macD, macKsatH, geo_vAreaF, KsatH);

    double deficit = AquiferDepth - ygw;
    double satn_new = 1.0;
    double theta = ThetaS;
    if (deficit <= 0.0) {
        deficit = 0.0;
        satn_new = 1.0;
        theta = ThetaS;
    } else {
        theta = (deficit > 0.0) ? (yus / deficit * ThetaS) : ThetaS;
        satn_new = (ThetaS - ThetaR > 0.0) ? ((theta - ThetaR) / (ThetaS - ThetaR)) : 0.0;
    }

    double satKr = 0.0;
    if (satn_new > 0.99) {
        satn_new = 1.0;
        satKr = 1.0;
        theta = ThetaS;
    } else if (satn_new <= ZERO) {
        satn_new = 0.0;
        satKr = 0.0;
        theta = ThetaR;
    } else {
        (void)satKr;
        theta = theta;
    }

    const bool satn_prev_valid = std::isfinite(satn_prev) && satn_prev >= 0.0 && satn_prev <= 1.0;
    if (!satn_prev_valid) {
        satn_prev = satn_new;
    }
    const double iBeta = soilMoistureStress(ThetaS, ThetaR, satn_prev);

    const double va = VegFrac;
    const double vb = 1.0 - VegFrac;
    const double pj = 1.0 - ImpAF;

    const double WetlandLevel = AquiferDepth - infD;
    const double RootReachLevel = AquiferDepth - RzD;

    double Es = h_min(h_max(0.0, ysf), qPotEvap) * vb;
    double Eu = 0.0;
    double Eg = 0.0;
    if (Es < qPotEvap) {
        if (ygw > WetlandLevel) {
            Eg = h_min(h_max(0.0, ygw), qPotEvap - Es) * pj * vb;
            Eu = 0.0;
        } else {
            Eg = 0.0;
            Eu = h_min(h_max(0.0, yus), iBeta * (qPotEvap - Es)) * pj * vb;
        }
    }

    double Tu = 0.0;
    double Tg = 0.0;
    if (lai > ZERO) {
        if (qEIC >= qPotTran) {
            Tg = 0.0;
            Tu = 0.0;
        } else {
            if (ygw > RootReachLevel) {
                Tg = h_min(h_max(0.0, ygw), (qPotTran - qEIC)) * pj * va;
                Tu = 0.0;
            } else {
                Tg = 0.0;
                Tu = h_min(h_max(0.0, yus), iBeta * (qPotTran - qEIC)) * pj * va;
            }
        }
    }

    out.qEs = Es;
    out.qEu = Eu;
    out.qEg = Eg;
    out.qTu = Tu;
    out.qTg = Tg;
    out.satn_new = satn_new;
    return out;
}

bool nearly_equal(double a, double b, double atol, double rtol)
{
    const double diff = std::fabs(a - b);
    const double denom = std::fmax(std::fabs(a), std::fabs(b));
    const double thr = std::fmax(atol, rtol * denom);
    return diff <= thr;
}

struct CaseInput {
    const char *name = "";
    double ysf = 0.0;
    double yus = 0.0;
    double ygw = 0.0;
    double satn_prev = 0.0;

    double AquiferDepth = 1.0;
    double infD = 0.1;
    double ThetaS = 0.45;
    double ThetaR = 0.05;
    double Beta = 4.0;
    double macD = 0.2;
    double macKsatH = 1.0e-4;
    double geo_vAreaF = 0.1;
    double KsatH = 1.0e-5;
    double VegFrac = 0.6;
    double ImpAF = 0.0;
    double RzD = 0.3;

    double qPotEvap = 1.0e-6;
    double qPotTran = 2.0e-6;
    double lai = 2.0;
    double qEIC = 0.0;
};

void cuda_check(cudaError_t err, const char *what)
{
    if (err == cudaSuccess) {
        return;
    }
    std::fprintf(stderr, "CUDA_ERROR: %s: %s\n", what, cudaGetErrorString(err));
    std::exit(2);
}

template <typename T>
T *cuda_alloc(size_t n, const char *what)
{
    T *p = nullptr;
    cuda_check(cudaMalloc(&p, n * sizeof(T)), what);
    return p;
}

template <typename T>
void cuda_upload(T *dst, const T *src, size_t n, const char *what)
{
    cuda_check(cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyHostToDevice), what);
}

template <typename T>
void cuda_download(T *dst, const T *src, size_t n, const char *what)
{
    cuda_check(cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyDeviceToHost), what);
}

bool run_case(const CaseInput &in)
{
    const int nEle = 1;

    int h_iLake = 0;
    double h_uYsf = in.ysf;
    double h_uYus = in.yus;
    double h_uYgw = in.ygw;

    double h_satn_prev = in.satn_prev;

    double h_AquiferDepth = in.AquiferDepth;
    double h_infD = in.infD;
    double h_ThetaS = in.ThetaS;
    double h_ThetaR = in.ThetaR;
    double h_ThetaFC = in.ThetaS * FieldCapacityRatio;
    double h_Alpha = 1.0;
    double h_Beta = in.Beta;
    double h_hAreaF = 0.0;
    double h_macKsatV = 1.0e-6;
    double h_infKsatV = 1.0e-6;
    double h_KsatV = 1.0e-6;
    double h_macD = in.macD;
    double h_macKsatH = in.macKsatH;
    double h_geo_vAreaF = in.geo_vAreaF;
    double h_KsatH = in.KsatH;
    double h_VegFrac = in.VegFrac;
    double h_ImpAF = in.ImpAF;
    double h_RzD = in.RzD;

    double h_fu_surf = 1.0;
    double h_fu_sub = 1.0;
    double h_netprcp = 0.0;
    double h_qPotEvap = in.qPotEvap;
    double h_qPotTran = in.qPotTran;
    double h_lai = in.lai;
    double h_qEIC = in.qEIC;

    double h_qEleInfil = 123.0;
    double h_qEleExfil = 456.0;
    double h_qEleRecharge = 789.0;
    double h_qEs = -1.0;
    double h_qEu = -2.0;
    double h_qEg = -3.0;
    double h_qTu = -4.0;
    double h_qTg = -5.0;
    double h_effKH = -6.0;

    int *d_iLake = cuda_alloc<int>(nEle, "cudaMalloc(ele_iLake)");
    double *d_uYsf = cuda_alloc<double>(nEle, "cudaMalloc(uYsf)");
    double *d_uYus = cuda_alloc<double>(nEle, "cudaMalloc(uYus)");
    double *d_uYgw = cuda_alloc<double>(nEle, "cudaMalloc(uYgw)");
    double *d_satn = cuda_alloc<double>(nEle, "cudaMalloc(ele_satn)");
    double *d_effKH = cuda_alloc<double>(nEle, "cudaMalloc(ele_effKH)");

    double *d_AquiferDepth = cuda_alloc<double>(nEle, "cudaMalloc(ele_AquiferDepth)");
    double *d_infD = cuda_alloc<double>(nEle, "cudaMalloc(ele_infD)");
    double *d_ThetaS = cuda_alloc<double>(nEle, "cudaMalloc(ele_ThetaS)");
    double *d_ThetaR = cuda_alloc<double>(nEle, "cudaMalloc(ele_ThetaR)");
    double *d_ThetaFC = cuda_alloc<double>(nEle, "cudaMalloc(ele_ThetaFC)");
    double *d_Alpha = cuda_alloc<double>(nEle, "cudaMalloc(ele_Alpha)");
    double *d_Beta = cuda_alloc<double>(nEle, "cudaMalloc(ele_Beta)");
    double *d_hAreaF = cuda_alloc<double>(nEle, "cudaMalloc(ele_hAreaF)");
    double *d_macKsatV = cuda_alloc<double>(nEle, "cudaMalloc(ele_macKsatV)");
    double *d_infKsatV = cuda_alloc<double>(nEle, "cudaMalloc(ele_infKsatV)");
    double *d_KsatV = cuda_alloc<double>(nEle, "cudaMalloc(ele_KsatV)");
    double *d_macD = cuda_alloc<double>(nEle, "cudaMalloc(ele_macD)");
    double *d_macKsatH = cuda_alloc<double>(nEle, "cudaMalloc(ele_macKsatH)");
    double *d_geo_vAreaF = cuda_alloc<double>(nEle, "cudaMalloc(ele_geo_vAreaF)");
    double *d_KsatH = cuda_alloc<double>(nEle, "cudaMalloc(ele_KsatH)");
    double *d_VegFrac = cuda_alloc<double>(nEle, "cudaMalloc(ele_VegFrac)");
    double *d_ImpAF = cuda_alloc<double>(nEle, "cudaMalloc(ele_ImpAF)");
    double *d_RzD = cuda_alloc<double>(nEle, "cudaMalloc(ele_RzD)");

    double *d_fu_surf = cuda_alloc<double>(nEle, "cudaMalloc(fu_Surf)");
    double *d_fu_sub = cuda_alloc<double>(nEle, "cudaMalloc(fu_Sub)");
    double *d_netprcp = cuda_alloc<double>(nEle, "cudaMalloc(qEleNetPrep)");
    double *d_qPotEvap = cuda_alloc<double>(nEle, "cudaMalloc(qPotEvap)");
    double *d_qPotTran = cuda_alloc<double>(nEle, "cudaMalloc(qPotTran)");
    double *d_lai = cuda_alloc<double>(nEle, "cudaMalloc(t_lai)");
    double *d_qEIC = cuda_alloc<double>(nEle, "cudaMalloc(qEleE_IC)");

    double *d_qEleInfil = cuda_alloc<double>(nEle, "cudaMalloc(qEleInfil)");
    double *d_qEleExfil = cuda_alloc<double>(nEle, "cudaMalloc(qEleExfil)");
    double *d_qEleRecharge = cuda_alloc<double>(nEle, "cudaMalloc(qEleRecharge)");
    double *d_qEs = cuda_alloc<double>(nEle, "cudaMalloc(qEs)");
    double *d_qEu = cuda_alloc<double>(nEle, "cudaMalloc(qEu)");
    double *d_qEg = cuda_alloc<double>(nEle, "cudaMalloc(qEg)");
    double *d_qTu = cuda_alloc<double>(nEle, "cudaMalloc(qTu)");
    double *d_qTg = cuda_alloc<double>(nEle, "cudaMalloc(qTg)");

    cuda_upload(d_iLake, &h_iLake, nEle, "upload ele_iLake");
    cuda_upload(d_uYsf, &h_uYsf, nEle, "upload uYsf");
    cuda_upload(d_uYus, &h_uYus, nEle, "upload uYus");
    cuda_upload(d_uYgw, &h_uYgw, nEle, "upload uYgw");
    cuda_upload(d_satn, &h_satn_prev, nEle, "upload ele_satn(prev)");
    cuda_upload(d_effKH, &h_effKH, nEle, "upload ele_effKH(init)");

    cuda_upload(d_AquiferDepth, &h_AquiferDepth, nEle, "upload ele_AquiferDepth");
    cuda_upload(d_infD, &h_infD, nEle, "upload ele_infD");
    cuda_upload(d_ThetaS, &h_ThetaS, nEle, "upload ele_ThetaS");
    cuda_upload(d_ThetaR, &h_ThetaR, nEle, "upload ele_ThetaR");
    cuda_upload(d_ThetaFC, &h_ThetaFC, nEle, "upload ele_ThetaFC");
    cuda_upload(d_Alpha, &h_Alpha, nEle, "upload ele_Alpha");
    cuda_upload(d_Beta, &h_Beta, nEle, "upload ele_Beta");
    cuda_upload(d_hAreaF, &h_hAreaF, nEle, "upload ele_hAreaF");
    cuda_upload(d_macKsatV, &h_macKsatV, nEle, "upload ele_macKsatV");
    cuda_upload(d_infKsatV, &h_infKsatV, nEle, "upload ele_infKsatV");
    cuda_upload(d_KsatV, &h_KsatV, nEle, "upload ele_KsatV");
    cuda_upload(d_macD, &h_macD, nEle, "upload ele_macD");
    cuda_upload(d_macKsatH, &h_macKsatH, nEle, "upload ele_macKsatH");
    cuda_upload(d_geo_vAreaF, &h_geo_vAreaF, nEle, "upload ele_geo_vAreaF");
    cuda_upload(d_KsatH, &h_KsatH, nEle, "upload ele_KsatH");
    cuda_upload(d_VegFrac, &h_VegFrac, nEle, "upload ele_VegFrac");
    cuda_upload(d_ImpAF, &h_ImpAF, nEle, "upload ele_ImpAF");
    cuda_upload(d_RzD, &h_RzD, nEle, "upload ele_RzD");

    cuda_upload(d_fu_surf, &h_fu_surf, nEle, "upload fu_Surf");
    cuda_upload(d_fu_sub, &h_fu_sub, nEle, "upload fu_Sub");
    cuda_upload(d_netprcp, &h_netprcp, nEle, "upload qEleNetPrep");
    cuda_upload(d_qPotEvap, &h_qPotEvap, nEle, "upload qPotEvap");
    cuda_upload(d_qPotTran, &h_qPotTran, nEle, "upload qPotTran");
    cuda_upload(d_lai, &h_lai, nEle, "upload t_lai");
    cuda_upload(d_qEIC, &h_qEIC, nEle, "upload qEleE_IC");

    cuda_upload(d_qEleInfil, &h_qEleInfil, nEle, "upload qEleInfil(init)");
    cuda_upload(d_qEleExfil, &h_qEleExfil, nEle, "upload qEleExfil(init)");
    cuda_upload(d_qEleRecharge, &h_qEleRecharge, nEle, "upload qEleRecharge(init)");
    cuda_upload(d_qEs, &h_qEs, nEle, "upload qEs(init)");
    cuda_upload(d_qEu, &h_qEu, nEle, "upload qEu(init)");
    cuda_upload(d_qEg, &h_qEg, nEle, "upload qEg(init)");
    cuda_upload(d_qTu, &h_qTu, nEle, "upload qTu(init)");
    cuda_upload(d_qTg, &h_qTg, nEle, "upload qTg(init)");

    DeviceModel h_model{};
    h_model.NumEle = nEle;
    h_model.NumRiv = 0;
    h_model.NumSeg = 0;
    h_model.NumLake = 0;
    h_model.ele_iLake = d_iLake;
    h_model.uYsf = d_uYsf;
    h_model.uYus = d_uYus;
    h_model.uYgw = d_uYgw;
    h_model.ele_satn = d_satn;
    h_model.ele_effKH = d_effKH;

    h_model.ele_AquiferDepth = d_AquiferDepth;
    h_model.ele_infD = d_infD;
    h_model.ele_ThetaS = d_ThetaS;
    h_model.ele_ThetaR = d_ThetaR;
    h_model.ele_ThetaFC = d_ThetaFC;
    h_model.ele_Alpha = d_Alpha;
    h_model.ele_Beta = d_Beta;
    h_model.ele_hAreaF = d_hAreaF;
    h_model.ele_macKsatV = d_macKsatV;
    h_model.ele_infKsatV = d_infKsatV;
    h_model.ele_KsatV = d_KsatV;
    h_model.ele_macD = d_macD;
    h_model.ele_macKsatH = d_macKsatH;
    h_model.ele_geo_vAreaF = d_geo_vAreaF;
    h_model.ele_KsatH = d_KsatH;
    h_model.ele_VegFrac = d_VegFrac;
    h_model.ele_ImpAF = d_ImpAF;
    h_model.ele_RzD = d_RzD;

    h_model.fu_Surf = d_fu_surf;
    h_model.fu_Sub = d_fu_sub;
    h_model.qEleNetPrep = d_netprcp;
    h_model.qPotEvap = d_qPotEvap;
    h_model.qPotTran = d_qPotTran;
    h_model.t_lai = d_lai;
    h_model.qEleE_IC = d_qEIC;

    h_model.qEleInfil = d_qEleInfil;
    h_model.qEleExfil = d_qEleExfil;
    h_model.qEleRecharge = d_qEleRecharge;
    h_model.qEs = d_qEs;
    h_model.qEu = d_qEu;
    h_model.qEg = d_qEg;
    h_model.qTu = d_qTu;
    h_model.qTg = d_qTg;

    DeviceModel *d_model = nullptr;
    cuda_check(cudaMalloc(&d_model, sizeof(DeviceModel)), "cudaMalloc(DeviceModel)");
    cuda_check(cudaMemcpy(d_model, &h_model, sizeof(DeviceModel), cudaMemcpyHostToDevice),
               "upload DeviceModel");

    k_ele_local<<<1, 256>>>(d_model);
    cuda_check(cudaGetLastError(), "launch k_ele_local");
    cuda_check(cudaDeviceSynchronize(), "sync after k_ele_local");

    cuda_download(&h_qEs, d_qEs, nEle, "download qEs");
    cuda_download(&h_qEu, d_qEu, nEle, "download qEu");
    cuda_download(&h_qEg, d_qEg, nEle, "download qEg");
    cuda_download(&h_qTu, d_qTu, nEle, "download qTu");
    cuda_download(&h_qTg, d_qTg, nEle, "download qTg");
    cuda_download(&h_satn_prev, d_satn, nEle, "download ele_satn(new)");
    cuda_download(&h_effKH, d_effKH, nEle, "download ele_effKH");

    const EtOut expect = host_reference(in.ysf,
                                        in.yus,
                                        in.ygw,
                                        in.satn_prev,
                                        in.AquiferDepth,
                                        in.infD,
                                        in.ThetaS,
                                        in.ThetaR,
                                        in.Beta,
                                        in.macD,
                                        in.macKsatH,
                                        in.geo_vAreaF,
                                        in.KsatH,
                                        in.VegFrac,
                                        in.ImpAF,
                                        in.RzD,
                                        in.qPotEvap,
                                        in.qPotTran,
                                        in.lai,
                                        in.qEIC);

    const double atol = 1.0e-12;
    const double rtol = 1.0e-10;
    bool ok = true;
    ok &= nearly_equal(h_qEs, expect.qEs, atol, rtol);
    ok &= nearly_equal(h_qEu, expect.qEu, atol, rtol);
    ok &= nearly_equal(h_qEg, expect.qEg, atol, rtol);
    ok &= nearly_equal(h_qTu, expect.qTu, atol, rtol);
    ok &= nearly_equal(h_qTg, expect.qTg, atol, rtol);
    ok &= nearly_equal(h_satn_prev, expect.satn_new, atol, rtol);
    ok &= nearly_equal(h_effKH, expect.effKH_val, atol, rtol);

    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", in.name);
        std::fprintf(stderr, "  got:    qEs=%.17g qEu=%.17g qEg=%.17g qTu=%.17g qTg=%.17g satn=%.17g effKH=%.17g\n",
                     h_qEs,
                     h_qEu,
                     h_qEg,
                     h_qTu,
                     h_qTg,
                     h_satn_prev,
                     h_effKH);
        std::fprintf(stderr, "  expect: qEs=%.17g qEu=%.17g qEg=%.17g qTu=%.17g qTg=%.17g satn=%.17g effKH=%.17g\n",
                     expect.qEs,
                     expect.qEu,
                     expect.qEg,
                     expect.qTu,
                     expect.qTg,
                     expect.satn_new,
                     expect.effKH_val);
    }

    cudaFree(d_model);
    cudaFree(d_iLake);
    cudaFree(d_uYsf);
    cudaFree(d_uYus);
    cudaFree(d_uYgw);
    cudaFree(d_satn);
    cudaFree(d_effKH);

    cudaFree(d_AquiferDepth);
    cudaFree(d_infD);
    cudaFree(d_ThetaS);
    cudaFree(d_ThetaR);
    cudaFree(d_ThetaFC);
    cudaFree(d_Alpha);
    cudaFree(d_Beta);
    cudaFree(d_hAreaF);
    cudaFree(d_macKsatV);
    cudaFree(d_infKsatV);
    cudaFree(d_KsatV);
    cudaFree(d_macD);
    cudaFree(d_macKsatH);
    cudaFree(d_geo_vAreaF);
    cudaFree(d_KsatH);
    cudaFree(d_VegFrac);
    cudaFree(d_ImpAF);
    cudaFree(d_RzD);

    cudaFree(d_fu_surf);
    cudaFree(d_fu_sub);
    cudaFree(d_netprcp);
    cudaFree(d_qPotEvap);
    cudaFree(d_qPotTran);
    cudaFree(d_lai);
    cudaFree(d_qEIC);

    cudaFree(d_qEleInfil);
    cudaFree(d_qEleExfil);
    cudaFree(d_qEleRecharge);
    cudaFree(d_qEs);
    cudaFree(d_qEu);
    cudaFree(d_qEg);
    cudaFree(d_qTu);
    cudaFree(d_qTg);

    return ok;
}

} // namespace

int main()
{
    int ndev = 0;
    const cudaError_t dev_err = cudaGetDeviceCount(&ndev);
    if (dev_err != cudaSuccess || ndev <= 0) {
        std::fprintf(stderr, "SKIP: no CUDA device found.\n");
        return 0;
    }

    std::vector<CaseInput> cases;
    {
        CaseInput c{};
        c.name = "valid_satn_prev";
        c.ysf = 0.0;
        c.yus = 0.02;
        c.ygw = 0.3;
        c.satn_prev = 0.5;
        c.qPotEvap = 1.0e-6;
        c.qPotTran = 2.0e-6;
        c.lai = 2.0;
        c.qEIC = 0.0;
        cases.push_back(c);
    }
    {
        CaseInput c{};
        c.name = "nan_satn_prev_fallback";
        c.ysf = 0.0;
        c.yus = 0.02;
        c.ygw = 0.3;
        c.satn_prev = std::numeric_limits<double>::quiet_NaN();
        c.qPotEvap = 1.0e-6;
        c.qPotTran = 2.0e-6;
        c.lai = 2.0;
        c.qEIC = 0.0;
        cases.push_back(c);
    }

    bool ok = true;
    for (const auto &c : cases) {
        ok &= run_case(c);
    }

    if (!ok) {
        std::fprintf(stderr, "==> FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "==> OK\n");
    return 0;
}
