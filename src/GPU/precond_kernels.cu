#include "precond_kernels.hpp"

#ifdef _CUDA_ON

#include "DeviceContext.hpp"
#include "Macros.hpp"
#include "Model_Data.hpp"
#include "Nvtx.hpp"

#include <cuda_runtime.h>

#include <cmath>

namespace {

__device__ inline double d_min(double a, double b) { return a < b ? a : b; }
__device__ inline double d_max(double a, double b) { return a > b ? a : b; }
__device__ inline double d_clamp_nonneg(double x) { return x >= 0.0 ? x : 0.0; }

__device__ inline double meanHarmonic(double k1, double k2, double d1, double d2)
{
    return (k1 * k2) * (d1 + d2) / (d1 * k2 + d2 * k1);
}

__device__ inline double satKfun(double elemSatn, double beta)
{
    const double temp = -1.0 + pow(1.0 - pow(elemSatn, beta / (beta - 1.0)), (beta - 1.0) / beta);
    return sqrt(elemSatn) * temp * temp;
}

__device__ inline double soilMoistureStress(double ThetaS, double ThetaR, double SatRatio)
{
    const double fc = ThetaS * FieldCapacityRatio;
    double beta_s = (SatRatio * (ThetaS - ThetaR) - ThetaR) / (fc - ThetaR);
    beta_s = d_min(d_max(beta_s, 0.0), 1.0);
    beta_s = 0.5 * (1.0 - cos(PI * beta_s));
    return beta_s;
}

__device__ inline double fun_CrossArea(double y, double w0, double s) { return y * (w0 + y * s); }

__device__ inline double fun_TopWidth(double y, double w0, double s) { return y * s * 2.0 + w0; }

__device__ inline double quadratic(double s, double w, double dA)
{
    const double ss = fabs(s);
    const double cc = w * w + 4.0 * ss * dA;
    if (cc < ZERO) {
        return -1.0 * w / (2.0 * ss);
    }
    return (-w + sqrt(cc)) / (2.0 * ss);
}

__device__ inline double fun_dAtodY(double dA, double w_top, double s)
{
    if (dA == 0.0) {
        return 0.0;
    }
    if (fabs(s) < EPS_SLOPE) {
        if (fabs(w_top) < ZERO) {
            return 0.0;
        }
        return dA / w_top;
    }
    return quadratic(s, w_top, dA);
}

__device__ inline double lake_toparea(const DeviceModel *m, int lake_idx, double y_abs)
{
    const int n = (m->lake_bathy_n != nullptr) ? m->lake_bathy_n[lake_idx] : 0;
    if (n <= 0 || m->lake_bathy_off == nullptr || m->bathy_yi == nullptr || m->bathy_ai == nullptr) {
        return 0.0;
    }
    const int off = m->lake_bathy_off[lake_idx];
    double ta = m->bathy_ai[off];
    if (y_abs <= m->bathy_yi[off]) {
        return ta;
    }
    for (int i = 1; i < n; i++) {
        const double yi_i = m->bathy_yi[off + i];
        if (y_abs < yi_i) {
            const double da = m->bathy_ai[off + i] - ta;
            const double dy = yi_i - y_abs;
            ta = da / dy * (y_abs - m->bathy_yi[off + i - 1]) + ta;
            break;
        }
        ta = m->bathy_ai[off + i];
    }
    return ta;
}

__device__ inline double river_local_rhs(const DeviceModel *m, int i, double yriv, bool dirichlet)
{
    if (dirichlet) {
        return 0.0;
    }

    const double w0 = (m->riv_BottomWidth != nullptr) ? m->riv_BottomWidth[i] : 0.0;
    const double bs = (m->riv_BankSlope != nullptr) ? m->riv_BankSlope[i] : 0.0;
    double topWidth = fun_TopWidth(yriv, w0, bs);
    double CSarea = fun_CrossArea(yriv, w0, bs);
    topWidth = d_max(0.0, topWidth);
    CSarea = d_max(0.0, CSarea);

    const double Qup = (m->QrivUp != nullptr) ? m->QrivUp[i] : 0.0;
    const double Qsurf = (m->QrivSurf != nullptr) ? m->QrivSurf[i] : 0.0;
    const double Qsub = (m->QrivSub != nullptr) ? m->QrivSub[i] : 0.0;
    const double Qdown = (m->QrivDown != nullptr) ? m->QrivDown[i] : 0.0;
    const double qBC = (m->riv_qBC != nullptr) ? m->riv_qBC[i] : 0.0;
    const double L = (m->riv_Length != nullptr) ? m->riv_Length[i] : 1.0;
    if (fabs(L) < ZERO) {
        return 0.0;
    }

    double dA = (-Qup - Qsurf - Qsub - Qdown + qBC) / L;
    if (dA < -1.0 * CSarea) {
        dA = -1.0 * CSarea;
    }
    return fun_dAtodY(dA, topWidth, bs);
}

__device__ inline double lake_local_rhs(const DeviceModel *m, int i, double yStage)
{
    const double zmin = (m->lake_zmin != nullptr) ? m->lake_zmin[i] : 0.0;
    const double area = lake_toparea(m, i, yStage + zmin);
    if (area <= ZERO) {
        return 0.0;
    }

    const double prcp = (m->qLakePrcp != nullptr) ? m->qLakePrcp[i] : 0.0;
    double evap = (m->qLakeEvap != nullptr) ? m->qLakeEvap[i] : 0.0;
    evap = d_min(evap, prcp + yStage);
    evap = d_max(0.0, evap);

    const double Qin = (m->QLakeRivIn != nullptr) ? m->QLakeRivIn[i] : 0.0;
    const double Qout = (m->QLakeRivOut != nullptr) ? m->QLakeRivOut[i] : 0.0;
    const double Qsub = (m->QLakeSub != nullptr) ? m->QLakeSub[i] : 0.0;
    const double Qsurf = (m->QLakeSurf != nullptr) ? m->QLakeSurf[i] : 0.0;

    return prcp - evap + (Qin - Qout + Qsub + Qsurf) / area;
}

__device__ inline void element_local_rhs(const DeviceModel *m,
                                         int i,
                                         double ysf,
                                         double yus,
                                         double ygw,
                                         bool gw_dirichlet,
                                         double *f_sf,
                                         double *f_us,
                                         double *f_gw)
{
    if (gw_dirichlet && m->ele_yBC != nullptr) {
        ygw = m->ele_yBC[i];
    }

    const double AquiferDepth = (m->ele_AquiferDepth != nullptr) ? m->ele_AquiferDepth[i] : 0.0;
    const double Sy = (m->ele_Sy != nullptr) ? m->ele_Sy[i] : 1.0;
    const double infD = (m->ele_infD != nullptr) ? m->ele_infD[i] : 0.0;
    const double infKsatV = (m->ele_infKsatV != nullptr) ? m->ele_infKsatV[i] : 0.0;
    const double ThetaS = (m->ele_ThetaS != nullptr) ? m->ele_ThetaS[i] : 0.0;
    const double ThetaR = (m->ele_ThetaR != nullptr) ? m->ele_ThetaR[i] : 0.0;
    const double ThetaFC = (m->ele_ThetaFC != nullptr) ? m->ele_ThetaFC[i] : 0.0;
    const double Beta = (m->ele_Beta != nullptr) ? m->ele_Beta[i] : 0.0;
    const double hAreaF = (m->ele_hAreaF != nullptr) ? m->ele_hAreaF[i] : 0.0;
    const double macKsatV = (m->ele_macKsatV != nullptr) ? m->ele_macKsatV[i] : 0.0;
    const double KsatV = (m->ele_KsatV != nullptr) ? m->ele_KsatV[i] : 0.0;
    const double VegFrac = (m->ele_VegFrac != nullptr) ? m->ele_VegFrac[i] : 0.0;
    const double ImpAF = (m->ele_ImpAF != nullptr) ? m->ele_ImpAF[i] : 0.0;
    const double RzD = (m->ele_RzD != nullptr) ? m->ele_RzD[i] : 0.0;

    const double fu_surf = (m->fu_Surf != nullptr) ? m->fu_Surf[i] : 1.0;
    const double fu_sub = (m->fu_Sub != nullptr) ? m->fu_Sub[i] : 1.0;
    const double netprcp = (m->qEleNetPrep != nullptr) ? m->qEleNetPrep[i] : 0.0;

    const double qPotEvap = (m->qPotEvap != nullptr) ? m->qPotEvap[i] : 0.0;
    const double qPotTran = (m->qPotTran != nullptr) ? m->qPotTran[i] : 0.0;
    const double lai = (m->t_lai != nullptr) ? m->t_lai[i] : 0.0;
    const double qEIC = (m->qEleE_IC != nullptr) ? m->qEleE_IC[i] : 0.0;

    const double Kmax = infKsatV * (1.0 - hAreaF) + macKsatV * hAreaF;

    /* updateElement (satn_new / theta / satKr) */
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
        satKr = satKfun(satn_new, Beta);
    }

    /* f_etFlux */
    const double iBeta = soilMoistureStress(ThetaS, ThetaR, satn_new);
    const double va = VegFrac;
    const double vb = 1.0 - VegFrac;
    const double pj = 1.0 - ImpAF;

    const double WetlandLevel = AquiferDepth - infD;
    const double RootReachLevel = AquiferDepth - RzD;

    double Es = d_min(d_max(0.0, ysf), qPotEvap) * vb;
    double Eu = 0.0;
    double Eg = 0.0;
    if (Es < qPotEvap) {
        if (ygw > WetlandLevel) {
            Eg = d_min(d_max(0.0, ygw), qPotEvap - Es) * pj * vb;
            Eu = 0.0;
        } else {
            Eg = 0.0;
            Eu = d_min(d_max(0.0, yus), iBeta * (qPotEvap - Es)) * pj * vb;
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
                Tg = d_min(d_max(0.0, ygw), (qPotTran - qEIC)) * pj * va;
                Tu = 0.0;
            } else {
                Tg = 0.0;
                Tu = d_min(d_max(0.0, yus), iBeta * (qPotTran - qEIC)) * pj * va;
            }
        }
    }

    /* Flux_Infiltration */
    double qi = 0.0;
    double qex = 0.0;
    const double av = ysf + netprcp;
    if (AquiferDepth > ZERO && (ygw + yus > AquiferDepth || deficit < yus)) {
        qex = fabs(ygw + yus - AquiferDepth) / AquiferDepth * Kmax;
        qi = 0.0;
    } else {
        qex = 0.0;
        if (av > 0.0 && deficit > infD && infD > ZERO) {
            const double grad = 1.0 + av / infD;
            double effkInfi = 0.0;
            if (av > Kmax) {
                effkInfi = infKsatV * (1.0 - hAreaF) + hAreaF * macKsatV * satn_new;
            } else if (av > infKsatV) {
                effkInfi = satKr * infKsatV * (1.0 - hAreaF) + hAreaF * macKsatV * satn_new;
            } else {
                effkInfi = satKr * infKsatV * (1.0 - hAreaF);
            }
            qi = grad * effkInfi;
            qi = d_min(av, d_max(0.0, qi));
        }
    }

    const double qin = qi * fu_surf;
    const double qexf = qex * fu_surf;

    /* Flux_Recharge */
    double qr = 0.0;
    if (!(ygw > AquiferDepth - infD && yus < deficit)) {
        double grad = 0.0;
        if (theta > ThetaR) {
            if (yus <= EPSILON) {
                grad = 0.0;
            } else {
                grad = (ThetaFC - ThetaR > 0.0) ? ((theta - ThetaR) / (ThetaFC - ThetaR)) : 0.0;
                grad = d_max(grad, 0.0);
            }
        }
        if (infKsatV > 0.0 && KsatV > 0.0) {
            const double ku = infKsatV * satKr;
            const double ke = meanHarmonic(ku, KsatV, deficit, ygw);
            qr = grad * ke;
        }
    }

    const double qrf = qr * fu_sub;

    double DYsf = netprcp - qin + qexf - Es;
    double DYus = qin - qrf - Eu - Tu;
    double DYgw = qrf - qexf - Eg - Tg;

    const double invSy = (Sy > ZERO) ? (1.0 / Sy) : 1.0;
    DYus *= invSy;
    DYgw *= invSy;

    if (gw_dirichlet) {
        DYgw = 0.0;
    }

    *f_sf = DYsf;
    *f_us = DYus;
    *f_gw = DYgw;
}

__device__ inline bool invert3x3(const double *A, double *invA)
{
    const double a00 = A[0];
    const double a01 = A[1];
    const double a02 = A[2];
    const double a10 = A[3];
    const double a11 = A[4];
    const double a12 = A[5];
    const double a20 = A[6];
    const double a21 = A[7];
    const double a22 = A[8];

    const double c00 = a11 * a22 - a12 * a21;
    const double c01 = -(a10 * a22 - a12 * a20);
    const double c02 = a10 * a21 - a11 * a20;

    const double det = a00 * c00 + a01 * c01 + a02 * c02;
    const double abs_det = fabs(det);
    if (det != det || abs_det < 1e-24 || abs_det > 1e300) {
        return false;
    }

    const double inv_det = 1.0 / det;

    invA[0] = c00 * inv_det;
    invA[1] = (-(a01 * a22 - a02 * a21)) * inv_det;
    invA[2] = (a01 * a12 - a02 * a11) * inv_det;

    invA[3] = c01 * inv_det;
    invA[4] = (a00 * a22 - a02 * a20) * inv_det;
    invA[5] = (-(a00 * a12 - a02 * a10)) * inv_det;

    invA[6] = c02 * inv_det;
    invA[7] = (-(a00 * a21 - a01 * a20)) * inv_det;
    invA[8] = (a00 * a11 - a01 * a10) * inv_det;

    return true;
}

__device__ inline void write_identity3(double *dst)
{
    dst[0] = 1.0;
    dst[1] = 0.0;
    dst[2] = 0.0;
    dst[3] = 0.0;
    dst[4] = 1.0;
    dst[5] = 0.0;
    dst[6] = 0.0;
    dst[7] = 0.0;
    dst[8] = 1.0;
}

__global__ void k_psetup(const realtype *dY, const DeviceModel *m, double gamma, int clamp_policy)
{
    if (dY == nullptr || m == nullptr || m->prec_inv == nullptr) {
        return;
    }

    const int nEle = m->NumEle;
    const int nRiv = m->NumRiv;
    const int nLake = m->NumLake;
    const int nBlocks = nEle + nRiv + nLake;

    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < nBlocks; b += blockDim.x * gridDim.x) {
        if (b < nEle) {
            const int i = b;
            const int iLake = (m->ele_iLake != nullptr) ? m->ele_iLake[i] : 0;
            double *inv = &m->prec_inv[static_cast<size_t>(i) * 9u];

            if (iLake > 0) {
                write_identity3(inv);
                continue;
            }

            const int bc = (m->ele_iBC != nullptr) ? m->ele_iBC[i] : 0;
            const bool gw_dirichlet = (bc > 0) && (m->ele_yBC != nullptr);

            double ysf = static_cast<double>(dY[i]);
            double yus = static_cast<double>(dY[i + nEle]);
            double ygw = static_cast<double>(dY[i + 2 * nEle]);

            if (clamp_policy) {
                ysf = d_clamp_nonneg(ysf);
                yus = d_clamp_nonneg(yus);
                ygw = d_clamp_nonneg(ygw);
            }

            if (gw_dirichlet) {
                ygw = m->ele_yBC[i];
            }

            double f_base[3];
            element_local_rhs(m, i, ysf, yus, ygw, gw_dirichlet, &f_base[0], &f_base[1], &f_base[2]);

            /* Finite-difference Jacobian (column-wise). */
            double J[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            const double y_base[3] = {ysf, yus, ygw};
            for (int col = 0; col < 3; col++) {
                const double yj = y_base[col];
                const double del = 1.0e-8 * (fabs(yj) + 1.0);
                double yp[3] = {y_base[0], y_base[1], y_base[2]};
                yp[col] += del;
                double f_pert[3];
                element_local_rhs(m, i, yp[0], yp[1], yp[2], gw_dirichlet, &f_pert[0], &f_pert[1], &f_pert[2]);
                const double inv_del = 1.0 / del;
                J[0 * 3 + col] = (f_pert[0] - f_base[0]) * inv_del;
                J[1 * 3 + col] = (f_pert[1] - f_base[1]) * inv_del;
                J[2 * 3 + col] = (f_pert[2] - f_base[2]) * inv_del;
            }

            /* M = I - gamma * J */
            double M[9];
            M[0] = 1.0 - gamma * J[0];
            M[1] = 0.0 - gamma * J[1];
            M[2] = 0.0 - gamma * J[2];
            M[3] = 0.0 - gamma * J[3];
            M[4] = 1.0 - gamma * J[4];
            M[5] = 0.0 - gamma * J[5];
            M[6] = 0.0 - gamma * J[6];
            M[7] = 0.0 - gamma * J[7];
            M[8] = 1.0 - gamma * J[8];

            double invM[9];
            if (!invert3x3(M, invM)) {
                write_identity3(inv);
                continue;
            }

            for (int k = 0; k < 9; k++) {
                inv[k] = invM[k];
            }
            continue;
        }

        const int off = b - nEle;
        const size_t base = static_cast<size_t>(nEle) * 9u;
        if (off < nRiv) {
            const int i = off;
            const int bc = (m->riv_BC != nullptr) ? m->riv_BC[i] : 0;
            const bool dirichlet = (bc > 0) && (m->riv_yBC != nullptr);
            const int idx = 3 * nEle + i;

            double yriv = static_cast<double>(dY[idx]);
            if (clamp_policy) {
                yriv = d_clamp_nonneg(yriv);
            }
            if (dirichlet) {
                yriv = m->riv_yBC[i];
            }

            const double f_base = river_local_rhs(m, i, yriv, dirichlet);
            const double del = 1.0e-8 * (fabs(yriv) + 1.0);
            const double f_pert = river_local_rhs(m, i, yriv + del, dirichlet);
            const double J = (f_pert - f_base) / del;

            const double M = 1.0 - gamma * J;
            double inv = 1.0;
            if (M == M && fabs(M) > 1e-24) {
                inv = 1.0 / M;
            }
            m->prec_inv[base + static_cast<size_t>(i)] = inv;
        } else {
            const int l = off - nRiv;
            if (l >= 0 && l < nLake) {
                const int idx = 3 * nEle + nRiv + l;
                double yStage = static_cast<double>(dY[idx]);
                if (clamp_policy) {
                    yStage = d_clamp_nonneg(yStage);
                }

                const double f_base = lake_local_rhs(m, l, yStage);
                const double del = 1.0e-8 * (fabs(yStage) + 1.0);
                const double f_pert = lake_local_rhs(m, l, yStage + del);
                const double J = (f_pert - f_base) / del;

                const double M = 1.0 - gamma * J;
                double inv = 1.0;
                if (M == M && fabs(M) > 1e-24) {
                    inv = 1.0 / M;
                }
                m->prec_inv[base + static_cast<size_t>(nRiv) + static_cast<size_t>(l)] = inv;
            }
        }
    }
}

__global__ void k_psolve(const realtype *r, realtype *z, const DeviceModel *m)
{
    if (r == nullptr || z == nullptr || m == nullptr || m->prec_inv == nullptr) {
        return;
    }

    const int nEle = m->NumEle;
    const int nRiv = m->NumRiv;
    const int nLake = m->NumLake;
    const int nBlocks = nEle + nRiv + nLake;
    const size_t inv_base = static_cast<size_t>(nEle) * 9u;

    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < nBlocks; b += blockDim.x * gridDim.x) {
        if (b < nEle) {
            const int i = b;
            const double *inv = &m->prec_inv[static_cast<size_t>(i) * 9u];

            const double r0 = static_cast<double>(r[i]);
            const double r1 = static_cast<double>(r[i + nEle]);
            const double r2 = static_cast<double>(r[i + 2 * nEle]);

            const double z0 = inv[0] * r0 + inv[1] * r1 + inv[2] * r2;
            const double z1 = inv[3] * r0 + inv[4] * r1 + inv[5] * r2;
            const double z2 = inv[6] * r0 + inv[7] * r1 + inv[8] * r2;

            z[i] = static_cast<realtype>(z0);
            z[i + nEle] = static_cast<realtype>(z1);
            z[i + 2 * nEle] = static_cast<realtype>(z2);
            continue;
        }

        const int off = b - nEle;
        const int base_y = 3 * nEle;
        if (off < nRiv) {
            const double inv_s = m->prec_inv[inv_base + static_cast<size_t>(off)];
            const int idx = base_y + off;
            z[idx] = static_cast<realtype>(inv_s * static_cast<double>(r[idx]));
        } else {
            const int l = off - nRiv;
            if (l >= 0 && l < nLake) {
                const double inv_s = m->prec_inv[inv_base + static_cast<size_t>(nRiv) + static_cast<size_t>(l)];
                const int idx = base_y + nRiv + l;
                z[idx] = static_cast<realtype>(inv_s * static_cast<double>(r[idx]));
            }
        }
    }
}

inline int cap_blocks(int blocks) { return (blocks > 65535) ? 65535 : blocks; }

} // namespace

int PSetup_cuda(realtype t,
                N_Vector y,
                N_Vector fy,
                booleantype jok,
                booleantype *jcurPtr,
                realtype gamma,
                void *user_data)
{
    (void)t;
    (void)fy;
    (void)jok;

    shud_nvtx::scoped_range range("PSetup_cuda");
    Model_Data *md = static_cast<Model_Data *>(user_data);
    if (md == nullptr || md->d_model == nullptr || md->h_model == nullptr || md->h_model->prec_inv == nullptr) {
        return -1;
    }

    if (N_VGetVectorID(y) != SUNDIALS_NVEC_CUDA) {
        fprintf(stderr, "ERROR: PSetup_cuda requires NVECTOR_CUDA for y.\n");
        return -1;
    }

    const realtype *dY = N_VGetDeviceArrayPointer_Cuda(y);
    if (dY == nullptr) {
        fprintf(stderr, "ERROR: PSetup_cuda: N_VGetDeviceArrayPointer_Cuda(y) returned NULL.\n");
        return -1;
    }

    const cudaStream_t stream = N_VGetCudaStream_Cuda(y);
    if (md->cuda_stream != stream && md->forcing_copy_event != nullptr && md->nGpuForcingCopy > 0) {
        const cudaError_t err = cudaStreamWaitEvent(stream, md->forcing_copy_event, 0);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "CUDA_ERROR: PSetup_cuda: cudaStreamWaitEvent(forcing_copy_event) failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
    }

    const int nBlocks = md->h_model->NumEle + md->h_model->NumRiv + md->h_model->NumLake;
    if (nBlocks > 0) {
        constexpr int kBlockSize = 256;
        const int blocks = cap_blocks((nBlocks + kBlockSize - 1) / kBlockSize);
        const int clamp_policy = CLAMP_POLICY;
        k_psetup<<<blocks, kBlockSize, 0, stream>>>(dY, md->d_model, static_cast<double>(gamma), clamp_policy);
        const cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA_ERROR: PSetup_cuda: kernel launch failed: %s\n", cudaGetErrorString(err));
            return -1;
        }
    }

    if (jcurPtr != nullptr) {
        *jcurPtr = SUNTRUE;
    }
    return 0;
}

int PSolve_cuda(realtype t,
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
    (void)gamma;
    (void)delta;
    (void)lr;

    shud_nvtx::scoped_range range("PSolve_cuda");
    Model_Data *md = static_cast<Model_Data *>(user_data);
    if (md == nullptr || md->d_model == nullptr || md->h_model == nullptr || md->h_model->prec_inv == nullptr) {
        return -1;
    }

    if (N_VGetVectorID(r) != SUNDIALS_NVEC_CUDA || N_VGetVectorID(z) != SUNDIALS_NVEC_CUDA) {
        fprintf(stderr, "ERROR: PSolve_cuda requires NVECTOR_CUDA for r and z.\n");
        return -1;
    }

    const realtype *dR = N_VGetDeviceArrayPointer_Cuda(r);
    realtype *dZ = N_VGetDeviceArrayPointer_Cuda(z);
    if (dR == nullptr || dZ == nullptr) {
        fprintf(stderr, "ERROR: PSolve_cuda: N_VGetDeviceArrayPointer_Cuda returned NULL.\n");
        return -1;
    }

    const cudaStream_t stream = N_VGetCudaStream_Cuda(z);
    const cudaStream_t r_stream = N_VGetCudaStream_Cuda(r);
    if (r_stream != stream) {
        const cudaError_t err = cudaStreamSynchronize(r_stream);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "CUDA_ERROR: PSolve_cuda: cudaStreamSynchronize(r_stream) failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
    }

    if (md->cuda_stream != stream && md->forcing_copy_event != nullptr && md->nGpuForcingCopy > 0) {
        const cudaError_t err = cudaStreamWaitEvent(stream, md->forcing_copy_event, 0);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "CUDA_ERROR: PSolve_cuda: cudaStreamWaitEvent(forcing_copy_event) failed: %s\n",
                    cudaGetErrorString(err));
            return -1;
        }
    }

    const int nBlocks = md->h_model->NumEle + md->h_model->NumRiv + md->h_model->NumLake;
    if (nBlocks > 0) {
        constexpr int kBlockSize = 256;
        const int blocks = cap_blocks((nBlocks + kBlockSize - 1) / kBlockSize);
        k_psolve<<<blocks, kBlockSize, 0, stream>>>(dR, dZ, md->d_model);
        const cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA_ERROR: PSolve_cuda: kernel launch failed: %s\n", cudaGetErrorString(err));
            return -1;
        }
    }

    return 0;
}

#endif /* _CUDA_ON */
