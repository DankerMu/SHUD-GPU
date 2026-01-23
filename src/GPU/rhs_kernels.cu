#include "rhs_kernels.hpp"

#ifdef _CUDA_ON

#include "DeviceContext.hpp"
#include "Macros.hpp"

#include <cuda_runtime.h>

#include <cmath>

namespace {

__device__ inline double d_min(double a, double b) { return a < b ? a : b; }
__device__ inline double d_max(double a, double b) { return a > b ? a : b; }
__device__ inline double d_clamp_nonneg(double x) { return x >= 0.0 ? x : 0.0; }

__device__ inline double pow23(double x)
{
    const double t = cbrt(x);
    return t * t;
}

__device__ inline double meanHarmonic(double k1, double k2, double d1, double d2)
{
    return (k1 * k2) * (d1 + d2) / (d1 * k2 + d2 * k1);
}

__device__ inline double meanArithmetic(double k1, double k2, double d1, double d2)
{
    return (k1 * d1 + k2 * d2) / (d1 + d2);
}

__device__ inline double manningEquation(double Area, double rough, double R, double S)
{
    if (S > 0.0) {
        return sqrt(S) * Area * pow23(R) / rough;
    }
    return -1.0 * sqrt(-S) * Area * pow23(R) / rough;
}

__device__ inline double avgY_sf(double z1, double y1, double z2, double y2, double threshold)
{
    const double h1 = z1 + y1;
    const double h2 = z2 + y2;
    if (h1 > h2) {
        return (y1 > threshold) ? y1 : 0.0;
    }
    return (y2 > threshold) ? y2 : 0.0;
}

__device__ inline double avgY_gw(double /*z1*/, double y1, double /*z2*/, double y2, double /*threshold*/)
{
    y1 = d_max(y1, 0.0);
    y2 = d_max(y2, 0.0);
    return 0.5 * (y1 + y2);
}

__device__ inline double satKfun(double elemSatn, double beta)
{
    const double temp = -1.0 + pow(1.0 - pow(elemSatn, beta / (beta - 1.0)), (beta - 1.0) / beta);
    return sqrt(elemSatn) * temp * temp;
}

__device__ inline double sat2psi(double elemSatn, double alpha, double beta)
{
    return -(pow(pow(elemSatn, beta / (1.0 - beta)) - 1.0, 1.0 / beta) / alpha);
}

__device__ inline double effKH(double Ygw, double aqDepth, double MacD, double Kmac, double AF, double Kmx)
{
    double effk = 0.0;
    if (MacD <= ZERO || Ygw < aqDepth - MacD) {
        effk = Kmx;
    } else {
        if (Ygw > aqDepth) {
            effk = (Kmac * MacD * AF + Kmx * (aqDepth - MacD * AF)) / aqDepth;
        } else {
            effk = (Kmac * (Ygw - (aqDepth - MacD)) * AF + Kmx * (aqDepth - MacD + (Ygw - (aqDepth - MacD)) * (1.0 - AF))) /
                   Ygw;
        }
    }
    return effk;
}

__device__ inline double soilMoistureStress(double ThetaS, double ThetaR, double SatRatio)
{
    const double fc = ThetaS * FieldCapacityRatio;
    double beta_s = (SatRatio * (ThetaS - ThetaR) - ThetaR) / (fc - ThetaR);
    beta_s = d_min(d_max(beta_s, 0.0), 1.0);
    beta_s = 0.5 * (1.0 - cos(PI * beta_s));
    return beta_s;
}

__device__ inline double weirFlow_jtoi(double zi,
                                       double yi,
                                       double zj,
                                       double yj,
                                       double zbank,
                                       double cwr,
                                       double width,
                                       double threshold)
{
    /* Positive = j -> i */
    const double hi = yi + zi;
    const double hj = yj + zj;
    const double dh = hj - hi;
    double Q = 0.0;
    if (dh > 0.0) { /* j -> i */
        double y = hi - zbank;
        if (y > 0.0 && yj > threshold) {
            if (hi > zbank) {
                y = dh;
            }
            Q = cwr * sqrt(2.0 * GRAV * y) * width * y * 60.0;
        }
    } else { /* i -> j */
        double y = hi - zbank;
        if (y > 0.0 && yi > threshold) {
            if (hj > zbank) {
                y = -dh;
            }
            Q = -1.0 * cwr * sqrt(2.0 * GRAV * y) * width * y * 60.0;
        }
    }
    return Q;
}

__device__ inline double flux_R2E_GW(double yr,
                                    double zr,
                                    double ye,
                                    double ze,
                                    double Kele,
                                    double Kriv,
                                    double L,
                                    double D_riv)
{
    if (Kele < ZERO || Kriv < ZERO) {
        return 0.0;
    }
    const double K = meanArithmetic(Kele, Kriv, 1.0, 1.0);
    const double he = ye + ze;
    const double hr = yr + zr;
    const double dh = hr - he;
    if (dh > ZERO) {
        if (yr < EPSILON) {
            return 0.0;
        }
        const double A = (he > zr) ? (0.5 * (yr + (he - zr)) * L) : (yr * L);
        return A * K * (dh / D_riv);
    }
    if (dh < -ZERO) {
        if (ye <= ZERO) {
            return 0.0;
        }
        const double A = 0.5 * (yr + (he - zr)) * L;
        return A * K * (dh / D_riv);
    }
    return 0.0;
}

__device__ inline double fun_CrossArea(double y, double w0, double s) { return y * (w0 + y * s); }

__device__ inline double fun_CrossPerem(double y, double w0, double s) { return 2.0 * sqrt(y * y + (y * s) * (y * s)) + w0; }

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

__global__ void k_apply_bc_and_sanitize_state(const realtype *dY, const DeviceModel *m, int clamp_policy)
{
    if (dY == nullptr || m == nullptr) {
        return;
    }

    const int nEle = m->NumEle;
    const int nRiv = m->NumRiv;
    const int nLake = m->NumLake;
    const int n = 3 * nEle + nRiv + nLake;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        const double y = static_cast<double>(dY[idx]);
        if (idx < nEle) {
            m->uYsf[idx] = clamp_policy ? d_clamp_nonneg(y) : y;
        } else if (idx < 2 * nEle) {
            const int i = idx - nEle;
            m->uYus[i] = clamp_policy ? d_clamp_nonneg(y) : y;
        } else if (idx < 3 * nEle) {
            const int i = idx - 2 * nEle;
            const int bc = (m->ele_iBC != nullptr) ? m->ele_iBC[i] : 0;
            if (bc > 0 && m->ele_yBC != nullptr) {
                m->uYgw[i] = m->ele_yBC[i];
            } else {
                m->uYgw[i] = clamp_policy ? d_clamp_nonneg(y) : y;
            }
        } else if (idx < 3 * nEle + nRiv) {
            const int i = idx - 3 * nEle;
            const int bc = (m->riv_BC != nullptr) ? m->riv_BC[i] : 0;
            if (bc > 0 && m->riv_yBC != nullptr) {
                m->uYriv[i] = m->riv_yBC[i];
            } else {
                m->uYriv[i] = clamp_policy ? d_clamp_nonneg(y) : y;
            }
        } else {
            const int i = idx - (3 * nEle + nRiv);
            m->uYlake[i] = clamp_policy ? d_clamp_nonneg(y) : y;
        }
    }
}

__global__ void k_ele_local(const DeviceModel *m)
{
    if (m == nullptr || m->NumEle <= 0) {
        return;
    }

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m->NumEle; i += blockDim.x * gridDim.x) {
        const int iLake = (m->ele_iLake != nullptr) ? m->ele_iLake[i] : 0;

        if (iLake > 0) {
            const int lake_idx = iLake - 1;
            if (m->qEleInfil != nullptr) m->qEleInfil[i] = 0.0;
            if (m->qEleExfil != nullptr) m->qEleExfil[i] = 0.0;
            if (m->qEleRecharge != nullptr) m->qEleRecharge[i] = 0.0;
            if (m->qEs != nullptr) m->qEs[i] = 0.0;
            if (m->qEu != nullptr) m->qEu[i] = 0.0;
            if (m->qEg != nullptr) m->qEg[i] = 0.0;
            if (m->qTu != nullptr) m->qTu[i] = 0.0;
            if (m->qTg != nullptr) m->qTg[i] = 0.0;

            if (m->ele_satn != nullptr) m->ele_satn[i] = 1.0;
            if (m->ele_effKH != nullptr && m->ele_KsatH != nullptr) m->ele_effKH[i] = m->ele_KsatH[i];

            if (m->qLakeEvap != nullptr && m->qPotEvap != nullptr && m->lake_invNumEle != nullptr) {
                atomicAdd(&m->qLakeEvap[lake_idx], m->qPotEvap[i] * m->lake_invNumEle[lake_idx]);
            }
            if (m->qLakePrcp != nullptr && m->qElePrep != nullptr && m->lake_invNumEle != nullptr) {
                atomicAdd(&m->qLakePrcp[lake_idx], m->qElePrep[i] * m->lake_invNumEle[lake_idx]);
            }
            continue;
        }

        const double ysf = (m->uYsf != nullptr) ? m->uYsf[i] : 0.0;
        const double yus = (m->uYus != nullptr) ? m->uYus[i] : 0.0;
        const double ygw = (m->uYgw != nullptr) ? m->uYgw[i] : 0.0;

        double satn_prev = (m->ele_satn != nullptr) ? m->ele_satn[i] : -1.0;

        const double AquiferDepth = (m->ele_AquiferDepth != nullptr) ? m->ele_AquiferDepth[i] : 0.0;
        const double infD = (m->ele_infD != nullptr) ? m->ele_infD[i] : 0.0;
        const double ThetaS = (m->ele_ThetaS != nullptr) ? m->ele_ThetaS[i] : 0.0;
        const double ThetaR = (m->ele_ThetaR != nullptr) ? m->ele_ThetaR[i] : 0.0;
        const double ThetaFC = (m->ele_ThetaFC != nullptr) ? m->ele_ThetaFC[i] : 0.0;
        const double Alpha = (m->ele_Alpha != nullptr) ? m->ele_Alpha[i] : 0.0;
        const double Beta = (m->ele_Beta != nullptr) ? m->ele_Beta[i] : 0.0;
        const double hAreaF = (m->ele_hAreaF != nullptr) ? m->ele_hAreaF[i] : 0.0;
        const double macKsatV = (m->ele_macKsatV != nullptr) ? m->ele_macKsatV[i] : 0.0;
        const double infKsatV = (m->ele_infKsatV != nullptr) ? m->ele_infKsatV[i] : 0.0;
        const double KsatV = (m->ele_KsatV != nullptr) ? m->ele_KsatV[i] : 0.0;
        const double macD = (m->ele_macD != nullptr) ? m->ele_macD[i] : 0.0;
        const double macKsatH = (m->ele_macKsatH != nullptr) ? m->ele_macKsatH[i] : 0.0;
        const double geo_vAreaF = (m->ele_geo_vAreaF != nullptr) ? m->ele_geo_vAreaF[i] : 0.0;
        const double KsatH = (m->ele_KsatH != nullptr) ? m->ele_KsatH[i] : 0.0;
        const double VegFrac = (m->ele_VegFrac != nullptr) ? m->ele_VegFrac[i] : 0.0;
        const double ImpAF = (m->ele_ImpAF != nullptr) ? m->ele_ImpAF[i] : 0.0;
        const double RzD = (m->ele_RzD != nullptr) ? m->ele_RzD[i] : 0.0;

        const double fu_surf = (m->fu_Surf != nullptr) ? m->fu_Surf[i] : 1.0;
        const double fu_sub = (m->fu_Sub != nullptr) ? m->fu_Sub[i] : 1.0;
        const double netprcp = (m->qEleNetPrep != nullptr) ? m->qEleNetPrep[i] : 0.0;

        const double Kmax = infKsatV * (1.0 - hAreaF) + macKsatV * hAreaF;

        /* updateElement (satn_new / effKH) */
        double effKH_val = effKH(ygw, AquiferDepth, macD, macKsatH, geo_vAreaF, KsatH);
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

        /* f_etFlux (uses satn from previous iteration) */
        const bool satn_prev_valid = (satn_prev == satn_prev) && satn_prev >= 0.0 && satn_prev <= 1.0;
        if (!satn_prev_valid) {
            satn_prev = satn_new;
        }
        const double iBeta = soilMoistureStress(ThetaS, ThetaR, satn_prev);
        const double va = VegFrac;
        const double vb = 1.0 - VegFrac;
        const double pj = 1.0 - ImpAF;

        const double qPotEvap = (m->qPotEvap != nullptr) ? m->qPotEvap[i] : 0.0;
        const double qPotTran = (m->qPotTran != nullptr) ? m->qPotTran[i] : 0.0;
        const double lai = (m->t_lai != nullptr) ? m->t_lai[i] : 0.0;
        const double qEIC = (m->qEleE_IC != nullptr) ? m->qEleE_IC[i] : 0.0;

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

        if (m->qEs != nullptr) m->qEs[i] = Es;
        if (m->qEu != nullptr) m->qEu[i] = Eu;
        if (m->qEg != nullptr) m->qEg[i] = Eg;
        if (m->qTu != nullptr) m->qTu[i] = Tu;
        if (m->qTg != nullptr) m->qTg[i] = Tg;

        /* Flux_Infiltration */
        double qi = 0.0;
        double qex = 0.0;
        const double av = ysf + netprcp;
        if (AquiferDepth > ZERO && (ygw + yus > AquiferDepth || deficit < yus)) {
            qex = fabs(ygw + yus - AquiferDepth) / AquiferDepth * Kmax;
            qi = 0.0;
        } else {
            qex = 0.0;
            if (av > 0.0 && deficit > infD) {
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

        if (m->qEleInfil != nullptr) m->qEleInfil[i] = qi * fu_surf;
        if (m->qEleExfil != nullptr) m->qEleExfil[i] = qex * fu_surf;

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

        if (m->qEleRecharge != nullptr) m->qEleRecharge[i] = qr * fu_sub;

        if (m->ele_effKH != nullptr) {
            m->ele_effKH[i] = effKH_val;
        }
        if (m->ele_satn != nullptr) {
            m->ele_satn[i] = satn_new;
        }
    }
}

__global__ void k_ele_edge_surface(const DeviceModel *m)
{
    if (m == nullptr || m->NumEle <= 0) {
        return;
    }

    const int n = m->NumEle * 3;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        const int i = idx / 3;
        const int j = idx - i * 3;
        const int inabr = (m->ele_nabr != nullptr) ? (m->ele_nabr[idx] - 1) : -1;
        const int ilake = (m->ele_lakenabr != nullptr) ? (m->ele_lakenabr[idx] - 1) : -1;
        const double isf = d_max(0.0, (m->uYsf != nullptr) ? m->uYsf[i] : 0.0);
        const double B = (m->ele_edge != nullptr) ? m->ele_edge[idx] : 0.0;
        double Q = 0.0;

        if (ilake >= 0) {
            double nsf = (m->uYlake != nullptr) ? m->uYlake[ilake] : 0.0;
            nsf = d_max(0.0, nsf);
            Q = weirFlow_jtoi(m->lake_zmin[ilake],
                              nsf,
                              m->ele_z_surf[i],
                              isf,
                              m->ele_z_surf[i],
                              0.6,
                              B,
                              0.01);
            if (m->QLakeSurf != nullptr) {
                atomicAdd(&m->QLakeSurf[ilake], Q);
            }
        } else if (inabr >= 0) {
            double nsf = (m->uYsf != nullptr) ? m->uYsf[inabr] : 0.0;
            nsf = d_max(0.0, nsf);
            const double dh = (isf + m->ele_z_surf[i]) - (nsf + m->ele_z_surf[inabr]);
            double Ymean = avgY_sf(m->ele_z_surf[i], isf, m->ele_z_surf[inabr], nsf, m->ele_depression[i]);
            Ymean = d_min(Ymean, static_cast<double>(MAXYSURF));
            if (Ymean <= 0.0) {
                Q = 0.0;
            } else {
                const double s = dh / m->ele_Dist2Nabor[idx];
                const double CrossA = Ymean * B;
                if ((s > 0.0 && isf <= 0.0) || (s < 0.0 && nsf <= 0.0)) {
                    Q = 0.0;
                } else {
                    Q = manningEquation(CrossA, m->ele_avgRough[idx], Ymean, s);
                }
            }
        } else {
            if (m->CloseBoundary) {
                Q = 0.0;
            } else {
                if (isf > m->ele_depression[i]) {
                    const double dist = (m->ele_Dist2Edge != nullptr) ? m->ele_Dist2Edge[idx] : 0.0;
                    const double s = (dist > 0.0) ? (isf / dist * 0.5) : 0.0;
                    if (s > 0.0) {
                        const double isf5 = isf * isf * isf * isf * isf;
                        Q = sqrt(s) * cbrt(isf5) * B / m->ele_Rough[i];
                    }
                }
            }
        }

        if (m->QeleSurf != nullptr) {
            m->QeleSurf[idx] = Q;
        }
    }
}

__global__ void k_ele_edge_sub(const DeviceModel *m)
{
    if (m == nullptr || m->NumEle <= 0) {
        return;
    }

    const int n = m->NumEle * 3;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        const int i = idx / 3;
        const int j = idx - i * 3;
        const int inabr = (m->ele_nabr != nullptr) ? (m->ele_nabr[idx] - 1) : -1;
        const int ilake = (m->ele_lakenabr != nullptr) ? (m->ele_lakenabr[idx] - 1) : -1;

        const double ygw = (m->uYgw != nullptr) ? m->uYgw[i] : 0.0;
        const double zbot = (m->ele_z_bottom != nullptr) ? m->ele_z_bottom[i] : 0.0;
        const double edge = (m->ele_edge != nullptr) ? m->ele_edge[idx] : 0.0;
        const double fu_sub = (m->fu_Sub != nullptr) ? m->fu_Sub[i] : 1.0;
        double Q = 0.0;

        if (ilake >= 0) {
            const double ylk = (m->uYlake != nullptr) ? m->uYlake[ilake] : 0.0;
            const double zlk = (m->lake_zmin != nullptr) ? m->lake_zmin[ilake] : 0.0;
            const double dh = (ygw + zbot) - (ylk + zlk);
            if ((dh > 0.0 && ygw <= 0.02) || (dh < 0.0 && ylk <= 0.02)) {
                Q = 0.0;
            } else {
                const double Ymean = avgY_gw(zbot, ygw, zlk, ylk, 0.002);
                const double grad = dh / m->ele_Dist2Nabor[idx];
                const double Kmean =
                    0.5 * ((m->ele_effKH != nullptr) ? m->ele_effKH[i] : 0.0) +
                    0.5 * ((m->ele_effKH != nullptr && inabr >= 0) ? m->ele_effKH[inabr] : 0.0);
                Q = Kmean * grad * Ymean * edge;
            }
            if (m->QLakeSub != nullptr) {
                atomicAdd(&m->QLakeSub[ilake], Q);
            }
        } else if (inabr >= 0) {
            const double ygw_n = (m->uYgw != nullptr) ? m->uYgw[inabr] : 0.0;
            const double zbot_n = (m->ele_z_bottom != nullptr) ? m->ele_z_bottom[inabr] : 0.0;
            const double dh = (ygw + zbot) - (ygw_n + zbot_n);
            if ((dh > 0.0 && ygw <= 0.02) || (dh < 0.0 && ygw_n <= 0.02)) {
                Q = 0.0;
            } else {
                const double Ymean = avgY_gw(zbot, ygw, zbot_n, ygw_n, 0.002);
                const double grad = dh / m->ele_Dist2Nabor[idx];
                const double Kmean =
                    0.5 * ((m->ele_effKH != nullptr) ? m->ele_effKH[i] : 0.0) +
                    0.5 * ((m->ele_effKH != nullptr) ? m->ele_effKH[inabr] : 0.0);
                Q = Kmean * grad * Ymean * edge;
            }
        } else {
            if (m->CloseBoundary) {
                Q = 0.0;
            } else {
                if ((m->ele_depression != nullptr) && (ygw > m->ele_depression[i] * 10.0)) {
                    const double dist = (m->ele_Dist2Edge != nullptr) ? m->ele_Dist2Edge[idx] : 0.0;
                    const double grad = (dist > 0.0) ? (ygw / dist * 0.5) : 0.0;
                    if (grad > 0.0) {
                        Q = ((m->ele_effKH != nullptr) ? m->ele_effKH[i] : 0.0) * grad;
                    }
                }
            }
        }

        if (m->QeleSub != nullptr) {
            m->QeleSub[idx] = Q * fu_sub;
        }
    }
}

__global__ void k_seg_exchange(const DeviceModel *m)
{
    if (m == nullptr || m->NumSeg <= 0) {
        return;
    }

    for (int seg = blockIdx.x * blockDim.x + threadIdx.x; seg < m->NumSeg; seg += blockDim.x * gridDim.x) {
        const int iEle = (m->seg_iEle != nullptr) ? (m->seg_iEle[seg] - 1) : -1;
        const int iRiv = (m->seg_iRiv != nullptr) ? (m->seg_iRiv[seg] - 1) : -1;
        if (iEle < 0 || iRiv < 0) {
            continue;
        }

        /* Surface exchange */
        double isf = (m->uYsf != nullptr) ? m->uYsf[iEle] : 0.0;
        const double qi = (m->qEleInfil != nullptr) ? m->qEleInfil[iEle] : 0.0;
        const double qex = (m->qEleExfil != nullptr) ? m->qEleExfil[iEle] : 0.0;
        isf = d_max(0.0, isf - qi + qex);
        const double QsegSurf = weirFlow_jtoi(m->ele_z_surf[iEle],
                                              isf,
                                              m->ele_z_surf[iEle] - m->riv_depth[iRiv],
                                              m->uYriv[iRiv],
                                              m->ele_z_surf[iEle] + m->riv_zbank[iRiv],
                                              m->seg_Cwr[seg],
                                              m->seg_length[seg],
                                              m->ele_depression[iEle]);

        if (m->QrivSurf != nullptr) atomicAdd(&m->QrivSurf[iRiv], QsegSurf);
        if (m->Qe2r_Surf != nullptr) atomicAdd(&m->Qe2r_Surf[iEle], -QsegSurf);

        /* Subsurface exchange */
        const double QsegSub_raw = flux_R2E_GW(m->uYriv[iRiv],
                                               m->ele_z_surf[iEle] - m->riv_depth[iRiv],
                                               m->uYgw[iEle],
                                               m->ele_z_bottom[iEle],
                                               m->ele_effKH[iEle],
                                               m->riv_KsatH[iRiv],
                                               m->seg_length[seg],
                                               m->riv_BedThick[iRiv]);
        const double QsegSub = QsegSub_raw * m->fu_Sub[iEle];

        if (m->QrivSub != nullptr) atomicAdd(&m->QrivSub[iRiv], QsegSub);
        if (m->Qe2r_Sub != nullptr) atomicAdd(&m->Qe2r_Sub[iEle], -QsegSub);
    }
}

__global__ void k_river_down_and_up(const DeviceModel *m)
{
    if (m == nullptr || m->NumRiv <= 0) {
        return;
    }

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m->NumRiv; i += blockDim.x * gridDim.x) {
        const double yriv = (m->uYriv != nullptr) ? m->uYriv[i] : 0.0;
        const double w0 = (m->riv_BottomWidth != nullptr) ? m->riv_BottomWidth[i] : 0.0;
        const double bs = (m->riv_BankSlope != nullptr) ? m->riv_BankSlope[i] : 0.0;
        double topWidth = fun_TopWidth(yriv, w0, bs);
        double CSarea = fun_CrossArea(yriv, w0, bs);
        double CSperem = fun_CrossPerem(yriv, w0, bs);
        topWidth = d_max(0.0, topWidth);
        CSarea = d_max(0.0, CSarea);
        CSperem = d_max(0.0, CSperem);

        if (m->riv_topWidth != nullptr) m->riv_topWidth[i] = topWidth;
        if (m->riv_CSarea != nullptr) m->riv_CSarea[i] = CSarea;
        if (m->riv_CSperem != nullptr) m->riv_CSperem[i] = CSperem;

        const int toLake = (m->riv_toLake != nullptr) ? m->riv_toLake[i] : -1;
        const int down = (m->riv_down_raw != nullptr) ? m->riv_down_raw[i] : -1;
        const int iDown = down - 1;
        const double n = (m->riv_avgRough != nullptr) ? m->riv_avgRough[i] : ((m->riv_rivRough != nullptr) ? m->riv_rivRough[i] : 1.0);

        double Qdown = 0.0;
        if (toLake >= 0) {
            const double s = m->riv_BedSlope[i] + yriv * 2.0 / m->riv_Length[i];
            const double R = (CSperem <= 0.0) ? 0.0 : (CSarea / CSperem);
            Qdown = manningEquation(CSarea, n, R, s);
            if (m->QLakeRivIn != nullptr) {
                atomicAdd(&m->QLakeRivIn[toLake], Qdown);
            }
        } else if (iDown >= 0) {
            const double sMean = 0.5 * (m->riv_BedSlope[i] + m->riv_BedSlope[iDown]);
            const double Distance = (m->riv_Dist2DownStream != nullptr) ? m->riv_Dist2DownStream[i] : m->riv_Length[i];
            const double s = ((yriv - m->riv_depth[i]) - (m->uYriv[iDown] - m->riv_depth[iDown])) / Distance + sMean;
            const double R = (CSperem <= ZERO) ? 0.0 : (CSarea / CSperem);
            Qdown = manningEquation(CSarea, n, R, s);
        } else {
            switch (down) {
                case -1:
                case -2:
                case -3: {
                    const double s = m->riv_BedSlope[i] + yriv * 2.0 / m->riv_Length[i];
                    const double R = (CSperem <= 0.0) ? 0.0 : (CSarea / CSperem);
                    Qdown = manningEquation(CSarea, n, R, s);
                    break;
                }
                case -4:
                    Qdown = CSarea * sqrt(GRAV * yriv) * 60.0;
                    break;
                default:
                    Qdown = 0.0;
                    break;
            }
        }

        if (m->QrivDown != nullptr) {
            m->QrivDown[i] = Qdown;
        }
        if (m->QrivUp != nullptr && iDown >= 0) {
            atomicAdd(&m->QrivUp[iDown], -Qdown);
        }
    }
}

__global__ void k_lake_toparea_and_scale(const DeviceModel *m)
{
    if (m == nullptr || m->NumLake <= 0) {
        return;
    }

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m->NumLake; i += blockDim.x * gridDim.x) {
        const double yStage = (m->uYlake != nullptr) ? m->uYlake[i] : 0.0;
        const double zmin = (m->lake_zmin != nullptr) ? m->lake_zmin[i] : 0.0;
        const double area = lake_toparea(m, i, yStage + zmin);
        if (m->y2LakeArea != nullptr) {
            m->y2LakeArea[i] = area;
        }

        if (m->qLakeEvap != nullptr && m->qLakePrcp != nullptr) {
            double evap = m->qLakeEvap[i];
            const double prcp = m->qLakePrcp[i];
            evap = d_min(evap, prcp + yStage);
            evap = d_max(0.0, evap);
            m->qLakeEvap[i] = evap;
        }
    }
}

__global__ void k_apply_dy_element(realtype *dYdot, const DeviceModel *m)
{
    if (dYdot == nullptr || m == nullptr || m->NumEle <= 0) {
        return;
    }

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m->NumEle; i += blockDim.x * gridDim.x) {
        const double area = m->ele_area[i];
        const double QeleSurfTot = m->Qe2r_Surf[i] + m->QeleSurf[i * 3 + 0] + m->QeleSurf[i * 3 + 1] + m->QeleSurf[i * 3 + 2];
        const double QeleSubTot = m->Qe2r_Sub[i] + m->QeleSub[i * 3 + 0] + m->QeleSub[i * 3 + 1] + m->QeleSub[i * 3 + 2];

        double DYsf = m->qEleNetPrep[i] - m->qEleInfil[i] + m->qEleExfil[i] - QeleSurfTot / area - m->qEs[i];
        double DYus = m->qEleInfil[i] - m->qEleRecharge[i] - m->qEu[i] - m->qTu[i];
        double DYgw = m->qEleRecharge[i] - m->qEleExfil[i] - QeleSubTot / area - m->qEg[i] - m->qTg[i];

        const int bc = (m->ele_iBC != nullptr) ? m->ele_iBC[i] : 0;
        if (bc > 0) {
            DYgw = 0.0;
        } else if (bc < 0) {
            DYgw += m->ele_QBC[i] / area;
        }

        const int ss = (m->ele_iSS != nullptr) ? m->ele_iSS[i] : 0;
        if (ss > 0) {
            DYsf += m->ele_QSS[i] / area;
        } else if (ss < 0) {
            DYgw += m->ele_QSS[i] / area;
        }

        const double Sy = m->ele_Sy[i];
        DYus /= Sy;
        DYgw /= Sy;

        if ((m->ele_iLake != nullptr) && (m->ele_iLake[i] > 0)) {
            DYsf = 0.0;
            DYus = 0.0;
            DYgw = 0.0;
        }

        dYdot[i] = static_cast<realtype>(DYsf);
        dYdot[i + m->NumEle] = static_cast<realtype>(DYus);
        dYdot[i + 2 * m->NumEle] = static_cast<realtype>(DYgw);
    }
}

__global__ void k_apply_dy_river(realtype *dYdot, const DeviceModel *m)
{
    if (dYdot == nullptr || m == nullptr || m->NumRiv <= 0) {
        return;
    }

    const int base = 3 * m->NumEle;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m->NumRiv; i += blockDim.x * gridDim.x) {
        const int bc = (m->riv_BC != nullptr) ? m->riv_BC[i] : 0;
        if (bc > 0) {
            dYdot[base + i] = static_cast<realtype>(0.0);
            continue;
        }
        double dA = (-m->QrivUp[i] - m->QrivSurf[i] - m->QrivSub[i] - m->QrivDown[i] + m->riv_qBC[i]) / m->riv_Length[i];
        const double CSarea = (m->riv_CSarea != nullptr) ? m->riv_CSarea[i] : 0.0;
        if (dA < -1.0 * CSarea) {
            dA = -1.0 * CSarea;
        }
        const double dy = fun_dAtodY(dA, (m->riv_topWidth != nullptr) ? m->riv_topWidth[i] : 0.0, m->riv_BankSlope[i]);
        dYdot[base + i] = static_cast<realtype>(dy);
    }
}

__global__ void k_apply_dy_lake(realtype *dYdot, const DeviceModel *m)
{
    if (dYdot == nullptr || m == nullptr || m->NumLake <= 0) {
        return;
    }

    const int base = 3 * m->NumEle + m->NumRiv;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < m->NumLake; i += blockDim.x * gridDim.x) {
        const double area = m->y2LakeArea[i];
        const double dy = m->qLakePrcp[i] - m->qLakeEvap[i] + (m->QLakeRivIn[i] - m->QLakeRivOut[i] + m->QLakeSub[i] + m->QLakeSurf[i]) / area;
        dYdot[base + i] = static_cast<realtype>(dy);
    }
}

} // namespace

void launch_rhs_kernels(realtype t,
                        const realtype *dY,
                        realtype *dYdot,
                        const DeviceModel *d_model,
                        const DeviceModel *h_model,
                        cudaStream_t stream)
{
    (void)t;
    if (dY == nullptr || dYdot == nullptr || d_model == nullptr || h_model == nullptr) {
        return;
    }

    const int nEle = h_model->NumEle;
    const int nRiv = h_model->NumRiv;
    const int nSeg = h_model->NumSeg;
    const int nLake = h_model->NumLake;
    const int nY = 3 * nEle + nRiv + nLake;

    const int clamp_policy = CLAMP_POLICY;
    constexpr int kBlockSize = 256;
    const auto cap_blocks = [](int blocks) { return (blocks > 65535) ? 65535 : blocks; };

    /* 0) memset / init (match CPU f_update semantics) */
    if (nEle > 0) {
        (void)cudaMemsetAsync(h_model->Qe2r_Surf, 0, static_cast<size_t>(nEle) * sizeof(double), stream);
        (void)cudaMemsetAsync(h_model->Qe2r_Sub, 0, static_cast<size_t>(nEle) * sizeof(double), stream);
    }
    if (nRiv > 0) {
        (void)cudaMemsetAsync(h_model->QrivSurf, 0, static_cast<size_t>(nRiv) * sizeof(double), stream);
        (void)cudaMemsetAsync(h_model->QrivSub, 0, static_cast<size_t>(nRiv) * sizeof(double), stream);
        (void)cudaMemsetAsync(h_model->QrivUp, 0, static_cast<size_t>(nRiv) * sizeof(double), stream);
    }
    if (nLake > 0) {
        (void)cudaMemsetAsync(h_model->QLakeSurf, 0, static_cast<size_t>(nLake) * sizeof(double), stream);
        (void)cudaMemsetAsync(h_model->QLakeSub, 0, static_cast<size_t>(nLake) * sizeof(double), stream);
        (void)cudaMemsetAsync(h_model->QLakeRivIn, 0, static_cast<size_t>(nLake) * sizeof(double), stream);
        (void)cudaMemsetAsync(h_model->QLakeRivOut, 0, static_cast<size_t>(nLake) * sizeof(double), stream);
        (void)cudaMemsetAsync(h_model->qLakePrcp, 0, static_cast<size_t>(nLake) * sizeof(double), stream);
        (void)cudaMemsetAsync(h_model->qLakeEvap, 0, static_cast<size_t>(nLake) * sizeof(double), stream);
    }
    (void)cudaMemsetAsync(dYdot, 0, static_cast<size_t>(nY) * sizeof(realtype), stream);

    /* 1) apply BC + sanitize */
    if (nY > 0) {
        const int blocks = cap_blocks((nY + kBlockSize - 1) / kBlockSize);
        k_apply_bc_and_sanitize_state<<<blocks, kBlockSize, 0, stream>>>(dY, d_model, clamp_policy);
    }

    /* 2) element local */
    if (nEle > 0) {
        const int blocks = cap_blocks((nEle + kBlockSize - 1) / kBlockSize);
        k_ele_local<<<blocks, kBlockSize, 0, stream>>>(d_model);
    }

    /* 3) element edge surface */
    if (nEle > 0) {
        const int nEdge = nEle * 3;
        const int blocks = cap_blocks((nEdge + kBlockSize - 1) / kBlockSize);
        k_ele_edge_surface<<<blocks, kBlockSize, 0, stream>>>(d_model);
    }

    /* 4) element edge subsurface */
    if (nEle > 0) {
        const int nEdge = nEle * 3;
        const int blocks = cap_blocks((nEdge + kBlockSize - 1) / kBlockSize);
        k_ele_edge_sub<<<blocks, kBlockSize, 0, stream>>>(d_model);
    }

    /* 5) segment exchange */
    if (nSeg > 0) {
        const int blocks = cap_blocks((nSeg + kBlockSize - 1) / kBlockSize);
        k_seg_exchange<<<blocks, kBlockSize, 0, stream>>>(d_model);
    }

    /* 6) river down + up */
    if (nRiv > 0) {
        const int blocks = cap_blocks((nRiv + kBlockSize - 1) / kBlockSize);
        k_river_down_and_up<<<blocks, kBlockSize, 0, stream>>>(d_model);
    }

    /* 7) lake toparea + evap cap */
    if (nLake > 0) {
        const int blocks = cap_blocks((nLake + kBlockSize - 1) / kBlockSize);
        k_lake_toparea_and_scale<<<blocks, kBlockSize, 0, stream>>>(d_model);
    }

    /* 8) apply DY element */
    if (nEle > 0) {
        const int blocks = cap_blocks((nEle + kBlockSize - 1) / kBlockSize);
        k_apply_dy_element<<<blocks, kBlockSize, 0, stream>>>(dYdot, d_model);
    }

    /* 9) apply DY river */
    if (nRiv > 0) {
        const int blocks = cap_blocks((nRiv + kBlockSize - 1) / kBlockSize);
        k_apply_dy_river<<<blocks, kBlockSize, 0, stream>>>(dYdot, d_model);
    }

    /* 10) apply DY lake */
    if (nLake > 0) {
        const int blocks = cap_blocks((nLake + kBlockSize - 1) / kBlockSize);
        k_apply_dy_lake<<<blocks, kBlockSize, 0, stream>>>(dYdot, d_model);
    }
}

#endif /* _CUDA_ON */
