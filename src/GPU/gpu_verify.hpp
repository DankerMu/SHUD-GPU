#ifndef SHUD_GPU_VERIFY_HPP
#define SHUD_GPU_VERIFY_HPP

#include <cstddef>
#include <cstdio>

struct GpuVerifySettings {
    bool enabled = true;
    unsigned long interval = 10; /* Verify every N RHS calls. */
    double atol = 1.0e-8;
    double rtol = 1.0e-6;
    int max_print = 8;           /* Print up to N mismatching entries. */
    bool stop_on_mismatch = true; /* Stop after first mismatch. */
    bool abort_on_mismatch = false; /* Abort process on mismatch. */
};

GpuVerifySettings gpuVerifySettingsFromEnv();

struct CompareResult {
    size_t n = 0;
    size_t mismatch_count = 0;
    size_t max_idx = 0;
    double max_abs = 0.0;
    double max_rel = 0.0;
    double cpu_value = 0.0;
    double gpu_value = 0.0;
    double threshold = 0.0;
};

template <typename CpuT, typename GpuT>
CompareResult compare_arrays(const CpuT *cpu, const GpuT *gpu, size_t n, double atol, double rtol)
{
    CompareResult r{};
    r.n = n;
    if (cpu == nullptr || gpu == nullptr || n == 0) {
        return r;
    }

    size_t first_idx = 0;
    bool have_first = false;
    for (size_t i = 0; i < n; i++) {
        const double c = static_cast<double>(cpu[i]);
        const double g = static_cast<double>(gpu[i]);
        const double diff = (c > g) ? (c - g) : (g - c);
        const double ac = (c >= 0.0) ? c : -c;
        const double ag = (g >= 0.0) ? g : -g;
        const double denom = (ac > ag) ? ac : ag;
        const double thr = (atol > rtol * denom) ? atol : (rtol * denom);
        if (diff > thr) {
            r.mismatch_count++;
            if (!have_first) {
                have_first = true;
                first_idx = i;
            }
        }

        if (diff > r.max_abs) {
            r.max_abs = diff;
            r.max_idx = i;
            r.cpu_value = c;
            r.gpu_value = g;
            r.threshold = thr;
            r.max_rel = (denom > 0.0) ? (diff / denom) : 0.0;
        }
    }

    if (!have_first && r.mismatch_count > 0) {
        r.max_idx = first_idx;
    }
    return r;
}

struct GpuVerifyContext {
    unsigned long step = 0;
    double t = 0.0;

    int NumEle = 0;
    int NumRiv = 0;
    int NumLake = 0;
    int NumY = 0;

    GpuVerifySettings settings{};

    /* CPU-side reference arrays. */
    const double *cpu_uYsf = nullptr;
    const double *cpu_uYus = nullptr;
    const double *cpu_uYgw = nullptr;
    const double *cpu_uYriv = nullptr;
    const double *cpu_yLakeStg = nullptr; /* Model_Data::yLakeStg (not global uYlake) */

    const double *cpu_ele_satn = nullptr;
    const double *cpu_ele_effKH = nullptr;

    const double *cpu_qEleInfil = nullptr;
    const double *cpu_qEleExfil = nullptr;
    const double *cpu_qEleRecharge = nullptr;
    const double *cpu_qEs = nullptr;
    const double *cpu_qEu = nullptr;
    const double *cpu_qEg = nullptr;
    const double *cpu_qTu = nullptr;
    const double *cpu_qTg = nullptr;

    const double *cpu_QeleSurf = nullptr; /* Flattened [NumEle*3]. */
    const double *cpu_QeleSub = nullptr;  /* Flattened [NumEle*3]. */

    const double *cpu_Qe2r_Surf = nullptr;
    const double *cpu_Qe2r_Sub = nullptr;

    const double *cpu_QrivSurf = nullptr;
    const double *cpu_QrivSub = nullptr;
    const double *cpu_QrivUp = nullptr;
    const double *cpu_QrivDown = nullptr;
    const double *cpu_riv_topWidth = nullptr;
    const double *cpu_riv_CSarea = nullptr;
    const double *cpu_riv_CSperem = nullptr;

    const double *cpu_QLakeSurf = nullptr;
    const double *cpu_QLakeSub = nullptr;
    const double *cpu_QLakeRivIn = nullptr;
    const double *cpu_QLakeRivOut = nullptr;
    const double *cpu_qLakePrcp = nullptr;
    const double *cpu_qLakeEvap = nullptr;
    const double *cpu_y2LakeArea = nullptr;

    const double *cpu_dYdot = nullptr; /* DY reference [NumY]. */
};

void gpuVerifyReport(FILE *out, const char *kernel, const char *field, const GpuVerifyContext &ctx, const CompareResult &r, const char *index_hint);

#endif /* SHUD_GPU_VERIFY_HPP */

