#include "gpu_verify.hpp"

#ifdef DEBUG_GPU_VERIFY

#include <cmath>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <strings.h>

namespace {

bool parseBoolEnv(const char *value, bool fallback)
{
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }

    if (strcmp(value, "1") == 0 || strcasecmp(value, "true") == 0 || strcasecmp(value, "yes") == 0 || strcasecmp(value, "on") == 0) {
        return true;
    }
    if (strcmp(value, "0") == 0 || strcasecmp(value, "false") == 0 || strcasecmp(value, "no") == 0 || strcasecmp(value, "off") == 0) {
        return false;
    }
    return fallback;
}

unsigned long parseULongEnv(const char *value, unsigned long fallback)
{
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    errno = 0;
    char *end = nullptr;
    const unsigned long v = std::strtoul(value, &end, 10);
    if (errno != 0 || end == value) {
        return fallback;
    }
    return v;
}

int parseIntEnv(const char *value, int fallback)
{
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    errno = 0;
    char *end = nullptr;
    const long v = std::strtol(value, &end, 10);
    if (errno != 0 || end == value) {
        return fallback;
    }
    return static_cast<int>(v);
}

double parseDoubleEnv(const char *value, double fallback)
{
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    errno = 0;
    char *end = nullptr;
    const double v = std::strtod(value, &end);
    if (errno != 0 || end == value) {
        return fallback;
    }
    return v;
}

bool parseFiniteDoubleEnv(const char *value, double *out)
{
    if (out == nullptr) {
        return false;
    }
    if (value == nullptr || value[0] == '\0') {
        return false;
    }

    errno = 0;
    char *end = nullptr;
    const double v = std::strtod(value, &end);
    if (errno != 0 || end == value || !std::isfinite(v)) {
        return false;
    }
    *out = v;
    return true;
}

} // namespace

GpuVerifySettings gpuVerifySettingsFromEnv()
{
    static GpuVerifySettings cached = []() {
        GpuVerifySettings s{};
        s.enabled = parseBoolEnv(std::getenv("SHUD_GPU_VERIFY"), true);
        s.interval = parseULongEnv(std::getenv("SHUD_GPU_VERIFY_INTERVAL"), s.interval);
        s.atol = parseDoubleEnv(std::getenv("SHUD_GPU_VERIFY_ATOL"), s.atol);
        s.rtol = parseDoubleEnv(std::getenv("SHUD_GPU_VERIFY_RTOL"), s.rtol);
        s.max_print = parseIntEnv(std::getenv("SHUD_GPU_VERIFY_MAX_PRINT"), s.max_print);
        s.stop_on_mismatch = parseBoolEnv(std::getenv("SHUD_GPU_VERIFY_STOP_ON_MISMATCH"), s.stop_on_mismatch);
        s.abort_on_mismatch = parseBoolEnv(std::getenv("SHUD_GPU_VERIFY_ABORT_ON_MISMATCH"), s.abort_on_mismatch);
        {
            const double kMinutesPerDay = 1440.0;
            double v = 0.0;
            if (parseFiniteDoubleEnv(std::getenv("SHUD_GPU_VERIFY_T_MIN"), &v)) {
                s.t_min = v;
            } else if (parseFiniteDoubleEnv(std::getenv("SHUD_GPU_VERIFY_T_MIN_DAY"), &v)) {
                s.t_min = v * kMinutesPerDay;
            }

            if (parseFiniteDoubleEnv(std::getenv("SHUD_GPU_VERIFY_T_MAX"), &v)) {
                s.t_max = v;
            } else if (parseFiniteDoubleEnv(std::getenv("SHUD_GPU_VERIFY_T_MAX_DAY"), &v)) {
                s.t_max = v * kMinutesPerDay;
            }

            if (s.t_max < s.t_min) {
                s.enabled = false;
            }
        }
        if (s.interval == 0) {
            s.enabled = false;
        }
        if (s.max_print < 0) {
            s.max_print = 0;
        }
        if (s.atol < 0.0) {
            s.atol = 0.0;
        }
        if (s.rtol < 0.0) {
            s.rtol = 0.0;
        }
        return s;
    }();
    return cached;
}

void gpuVerifyReport(FILE *out, const char *kernel, const char *field, const GpuVerifyContext &ctx, const CompareResult &r, const char *index_hint)
{
    if (out == nullptr) {
        out = stderr;
    }
    if (kernel == nullptr) {
        kernel = "(unknown)";
    }
    if (field == nullptr) {
        field = "(unknown)";
    }
    if (index_hint == nullptr) {
        index_hint = "";
    }

    fprintf(out,
            "[GPU_VERIFY] step=%lu t=%.10g kernel=%s field=%s n=%zu mismatches=%zu "
            "worst_ratio=%.6e diff=%.6e rel=%.6e idx=%zu%s cpu=%.10g gpu=%.10g thr=%.6e (atol=%.3e rtol=%.3e)\n",
            ctx.step,
            ctx.t,
            kernel,
            field,
            r.n,
            r.mismatch_count,
            r.worst_ratio,
            r.max_abs,
            r.max_rel,
            r.max_idx,
            index_hint,
            r.cpu_value,
            r.gpu_value,
            r.threshold,
            ctx.settings.atol,
            ctx.settings.rtol);
}

#else /* DEBUG_GPU_VERIFY */

GpuVerifySettings gpuVerifySettingsFromEnv()
{
    return GpuVerifySettings{};
}

void gpuVerifyReport(FILE *out, const char *kernel, const char *field, const GpuVerifyContext &ctx, const CompareResult &r, const char *index_hint)
{
    (void)out;
    (void)kernel;
    (void)field;
    (void)ctx;
    (void)r;
    (void)index_hint;
}

#endif /* DEBUG_GPU_VERIFY */
