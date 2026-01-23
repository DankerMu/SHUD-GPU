#ifndef SHUD_GPU_NVTX_HPP
#define SHUD_GPU_NVTX_HPP

#if defined(_CUDA_ON)
#if __has_include(<nvtx3/nvtx3.hpp>)
#include <nvtx3/nvtx3.hpp>
#define SHUD_NVTX_BACKEND_NVTX3 1
#elif __has_include(<nvToolsExt.h>)
#include <nvToolsExt.h>
#define SHUD_NVTX_BACKEND_NVTX2 1
#else
#define SHUD_NVTX_BACKEND_NONE 1
#endif
#else
#define SHUD_NVTX_BACKEND_NONE 1
#endif

namespace shud_nvtx {

#if defined(SHUD_NVTX_BACKEND_NVTX3)
using scoped_range = nvtx3::scoped_range;
#elif defined(SHUD_NVTX_BACKEND_NVTX2)
class scoped_range {
public:
    explicit scoped_range(const char *message)
    {
        nvtxRangePushA((message != nullptr) ? message : "");
    }

    ~scoped_range() { nvtxRangePop(); }

    scoped_range(const scoped_range &) = delete;
    scoped_range &operator=(const scoped_range &) = delete;
};
#else
class scoped_range {
public:
    explicit scoped_range(const char *) {}
    ~scoped_range() = default;

    scoped_range(const scoped_range &) = delete;
    scoped_range &operator=(const scoped_range &) = delete;
};
#endif

} // namespace shud_nvtx

#endif /* SHUD_GPU_NVTX_HPP */
